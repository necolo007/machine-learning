"""主流程：串联数据分析、特征工程、模型训练与评估。"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from . import config
from .eda import run_eda
from .evaluate import (evaluate_horizons, plot_metric_by_horizon, plot_predictions,
                       plot_summary_bar, plot_timeseries)
from .features import prepare_dataset
from .models import ann, ensemble, lstm, rf

warnings.filterwarnings("ignore")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_all(skip_eda=False, verbose=0, use_cache=True):
    # ---------- 一、数据分析 ----------
    if not skip_eda:
        log("一、数据分析 (EDA) ...")
        run_eda()

    # ---------- 二、特征工程 ----------
    log("二、特征工程：时间/滞后/窗口特征 → 特征选择 → 归一化 ...")
    ds = prepare_dataset()
    for k, v in ds.info.items():
        log(f"    {k}: {v}")
    log(f"    选中特征: {ds.feature_cols}")

    # ---------- 三、径流预测 ----------
    log("三、径流预测：24 个个体模型 + 16 个集成模型 ...")
    cache_dir = config.RESULT_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)

    def run_model(name, fn, *args):
        """带磁盘缓存的模型训练，避免中途失败需要重跑全部模型。"""
        cache = cache_dir / f"{name}.npz"
        if use_cache and cache.exists():
            d = np.load(cache, allow_pickle=True)
            p = d["params"]
            log(f"  [cache] {name}")
            return d["pred"], (p.item() if p.ndim == 0 else list(p))
        log(f"  训练 {name} ...")
        t0 = time.time()
        out = fn(*args)
        pred, params = out if isinstance(out, tuple) else (out, None)
        np.savez(cache, pred=pred, params=np.array(params, dtype=object))
        log(f"  {name} 完成 ({time.time() - t0:.1f}s)")
        return pred, params

    X, Y, Xte = ds.X_train, ds.Y_train, ds.X_test
    S, Ste = ds.S_train, ds.S_test
    H = config.HORIZON

    results = {
        # 参照基线：预测值恒等于 origin 当天实测流量
        "Persistence": (np.repeat(ds.q_now_test[:, None], H, axis=1), None),
    }
    results["ANN_Direct"] = run_model("ANN_Direct", ann.build_ann_direct, X, Y, Xte, verbose)
    results["ANN_MultiOutput"] = run_model("ANN_MultiOutput", ann.build_ann_multioutput, X, Y, Xte, verbose)
    results["RF_Direct"] = run_model("RF_Direct", rf.build_rf_direct, X, Y, Xte, verbose)
    results["RF_MultiOutput"] = run_model("RF_MultiOutput", rf.build_rf_multioutput, X, Y, Xte, verbose)
    results["LSTM_Direct"] = run_model("LSTM_Direct", lstm.build_lstm_direct, S, Y, Ste, verbose)
    results["LSTM_MultiOutput"] = run_model("LSTM_MultiOutput", lstm.build_lstm_multioutput, S, Y, Ste, verbose)
    results["Avg_Direct"] = run_model("Avg_Direct", ensemble.simple_average_direct, X, Y, Xte, S, Ste)
    results["Avg_MultiOutput"] = run_model("Avg_MultiOutput", ensemble.simple_average_multioutput, X, Y, Xte, S, Ste)
    results["Stack_Direct"] = run_model("Stack_Direct", ensemble.stacking_direct, X, Y, Xte, S, Ste)
    results["Stack_MultiOutput"] = run_model("Stack_MultiOutput", ensemble.stacking_multioutput, X, Y, Xte, S, Ste)

    # ---------- 四、评估与保存 ----------
    log("四、模型评估与结果保存 ...")
    summary_rows, all_metrics, per_horizon = [], {}, {}
    for name, (pred, params) in results.items():
        df_h = evaluate_horizons(ds.Y_test, pred)
        per_horizon[name] = df_h
        overall = {m: float(df_h[m].mean()) for m in config.METRICS}
        all_metrics[name] = {
            "per_horizon": df_h.round(4).to_dict(orient="index"),
            "overall": {k: round(v, 4) for k, v in overall.items()},
            "params": params,
        }
        if name != "Persistence":
            plot_predictions(ds.Y_test, pred, name)
            plot_timeseries(ds.Y_test, pred, name, ds.origins_test)
        summary_rows.append({"model": name, **overall})
        log(f"  {name:18s} NSE={overall['NSE']:6.3f} RMSE={overall['RMSE']:7.1f} "
            f"MAE={overall['MAE']:6.1f} R2={overall['R2']:6.3f} KGE={overall['KGE']:6.3f}")

    summary = pd.DataFrame(summary_rows).set_index("model")
    summary.round(4).to_csv(config.RESULT_DIR / "summary.csv", encoding="utf-8-sig")

    nse_tbl = pd.DataFrame({n: d["NSE"] for n, d in per_horizon.items()})
    nse_tbl.round(4).to_csv(config.RESULT_DIR / "nse_by_horizon.csv", encoding="utf-8-sig")

    with open(config.RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": ds.info, "feature_cols": ds.feature_cols, "models": all_metrics},
                  f, ensure_ascii=False, indent=2)

    plot_metric_by_horizon(per_horizon, "NSE")
    plot_metric_by_horizon(per_horizon, "KGE")
    plot_summary_bar(summary, "NSE")

    log(f"结果已保存: {config.RESULT_DIR}")
    print("\n各预见期 NSE：")
    print(nse_tbl.round(3).to_string())
    log("完成！")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于机器学习的径流预测")
    parser.add_argument("--skip-eda", action="store_true", help="跳过 EDA")
    parser.add_argument("--no-cache", action="store_true", help="忽略已有缓存，全部重训")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    run_all(skip_eda=args.skip_eda, verbose=args.verbose, use_cache=not args.no_cache)
