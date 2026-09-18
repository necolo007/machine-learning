"""主流程：串联数据分析、特征工程、模型训练与评估。"""
import json
import time
import warnings

import numpy as np
import pandas as pd

from . import config
from .eda import run_eda
from .evaluate import evaluate_horizons, plot_predictions, plot_timeseries
from .features import prepare_dataset, prepare_sequence_dataset, prepare_both
from .models import ann, rf, lstm, ensemble

warnings.filterwarnings("ignore")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_all(skip_eda=False, verbose=0):
    results = {}

    # ---------- 一、数据分析 ----------
    if not skip_eda:
        log("一、数据分析 EDA ...")
        run_eda()

    # ---------- 二、特征工程 ----------
    log("二、特征工程：构建监督样本（扁平特征 + 序列特征，按相同 origin 对齐）...")
    Xtr, Ytr, Xte, Yte, Xtr_seq, Xte_seq, info = prepare_both(use_selection="pearson", k=20)
    log(f"特征信息: {info}")
    assert Ytr.shape[0] == Xtr_seq.shape[0]
    assert Yte.shape[0] == Xte_seq.shape[0]

    # ---------- 三、径流预测 ----------
    log("三、径流预测 ...")

    # 1) ANN
    log("  [1/5] ANN 多步直接预测 + GridSearchCV ...")
    pred, params = ann.build_ann_direct(Xtr, Ytr, Xte, verbose=verbose)
    results["ANN_Direct"] = {"pred": pred, "params": params}
    log("  [1/5] ANN 多输出预测 + GridSearchCV ...")
    pred, params = ann.build_ann_multioutput(Xtr, Ytr, Xte, verbose=verbose)
    results["ANN_MultiOutput"] = {"pred": pred, "params": params}

    # 2) RandomForest
    log("  [2/5] RandomForest 多步直接预测 + GridSearchCV ...")
    pred, params = rf.build_rf_direct(Xtr, Ytr, Xte, verbose=verbose)
    results["RF_Direct"] = {"pred": pred, "params": params}
    log("  [2/5] RandomForest 多输出预测 + GridSearchCV ...")
    pred, params = rf.build_rf_multioutput(Xtr, Ytr, Xte, verbose=verbose)
    results["RF_MultiOutput"] = {"pred": pred, "params": params}

    # 3) LSTM
    log("  [3/5] LSTM 多步直接预测 + 网格搜索 ...")
    pred, params = lstm.build_lstm_direct(Xtr_seq, Ytr, Xte_seq, verbose=verbose)
    results["LSTM_Direct"] = {"pred": pred, "params": params}
    log("  [3/5] LSTM 多输出预测 + 网格搜索 ...")
    pred, params = lstm.build_lstm_multioutput(Xtr_seq, Ytr, Xte_seq, verbose=verbose)
    results["LSTM_MultiOutput"] = {"pred": pred, "params": params}

    # 4) 集成学习 - 简单平均
    log("  [4/5] 集成学习 简单平均 (直接 + 多输出) ...")
    results["Avg_Direct"] = {"pred": ensemble.simple_average_direct(Xtr, Ytr, Xte, Xtr_seq, Xte_seq)}
    results["Avg_MultiOutput"] = {"pred": ensemble.simple_average_multioutput(Xtr, Ytr, Xte, Xtr_seq, Xte_seq)}

    # 5) 集成学习 - Stacking
    log("  [5/5] 集成学习 Stacking (直接 + 多输出) ...")
    results["Stack_Direct"] = {"pred": ensemble.stacking_direct(Xtr, Ytr, Xte, Xtr_seq, Xte_seq)}
    results["Stack_MultiOutput"] = {"pred": ensemble.stacking_multioutput(Xtr, Ytr, Xte, Xtr_seq, Xte_seq)}

    # ---------- 评估与保存 ----------
    log("四、模型评估与结果保存 ...")
    summary_rows = []
    all_metrics = {}
    for name, r in results.items():
        pred = r["pred"]
        df_h = evaluate_horizons(Yte, pred)
        overall = {m: float(df_h[m].mean()) for m in config.METRICS}
        all_metrics[name] = {
            "per_horizon": df_h.round(4).to_dict(orient="index"),
            "overall": {k: round(v, 4) for k, v in overall.items()},
            "params": r.get("params"),
        }
        plot_predictions(Yte, pred, name)
        plot_timeseries(Yte, pred, name)
        row = {"model": name}
        row.update({m: overall[m] for m in config.METRICS})
        summary_rows.append(row)
        log(f"  {name:20s} NSE={overall['NSE']:.3f} RMSE={overall['RMSE']:.2f} "
            f"MAE={overall['MAE']:.2f} R2={overall['R2']:.3f} KGE={overall['KGE']:.3f}")

    summary = pd.DataFrame(summary_rows).set_index("model")
    summary.to_csv(config.RESULT_DIR / "summary.csv", encoding="utf-8-sig")
    with open(config.RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    log(f"汇总结果保存至: {config.RESULT_DIR / 'summary.csv'}")
    log(f"详细指标保存至: {config.RESULT_DIR / 'metrics.json'}")
    log("完成！")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eda", action="store_true", help="跳过 EDA")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    run_all(skip_eda=args.skip_eda, verbose=args.verbose)
