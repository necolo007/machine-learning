"""评估指标与可视化。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def nse(obs, sim):
    """Nash-Sutcliffe 效率系数：1 为完美，0 相当于用观测均值预测。"""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(obs, sim):
    """Kling-Gupta 效率系数：综合相关性、变差比与偏差比。"""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if obs.std() == 0:
        return 0.0
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean() if obs.mean() != 0 else 0.0
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def evaluate(obs, sim):
    """计算 NSE / RMSE / MAE / R2 / KGE。"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    obs = np.asarray(obs, float).ravel()
    sim = np.asarray(sim, float).ravel()
    return {
        "NSE": float(nse(obs, sim)),
        "RMSE": float(np.sqrt(mean_squared_error(obs, sim))),
        "MAE": float(mean_absolute_error(obs, sim)),
        "R2": float(r2_score(obs, sim)),
        "KGE": float(kge(obs, sim)),
    }


def evaluate_horizons(Y_true: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
    """对每个预测步长 (t+1 .. t+HORIZON) 分别计算指标。"""
    rows = []
    for h in range(Y_true.shape[1]):
        m = evaluate(Y_true[:, h], Y_pred[:, h])
        m["horizon"] = h + 1
        rows.append(m)
    return pd.DataFrame(rows).set_index("horizon")


def plot_predictions(Y_true, Y_pred, name: str):
    """散点图：t+1 / t+3 / t+7 的预测 vs 实测（双对数坐标，兼顾枯水与洪峰）。"""
    H = Y_true.shape[1]
    hs = [h for h in (0, 2, 6) if h < H]
    fig, axes = plt.subplots(1, len(hs), figsize=(5 * len(hs), 4.6))
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, hs):
        obs, sim = Y_true[:, h], np.clip(Y_pred[:, h], 1e-3, None)
        ax.scatter(obs, sim, s=10, alpha=0.5, edgecolors="none")
        lim = [min(obs.min(), sim.min()) * 0.8, max(obs.max(), sim.max()) * 1.2]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("实测流量 (cfs)")
        ax.set_ylabel("预测流量 (cfs)")
        ax.set_title(f"t+{h + 1}  NSE={nse(obs, sim):.3f}")
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle(f"{name} 预测 vs 实测")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"pred_{name}.png", dpi=120)
    plt.close(fig)


def plot_timeseries(Y_true, Y_pred, name: str, origins=None):
    """时序对比：t+1 与 t+7 两个预见期。"""
    H = Y_true.shape[1]
    hs = [h for h in (0, H - 1) if h < H]
    fig, axes = plt.subplots(len(hs), 1, figsize=(12, 3.2 * len(hs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, hs):
        # 横轴为被预测日期，即 origin + (h+1) 天
        x = (origins + pd.Timedelta(days=h + 1)) if origins is not None else np.arange(len(Y_true))
        ax.plot(x, Y_true[:, h], label="实测", lw=1.1, color="k")
        ax.plot(x, Y_pred[:, h], label="预测", lw=1.1, color="tab:red", alpha=0.85)
        ax.set_ylabel("流量 (cfs)")
        ax.set_title(f"预见期 t+{h + 1}  NSE={nse(Y_true[:, h], Y_pred[:, h]):.3f}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{name} 测试集时序对比 ({config.TEST_YEAR} 年)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"ts_{name}.png", dpi=120)
    plt.close(fig)


def plot_metric_by_horizon(per_horizon: dict, metric: str = "NSE"):
    """各模型指标随预见期变化的折线图。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, df in per_horizon.items():
        style = "--o" if name == "Persistence" else "-o"
        ax.plot(df.index, df[metric], style, ms=4, lw=1.4, label=name)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("预见期 (天)")
    ax.set_ylabel(metric)
    ax.set_title(f"各模型 {metric} 随预见期的变化")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"metric_{metric}_by_horizon.png", dpi=120)
    plt.close(fig)


def plot_summary_bar(summary: pd.DataFrame, metric: str = "NSE"):
    """各模型平均指标柱状图。"""
    s = summary[metric].sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = ["tab:grey" if i == "Persistence" else "tab:blue" for i in s.index]
    ax.barh(s.index, s.values, color=colors)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set_xlabel(f"平均 {metric}（1-7 天）")
    ax.set_title(f"模型平均 {metric} 对比")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"summary_{metric}.png", dpi=120)
    plt.close(fig)
