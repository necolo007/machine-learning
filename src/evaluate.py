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
    """Nash-Sutcliffe 效率系数。"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(obs, sim):
    """Kling-Gupta 效率系数。"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    r = np.corrcoef(obs, sim)[0, 1] if obs.std() > 0 else 0.0
    alpha = sim.std() / obs.std() if obs.std() > 0 else 0.0
    beta = sim.mean() / obs.mean() if obs.mean() != 0 else 0.0
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def evaluate(obs, sim):
    """计算多指标。"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    obs = np.asarray(obs, dtype=float).ravel()
    sim = np.asarray(sim, dtype=float).ravel()
    rmse = float(np.sqrt(mean_squared_error(obs, sim)))
    mae = float(mean_absolute_error(obs, sim))
    r2 = float(r2_score(obs, sim))
    return {
        "NSE": float(nse(obs, sim)),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "KGE": float(kge(obs, sim)),
    }


def evaluate_horizons(Y_true: np.ndarray, Y_pred: np.ndarray):
    """对每个预测步长 (1..HORIZON) 计算指标。"""
    rows = []
    for h in range(Y_true.shape[1]):
        m = evaluate(Y_true[:, h], Y_pred[:, h])
        m["horizon"] = h + 1
        rows.append(m)
    return pd.DataFrame(rows).set_index("horizon")


def plot_predictions(Y_true, Y_pred, name: str, origins=None):
    """绘制 1/3/7 天预测 vs 实测散点与时序。"""
    H = Y_true.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, h in zip(axes, [0, 2, 6]):
        if h >= H:
            continue
        ax.scatter(Y_true[:, h], Y_pred[:, h], s=8, alpha=0.5)
        lims = [min(Y_true[:, h].min(), Y_pred[:, h].min()),
                max(Y_true[:, h].max(), Y_pred[:, h].max())]
        ax.plot(lims, lims, "r--", lw=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name} t+{h+1}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{name} 预测 vs 实测")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"pred_{name}.png", dpi=120)
    plt.close(fig)


def plot_timeseries(Y_true, Y_pred, name: str, origins=None):
    """绘制 t+1 预测时序对比。"""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(Y_true))
    ax.plot(x, Y_true[:, 0], label="Observed", lw=1)
    ax.plot(x, Y_pred[:, 0], label="Predicted", lw=1, alpha=0.8)
    ax.set_title(f"{name} - t+1 时序对比 (测试集)")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Discharge")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"ts_{name}.png", dpi=120)
    plt.close(fig)
