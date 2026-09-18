"""EDA：数据分析与可视化（纯 matplotlib 实现，无需 seaborn）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data_loader import load_data, describe_fields

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_time_series(df: pd.DataFrame):
    """时间序列可视化：径流与主要气象要素。"""
    cols = [config.TARGET, "Prcp", "Tmax", "Tmin", "Srad"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.4 * len(cols)), sharex=True)
    for ax, c in zip(axes, cols):
        ax.plot(df.index, df[c], linewidth=0.7)
        ax.set_ylabel(c)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("时间序列可视化 (2000-2004)")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "ts_series.png", dpi=120)
    plt.close(fig)


def plot_histograms(df: pd.DataFrame):
    """直方图：各变量分布。"""
    cols = [config.TARGET] + config.METEO_COLS
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(df[c], bins=50, color="steelblue", edgecolor="white")
        ax.set_title(c)
        ax.set_yscale("log")
    fig.suptitle("变量分布直方图")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "histograms.png", dpi=120)
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame):
    """箱型图：按月份查看径流/降水分布。"""
    df = df.copy()
    df["month"] = df.index.month
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    months = sorted(df["month"].unique())
    for ax, c in zip(axes, [config.TARGET, "Prcp"]):
        data = [df.loc[df["month"] == m, c].values for m in months]
        ax.boxplot(data, tick_labels=months, showfliers=True)
        ax.set_xlabel("Month")
        ax.set_ylabel(c)
        ax.set_title(f"各月{c}箱型图")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "boxplots.png", dpi=120)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame):
    """皮尔逊相关系数热力图。"""
    cols = [config.TARGET] + config.METEO_COLS
    corr = df[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("皮尔逊相关系数热力图")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)
    return corr


def run_eda():
    """执行完整 EDA 流程。"""
    print(describe_fields())
    df = load_data()
    print(f"数据规模: {df.shape}, 时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    print("\n描述性统计:")
    print(df.describe().round(2).to_string())

    plot_time_series(df)
    plot_histograms(df)
    plot_boxplots(df)
    corr = plot_correlation(df)
    print("\n皮尔逊相关系数:")
    print(corr.round(3).to_string())
    print(f"\nEDA 图表已保存至: {config.FIG_DIR}")
    return df


if __name__ == "__main__":
    run_eda()
