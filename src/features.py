"""特征工程：时间特征、滞后特征、窗口特征、归一化、特征选择。"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config

TARGET = config.TARGET
LOOKBACK = config.LOOKBACK
HORIZON = config.HORIZON


# ----------------------------------------------------------------------
# 1. 时间序列特征提取
# ----------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """对四季和月份进行编码（三角化 + 季节 one-hot）。"""
    out = df.copy()
    month = out.index.month.to_numpy()
    dayofyear = out.index.dayofyear.to_numpy()
    # 月份三角化编码
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    # 年内日序三角化
    out["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)
    # 季节 one-hot（3-5 春=1，6-8 夏=2，9-11 秋=3，12-2 冬=4）
    season = np.where((month >= 3) & (month <= 5), 1,
             np.where((month >= 6) & (month <= 8), 2,
             np.where((month >= 9) & (month <= 11), 3, 4)))
    for s, name in zip([1, 2, 3, 4], ["spring", "summer", "autumn", "winter"]):
        out[f"season_{name}"] = (season == s).astype(int)
    return out


def add_lag_and_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取滞后特征和窗口（滚动）特征。"""
    out = df.copy()
    # 滞后特征：径流的 lag 0..LOOKBACK
    for lag in range(LOOKBACK + 1):
        out[f"{TARGET}_lag{lag}"] = out[TARGET].shift(lag)
    # 窗口特征：过去 LOOKBACK 天的均值/标准差/最大值
    roll = out[TARGET].rolling(LOOKBACK)
    out[f"{TARGET}_roll_mean"] = roll.mean().shift(1)
    out[f"{TARGET}_roll_std"] = roll.std().shift(1)
    out[f"{TARGET}_roll_max"] = roll.max().shift(1)
    # 降水窗口累计
    out["Prcp_roll_sum"] = out["Prcp"].rolling(LOOKBACK).sum().shift(1)
    return out


# ----------------------------------------------------------------------
# 2. 特征归一化
# ----------------------------------------------------------------------
@dataclass
class Normalizer:
    """最小-最大归一化（按列保存 min/max）。"""
    mins: np.ndarray = None
    maxs: np.ndarray = None
    cols: List[str] = None

    def fit(self, X: np.ndarray, cols: List[str]):
        self.cols = cols
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rng = np.where(self.maxs - self.mins < 1e-9, 1.0, self.maxs - self.mins)
        return (X - self.mins) / rng


def zscore_standardize(X: np.ndarray):
    """Z-score 标准化（返回标准化后数据及统计量）。"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (X - mean) / std, mean, std


# ----------------------------------------------------------------------
# 3. 特征选择
# ----------------------------------------------------------------------
def pearson_select(features: pd.DataFrame, y: pd.Series, k: int = 15) -> List[str]:
    """皮尔逊相关系数法：选择与目标相关性绝对值最大的 k 个特征。"""
    corrs = features.corrwith(y).abs().sort_values(ascending=False)
    return corrs.head(k).index.tolist()


def mutual_info_select(features: pd.DataFrame, y: pd.Series, k: int = 15) -> List[str]:
    """互信息法选择特征。"""
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(features.fillna(0), y, random_state=config.SEED)
    s = pd.Series(mi, index=features.columns).sort_values(ascending=False)
    return s.head(k).index.tolist()


# ----------------------------------------------------------------------
# 构建监督学习样本
# ----------------------------------------------------------------------
def build_supervised(df: pd.DataFrame, feature_cols: List[str]):
    """
    构建监督学习样本：
      X[i] = 时刻 t 的特征（已含滞后/窗口/时间编码）
      Y[i] = [Discharge[t+1], ..., Discharge[t+HORIZON]]
    origin t 取自 df.index，要求 t-HORIZON..t+HORIZON 均存在。
    """
    X, Y, origins = [], [], []
    vals = df[feature_cols].values
    tgt = df[TARGET].values
    idx = df.index
    n = len(df)
    for t in range(n):
        if t + HORIZON >= n:
            break
        X.append(vals[t])
        Y.append(tgt[t + 1: t + 1 + HORIZON])
        origins.append(idx[t])
    X = np.array(X)
    Y = np.array(Y)
    origins = pd.DatetimeIndex(origins)
    return X, Y, origins


def split_by_time(origins: pd.DatetimeIndex, X: np.ndarray, Y: np.ndarray):
    """
    按时间顺序切分：origin 时刻落在训练期/测试期。
    训练集：origin 时刻 <= TRAIN_END_YEAR 且其预测窗口完全在训练年内
    测试集：origin 时刻的预测窗口完全落在 TEST_YEAR 内
    """
    train_mask = origins.year <= config.TRAIN_END_YEAR
    # 测试 origin：t+1..t+HORIZON 全部落在 TEST_YEAR
    test_mask = pd.DatetimeIndex([
        origins[i] + pd.Timedelta(days=h) for i in range(len(origins)) for h in [1]
    ])
    # 用 origin + HORIZON 天判断窗口末端
    end_window = origins + pd.Timedelta(days=config.HORIZON)
    start_window = origins + pd.Timedelta(days=1)
    test_mask = (start_window.year == config.TEST_YEAR) & (end_window.year == config.TEST_YEAR)
    return (X[train_mask], Y[train_mask], origins[train_mask],
            X[test_mask], Y[test_mask], origins[test_mask])


def build_sequences(df: pd.DataFrame, seq_cols: List[str]):
    """
    为 LSTM 构造 3D 序列样本：
      X[i] = [seq[t-LOOKBACK+1], ..., seq[t]]  形状 (LOOKBACK, F)
      Y[i] = [Discharge[t+1], ..., Discharge[t+HORIZON]]
    """
    data = df[seq_cols].values
    tgt = df[TARGET].values
    idx = df.index
    n = len(df)
    X, Y, origins = [], [], []
    for t in range(LOOKBACK - 1, n):
        if t + HORIZON >= n:
            break
        X.append(data[t - LOOKBACK + 1: t + 1])
        Y.append(tgt[t + 1: t + 1 + HORIZON])
        origins.append(idx[t])
    return np.array(X), np.array(Y), pd.DatetimeIndex(origins)


def prepare_sequence_dataset():
    """为 LSTM 准备 3D 序列数据，并按时间切分 + 归一化。"""
    from .data_loader import load_data
    df = load_data()
    seq_cols = [TARGET] + ["Prcp", "Tmax", "Tmin", "Srad", "Vp", "Dayl"]
    X, Y, origins = build_sequences(df, seq_cols)
    # 按 origin 时间切分
    end_window = origins + pd.Timedelta(days=config.HORIZON)
    start_window = origins + pd.Timedelta(days=1)
    train_mask = origins.year <= config.TRAIN_END_YEAR
    test_mask = (start_window.year == config.TEST_YEAR) & (end_window.year == config.TEST_YEAR)
    Xtr, Ytr = X[train_mask], Y[train_mask]
    Xte, Yte = X[test_mask], Y[test_mask]

    # 在训练集上拟合 MinMax 归一化（按特征列）
    n_feat = Xtr.shape[2]
    mins = Xtr.reshape(-1, n_feat).min(axis=0)
    maxs = Xtr.reshape(-1, n_feat).max(axis=0)
    rng = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
    Xtr_n = (Xtr - mins) / rng
    Xte_n = (Xte - mins) / rng
    info = {
        "seq_cols": seq_cols,
        "n_features": len(seq_cols),
        "lookback": config.LOOKBACK,
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }
    return Xtr_n, Ytr, Xte_n, Yte, info


def prepare_both(use_selection: str = "pearson", k: int = 20):
    """
    同时构建扁平特征与序列特征，并保证两者样本按相同 origin 对齐。
    返回: (Xtr_flat, Ytr, Xte_flat, Yte, Xtr_seq, Xte_seq, info)
    """
    from .data_loader import load_data
    df = load_data()
    df_time = add_time_features(df)
    df_feat = add_lag_and_window_features(df_time)
    df_clean = df_feat.dropna()

    # 扁平特征候选 + 特征选择
    feature_cols = [c for c in df_clean.columns if c != TARGET]
    y_series = df_clean[TARGET]
    if use_selection == "pearson":
        selected = pearson_select(df_clean[feature_cols], y_series, k=k)
    elif use_selection == "mi":
        selected = mutual_info_select(df_clean[feature_cols], y_series, k=k)
    else:
        selected = feature_cols

    Xf, Y, origins = build_supervised(df_clean, selected)

    # 序列特征：从 df_time（含全部行）构造，再按 origin 对齐到扁平样本
    seq_cols = [TARGET] + ["Prcp", "Tmax", "Tmin", "Srad", "Vp", "Dayl"]
    Xs_all, Ys_all, origins_seq = build_sequences(df_time, seq_cols)
    # 对齐 origins
    seq_idx = {d: i for i, d in enumerate(origins_seq)}
    keep = [seq_idx[d] for d in origins if d in seq_idx]
    Xs = Xs_all[keep]
    assert len(Xs) == len(Xf), (len(Xs), len(Xf))

    # 按时间切分
    end_window = origins + pd.Timedelta(days=config.HORIZON)
    start_window = origins + pd.Timedelta(days=1)
    train_mask = origins.year <= config.TRAIN_END_YEAR
    test_mask = (start_window.year == config.TEST_YEAR) & (end_window.year == config.TEST_YEAR)

    Xtr_f, Ytr = Xf[train_mask], Y[train_mask]
    Xte_f, Yte = Xf[test_mask], Y[test_mask]
    Xtr_s = Xs[train_mask]
    Xte_s = Xs[test_mask]

    # 扁平特征 MinMax 归一化
    mins = Xtr_f.min(axis=0)
    maxs = Xtr_f.max(axis=0)
    rng = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
    Xtr_f_n = (Xtr_f - mins) / rng
    Xte_f_n = (Xte_f - mins) / rng

    # 序列特征 MinMax 归一化
    n_feat = Xtr_s.shape[2]
    s_mins = Xtr_s.reshape(-1, n_feat).min(axis=0)
    s_maxs = Xtr_s.reshape(-1, n_feat).max(axis=0)
    s_rng = np.where(s_maxs - s_mins < 1e-9, 1.0, s_maxs - s_mins)
    Xtr_s_n = (Xtr_s - s_mins) / s_rng
    Xte_s_n = (Xte_s - s_mins) / s_rng

    info = {
        "feature_cols": selected,
        "n_features_flat": len(selected),
        "n_features_seq": len(seq_cols),
        "lookback": config.LOOKBACK,
        "n_train": len(Xtr_f),
        "n_test": len(Xte_f),
        "train_period": f"{origins[train_mask].min().date()} ~ {origins[train_mask].max().date()}",
        "test_period": f"{origins[test_mask].min().date()} ~ {origins[test_mask].max().date()}",
    }
    return Xtr_f_n, Ytr, Xte_f_n, Yte, Xtr_s_n, Xte_s_n, info


def prepare_dataset(use_selection: str = "pearson", k: int = 20):
    """端到端：加载→特征工程→特征选择→切分→归一化。"""
    from .data_loader import load_data
    df = load_data()
    df = add_time_features(df)
    df = add_lag_and_window_features(df)
    df = df.dropna()

    # 候选特征：去掉目标本身
    feature_cols = [c for c in df.columns if c != TARGET]
    # 与目标相关性
    y_series = df[TARGET]
    if use_selection == "pearson":
        selected = pearson_select(df[feature_cols], y_series, k=k)
    elif use_selection == "mi":
        selected = mutual_info_select(df[feature_cols], y_series, k=k)
    else:  # "all"
        selected = feature_cols

    X, Y, origins = build_supervised(df, selected)
    Xtr, Ytr, otr, Xte, Yte, ote = split_by_time(origins, X, Y)

    # 最小-最大归一化（仅在训练集上拟合）
    norm = Normalizer().fit(Xtr, selected)
    Xtr_n = norm.transform(Xtr)
    Xte_n = norm.transform(Xte)

    info = {
        "feature_cols": selected,
        "n_features": len(selected),
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "train_period": f"{otr.min().date()} ~ {otr.max().date()}",
        "test_period": f"{ote.min().date()} ~ {ote.max().date()}",
    }
    return Xtr_n, Ytr, Xte_n, Yte, info
