"""特征工程：时间序列特征提取、特征归一化、特征选择。

对应实验指导书"二、特征工程"：
  1. 时间序列特征提取：四季/月份编码、滞后特征、窗口特征
  2. 特征归一化：最小-最大归一化（MinMax）与 Z-score 标准化
  3. 特征选择：皮尔逊相关系数法与互信息法

同时产出两套对齐的样本：
  - 扁平特征 (n, F)：供 MLP / 随机森林使用
  - 序列特征 (n, L, C)：供 LSTM 使用
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from . import config
from .data_loader import load_data

TARGET = config.TARGET
LOOKBACK = config.LOOKBACK
HORIZON = config.HORIZON


# ----------------------------------------------------------------------
# 1. 时间序列特征提取
# ----------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """对四季和月份进行编码：月份/年内日序三角化 + 季节 one-hot。"""
    out = df.copy()
    month = out.index.month.to_numpy()
    doy = out.index.dayofyear.to_numpy()
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # 季节：3-5 春，6-8 夏，9-11 秋，12-2 冬
    season = np.where((month >= 3) & (month <= 5), 1,
             np.where((month >= 6) & (month <= 8), 2,
             np.where((month >= 9) & (month <= 11), 3, 4)))
    for code, name in zip([1, 2, 3, 4], ["spring", "summer", "autumn", "winter"]):
        out[f"season_{name}"] = (season == code).astype(float)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """滞后特征：对数径流 lag0~lag7、径流涨落率、气象要素近期滞后。"""
    out = df.copy()
    logq = np.log(out[TARGET])
    for lag in range(LOOKBACK + 1):
        out[f"logQ_lag{lag}"] = logq.shift(lag)
    # 涨落率：区分洪水的涨水段与退水段，对多步预测非常关键
    out["dlogQ_1"] = logq.diff(1)
    out["dlogQ_1_prev"] = logq.diff(1).shift(1)
    out["dlogQ_3"] = logq.diff(3)
    for col in config.FORCING_COLS:
        for lag in range(1, 4):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """窗口（滚动）特征：径流统计量、前期降水指数、正积温（融雪代理）。"""
    out = df.copy()
    logq = np.log(out[TARGET])
    for w in config.ROLL_WINDOWS:
        roll = logq.rolling(w)
        out[f"logQ_mean{w}"] = roll.mean()
        out[f"logQ_std{w}"] = roll.std()
        out[f"logQ_max{w}"] = roll.max()
        out[f"logQ_min{w}"] = roll.min()
        # 前期降水指数 API：流域蓄水状态的代理量
        out[f"Prcp_sum{w}"] = out["Prcp"].rolling(w).sum()
        out[f"Tmax_mean{w}"] = out["Tmax"].rolling(w).mean()
    # 正积温：融雪径流的主要驱动
    pdd = out["Tmax"].clip(lower=0)
    out["PDD"] = pdd
    for w in [3, 7, 14]:
        out[f"PDD_sum{w}"] = pdd.rolling(w).sum()
    out["frozen_days7"] = (out["Tmax"] < 0).rolling(7).sum()
    # 降雨与融雪的联合作用（雨夹雪/雨引发融雪）
    out["Prcp_x_PDD"] = out["Prcp"] * pdd
    return out


def add_future_forcing(df: pd.DataFrame) -> pd.DataFrame:
    """未来气象强迫（"完美预报"假设）：未来 1~HORIZON 天的降水与正积温。

    径流预报的业务流程是"数值天气预报 → 水文模型 → 流量预报"，
    未来降水是必备输入；此处以实测气象代替预报值。
    """
    out = df.copy()
    pdd = out["Tmax"].clip(lower=0)
    for h in range(1, HORIZON + 1):
        out[f"fut_Prcp_{h}"] = out["Prcp"].shift(-h)
        out[f"fut_PDD_{h}"] = pdd.shift(-h)
    out["fut_Prcp_sum"] = sum(out["Prcp"].shift(-h) for h in range(1, HORIZON + 1))
    out["fut_PDD_sum"] = sum(pdd.shift(-h) for h in range(1, HORIZON + 1))
    return out


def build_feature_frame(use_future_forcing: Optional[bool] = None) -> pd.DataFrame:
    """加载原始数据并完成全部特征构造，返回含目标列的特征表。"""
    if use_future_forcing is None:
        use_future_forcing = config.USE_FUTURE_FORCING
    df = load_data()
    # Swe 全为 0，无信息量，剔除
    df = df.drop(columns=[c for c in ["Swe"] if c in df.columns])
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_window_features(df)
    if use_future_forcing:
        df = add_future_forcing(df)
    return df


# ----------------------------------------------------------------------
# 2. 特征归一化
# ----------------------------------------------------------------------
# 常用方法小结：
#   - 最小-最大归一化 (MinMax)：线性映射到 [0,1]，保留原分布形状，对异常值敏感
#   - Z-score 标准化：减均值除标准差，得到零均值单位方差，受异常值影响较小
#   - 最大绝对值 / 稳健(分位数)缩放：适合稀疏或重尾数据
# 本项目默认使用 MinMax（仅在训练集上拟合，再应用到测试集，避免信息泄漏）。
@dataclass
class MinMaxNormalizer:
    """最小-最大归一化：x' = (x - min) / (max - min)。"""
    mins: np.ndarray = None
    maxs: np.ndarray = None
    cols: List[str] = None

    def fit(self, X: np.ndarray, cols: Optional[List[str]] = None):
        self.cols = cols
        self.mins = np.nanmin(X, axis=0)
        self.maxs = np.nanmax(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rng = np.where(self.maxs - self.mins < 1e-9, 1.0, self.maxs - self.mins)
        return (X - self.mins) / rng

    def fit_transform(self, X, cols=None):
        return self.fit(X, cols).transform(X)


@dataclass
class ZScoreNormalizer:
    """Z-score 标准化：x' = (x - mean) / std。"""
    mean: np.ndarray = None
    std: np.ndarray = None
    cols: List[str] = None

    def fit(self, X: np.ndarray, cols: Optional[List[str]] = None):
        self.cols = cols
        self.mean = np.nanmean(X, axis=0)
        self.std = np.where(np.nanstd(X, axis=0) < 1e-9, 1.0, np.nanstd(X, axis=0))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit_transform(self, X, cols=None):
        return self.fit(X, cols).transform(X)


# ----------------------------------------------------------------------
# 3. 特征选择
# ----------------------------------------------------------------------
# 常用方法小结：
#   - 过滤式：皮尔逊相关系数（线性相关）、互信息（含非线性）、方差阈值、卡方
#   - 包裹式：递归特征消除 RFE、前向/后向搜索
#   - 嵌入式：Lasso 系数、树模型特征重要性
# 本项目实现皮尔逊相关系数法与互信息法；为避免信息泄漏，只在训练集样本上做选择。
#
# 注意：径流滞后项之间相关系数普遍 > 0.95，若只按"与目标相关性"排序取 Top-k，
# 结果几乎全是彼此冗余的滞后项，而降水等互补信息会被挤出。因此在打分之后
# 再做一步贪心冗余剔除（相关性高于阈值的候选跳过），即 max-relevance/min-redundancy 思路。
REDUNDANCY_THRESHOLD = 0.95


def _greedy_pick(scores: pd.Series, features: pd.DataFrame, k: int,
                 threshold: float = REDUNDANCY_THRESHOLD) -> List[str]:
    """按得分从高到低贪心选取，跳过与已选特征相关性超过阈值的候选。"""
    corr = features.corr().abs().fillna(0.0)
    picked: List[str] = []
    for col in scores.sort_values(ascending=False).index:
        if len(picked) >= k:
            break
        if picked and corr.loc[col, picked].max() > threshold:
            continue
        picked.append(col)
    return picked


def pearson_scores(features: pd.DataFrame, Y: np.ndarray) -> pd.Series:
    """各特征与目标的皮尔逊相关系数（取各预见期中的最大 |r|）。"""
    scores = [features.corrwith(pd.Series(np.log(Y[:, h]), index=features.index)).abs()
              for h in range(Y.shape[1])]
    return pd.concat(scores, axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def mutual_info_scores(features: pd.DataFrame, Y: np.ndarray) -> pd.Series:
    """各特征与目标的互信息（取各预见期中的最大值），可捕捉非线性依赖。"""
    from sklearn.feature_selection import mutual_info_regression
    mi = [mutual_info_regression(features.values, np.log(Y[:, h]), random_state=config.SEED)
          for h in range(Y.shape[1])]
    return pd.Series(np.max(mi, axis=0), index=features.columns)


def pearson_select(features: pd.DataFrame, Y: np.ndarray, k: int = 30) -> List[str]:
    """皮尔逊相关系数法 + 冗余剔除。"""
    return _greedy_pick(pearson_scores(features, Y), features, k)


def mutual_info_select(features: pd.DataFrame, Y: np.ndarray, k: int = 30) -> List[str]:
    """互信息法 + 冗余剔除。"""
    return _greedy_pick(mutual_info_scores(features, Y), features, k)


def select_features(features: pd.DataFrame, Y: np.ndarray,
                    method: str = "pearson", k: int = 30) -> List[str]:
    if method == "pearson":
        return pearson_select(features, Y, k)
    if method == "mi":
        return mutual_info_select(features, Y, k)
    return list(features.columns)


# ----------------------------------------------------------------------
# 4. 监督样本构造与按时间切分
# ----------------------------------------------------------------------
def build_supervised(df: pd.DataFrame, feature_cols: List[str]):
    """X[i] = 时刻 t 的特征向量；Y[i] = [Q(t+1), ..., Q(t+HORIZON)]。"""
    vals = df[feature_cols].to_numpy(dtype=float)
    tgt = df[TARGET].to_numpy(dtype=float)
    n = len(df)
    n_samples = n - HORIZON
    X = vals[:n_samples]
    Y = np.stack([tgt[t + 1: t + 1 + HORIZON] for t in range(n_samples)])
    return X, Y, df.index[:n_samples]


def build_sequences(df: pd.DataFrame, use_future_forcing: bool):
    """LSTM 的 3D 序列样本。

    编码窗口为 [t-LOOKBACK+1, t]；若使用未来气象强迫，则再向后拼接
    [t+1, t+HORIZON] 共 HORIZON 步，其中径流通道用 t 时刻值保持（未知），
    气象通道用实测（预报）值，并附 is_future 标志位区分两段。
    """
    logq = np.log(df[TARGET]).to_numpy(dtype=float)
    pdd = df["Tmax"].clip(lower=0).to_numpy(dtype=float)
    forcing = df[config.FORCING_COLS].to_numpy(dtype=float)
    channels = np.column_stack([logq, pdd, forcing])  # (n, C-1)
    tgt = df[TARGET].to_numpy(dtype=float)
    n, c = channels.shape
    steps = LOOKBACK + (HORIZON if use_future_forcing else 0)

    X, Y, origins = [], [], []
    for t in range(LOOKBACK - 1, n - HORIZON):
        hist = channels[t - LOOKBACK + 1: t + 1]
        if use_future_forcing:
            fut = channels[t + 1: t + 1 + HORIZON].copy()
            fut[:, 0] = channels[t, 0]  # 未来径流未知，用 t 时刻值占位
            seq = np.vstack([hist, fut])
            flag = np.concatenate([np.zeros(LOOKBACK), np.ones(HORIZON)])[:, None]
            seq = np.hstack([seq, flag])
        else:
            seq = np.hstack([hist, np.zeros((LOOKBACK, 1))])
        X.append(seq)
        Y.append(tgt[t + 1: t + 1 + HORIZON])
        origins.append(df.index[t])
    X = np.asarray(X, dtype=float)
    assert X.shape[1] == steps and X.shape[2] == c + 1
    return X, np.asarray(Y, dtype=float), pd.DatetimeIndex(origins)


def time_split_masks(origins: pd.DatetimeIndex):
    """按时间先后切分：训练集 origin 在前 4 年；测试集预测窗口完整落在测试年内。"""
    train_mask = np.asarray(origins.year <= config.TRAIN_END_YEAR)
    start = origins + pd.Timedelta(days=1)
    end = origins + pd.Timedelta(days=HORIZON)
    test_mask = np.asarray((start.year == config.TEST_YEAR) & (end.year == config.TEST_YEAR))
    return train_mask, test_mask


# ----------------------------------------------------------------------
# 5. 端到端数据准备
# ----------------------------------------------------------------------
@dataclass
class Dataset:
    """训练/测试数据容器（扁平特征与序列特征按同一组 origin 对齐）。"""
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    S_train: np.ndarray
    S_test: np.ndarray
    origins_train: pd.DatetimeIndex
    origins_test: pd.DatetimeIndex
    q_now_test: np.ndarray          # 测试集 origin 当天实测流量（持续性基线用）
    feature_cols: List[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def prepare_dataset(method: Optional[str] = None, k: Optional[int] = None,
                    use_future_forcing: Optional[bool] = None,
                    normalizer: str = "minmax") -> Dataset:
    """加载 → 特征工程 → 按时间切分 → 训练集上做特征选择与归一化。"""
    method = method or config.SELECT_METHOD
    k = k or config.N_SELECT
    if use_future_forcing is None:
        use_future_forcing = config.USE_FUTURE_FORCING

    df = build_feature_frame(use_future_forcing)
    # 序列样本先于 dropna 构造，保证 LSTM 能用到完整历史
    S_all, Y_seq, origins_seq = build_sequences(df, use_future_forcing)

    df_clean = df.dropna()
    candidate_cols = [c for c in df_clean.columns if c != TARGET]
    X_all, Y_all, origins = build_supervised(df_clean, candidate_cols)

    # 两套样本按 origin 对齐
    pos = {d: i for i, d in enumerate(origins_seq)}
    keep = np.array([i for i, d in enumerate(origins) if d in pos])
    X_all, Y_all, origins = X_all[keep], Y_all[keep], origins[keep]
    S_all = S_all[[pos[d] for d in origins]]
    assert np.allclose(Y_all, Y_seq[[pos[d] for d in origins]])

    train_mask, test_mask = time_split_masks(origins)
    Xtr_raw, Ytr = X_all[train_mask], Y_all[train_mask]
    Xte_raw, Yte = X_all[test_mask], Y_all[test_mask]
    Str_raw, Ste_raw = S_all[train_mask], S_all[test_mask]

    # 特征选择：仅用训练集样本，目标为对数流量（抑制极值主导）
    feat_df = pd.DataFrame(Xtr_raw, columns=candidate_cols)
    selected = select_features(feat_df, Ytr, method, k)
    # 保持与原特征表一致的列序，便于阅读
    selected = [c for c in candidate_cols if c in selected]
    col_idx = [candidate_cols.index(c) for c in selected]

    Norm = MinMaxNormalizer if normalizer == "minmax" else ZScoreNormalizer
    norm = Norm().fit(Xtr_raw[:, col_idx], selected)
    Xtr = norm.transform(Xtr_raw[:, col_idx])
    Xte = norm.transform(Xte_raw[:, col_idx])

    # 序列特征按通道归一化（在训练集上拟合）
    n_ch = Str_raw.shape[2]
    snorm = Norm().fit(Str_raw.reshape(-1, n_ch))
    Str = snorm.transform(Str_raw.reshape(-1, n_ch)).reshape(Str_raw.shape)
    Ste = snorm.transform(Ste_raw.reshape(-1, n_ch)).reshape(Ste_raw.shape)

    q_now = df_clean[TARGET].reindex(origins[test_mask]).to_numpy(dtype=float)

    info = {
        "n_candidate_features": len(candidate_cols),
        "n_selected_features": len(selected),
        "select_method": method,
        "normalizer": normalizer,
        "use_future_forcing": bool(use_future_forcing),
        "log_target": bool(config.LOG_TARGET),
        "seq_shape": list(Str.shape[1:]),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "train_period": f"{origins[train_mask].min().date()} ~ {origins[train_mask].max().date()}",
        "test_period": f"{(origins[test_mask].min() + pd.Timedelta(days=1)).date()}"
                       f" ~ {(origins[test_mask].max() + pd.Timedelta(days=HORIZON)).date()}",
    }
    return Dataset(Xtr, Ytr, Xte, Yte, Str, Ste,
                   origins[train_mask], origins[test_mask], q_now, selected, info)
