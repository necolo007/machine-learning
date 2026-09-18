"""各模型共用的工具：目标变换、时序交叉验证、预测后处理。"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from .. import config


def cv_splitter(n_splits: int = None):
    """时序交叉验证。时间序列不能打乱顺序，故用 TimeSeriesSplit 替代 KFold。"""
    return TimeSeriesSplit(n_splits=n_splits or config.CV_SPLITS)


def forward(y: np.ndarray) -> np.ndarray:
    """目标正变换（默认取自然对数）。"""
    if not config.LOG_TARGET:
        return y
    return np.log(np.clip(y, 1e-6, None))


def inverse(z: np.ndarray) -> np.ndarray:
    """目标反变换。"""
    if not config.LOG_TARGET:
        return z
    return np.exp(np.clip(z, -20, 20))


def clip_to_range(pred: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """将预测限制在训练期观测流量的合理范围内，避免元学习器/线性外推产生离谱值。"""
    lo = float(np.min(y_train)) * 0.5
    hi = float(np.max(y_train)) * 1.2
    return np.clip(pred, lo, hi)
