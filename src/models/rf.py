"""基于随机森林的径流多步预测模型。

  - 多步直接预测 (Direct)：每个 horizon 训练一个 RandomForestRegressor（共 7 个）
  - 多输出预测 (MultiOutput)：单个 MultiOutputRegressor(RandomForest) 同时输出 7 步
  - GridSearchCV（cv=TimeSeriesSplit，不打乱时间顺序）确定超参数
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

from .. import config
from .common import cv_splitter, forward, inverse, clip_to_range

PARAM_GRID = {
    "n_estimators": [300],
    "max_depth": [None, 12, 20],
    "min_samples_leaf": [1, 2, 5],
    "max_features": [0.5, 1.0],
}


def _base_rf():
    return RandomForestRegressor(random_state=config.SEED, n_jobs=-1)


def build_rf_direct(Xtr, Ytr, Xte, verbose=0):
    """多步直接预测：每个 horizon 独立 RF + GridSearchCV。"""
    H = Ytr.shape[1]
    preds = np.zeros((len(Xte), H))
    best_params = []
    for h in range(H):
        gs = GridSearchCV(_base_rf(), PARAM_GRID, cv=cv_splitter(),
                          scoring="neg_root_mean_squared_error",
                          n_jobs=-1, verbose=verbose)
        gs.fit(Xtr, forward(Ytr[:, h]))
        preds[:, h] = clip_to_range(inverse(gs.predict(Xte)), Ytr[:, h])
        best_params.append(gs.best_params_)
    return preds, best_params


def build_rf_multioutput(Xtr, Ytr, Xte, verbose=0):
    """多输出预测：单个 MultiOutputRegressor(RandomForest) + GridSearchCV。"""
    grid = {f"estimator__{k}": v for k, v in PARAM_GRID.items()}
    gs = GridSearchCV(MultiOutputRegressor(_base_rf()), grid, cv=cv_splitter(),
                      scoring="neg_root_mean_squared_error",
                      n_jobs=-1, verbose=verbose)
    gs.fit(Xtr, forward(Ytr))
    preds = clip_to_range(inverse(gs.predict(Xte)), Ytr)
    return preds, gs.best_params_
