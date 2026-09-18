"""基于随机森林的径流多步预测模型。

包含：
  - 多步直接预测 (Direct)：每个 horizon 训练一个 RandomForestRegressor
  - 多输出预测 (MultiOutput)：单个 MultiOutputRegressor(RandomForest)
  - GridSearchCV 确定超参数
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from .. import config


def build_rf_direct(Xtr, Ytr, Xte, verbose=0):
    """多步直接预测：每个 horizon 独立 RF + GridSearchCV。"""
    H = Ytr.shape[1]
    preds = np.zeros((len(Xte), H))
    best_params = []
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    for h in range(H):
        yh = Ytr[:, h]
        base = RandomForestRegressor(random_state=config.SEED, n_jobs=-1)
        gs = GridSearchCV(base, param_grid, cv=3, scoring="neg_root_mean_squared_error",
                          n_jobs=-1, verbose=verbose)
        gs.fit(Xtr, yh)
        preds[:, h] = gs.predict(Xte)
        best_params.append(gs.best_params_)
    return preds, best_params


def build_rf_multioutput(Xtr, Ytr, Xte, verbose=0):
    """多输出预测：单个 MultiOutputRegressor(RandomForest) + GridSearchCV。"""
    param_grid = {
        "estimator__n_estimators": [100, 300],
        "estimator__max_depth": [None, 10, 20],
        "estimator__min_samples_split": [2, 5],
    }
    base = MultiOutputRegressor(
        RandomForestRegressor(random_state=config.SEED, n_jobs=-1))
    gs = GridSearchCV(base, param_grid, cv=3, scoring="neg_root_mean_squared_error",
                      n_jobs=-1, verbose=verbose)
    gs.fit(Xtr, Ytr)
    preds = gs.predict(Xte)
    return preds, gs.best_params_
