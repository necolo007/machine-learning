"""基于人工神经网络 (MLP) 的径流多步预测模型。

包含：
  - 多步直接预测 (Direct)：每个 horizon 训练一个独立的 MLPRegressor
  - 多输出预测 (MultiOutput)：单个 MultiOutputRegressor(MLPRegressor) 同时输出 7 步
  - GridSearchCV 确定超参数
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

from .. import config


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_ann_direct(Xtr, Ytr, Xte, verbose=0):
    """多步直接预测：为每个 horizon 训练独立 MLP，GridSearchCV 调参。"""
    H = Ytr.shape[1]
    preds = np.zeros((len(Xte), H))
    best_params = []
    param_grid = {
        "hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "alpha": [1e-4, 1e-3],
        "learning_rate_init": [1e-3, 5e-4],
    }
    for h in range(H):
        yh = Ytr[:, h]
        base = MLPRegressor(max_iter=500, random_state=config.SEED, early_stopping=True)
        gs = GridSearchCV(base, param_grid, cv=3, scoring="neg_root_mean_squared_error",
                          n_jobs=-1, verbose=verbose)
        gs.fit(Xtr, yh)
        preds[:, h] = gs.predict(Xte)
        best_params.append(gs.best_params_)
    return preds, best_params


def build_ann_multioutput(Xtr, Ytr, Xte, verbose=0):
    """多输出预测：单个 MultiOutputRegressor(MLP)，GridSearchCV 调参。"""
    param_grid = {
        "estimator__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
        "estimator__alpha": [1e-4, 1e-3],
        "estimator__learning_rate_init": [1e-3, 5e-4],
    }
    base = MultiOutputRegressor(
        MLPRegressor(max_iter=500, random_state=config.SEED, early_stopping=True))
    gs = GridSearchCV(base, param_grid, cv=3, scoring="neg_root_mean_squared_error",
                      n_jobs=-1, verbose=verbose)
    gs.fit(Xtr, Ytr)
    preds = gs.predict(Xte)
    return preds, gs.best_params_
