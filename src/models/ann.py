"""基于人工神经网络 (MLP) 的径流多步预测模型。

  - 多步直接预测 (Direct)：每个 horizon 训练一个独立 MLPRegressor（共 7 个）
  - 多输出预测 (MultiOutput)：单个 MultiOutputRegressor(MLPRegressor) 同时输出 7 步
  - GridSearchCV（cv=TimeSeriesSplit，不打乱时间顺序）确定超参数
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

from .. import config
from .common import cv_splitter, forward, inverse, clip_to_range

PARAM_GRID = {
    "hidden_layer_sizes": [(64,), (128, 64), (256, 128)],
    "alpha": [1e-4, 1e-3, 1e-2],
    "learning_rate_init": [1e-3, 3e-4],
}


def _base_mlp():
    return MLPRegressor(max_iter=1500, random_state=config.SEED,
                        early_stopping=True, n_iter_no_change=30,
                        validation_fraction=0.15)


def build_ann_direct(Xtr, Ytr, Xte, verbose=0):
    """多步直接预测：为每个 horizon 独立训练 MLP 并调参。"""
    H = Ytr.shape[1]
    preds = np.zeros((len(Xte), H))
    best_params = []
    for h in range(H):
        gs = GridSearchCV(_base_mlp(), PARAM_GRID, cv=cv_splitter(),
                          scoring="neg_root_mean_squared_error",
                          n_jobs=-1, verbose=verbose)
        gs.fit(Xtr, forward(Ytr[:, h]))
        preds[:, h] = clip_to_range(inverse(gs.predict(Xte)), Ytr[:, h])
        best_params.append(gs.best_params_)
    return preds, best_params


def build_ann_multioutput(Xtr, Ytr, Xte, verbose=0):
    """多输出预测：单个模型同时输出 7 步。"""
    grid = {f"estimator__{k}": v for k, v in PARAM_GRID.items()}
    gs = GridSearchCV(MultiOutputRegressor(_base_mlp()), grid, cv=cv_splitter(),
                      scoring="neg_root_mean_squared_error",
                      n_jobs=-1, verbose=verbose)
    gs.fit(Xtr, forward(Ytr))
    preds = clip_to_range(inverse(gs.predict(Xte)), Ytr)
    return preds, gs.best_params_
