"""基于集成学习的径流多步预测模型。

  - 简单平均 (Simple Average)：对 MLP / 随机森林 / LSTM 三个基模型的预测取算术平均
  - Stacking：用 TimeSeriesSplit 生成"折外 (out-of-fold) 元特征"训练 Ridge 元学习器，
    避免元学习器在基模型的训练集内预测上拟合，从而消除严重过拟合
  - 两种预测策略（多步直接 / 多输出）各一套

基模型采用固定的合理超参（集成部分按指导书只要求评价精度），以控制计算开销。
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader, TensorDataset

from .. import config
from .common import cv_splitter, forward, inverse, clip_to_range
from .lstm import LSTMNet, DEVICE

CV_SPLITS = 3


# ---------------------------------------------------------------- 基模型
def _fit_mlp(Xtr, ztr):
    """ztr 为已变换的目标（一维或二维）。返回 predict 函数（变换空间）。"""
    est = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1500, alpha=1e-3,
                       random_state=config.SEED, early_stopping=True,
                       n_iter_no_change=30, validation_fraction=0.15)
    model = MultiOutputRegressor(est) if ztr.ndim == 2 else est
    model.fit(Xtr, ztr)
    return model.predict


def _fit_rf(Xtr, ztr):
    est = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                max_features=0.5, random_state=config.SEED, n_jobs=-1)
    model = MultiOutputRegressor(est) if ztr.ndim == 2 else est
    model.fit(Xtr, ztr)
    return model.predict


def _fit_lstm(Str, ztr, epochs=120):
    """LSTM 基模型；Str 形状 (n, L, C)。"""
    z = ztr.reshape(-1, 1) if ztr.ndim == 1 else ztr
    torch.manual_seed(config.SEED)
    model = LSTMNet(Str.shape[2], 64, 1, z.shape[1], 0.1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(Str, dtype=torch.float32),
                      torch.tensor(z, dtype=torch.float32)),
        batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()

    def predict(X):
        with torch.no_grad():
            p = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        return p.ravel() if z.shape[1] == 1 else p

    return predict


def _base_predictions(Xtr, Str, ztr, Xte, Ste):
    """训练三个基模型并给出各自在测试集上的预测（均在变换空间）。"""
    return [
        _fit_mlp(Xtr, ztr)(Xte),
        _fit_rf(Xtr, ztr)(Xte),
        _fit_lstm(Str, ztr)(Ste),
    ]


# ------------------------------------------------------------ 简单平均
def simple_average_direct(Xtr, Ytr, Xte, Str, Ste):
    """多步直接预测 + 简单平均：逐 horizon 训练三个基模型后取算术平均。"""
    H = Ytr.shape[1]
    preds = np.zeros((len(Xte), H))
    for h in range(H):
        ps = _base_predictions(Xtr, Str, forward(Ytr[:, h]), Xte, Ste)
        preds[:, h] = np.mean([inverse(np.asarray(p).ravel()) for p in ps], axis=0)
    return clip_to_range(preds, Ytr)


def simple_average_multioutput(Xtr, Ytr, Xte, Str, Ste):
    """多输出预测 + 简单平均。"""
    ps = _base_predictions(Xtr, Str, forward(Ytr), Xte, Ste)
    return clip_to_range(np.mean([inverse(np.asarray(p)) for p in ps], axis=0), Ytr)


# -------------------------------------------------------------- Stacking
def _oof_meta_features(Xtr, Str, ztr, Xte, Ste):
    """用 TimeSeriesSplit 生成折外元特征。

    返回 (meta_train, z_meta_train, meta_test)：
      meta_train 只包含各折验证段的样本（基模型未见过这些样本），
      meta_test 由在全部训练数据上重训的基模型给出。
    """
    single = ztr.ndim == 1
    n_out = 1 if single else ztr.shape[1]
    rows, targets = [], []
    for tr, va in cv_splitter(CV_SPLITS).split(Xtr):
        ps = _base_predictions(Xtr[tr], Str[tr], ztr[tr], Xtr[va], Str[va])
        rows.append(np.hstack([np.asarray(p).reshape(len(va), n_out) for p in ps]))
        targets.append(ztr[va])
    meta_train = np.vstack(rows)
    z_meta = np.concatenate(targets) if single else np.vstack(targets)

    ps_te = _base_predictions(Xtr, Str, ztr, Xte, Ste)
    meta_test = np.hstack([np.asarray(p).reshape(len(Xte), n_out) for p in ps_te])
    return meta_train, z_meta, meta_test


def stacking_direct(Xtr, Ytr, Xte, Str, Ste):
    """多步直接预测 + Stacking（Ridge 元学习器）。"""
    H = Ytr.shape[1]
    preds = np.zeros((len(Xte), H))
    for h in range(H):
        m_tr, z_tr, m_te = _oof_meta_features(Xtr, Str, forward(Ytr[:, h]), Xte, Ste)
        meta = Ridge(alpha=1.0).fit(m_tr, z_tr)
        preds[:, h] = inverse(meta.predict(m_te))
    return clip_to_range(preds, Ytr)


def stacking_multioutput(Xtr, Ytr, Xte, Str, Ste):
    """多输出预测 + Stacking（多输出 Ridge 元学习器）。"""
    m_tr, z_tr, m_te = _oof_meta_features(Xtr, Str, forward(Ytr), Xte, Ste)
    meta = MultiOutputRegressor(Ridge(alpha=1.0)).fit(m_tr, z_tr)
    return clip_to_range(inverse(meta.predict(m_te)), Ytr)
