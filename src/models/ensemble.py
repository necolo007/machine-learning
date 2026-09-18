"""基于集成学习的径流多步预测模型。

包含：
  - 简单平均 (Simple Average)：对基模型预测取平均
  - Stacking：以基模型预测作为元特征，训练 Ridge 元学习器
  - 多步直接预测策略与多输出策略各一套
集成学习不要求 GridSearch（按指导书仅需评价精度），基模型采用固定合理超参以控制开销。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from .. import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- 基模型（固定超参） ----------------
def fit_mlp(Xtr, Ytr, horizon):
    """单输出 MLP，返回 predict 函数。"""
    if Ytr.ndim == 1:
        y = Ytr
    else:
        y = Ytr[:, horizon] if horizon is not None else Ytr
    m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400,
                     random_state=config.SEED, early_stopping=True)
    m.fit(Xtr, y)
    return lambda X: m.predict(X)


def fit_rf(Xtr, Ytr, horizon):
    if Ytr.ndim == 1:
        y = Ytr
    else:
        y = Ytr[:, horizon] if horizon is not None else Ytr
    m = RandomForestRegressor(n_estimators=200, max_depth=15,
                              random_state=config.SEED, n_jobs=-1)
    m.fit(Xtr, y)
    return lambda X: m.predict(X)


class _LSTM(nn.Module):
    def __init__(self, n_feat, hidden, horizon):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def fit_lstm(Xtr, Ytr, horizon, epochs=60):
    """LSTM 基模型。Xtr 形状 (n, L, F)。"""
    if Ytr.ndim == 1:
        y = Ytr
    else:
        y = Ytr[:, horizon] if horizon is not None else Ytr
    H = 1 if y.ndim == 1 else y.shape[1]
    n_feat = Xtr.shape[2]
    model = _LSTM(n_feat, 32, H).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(y if y.ndim == 1 else y, dtype=torch.float32).unsqueeze(-1) if y.ndim == 1 else torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(Xt, Yt)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    def predict(X):
        with torch.no_grad():
            p = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        return p
    return predict


# ---------------- 集成策略 ----------------
def _base_predictors_flat(Xtr, Ytr, Xte, horizon):
    """返回各基模型在测试集上的预测（用于扁平特征：MLP、RF）。"""
    p_mlp = fit_mlp(Xtr, Ytr, horizon)(Xte)
    p_rf = fit_rf(Xtr, Ytr, horizon)(Xte)
    return p_mlp, p_rf


def _base_predictors_seq(Xtr_seq, Ytr, Xte_seq, horizon):
    """序列特征：LSTM。"""
    return fit_lstm(Xtr_seq, Ytr, horizon)(Xte_seq)


def simple_average_direct(Xtr, Ytr, Xte, Xtr_seq, Xte_seq):
    """多步直接 + 简单平均。"""
    H = Ytr.shape[1]
    preds = np.zeros((len(Xte), H))
    for h in range(H):
        p_mlp = fit_mlp(Xtr, Ytr[:, h], h)(Xte)
        p_rf = fit_rf(Xtr, Ytr[:, h], h)(Xte)
        p_lstm = fit_lstm(Xtr_seq, Ytr[:, h], h)(Xte_seq)
        preds[:, h] = (p_mlp + p_rf + p_lstm.ravel()) / 3.0
    return preds


def simple_average_multioutput(Xtr, Ytr, Xte, Xtr_seq, Xte_seq):
    """多输出 + 简单平均。"""
    p_mlp = fit_mlp(Xtr, Ytr, None)(Xte)
    p_rf = fit_rf(Xtr, Ytr, None)(Xte)
    p_lstm = fit_lstm(Xtr_seq, Ytr, None)(Xte_seq)
    return (p_mlp + p_rf + p_lstm) / 3.0


def stacking_direct(Xtr, Ytr, Xte, Xtr_seq, Xte_seq, val_frac=0.2):
    """多步直接 + Stacking（Ridge 元学习器）。"""
    H = Ytr.shape[1]
    # 时间顺序切分内部验证集，构造元特征
    n_val = int(len(Xtr) * val_frac)
    Xa, Ya = Xtr[:-n_val], Ytr[:-n_val]
    Xb, Yb = Xtr[-n_val:], Ytr[-n_val:]
    Xa_seq, Ya_seq = Xtr_seq[:-n_val], Ytr[:-n_val]
    Xb_seq = Xtr_seq[-n_val:]
    preds = np.zeros((len(Xte), H))
    for h in range(H):
        # 在 a 上训练，预测 b 得元特征
        meta_b = np.column_stack([
            fit_mlp(Xa, Ya[:, h], h)(Xb),
            fit_rf(Xa, Ya[:, h], h)(Xb),
            fit_lstm(Xa_seq, Ya[:, h], h)(Xb_seq).ravel(),
        ])
        # 在全量训练集训练，预测测试集
        meta_te = np.column_stack([
            fit_mlp(Xtr, Ytr[:, h], h)(Xte),
            fit_rf(Xtr, Ytr[:, h], h)(Xte),
            fit_lstm(Xtr_seq, Ytr[:, h], h)(Xte_seq).ravel(),
        ])
        meta = Ridge(alpha=1.0)
        meta.fit(meta_b, Yb[:, h])
        preds[:, h] = meta.predict(meta_te)
    return preds


def stacking_multioutput(Xtr, Ytr, Xte, Xtr_seq, Xte_seq, val_frac=0.2):
    """多输出 + Stacking。"""
    n_val = int(len(Xtr) * val_frac)
    Xa, Ya = Xtr[:-n_val], Ytr[:-n_val]
    Xb, Yb = Xtr[-n_val:], Ytr[-n_val:]
    Xa_seq = Xtr_seq[:-n_val]
    Xb_seq = Xtr_seq[-n_val:]
    meta_b = np.column_stack([
        fit_mlp(Xa, Ya, None)(Xb).ravel(),
        fit_rf(Xa, Ya, None)(Xb).ravel(),
        fit_lstm(Xa_seq, Ya, None)(Xb_seq).ravel(),
    ])
    meta_te = np.column_stack([
        fit_mlp(Xtr, Ytr, None)(Xte).ravel(),
        fit_rf(Xtr, Ytr, None)(Xte).ravel(),
        fit_lstm(Xtr_seq, Ytr, None)(Xte_seq).ravel(),
    ])
    # 元学习器：多输出 Ridge
    meta = MultiOutputRegressor(Ridge(alpha=1.0))
    meta.fit(meta_b, Yb)
    return meta.predict(meta_te)
