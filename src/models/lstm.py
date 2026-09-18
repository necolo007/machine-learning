"""基于 LSTM 的径流多步预测模型 (PyTorch)。

  - 多步直接预测 (Direct)：每个 horizon 训练一个独立 LSTM（共 7 个）
  - 多输出预测 (MultiOutput)：单个 LSTM 同时输出 7 步
  - 网格搜索确定超参数：PyTorch 无原生 GridSearchCV，此处用与 GridSearchCV 等价的
    "网格 + TimeSeriesSplit 交叉验证"手动实现，保证不打乱时间顺序
"""
import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .. import config
from .common import cv_splitter, forward, inverse, clip_to_range

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 本项目的 LSTM 很小，CPU 上开满线程反而因线程争用大幅变慢，故限制线程数
torch.set_num_threads(min(4, torch.get_num_threads()))

GRID = {
    "hidden": [32, 64],
    "layers": [1, 2],
    "dropout": [0.1],
    "lr": [1e-3],
    "wd": [1e-5],
    "batch": [64],
}
EPOCHS = 150
PATIENCE = 20
CV_SPLITS = 3


class LSTMNet(nn.Module):
    """编码序列后由全连接头一次性输出 horizon 个预测值。"""

    def __init__(self, n_features, hidden, n_layers, horizon, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, horizon))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])          # 始终保持 (batch, horizon)


def _as2d(y: np.ndarray) -> np.ndarray:
    return y.reshape(-1, 1) if y.ndim == 1 else y


def _train_one(Xtr, Ytr, Xval, Yval, params, epochs=EPOCHS, patience=PATIENCE):
    """训练单个 LSTM，用验证集 RMSE 做早停，返回模型与最优验证 RMSE。"""
    Ytr, Yval = _as2d(Ytr), _as2d(Yval)
    torch.manual_seed(config.SEED)
    model = LSTMNet(Xtr.shape[2], params["hidden"], params["layers"],
                    Ytr.shape[1], params["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["wd"])
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                      torch.tensor(Ytr, dtype=torch.float32)),
        batch_size=params["batch"], shuffle=True)
    Xv = torch.tensor(Xval, dtype=torch.float32).to(DEVICE)

    best_val, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)          # 形状严格一致，不发生广播
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv).cpu().numpy()
        val_rmse = float(np.sqrt(np.mean((vp - Yval) ** 2)))
        if val_rmse < best_val - 1e-5:
            best_val, best_state, bad = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def _grid_search(Xtr, Ytr, verbose=0):
    """网格搜索：每组超参在 TimeSeriesSplit 各折上取平均验证 RMSE，选最优后重训。"""
    keys = list(GRID)
    splits = list(cv_splitter(CV_SPLITS).split(Xtr))
    best_params, best_score = None, np.inf
    for combo in itertools.product(*[GRID[k] for k in keys]):
        params = dict(zip(keys, combo))
        scores = [_train_one(Xtr[tr], Ytr[tr], Xtr[va], Ytr[va], params)[1]
                  for tr, va in splits]
        score = float(np.mean(scores))
        if verbose:
            print(f"    params={params} cv_rmse={score:.4f}")
        if score < best_score:
            best_params, best_score = params, score
    # 用最优超参在全部训练数据上重训，末尾 15% 作为早停验证集
    n_val = max(int(len(Xtr) * 0.15), 1)
    model, _ = _train_one(Xtr[:-n_val], Ytr[:-n_val], Xtr[-n_val:], Ytr[-n_val:], best_params)
    return model, best_params


def _predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()


def build_lstm_direct(Xtr, Ytr, Xte, verbose=0):
    """多步直接预测：每个 horizon 一个独立 LSTM。"""
    H = Ytr.shape[1]
    preds = np.zeros((len(Xte), H))
    all_params = []
    for h in range(H):
        model, params = _grid_search(Xtr, forward(Ytr[:, h]), verbose)
        preds[:, h] = clip_to_range(inverse(_predict(model, Xte).ravel()), Ytr[:, h])
        all_params.append(params)
    return preds, all_params


def build_lstm_multioutput(Xtr, Ytr, Xte, verbose=0):
    """多输出预测：单个 LSTM 同时输出 7 步。"""
    model, params = _grid_search(Xtr, forward(Ytr), verbose)
    return clip_to_range(inverse(_predict(model, Xte)), Ytr), params
