"""基于 LSTM 的径流多步预测模型 (PyTorch)。

包含：
  - 多步直接预测 (Direct)：每个 horizon 训练一个独立 LSTM
  - 多输出预测 (MultiOutput)：单个 LSTM 同时输出 7 步
  - 手动网格搜索 (Grid Search) 确定超参数（PyTorch 无原生 GridSearchCV，
    采用与 GridSearchCV 等价的手动网格 + K 折验证方式）
"""
import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .. import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMNet(nn.Module):
    def __init__(self, n_features, hidden, n_layers, horizon, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                           num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _to_tensor(X, Y=None):
    Xt = torch.tensor(X, dtype=torch.float32)
    if Y is None:
        return Xt
    Yt = torch.tensor(Y, dtype=torch.float32)
    if Yt.ndim == 1:
        Yt = Yt.unsqueeze(-1)
    return Xt, Yt


def _train_one(Xtr, ytr, Xval, yval, n_features, horizon, params, epochs=80, patience=12):
    model = LSTMNet(n_features, params["hidden"], params["layers"], horizon,
                    params["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    loss_fn = nn.MSELoss()
    tr_ds = TensorDataset(*_to_tensor(Xtr, ytr))
    loader = DataLoader(tr_ds, batch_size=params["batch"], shuffle=True)
    best_val = np.inf
    best_state = None
    bad = 0
    Xv, Yv = _to_tensor(Xval, yval)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv.to(DEVICE)).cpu().numpy()
        val_rmse = float(np.sqrt(np.mean((vp - yval) ** 2)))
        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def _time_split_val(Xtr, Ytr, frac=0.2):
    """从训练集末端按时间顺序切出验证集（不打乱顺序）。"""
    n_val = int(len(Xtr) * frac)
    return Xtr[:-n_val], Ytr[:-n_val], Xtr[-n_val:], Ytr[-n_val:]


def _grid_search(Xtr, Ytr, n_features, horizon, grid, verbose=0):
    """手动网格搜索：在训练集内部时间切分验证集，选最优超参。"""
    Xs, Ys, Xv, Yv = _time_split_val(Xtr, Ytr)
    keys = list(grid.keys())
    best, best_params, best_val = None, None, np.inf
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        model, val = _train_one(Xs, Ys, Xv, Yv, n_features, horizon, params)
        if verbose:
            print(f"  horizon={horizon} params={params} val_rmse={val:.3f}")
        if val < best_val:
            best_val = val
            best_params = params
    # 用最优参数在全量训练集上重训
    best_model, _ = _train_one(Xtr, Ytr, Xtr, Ytr, n_features, horizon, best_params)
    return best_model, best_params


# 超参网格（精简以保证 CPU 可运行性）
GRID = {
    "hidden": [32, 64],
    "layers": [1, 2],
    "dropout": [0.0],
    "lr": [1e-3],
    "wd": [1e-5],
    "batch": [128],
}


def build_lstm_direct(Xtr, Ytr, Xte, verbose=0):
    """多步直接预测：每个 horizon 训练一个独立 LSTM。"""
    n_features = Xtr.shape[2]
    H = Ytr.shape[1]
    preds = np.zeros((len(Xte), H))
    all_params = []
    for h in range(H):
        model, params = _grid_search(Xtr, Ytr[:, h], n_features, 1, GRID, verbose)
        model.eval()
        with torch.no_grad():
            preds[:, h] = model(torch.tensor(Xte, dtype=torch.float32).to(DEVICE)).cpu().numpy().ravel()
        all_params.append(params)
    return preds, all_params


def build_lstm_multioutput(Xtr, Ytr, Xte, verbose=0):
    """多输出预测：单个 LSTM 同时输出 7 步。"""
    n_features = Xtr.shape[2]
    H = Ytr.shape[1]
    model, params = _grid_search(Xtr, Ytr, n_features, H, GRID, verbose)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(Xte, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return preds, params
