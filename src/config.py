"""全局配置：路径、预测参数、随机种子。"""
from pathlib import Path

# 路径
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "01047000.csv"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# 数据列
DATE_COL = "Date"
TARGET = "Discharge"
# 气象特征（Swe 全为 0，会在特征工程中被剔除）
METEO_COLS = ["Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]
# 参与滞后/窗口特征构造的气象要素
FORCING_COLS = ["Prcp", "Tmax", "Tmin", "Srad", "Vp", "Dayl"]

# 预测参数
LOOKBACK = 7           # 使用过去 7 天构造特征
HORIZON = 7            # 预测未来 1-7 天径流
TRAIN_END_YEAR = 2003  # 前 4 年训练
TEST_YEAR = 2004       # 第 5 年测试
ROLL_WINDOWS = [3, 7, 14, 30]  # 滚动窗口长度

# 未来气象强迫：径流预报中未来 1-7 天的降水/气温来自数值天气预报，
# 此处用实测值代替（"完美预报"假设），是雨洪预报建模的常规设定。
# 置为 False 可退回仅用历史信息的纯自回归设定。
USE_FUTURE_FORCING = True

# 目标变换：径流分布右偏严重，取对数后建模再指数还原，可显著稳定 MLP/LSTM 训练
LOG_TARGET = True

# 特征选择
SELECT_METHOD = "pearson"  # "pearson" | "mi" | "all"
N_SELECT = 45

# 时序交叉验证折数（GridSearchCV 使用 TimeSeriesSplit，保证不打乱时间顺序）
CV_SPLITS = 4

# 评估指标
METRICS = ["NSE", "RMSE", "MAE", "R2", "KGE"]

# 随机种子
SEED = 42
