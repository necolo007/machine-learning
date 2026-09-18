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

# 预测参数
LOOKBACK = 7          # 使用过去 7 天构造特征
HORIZON = 7            # 预测未来 1-7 天径流
TRAIN_END_YEAR = 2003  # 前 4 年训练
TEST_YEAR = 2004       # 第 5 年测试

# 评估指标
METRICS = ["NSE", "RMSE", "MAE", "R2", "KGE"]

# 随机种子
SEED = 42
