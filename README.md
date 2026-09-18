# 基于机器学习的径流预测

本项目依据《机器学习课程实践》实验指导书完成，综合利用 **特征工程、机器学习、深度学习与集成学习** 技术实现给定流域的径流（流量）多步预测。

## 一、数据集

`01047000.csv`：USGS 站点 01047000 的日均流量与 Daymet V4R1 气象数据（2000-01-01 ~ 2004-12-31，共 1827 天）。

| 字段 | 含义 |
|------|------|
| Date | 日期 |
| Discharge | 日均径流/流量 (cfs)，预测目标 |
| Dayl | 日长 (秒) |
| Prcp | 降水量 (mm/天) |
| Srad | 短波辐射 (W/m²) |
| Swe | 雪水当量 (mm)（本数据集全为 0，特征工程中剔除） |
| Tmax | 日最高气温 (℃) |
| Tmin | 日最低气温 (℃) |
| Vp | 水汽压 (Pa) |

参考：[Daymet Daily V4R1](https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4R1.html)、[USGS HUCs](https://nas.er.usgs.gov/hucs.aspx)

## 二、项目结构

```
machine-learning/
├── 01047000.csv                 # 原始数据
├── 实验指导书-2026(1).docx
├── requirements.txt
├── README.md
├── src/
│   ├── config.py                # 全局配置（路径/预测参数/种子）
│   ├── data_loader.py           # 数据加载与字段说明
│   ├── eda.py                   # 数据分析与可视化
│   ├── features.py              # 特征工程（时间/滞后/窗口/归一化/选择）
│   ├── evaluate.py              # 评估指标（NSE/RMSE/MAE/R2/KGE）与绘图
│   ├── train.py                 # 主流程入口
│   └── models/
│       ├── ann.py               # 人工神经网络（MLP）直接 + 多输出 + GridSearchCV
│       ├── rf.py                # 随机森林 直接 + 多输出 + GridSearchCV
│       ├── lstm.py              # LSTM 直接 + 多输出 + 网格搜索（PyTorch）
│       └── ensemble.py          # 集成学习：简单平均 + Stacking
├── figures/                     # EDA 与预测图表
└── results/                     # 汇总指标 summary.csv / metrics.json
```

## 三、技术路线

### 1. 数据分析（EDA）
- 字段中英文含义说明
- 时间序列可视化、直方图、箱型图、皮尔逊相关系数热力图

### 2. 特征工程
- **时间序列特征提取**：月份/年内日序三角化编码、季节 one-hot 编码；径流滞后特征 (lag0~lag7)；滚动窗口特征 (均值/标准差/最大值/降水累计)
- **特征归一化**：实现最小-最大归一化（MinMax），并提供 Z-score 标准化函数
- **特征选择**：实现皮尔逊相关系数法与互信息法，默认采用皮尔逊法选取 Top-20 特征

### 3. 径流预测
预测目标：未来 **1~7 天** 的径流，采用两种策略：
- **多步直接预测 (Direct)**：每个预测步长训练一个独立模型（共 7 个模型）
- **多输出预测 (MultiOutput)**：单个模型同时输出 7 步（共 1 个模型）

模型清单（共 24 个个体模型 + 16 个集成模型，符合指导书要求）：

| 方法 | 类型 | 直接(7) | 多输出(1) | 调参 |
|------|------|:---:|:---:|------|
| ANN (MLP) | 机器学习 | ✓ | ✓ | GridSearchCV |
| RandomForest | 机器学习 | ✓ | ✓ | GridSearchCV |
| LSTM | 深度学习 | ✓ | ✓ | 网格搜索（手动等价 GridSearch） |
| 简单平均 | 集成学习 | ✓ | ✓ | 固定超参 |
| Stacking(Ridge 元学习器) | 集成学习 | ✓ | ✓ | 固定超参 |

> 个体模型数：3 方法 × (7+1) = 24；集成模型数：2 方法 × (7+1) = 16。

### 4. 数据划分
按时间先后顺序切分，**不打乱**：前 4 年 (2000-2003) 训练，第 5 年 (2004) 测试。测试集 origin 的预测窗口完全落在 2004 年内，覆盖 2004 全年。

### 5. 评估指标
NSE、RMSE、MAE、R²、KGE，对每个预测步长分别计算并取平均。

## 四、运行方式

```bash
pip install -r requirements.txt

# 1) 数据分析（生成 figures/ 下 EDA 图表）
python -m src.eda

# 2) 完整训练与评估（含 EDA）
python -m src.train

# 或跳过 EDA 直接训练
python -m src.train --skip-eda
```

结果输出：
- `results/summary.csv`：各模型平均指标汇总
- `results/metrics.json`：各模型逐预测步长指标与最优超参
- `figures/`：EDA 图表 + 各模型预测散点图与时序对比图

## 五、环境
- Python 3.11、scikit-learn 1.6、PyTorch 2.12 (CPU)、pandas、numpy、matplotlib
- 无需 TensorFlow / xgboost / seaborn
