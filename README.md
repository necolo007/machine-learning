# 基于机器学习的径流预测

本项目依据《机器学习课程实践》实验指导书完成，综合利用 **特征工程、机器学习、深度学习与集成学习** 技术实现给定流域的径流（流量）多步预测。

## 一、数据集

`01047000.csv`：USGS 站点 01047000 的日均流量与 Daymet V4R1 气象数据（2000-01-01 ~ 2004-12-31，共 1827 天，无缺失、无重复日期）。

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

该流域响应很快：流量的滞后 1 天自相关仅 0.77，滞后 7 天降到 0.40，且分布严重右偏（均值 611 cfs、最大 13700 cfs）。因此建模时对流量取对数，评估时再还原到原始量纲。

## 二、项目结构

```
machine-learning/
├── 01047000.csv                 # 原始数据
├── 实验指导书-2026(1).docx
├── requirements.txt
├── README.md
├── src/
│   ├── config.py                # 全局配置（路径/预测参数/特征选择/种子）
│   ├── data_loader.py           # 数据加载与字段说明
│   ├── eda.py                   # 数据分析与可视化
│   ├── features.py              # 特征工程（时间/滞后/窗口/归一化/选择）
│   ├── evaluate.py              # 评估指标（NSE/RMSE/MAE/R2/KGE）与绘图
│   ├── train.py                 # 主流程入口
│   └── models/
│       ├── common.py            # 目标变换、时序交叉验证、预测后处理
│       ├── ann.py               # 人工神经网络（MLP）直接 + 多输出 + GridSearchCV
│       ├── rf.py                # 随机森林 直接 + 多输出 + GridSearchCV
│       ├── lstm.py              # LSTM 直接 + 多输出 + 网格搜索（PyTorch）
│       └── ensemble.py          # 集成学习：简单平均 + Stacking
├── figures/                     # EDA 图表 + 各模型预测图 + 指标对比图
└── results/                     # summary.csv / nse_by_horizon.csv / metrics.json
```

## 三、技术路线

### 1. 数据分析（EDA）
- 字段中英文含义说明、描述性统计
- 时间序列可视化、直方图、按月箱型图、皮尔逊相关系数热力图

### 2. 特征工程

**时间序列特征提取**（共 89 个候选特征）

| 类别 | 特征 |
|------|------|
| 时间编码 | 月份/年内日序三角化编码，四季 one-hot |
| 滞后特征 | 对数流量 lag0~lag7；流量涨落率 `dlogQ`（区分涨水段与退水段）；各气象要素 lag1~lag3 |
| 窗口特征 | 对数流量的 3/7/14/30 天均值、标准差、最大值、最小值；前期降水指数 `Prcp_sum`（流域蓄水状态代理）；正积温 `PDD_sum`（融雪代理）；`frozen_days7`、`Prcp × PDD` |
| 未来气象强迫 | 未来 1~7 天的降水与正积温 |

> **关于未来气象强迫**：径流预报的业务流程是"数值天气预报 → 水文模型 → 流量预报"，未来降水是必备输入，本项目以实测气象代替预报值（"完美预报"假设）。将 `config.USE_FUTURE_FORCING` 置为 `False` 可退回仅用历史信息的纯自回归设定。两者差别很大：纯自回归时平均 NSE 约 0.2，加入未来气象后可达 0.74。

**特征归一化**：实现 `MinMaxNormalizer`（最小-最大归一化，默认）与 `ZScoreNormalizer`（Z-score 标准化）。**仅在训练集上拟合**统计量后应用到测试集，避免信息泄漏。

**特征选择**：实现皮尔逊相关系数法与互信息法，默认皮尔逊法选取 45 个特征。两种方法都在打分后追加一步**贪心冗余剔除**（跳过与已选特征相关性 > 0.95 的候选）——因为流量各滞后项之间相关系数普遍超过 0.95，若只按"与目标相关性"排序取 Top-k，结果几乎全是彼此冗余的滞后项，降水等互补信息会被完全挤出。特征选择同样只在训练集样本上进行。

### 3. 径流预测
预测目标：未来 **1~7 天** 的径流，采用两种策略：
- **多步直接预测 (Direct)**：每个预测步长训练一个独立模型（共 7 个模型）
- **多输出预测 (MultiOutput)**：单个模型同时输出 7 步（共 1 个模型）

| 方法 | 类型 | 直接(7) | 多输出(1) | 调参 |
|------|------|:---:|:---:|------|
| ANN (MLP) | 机器学习 | ✓ | ✓ | GridSearchCV |
| RandomForest | 机器学习 | ✓ | ✓ | GridSearchCV |
| LSTM | 深度学习 | ✓ | ✓ | 网格搜索（手动等价 GridSearch） |
| 简单平均 | 集成学习 | ✓ | ✓ | 固定超参 |
| Stacking(Ridge 元学习器) | 集成学习 | ✓ | ✓ | 固定超参 |

> 个体模型数：3 方法 × (7+1) = 24；集成模型数：2 方法 × (7+1) = 16。

几点实现要点：
- **调参不打乱时间顺序**：所有 `GridSearchCV` 的 `cv` 均使用 `TimeSeriesSplit`；LSTM 用等价的"手动网格 + TimeSeriesSplit"实现。
- **LSTM 输入**：编码窗口 `[t-6, t]` 共 7 步；启用未来气象强迫时再向后拼接 `[t+1, t+7]` 共 7 步（流量通道用 t 时刻值占位，附 `is_future` 标志位），即"已知未来协变量"的编码方式。
- **Stacking 的元特征来自折外预测**：用 `TimeSeriesSplit` 逐折训练基模型、在验证段上预测，拼成元特征后训练 Ridge 元学习器。若直接用基模型在自己训练集上的预测作为元特征，元学习器会严重过拟合。

### 4. 数据划分
按时间先后顺序切分，**不打乱**：前 4 年 (2000-2003) 训练，第 5 年 (2004) 测试。测试集 origin 的预测窗口完全落在 2004 年内。训练样本 1432 个，测试样本 353 个。

### 5. 评估指标
NSE、RMSE、MAE、R²、KGE，对每个预测步长分别计算并取平均。另提供**持续性基线**（预测值恒等于起报日实测流量）作为参照。

## 四、结果

各模型 1-7 天平均指标（测试集 = 2004 年）：

| 模型 | NSE | RMSE | MAE | KGE |
|------|----:|-----:|----:|----:|
| Persistence（基线） | -0.099 | 824.8 | 391.3 | 0.444 |
| ANN_Direct | 0.689 | 438.4 | 192.2 | 0.697 |
| **ANN_MultiOutput** | **0.737** | **403.9** | 178.1 | 0.689 |
| RF_Direct | 0.540 | 535.8 | 228.3 | 0.561 |
| RF_MultiOutput | 0.539 | 538.1 | 228.0 | 0.573 |
| LSTM_Direct | 0.641 | 474.4 | 194.8 | **0.762** |
| LSTM_MultiOutput | 0.627 | 483.4 | 203.0 | 0.729 |
| Avg_Direct | 0.717 | 421.4 | **169.0** | 0.667 |
| Avg_MultiOutput | 0.685 | 443.2 | 183.8 | 0.675 |
| Stack_Direct | 0.716 | 421.0 | 169.1 | 0.637 |
| Stack_MultiOutput | 0.613 | 490.5 | 203.7 | 0.596 |

逐预见期 NSE 见 `results/nse_by_horizon.csv`，逐步长全部指标与最优超参见 `results/metrics.json`。

主要结论：
- 全部 40 个模型在所有预见期上都显著优于持续性基线（基线从 t+1 的 0.48 迅速衰减到 t+7 的 -0.49）。
- ANN 表现最好（t+1 的 NSE 达 0.84，t+7 仍有 0.63）；集成学习的两种方法把 MAE 降到最低，简单平均与 Stacking 精度相当。
- 随机森林明显弱于 ANN 与 LSTM，主要因为树模型无法外推：训练期未出现的量级它预测不出来，洪峰被系统性低估。
- LSTM 的 KGE 最高，说明它在流量的变差与偏差上还原得最好，但对个别洪峰的把握不如 ANN。

## 五、运行方式

```bash
pip install -r requirements.txt

# 1) 仅做数据分析（生成 figures/ 下 EDA 图表）
python -m src.eda

# 2) 完整训练与评估（含 EDA），约 20-30 分钟（CPU）
python -m src.train

# 跳过 EDA；忽略缓存全部重训
python -m src.train --skip-eda
python -m src.train --no-cache
```

训练结果会缓存到 `results/cache/*.npz`，中途失败后重跑不必重复训练已完成的模型。

输出：
- `results/summary.csv`：各模型 1-7 天平均指标
- `results/nse_by_horizon.csv`：各模型逐预见期 NSE
- `results/metrics.json`：数据集信息、选中特征、逐步长指标与最优超参
- `figures/`：EDA 图表、各模型预测散点图与时序对比图、指标随预见期变化图

## 六、环境
- Python 3.11、scikit-learn、PyTorch (CPU)、pandas、numpy、matplotlib
- 无需 TensorFlow / xgboost / seaborn
