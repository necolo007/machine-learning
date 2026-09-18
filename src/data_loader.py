"""数据加载与字段说明。"""
import pandas as pd

from . import config

# 数据字段中英文含义（Daymet V4R1 + USGS 流量）
FIELD_DESCRIPTION = {
    "Date": "日期 (YYYY-MM-DD)",
    "Discharge": "日均径流/流量 (cfs, 立方英尺/秒)，USGS 站点 01047000",
    "Dayl": "日长 (秒)，一天内太阳在地平线以上的时长",
    "Prcp": "降水量 (mm/天)",
    "Srad": "短波辐射 (W/m^2)",
    "Swe": "雪水当量 (mm)，积雪融化等效水量",
    "Tmax": "日最高气温 (℃)",
    "Tmin": "日最低气温 (℃)",
    "Vp": "水汽压 (Pa)",
}


def load_data() -> pd.DataFrame:
    """读取 CSV，解析日期并设为索引。"""
    df = pd.read_csv(config.DATA_PATH)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    df = df.sort_values(config.DATE_COL).reset_index(drop=True)
    df = df.set_index(config.DATE_COL)
    return df


def describe_fields() -> str:
    """返回字段中文说明文本。"""
    lines = ["数据字段说明（Daymet V4R1 气象数据 + USGS 流量数据）："]
    for k, v in FIELD_DESCRIPTION.items():
        lines.append(f"  {k:10s} : {v}")
    return "\n".join(lines)
