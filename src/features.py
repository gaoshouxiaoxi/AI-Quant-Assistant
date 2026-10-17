"""第 5-6 周：特征工程。

把 OHLCV 序列变成"每行一个样本"的特征矩阵，用于预测**次日涨跌**。

设计原则：
1. **只用 t 时刻及之前的数据构造特征**，标签是 t+1 的方向。任何用到 t+1
   信息（比如 t+1 的收益率、或者用整段数据算的均值做标准化）都是数据泄漏。
2. 标准化放在交叉验证**内部**做，只用训练折的统计量，不能先全量标准化再切分。
3. 特征分三类，便于解释：
   - 滞后收益（过去 20 天每天的涨跌）
   - 技术指标（均线偏离、波动率、RSI、成交量变化）
   - 形态特征（实体大小、上下影线相对位置）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 20


def build_features(stock: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """构造特征矩阵与标签。

    Parameters
    ----------
    stock : 含 Open/High/Low/Close/Volume 的日线 DataFrame
    lookback : 使用过去多少个交易日

    Returns
    -------
    DataFrame：索引与 stock 对齐，包含
        - 特征列（f_*）
        - label：次日收盘价涨 = 1，跌 = 0
        - forward_return：次日收益率（用于评估，切勿作为特征）
    末尾若干行因无法构造标签会被丢弃。
    """
    df = stock.copy().sort_index()
    close = df["Close"].astype(float)
    high, low = df["High"].astype(float), df["Low"].astype(float)
    open_, volume = df["Open"].astype(float), df["Volume"].astype(float)

    ret = close.pct_change()
    out = pd.DataFrame(index=df.index)

    # ---- 1) 滞后收益：过去 lookback 天 ----
    for lag in range(1, lookback + 1):
        out[f"f_ret_lag{lag}"] = ret.shift(lag)

    # ---- 2) 技术指标 ----
    for window in (5, 10, 20):
        ma = close.rolling(window).mean()
        out[f"f_close_ma{window}_ratio"] = close / ma - 1.0  # 价格相对均线偏离

    out["f_vol5"] = ret.rolling(5).std()
    out["f_vol20"] = ret.rolling(20).std()
    out["f_vol_ratio"] = out["f_vol5"] / out["f_vol20"]

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["f_rsi14"] = 100 - 100 / (1 + rs)

    # 成交量：相对 20 日均量的倍数（对数化，压缩极值）
    out["f_volume_ratio"] = np.log1p(volume / volume.rolling(20).mean())

    # 动量
    out["f_momentum5"] = close / close.shift(5) - 1.0
    out["f_momentum20"] = close / close.shift(20) - 1.0

    # ---- 3) 形态特征 ----
    rng = (high - low).replace(0, np.nan)
    out["f_body"] = (close - open_) / rng          # 实体占振幅比例
    out["f_upper_wick"] = (high - close.combine(open_, max)) / rng
    out["f_lower_wick"] = (close.combine(open_, min) - low) / rng
    out["f_gap"] = open_ / close.shift(1) - 1.0    # 跳空

    # ---- 标签：次日方向 ----
    forward_return = close.pct_change().shift(-1)
    out["forward_return"] = forward_return
    out["label"] = (forward_return > 0).astype("int64")
    # 最后一行的 forward_return 是 NaN，标签无意义
    out.loc[forward_return.isna(), "label"] = -1

    out = out.replace([np.inf, -np.inf], np.nan)
    # 丢掉滚动窗口未成型的前段，以及标签无效的末行
    out = out.dropna()
    out = out[out["label"] >= 0]
    return out


FEATURE_PREFIX = "f_"


def split_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """拆出特征矩阵与标签。"""
    feature_cols = [c for c in data.columns if c.startswith(FEATURE_PREFIX)]
    return data[feature_cols], data["label"]
