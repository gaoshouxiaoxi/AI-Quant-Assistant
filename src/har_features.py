"""HAR 模型的特征工程（无网络依赖）。

从 `har_extended.py` 抽出来，目的是让"可复现研究"这条路径
只依赖 pandas / numpy / scikit-learn，不碰 yfinance、streamlit 等。

依赖：pandas, numpy（以及本项目内同样轻量的 rv_estimators）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rv_estimators import TRADING_DAYS

# 经典 HAR 三尺度
HAR3 = ["rv_d", "rv_w", "rv_m"]
# 多因子扩展（审计结论：不加分，保留用于对照）
HARX = HAR3 + ["vol_ratio", "overnight_share", "range_ratio",
               "skew22", "kurt22", "vol_asym", "absret"]
BASELINES = {"持续性(当日 RV)": ["rv_d"], "持续性(周均 RV)": ["rv_w"]}
ALL_COLS = sorted(set(HAR3 + HARX + [c for cs in BASELINES.values() for c in cs]))


def build_features(df: pd.DataFrame, forward: int = 5) -> pd.DataFrame:
    """由日线 OHLCV + 日度 RV 构造特征与标签。

    Parameters
    ----------
    df : 必须含 Open/High/Low/Close/Volume 与 rv（日度方差）
    forward : 标签的前视窗口长度（交易日）

    Returns
    -------
    索引与原数据一致；含特征列、`target`（log 未来 forward 日均值 RV）、
    `rv_is_true`（若上游提供）。
    """
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    lo = df["Low"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)
    rv = df["rv"].clip(lower=1e-12)

    out = pd.DataFrame(index=df.index)
    out["rv_is_true"] = df["rv_is_true"] if "rv_is_true" in df else 1
    out["rv"] = rv
    out["rv_d"] = np.log(rv)
    out["rv_w"] = np.log(rv.rolling(5).mean())
    out["rv_m"] = np.log(rv.rolling(22).mean())

    out["vol_ratio"] = np.log1p(v / v.rolling(22).mean())
    overnight = np.log(o / c.shift(1))
    out["overnight_share"] = (overnight ** 2 / rv).clip(0, 5)
    out["range_ratio"] = (h - lo) / c

    ret = c.pct_change()
    out["skew22"] = ret.rolling(22, min_periods=10).skew()
    out["kurt22"] = ret.rolling(22, min_periods=10).kurt()
    out["absret"] = ret.abs()

    # 不能用 ret.where(ret<0).rolling(22).std()：被 mask 后窗口几乎全 NaN，
    # rolling 默认要求满窗口非空，会整列变 NaN。
    sq = ret ** 2
    down = np.sqrt(sq.where(ret < 0).rolling(22, min_periods=5).mean()
                   * TRADING_DAYS)
    up = np.sqrt(sq.where(ret > 0).rolling(22, min_periods=5).mean()
                 * TRADING_DAYS)
    out["vol_asym"] = down / up

    # 标签：未来 forward 日的平均日度方差（对数）
    out["target"] = np.log(rv.rolling(forward).mean().shift(-forward))
    return out.replace([np.inf, -np.inf], np.nan)
