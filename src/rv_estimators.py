"""波动率估计量：用 OHLC 提高估计效率。

为什么需要这些估计量
--------------------
用"日收益率的标准差"估计波动率，信息利用率很低：一天只有收盘价这一个观测点。
而 OHLC 四个价格里，最高/最低价包含了当日的**波动区间**信息。

估计效率（相对收盘价法的方差节省）：
- Parkinson（1980）：约 5 倍
- Garman-Klass（1980）：约 7 倍
- Yang-Zhang（2000）：约 7 倍，且能处理隔夜跳空和趋势漂移

这些效率收益意味着：同样的历史长度，噪声更小，预测更容易。

各估计量（单日方差，之后按窗口平均或求和）
------------------------------------------
令 O/H/L/C 为当日开盘/最高/最低/收盘，C_prev 为前一日收盘。

- 收盘价法 (close-to-close)：r_t = log(C_t / C_{t-1})，取方差
- Parkinson：        log(H/L)² / (4·ln2)
- Garman-Klass：     ½·log(H/L)² − (2ln2−1)·log(C/O)²
- Rogers-Satchell：  log(H/C)·log(H/O) + log(L/C)·log(L/O)
- Yang-Zhang：       σ²_overnight + k·σ²_intraday + (1−k)·σ²_RS
  其中 k = 0.34 / (1.34 + (N+1)/(N−1))，N 为窗口长度

参考
----
Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance
    of the Rate of Return. Journal of Business.
Garman, M. & Klass, M. (1980). On the Estimation of Security Price Volatilities
    from Historical Data. Journal of Business.
Yang, D. & Zhang, Q. (2000). Drift-Independent Volatility Estimation Based on
    High, Low, Open, and Close Prices. Journal of Business.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
LN2 = np.log(2.0)


# ---------------------------------------------------------------- 单日方差

def daily_var_close(close: pd.Series) -> pd.Series:
    """收盘价法：日对数收益率的平方（用于滚动求均值）。"""
    return np.log(close / close.shift(1)) ** 2


def daily_var_parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson 单日方差。"""
    return np.log(high / low) ** 2 / (4.0 * LN2)


def daily_var_garman_klass(open_: pd.Series, high: pd.Series,
                           low: pd.Series, close: pd.Series) -> pd.Series:
    """Garman-Klass 单日方差。"""
    hl = np.log(high / low) ** 2
    co = np.log(close / open_) ** 2
    return 0.5 * hl - (2.0 * LN2 - 1.0) * co


def daily_rs(open_: pd.Series, high: pd.Series, low: pd.Series,
             close: pd.Series) -> pd.Series:
    """Rogers-Satchell 单日方差（对趋势稳健，但不处理跳空）。"""
    return (np.log(high / close) * np.log(high / open_)
            + np.log(low / close) * np.log(low / open_))


# ---------------------------------------------------------------- 滚动年化

def rolling_vol_close(close: pd.Series, window: int = 20) -> pd.Series:
    """收盘价法滚动年化波动率。"""
    var = daily_var_close(close).rolling(window).mean()
    return np.sqrt(var.clip(lower=0) * TRADING_DAYS)


def rolling_vol_parkinson(high: pd.Series, low: pd.Series,
                          window: int = 20) -> pd.Series:
    var = daily_var_parkinson(high, low).rolling(window).mean()
    return np.sqrt(var.clip(lower=0) * TRADING_DAYS)


def rolling_vol_garman_klass(open_: pd.Series, high: pd.Series,
                             low: pd.Series, close: pd.Series,
                             window: int = 20) -> pd.Series:
    var = daily_var_garman_klass(open_, high, low, close).rolling(window).mean()
    return np.sqrt(var.clip(lower=0) * TRADING_DAYS)


def rolling_vol_yang_zhang(open_: pd.Series, high: pd.Series,
                           low: pd.Series, close: pd.Series,
                           window: int = 20) -> pd.Series:
    """Yang-Zhang 滚动年化波动率（同时考虑隔夜跳空与日内波动）。"""
    log_oc = np.log(open_ / close.shift(1))     # 隔夜收益
    log_co = np.log(close / open_)              # 日内收益
    rs = daily_rs(open_, high, low, close)

    var_o = log_oc.rolling(window).var(ddof=1)
    var_c = log_co.rolling(window).var(ddof=1)
    var_rs = rs.rolling(window).mean()

    n = window
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    var = var_o + k * var_c + (1.0 - k) * var_rs
    return np.sqrt(var.clip(lower=0) * TRADING_DAYS)


def all_estimators(stock: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """一次算出所有估计量，便于对比。"""
    o = stock["Open"].astype(float)
    h = stock["High"].astype(float)
    lo = stock["Low"].astype(float)
    c = stock["Close"].astype(float)

    return pd.DataFrame({
        "close": rolling_vol_close(c, window),
        "parkinson": rolling_vol_parkinson(h, lo, window),
        "garman_klass": rolling_vol_garman_klass(o, h, lo, c, window),
        "yang_zhang": rolling_vol_yang_zhang(o, h, lo, c, window),
    })


def comparison_table(stock: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """各估计量的描述统计（便于看是否有系统性偏差）。"""
    est = all_estimators(stock, window)
    return pd.DataFrame({
        "mean": est.mean(),
        "std": est.std(),
        "min": est.min(),
        "max": est.max(),
    })
