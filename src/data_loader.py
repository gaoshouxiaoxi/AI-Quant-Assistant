"""数据获取层：优先 yfinance，被限流时自动降级到腾讯行情接口。

这样第 1 周的目标不会因为单一数据源限流而卡死。

对外主要接口：
    load_daily(ticker, start, end) -> pandas.DataFrame
"""

from __future__ import annotations

import json
import urllib.request

import pandas as pd
import yfinance as yf

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_daily(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """获取日线数据。

    返回 (DataFrame, 数据源名称)。两个源都失败时抛 RuntimeError。
    """
    errors: list[str] = []

    try:
        df = _from_yfinance(ticker, start, end)
        if not df.empty:
            return df, "yfinance"
        errors.append("yfinance: 返回空数据")
    except Exception as exc:  # noqa: BLE001 - 需要把真实原因带给调用者
        errors.append(f"yfinance: {type(exc).__name__}: {exc}")

    try:
        df = _from_tencent(ticker, start, end)
        if not df.empty:
            return df, "tencent"
        errors.append("tencent: 返回空数据")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tencent: {type(exc).__name__}: {exc}")

    raise RuntimeError("所有数据源都失败了：\n  - " + "\n  - ".join(errors))


def _from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame()

    # 新版 yfinance 可能返回多层列索引，压平它
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[[c for c in COLUMNS if c in raw.columns]].copy()
    df.index.name = "Date"
    return df


def _tencent_symbol(ticker: str) -> str:
    """把常见代码映射成腾讯接口的写法。"""
    t = ticker.strip().upper()
    if t.isdigit() and len(t) == 6:
        # A 股：6 开头为沪市，其余按深市处理
        return ("sh" if t.startswith("6") else "sz") + t
    return f"us{t}.OQ"


def _from_tencent(ticker: str, start: str, end: str) -> pd.DataFrame:
    symbol = _tencent_symbol(ticker)
    url = (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={symbol},day,{start},{end},640,qfq"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    node = (payload.get("data") or {}).get(symbol) or {}
    rows = node.get("qfqday") or node.get("day") or []
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["Date", "Open", "Close", "High", "Low", "Volume"],
    )
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.set_index("Date")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame[COLUMNS]
