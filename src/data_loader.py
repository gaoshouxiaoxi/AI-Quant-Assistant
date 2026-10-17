"""数据获取层：多源自动降级。

优先级：akshare（能取全窗口）→ yfinance → 腾讯行情（单次上限 640 根日线）。

说明：
- 本机开着系统代理（Clash 等），东方财富接口会被代理拒绝，因此这里对
  A 股接口显式绕过代理。
- 腾讯接口单次最多返回 640 根日线，属于降级方案，覆盖范围会短于请求区间，
  返回值里的 source 会标明实际用的是哪个源，便于调用方判断数据可信范围。

对外主要接口：
    load_daily(ticker, start, end) -> (DataFrame, source_name)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# 单日跌幅超过这个阈值就认为是未复权造成的假暴跌，拒绝该数据源
SPLIT_GLITCH_THRESHOLD = -0.5


def _bypass_proxy() -> None:
    """让 akshare 的国内数据接口直连，避免被本机代理拦掉。"""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def _is_a_share(ticker: str) -> bool:
    t = ticker.strip()
    return t.isdigit() and len(t) == 6


def _validate(df: pd.DataFrame, name: str) -> None:
    """体检：拦住未复权数据导致的假暴跌。

    典型症状：AAPL 在 2020-08-31 拆股（4:1），未复权数据会出现 -74% 的
    假单日跌幅，最大回撤会被算成 -78%。真实个股单日波动极少超过 ±50%。
    """
    if len(df) < 2:
        raise ValueError("数据不足 2 行，无法判断")
    ret = df["Close"].pct_change().dropna()
    if ret.empty:
        raise ValueError("无法计算收益率")
    worst = float(ret.min())
    if worst < SPLIT_GLITCH_THRESHOLD:
        raise ValueError(
            f"疑似未复权（单日跌幅 {worst * 100:.1f}%，超过 "
            f"{abs(SPLIT_GLITCH_THRESHOLD) * 100:.0f}% 阈值）"
        )


def _cache_path(cache_dir: Path | None, ticker: str, start: str, end: str) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"{ticker}_{start}_{end}.csv"


def load_daily(ticker: str, start: str, end: str,
               cache_dir: Path | str | None = None,
               retries: int = 2) -> tuple[pd.DataFrame, str]:
    """获取日线数据，返回 (DataFrame, 数据源名称)。

    顺序：
    1. 若给了 `cache_dir` 且已有对应 CSV，直接读本地缓存（source="cache"）。
       免费数据源经常限流/断连，缓存能保证重复实验不被网络卡住。
    2. 依次尝试 yfinance（支持复权、覆盖全）→ 腾讯（前复权、较稳，单次 640 行）
       → akshare。每个源取到数据后都要通过 `_validate` 体检，否则降级。
    3. 网络全失败但有缓存时，回退到缓存（source="cache(降级)"）。

    所有路径都失败时抛 RuntimeError，并列出每个源的具体原因。
    """
    cache_file = _cache_path(cache_dir if isinstance(cache_dir, Path) else
                             (Path(cache_dir) if cache_dir else None),
                             ticker, start, end)
    if cache_file is not None and cache_file.exists():
        try:
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            cached = _normalize(cached)
            _validate(cached, "cache")
            return cached, "cache"
        except Exception:  # noqa: BLE001 - 缓存坏了就继续走网络
            pass

    errors: list[str] = []
    sources = (("yfinance", _from_yfinance),
               ("tencent", _from_tencent),
               ("akshare", _from_akshare))

    for name, loader in sources:
        for attempt in range(retries + 1):
            try:
                df = loader(ticker, start, end)
                if df.empty:
                    if attempt < retries:
                        time.sleep(0.8 * (attempt + 1))
                        continue
                    errors.append(f"{name}: 返回空数据")
                    break
                _validate(df, name)
                return df, name
            except Exception as exc:  # noqa: BLE001
                if attempt < retries:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if cache_file is not None and cache_file.exists():
        # 降级读缓存也必须体检：缓存文件可能是早期未加体检时写下的脏数据
        # （例如 AAPL 2020-08-31 的拆股断层），直接返回会把错误带进下游。
        try:
            cached = _normalize(pd.read_csv(cache_file, index_col=0,
                                            parse_dates=True))
            _validate(cached, "cache(降级)")
            return cached, "cache(降级)"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cache(降级): {type(exc).__name__}: {exc}")

    raise RuntimeError("所有数据源都失败或未通过体检：\n  - " + "\n  - ".join(errors))


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名与索引，按日期升序。"""
    out = df[COLUMNS].copy()
    out.index.name = "Date"
    return out.sort_index()


def _from_akshare(ticker: str, start: str, end: str) -> pd.DataFrame:
    _bypass_proxy()
    import akshare as ak

    if _is_a_share(ticker):
        raw = ak.stock_zh_a_hist(
            symbol=ticker, period="daily",
            start_date=start.replace("-", ""), end_date=end.replace("-", ""),
            adjust="qfq",
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.rename(columns={
            "日期": "Date", "开盘": "Open", "收盘": "Close",
            "最高": "High", "最低": "Low", "成交量": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        return _normalize(df.apply(pd.to_numeric, errors="coerce"))

    raw = ak.stock_us_daily(symbol=ticker.upper())
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns={"date": "Date", "open": "Open", "close": "Close",
                             "high": "High", "low": "Low", "volume": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.loc[(df.index >= start) & (df.index <= end)]
    return _normalize(df.apply(pd.to_numeric, errors="coerce"))


def _from_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    # auto_adjust=True：让拆股/分红被正确复权，避免出现假暴跌
    raw = yf.download(ticker, start=start, end=end, progress=False,
                      auto_adjust=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return _normalize(raw[[c for c in COLUMNS if c in raw.columns]])


def _tencent_symbol(ticker: str) -> str:
    t = ticker.strip().upper()
    if _is_a_share(t):
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
    return _normalize(frame.apply(pd.to_numeric, errors="coerce"))
