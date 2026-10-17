"""日度 RV 与 HAR 模型（60 分钟数据）。

背景
----
15 分钟数据只有 122 个交易日，样本太少；东方财富接口在本机整站不通
（连日报都 RemoteDisconnected）。新浪接口的 datalen 有 1970 硬上限，
但**周期越长覆盖的历史越久**：

    15 分钟 → 约 122 个交易日
    30 分钟 → 约 250 个交易日
    60 分钟 → 约 490 个交易日

因此改用 60 分钟 bar（每个交易日 4 根）构造日度 RV，样本量提升约 4 倍。

日度 RV 的定义
--------------
    RV_t = Σ r_i²  （当日各 60 分钟 bar 的对数收益）
         + 隔夜收益²（计入隔夜跳空）

HAR 模型（Corsi, 2009）
-----------------------
波动率预测的行业基准模型。用三个不同时间尺度的 RV 做线性回归：

    RV_{t+1} = a + b1·RV_t + b2·RV_week + b3·RV_month + ε

其中 RV_week 是过去 5 日均值、RV_month 是过去 22 日均值。
它用极少的参数捕捉了"长记忆性"，在大量实证中难以被复杂模型击败。
本脚本把 HAR 与"单纯用当前 RV 外推"的持续性基线做对比。
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from rv_estimators import (TRADING_DAYS, daily_rs, daily_var_close,
                           daily_var_garman_klass, daily_var_parkinson)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MINUTE_DIR = ROOT / "data" / "minute"
REPORT_DIR = ROOT / "reports"

SCALE = "60"
TICKERS = {"sh600519": "贵州茅台", "sz000858": "五粮液", "sz000001": "平安银行"}

FORWARD = 5       # 预测未来 5 日平均 RV
N_SPLITS = 5


# ---------------------------------------------------------------- 抓取

def _no_proxy() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def fetch_sina_bars(symbol: str, scale: str = SCALE) -> pd.DataFrame:
    """直接调新浪接口（akshare 把 datalen 写死了，这里用满 1970）。"""
    _no_proxy()
    cache = MINUTE_DIR / f"{symbol}_{scale}m.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["day"])
        return df.set_index("day").sort_index()

    url = ("https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_t=/"
           "CN_MarketDataService.getKLineData"
           f"?symbol={symbol}&scale={scale}&ma=no&datalen=1970")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8", "ignore")
    m = re.search(r"\((\[.*\])\)", text, re.S)
    if not m:
        raise RuntimeError(f"{symbol}: 无法解析新浪返回")
    raw = pd.DataFrame(json.loads(m.group(1)))
    if raw.empty:
        raise RuntimeError(f"{symbol}: 返回空数据")

    raw["day"] = pd.to_datetime(raw["day"])
    for col in ("open", "high", "low", "close", "volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    df = raw.set_index("day").sort_index()

    MINUTE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


# ---------------------------------------------------------------- 日度 RV

def build_daily_rv(bars: pd.DataFrame) -> pd.DataFrame:
    """把 60 分钟 bar 聚合成日线 OHLC + 日度 RV。"""
    bars = bars.copy()
    bars["date"] = bars.index.date

    rows = []
    for date, g in bars.groupby("date"):
        g = g.sort_index()
        if len(g) < 2:
            continue
        closes = g["close"].to_numpy()
        opens = g["open"].to_numpy()
        # 第一根用 open→close，其余用相邻 bar 收盘
        prev = np.r_[opens[0], closes[:-1]]
        r = np.log(closes / prev)
        r = r[np.isfinite(r)]

        rows.append({
            "date": pd.Timestamp(date),
            "Open": float(opens[0]),
            "High": float(g["high"].max()),
            "Low": float(g["low"].min()),
            "Close": float(closes[-1]),
            "Volume": float(g["volume"].sum()),
            "rv_intraday": float(np.sum(r ** 2)),
            "n_bars": len(g),
        })

    daily = pd.DataFrame(rows).set_index("date").sort_index()
    daily["overnight"] = np.log(daily["Open"] / daily["Close"].shift(1))
    daily = daily.dropna(subset=["overnight"])
    daily["rv_total"] = daily["rv_intraday"] + daily["overnight"] ** 2
    return daily


# ---------------------------------------------------------------- 估计量对照

def compare_estimators(daily: pd.DataFrame) -> pd.DataFrame:
    """在更长的样本上重做估计量验证（日内 vs rv_intraday）。"""
    o, h, lo, c = (daily["Open"], daily["High"], daily["Low"], daily["Close"])
    ests = {
        "Parkinson": daily_var_parkinson(h, lo),
        "Garman-Klass": daily_var_garman_klass(o, h, lo, c),
        "Rogers-Satchell": daily_rs(o, h, lo, c),
    }
    rows = []
    for name, est in ests.items():
        # 5 日均值后再比，降低单日噪声
        d = pd.DataFrame({"est": est.rolling(5).mean(),
                          "tgt": daily["rv_intraday"].rolling(5).mean()}).dropna()
        d = d[(d["est"] > 0) & (d["tgt"] > 0)]
        if len(d) < 50:
            continue
        le, lt = np.log(d["est"]), np.log(d["tgt"])
        b, a = np.polyfit(le, lt, 1)
        rows.append({
            "estimator": name, "n": len(d),
            "corr": float(le.corr(lt)),
            "slope": float(b),
            "rmse_log": float(np.sqrt(np.mean((lt - (a + b * le)) ** 2))),
            "median_ratio": float((d["est"] / d["tgt"]).median()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- HAR

def build_har_dataset(daily: pd.DataFrame) -> pd.DataFrame:
    """HAR 特征：日 / 周 / 月 三个尺度的 RV（对数）。"""
    rv = daily["rv_total"].copy()
    out = pd.DataFrame(index=daily.index)
    out["rv"] = rv
    out["rv_week"] = rv.rolling(5).mean()
    out["rv_month"] = rv.rolling(22).mean()
    # 目标：未来 FORWARD 日的平均 RV（对数）
    out["target"] = rv.rolling(FORWARD).mean().shift(-FORWARD)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out["rv"] > 0) & (out["target"] > 0)]
    for col in ("rv", "rv_week", "rv_month", "target"):
        out[f"log_{col}"] = np.log(out[col])
    return out


def walk_forward(n: int, n_splits: int = N_SPLITS, min_train: float = 0.4):
    start = int(n * min_train)
    step = max(1, (n - start) // n_splits)
    for k in range(n_splits):
        b = start + k * step
        end = n if k == n_splits - 1 else b + step
        yield np.arange(0, b), np.arange(b, end)


def evaluate_har(data: pd.DataFrame) -> dict:
    """HAR 与持续性基线的样本外对比（都在对数空间）。"""
    X = data[["log_rv", "log_rv_week", "log_rv_month"]]
    y = data["log_target"]
    base = data["log_rv_week"]      # 持续性基线：用周均值 RV 外推

    har_pred, base_pred, truth = [], [], []
    for tr, te in walk_forward(len(X)):
        model = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
        har_pred.append(pd.Series(model.predict(X.iloc[te]), index=X.iloc[te].index))
        base_pred.append(base.iloc[te])
        truth.append(y.iloc[te])

    har = pd.concat(har_pred).sort_index()
    bas = pd.concat(base_pred).sort_index()
    tru = pd.concat(truth).sort_index()

    mse_har = float(np.mean((har - tru) ** 2))
    mse_base = float(np.mean((bas - tru) ** 2))
    return {
        "samples": len(tru),
        "r2_har": r2_score(tru, har),
        "r2_base": r2_score(tru, bas),
        "mae_har": mean_absolute_error(tru, har),
        "mae_base": mean_absolute_error(tru, bas),
        "skill": 1.0 - mse_har / mse_base if mse_base > 0 else float("nan"),
        "spearman_har": float(tru.corr(har, method="spearman")),
        "spearman_base": float(tru.corr(bas, method="spearman")),
        "pred": har, "base": bas, "true": tru,
    }


def plot_har(result: dict, title: str, out_path: Path) -> Path:
    """HAR 预测 vs 持续性基线 vs 实际（对数 RV）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from charts import setup_chinese_font
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(result["true"].index, np.exp(result["true"]).values * TRADING_DAYS,
            color="#212121", linewidth=1.1, label="实际 RV（年化）")
    ax.plot(result["pred"].index, np.exp(result["pred"]).values * TRADING_DAYS,
            color="#d32f2f", linewidth=1.1, alpha=0.85, label="HAR 预测")
    ax.plot(result["base"].index, np.exp(result["base"]).values * TRADING_DAYS,
            color="#1565c0", linewidth=1.0, linestyle="--", alpha=0.75,
            label="持续性基线")
    ax.set_ylabel("年化波动率")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    print("=" * 78)
    print("日度 RV（60 分钟）与 HAR 模型")
    print("=" * 78)

    estimator_rows, har_rows = [], []

    for symbol, name in TICKERS.items():
        bars = fetch_sina_bars(symbol)
        daily = build_daily_rv(bars)
        print(f"\n----- {name} ({symbol}) -----")
        print(f"  {len(bars)} 根 60 分钟 bar → {len(daily)} 个交易日  "
              f"{daily.index.min():%Y-%m-%d} ~ {daily.index.max():%Y-%m-%d}")

        est = compare_estimators(daily)
        if not est.empty:
            print("  估计量 vs 日内 RV（5日均值口径）：")
            for _, r in est.iterrows():
                print(f"    {r['estimator']:<18} n={int(r['n']):>4}  "
                      f"corr={r['corr']:.3f}  斜率={r['slope']:.2f}  "
                      f"中位比值={r['median_ratio']:.2f}")
                estimator_rows.append({"symbol": symbol, "name": name, **r.to_dict()})

        har_data = build_har_dataset(daily)
        if len(har_data) < 120:
            print(f"  HAR 样本不足（{len(har_data)}），跳过")
            continue
        m = evaluate_har(har_data)
        print(f"  HAR 模型（样本外 {m['samples']} 天）：")
        print(f"    HAR       R²={m['r2_har']:>7.3f}  MAE={m['mae_har']:.4f}  "
              f"Skill={m['skill']:>+7.2%}  Spearman={m['spearman_har']:.3f}")
        print(f"    持续性基线 R²={m['r2_base']:>7.3f}  MAE={m['mae_base']:.4f}  "
              f"Skill=  +0.00%  Spearman={m['spearman_base']:.3f}")
        har_rows.append({"symbol": symbol, "name": name, "samples": m["samples"],
                         "r2_har": m["r2_har"], "r2_base": m["r2_base"],
                         "mae_har": m["mae_har"], "mae_base": m["mae_base"],
                         "skill": m["skill"],
                         "spearman_har": m["spearman_har"],
                         "spearman_base": m["spearman_base"]})
        try:
            plot_har(m, f"{name} HAR 模型 vs 持续性基线（60分钟 RV）",
                     ROOT / "charts" / f"har_{symbol}.png")
        except Exception as exc:  # noqa: BLE001
            print(f"    图表失败: {type(exc).__name__}: {exc}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if estimator_rows:
        pd.DataFrame(estimator_rows).to_csv(
            REPORT_DIR / "rv60_estimator_validation.csv", index=False)
    if har_rows:
        df = pd.DataFrame(har_rows)
        df.to_csv(REPORT_DIR / "har_model_results.csv", index=False)
        print("\n" + "=" * 78)
        print("HAR 汇总")
        print("=" * 78)
        print(f"{'标的':<12}{'样本':>7}{'HAR R²':>10}{'基线 R²':>10}"
              f"{'Skill':>10}{'Spearman':>11}")
        print("-" * 78)
        for _, r in df.iterrows():
            print(f"{r['name']:<12}{int(r['samples']):>7}{r['r2_har']:>10.3f}"
                  f"{r['r2_base']:>10.3f}{r['skill']:>+10.2%}"
                  f"{r['spearman_har']:>11.3f}")
        print(f"\n平均：HAR R²={df['r2_har'].mean():.3f}  "
              f"基线 R²={df['r2_base'].mean():.3f}  "
              f"Skill={df['skill'].mean():+.2%}")

    print("\n说明：HAR 只有 3 个参数，样本外表现稳定；")
    print("      若 Skill > 0，说明它比'直接用当前 RV 外推'更准。")


if __name__ == "__main__":
    main()
