"""验证：哪种波动率估计量最接近"真值"？

思路
----
用 15 分钟高频数据算出**已实现波动率 RV**（当天所有 bar 收益率平方和），
把它当作当期真值的近似；再用各种 OHLC 估计量（收盘价 / Parkinson /
Garman-Klass / Yang-Zhang）去逼近它，看谁最准。

关键细节：目标量要对齐
----------------------
- **日内 RV**（只含开盘到收盘）↔ 对应 Parkinson / Garman-Klass / Rogers-Satchell
  这些估计量只用了日内高低价，不含隔夜跳空。
- **总 RV**（日内 + 隔夜跳空）↔ 对应收盘价法 / Yang-Zhang
  这两个把隔夜收益也算进去。

如果混着比，会得出"Yang-Zhang 有系统性偏差"的错误结论——其实是量的定义不同。

数据说明
--------
15 分钟数据只有约 122 个交易日，且是 2026 年的，与 2020-2025 的日线数据不重叠。
因此这里**直接从 15 分钟 bar 聚合出日线 OHLC**（日高=各 bar 最高价的最大值），
保证估计量与 RV 来自同一批交易日。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from rv_estimators import (TRADING_DAYS, daily_var_close, daily_var_garman_klass,
                           daily_var_parkinson, daily_rs)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "minute"


def load_bars() -> pd.DataFrame:
    path = DATA_DIR / "sh600519_15m.csv"
    df = pd.read_csv(path, parse_dates=["day"])
    df = df.set_index("day").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_daily_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """从 15 分钟 bar 聚合日线 OHLC + 日内 RV + 总 RV。"""
    bars = bars.copy()
    bars["date"] = bars.index.date

    rows = []
    for date, g in bars.groupby("date"):
        g = g.sort_index()
        if len(g) < 8:
            continue
        # 日内 bar 收益率：第一根用 open→close，其余用相邻收盘价
        r = np.log(g["close"].to_numpy() / np.r_[g["open"].iloc[0],
                                                  g["close"].to_numpy()[:-1]])
        r = r[np.isfinite(r)]
        rv_intraday = float(np.sum(r ** 2))

        rows.append({
            "date": pd.Timestamp(date),
            "Open": float(g["open"].iloc[0]),
            "High": float(g["high"].max()),
            "Low": float(g["low"].min()),
            "Close": float(g["close"].iloc[-1]),
            "Volume": float(g["volume"].sum()),
            "rv_intraday": rv_intraday,
            "n_bars": len(g),
        })

    daily = pd.DataFrame(rows).set_index("date").sort_index()
    # 隔夜收益（用前一日最后一根 bar 收盘 → 当日第一根 bar 开盘）
    daily["overnight"] = np.log(daily["Open"] / daily["Close"].shift(1))
    daily["rv_total"] = daily["rv_intraday"] + daily["overnight"] ** 2
    return daily.dropna(subset=["overnight"])


def evaluate(estimator: pd.Series, target: pd.Series, label: str) -> dict:
    """比较估计量与目标 RV（在对数空间做，因为波动率分布的误差是乘性的）。"""
    d = pd.DataFrame({"est": estimator, "tgt": target}).dropna()
    d = d[(d["est"] > 0) & (d["tgt"] > 0)]
    if len(d) < 30:
        return {"label": label, "n": len(d)}

    le, lt = np.log(d["est"]), np.log(d["tgt"])
    # 线性回归：log(RV) = a + b·log(估计)，b 接近 1 说明无系统偏置
    b, a = np.polyfit(le, lt, 1)
    resid = lt - (a + b * le)
    return {
        "label": label,
        "n": len(d),
        "corr": float(le.corr(lt)),
        "slope": float(b),
        "intercept": float(a),
        "rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        "bias_mean_ratio": float((d["est"] / d["tgt"]).mean()),
        "median_ratio": float((d["est"] / d["tgt"]).median()),
    }


def main() -> None:
    bars = load_bars()
    daily = build_daily_from_bars(bars)

    print(f"15 分钟数据：{len(bars)} 根 bar")
    print(f"聚合为 {len(daily)} 个交易日："
          f"{daily.index.min():%Y-%m-%d} ~ {daily.index.max():%Y-%m-%d}\n")

    # 用与 RV 相同的窗口（单日）比较各估计量
    o, h, lo, c = (daily["Open"], daily["High"], daily["Low"], daily["Close"])

    intraday_ests = {
        "Parkinson": daily_var_parkinson(h, lo),
        "Garman-Klass": daily_var_garman_klass(o, h, lo, c),
        "Rogers-Satchell": daily_rs(o, h, lo, c),
    }
    total_ests = {
        "收盘价法(含隔夜)": daily_var_close(c),
    }

    print("=" * 74)
    print("A. 日内估计量 vs 日内 RV（rv_intraday）")
    print("=" * 74)
    print(f"{'估计量':<20}{'n':>5}{'corr':>8}{'斜率':>8}{'RMSE(log)':>11}{'中位比值':>10}")
    print("-" * 74)
    rows = []
    for name, est in intraday_ests.items():
        # 单日估计噪声大，用 5 日窗口平均后再比，更接近 RV 的可比口径
        est5 = est.rolling(5).mean()
        rv5 = daily["rv_intraday"].rolling(5).mean()
        r = evaluate(est5, rv5, name)
        rows.append(r)
        if r.get("n", 0) >= 30:
            print(f"{name:<20}{r['n']:>5}{r['corr']:>8.3f}{r['slope']:>8.2f}"
                  f"{r['rmse_log']:>11.3f}{r['median_ratio']:>10.2f}")

    print("\n" + "=" * 74)
    print("B. 含隔夜估计量 vs 总 RV（rv_total = 日内 + 隔夜²）")
    print("=" * 74)
    print(f"{'估计量':<20}{'n':>5}{'corr':>8}{'斜率':>8}{'RMSE(log)':>11}{'中位比值':>10}")
    print("-" * 74)
    total_results = []
    from rv_estimators import rolling_vol_yang_zhang
    yz = rolling_vol_yang_zhang(o, h, lo, c, 5)
    yz_var = (yz / np.sqrt(TRADING_DAYS)) ** 2
    total_ests["Yang-Zhang(5日)"] = yz_var

    for name, est in total_ests.items():
        est5 = est.rolling(5).mean()
        rv5 = daily["rv_total"].rolling(5).mean()
        r = evaluate(est5, rv5, name)
        total_results.append(r)
        if r.get("n", 0) >= 30:
            print(f"{name:<20}{r['n']:>5}{r['corr']:>8.3f}{r['slope']:>8.2f}"
                  f"{r['rmse_log']:>11.3f}{r['median_ratio']:>10.2f}")

    # 单日层面（不滚动）的对比，作为补充
    print("\n" + "=" * 74)
    print("C. 单日估计量 vs 单日 RV（噪声最大，仅作参考）")
    print("=" * 74)
    for name, est in {**intraday_ests}.items():
        r = evaluate(est, daily["rv_intraday"], name)
        if r.get("n", 0) >= 30:
            print(f"  {name:<20} corr={r['corr']:.3f}  斜率={r['slope']:.2f}  "
                  f"中位比值={r['median_ratio']:.2f}")

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    best = max((r for r in rows if r.get("corr")),
               key=lambda x: x["corr"], default=None)
    if best:
        print(f"  日内估计量中最接近 RV 的是 {best['label']}："
              f"corr={best['corr']:.3f}，斜率={best['slope']:.2f}")
    if total_results:
        best_t = max((r for r in total_results if r.get("corr")),
                     key=lambda x: x["corr"], default=None)
        if best_t:
            print(f"  含隔夜估计量中最接近总 RV 的是 {best_t['label']}："
                  f"corr={best_t['corr']:.3f}，斜率={best_t['slope']:.2f}")
    print("\n  斜率接近 1、中位比值接近 1 的估计量没有系统性偏置，可直接用作目标。")
    print("  斜率明显偏离 1 说明它估计的波动水平不同，需要先做尺度校正。")

    out = ROOT / "reports" / "rv_estimator_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for r in rows + total_results if r.get("n")]).to_csv(
        out, index=False)
    print(f"\n明细已保存 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
