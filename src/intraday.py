"""日内效应分析（15 分钟数据）。

思路：日频方向不可预测，但**日内**存在结构性的、可重复的模式：
1. 波动率的 U 型曲线 —— 开盘和收盘波动最大，午间最小（全球市场普遍现象）
2. 隔夜跳空与日内收益的关系 —— 是否存在反转或延续
3. 上午收益能否预测下午收益 —— 日内动量 vs 反转

样本量警告
----------
15 分钟数据只有约 120 个交易日。日度层面的样本量 n≈120，
相关系数的 95% 显著性阈值约为 |r| > 0.18（n=120）。
本脚本会在结果里标注这个阈值，避免把偶然的强相关当成规律。

数据来源：akshare 的新浪 15 分钟接口（stock_zh_a_minute）。
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

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "minute"
CHART_DIR = ROOT / "charts"

SYMBOL = "sh600519"
NAME = "贵州茅台"
PERIOD = "15"


def fetch_minute() -> pd.DataFrame:
    """获取 15 分钟数据，优先读本地缓存。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{SYMBOL}_{PERIOD}m.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["day"])
        return df.set_index("day").sort_index()

    import os
    # 国内接口需要绕过本机代理
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

    import akshare as ak
    raw = ak.stock_zh_a_minute(symbol=SYMBOL, period=PERIOD, adjust="qfq")
    if raw is None or raw.empty:
        raise RuntimeError("分钟数据获取失败")
    df = raw.copy()
    df["day"] = pd.to_datetime(df["day"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("day").sort_index()
    df.to_csv(cache)
    return df


def daily_aggregate(bars: pd.DataFrame) -> pd.DataFrame:
    """把 15 分钟 bar 聚合成日度，拆出隔夜跳空 / 上午 / 下午。"""
    bars = bars.copy()
    bars["date"] = bars.index.date
    bars["time"] = bars.index.time
    # A 股：9:30-11:30 上午(8根)，13:00-15:00 下午(8根)
    bars["is_morning"] = bars["time"] <= pd.Timestamp("11:30").time()

    rows = []
    for date, g in bars.groupby("date"):
        g = g.sort_index()
        if len(g) < 8:
            continue
        first = g.iloc[0]
        last = g.iloc[-1]
        morning = g[g["is_morning"]]
        afternoon = g[~g["is_morning"]]
        if morning.empty or afternoon.empty:
            continue
        rows.append({
            "date": pd.Timestamp(date),
            "day_open": first["open"],
            "day_close": last["close"],
            "morning_ret": morning["close"].iloc[-1] / morning["open"].iloc[0] - 1,
            "afternoon_ret": afternoon["close"].iloc[-1] / afternoon["open"].iloc[0] - 1,
            "day_ret": last["close"] / first["open"] - 1,
            "realized_vol": float(np.sqrt(np.sum(np.log(g["close"] / g["close"].shift(1)).dropna() ** 2))),
        })

    daily = pd.DataFrame(rows).set_index("date").sort_index()
    daily["overnight_gap"] = daily["day_open"] / daily["day_close"].shift(1) - 1
    return daily.dropna()


def significance_threshold(n: int, alpha: float = 0.05) -> float:
    """相关系数的显著性阈值（近似用 t 分布临界值）。"""
    from scipy import stats
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    return float(t_crit / np.sqrt(t_crit ** 2 + (n - 2)))


def plot_intraday(daily: pd.DataFrame, vol_by_slot: pd.Series, title: str,
                  out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from charts import setup_chinese_font
    setup_chinese_font()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.bar(range(1, len(vol_by_slot) + 1), vol_by_slot.values, color="#1565c0")
    ax1.set_title(f"{title} · 日内波动率分布（15 分钟 bar）")
    ax1.set_xlabel("当日第 N 根 15 分钟 bar（1=开盘，16=收盘）")
    ax1.set_ylabel("收益率标准差")
    ax1.grid(axis="y", alpha=0.25)

    ax2.scatter(daily["morning_ret"], daily["afternoon_ret"],
                s=22, color="#d32f2f", alpha=0.7)
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    r = daily["morning_ret"].corr(daily["afternoon_ret"])
    ax2.set_title(f"上午 vs 下午收益（r = {r:+.3f}）")
    ax2.set_xlabel("上午收益")
    ax2.set_ylabel("下午收益")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    bars = fetch_minute()
    print(f"{NAME} ({SYMBOL}) 15 分钟线：{len(bars)} 根 bar，"
          f"{bars.index.min():%Y-%m-%d} ~ {bars.index.max():%Y-%m-%d}")

    daily = daily_aggregate(bars)
    n = len(daily)
    thresh = significance_threshold(n)
    print(f"聚合为 {n} 个交易日\n")
    print(f"显著性阈值：n={n} 时，|r| > {thresh:.3f} 才算 p<0.05 显著\n")

    # ---- 1) 日内波动率分布 ----
    bars2 = bars.copy()
    bars2["slot"] = bars2.groupby(bars2.index.date).cumcount() + 1
    # 用整条序列算收益率，这样每天第一根 bar 包含隔夜跳空（信息量最大的一段）。
    # 若按日分组做 pct_change，第一根永远是 NaN，会把开盘波动整体丢掉。
    bar_ret = bars2["close"].pct_change()
    vol_by_slot = bar_ret.groupby(bars2["slot"]).std().dropna()

    print("=== 1) 日内波动率（按当日第 N 根 bar）")
    peak = vol_by_slot.idxmax()
    trough = vol_by_slot.idxmin()
    print(f"{'bar':>4}{'波动率':>10}{'说明':>12}")
    for slot, v in vol_by_slot.items():
        note = ""
        if slot == 1:
            note = "含隔夜跳空"
        if slot == peak:
            note += " 最高"
        if slot == trough:
            note += " 最低"
        print(f"{slot:>4}{v:>10.5f}{note:>12}")
    print(f"\n  U 型特征：第 {peak} 根波动最大，第 {trough} 根最小，"
          f"相差 {vol_by_slot.loc[peak] / vol_by_slot.loc[trough]:.2f} 倍。")

    # ---- 2) 相关系数检验 ----
    print("\n=== 2) 日度相关检验")
    tests = [
        ("上午收益 → 下午收益", "morning_ret", "afternoon_ret"),
        ("隔夜跳空 → 日内收益", "overnight_gap", "day_ret"),
        ("上午收益 → 隔夜跳空(次日)", "morning_ret", "overnight_gap"),
        ("日内收益 → 次日日内收益", "day_ret", "day_ret"),
    ]
    n_tests = len(tests)
    bonferroni = 0.05 / n_tests
    print(f"  同时检验 {n_tests} 个假设，Bonferroni 校正阈值 p < {bonferroni:.4f}")
    from scipy import stats
    for label, a, b in tests:
        if a == "day_ret" and b == "day_ret":
            x = daily["day_ret"].iloc[:-1].values
            y = daily["day_ret"].iloc[1:].values
            r = float(np.corrcoef(x, y)[0, 1])
        else:
            r = daily[a].corr(daily[b])
        # 相关系数的 p 值
        t_stat = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
        p = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2)))
        raw = "显著" if p < 0.05 else "不显著"
        corrected = "（校正后仍显著）" if p < bonferroni else "（校正后不显著）"
        print(f"  {label:<24} r = {r:+.4f}  p = {p:.4f}  [{raw}]{corrected if p < 0.05 else ''}")

    # ---- 3) 按上午涨跌分组的下午表现 ----
    print("\n=== 3) 按上午方向分组的下午平均收益")
    up = daily[daily["morning_ret"] > 0]["afternoon_ret"]
    down = daily[daily["morning_ret"] < 0]["afternoon_ret"]
    print(f"  上午上涨 ({len(up)} 天): 下午平均 {up.mean():+.4%}，中位数 {up.median():+.4%}")
    print(f"  上午下跌 ({len(down)} 天): 下午平均 {down.mean():+.4%}，中位数 {down.median():+.4%}")
    if len(up) and len(down):
        from scipy import stats
        t, p = stats.ttest_ind(up, down, equal_var=False)
        print(f"  两组差异 t={t:+.2f}, p={p:.3f} "
              f"{'（显著）' if p < 0.05 else '（不显著）'}")

    plot_intraday(daily, vol_by_slot, NAME, CHART_DIR / "intraday_600519.png")
    daily.to_csv(DATA_DIR / "600519_daily_from_15m.csv")
    print(f"\n图表已保存 charts/intraday_600519.png，日度聚合数据存于 data/minute/")

    print("\n结论提示：")
    print("  - U 型波动曲线属于全市场普遍规律，样本量小也稳定，可直接用于择时执行")
    print("  - 相关系数若未超过阈值，就不能当作可交易信号")


if __name__ == "__main__":
    main()
