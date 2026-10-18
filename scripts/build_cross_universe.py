"""按预注册规则构建跨股票样本（抓取 60 分钟 bar）。

规则见 `docs/preregistration_cross_stock.md` §2：
沪深300 成分股快照 → 只留 A 股 → 按代码升序 → 逐个检查完整性 → 取前 25 只。
被排除的股票与原因全部记录，不允许按表现替换。

用法：
    python scripts/build_cross_universe.py            # 抓取并生成名单
    python scripts/build_cross_universe.py --dry-run  # 只打印名单，不下载
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 国内接口必须绕过系统代理，否则会被劫持
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

CFG_PATH = ROOT / "configs" / "cross_stock.json"
MINUTE_DIR = ROOT / "data" / "minute"
OUT_DIR = ROOT / "reports" / "cross_stock"


def fetch_pool(snapshot_date: str) -> pd.DataFrame:
    import akshare as ak
    df = ak.index_stock_cons_csindex(symbol="000300")
    df = df.rename(columns={"成分券代码": "code", "成分券名称": "name",
                            "交易所": "exchange", "日期": "date"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
        df = df[df["date"] == snapshot_date] if (df["date"] == snapshot_date).any() else df
    return df[["code", "name", "exchange"]]


def to_symbol(code: str, exchange: str) -> str:
    return ("sh" if "上海" in str(exchange) else "sz") + code


def check_completeness(bars: pd.DataFrame, rule: dict) -> tuple[bool, str]:
    if bars is None or len(bars) == 0:
        return False, "无数据"
    if len(bars) < rule["min_bars"]:
        return False, f"bar 数不足（{len(bars)} < {rule['min_bars']}）"
    first = bars.index.min()
    last = bars.index.max()
    if first > pd.Timestamp(rule["first_bar_not_after"]):
        return False, f"起始过晚（{first.date()}）"
    if last < pd.Timestamp(rule["last_bar_not_before"]):
        return False, f"结束过早（{last.date()}）"
    per_day = bars.groupby(bars.index.date).size()
    if int(per_day.median()) != rule["median_bars_per_day"]:
        return False, f"每日 bar 中位数 {int(per_day.median())}（要求 {rule['median_bars_per_day']}）"
    return True, "合格"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印，不下载")
    ap.add_argument("--candidates", type=int, default=40,
                    help="最多尝试的候选数量")
    args = ap.parse_args()

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    rule = cfg["universe_rule"]
    comp = rule["completeness"]
    target_n = rule["target_n"]

    print("=" * 78)
    print("按预注册规则构建跨股票样本")
    print("=" * 78)

    pool = fetch_pool(rule["snapshot_date"])
    pool = pool[pool["exchange"].isin(rule["exchange_filter"])]
    pool = pool.sort_values("code").reset_index(drop=True)
    print(f"股票池：{len(pool)} 只 A 股（快照 {rule['snapshot_date']}），按代码升序")

    from daily_rv import fetch_sina_bars

    selected, excluded = [], []
    tried = 0
    for _, row in pool.iterrows():
        if len(selected) >= target_n or tried >= args.candidates:
            break
        tried += 1
        code, name = row["code"], row["name"]
        symbol = to_symbol(code, row["exchange"])

        if args.dry_run:
            selected.append({"ticker": code, "name": name, "symbol": symbol})
            print(f"  [dry] {tried:>2}. {code} {name} → {symbol}")
            continue

        try:
            bars = fetch_sina_bars(symbol, "60")
        except Exception as exc:  # noqa: BLE001
            excluded.append({"ticker": code, "name": name, "symbol": symbol,
                             "reason": f"抓取失败：{type(exc).__name__}"})
            print(f"  {tried:>2}. {code} {name} → 抓取失败 {type(exc).__name__}")
            continue

        ok, reason = check_completeness(bars, comp)
        if not ok:
            excluded.append({"ticker": code, "name": name, "symbol": symbol,
                             "reason": reason})
            print(f"  {tried:>2}. {code} {name} → 排除：{reason}")
            continue

        selected.append({
            "ticker": code, "name": name, "symbol": symbol,
            "bars": f"data/minute/{symbol}_60m.csv",
            "n_bars": int(len(bars)),
            "start": str(bars.index.min()),
            "end": str(bars.index.max()),
        })
        print(f"  {tried:>2}. {code} {name} → 合格（{len(bars)} 根）")

    print("-" * 78)
    print(f"选中 {len(selected)} 只，尝试 {tried} 只，排除 {len(excluded)} 只")

    if args.dry_run:
        return 0

    if len(selected) < target_n:
        print(f"警告：只选到 {len(selected)} 只（目标 {target_n}），"
              f"可加大 --candidates 重试。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "protocol_version": cfg["protocol_version"],
        "snapshot_date": rule["snapshot_date"],
        "ordering": rule["ordering"],
        "target_n": target_n,
        "selected_n": len(selected),
        "universe": selected,
        "excluded": excluded,
    }
    out_path = ROOT / "configs" / "universe_cs300_25.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    pd.DataFrame(excluded).to_csv(OUT_DIR / "universe_exclusions.csv", index=False)
    print(f"\n名单：{out_path.relative_to(ROOT)}")
    print(f"排除记录：{(OUT_DIR / 'universe_exclusions.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
