"""按预注册规则构建分层样本（3 板块 x 3 市值层，每格 7 只）。

规则见 `docs/preregistration_stratified.md` §2：
- 股票池：沪深300（大）/ 中证500（中）/ 中证1000（小），快照 2026-10-16
- 板块：沪市主板 / 深市主板 / 创业板（科创板排除）
- 每格：候选按代码升序 → 用 seed=42 的随机置换 → 依序取合格者，满 7 只为止
- 只允许数据完整性原因排除

用法：
    python scripts/build_stratified_universe.py --dry-run   # 只看抽样名单，不下载
    python scripts/build_stratified_universe.py             # 下载并生成名单
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

CFG_PATH = ROOT / "configs" / "stratified_universe.json"
CROSS_UNIVERSE = ROOT / "configs" / "universe_cs300_25.json"
OUT_DIR = ROOT / "reports" / "stratified"

# 固定的格子顺序：决定随机置换的消耗顺序，必须写死以保证可复现
BOARD_ORDER = ["沪市主板", "深市主板", "创业板"]
SIZE_ORDER = ["大市值", "中市值", "小市值"]


def board_of(code: str) -> str:
    if code.startswith(("600", "601", "603", "605")):
        return "沪市主板"
    if code.startswith(("000", "001", "002", "003", "302")):
        return "深市主板"
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith("688"):
        return "科创板"
    return "其他"


def exchange_of(code: str) -> str:
    return "sh" if code.startswith("6") else "sz"


def fetch_pool(cfg: dict) -> pd.DataFrame:
    import akshare as ak
    rows = []
    for size, idx_code in cfg["universe_rule"]["pool_indices"].items():
        df = ak.index_stock_cons_csindex(symbol=idx_code)
        for c, name in zip(df["成分券代码"], df["成分券名称"]):
            code = str(c).zfill(6)
            rows.append({"code": code, "name": str(name).replace(" ", ""),
                         "size": size, "board": board_of(code)})
    pool = pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True)
    return pool


def check_completeness(bars: pd.DataFrame, rule: dict) -> tuple[bool, str]:
    if bars is None or len(bars) == 0:
        return False, "无数据"
    if len(bars) < rule["min_bars"]:
        return False, f"bar 数不足（{len(bars)}）"
    if bars.index.min() > pd.Timestamp(rule["first_bar_not_after"]):
        return False, f"起始过晚（{bars.index.min().date()}）"
    if bars.index.max() < pd.Timestamp(rule["last_bar_not_before"]):
        return False, f"结束过早（{bars.index.max().date()}）"
    per_day = bars.groupby(bars.index.date).size()
    if int(per_day.median()) != rule["median_bars_per_day"]:
        return False, f"每日 bar 中位数 {int(per_day.median())}"
    return True, "合格"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    rule = cfg["universe_rule"]
    comp = rule["completeness"]
    quota = rule["quota_per_cell"]
    seed = cfg["seed"]

    print("=" * 78)
    print("按预注册规则构建分层样本")
    print("=" * 78)

    pool = fetch_pool(cfg)
    allowed = set(BOARD_ORDER)
    pool = pool[pool["board"].isin(allowed)]

    print("\n候选数量（板块 x 市值层）：")
    print(pd.crosstab(pool["board"], pool["size"]).reindex(
        index=BOARD_ORDER, columns=SIZE_ORDER).to_string())

    from daily_rv import fetch_sina_bars

    rng = np.random.default_rng(seed)
    selected, excluded = [], []
    cell_counts = {}

    for board in BOARD_ORDER:
        for size in SIZE_ORDER:
            cell = pool[(pool["board"] == board) & (pool["size"] == size)]
            cell = cell.sort_values("code").reset_index(drop=True)
            order = rng.permutation(len(cell))   # 固定顺序消耗随机数 → 可复现
            picked = 0
            print(f"\n[{board} × {size}] 候选 {len(cell)} 只")

            for pos in order:
                if picked >= quota:
                    break
                row = cell.iloc[int(pos)]
                code, symbol = row["code"], exchange_of(row["code"]) + row["code"]

                if args.dry_run:
                    selected.append({"ticker": code, "name": row["name"],
                                     "board": board, "size": size,
                                     "symbol": symbol})
                    picked += 1
                    continue

                try:
                    bars = fetch_sina_bars(symbol, "60")
                except Exception as exc:  # noqa: BLE001
                    excluded.append({"ticker": code, "name": row["name"],
                                     "board": board, "size": size,
                                     "reason": f"抓取失败 {type(exc).__name__}"})
                    continue

                ok, reason = check_completeness(bars, comp)
                if not ok:
                    excluded.append({"ticker": code, "name": row["name"],
                                     "board": board, "size": size,
                                     "reason": reason})
                    continue

                selected.append({
                    "ticker": code, "name": row["name"], "board": board,
                    "size": size, "symbol": symbol,
                    "bars": f"data/minute/{symbol}_60m.csv",
                    "n_bars": int(len(bars)),
                    "start": str(bars.index.min()), "end": str(bars.index.max()),
                })
                picked += 1

            cell_counts[f"{board}|{size}"] = picked
            if picked < quota:
                print(f"  该格仅取到 {picked} 只（配额 {quota}），不跨格借调")

    print("\n" + "-" * 78)
    print(f"共选中 {len(selected)} 只，排除 {len(excluded)} 只")
    gaps = {k: v for k, v in cell_counts.items() if v < quota}
    if gaps:
        print(f"未满额的格子：{gaps}")

    if args.dry_run:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "protocol_version": cfg["protocol_version"],
        "snapshot_date": rule["snapshot_date"],
        "seed": seed,
        "quota_per_cell": quota,
        "selected_n": len(selected),
        "cell_counts": cell_counts,
        "universe": selected,
    }
    path = ROOT / "configs" / "universe_stratified_63.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(excluded, columns=["ticker", "name", "board", "size", "reason"]).to_csv(
        OUT_DIR / "exclusions.csv", index=False)

    # 与既有 25 只实验的重叠
    if CROSS_UNIVERSE.exists():
        prev = json.loads(CROSS_UNIVERSE.read_text(encoding="utf-8"))
        prev_codes = {u["ticker"] for u in prev["universe"]}
        new_codes = {u["ticker"] for u in selected}
        overlap = sorted(prev_codes & new_codes)
        pd.DataFrame({"ticker": overlap}).to_csv(
            OUT_DIR / "overlap_with_cross_stock.csv", index=False)
        print(f"与 25 只实验重叠：{len(overlap)} 只 {overlap}")

    print(f"\n名单：{path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
