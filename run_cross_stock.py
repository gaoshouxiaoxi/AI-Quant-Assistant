"""跨股票验证运行器。

用法
----
    python run_cross_stock.py --gate     # 复现闸门：用新流程跑原有 3 只，对比主结果
    python run_cross_stock.py            # 跑 25 只完整实验

闸门不通过（与主结果不一致）就应停止，不要进入扩样本分析。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from cross_stock_study import (analyze_universe, plot_study,  # noqa: E402
                               run_sensitivity, summarize)

CFG_PATH = ROOT / "configs" / "cross_stock.json"
UNIVERSE_PATH = ROOT / "configs" / "universe_cs300_25.json"
MAIN_CFG = ROOT / "configs" / "research_main.json"
MAIN_RESULTS = ROOT / "reports" / "research" / "results_main.csv"


def run_gate(cfg: dict) -> int:
    """复现闸门：新流程必须能重现主实验的 3 只股票结果。"""
    print("=" * 78)
    print("复现闸门：用跨股票流程重跑原有 3 只，与主实验逐项比对")
    print("=" * 78)

    main_cfg = json.loads(MAIN_CFG.read_text(encoding="utf-8"))
    universe = [{"ticker": u["ticker"], "name": u["name"], "bars": u["bars"]}
                for u in main_cfg["universe"]]

    per_stock, _ = analyze_universe(universe, cfg)

    ref = pd.read_csv(MAIN_RESULTS)
    ref = ref[(ref["model"] == "HAR-3")].copy()
    ref["ticker"] = ref["ticker"].astype(str).str.zfill(6)

    print("\n比对（容差 1e-9）：")
    ok = True
    for _, r in per_stock.iterrows():
        m = ref[ref["ticker"] == r["ticker"]]
        if len(m) != 1:
            print(f"  {r['ticker']}: 参考结果缺失")
            ok = False
            continue
        m = m.iloc[0]
        diffs = {
            "skill": abs(float(m["skill"]) - float(r["skill"])),
            "spearman": abs(float(m["spearman"]) - float(r["spearman"])),
            "disp_ratio": abs(float(m["disp_ratio"]) - float(r["disp_ratio"])),
            "n": abs(int(m["n"]) - int(r["n_oos"])),
        }
        bad = {k: v for k, v in diffs.items() if v > 1e-9}
        status = "✅ 一致" if not bad else f"❌ 不一致 {bad}"
        print(f"  {r['ticker']} {r['name']:<6} "
              f"Skill {r['skill']:+.6f} vs {float(m['skill']):+.6f}   {status}")
        if bad:
            ok = False

    print()
    if ok:
        print("闸门通过：新流程可以复现主实验，允许进入扩样本分析。")
        return 0
    print("闸门未通过：请先排查流程差异，不要继续扩样本分析。")
    return 1


def run_full(cfg: dict) -> int:
    data = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    universe = data["universe"]
    print("=" * 78)
    print(f"跨股票验证：{len(universe)} 只沪深300 成分股")
    print(f"选股规则：{data['ordering']}（快照 {data['snapshot_date']}）")
    print("=" * 78)

    per_stock, frames = analyze_universe(universe, cfg)
    summary = summarize(per_stock, cfg)
    sens = run_sensitivity(frames, cfg)

    out_dir = ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_stock.to_csv(ROOT / cfg["outputs"]["per_stock_csv"], index=False)
    sens.to_csv(ROOT / cfg["outputs"]["sensitivity_csv"], index=False)
    pd.DataFrame([summary]).to_csv(ROOT / cfg["outputs"]["summary_csv"], index=False)

    print("\n" + "=" * 78)
    print("主要指标（预注册）：跨股票 Skill 中位数")
    print("=" * 78)
    print(f"  股票数            {summary['n_stocks']}")
    print(f"  中位数 Skill      {summary['median_skill']:+.2%}")
    print(f"  95% CI（cluster） [{summary['median_skill_ci_lo']:+.2%}, "
          f"{summary['median_skill_ci_hi']:+.2%}]")
    print(f"  均值 / 标准差     {summary['mean_skill']:+.2%} / "
          f"{summary['std_skill']:.2%}")
    print(f"  四分位区间        [{summary['skill_q1']:+.2%}, "
          f"{summary['skill_q3']:+.2%}]")
    print(f"  最小 / 最大       {summary['skill_min']:+.2%} / "
          f"{summary['skill_max']:+.2%}")
    print(f"  正 Skill 占比     {summary['positive_share']:.1%}")
    print(f"  Spearman 中位数   {summary['median_spearman']:.3f} "
          f"（基线 {summary['median_spearman_base']:.3f}）")
    print(f"  离散度比中位数    {summary['median_disp_ratio']:.2f}")
    print(f"\n  >>> 按预注册门槛判定：{summary['verdict']} <<<")

    print("\n" + "=" * 78)
    print("预注册敏感性检验")
    print("=" * 78)
    print(f"{'检验':<16}{'股票数':>6}{'中位数 Skill':>14}{'正占比':>9}"
          f"{'中位偏差':>10}")
    print("-" * 78)
    for _, r in sens.iterrows():
        bias = f"{r['median_bias']:+.2f}" if "median_bias" in r and pd.notna(
            r.get("median_bias")) else "—"
        print(f"{r['check']:<16}{int(r['n_stocks']):>6}"
              f"{r['median_skill']:>+14.2%}{r['positive_share']:>9.0%}"
              f"{bias:>10}")

    try:
        paths = plot_study(per_stock, summary, ROOT / cfg["outputs"]["chart_dir"])
        print(f"\n图表：{[str(p.relative_to(ROOT)) for p in paths]}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n图表失败（不影响结果）：{type(exc).__name__}: {exc}")

    print(f"\n逐股票：{cfg['outputs']['per_stock_csv']}")
    print(f"汇总：{cfg['outputs']['summary_csv']}")
    print(f"敏感性：{cfg['outputs']['sensitivity_csv']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true",
                    help="只跑复现闸门（原有 3 只）")
    args = ap.parse_args()

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    if args.gate:
        return run_gate(cfg)
    return run_full(cfg)


if __name__ == "__main__":
    sys.exit(main())
