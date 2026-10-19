"""B1 运行器：滚动截距修正 vs 原始 HAR-3（同一批股票、同一测试日期）。

用法
----
    python run_b1.py           # 跑 63 只
    python run_b1.py --limit 5 # 先小样本试跑

严格按 `docs/preregistration_b1.md` 的报告与判定规则执行。
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

from b1_rolling_intercept import (cluster_bootstrap_median,  # noqa: E402
                                  high_vol_group,
                                  paired_block_bootstrap, run_stock,
                                  stock_metrics)

CFG_PATH = ROOT / "configs" / "b1_rolling_intercept.json"
UNIVERSE_PATH = ROOT / "configs" / "universe_stratified_63.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 只（调试用）")
    args = ap.parse_args()

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))["universe"]
    if args.limit:
        universe = universe[: args.limit]

    print("=" * 82)
    print(f"B1 滚动截距修正：{len(universe)} 只股票")
    print(f"窗口 W={cfg['correction']['window']}  前视 h={cfg['correction']['horizon']}  "
          f"初始化={cfg['initialization']['mode']}  "
          f"min_history_train={cfg['initialization']['min_history_train']}")
    print("=" * 82)

    rows, failures, errors = [], [], []
    for k, item in enumerate(universe, 1):
        try:
            res = run_stock(item, cfg)
        except Exception as exc:  # noqa: BLE001
            errors.append({"ticker": item["ticker"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{k:>2}/{len(universe)}] {item['ticker']} 运行失败 {type(exc).__name__}")
            continue

        if not res["ok"]:
            for f in res["failures"]:
                failures.append({"ticker": item["ticker"], **f})
            print(f"  [{k:>2}/{len(universe)}] {item['ticker']} {item['name']:<6} "
                  f"协议失败：{res['failures'][0]['reason']}")
            continue

        for f in res["failures"]:
            failures.append({"ticker": item["ticker"], **f})

        oos = res["oos"]
        m = stock_metrics(oos, res["dataset"]["rv_w"])
        m["hv_vol"] = high_vol_group(res["dataset"], res["daynum"], cfg)

        err_base = (oos["pred_base"] - oos["y"]).to_numpy()
        err_b1 = (oos["pred_b1"] - oos["y"]).to_numpy()
        lo, hi = paired_block_bootstrap(
            err_base, err_b1,
            block=cfg["confidence_intervals"]["per_stock_paired"]["block_length_days"],
            n_boot=cfg["confidence_intervals"]["per_stock_paired"]["n_resamples"])
        m["paired_ci_lo"], m["paired_ci_hi"] = lo, hi
        m.update({"ticker": item["ticker"], "name": item["name"]})
        rows.append(m)

        print(f"  [{k:>2}/{len(universe)}] {item['ticker']} {item['name']:<6} "
              f"n={m['n']:>3}  Skill base {m['skill_base']:>+7.2%} → B1 {m['skill_b1']:>+7.2%}  "
              f"配对 Δ {m['paired_rel_improvement']:>+7.2%}  δ̄={m['mean_delta']:>+.3f}")

    if not rows:
        print("\n没有任何股票完成，终止。")
        return 1

    per = pd.DataFrame(rows)
    out_dir = ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 调试用的 --limit 绝不能覆盖主结果：加后缀单独存放
    suffix = f".limit{args.limit}" if args.limit else ""

    def out_path(key: str) -> Path:
        p = ROOT / cfg["outputs"][key]
        return p.with_name(p.stem + suffix + p.suffix) if suffix else p

    per.to_csv(out_path("per_stock_csv"), index=False)
    pd.DataFrame(failures).to_csv(out_path("failures_csv"), index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / f"errors{suffix}.csv", index=False)

    # ---- 高波动分组：横截面中位数切分（运行前一次性确定） ----
    cutoff = float(per["hv_vol"].median())
    per["hv_group"] = np.where(per["hv_vol"] >= cutoff, "高波动", "低波动")
    per[["ticker", "name", "hv_vol", "hv_group"]].to_csv(
        out_path("high_vol_groups_csv"), index=False)

    paired = per[["ticker", "name", "mse_base", "mse_b1", "skill_base",
                  "skill_b1", "paired_rel_improvement", "paired_ci_lo",
                  "paired_ci_hi", "mean_delta", "hv_group"]]
    paired.to_csv(out_path("paired_csv"), index=False)

    med_skill_b1 = float(per["skill_b1"].median())
    med_skill_base = float(per["skill_base"].median())
    med_paired = float(per["paired_rel_improvement"].median())
    ci_lo, ci_hi = cluster_bootstrap_median(
        per["paired_rel_improvement"].to_numpy(),
        n_boot=cfg["confidence_intervals"]["across_stocks"]["n_resamples"])

    rule = cfg["decision_metric"]["rule"]
    if med_paired <= 0:
        verdict = "无效"
    elif ci_lo > 0:
        verdict = "B1 有效"
    else:
        verdict = "部分有效"

    summary = {
        "n_stocks": int(len(per)),
        "n_failures": len(failures),
        "median_skill_base": med_skill_base,
        "median_skill_b1": med_skill_b1,
        "median_paired_improvement": med_paired,
        "paired_ci_lo": ci_lo, "paired_ci_hi": ci_hi,
        "positive_paired_share": float((per["paired_rel_improvement"] > 0).mean()),
        "median_mean_delta": float(per["mean_delta"].median()),
        "median_spearman_base": float(per["spearman_base"].median()),
        "median_spearman_b1": float(per["spearman_b1"].median()),
        "median_disp_base": float(per["disp_ratio_base"].median()),
        "median_disp_b1": float(per["disp_ratio_b1"].median()),
        "hv_cutoff": cutoff,
        "verdict": verdict,
    }
    pd.DataFrame([summary]).to_csv(out_path("summary_csv"), index=False)

    print("\n" + "=" * 82)
    print("主指标（预注册）：B1 的跨股票 Skill 中位数")
    print("=" * 82)
    print(f"  股票数（完成）      {summary['n_stocks']}")
    print(f"  协议失败            {summary['n_failures']}")
    print(f"  Skill 中位数 HAR-3  {med_skill_base:+.2%}")
    print(f"  Skill 中位数 B1     {med_skill_b1:+.2%}")
    print(f"  配对相对改善中位数  {med_paired:+.2%}  "
          f"95% CI [{ci_lo:+.2%}, {ci_hi:+.2%}]")
    print(f"  正改善占比          {summary['positive_paired_share']:.1%}")
    print(f"  平均截距修正 δ̄      {summary['median_mean_delta']:+.4f}")
    print(f"  Spearman 中位数     HAR-3 {summary['median_spearman_base']:.3f} → "
          f"B1 {summary['median_spearman_b1']:.3f}")
    print(f"  离散度比中位数      HAR-3 {summary['median_disp_base']:.2f} → "
          f"B1 {summary['median_disp_b1']:.2f}")
    print(f"\n  >>> 判定：{verdict} <<<")

    print("\n" + "=" * 82)
    print(f"高波动子集（按第 1 折训练期波动，切点 {cutoff:.1%}）")
    print("=" * 82)
    for g, sub in per.groupby("hv_group"):
        print(f"  {g}（n={len(sub)}）："
              f"Skill HAR-3 {sub['skill_base'].median():+.2%} → "
              f"B1 {sub['skill_b1'].median():+.2%}，"
              f"配对 Δ 中位 {sub['paired_rel_improvement'].median():+.2%}，"
              f"正改善 {(sub['paired_rel_improvement'] > 0).mean():.0%}")

    if failures:
        print(f"\n协议失败 {len(failures)} 条，见 {cfg['outputs']['failures_csv']}")
    print(f"\n逐股票：{cfg['outputs']['per_stock_csv']}")
    print(f"汇总：{cfg['outputs']['summary_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
