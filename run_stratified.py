"""分层抽样验证运行器。

用法
----
    python run_stratified.py --gate   # 复现闸门（用新流程重跑主实验 3 只）
    python run_stratified.py          # 63 只分层实验

严格遵循 `docs/preregistration_stratified.md`：
主要指标是**全部 63 只**的中位数 Skill；分层结果为预先声明的次要描述性分析。
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

from cross_stock_study import (analyze_universe, block_bootstrap_skill,  # noqa: E402
                               cluster_bootstrap_median, decide, run_sensitivity,
                               skill_of)
from cross_stock_study import plot_study  # noqa: E402

CFG_PATH = ROOT / "configs" / "stratified_universe.json"
UNIVERSE_PATH = ROOT / "configs" / "universe_stratified_63.json"
MAIN_CFG = ROOT / "configs" / "research_main.json"
MAIN_RESULTS = ROOT / "reports" / "research" / "results_main.csv"

BOARD_ORDER = ["沪市主板", "深市主板", "创业板"]
SIZE_ORDER = ["大市值", "中市值", "小市值"]


def run_gate(cfg: dict) -> int:
    print("=" * 78)
    print("复现闸门：新流程重跑主实验 3 只，与主结果逐项比对")
    print("=" * 78)
    main_cfg = json.loads(MAIN_CFG.read_text(encoding="utf-8"))
    universe = [{"ticker": u["ticker"], "name": u["name"], "bars": u["bars"]}
                for u in main_cfg["universe"]]
    per_stock, _ = analyze_universe(universe, cfg)

    ref = pd.read_csv(MAIN_RESULTS, dtype={"ticker": str})
    ref = ref[ref["model"] == "HAR-3"]

    ok = True
    print()
    for _, r in per_stock.iterrows():
        m = ref[ref["ticker"] == r["ticker"]]
        if len(m) != 1:
            print(f"  {r['ticker']}: 参考缺失")
            ok = False
            continue
        d_skill = abs(float(m.iloc[0]["skill"]) - r["skill"])
        d_spear = abs(float(m.iloc[0]["spearman"]) - r["spearman"])
        good = d_skill < 1e-9 and d_spear < 1e-9
        print(f"  {r['ticker']} {r['name']:<6} Skill {r['skill']:+.6f}  "
              f"{'✅' if good else f'❌ Δskill={d_skill:.2e}'}")
        ok = ok and good

    print("\n闸门" + ("通过：流程一致，允许继续。" if ok else "未通过：停止。"))
    return 0 if ok else 1


def cell_table(per: pd.DataFrame) -> pd.DataFrame:
    """9 个格子的描述性统计（预先声明的次要分析）。"""
    rows = []
    for board in BOARD_ORDER:
        for size in SIZE_ORDER:
            sub = per[(per["board"] == board) & (per["size"] == size)]
            if len(sub) == 0:
                continue
            rows.append({
                "board": board, "size": size, "n": len(sub),
                "median_skill": float(sub["skill"].median()),
                "mean_skill": float(sub["skill"].mean()),
                "positive_share": float((sub["skill"] > 0).mean()),
                "median_spearman": float(sub["spearman"].median()),
                "median_disp_ratio": float(sub["disp_ratio"].median()),
                "median_mean_vol": float(sub["mean_vol"].median()),
            })
    return pd.DataFrame(rows)


def grouped_table(per: pd.DataFrame, by: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for key in order:
        sub = per[per[by] == key]
        if len(sub) == 0:
            continue
        rows.append({"group": key, "n": len(sub),
                     "median_skill": float(sub["skill"].median()),
                     "mean_skill": float(sub["skill"].mean()),
                     "positive_share": float((sub["skill"] > 0).mean()),
                     "median_spearman": float(sub["spearman"].median()),
                     "median_mean_vol": float(sub["mean_vol"].median())})
    return pd.DataFrame(rows)


def overlap_exclusion(frames: list[tuple], overlap: set[str]) -> float:
    """剔除与前一实验重叠的股票后，中位数 Skill。"""
    skills = []
    for ticker, df, _ in frames:
        if ticker in overlap:
            continue
        skills.append(skill_of(df["err_m"].to_numpy(), df["err_b"].to_numpy()))
    return float(np.nanmedian(skills)) if skills else float("nan")


def plot_stratified(per: pd.DataFrame, cells: pd.DataFrame,
                    out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from charts import setup_chinese_font
        setup_chinese_font()
    except Exception:  # noqa: BLE001
        pass

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 热力图：板块 x 市值层的中位数 Skill
    mat = cells.pivot(index="board", columns="size", values="median_skill")
    mat = mat.reindex(index=BOARD_ORDER, columns=SIZE_ORDER) * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(mat.to_numpy(), cmap="RdYlGn", vmin=-40, vmax=40, aspect="auto")
    ax.set_xticks(range(len(SIZE_ORDER)))
    ax.set_xticklabels(SIZE_ORDER)
    ax.set_yticks(range(len(BOARD_ORDER)))
    ax.set_yticklabels(BOARD_ORDER)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=9)
    ax.set_title("各分层的中位数 Skill（%）")
    fig.colorbar(im, ax=ax, label="Skill (%)")
    fig.tight_layout()
    p = out_dir / "stratified_heatmap.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # 按板块 / 按市值的分布
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, by, order, title in (
            (axes[0], "board", BOARD_ORDER, "按板块"),
            (axes[1], "size", SIZE_ORDER, "按市值层")):
        data = [per[per[by] == k]["skill"].to_numpy() * 100 for k in order]
        # matplotlib 3.9+ 移除了 boxplot(labels=...)，必须用 tick_labels
        ax.boxplot(data, tick_labels=order)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Skill（%）")
        ax.set_title(f"{title}：Skill 分布")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = out_dir / "stratified_groups.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


def run_full(cfg: dict) -> int:
    data = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    universe = data["universe"]
    meta = {u["ticker"]: (u["board"], u["size"]) for u in universe}

    print("=" * 78)
    print(f"分层验证：{len(universe)} 只（3 板块 x 3 市值层 x 7 只）")
    print("=" * 78)

    per, frames = analyze_universe(universe, cfg)
    per["board"] = per["ticker"].map(lambda t: meta[t][0])
    per["size"] = per["ticker"].map(lambda t: meta[t][1])
    frames = [(t, df, ov) for (t, df, ov) in frames]

    vals = per["skill"].to_numpy()
    median = float(np.nanmedian(vals))
    lo, hi = cluster_bootstrap_median(
        vals, n_boot=cfg["confidence_intervals"]["across_stocks"]["n_resamples"])
    pos_share = float(np.mean(vals > 0))
    verdict = decide(median, lo, pos_share, cfg["decision_rule"])

    summary = {
        "n_stocks": int(len(per)),
        "median_skill": median,
        "median_skill_ci_lo": lo, "median_skill_ci_hi": hi,
        "mean_skill": float(np.nanmean(vals)),
        "std_skill": float(np.nanstd(vals, ddof=1)),
        "skill_q1": float(np.nanpercentile(vals, 25)),
        "skill_q3": float(np.nanpercentile(vals, 75)),
        "positive_share": pos_share,
        "median_spearman": float(np.nanmedian(per["spearman"])),
        "median_spearman_base": float(np.nanmedian(per["spearman_base"])),
        "median_disp_ratio": float(np.nanmedian(per["disp_ratio"])),
        "verdict": verdict,
    }

    cells = cell_table(per)
    by_board = grouped_table(per, "board", BOARD_ORDER)
    by_size = grouped_table(per, "size", SIZE_ORDER)

    out_dir = ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(ROOT / cfg["outputs"]["per_stock_csv"], index=False)
    cells.to_csv(ROOT / cfg["outputs"]["cell_csv"], index=False)
    pd.DataFrame([summary]).to_csv(ROOT / cfg["outputs"]["summary_csv"], index=False)
    pd.concat([by_board.assign(groupby="board"),
               by_size.assign(groupby="size")]).to_csv(
        out_dir / "grouped.csv", index=False)

    sens = run_sensitivity(frames, cfg)

    # 额外：剔除与前一实验重叠的股票
    overlap_path = out_dir / "overlap_with_cross_stock.csv"
    if overlap_path.exists():
        overlap = set(pd.read_csv(overlap_path, dtype={"ticker": str})["ticker"])
        m_wo = overlap_exclusion(frames, overlap)
        sens = pd.concat([sens, pd.DataFrame([{
            "check": "剔除与25只重叠", "n_stocks": len(frames) - len(overlap),
            "median_skill": m_wo,
            "positive_share": float("nan")}])], ignore_index=True)
    sens.to_csv(ROOT / cfg["outputs"]["sensitivity_csv"], index=False)

    print("\n" + "=" * 78)
    print("主要指标（预注册）：全部 63 只的 Skill 中位数")
    print("=" * 78)
    print(f"  中位数 Skill      {median:+.2%}")
    print(f"  95% CI（cluster） [{lo:+.2%}, {hi:+.2%}]")
    print(f"  均值 / 标准差     {summary['mean_skill']:+.2%} / {summary['std_skill']:.2%}")
    print(f"  四分位            [{summary['skill_q1']:+.2%}, {summary['skill_q3']:+.2%}]")
    print(f"  正 Skill 占比     {pos_share:.1%}")
    print(f"  Spearman 中位数   {summary['median_spearman']:.3f} "
          f"（基线 {summary['median_spearman_base']:.3f}）")
    print(f"  离散度比中位数    {summary['median_disp_ratio']:.2f}")
    print(f"\n  >>> 判定：{verdict} <<<")

    print("\n" + "=" * 78)
    print("分层结果（预先声明的描述性分析，不据此做显著性结论）")
    print("=" * 78)
    print(f"\n{'板块':<8}{'市值层':<8}{'n':>3}{'中位 Skill':>12}{'正占比':>8}"
          f"{'中位 Spearman':>14}{'中位波动':>10}")
    print("-" * 78)
    for _, r in cells.iterrows():
        print(f"{r['board']:<8}{r['size']:<8}{int(r['n']):>3}"
              f"{r['median_skill']:>+12.2%}{r['positive_share']:>8.0%}"
              f"{r['median_spearman']:>14.3f}{r['median_mean_vol']:>10.1%}")

    print("\n按板块合并：")
    for _, r in by_board.iterrows():
        print(f"  {r['group']:<8} n={int(r['n']):>2}  中位 Skill {r['median_skill']:>+7.2%}"
              f"  正占比 {r['positive_share']:>4.0%}  中位波动 {r['median_mean_vol']:.1%}")
    print("按市值层合并：")
    for _, r in by_size.iterrows():
        print(f"  {r['group']:<8} n={int(r['n']):>2}  中位 Skill {r['median_skill']:>+7.2%}"
              f"  正占比 {r['positive_share']:>4.0%}  中位波动 {r['median_mean_vol']:.1%}")

    print("\n" + "=" * 78)
    print("预注册敏感性检验")
    print("=" * 78)
    print(f"{'检验':<18}{'股票数':>6}{'中位数 Skill':>14}{'正占比':>9}")
    print("-" * 78)
    for _, r in sens.iterrows():
        ps = f"{r['positive_share']:.0%}" if pd.notna(r["positive_share"]) else "—"
        print(f"{r['check']:<18}{int(r['n_stocks']):>6}"
              f"{r['median_skill']:>+14.2%}{ps:>9}")

    try:
        paths = plot_stratified(per, cells, ROOT / cfg["outputs"]["chart_dir"])
        paths += plot_study(per, summary, ROOT / cfg["outputs"]["chart_dir"])
        print(f"\n图表：{[str(p.relative_to(ROOT)) for p in paths]}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n图表失败：{type(exc).__name__}: {exc}")

    print(f"\n逐股票：{cfg['outputs']['per_stock_csv']}")
    print(f"分层：{cfg['outputs']['cell_csv']}")
    print(f"汇总：{cfg['outputs']['summary_csv']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    return run_gate(cfg) if args.gate else run_full(cfg)


if __name__ == "__main__":
    sys.exit(main())
