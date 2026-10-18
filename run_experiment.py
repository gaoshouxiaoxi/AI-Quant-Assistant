"""统一实验入口：一条命令从入库数据重新生成主报告。

用法
----
    python run_experiment.py                 # 跑主实验，写结果与 manifest
    python run_experiment.py --check         # 跑并与已提交结果逐项比对
    python run_experiment.py --config configs/research_main.json

设计要点
--------
- 只读仓库内已入库的 60 分钟 bar，**不需要联网**
- 所有协议参数在 configs/research_main.json，代码无魔法数字
- 输出 reports/research/results_main.csv 与 manifest.json（含 git commit、
  依赖版本、数据哈希、随机种子）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from research_pipeline import (compare_to_reference, load_protocol,  # noqa: E402
                               plot_study, plot_summary, run_experiment,
                               run_robustness, summarize_robustness)


def main() -> int:
    ap = argparse.ArgumentParser(description="AI-Quant-Assistant 可复现实验入口")
    ap.add_argument("--config", default="configs/research_main.json",
                    help="协议配置文件（默认 configs/research_main.json）")
    ap.add_argument("--check", action="store_true",
                    help="与已提交的结果逐项比对（用于复现性校验）")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="--check 的数值容差（默认 1e-6）")
    ap.add_argument("--no-chart", action="store_true", help="不生成图表")
    ap.add_argument("--study", action="store_true",
                    help="额外跑稳健性研究（时间子区间 / 波动档位）")
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        print(f"找不到配置：{cfg_path}")
        return 2

    cfg, _ = load_protocol(cfg_path)
    print("=" * 78)
    print(f"可复现实验：{cfg['description']}")
    print(f"协议版本 {cfg['protocol_version']}   随机种子 {cfg['seed']}")
    print(f"目标 log(未来 {cfg['target']['horizon_days']} 日均值 RV)   "
          f"切分 {cfg['split']['n_splits']} 折  purge={cfg['split']['purge_rows']}")
    print("=" * 78)

    results, manifest = run_experiment(cfg, cfg_path)

    ref_csv = ROOT / cfg["outputs"]["results_csv"]
    out_manifest = ROOT / cfg["outputs"]["manifest"]
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    # 关键：--check 时绝不能覆盖参考结果，否则等于和自己比、永远通过。
    # 校验模式把重算结果写到 *.recomputed.csv，参考文件保持不动。
    if args.check:
        out_csv = ref_csv.with_name(ref_csv.stem + ".recomputed.csv")
    else:
        out_csv = ref_csv
    results.to_csv(out_csv, index=False)
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print(f"\n{'标的':<8}{'模型':<20}{'n':>5}{'R²':>9}{'Skill':>9}"
          f"{'Spearman':>10}{'离散度':>9}")
    print("-" * 78)
    for _, r in results.iterrows():
        print(f"{r['name']:<8}{r['model']:<20}{int(r['n']):>5}"
              f"{r['r2']:>9.3f}{r['skill']:>+9.2%}{r['spearman']:>10.3f}"
              f"{r['disp_ratio']:>9.2f}")

    print("\n跨标的平均：")
    agg = results.groupby("model")[["skill", "spearman", "disp_ratio"]].mean()
    for model, r in agg.sort_values("skill", ascending=False).iterrows():
        print(f"  {model:<20} Skill {r['skill']:>+7.2%}   "
              f"Spearman {r['spearman']:.3f}   离散度 {r['disp_ratio']:.2f}")

    if not args.no_chart:
        try:
            charts = plot_summary(results, ROOT / cfg["outputs"]["chart_dir"])
            print(f"\n图表：{[str(c.relative_to(ROOT)) for c in charts]}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n图表生成失败（不影响结果）：{type(exc).__name__}: {exc}")

    print(f"\n结果：{out_csv.relative_to(ROOT)}")
    print(f"元数据：{out_manifest.relative_to(ROOT)}")
    print(f"耗时 {manifest['runtime_seconds']} 秒   "
          f"git {manifest['git']['commit'][:7] or 'n/a'}"
          f"{' (dirty)' if manifest['git']['dirty'] else ''}")

    if args.study:
        print("\n" + "=" * 78)
        print("稳健性研究：把样本外窗口切开，看结论是否稳定")
        print("=" * 78)
        study = run_robustness(cfg)
        study_path = ref_csv.parent / "study_cells.csv"
        study.to_csv(study_path, index=False)

        summary = summarize_robustness(study)
        summary_path = ref_csv.parent / "study_summary.csv"
        summary.to_csv(summary_path, index=False)

        print(f"\n{'切法':<10}{'单元数':>6}{'Skill>0':>9}{'Spearman>基线':>15}"
              f"{'平均Skill':>11}{'平均Spearman':>14}{'平均离散度':>11}")
        print("-" * 78)
        for _, r in summary.iterrows():
            print(f"{r['切法']:<10}{int(r['单元数']):>6}{int(r['Skill>0']):>9}"
                  f"{int(r['Spearman>基线']):>15}"
                  f"{r['平均 Skill']:>+11.2%}{r['平均 Spearman']:>14.3f}"
                  f"{r['平均离散度比']:>11.2f}")

        print("\n按波动档位的明细（跨标的平均）：")
        reg = study[study["scheme"] == "regime"]
        for g, sub in reg.groupby("group"):
            print(f"  {g:<6} n≈{int(sub['n'].mean()):>3}  "
                  f"Skill {sub['skill'].mean():>+7.2%}  "
                  f"Spearman {sub['spearman'].mean():.3f}"
                  f"（基线 {sub['spearman_base'].mean():.3f}）  "
                  f"离散度 {sub['disp_ratio'].mean():.2f}")

        if not args.no_chart:
            try:
                paths = plot_study(study, ROOT / cfg["outputs"]["chart_dir"])
                print(f"\n图表：{[str(p.relative_to(ROOT)) for p in paths]}")
            except Exception as exc:  # noqa: BLE001
                print(f"\n图表失败（不影响结果）：{type(exc).__name__}: {exc}")
        print(f"\n单元明细：{study_path.relative_to(ROOT)}")
        print(f"汇总：{summary_path.relative_to(ROOT)}")

    if args.check:
        print("\n" + "=" * 78)
        print("复现性校验（与已提交结果比对）")
        print("=" * 78)
        issues = compare_to_reference(results, ref_csv, tol=args.tol)
        if issues:
            print(f"发现 {len(issues)} 处不一致（容差 {args.tol}）：")
            for i in issues[:30]:
                print(f"  - {i}")
            print(f"\n本次重算结果已存到 {out_csv.relative_to(ROOT)}，"
                  f"参考文件未被改动。")
            return 1
        print(f"全部一致（容差 {args.tol}）—— 复现成功")
    return 0


if __name__ == "__main__":
    sys.exit(main())
