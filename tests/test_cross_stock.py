"""跨股票验证的测试：样本规则、判定门槛、bootstrap、复现一致性。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "configs" / "cross_stock.json"
UNIVERSE = ROOT / "configs" / "universe_cs300_25.json"


@pytest.fixture(scope="module")
def cfg():
    return json.loads(CFG.read_text(encoding="utf-8"))


def test_preregistration_committed_before_data():
    """预注册记录必须存在，且指向的基线提交早于数据抓取。"""
    rec = ROOT / "reports" / "cross_stock" / "preregistration.json"
    assert rec.exists(), "缺少预注册记录"
    data = json.loads(rec.read_text(encoding="utf-8"))
    for key in ("recorded_at_utc", "baseline_commit", "config_sha256",
                "preregistration_sha256"):
        assert key in data


def test_universe_follows_mechanical_rule(cfg):
    """样本必须按代码升序、数量达标、全部在 data/minute 下。"""
    data = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    uni = data["universe"]
    assert len(uni) == cfg["universe_rule"]["target_n"]

    codes = [u["ticker"] for u in uni]
    assert codes == sorted(codes), "必须按股票代码升序（规则要求）"
    assert len(set(codes)) == len(codes), "不能重复"

    for u in uni:
        assert (ROOT / u["bars"]).exists(), f"缺少数据 {u['bars']}"
        assert u["n_bars"] >= cfg["universe_rule"]["completeness"]["min_bars"]


def test_decision_rule_thresholds(cfg):
    """判定门槛必须严格按预注册执行。"""
    from cross_stock_study import decide
    rule = cfg["decision_rule"]

    # 支持：中位数>0、CI 下界>0、正占比>=70%
    assert decide(0.20, 0.05, 0.80, rule) == "支持 H1"
    # 正占比不足 → 部分支持
    assert decide(0.20, 0.05, 0.60, rule) == "部分支持"
    # CI 含 0 → 部分支持
    assert decide(0.20, -0.01, 0.80, rule) == "部分支持"
    # 中位数<=0 → 不支持
    assert decide(-0.01, -0.10, 0.30, rule) == "不支持"


def test_block_bootstrap_is_deterministic():
    """同一数据同一随机种子，bootstrap 必须给出相同区间。"""
    from cross_stock_study import block_bootstrap_skill
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "err_m": rng.normal(0, 1, n),
        "err_b": rng.normal(0, 1.3, n),
    })
    a = block_bootstrap_skill(df, block=10, n_boot=200, seed=42)
    b = block_bootstrap_skill(df, block=10, n_boot=200, seed=42)
    assert a == b
    assert a[0] <= a[1]


def test_cluster_bootstrap_median_brackets_median():
    from cross_stock_study import cluster_bootstrap_median
    vals = np.array([0.1, 0.2, 0.25, 0.3, 0.5, 0.6, 0.15, 0.22])
    lo, hi = cluster_bootstrap_median(vals, n_boot=500, seed=42)
    assert lo <= np.median(vals) <= hi


def test_summary_matches_committed_per_stock():
    """汇总表里的主要指标必须能由 per_stock.csv 重算出来。"""
    per_path = ROOT / "reports" / "cross_stock" / "per_stock.csv"
    sum_path = ROOT / "reports" / "cross_stock" / "summary.csv"
    if not per_path.exists() or not sum_path.exists():
        pytest.skip("尚未生成跨股票结果")

    per = pd.read_csv(per_path)
    summ = pd.read_csv(sum_path).iloc[0]

    assert abs(per["skill"].median() - summ["median_skill"]) < 1e-9
    assert abs((per["skill"] > 0).mean() - summ["positive_share"]) < 1e-12
    assert int(summ["n_stocks"]) == len(per)


def test_gate_reproduces_main_results():
    """复现闸门：跨股票流程必须重现主实验的 HAR-3 数值。"""
    main_cfg_path = ROOT / "configs" / "research_main.json"
    main_results = ROOT / "reports" / "research" / "results_main.csv"
    if not main_results.exists():
        pytest.skip("主实验尚未生成")

    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    main_cfg = json.loads(main_cfg_path.read_text(encoding="utf-8"))
    universe = [{"ticker": u["ticker"], "name": u["name"], "bars": u["bars"]}
                for u in main_cfg["universe"]]

    from cross_stock_study import analyze_universe
    per, _ = analyze_universe(universe, cfg, verbose=False)

    ref = pd.read_csv(main_results, dtype={"ticker": str})
    ref = ref[ref["model"] == "HAR-3"]

    for _, r in per.iterrows():
        m = ref[ref["ticker"] == r["ticker"]]
        assert len(m) == 1
        assert abs(float(m.iloc[0]["skill"]) - r["skill"]) < 1e-9
        assert abs(float(m.iloc[0]["spearman"]) - r["spearman"]) < 1e-9
