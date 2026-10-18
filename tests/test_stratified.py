"""分层抽样实验的测试：预注册完整性、名单结构、判定规则、结果自洽。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "configs" / "stratified_universe.json"
UNIVERSE = ROOT / "configs" / "universe_stratified_63.json"
BOARDS = ["沪市主板", "深市主板", "创业板"]
SIZES = ["大市值", "中市值", "小市值"]


@pytest.fixture(scope="module")
def cfg():
    return json.loads(CFG.read_text(encoding="utf-8"))


def test_preregistration_record_is_bom_free():
    """预注册记录必须是标准 JSON（上一轮踩过 BOM 的坑）。"""
    p = ROOT / "reports" / "stratified" / "preregistration.json"
    assert p.exists()
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "含有 UTF-8 BOM"
    d = json.loads(raw.decode("utf-8"))
    for k in ("recorded_at_utc", "base_commit", "config_sha256",
              "preregistration_sha256"):
        assert k in d


def test_preregistration_hashes_match_files():
    """记录的哈希必须与当前文件一致——否则预注册已被事后修改。"""
    import hashlib
    rec = json.loads((ROOT / "reports" / "stratified" /
                      "preregistration.json").read_text(encoding="utf-8"))

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assert sha(CFG) == rec["config_sha256"], "配置文件在预注册后被改动"
    assert sha(ROOT / "docs" / "preregistration_stratified.md") == \
        rec["preregistration_sha256"], "预注册文档被改动"


def test_stratified_universe_structure(cfg):
    """名单必须满足：63 只、9 格各 7 只、板块在白名单内、无科创板。"""
    data = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    uni = data["universe"]
    assert len(uni) == cfg["universe_rule"]["target_n"] == 63

    counts = pd.Series([f"{u['board']}|{u['size']}" for u in uni]).value_counts()
    assert len(counts) == 9, "应有 9 个格子"
    assert (counts == cfg["universe_rule"]["quota_per_cell"]).all(), \
        f"每格必须 7 只，实际 {counts.to_dict()}"

    for u in uni:
        assert u["board"] in BOARDS, f"非法板块 {u['board']}"
        assert u["size"] in SIZES, f"非法市值层 {u['size']}"
        assert not u["ticker"].startswith("688"), "科创板应被排除"
        assert (ROOT / u["bars"]).exists(), f"缺少数据 {u['bars']}"


def test_no_performance_based_exclusion():
    """排除清单只能包含数据完整性原因。"""
    p = ROOT / "reports" / "stratified" / "exclusions.csv"
    if not p.exists():
        pytest.skip("无排除记录")
    df = pd.read_csv(p)
    if len(df) == 0:
        return
    allowed = ("bar 数不足", "起始过晚", "结束过早", "每日 bar", "抓取失败", "无数据")
    for _, r in df.iterrows():
        assert any(k in str(r["reason"]) for k in allowed), \
            f"可疑的排除原因：{r['reason']}"


def test_purge_equals_horizon(cfg):
    assert cfg["split"]["purge_rows"] == cfg["target"]["horizon_days"]


def test_summary_recomputable_from_per_stock():
    per_path = ROOT / "reports" / "stratified" / "per_stock.csv"
    sum_path = ROOT / "reports" / "stratified" / "summary.csv"
    if not per_path.exists() or not sum_path.exists():
        pytest.skip("尚未生成结果")
    per = pd.read_csv(per_path)
    s = pd.read_csv(sum_path).iloc[0]
    assert abs(per["skill"].median() - s["median_skill"]) < 1e-9
    assert abs((per["skill"] > 0).mean() - s["positive_share"]) < 1e-12
    assert int(s["n_stocks"]) == 63


def test_cells_sum_to_63():
    p = ROOT / "reports" / "stratified" / "cells.csv"
    if not p.exists():
        pytest.skip("尚未生成分层结果")
    cells = pd.read_csv(p)
    assert len(cells) == 9
    assert int(cells["n"].sum()) == 63


def test_reproducibility_gate_matches_main():
    """分层流程必须能重现主实验的 3 只股票。"""
    main_results = ROOT / "reports" / "research" / "results_main.csv"
    if not main_results.exists():
        pytest.skip("主实验未生成")
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    main_cfg = json.loads((ROOT / "configs" / "research_main.json")
                          .read_text(encoding="utf-8"))
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


def test_negative_skill_stocks_are_high_volatility():
    """记录一个描述性事实：本样本中负 Skill 的股票波动水平全部偏高。

    这不是统计检验，只是防止以后改数据/改流程时该模式被悄悄破坏。
    """
    p = ROOT / "reports" / "stratified" / "per_stock.csv"
    if not p.exists():
        pytest.skip("尚未生成结果")
    d = pd.read_csv(p)
    neg = d[d["skill"] < 0]
    if len(neg) == 0:
        pytest.skip("本次样本无负 Skill")
    assert neg["mean_vol"].min() > 0.45, \
        "出现了低波动的负 Skill 股票，与报告描述不符，需复查"
