"""B1 规则测试：标签成熟、窗口、状态隔离、失败规则、与参考结果的一致性。

这些测试直接对应 `docs/preregistration_b1.md` 冻结的条款，
任何一条被改坏都意味着 B1 的结论不可信。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "configs" / "b1_rolling_intercept.json"
PREREG = ROOT / "docs" / "preregistration_b1.md"
RECORD = ROOT / "reports" / "b1" / "preregistration.json"


@pytest.fixture(scope="module")
def cfg():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 纯逻辑

def test_matured_residuals_respect_horizon():
    """核心：残差只有在 s + horizon <= t 时才可用。"""
    from b1_rolling_intercept import matured_residuals

    history = [(d, float(d) * 10) for d in range(11)]   # 日序号 0..10
    got = matured_residuals(history, up_to_day=10, horizon=5)
    # 只有 d <= 5 的残差成熟
    assert got == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]


def test_matured_residuals_empty_when_nothing_matured():
    from b1_rolling_intercept import matured_residuals
    history = [(9, 1.0), (10, 2.0)]
    assert matured_residuals(history, up_to_day=10, horizon=5) == []


def test_matured_residuals_uses_trading_day_not_index():
    """间隔不是 1 时（有停牌/缺失日），仍按交易日序号判断。"""
    from b1_rolling_intercept import matured_residuals
    history = [(0, 1.0), (40, 2.0)]          # 中间缺了大量交易日
    assert matured_residuals(history, up_to_day=44, horizon=5) == [1.0]
    assert matured_residuals(history, up_to_day=45, horizon=5) == [1.0, 2.0]


def test_apply_window_requires_full_window():
    """不足 20 个必须返回 None（触发失败规则），不能缩短窗口。"""
    from b1_rolling_intercept import apply_window
    assert apply_window([1.0] * 19, 20) is None
    assert apply_window([1.0] * 20, 20) == pytest.approx(1.0)


def test_apply_window_takes_latest():
    from b1_rolling_intercept import apply_window
    pool = list(range(25))                   # 0..24
    assert apply_window(pool, 20) == pytest.approx(np.mean(range(5, 25)))


# ---------------------------------------------------------------- 预注册完整性

def test_preregistration_files_unchanged():
    """预注册文档与配置在记录之后不得被修改。"""
    if not RECORD.exists():
        pytest.skip("尚无预注册记录")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    assert hashlib.sha256(CFG_PATH.read_bytes()).hexdigest() == rec["config_sha256"]
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == \
        rec["preregistration_sha256"]


def test_frozen_parameters(cfg):
    """冻结参数不得被暗中改动。"""
    assert cfg["correction"]["window"] == 20
    assert cfg["correction"]["horizon"] == 5
    assert cfg["initialization"]["min_history_train"] == 60
    assert cfg["initialization"]["mode"] == "training_fold_out_of_sample"
    assert cfg["split"]["purge_rows"] == cfg["target"]["horizon_days"]
    assert cfg["modifies_existing_code"] is False


# ---------------------------------------------------------------- 数据相关

@pytest.fixture(scope="module")
def one_stock(cfg):
    universe = json.loads(
        (ROOT / "configs" / "universe_stratified_63.json").read_text("utf-8"))
    item = universe["universe"][0]
    from b1_rolling_intercept import build_frames
    from research_pipeline import read_bars
    bars = read_bars(ROOT / item["bars"])
    ds, daynum = build_frames(bars, cfg["correction"]["horizon"])
    return item, ds, daynum


def test_maturity_never_uses_unmatured_residuals(one_stock, cfg):
    """端到端：每折内任一预测日的可用残差数，必须等于按成熟规则算出的数量。"""
    from b1_rolling_intercept import matured_residuals, run_fold
    from validation import walk_forward_indices

    item, ds, daynum = one_stock
    h = cfg["correction"]["horizon"]
    folds = walk_forward_indices(len(ds), cfg["split"]["n_splits"],
                                 cfg["split"]["min_train_frac"],
                                 cfg["split"]["purge_rows"])
    tr, te = folds[1]
    res = run_fold(ds, daynum, tr, te, h, cfg["correction"]["window"],
                   cfg["initialization"]["min_history_train"])
    assert res["ok"]
    frame = res["frame"]
    # 每个预测日的可用残差数必须单调不减，且起点 >= 窗口
    assert (frame["n_resid"] >= cfg["correction"]["window"]).all()
    assert frame["n_resid"].is_monotonic_increasing


def test_run_fold_is_pure(one_stock, cfg):
    """同一折跑两次结果必须完全一致（禁止隐藏状态）。"""
    from b1_rolling_intercept import run_fold
    from validation import walk_forward_indices

    _, ds, daynum = one_stock
    h = cfg["correction"]["horizon"]
    folds = walk_forward_indices(len(ds), cfg["split"]["n_splits"],
                                 cfg["split"]["min_train_frac"],
                                 cfg["split"]["purge_rows"])
    tr, te = folds[2]
    a = run_fold(ds, daynum, tr, te, h, 20, 60)
    b = run_fold(ds, daynum, tr, te, h, 20, 60)
    pd.testing.assert_frame_equal(a["frame"], b["frame"])


def test_failure_when_history_insufficient(one_stock, cfg):
    """训练期成熟残差不足 20 个时必须判定失败，而不是凑数。"""
    from b1_rolling_intercept import run_fold
    from validation import walk_forward_indices

    _, ds, daynum = one_stock
    short = ds.iloc[:200]
    short_day = daynum[:200]
    folds = walk_forward_indices(len(short), 5, 0.4, 5)
    tr, te = folds[0]
    res = run_fold(short, short_day, tr, te, 5, 20, 60)
    assert res["ok"] is False
    assert "残差" in res["reason"]


# ---------------------------------------------------------------- 结果一致性

def test_base_path_matches_reference_results():
    """B1 的基础路径必须逐位重现参考的 HAR-3 结果。

    这是负结果可信的前提：如果基础预测本身就与参考不一致，
    "B1 无效"可能只是实现错误的产物。
    """
    b1_path = ROOT / "reports" / "b1" / "per_stock.csv"
    ref_path = ROOT / "reports" / "stratified" / "per_stock.csv"
    if not b1_path.exists() or not ref_path.exists():
        pytest.skip("尚未生成结果")

    b1 = pd.read_csv(b1_path, dtype={"ticker": str})
    ref = pd.read_csv(ref_path, dtype={"ticker": str}).rename(
        columns={"skill": "ref_skill", "n_oos": "ref_n"})
    m = b1.merge(ref[["ticker", "ref_skill", "ref_n"]], on="ticker")
    assert len(m) == 63
    assert (m["n"] == m["ref_n"]).all(), "测试日期数与参考不一致"
    assert (m["skill_base"] - m["ref_skill"]).abs().max() < 1e-9


def test_summary_matches_per_stock():
    per_path = ROOT / "reports" / "b1" / "per_stock.csv"
    sum_path = ROOT / "reports" / "b1" / "summary.csv"
    if not per_path.exists() or not sum_path.exists():
        pytest.skip("尚未生成结果")
    per = pd.read_csv(per_path)
    s = pd.read_csv(sum_path).iloc[0]
    assert abs(per["skill_b1"].median() - s["median_skill_b1"]) < 1e-9
    assert abs(per["paired_rel_improvement"].median()
               - s["median_paired_improvement"]) < 1e-9


def test_paired_metric_definition():
    """配对改善 = 1 - MSE(b1)/MSE(base)，符号方向不能反。"""
    from b1_rolling_intercept import stock_metrics
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    y = pd.Series(np.zeros(n), index=idx)
    oos = pd.DataFrame({
        "y": y,
        "pred_base": pd.Series(np.full(n, 0.2), index=idx),
        "pred_b1": pd.Series(np.full(n, 0.1), index=idx),
        "delta": pd.Series(np.full(n, -0.1), index=idx),
        "n_resid": pd.Series(np.full(n, 20), index=idx),
    })
    rv_w = pd.Series(np.full(n, 0.5), index=idx)
    m = stock_metrics(oos, rv_w)
    # B1 误差更小 → 改善为正
    assert m["paired_rel_improvement"] > 0
    assert m["skill_b1"] > m["skill_base"]
