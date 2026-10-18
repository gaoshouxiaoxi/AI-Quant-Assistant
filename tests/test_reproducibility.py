"""可复现性测试：协议完整性、确定性、离线可运行、基线与模型同样本。

这些测试守住"别人能从干净环境复现核心结果"这条底线。
"""

import json
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "configs" / "research_main.json"


@pytest.fixture(scope="module")
def cfg():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def test_protocol_has_required_keys(cfg):
    for key in ("protocol_version", "seed", "universe", "target", "split",
                "models", "baselines", "metrics", "outputs"):
        assert key in cfg, f"协议缺少字段 {key}"
    assert cfg["seed"] == 42
    assert len(cfg["universe"]) >= 3


def test_purge_equals_label_horizon(cfg):
    """协议自洽：purge 行数必须等于标签前视窗口。"""
    assert cfg["split"]["purge_rows"] == cfg["target"]["horizon_days"]


def test_universe_files_exist_and_are_committed_data(cfg):
    for item in cfg["universe"]:
        path = ROOT / item["bars"]
        assert path.exists(), f"缺少数据文件 {item['bars']}"
        assert "data/minute" in item["bars"]


def test_protocol_declares_no_network(cfg):
    assert cfg["data"]["network_required"] is False


def _load_one(ticker: str):
    from research_pipeline import (daily_rv, evaluate_baseline,  # noqa: PLC0415
                                   evaluate_model, prepare_dataset, read_bars)
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    item = next(u for u in cfg["universe"] if u["ticker"] == ticker)
    bars = read_bars(ROOT / item["bars"])
    ds = prepare_dataset(bars, cfg)
    return cfg, ds


def test_pipeline_is_deterministic():
    """同一输入跑两次，结果必须完全一致（含 RF，已固定随机种子）。"""
    from research_pipeline import (evaluate_model, walk_forward_indices)  # noqa: PLC0415
    cfg, ds = _load_one("600519")
    folds = walk_forward_indices(len(ds), cfg["split"]["n_splits"],
                                 cfg["split"]["min_train_frac"],
                                 cfg["split"]["purge_rows"])
    test_index = ds.index[np.concatenate([te for _, te in folds])].sort_values()

    spec = next(m for m in cfg["models"] if m["name"] == "RandomForest")
    a = evaluate_model(ds, spec, cfg, folds, test_index)
    b = evaluate_model(ds, spec, cfg, folds, test_index)
    for k in ("r2", "mae", "skill", "spearman", "disp_ratio", "calib_slope"):
        assert a[k] == b[k], f"{k} 不确定：{a[k]} vs {b[k]}"


def test_baselines_share_model_sample_size():
    """基线与模型必须用同一批测试日期，n 必须相等。"""
    from research_pipeline import (evaluate_baseline, evaluate_model,  # noqa: PLC0415
                                   walk_forward_indices)
    cfg, ds = _load_one("600519")
    folds = walk_forward_indices(len(ds), cfg["split"]["n_splits"],
                                 cfg["split"]["min_train_frac"],
                                 cfg["split"]["purge_rows"])
    test_index = ds.index[np.concatenate([te for _, te in folds])].sort_values()

    model = evaluate_model(ds, cfg["models"][0], cfg, folds, test_index)
    base = evaluate_baseline(ds, cfg["baselines"][0], test_index)
    assert model["n"] == base["n"] == len(test_index)


def test_runs_with_network_disabled(monkeypatch):
    """禁用 socket 后仍能跑完一个标的——证明主实验不依赖网络。"""
    from research_pipeline import (evaluate_model, prepare_dataset,  # noqa: PLC0415
                                   read_bars, walk_forward_indices)

    def boom(*args, **kwargs):
        raise RuntimeError("测试禁用了网络")

    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "socket", boom)

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    item = cfg["universe"][0]
    bars = read_bars(ROOT / item["bars"])
    ds = prepare_dataset(bars, cfg)
    folds = walk_forward_indices(len(ds), cfg["split"]["n_splits"],
                                 cfg["split"]["min_train_frac"],
                                 cfg["split"]["purge_rows"])
    test_index = ds.index[np.concatenate([te for _, te in folds])].sort_values()
    r = evaluate_model(ds, cfg["models"][0], cfg, folds, test_index)
    assert r["n"] > 0


def test_reproduces_committed_reference():
    """重算结果必须与已提交的参考结果一致（容差 1e-9，线性模型为确定性）。"""
    from research_pipeline import (evaluate_model,  # noqa: PLC0415
                                   walk_forward_indices)
    cfg, ds = _load_one("600519")
    ref_path = ROOT / cfg["outputs"]["results_csv"]
    if not ref_path.exists():
        pytest.skip("尚未生成参考结果")

    folds = walk_forward_indices(len(ds), cfg["split"]["n_splits"],
                                 cfg["split"]["min_train_frac"],
                                 cfg["split"]["purge_rows"])
    test_index = ds.index[np.concatenate([te for _, te in folds])].sort_values()
    spec = next(m for m in cfg["models"] if m["name"] == "HAR-3")
    r = evaluate_model(ds, spec, cfg, folds, test_index)

    ref = pd.read_csv(ref_path)
    hit = ref[(ref["ticker"].astype(str).str.zfill(6) == "600519")
              & (ref["model"] == "HAR-3")]
    assert len(hit) == 1, "参考结果里应有且只有一条 600519 / HAR-3"

    for col in ("n", "r2", "mae", "skill", "spearman", "disp_ratio",
                "calib_slope"):
        assert abs(float(hit[col].iloc[0]) - float(r[col])) < 1e-9, \
            f"{col} 与参考结果不一致"


def test_compare_detects_drift(tmp_path):
    """校验函数必须能发现数值漂移，否则 --check 形同虚设。"""
    from research_pipeline import compare_to_reference
    ref = pd.DataFrame([{"ticker": "600519", "model": "HAR-3", "n": 278,
                         "r2": 0.17, "mae": 0.5, "skill": 0.25,
                         "spearman": 0.4, "disp_ratio": 0.36,
                         "calib_slope": 1.1}])
    path = tmp_path / "ref.csv"
    ref.to_csv(path, index=False)

    same = ref.copy()
    assert compare_to_reference(same, path) == []

    drifted = ref.copy()
    drifted.loc[0, "skill"] = 0.26
    issues = compare_to_reference(drifted, path)
    assert issues and "skill" in issues[0]


def test_robustness_study_structure():
    """稳健性研究必须产出约定的单元数与列。"""
    from research_pipeline import run_robustness, summarize_robustness  # noqa: PLC0415
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    study = run_robustness(cfg)

    n_tickers = len(cfg["universe"])
    assert len(study) == n_tickers * (1 + 2 + 3 + 3), "单元数应为 3 x 9"
    for col in ("ticker", "scheme", "group", "n", "skill", "spearman",
                "spearman_base", "disp_ratio"):
        assert col in study.columns

    summary = summarize_robustness(study)
    assert set(summary["切法"]) == {"时间二分", "时间三分", "波动三档"}


def test_robustness_reproduces_committed_cells():
    """重算的稳健性单元必须与已提交的 study_cells.csv 一致。"""
    from research_pipeline import run_robustness  # noqa: PLC0415
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    ref_path = ROOT / "reports" / "research" / "study_cells.csv"
    if not ref_path.exists():
        pytest.skip("尚未生成稳健性参考结果")

    ref = pd.read_csv(ref_path)
    new = run_robustness(cfg)

    key = ["ticker", "scheme", "group"]
    for df in (ref, new):
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    merged = ref.merge(new, on=key, suffixes=("_ref", "_new"))
    assert len(merged) == len(ref), "单元集合与参考不一致"

    for col in ("n", "skill", "spearman", "spearman_base", "disp_ratio"):
        diff = (merged[f"{col}_ref"] - merged[f"{col}_new"]).abs().max()
        assert diff < 1e-9, f"{col} 与参考结果不一致（最大差异 {diff}）"
