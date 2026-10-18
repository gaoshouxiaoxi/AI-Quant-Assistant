"""审计修复的回归测试：purging 与统一样本。

这些测试是为 2026-10 代码审查发现的问题写的，防止以后改回去。
"""

import numpy as np
import pandas as pd
import pytest

from validation import (assert_same_samples, common_index,
                        verify_no_overlap, walk_forward_indices)

HORIZON = 5


def test_walk_forward_without_purge_touches_boundary():
    """不 purge 时，训练集最后一行紧贴测试起点（这是被修复的旧行为）。"""
    folds = walk_forward_indices(200, n_splits=4, min_train=0.4, purge=0)
    for train_idx, test_idx in folds:
        assert train_idx[-1] == test_idx[0] - 1


def test_walk_forward_purge_removes_horizon_rows():
    """purge=h 时，训练集末尾应比边界早 h 行。"""
    folds = walk_forward_indices(200, n_splits=4, min_train=0.4, purge=HORIZON)
    for train_idx, test_idx in folds:
        assert train_idx[-1] == test_idx[0] - 1 - HORIZON


def test_no_train_label_window_overlaps_test():
    """核心断言：训练集任何一行的标签窗口都不得进入测试区间。"""
    folds = walk_forward_indices(300, n_splits=5, min_train=0.4, purge=HORIZON)
    for train_idx, test_idx in folds:
        test_start = test_idx[0]
        # 训练最后一行的标签用到 t+1..t+HORIZON
        assert train_idx[-1] + HORIZON < test_start


def test_verify_no_overlap_detects_bad_purge():
    """purge 不足时，自检函数必须报错，而不是静默放过。"""
    bad = walk_forward_indices(200, n_splits=4, min_train=0.4, purge=0)
    with pytest.raises(AssertionError):
        verify_no_overlap(200, 0, HORIZON, bad)


def test_verify_no_overlap_accepts_correct_purge():
    good = walk_forward_indices(200, n_splits=4, min_train=0.4, purge=HORIZON)
    verify_no_overlap(200, HORIZON, HORIZON, good)  # 不应抛异常


def test_folds_are_contiguous_and_ordered():
    folds = walk_forward_indices(250, n_splits=5, min_train=0.4, purge=HORIZON)
    prev_end = 0
    for train_idx, test_idx in folds:
        assert np.array_equal(train_idx, np.arange(0, len(train_idx)))
        assert test_idx[0] == len(train_idx) + HORIZON or test_idx[0] > len(train_idx)
        assert test_idx[0] >= prev_end
        prev_end = test_idx[-1] + 1


def test_too_small_sample_raises():
    with pytest.raises(ValueError):
        walk_forward_indices(10, n_splits=5, min_train=0.4, purge=HORIZON)


def test_common_index_uses_all_features():
    """统一样本：只要任一特征缺失，该行就不进入共同索引。"""
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    data = pd.DataFrame({
        "target": np.arange(10.0),
        "a": np.arange(10.0),
        "b": np.arange(10.0),
        "c": np.arange(10.0),
    }, index=idx)
    data.loc[idx[3], "b"] = np.nan
    data.loc[idx[7], "c"] = np.nan

    only_ab = common_index(data, ["a", "b"])
    all_three = common_index(data, ["a", "b", "c"])

    assert len(only_ab) == 9          # 只排除 b 缺失那行
    assert len(all_three) == 8        # 两行都排除
    assert idx[3] not in all_three
    assert idx[7] not in all_three


def test_assert_same_samples_detects_mismatch():
    a = {"model1": {"n": 3, "index": pd.Index([1, 2, 3])},
         "model2": {"n": 3, "index": pd.Index([1, 2, 3])}}
    assert_same_samples(a)  # 一致，不报错

    b = {"model1": {"n": 3, "index": pd.Index([1, 2, 3])},
         "model2": {"n": 2, "index": pd.Index([1, 2])}}
    with pytest.raises(AssertionError):
        assert_same_samples(b)
