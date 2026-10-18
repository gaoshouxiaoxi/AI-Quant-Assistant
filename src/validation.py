"""验证工具：带 purging 的扩展窗口切分，以及统一样本的对齐。

为什么需要 purging
------------------
若标签是"未来 h 天的均值"（例如未来 5 日 RV），索引 t 的标签用到了 t+1..t+h。
按 b 切开训练/测试后，训练集最后 h 行的标签窗口会**伸进测试区间**：

    train: [0, b)
    test : [b, ...)
    训练行 t = b-1 的标签用到 b..b+h-1 —— 落在测试期内

这不是特征读取未来，而是标签窗口与测试期重叠，会让样本外验证不再独立。
修法：训练集末尾剔除 h 行（purging），即训练区间改为 [0, b-h)。

为什么需要统一样本
------------------
若每个模型各自 dropna，特征多的模型会丢掉更多行，导致不同模型在不同日期集合上
评估，Skill 差异里混进了样本差异。正确做法是先对"所有模型特征并集"求有效行，
再让所有模型、所有基线都用同一批索引。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_indices(n: int, n_splits: int = 5, min_train: float = 0.4,
                         purge: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """扩展窗口切分，带 purging。

    Parameters
    ----------
    n : 样本数
    n_splits : 折数
    min_train : 初始训练占比
    purge : 训练集末尾要剔除的行数，应等于标签的前视窗口长度

    Returns
    -------
    [(train_idx, test_idx), ...]
    """
    if n_splits <= 0:
        raise ValueError("n_splits 必须为正")
    start = max(int(n * min_train), 1)
    remaining = n - start
    if remaining < n_splits:
        raise ValueError(f"样本太少：n={n}，无法切成 {n_splits} 折")

    step = remaining // n_splits
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for k in range(n_splits):
        boundary = start + k * step
        end = n if k == n_splits - 1 else boundary + step
        train_end = boundary - purge
        if train_end < 10:          # 训练样本过少则跳过该折
            continue
        folds.append((np.arange(0, train_end), np.arange(boundary, end)))

    # 关键：一折都生不出来时必须报错。
    # 早期版本只是 continue，函数会静默返回空列表，调用方拿到空结果却以为"跑完了"。
    if not folds:
        raise ValueError(
            f"没有任何有效折：n={n}, n_splits={n_splits}, "
            f"min_train={min_train}, purge={purge}。"
            f"purge 过大或样本过少都会导致此错误")
    return folds


def verify_no_overlap(n: int, purge: int, horizon: int,
                      folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """自检：训练集里任何一行的标签窗口都不得触及测试区间。

    标签窗口是 [t+1, t+horizon]，测试起点是 test[0]。
    条件：t + horizon < test_start
    """
    for train_idx, test_idx in folds:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        test_start = int(test_idx[0])
        worst = int(train_idx[-1]) + horizon
        if worst >= test_start:
            raise AssertionError(
                f"purge={purge} 不足：训练最后一行的标签窗口到 {worst}，"
                f"测试起点 {test_start}，存在重叠")


def common_index(data: pd.DataFrame, columns: list[str],
                 target: str = "target") -> pd.Index:
    """所有特征并集都不缺失的索引，用作统一样本。"""
    needed = list(dict.fromkeys([target] + columns))
    return data.dropna(subset=needed).index


def assert_same_samples(results: dict[str, dict]) -> None:
    """确认各模型评估样本完全一致，否则对比无效。"""
    sets = {k: set(v.get("index", [])) for k, v in results.items() if v.get("n")}
    if len(sets) <= 1:
        return
    base_key = next(iter(sets))
    base = sets[base_key]
    for key, s in sets.items():
        if s != base:
            raise AssertionError(
                f"样本不一致：{base_key} 有 {len(base)} 行，"
                f"{key} 有 {len(s)} 行，差异 {len(base ^ s)} 行")
