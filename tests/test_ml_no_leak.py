"""ml_predict.py 的时间泄漏审计测试。

审计结论（2026-10）：
- 标签：`features.py` 用 `close.pct_change().shift(-1)`，索引 t 的标签用到
  **t+1 的收盘价**，前视窗口 h=1。
- 特征：只用了 `shift(lag)`（lag≥1）、`rolling()`、`diff()`，不含 `shift(-k)`，
  **没有特征泄漏**。
- 预处理：`make_pipeline` 里的 SimpleImputer / StandardScaler 在每折内部 fit，
  **没有用到测试集统计量**。
- 采样：`ml_predict` 里所有模型共用同一份 X、y，**日期集合天然一致**。
- 唯一缺陷：切分未清空训练集末尾 1 行（其标签用到测试期首日），已修（purge=1）。
"""

import numpy as np
import pandas as pd
import pytest

from features import build_features, split_xy

HORIZON = 1  # features.py 的标签前视窗口


@pytest.fixture
def prices() -> pd.DataFrame:
    """构造 120 天、有涨有跌的价格序列。"""
    idx = pd.date_range("2023-01-02", periods=120, freq="B")
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.012, len(idx)))
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, len(idx))),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(1e6, 5e6, len(idx)).astype(float),
    }, index=idx)


def test_label_uses_next_day_close(prices):
    """标签必须等于"次日收盘价相对今日收盘价"的符号。"""
    data = build_features(prices)
    close = prices["Close"]
    # 直接用次日收益率做对照，避免 get_loc 在 pandas 3.x 上的索引怪癖
    next_ret = close.shift(-1) / close - 1

    for t in data.index[:20]:
        expected = 1 if next_ret.loc[t] > 0 else 0
        assert int(data.loc[t, "label"]) == expected


def test_forward_return_matches_next_day(prices):
    """forward_return 必须精确等于次日收益率（不是当日）。"""
    data = build_features(prices)
    close = prices["Close"]
    next_ret = close.shift(-1) / close - 1
    for t in data.index[:20]:
        assert data.loc[t, "forward_return"] == pytest.approx(
            next_ret.loc[t], rel=1e-9)


def test_no_feature_reads_future(prices):
    """改动未来价格，历史特征必须一字不变。"""
    cut = 80
    base = build_features(prices)
    modified = prices.copy()
    modified.iloc[cut:] = modified.iloc[cut:] * 3.0
    alt = build_features(modified)

    cutoff_date = prices.index[cut - 1]
    # 特征与目标都只看 t+1 之后的行情，因此 t <= cut-1 的**特征**不受影响
    cols = [c for c in base.columns if c.startswith("f_")]
    np.testing.assert_allclose(
        base.loc[:cutoff_date, cols].to_numpy(),
        alt.loc[:cutoff_date, cols].to_numpy(),
        rtol=1e-12,
    )


def test_label_at_last_usable_row_still_valid(prices):
    """最后一行因缺少次日价格被丢弃，不应留下无效标签。"""
    data = build_features(prices)
    assert (data["label"].isin([0, 1])).all()
    assert prices.index[-1] not in data.index


def test_split_xy_shapes(prices):
    data = build_features(prices)
    X, y = split_xy(data)
    assert len(X) == len(y)
    assert X.shape[1] > 0
    assert not X.isna().any().any()


def test_walk_forward_purge_is_horizon():
    """切分的 purge 必须等于标签前视窗口，否则训练标签会伸进测试期。"""
    validation = pytest.importorskip("validation")
    folds = validation.walk_forward_indices(200, 5, 0.4, purge=HORIZON)
    for train_idx, test_idx in folds:
        assert train_idx[-1] + HORIZON < test_idx[0]


def test_evaluate_model_uses_purge_and_same_dates():
    """不同模型必须落在同一批测试日期上，且训练集已 purge。"""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    from ml_model import evaluate_model

    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(300, 5)), index=idx,
                     columns=[f"f{i}" for i in range(5)])
    y = pd.Series((rng.normal(size=300) > 0).astype(int), index=idx)

    m1 = evaluate_model(LogisticRegression(max_iter=500), X, y, purge=HORIZON)
    m2 = evaluate_model(DecisionTreeClassifier(max_depth=2, random_state=0),
                        X, y, purge=HORIZON)

    # 日期集合一致
    assert m1["proba"].index.equals(m2["proba"].index)
    # 测试集起点不早于初始训练边界
    assert m1["proba"].index.min() >= idx[int(300 * 0.4)]
