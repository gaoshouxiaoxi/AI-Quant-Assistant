"""第 5-6 周：机器学习预测（RandomForest / XGBoost）。

任务：用过去 20 个交易日的 OHLCV + 技术指标，预测**次日收盘价是否上涨**。

用法：
    python src/ml_predict.py                     # 默认 AAPL 与 600519
    python src/ml_predict.py AAPL,600519,TSLA

评估口径（重要，见 src/ml_model.py 注释）：
- 扩展窗口前向验证，训练集永远是测试集的过去，不做随机切分
- 标准化在每折内部完成，避免泄漏
- 同时给出"永远预测涨/跌"两个基线
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台 GBK 会让 emoji 等字符输出失败，导致"退出码 0 但零输出"
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from charts import setup_chinese_font  # noqa: E402
from data_loader import load_daily  # noqa: E402
from features import build_features, split_xy  # noqa: E402
from ml_charts import plot_feature_importance, plot_prediction_curve  # noqa: E402
from ml_model import build_models, evaluate_model, format_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHART_DIR = ROOT / "charts"
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

# 标签前视窗口：features.py 用 shift(-1)，即次日收盘价，h=1
LABEL_HORIZON = 1

DEFAULT_TICKERS = ["AAPL", "600519"]
DEFAULT_START = "2022-06-15"
DEFAULT_END = "2024-12-31"

LABELS = {
    "AAPL": "Apple (AAPL)",
    "600519": "贵州茅台 (600519)",
    "TSLA": "Tesla (TSLA)",
    "MSFT": "Microsoft (MSFT)",
    "NVDA": "NVIDIA (NVDA)",
}


def main() -> None:
    argv = [a for a in sys.argv[1:] if a]
    # 支持两种写法：
    #   ml_predict.py AAPL,600519            → 逗号分隔的多个标的
    #   ml_predict.py AAPL 600519 2022-06-15 2024-12-31
    dates = [a for a in argv if len(a) == 10 and a[4] == "-"]
    tickers = [a for a in argv if a not in dates]
    # 若最后两个是日期，它们就是区间
    if len(dates) >= 2:
        start, end = dates[-2], dates[-1]
        tickers = [a for a in tickers if a not in (start, end)]
    elif len(dates) == 1:
        start, end = dates[0], DEFAULT_END
        tickers = [a for a in tickers if a != start]
    else:
        start, end = DEFAULT_START, DEFAULT_END

    flat: list[str] = []
    for t in tickers:
        flat.extend([x for x in t.split(",") if x])
    tickers = flat or DEFAULT_TICKERS

    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    font = setup_chinese_font()
    print(f"中文字体: {font or '未找到'}")
    print(f"标的    : {', '.join(tickers)}")
    print(f"区间    : {start} ~ {end}")
    print("任务    : 预测次日涨跌（1=涨，0=跌）")
    print("验证    : 扩展窗口前向验证（训练集永远在测试集之前）\n")

    models = build_models()
    print(f"模型    : {', '.join(models)}")
    print("=" * 76)

    summary: list[dict] = []
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        label = LABELS.get(ticker, ticker)
        print(f"\n----- {label} -----")

        stock, source = load_daily(ticker, start, end, cache_dir=DATA_DIR)
        data = build_features(stock)
        X, y = split_xy(data)
        print(f"数据源 {source}   样本 {len(X)} 行   特征 {X.shape[1]} 个")
        print(f"次日上涨占比 {y.mean():.2%}")

        header = f"{'模型':<14}{'准确率':>9}{'AUC':>9}{'精确率':>9}{'召回率':>9}{'F1':>9}"
        print(header)
        print("-" * 76)

        for name in models:
            m = evaluate_model(models[name], X, y, purge=LABEL_HORIZON)
            print(format_metrics(name, m))
            summary.append({"ticker": ticker, "label": label, "model": name, **m})

            plot_prediction_curve(
                m["proba"], m["y_true"],
                f"{label} · {name} 样本外预测概率",
                CHART_DIR / f"{ticker}_{name}_proba.png",
            )

            if name == "RandomForest":
                clf = models[name].fit(X, y)  # 仅用于画特征重要性（全量，不参与评估）
                if hasattr(clf, "feature_importances_"):
                    plot_feature_importance(
                        X.columns, clf.feature_importances_,
                        f"{label} · 特征重要性",
                        CHART_DIR / f"{ticker}_feature_importance.png",
                    )

        # 基线：必须在**本标的**的样本外标签上计算，不能用上一个标的的
        y_true = m["y_true"]
        up = float(y_true.mean())
        majority = max(up, 1 - up)
        side = "涨" if up >= 0.5 else "跌"
        print(f"{'基线:' + side + '占比多数':<14}{majority:>8.2%}       -"
              f"       -       -       -")
        print(f"（样本外上涨占比 {up:.2%}；恒预测多数派即为 {majority:.2%}，"
              f"准确率必须显著高于它才有意义）")

    # ---- 汇总 ----
    print("\n" + "=" * 76)
    print("汇总：各模型相对基线的增益")
    print("=" * 76)
    print(f"{'标的':<18}{'模型':<14}{'准确率':>9}{'AUC':>8}{'超基线':>10}")
    print("-" * 76)
    for row in summary:
        print(f"{row['label']:<18}{row['model']:<14}{row['accuracy']:>8.2%}"
              f"{row['auc']:>8.3f}{row['edge_over_baseline']:>+10.2%}")

    report = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("proba", "y_true", "folds")}
                           for r in summary])
    out = REPORT_DIR / "ml_results.csv"
    report.to_csv(out, index=False)
    print(f"\n明细已保存到 {out.relative_to(ROOT)}")

    print("\n判读建议：")
    print("  - AUC 0.5 左右 = 无预测能力；0.55~0.60 已经算有微弱信号")
    print("  - 准确率低于基线时，模型其实是偏向'预测跌'的多数派，要看 AUC 而非准确率")
    print("  - 本结果基于单一区间、未做超参搜索，不能直接当投资依据")


if __name__ == "__main__":
    main()
