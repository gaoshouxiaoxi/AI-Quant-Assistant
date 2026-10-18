"""可复现研究管线：从入库的 60 分钟 bar 重新生成主实验报告。

设计原则
--------
1. **不联网**：只读仓库里 `data/minute/*_60m.csv`（已入库），干净克隆即可跑。
2. **协议外置**：所有可调项写在 `configs/research_main.json`，代码不含魔法数字。
3. **可验证**：`--check` 会把本次结果与已提交的 `results_main.csv` 对比，
   任何数值漂移都会报出来。
4. **可追溯**：`manifest.json` 记录 git 提交、依赖版本、数据文件哈希、配置哈希、
   随机种子与运行耗时。

依赖：pandas / numpy / scikit-learn（+ 项目内轻量模块 validation、har_features）。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from daily_rv import build_daily_rv  # noqa: E402
from har_features import ALL_COLS, build_features  # noqa: E402
from validation import common_index, walk_forward_indices  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 基础设施

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_info() -> dict:
    def run(args: list[str]) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["status", "--porcelain"])),
    }


def load_protocol(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text), sha256_text(text)


def read_bars(path: Path) -> pd.DataFrame:
    """读取入库的 60 分钟 bar，统一成 daily_rv.build_daily_rv 需要的格式。"""
    df = pd.read_csv(path, parse_dates=["day"], index_col="day")
    df = df.rename(columns={"open": "open", "high": "high", "low": "low",
                            "close": "close", "volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def daily_rv(bars: pd.DataFrame, min_bars: int = 2) -> pd.DataFrame:
    """日线 OHLC + 日度 RV（委托给 daily_rv.build_daily_rv，避免重复实现）。"""
    if min_bars != 2:
        raise ValueError("build_daily_rv 内置 min_bars=2，改协议需同步改实现")
    d = build_daily_rv(bars)
    d = d.rename(columns={"rv_total": "rv"})
    d["rv_is_true"] = 1
    return d[["Open", "High", "Low", "Close", "Volume", "rv", "rv_is_true"]]


def prepare_dataset(bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """bars → 特征 + 标签，并按协议做缺失值处理与样本对齐。"""
    daily = daily_rv(bars, cfg["data"]["min_bars_per_day"])
    feats = build_features(daily, forward=cfg["target"]["horizon_days"])
    idx = common_index(feats, ALL_COLS)
    return feats.loc[idx]


# ---------------------------------------------------------------- 评估

def _metrics(pred: pd.Series, true: pd.Series, base: pd.Series) -> dict:
    mse_m = float(np.mean((pred - true) ** 2))
    mse_b = float(np.mean((base - true) ** 2))
    slope = float(np.polyfit(pred.to_numpy(), true.to_numpy(), 1)[0])
    return {
        "n": int(len(true)),
        "r2": float(r2_score(true, pred)),
        "mae": float(mean_absolute_error(true, pred)),
        "skill": float(1 - mse_m / mse_b) if mse_b > 0 else float("nan"),
        "spearman": float(true.corr(pred, method="spearman")),
        "spearman_base": float(true.corr(base, method="spearman")),
        "disp_ratio": float(pred.std() / true.std()),
        "calib_slope": slope,
    }


def _make_estimator(spec: dict, cfg: dict):
    if spec["estimator"] == "ols":
        return Pipeline([("scaler", StandardScaler()),
                         ("model", LinearRegression())])
    if spec["estimator"] == "random_forest":
        params = dict(cfg["sklearn_params"]["random_forest"])
        params["random_state"] = cfg["seed"]
        return Pipeline([("scaler", StandardScaler()),
                         ("model", RandomForestRegressor(**params))])
    raise ValueError(f"未知 estimator：{spec['estimator']}")


def evaluate_model(dataset: pd.DataFrame, spec: dict, cfg: dict,
                   folds: list[tuple[np.ndarray, np.ndarray]],
                   test_index: pd.Index) -> dict:
    """单模型前向验证。样本已对齐，切分带 purge。

    注意：模型只在测试折上评估，所以 n 等于各折测试样本数之和。
    基线必须用**同一批 test_index**，否则两者 n 不同、对比无意义。
    """
    cols = spec["features"]
    X, y = dataset[cols], dataset["target"]

    preds, trues = [], []
    for tr, te in folds:
        est = _make_estimator(spec, cfg)
        est.fit(X.iloc[tr], y.iloc[tr])
        preds.append(pd.Series(est.predict(X.iloc[te]), index=X.iloc[te].index))
        trues.append(y.iloc[te])

    pred = pd.concat(preds).sort_index()
    true = pd.concat(trues).sort_index()
    if not pred.index.equals(test_index):
        raise AssertionError("模型评估日期与统一测试索引不一致")
    base = dataset.loc[true.index, "rv_w"]
    out = _metrics(pred, true, base)
    out.update({"model": spec["name"], "kind": "model"})
    return out


def evaluate_baseline(dataset: pd.DataFrame, spec: dict,
                      test_index: pd.Index) -> dict:
    """基线：直接用某列的当前值作为预测（无参数拟合）。

    只在 models 用过的同一批测试日期上评估，保证 n 与模型一致。
    """
    sub = dataset.loc[test_index]
    true = sub["target"]
    pred = sub[spec["feature"]]
    base = sub["rv_w"]
    out = _metrics(pred, true, base)
    out.update({"model": spec["name"], "kind": "baseline"})
    return out


def run_experiment(cfg: dict, cfg_path: Path) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    rows, data_manifest = [], []

    for item in cfg["universe"]:
        bars_path = ROOT / item["bars"]
        if not bars_path.exists():
            raise FileNotFoundError(
                f"缺少数据文件 {item['bars']}。主实验只依赖已入库的 60 分钟 bar，"
                f"请确认克隆完整。")
        bars = read_bars(bars_path)
        data_manifest.append({
            "ticker": item["ticker"],
            "path": item["bars"],
            "sha256": sha256_file(bars_path),
            "rows": int(len(bars)),
            "start": str(bars.index.min()),
            "end": str(bars.index.max()),
        })

        ds = prepare_dataset(bars, cfg)

        # 折与测试索引只算一次，模型与基线共用，保证 n 一致
        folds = walk_forward_indices(
            len(ds), cfg["split"]["n_splits"],
            cfg["split"]["min_train_frac"], cfg["split"]["purge_rows"])
        test_pos = np.concatenate([te for _, te in folds])
        test_index = ds.index[test_pos].sort_values()

        for spec in cfg["models"]:
            r = evaluate_model(ds, spec, cfg, folds, test_index)
            r.update({"ticker": item["ticker"], "name": item["name"]})
            rows.append(r)
        for spec in cfg["baselines"]:
            r = evaluate_baseline(ds, spec, test_index)
            r.update({"ticker": item["ticker"], "name": item["name"]})
            rows.append(r)

    results = pd.DataFrame(rows)
    order = ["ticker", "name", "kind", "model", *cfg["metrics"]]
    results = results[[c for c in order if c in results.columns]]

    manifest = {
        "protocol_version": cfg["protocol_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 2),
        "seed": cfg["seed"],
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "config": {"path": str(cfg_path.relative_to(ROOT)),
                   "sha256": sha256_text(cfg_path.read_text(encoding="utf-8"))},
        "data": data_manifest,
        "target": cfg["target"],
        "split": cfg["split"],
        "missing": cfg["missing"],
        "network_required": cfg["data"]["network_required"],
        "environment": {
            "numpy": np.__version__,
            "random_seed_applied": cfg["seed"],
        },
    }
    return results, manifest


def _package_versions() -> dict:
    import importlib
    names = ["pandas", "numpy", "sklearn", "scipy", "matplotlib"]
    out = {}
    for n in names:
        try:
            mod = importlib.import_module(n)
            out[n] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[n] = None
    return out


# ---------------------------------------------------------------- 校验

def compare_to_reference(results: pd.DataFrame, ref_path: Path,
                         tol: float = 1e-6) -> list[str]:
    """把本次结果与已提交结果逐项比对，返回不一致项（空列表 = 完全一致）。

    注意：CSV 读回的 ticker 会变成 int64（600519），而配置里是字符串，
    直接 merge 会抛 "merge on int64 and str"。这里统一转成字符串再比。
    """
    if not ref_path.exists():
        return [f"缺少参考结果 {ref_path}，无法校验"]
    ref = pd.read_csv(ref_path)
    key = ["ticker", "model"]
    for k in key:
        if k not in ref.columns or k not in results.columns:
            return [f"参考结果或本次结果缺少列 {k}"]
    ref = ref.copy()
    new = results.copy()
    for df in (ref, new):
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        df["model"] = df["model"].astype(str)

    merged = ref.merge(new, on=key, suffixes=("_ref", "_new"), how="outer",
                       indicator=True)
    issues: list[str] = []
    for _, r in merged.iterrows():
        if r["_merge"] != "both":
            issues.append(f"{r['ticker']}/{r['model']} 只出现在"
                          f"{'参考' if r['_merge'] == 'left_only' else '本次'}结果中")
            continue
        for col in ["n", "r2", "mae", "skill", "spearman", "disp_ratio",
                    "calib_slope"]:
            a, b = r.get(f"{col}_ref"), r.get(f"{col}_new")
            if a is None or b is None or (isinstance(a, float) and np.isnan(a)
                                          and isinstance(b, float) and np.isnan(b)):
                continue
            if abs(float(a) - float(b)) > tol:
                issues.append(f"{r['ticker']}/{r['model']} 的 {col}: "
                              f"参考 {a} vs 本次 {b}")
    return issues


def plot_summary(results: pd.DataFrame, out_dir: Path) -> list[Path]:
    """跨标的平均 Skill / Spearman 对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT / "src"))
    try:
        from charts import setup_chinese_font
        setup_chinese_font()
    except Exception:  # noqa: BLE001
        pass

    agg = (results.groupby("model")[["skill", "spearman"]].mean()
           .sort_values("skill", ascending=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "protocol_summary.png"

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(agg))
    ax.bar(x - 0.2, agg["skill"] * 100, width=0.4, color="#d32f2f",
           label="Skill（%）")
    ax.bar(x + 0.2, agg["spearman"], width=0.4, color="#1565c0",
           label="Spearman")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=15, ha="right")
    ax.axhline(0, color="gray", linewidth=0.9)
    ax.set_title("跨标的平均：Skill 与 Spearman（严格样本外）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------- 稳健性研究

STUDY_MODEL = "HAR-3"


def _oos_frame(dataset: pd.DataFrame, spec: dict, cfg: dict) -> pd.DataFrame:
    """取某模型的样本外预测，附上真实值、基线、年化波动。"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from rv_estimators import TRADING_DAYS

    folds = walk_forward_indices(
        len(dataset), cfg["split"]["n_splits"],
        cfg["split"]["min_train_frac"], cfg["split"]["purge_rows"])
    X, y = dataset[spec["features"]], dataset["target"]
    preds, trues = [], []
    for tr, te in folds:
        est = Pipeline([("s", StandardScaler()), ("m", LinearRegression())])
        est.fit(X.iloc[tr], y.iloc[tr])
        preds.append(pd.Series(est.predict(X.iloc[te]), index=X.iloc[te].index))
        trues.append(y.iloc[te])

    pred = pd.concat(preds).sort_index()
    true = pd.concat(trues).sort_index()
    df = pd.DataFrame({"pred": pred, "true": true,
                       "base": dataset.loc[true.index, "rv_w"]})
    df["actual_vol"] = np.sqrt(np.exp(df["true"]) * TRADING_DAYS)
    return df


def _cell_stats(df: pd.DataFrame) -> dict:
    if len(df) < 20:
        return {}
    mse_m = float(np.mean((df["pred"] - df["true"]) ** 2))
    mse_b = float(np.mean((df["base"] - df["true"]) ** 2))
    return {
        "n": int(len(df)),
        "skill": float(1 - mse_m / mse_b) if mse_b > 0 else float("nan"),
        "spearman": float(df["true"].corr(df["pred"], method="spearman")),
        "spearman_base": float(df["true"].corr(df["base"], method="spearman")),
        "disp_ratio": float(df["pred"].std() / df["true"].std()),
        "mean_vol": float(df["actual_vol"].mean()),
    }


def run_robustness(cfg: dict) -> pd.DataFrame:
    """把样本外区间切开，检查结论在不同子区间/波动档位上是否稳定。

    两种切法：
    - 时间子区间：把样本外窗口按时间等分成 2 段和 3 段
    - 波动档位：按真实波动三分位分组

    输出长表，每行是一个 (标的, 切法, 分组) 单元。
    """
    rows = []
    for item in cfg["universe"]:
        bars = read_bars(ROOT / item["bars"])
        ds = prepare_dataset(bars, cfg)
        spec = next(m for m in cfg["models"] if m["name"] == STUDY_MODEL)
        frame = _oos_frame(ds, spec, cfg)

        def add(scheme: str, group: str, sub: pd.DataFrame) -> None:
            st = _cell_stats(sub)
            if st:
                rows.append({"ticker": item["ticker"], "name": item["name"],
                             "scheme": scheme, "group": group, **st})

        add("overall", "全区间", frame)

        for k in (2, 3):
            edges = np.linspace(0, len(frame), k + 1).astype(int)
            for i in range(k):
                sub = frame.iloc[edges[i]:edges[i + 1]]
                label = (f"{sub.index.min().date()} ~ {sub.index.max().date()}"
                         if len(sub) else f"第{i+1}/{k}段")
                add(f"time_{k}", label, sub)

        terciles = pd.qcut(frame["actual_vol"], 3,
                           labels=["低波动", "中波动", "高波动"])
        for name, sub in frame.groupby(terciles, observed=True):
            add("regime", str(name), sub)

    return pd.DataFrame(rows)


def summarize_robustness(study: pd.DataFrame) -> pd.DataFrame:
    """把单元结果汇总成"某结论在多少个单元里成立"。"""
    out = []
    cells = study[study["scheme"] != "overall"]
    for scheme, g in cells.groupby("scheme"):
        out.append({
            "切法": {"time_2": "时间二分", "time_3": "时间三分",
                     "regime": "波动三档"}.get(scheme, scheme),
            "单元数": len(g),
            "Skill>0": int((g["skill"] > 0).sum()),
            "Spearman>基线": int((g["spearman"] > g["spearman_base"]).sum()),
            "平均 Skill": float(g["skill"].mean()),
            "平均 Spearman": float(g["spearman"].mean()),
            "平均离散度比": float(g["disp_ratio"].mean()),
        })
    return pd.DataFrame(out)


def plot_study(study: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from charts import setup_chinese_font
        setup_chinese_font()
    except Exception:  # noqa: BLE001
        pass

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    reg = study[study["scheme"] == "regime"]
    if len(reg):
        order = ["低波动", "中波动", "高波动"]
        agg = reg.groupby("group")[["skill", "spearman", "disp_ratio"]].mean()
        agg = agg.reindex([o for o in order if o in agg.index])
        p = out_dir / "study_by_regime.png"
        fig, ax = plt.subplots(figsize=(9, 4.4))
        x = np.arange(len(agg))
        ax.bar(x - 0.2, agg["skill"] * 100, width=0.4, color="#d32f2f",
               label="Skill（%）")
        ax.bar(x + 0.2, agg["spearman"], width=0.4, color="#1565c0",
               label="Spearman")
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index)
        ax.axhline(0, color="gray", linewidth=0.9)
        ax.set_title("按波动档位：Skill 与 Spearman（跨标的平均）")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    t = study[study["scheme"].isin(["time_2", "time_3"])]
    if len(t):
        p = out_dir / "study_by_time.png"
        fig, ax = plt.subplots(figsize=(11, 4.6))
        labels = [f"{r['name']}\n{r['group']}" for _, r in t.iterrows()]
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["skill"] * 100, width=0.4, color="#d32f2f",
               label="Skill（%）")
        ax.bar(x + 0.2, t["spearman"], width=0.4, color="#1565c0",
               label="Spearman")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="gray", linewidth=0.9)
        ax.set_title("按时间子区间：Skill 与 Spearman（每根柱子一个子区间）")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths
