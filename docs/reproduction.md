# 复现指南（Reproduction）

> 目标：**任何人从干净环境，只用仓库内已提交的文件，就能重算出本项目的核心结果。**
> 主实验不需要联网、不需要下载行情。

---

## 1. 三十秒版

```bash
git clone git@github.com:gaoshouxiaoxi/AI-Quant-Assistant.git
cd AI-Quant-Assistant
python -m venv venv
venv\Scripts\activate            # Windows；macOS/Linux 用 source venv/bin/activate
pip install -r requirements-dev.txt

python run_experiment.py         # 重算主实验，约 4 秒
python run_experiment.py --check # 与已提交结果逐项比对，应输出"复现成功"
```

预期最后一行：

```
全部一致（容差 1e-06）—— 复现成功
```

---

## 2. 环境

| 项 | 要求 |
| --- | --- |
| Python | 3.11（生成结果所用版本：3.11.4） |
| 操作系统 | 任意；结果不含平台相关随机性 |
| 网络 | **不需要**（主实验只读已入库 CSV） |
| 内存 | < 1 GB |

两条安装路径：

| 文件 | 用途 |
| --- | --- |
| `requirements-dev.txt` | 跑实验 + 测试所需的最小集合 |
| `requirements-lock.txt` | `pip freeze` 全量锁定（82 个包），用于逐位复现 |

**版本敏感性**：线性模型（HAR-3 / HAR-X）的结果跨版本稳定；`RandomForest`
在 sklearn 小版本间可能有 1e-3 量级差异，`--check` 默认容差 1e-6 只对线性模型
严格，若 RF 报不一致可放宽 `--tol 1e-2`。

---

## 3. 数据从哪来

主实验只依赖**已提交到仓库**的 60 分钟 bar：

| 文件 | 行数 | 覆盖区间 |
| --- | --- | --- |
| `data/minute/sh600519_60m.csv` | 1970 | 2024-09-30 ~ 2026-10-16 |
| `data/minute/sz000858_60m.csv` | 1970 | 同上 |
| `data/minute/sz000001_60m.csv` | 1970 | 同上 |

来源：新浪财经分钟线接口（`quotes.sina.cn`，`datalen` 上限 1970 根）。
`datalen` 是硬上限，所以 60 分钟周期只能拿到约 492 个交易日——这是 60 分钟
而非更细周期的原因（15 分钟只能覆盖 122 天）。

`configs/research_main.json` 里 `network_required: false`，
并有测试 `test_runs_with_network_disabled` 在禁用 socket 后仍要求跑通。

**未入库**的是日线 CSV（`data/*_2020-01-01_2025-01-01.csv`），它们只用于
「延长 RV 目标」的**探索性**实验。该实验依赖用 2024-2026 高频数据校准的代理，
本身带有前视问题，已明确排除在可复现核心之外。

---

## 4. 实验协议（`configs/research_main.json`）

协议外置，代码里没有魔法数字。关键字段：

| 字段 | 值 | 说明 |
| --- | --- | --- |
| `seed` | 42 | 随机森林的 `random_state` |
| `data.rv_definition` | `Σr² + overnight²` | 日度**方差**，未年化 |
| `target.horizon_days` | 5 | 标签 = 未来 5 日均值 RV 的对数 |
| `split.scheme` | expanding walk-forward | 扩展窗口前向验证 |
| `split.n_splits` | 5 | 折数 |
| `split.min_train_frac` | 0.4 | 初始训练占比 |
| `split.purge_rows` | 5 | **必须等于 `horizon_days`**，有测试守护 |
| `missing.rule` | drop NaN rows | 缺失值处理规则 |
| `missing.sample_alignment` | common index | 模型与基线共用同一批日期 |

**三条硬约束（由测试保证）**：

1. `purge_rows == horizon_days`
2. 基线与模型的 `n` 必须相等
3. 同一输入两次运行结果必须逐位一致

---

## 5. 预期结果

`reports/research/results_main.csv`，严格样本外（仅真实 RV）：

| 标的 | HAR-3 Skill | HAR-3 Spearman | 基线 Spearman | 离散度比 |
| --- | --- | --- | --- | --- |
| 贵州茅台 | +25.76% | 0.448 | 0.472 | 0.36 |
| 五粮液 | +30.51% | 0.297 | 0.368 | 0.39 |
| 平安银行 | +29.09% | 0.225 | 0.321 | 0.29 |
| **平均** | **+28.45%** | 0.324 | 0.387 | 0.35 |

跨模型平均：

| 模型 | Skill | Spearman |
| --- | --- | --- |
| HAR-3 | **+28.45%** | 0.324 |
| HAR-3+kurt22 | +24.17% | 0.261 |
| HAR-X（多因子） | +20.80% | 0.217 |
| RandomForest | +16.91% | 0.139 |
| 持续性（周均 RV） | 0.00%（基准） | 0.387 |
| 持续性（当日 RV） | −108.01% | 0.296 |

**读这张表时请注意**：HAR-3 的 Skill 最高，但 Spearman 低于基线。
两者衡量不同能力（幅度 vs 排序），详见 [README 的诊断章节](../README.md#skill-与-spearman-的矛盾诊断)。

---

## 6. 校验机制

`--check` 不覆盖参考结果，而是：

1. 重算主实验
2. 与已提交的 `results_main.csv` 逐行逐列比对（默认容差 `1e-6`）
3. 把重算结果写到 `results_main.recomputed.csv` 供人工对比
4. 有不一致则打印具体字段与数值，并以退出码 1 结束

```bash
python run_experiment.py --check          # 默认容差
python run_experiment.py --check --tol 1e-2   # 放宽（用于跨版本 RF 差异）
```

元数据写在 `reports/research/manifest.json`：

- git commit / branch / 是否有未提交改动
- Python 版本、平台
- pandas / numpy / sklearn / scipy / matplotlib 版本
- 每个数据文件的 **sha256**、行数、起止日期
- 配置文件的 sha256
- 随机种子、运行耗时

> 注意：manifest 在提交前生成，所以其 `git.dirty` 常为 `true`。
> 这是"记录生成时的状态"，不是缺陷；需要干净记录时，在提交后重跑一次即可。

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `缺少数据文件 data/minute/...` | 克隆不完整或用了浅克隆 | `git checkout` 完整仓库 |
| `--check` 报 RF 的 r2/skill 不一致 | sklearn 版本差异 | `--tol 1e-2`；或在 `requirements-lock.txt` 环境里跑 |
| 中文图表显示方框 | 未安装中文字体 | 不影响数值；`charts.setup_chinese_font()` 会回退 |
| `ValueError: 没有任何有效折` | 样本太少或 purge 过大 | 检查 `purge_rows` 与数据行数 |
| 退出码 1 但无输出 | Windows GBK 下 print 特殊字符 | 已在脚本顶部强制 UTF-8；若仍出现请提 issue |

---

## 8. 代码地图（复现相关）

| 文件 | 作用 |
| --- | --- |
| `run_experiment.py` | 统一入口（CLI） |
| `configs/research_main.json` | 协议配置 |
| `src/research_pipeline.py` | 管线实现（读数据 → 特征 → 切分 → 评估 → manifest） |
| `src/har_features.py` | 特征工程（无网络依赖） |
| `src/validation.py` | purging 切分 + 统一样本 + 自检 |
| `src/daily_rv.py` | 60 分钟 bar → 日度 RV |
| `tests/test_reproducibility.py` | 9 个复现性测试 |
| `tests/test_validation.py` | 9 个 purging / 对齐测试 |

---

## 9. 复现范围声明

**能复现的**：上表全部数值、图表 `charts/research/protocol_summary.png`。

**不能复现的**：

- 重新抓取行情（新浪接口有 `datalen` 上限且可能变更；已入库 CSV 即为数据快照）
- 「延长 RV 目标」实验（依赖未入库的日线数据，且校准含前视，见 README）
- 方向预测（`ml_predict.py`）的原始结果——它需要 `data/*.csv`；
  该结论本身是负面的（AUC≈0.5），不影响核心结论
