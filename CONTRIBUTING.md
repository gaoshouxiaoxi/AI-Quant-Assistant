# 贡献指南

感谢你愿意改进这个项目。这个仓库的目标不只是"能跑"，而是**每个结论都有可复现的证据**。

## 本地开发

```bash
# 1. 环境
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements-dev.txt

# 2. 跑测试（不需要联网）
python -m pytest tests -v

# 3. 启动界面
streamlit run app.py
```

## 提交前请自检

1. **测试必须全绿**：`python -m pytest tests`
2. **语法检查**：`python -m compileall -q src app.py`
3. **不要提交数据**：`data/*.csv` 已被忽略（分钟线数据是例外，体量小且用于复现）

## 代码约定

- **不使用随机切分**：涉及时间序列的所有评估必须用扩展窗口前向验证
  （`walk_forward_splits` / `walk_forward`），训练集永远是测试集的过去。
- **标准化放在交叉验证内部**：先全量标准化再切分会泄漏统计量。
- **每个结论都要有基线**：持续性、多数派、随机，三者至少给一个。
- **报告置信区间**：稀疏事件（如尖峰）的点估计不可靠，必须给 bootstrap 区间。
- **数据先体检**：新增数据源必须通过 `data_loader._validate()`（拦截拆股造成的假暴跌）。

## 常见坑（已踩过，别再踩）

| 现象 | 原因 | 修法 |
| --- | --- | --- |
| 准确率 99%+ | 特征里混入了未来信息 | 检查特征是否用到 `shift(-1)` 之后的数据 |
| 策略净值异常低 | `ffill` 把"无信号日"写成 0 | 用 `NaN` 表示无信号，空仓仍用 0 |
| 波动率 R² 为负 | 目标用了有偏估计量 | 用 Yang-Zhang / Parkinson 而非收盘价法 |
| 中文图表显示方框 | 未设置字体 | 调用 `charts.setup_chinese_font()` |
| 脚本退出码 0 但无输出 | GBK 下 print emoji 抛异常 | `sys.stdout.reconfigure(encoding="utf-8")` |

## 提 PR

- 一个 PR 只做一件事，描述里写清**动机**和**验证方式**
- 涉及数值结论的改动，请附上改动前后的对比数字
- 新增研究脚本请放到 `src/`，结果表放到 `reports/`，图表放到 `charts/`

## 免责声明

本项目仅用于研究与教学。任何基于本仓库的分析结果都不构成投资建议。
