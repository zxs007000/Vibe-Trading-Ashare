# OOS 框架（Vibe-Trading A股量化因子 · 严格样本外验证）

> 这是「因子有寿命，什么状态用什么因子」这条主线的工程落地：**选股层冻结 IS 胜者 + 配置层 regime 总闸**。
> 本目录是一套**自包含、可本地运行**的严格样本外（Out-of-Sample）因子验证 + 生产选股框架。

---

## 1. 目录结构

```
OOS_Framework/
├── README.md                      # 本文件
├── config 通过环境变量 STOCK_WORM_DATA 注入（见 §4）
│
├── oos_validation.py              # 【核心引擎】回测引擎 + build_signal + 四策略头对头(A/B/Frozen/Random)
├── oos_validation_corrected.py    # 修正版：去生存者偏差面板 + 行业中性化（M.load_wide_sf / build_zarr）
├── factor_zoo_daily.py            # 30 个技术因子（build_factors）+ 行业中性化（neutralize_factors）
├── factor_zoo_ortho.py            # 5 个状态正交防御因子（build_ortho_factors）
├── oos_engine_prod.py             # 生产引擎：Frozen + 质量层，输出实际持仓 CSV + 最新买入清单
├── factor_state_review.py         # 因子状态复盘 + 合流系统(B 选股 + C ETF轮动)回测 + regime 扫描
├── factor_window_check.py         # 2020–2023 跨周期因子家族检验（含新冠熔断 + 2022 熊）
├── etf_rotation_ext.py            # 方向 C ETF 轮动（沪深300 vs MA120 regime），被复盘脚本调用
├── build_fundamental_factors.py   # 3 个质量因子构造（ROE / rev_yoy / profit_yoy，财报→日线→中性化）
│
└── backtest/
    ├── __init__.py
    └── validation.py              # 最小化的 _sharpe（避免引入原项目 backtest.models 重依赖）
```

> 说明：所有脚本从本目录直接 `python xxx.py` 运行即可（脚本已把自身目录加入 `sys.path`，`import oos_validation`、
> `from factor_zoo_daily import ...`、`from backtest.validation import _sharpe` 都能解析）。

---

## 2. 各文件作用一览

| 文件 | 入口/函数 | 干什么 |
|---|---|---|
| `oos_validation.py` | `main()` | 严格 OOS 验证（分界 2024-09-01）。含 `long_only_topk` / `random_topk` / `_stat_block` / `build_signal`。对比 A(无选择)、B(自适应IC闸门)、Frozen(IS锁定因子集永远开)、Random(安慰剂)、等权基准、随机选股。 |
| `oos_validation_corrected.py` | `load_wide_sf()` / `build_zarr()` / `run_pipeline()` | 用**去生存者偏差**面板 + **行业中性化**，对原始因子 vs 中性化因子头对头，验证结论是否稳健。 |
| `factor_zoo_daily.py` | `build_factors()` / `neutralize_factors()` / `daily_rank_ic()` | 30 个技术因子的标准公式实现 + 证监会行业中性化（逐日减行业均值）。 |
| `factor_zoo_ortho.py` | `build_ortho_factors()` | 5 个状态正交因子（beta_60 / lowvol_60 / lowivol_60 / liq_stress_20 / distress_60）。 |
| `oos_engine_prod.py` | `main()` | 生产引擎：Frozen(30技术 IS锁定) + 质量层(ROE/rev_yoy/profit_yoy)，输出每调仓日 top-K 持仓 CSV 与最新买入清单。 |
| `factor_state_review.py` | `main()` | 38 因子状态总览（IS/OOS/牛/熊 IC）+ 合流系统(B+C)回测 + regime 信号尺度扫描。 |
| `factor_window_check.py` | `main()` | 钉死 2020–2023 窗口，用 pre-2020 冻结因子做严格 OOS，看因子家族成色（跨牛熊）。 |
| `etf_rotation_ext.py` | `backtest()` | 方向 C 的 ETF 轮动（CORE20Y 宇宙 + 动量 L20 top1 + 沪深300 vs MA120 regime + 月度再平衡）。 |
| `build_fundamental_factors.py` | `build_daily()` | 从财报构造 3 个质量因子日线序列（前向填充→规模桶/行业 demean→z-score）。 |

---

## 3. 依赖

```bash
pip install pandas numpy matplotlib
# 可选（仅当你要重爬质量因子 / 跑 ETF 轮动实时行情时用到）:
pip install akshare
```

Python ≥ 3.10（用到 `from __future__ import annotations`）。

---

## 4. 数据准备（关键）

框架需要 `stock_worm` 生成的几个 parquet（**不含在本压缩包内**，数据较大，约 240MB，请用自己的 stock_worm 数据）：

| 文件 | 说明 |
|---|---|
| `ashare_daily_panel.parquet` | 当前存活快照日线面板（open/high/low/close/volume/amount） |
| `ashare_daily_panel_survivorfree.parquet` | **去生存者偏差**面板（1489 alive + 358 delisted），复盘/修正版/2020检验都用它 |
| `csrc_industry_map.parquet` | 证监会行业映射（code → csrc_industry），用于行业中性化 |
| `fundamentals/fund_factors_daily.parquet` | 3 个质量因子日线序列（ROE / rev_yoy / profit_yoy），由 `build_fundamental_factors.py` 生成 |
| `etf_rotation_ext_cache.parquet` | ETF 轮动缓存（含国债ETF日线），仅 `factor_state_review.py` 用到 |

**设置数据根目录**（所有路径都从环境变量读取，默认回退到 `/workspace/stock_worm/data`）：

```bash
export STOCK_WORM_DATA=/your/local/path/to/stock_worm/data
```

例如你的数据在 `~/stock_worm/data`，就 `export STOCK_WORM_DATA=$HOME/stock_worm/data`。
没有 stock_worm 数据时，可用 akshare/tushare 自行构造等价的 parquet（列名见 `factor_zoo_daily.load_wide`）。

---

## 5. 关键参数（在 `oos_validation.py` 顶部，全框架共用）

| 参数 | 默认 | 含义 |
|---|---|---|
| `SPLIT` | 2024-09-01 | 严格 OOS 分界（「924 政策行情」regime 切换） |
| `TOP_K` | 0.30 | 每期取截面前 30% |
| `HOLD` | 5 | 非重叠 5 交易日（约一周）再平衡 |
| `COST` | 0.001 | 单边千一成本 |
| `TRAIL` | 250 | 滚动 IC 统计窗口（≈1 年） |
| `RNG` | 42 | 随机种子 |

`factor_window_check.py` 另有两个窗口参数：`TEST_START=2020-01-01`、`TEST_END=2023-12-31`、`FREEZE_CUT=2020-01-01`。

---

## 6. 运行命令

```bash
cd OOS_Framework
export STOCK_WORM_DATA=/your/path/to/stock_worm/data

# (1) 严格 OOS 验证（A/B/Frozen/Random 头对头）
python oos_validation.py

# (2) 修正版（去生存偏差 + 行业中性化，看结论是否稳健）
python oos_validation_corrected.py

# (3) 生产引擎（Frozen + 质量层，输出持仓 CSV + 最新买入清单）
python oos_engine_prod.py

# (4) 因子状态复盘 + 合流系统回测 + regime 扫描
python factor_state_review.py

# (5) 2020–2023 跨周期因子家族检验（含熔断+2022熊）
python factor_window_check.py
```

每个脚本会在 `screen_results/` 下产出报告（.md）+ 图表（.png）；生产引擎额外产出 `OOS生产引擎_持仓.csv`。

---

## 7. 核心设计要点（看代码时抓住这几条）

1. **严格 OOS / 冻结（Frozen）**：因子集与权重在 IS（≤`SPLIT`）一次性锁定，`frozen_set = {f | IS_IC>0 且 IS_ICIR>0}`；OOS 零重学习，`build_signal(..., gate=False, weight_src="is")`。
2. **信号合成（`build_signal`）**：`composite = Σ orient(f)·w(f)·z(f) / Σ w(f)`。
   - `orient(f) = +1 if IS_mean_IC≥0 else -1`：**按 IS 期 IC 符号自动翻转方向**（IC 为负的因子自动反向取，低分加分）。
   - `w(f) = is_icir(f)`：权重 = IS 期 ICIR（信息比率），越稳权重越大。
3. **预处理**：横截面 z-score 标准化 + 证监会行业中性化（逐日减行业均值）；质量因子额外做规模 10 分桶 demean。
4. **配置层 regime 总闸（合流系统）**：沪深300 vs MA120 决定 risk-on 持股票 / risk-off 切国债。**关键约束：regime 信号尺度必须与 alpha 再平衡尺度（HOLD=5）匹配**——快信号（等权 MA20 / 波动率状态）才有效，慢信号（MA120/200）反而更差。详见 `factor_state_review.py` 的 §3b 扫描。

---

## 8. 注意事项

- **内存**：去生存者偏差面板 1803 只 × ~5000 交易日，因子矩阵较大；`oos_validation_corrected` / `factor_state_review` 已做分阶段释放。建议在 ≥8GB 内存环境运行。
- **生存者偏差**：`ashare_daily_panel.parquet`（快照）绝对数字虚高；复盘/修正/跨周期检验均用 `survivorfree` 版本。相对排序不受影响，绝对数字须在去偏差后面重算。
- **单一 OOS regime**：只有 2024-09 一个干净切点、OOS 约 1.75 年，属提示性非结论性；`factor_window_check.py` 用 2020–2023 另一窗口做交叉验证。
- **质量因子**：若没有 `fund_factors_daily.parquet`，先跑 `python build_fundamental_factors.py`（需 akshare 可达）生成；或直接用自己的质量因子日线序列替换该文件。

---

## 9. 与已生成报告的对应

完整的方法论、38 因子公式、逐因子数字与回测结果见 `因子家族与策略复盘_完整报告.pdf`（同工作目录下）。
本框架是该报告里所有结论的可复现代码。
