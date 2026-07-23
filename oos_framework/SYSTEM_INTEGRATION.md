# Vibe-Trading-Ashare · 量化选股系统端到端整合文档

> 范围：从数据湖构建 → 数据湖组成 → 因子挖掘 → 因子产出 → 因子训练(WFA) → 回测 → 因子终验(SHAP) → 防御门控，一整套管线。
> 代码根：`D:\Vibe-Trading-Ashare\oos_framework\factor_mining\`
> 数据/宏观根：`stockworm`（`D:\stcok-worm\`，经 `STOCKLAKE` 环境变量注入 `factor_mining/base_data.py`）
> 文档状态：截至 2026-07-23 全部阶段均已跑通并有实证结果（含 SHAP 因子终验 P6）。

---

## 0. 总览：七阶段流水线

```
[1 数据湖构建]            [2 数据湖组成]           [3 因子挖掘]            [4 因子产出]
 stockworm / akshare  →   daily/ turnover/    →   mine_v2.py          →  factors_v2_3dim.json
 构建 OHLCV/换手率/        chip_panels/          四方向挖矿               (118 候选)
 筹码/宏观/流通股本         macro/ universe       (grid+GP+MCTS+XGB)         │
                           过滤(5596→5175)                                 ▼
                                                                   factor_screen.py
                                                                   (P3 衰减感知筛选)
                                                                            │
                                                       factors_v2_selected.json (40 入选)
                                                                            │
[5 因子训练 WFA]   [6 回测]   [6.5 因子终验 SHAP]   [7 防御门控(完整双尾)]
 build_feature_table_chunked  backtest(top30%)  shap_validate_v2.py  gate_full_v4.py + defensive_gating.py
   → run_wfa_chunked (4折) → 年化+16.1%   → TreeExplainer精确SHAP ① 上证右侧危机确认 → CRISIS_POS=0.6
   → oos_detail_v2.parquet    回撤-37.0%   → 方向验证+chip 6维 ② 巴指左侧双尾预警(缓冲≤20%)
                                       │    (P6 收尾体检)     ③ 因子衰减预警(降仓≤20%)
                                       │                      ④ 敏感耦合(阈值左移)
                                       ▼                     + P5c 持仓结构倾斜(TILT_CAP=0.7)
                               裸多头 vs 双尾+倾斜             → 回撤-37%→-26.5% 年化不掉
```

---

## 1. 数据湖构建（Data Lake Construction）

数据湖由 **stockworm** 多源采集 + `factor_mining` 派生面板两层构成。根路径 `LAKE = os.environ.get("STOCKLAKE", "/workspace/stocklake")`（`base_data.py`）。

### 1.1 基础层（stockworm 采集，akshare/腾讯/东财）
| 面板 | 路径 | 来源 | 说明 |
|---|---|---|---|
| 日线 OHLCV | `LAKE/daily/<code>.parquet` | akshare `stock_zh_a_hist` / tencent | `open,high,low,close,volume,amount`，date×stock 面板 |
| 准确换手率 | `LAKE/turnover/<code>.parquet` | akshare（交易所披露真实换手率） | **B1 层**：对限售股解禁免疫（关键，见 §2.4） |
| 流通股本快照 | `LAKE/float_shares_panel.parquet` | stockworm `company_info` | 横截面规模特征；**非**筹码成本分布依赖项 |

### 1.2 派生层（factor_mining 计算，落盘）
| 面板 | 构建脚本 | 说明 |
|---|---|---|
| 筹码 6 维 | `build_chip_lake.py` → `LAKE/chip_panels/<field>.parquet` | 逐股衰减成本分布模型（§2.4） |
| 防御面板 | `build_defense_panel.py` → `factor_mining/defense_panel.parquet` | 价量波动 + chip 防御度（P5c 用） |
| 股票池元数据 | `universe.py` → `LAKE/metadata/universe_stats.json` + `stock_names.parquet` | ST/次新/低流动性过滤名单 |

### 1.3 宏观层（防御门控右翼，stockworm 侧）
| 信号 | 路径 | 构建 | 说明 |
|---|---|---|---|
| 巴菲特指标 | `stockworm/data/macro/buffett_ratio.parquet` | `build_buffett_ratio.py` → `macro.buffett_ratio_daily(publish_lag_days=60)` | 日频 = A股总市值 / 名义GDP（年化），GDP **按发布日对齐**防前视 |
| M2 同比 | `stockworm/data/macro/m2_growth.parquet` | `macro.m2_growth_daily(win=12)` | 流动性共振，阻尼巴指假信号 |

### 1.4 筹码成本分布模型（解禁免疫的核心）
`chip_structure.py` 用**衰减成本分布**（移动筹码分布/无限衰减）：
1. 每股票维护价格分桶直方图 `H`（100 bins，固定区间=全历史 1%~99% 分位）。
2. 每日：`H = H * (1 - turnover/100)`（老筹码离场）；在 `[low, high]` 上按三角分布（峰在 `(open+close)/2`）注入 `volume`。
3. **流通股本由 `volume / (turnover/100)` 反推并注入衰减** → 解禁日换手率骤降自动使反推流通股本骤升、成本峰下移，**对解禁天然稳健**（无需逐日流通股本历史）。
4. 输出 6 维：`chip_profit_ratio`(获利盘比例)、`chip_cost_dev`(平均成本偏离)、`chip_conc70/90`(成本集中度)、`chip_disp`(成本离散度/变异系数)、`chip_skew`(成本分布偏度)。

---

## 2. 数据湖组成（Data Lake Composition）

### 2.1 股票池过滤（Universe，三规则）
`universe.py::build_universe()`：全市场 → 剔 ST/*ST/退市（按 `stock_names` 名称匹配）→ 剔次新（上市< `MIN_LISTED_DAYS=250` 交易日）→ 剔低流动性（近 `LIQ_WINDOW=60` 日均成交额 < `MIN_AVG_AMOUNT=3e7` 即 3000 万元）。
- **实测**：5596 → **5175** 只（组合已剔 ST/退市，**无生存者偏差**，这是股灾抗压测试结论的诚实边界之一）。
- 名称表为当前快照（历史 ST 状态无 PIT），对挖因子影响有限（ST 股权重本就低）。

### 2.2 变量池（三维度）
`base_data.py::derive_variables()` 从 6 基础字段派生 ~19 维标准 alpha 变量（收益类 `ret_1/5/10/20`、`vol_20`、`amp_1`、`range_20`、`vrate_5/20`、`arate_5`、`dist_high/low_20` 等）+ 自动并入 chip 6 维 → **价量 + 筹码 两维度已入池**。
- **拥挤度 18 维**（用户方法论维度：资金流/北向/两融/订单流集中度等）：当前**尚未接入变量池**（待办，§10），因子挖掘目前以价量+chip 为主。

### 2.3 面板 Schema 约定
- 单字段面板 `date×stock`，`float32`，约 42MB/全市场 5448 只（远小于 XGBoost 特征矩阵，故挖掘阶段可放宽样本）。
- `load_panel(field, codes, start, end)` 兼容 date 作列/作索引两种存储；`index` 统一为 `Timestamp`。

### 2.4 关键设计：解禁免疫
旧快照法用 `成交量/当前流通股本` 反推换手率，在解禁日因流通股本突变系统性低估早期换手 → 污染成本分布。本湖用 akshare **交易所披露真实换手率** + 反推逐日流通股本注入衰减，从根本上消除该偏误。

---

## 3. 因子挖掘（Factor Mining）

编排脚本：`mine_v2.py`。变量池 → 四方向并行挖掘 → `factors_v2_3dim.json`。

| 方向 | 模块 | 方法 | 产出 |
|---|---|---|---|
| 1 Grid | `grid_search.py` | 算子树全组合 + `require_ic_pos`（只保留 IC>0） | 候选 |
| 2 GP | `genetic_programming.py` | 遗传规划 pareto 前沿（pop 30×12 代×3 seed） | 候选 |
| 3 MCTS | `llm_mcts.py` | LLM+蒙特卡洛树搜索（`max_depth=2`, 150 迭代, `icir>0`） | 候选 |
| 4 XGB 交互 | `xgb_interaction.py` | XGBoost 交互挖掘（`max_depth≤3`，树路径特共现对→DSL 公式） | 候选（chip 交互加权） |

- 每方向独立 `try/except` + **增量落盘**（`save(seen)` 每次写完），单方向崩不影响整体。
- 样本：过滤池随机采样 `N_MINE=300` 只（seed 42）；标签 `forward_returns(horizons=5/10/20/60)`。
- **产出**：`factors_v2_3dim.json`，**118 个去重候选因子**（含 `expr_tuple`、`dir`、`ic20`、`icir20`）。其中含 chip 变量因子占显著比例。

---

## 4. 因子产出（Factor Output）

两层 JSON：候选 → 入选。

### 4.1 候选：`factors_v2_3dim.json`
118 个正 IC 去重因子，附四方向标签与整样本 IC/ICIR。

### 4.2 入选：`factors_v2_selected.json`（P3 衰减感知筛选）
`factor_screen.py::screen_factors()` —— **专治"450 通用因子远期行、近期衰减"病**：
1. **按年切块**：`DECAY_BLOCKS = [2019..2025]`，每块算 Rank-IC（trailing 口径防前视）。
2. **指数衰减加权**：`decayed_ic = Σ(ic_block × w_block)`，`w` 半衰期 `DECAY_HALFLIFE=2.0` 年（2025 权重最高）。
3. **多重检验校正**：`decayed_ic` 正态近似 `z = ic·√n_days` → 双尾 p → **Benjamini-Hochberg FDR**（`FDR_Q=0.10`），只保留 `q<0.10` 且 `decayed_ic>0.006` 且有效块≥3。
4. **方向 4 tie-break**：XGB 挖出的 chip 交互因子 `decayed_ic × 1.15`（**不绕过 FDR**，仅作同分优先）。
5. 取 `keep=60` 上限，实际 **入选 40 个**（`RESULT_v2_3dim.md` 印证）。

---

## 5. 因子训练（Factor Training — 分块 WFA）

编排：`run_wfa_v2.py` → `factor_wfa.py`。

### 5.1 特征表（Pass 自包含，断点续传）
`build_feature_table_chunked(codes, exprs, out_dir, chunk_months=12, lookback=252)`：
- 日期切成 12 月不重叠窗（`_month_chunks`）；Pass-1 面板落盘 + 在线 z-score；`resume=True` 断点续传。
- 输出 `wfa_v2_store/feat_*.parquet`（long：`date,code,fwd_ret_*,cls_*,<feat_cols>`）。

### 5.2 WFA（滚动折，防前视）
`run_wfa_chunked(feat_dir, feat_cols, train_cap=2_000_000)`：
- **4 折**：训练 3 年 / 测试 1 年 / 步长 1 年（`wfa_folds`）。
- IS 子采样 `TRAIN_CAP=2M`（控内存/提速）；**OOS 窗全量**（评估无偏）。
- 每折 XGBoost 融合 **5/20/60 日标签**（`HORIZONS`），输出 `fused`（融合概率/分数）。
- 落盘 `factor_mining/oos_detail_v2.parquet`（long：`date,code,fused,fwd_ret_1,cls_5/20/60`）。

### 5.3 衰减治理验证（P4 核心结论）
最近期折 4 的 `ic_fuse_20 = +0.102` **远高于** 折 1 的 `+0.034` → 完全逆转"远期行、近期衰减"病，证明 P3 衰减感知筛选有效。

---

## 6. 回测（Backtest）

`factor_wfa.py::backtest(oos_detail, top_frac=0.3)`：
- 每日按 `fused` 排序，买**前 30% 等权**，持有一日（T+1 收益），次日再平衡。
- 基准 = 全市场等权（`fwd_ret_1` 截面均值）。

**实证（`RESULT_v2_3dim.md`，2021-10-25 ~ 2025-09-18，901 交易日）**：
| 指标 | 因子组合 | 全市场等权基准 |
|---|---|---|
| 总收益 | +70.4% | +8.4% |
| 年化 | **+16.1%** | +2.3% |
| 年化夏普 | 0.71 | — |
| 最大回撤 | **-37.0%** | -33.7% |
| Calmar | 0.43 | — |

> 注：回测为**毛收益未扣费**；覆盖 WFA 全部 OOS 窗、无前视泄露。

---

## 6.5 因子终验（SHAP 可解释性验证，P6）

`shap_validate_v2.py`：复现末折（2024-25 近期）IS 模型（2021-10~2024-10 训练，目标 20 日），用 **TreeExplainer** 在 OOS **4 万样本**上算精确 SHAP；输出 `shap_summary_v2.parquet` + `RESULT_v2_shap.md`（6.3min）。

**定位**：SHAP = 流程最后的**因子终验关**——收尾体检，验证入选因子方向是否符合经济直觉（**非替代** gain 选因子）；`gain` 只有大小无方向、且偏向高基数特征，SHAP 补方向与可信度。

### 6.5.1 SHAP Top15（按 mean|SHAP| 降序）

| 排名 | 因子 | mean\|SHAP\| | 方向corr | gain排名 | chip? |
|---|---|---|---|---|---|
| 1 | `ts_max(chip_conc70,10) sub high sub cs_rank(ts_rank(dist_high_20,60))` | 0.1083 | +0.41 | 3 | |
| 2 | `0.0 sub chip_cost_dev div vol_20` | 0.0564 | -0.81 | 16 | |
| 3 | `chip_disp` | 0.0563 | +0.54 | 8 | ✓ |
| 4 | `0.0 sub cs_rank(chip_cost_dev) mul cs_rank(vol_20)` | 0.0560 | +0.44 | 15 | |
| 5 | `chip_conc90` | 0.0539 | -0.53 | 9 | ✓ |
| 6 | `ts_rank(ret_10,20) sub high sub chip_conc70` | 0.0484 | +0.45 | 4 | |
| 7 | `0.0 sub ret_20 div vol_20` | 0.0405 | -0.72 | 37 | |
| 8 | `chip_profit_ratio` | 0.0405 | +0.48 | 12 | ✓ |
| 9 | `0.0 sub cs_rank(chip_conc90) sub cs_rank(chip_disp)` | 0.0394 | +0.63 | 7 | |
| 10 | `cs_rank(chip_disp) mul 1.0 sub cs_rank(ret_20)` | 0.0384 | +0.60 | 1 | |
| 11 | `cs_rank(vol_20 sub ret_20)` | 0.0376 | +0.77 | 36 | |
| 12 | `cs_rank(vol_20 div amount)` | 0.0360 | +0.38 | 2 | |
| 13 | `ts_min(ret_1 sub amount div ret_5 div log_ret_1,10)` | 0.0332 | +0.07 | 10 | |
| 14 | `cs_rank(chip_disp) sub cs_rank(vol_20)` | 0.0325 | -0.78 | 35 | |
| 15 | `0.0 sub cs_rank(ret_20) mul cs_rank(vol_20)` | 0.0324 | +0.28 | 6 | |

> 方向 corr >0 = 因子值越大越利多（SHAP↑），<0 = 越大越利空。

### 6.5.2 筹码（chip）6 维 SHAP 表现

| 因子 | SHAP排名 | mean\|SHAP\| | 方向corr |
|---|---|---|---|
| `chip_disp` | 3 | 0.0563 | +0.54 |
| `chip_conc90` | 5 | 0.0539 | -0.53 |
| `chip_profit_ratio` | 8 | 0.0405 | +0.48 |
| `chip_disp div vol_20` | 17 | 0.0280 | +0.47 |
| `chip_skew` | 18 | 0.0277 | -0.30 |
| `chip_cost_dev` | 22 | 0.0252 | -0.72 |
| `chip_conc70` | 25 | 0.0248 | +0.31 |

### 6.5.3 结论
- **方向验证符合经济直觉**：`chip_disp` 分散→好（+0.54）、`chip_cost_dev` 偏离成本→回归（-0.72）、`chip_conc90` 集中→差（-0.53）、`chip_profit_ratio` 获利盘高→好（+0.48，与"获利盘高=抛压"的朴素直觉略反，需后续盯）。
- **SHAP 排序 vs gain 排序 Spearman = 0.57**（中等分歧，坐实 gain 有偏、SHAP 更可信）：如 `vol_20 sub ret_20` SHAP 第 11 / gain 第 36。
- **chip 6 维平均 SHAP 排名 14/46**，验证筹码维度确有 alpha 且可解释，与 §3/§4 的"chip 主导 XGBoost 重要性"互相印证 → 三维度因子设计成立。

---

## 7. 防御门控（Defensive Gating — 完整双尾）

用户设计的**完整双尾** = 宏观三腿（上证基准 + 左侧巴指 + 因子衰减 + 敏感耦合）+ P5c 持仓倾斜。代码：`defensive_gating.py`（真组件）+ `gate_full_v3.py`/`gate_full_v4.py`（整合）。

### 7.1 四腿定义
| 腿 | 信号 | 动作 | 参数 |
|---|---|---|---|
| ① 右侧危机确认 | 上证综指跌破 250 日线 **-10%** 或 波动 z **>2** | 仓位压到 `CRISIS_POS=0.60`（**不归零**） | `CRISIS_MA=250`, `CRISIS_MA_THR=-0.10`, `CRISIS_VOL_Z=2.0` |
| ② 左侧巴指双尾 | 巴指 5Y 分位 **<P20（底部）/ >P80（顶部）** → `defensive_tilt∈[0,1]`（P5/P95 达 1） | 缓冲降仓 ≤ `MAX_POS_REDUCE=0.20`（**保持满仓只调结构**） | 双尾线性，M2 同比共振阻尼 |
| ③ 因子衰减预警 | 组合滚动 IC（60d）< `GATE_DECAY_FRAC=0.30 ×` 历史 IC（250d） | 判熊，降仓 ≤ `MAX_POS_REDUCE_DECAY=0.20` | `daily_cross_ic` + `factor_decay_flag` |
| ④ 敏感耦合 | `tilt>0.5`（仅极端预警区）时右侧阈值左移 | MA 阈值 -10%→**-5%**、vol z 2→**1**（更早触发） | `SENS_MA=0.05`, `SENS_VOL=1.0`, `SENS_GATE=0.5` |

- 仓位合成：`buf = max(tilt×0.20, decay×0.20)`；`pos = 1-buf`；若危机确认 `pos = min(pos, 0.60)`；空仓部分吃 `DEF_ANN=0.04` 防御资产年化。
- **ML 双模型（市场 regime + 个股 name-risk）为第三腿，本次未建**（待办 §10）。

### 7.2 P5c 持仓结构倾斜（巴菲特左侧驱动，价量波动防御度）
`gate_full_v4.py::tilted_daily()`：
- 选股：`top-30% by fused`（alpha 来源与裸多头/旧双尾完全一致，保收益）。
- 防御度：`D̃ = clip(1 + 0.5·zmean[-vol_20, -downside_vol_60, +chip_disp, -chip_conc90], 0, 2)`（纯价量+chip，**无基本面/PIT 泄漏**；`ivol_60` 面板 bug 已修待重算，暂用 4 有效列）。
- 权重：`w = (1-tc)·等权 + tc·softmax(fused·D̃)`，`tc = min(tilt, TILT_CAP=0.70)`。
- **关键性质**：`tc=0`（巴指无预警）→ 纯等权，**收益零损耗**；`tc>0` → 向"高 alpha×高防御"滑。巴指阴跌钝化无碍——它只管"何时开启倾斜"，倾斜本身用快变量。

### 7.3 实证结果

**P5c 完整双尾+倾斜（`RESULT_v2_gate_full_v4.md`）**：
| 组合 | 年化 | 夏普 | Sortino | 最大回撤 | Calmar |
|---|---|---|---|---|---|
| 裸多头（等权，无门） | +16.0% | 0.71 | 0.82 | -37.0% | 0.43 |
| 持仓倾斜（无危机门） | +16.1% | 0.72 | 0.82 | -36.9% | 0.44 |
| **完整双尾+倾斜(v4)** | **+16.4%** | **0.81** | **0.95** | **-26.5%** | **0.62** |
| 基准（全市场等权） | +2.2% | 0.21 | 0.25 | -33.7% | 0.06 |

→ 回撤 **-37% → -26.5%**（削 10.5pp），年化**不降反升**（+16.0%→+16.4%）；危机段内最大回撤 **-29.5% → -18.6%**（少亏 10.9pp）。倾斜单独对回撤几乎无贡献（巴指稀疏，`tilt` 均值 0.09），回撤削减主要来自危机门。

**P6 SHAP 可解释性**：详见 **§6.5 因子终验（SHAP 可解释性验证）**——含 Top15 表、chip 6 维表、方向验证与 Spearman 一致性（SHAP vs gain = 0.57，chip 6 维平均排名 14/46）。

**极端股灾抗压（`RESULT_crash_gate_hist.md`，等权全市场组合代理，2005-2026）**：
| 窗口 | 裸回撤 | 门控回撤 | 削减 | 危机门激活 |
|---|---|---|---|---|
| 2008 全球金融危机 | -71.3% | -54.1% | +17.2pp | 85% |
| 2015 年中 acute(6-8月) | -44.7% | -31.4% | +13.3pp | 68% |
| 2016 熔断 | -24.2% | -15.0% | +9.3pp | 44% |
| 2022 慢熊 | -27.9% | -20.3% | +7.6pp | 18% |
| 2024.1-2 微盘/DMA踩踏 | -28.6% | -19.6% | +9.0pp | 68% |
| 全历史 2005-2026 | -71.3% | -54.1% | +17.2pp | 25% |

→ 全历史年化 **+9.5% → +13.9%**（门控反而提升长期收益）。

### 7.4 诚实边界（用户 21:45 三点纠正）
1. 股灾抗压测试用**等权全市场组合作被保卫仓位（透明代理，不吃 alpha）**即足够验证门控抗压性，无需融合 alpha。
2. 组合已剔 ST/退市，**无生存者偏差**。
3. **巴指仅针对"系统化过热→可能雪崩"**；2008（外因全球危机）、2016（熔断机制缺陷）**非过热**，巴指沉默是**正确行为而非失败**。通用安全网由**①危机门（上证破位）**承接所有崩盘类型。

---

## 8. 关键实证数字汇总

| 阶段 | 指标 | 值 |
|---|---|---|
| 股票池 | 全市场→过滤池 | 5596 → 5175 |
| 因子挖掘 | 候选因子 | 118（四方向） |
| P3 筛选 | 入选因子 | 40 |
| WFA | 折数 / 训练窗 | 4 折 / 训练3y测1y |
| 衰减治理 | 折4 vs 折1 `ic_fuse_20` | +0.102 vs +0.034（近期反超） |
| 回测 | 年化 / 回撤 / 夏普 | +16.1% / -37.0% / 0.71 |
| 双尾+倾斜 | 年化 / 回撤 / 夏普 | +16.4% / -26.5% / 0.81 |
| 危机段内 | 裸 vs v4 回撤 | -29.5% → -18.6% |
| SHAP | chip 平均排名 / Spearman | 14/46 / 0.57 |
| 因子图书馆 | 冻结/衰减/再冻结 | 40因子 / 21冻结 / 4衰减 / 未触发再冻结 |
| 信号融合 | frozen vs xgb OOS rank-IC；日/周换手 A/B | +0.083 vs +0.014（元融合收敛为信任 frozen）；周换手把年化 −20%~−29% 翻正至 +1.5%~+11.2%（M1 最优 +11.2%/夏普0.60） |
| 股灾抗压 | 2008 回撤削减 | +17.2pp |

---

## 9. 文件地图（关键脚本/产出）

```
oos_framework/factor_mining/
├── base_data.py            # 数据底座：load_panel/load_turnover/load_chip_panels/derive_variables
├── universe.py             # 股票池三规则过滤 → metadata/universe_stats.json
├── chip_structure.py       # 衰减成本分布模型 → chip 6 维
├── build_chip_lake.py      # 全市场 chip 面板落盘
├── build_defense_panel.py  # P5c 防御面板（价量波动+chip）
├── mine_v2.py              # 四方向挖掘编排 → factors_v2_3dim.json
├── factor_screen.py        # P3 衰减感知筛选 → factors_v2_selected.json (40)
├── factor_wfa.py           # build_feature_table_chunked / run_wfa_chunked / backtest
├── factor_zoo.py           # 因子图书馆: register→freeze→monitor→maybe_refreeze 生命周期管理
├── factor_zoo_state.json   # 因子图书馆状态落盘(ICIR权重/衰减标志/冻结集/再冻结次数)
├── FACTOR_ZOO_REPORT.md    # 因子图书馆生命周期报告(40因子/21冻结/4衰减)
├── signal_meta_learner.py  # 信号融合器: frozen ICIR × XGBoost 元学习融合(体制条件)
├── META_SIGNAL_RESULTS.md  # 信号融合器实证报告(日/周换手 A/B 四路对照 + 体制分段)
├── portfolio_optimizer.py  # 组合优化器: 信号→约束型多头组合(可落地的配置层)
├── demo_portfolio_optimizer.py  # 真实数据 A/B: naive 等权 vs 约束优化
├── run_wfa_v2.py           # P4 全池 WFA 编排
├── shap_validate_v2.py     # P6 SHAP 终验
├── gate_full_v3.py         # 双尾(上证+巴指+衰减+耦合)
├── gate_full_v4.py         # P5c 双尾+持仓倾斜（最终）
├── crash_gate_full_hist.py # 极端股灾抗压测试
├── defensive_gating.py     # 宏观/危机真组件（_load_macro/_macro_gating/_crisis_signal）
├── oos_detail_v2.parquet   # WFA OOS 明细（门控输入）
├── defense_panel.parquet   # P5c 防御面板
├── factors_v2_3dim.json / factors_v2_selected.json
├── RESULT_v2_3dim.md / RESULT_v2_gate_full_v4.md / RESULT_v2_shap.md / RESULT_crash_gate_hist.md
└── wfa_v2_store/           # 分块特征 store

stcok-worm/
├── stcok_worm/macro.py     # buffett_ratio_daily / m2_growth_daily
├── build_buffett_ratio.py  # 宏观信号落盘 → data/macro/
└── data/macro/             # buffett_ratio.parquet / m2_growth.parquet
```

---

## 10. 待办 / 下一步（Open Items）

1. **`ivol_60` 重算**：`build_defense_panel.py` 的 beta/resid 计算 bug 已修（改用 numpy 广播），需重跑防御面板把 `ivol_60` 纳入 `D̃`（目前暂用 4 有效列）。
2. **ML 双模型第三腿**：市场 regime 模型 + 个股 name-risk 模型（用户双尾设计中的 ML 腿），尚未建。
3. **拥挤度 18 维接入变量池**：当前仅价量+chip 两维度入池，方法论第三维度待接通。
4. **严格 PIT ST 历史**：`universe.py` 名称表为当前快照，历史 ST 状态需接历史记录以增强严谨性。
5. **扣费回测 / 组合优化**：✅ 已解决。`portfolio_optimizer.py` 在 `backtest` 的等权 top-N 之外新增**约束型配置层**（个股上限 + 分组中性 + 换手上限 + 20bps 交易成本模型），`backtest_optimized()` 与 `backtest` 同指标口径。详见第 11 节与 `PORTFOLIO_OPT_RESULTS.md`。

---

## 11. 组合优化器模块（新增 2026-07-24）

把"选股信号"升级为"可落地的组合"。两阶段（选股 → 配置）、投影+收缩法求解（纯 numpy，无外部求解器依赖），约束：多头 / 满仓 / 个股权重上限 / 分组中性（行业·市值·自定义）/ 换手上限。

接口：
```python
from factor_mining.portfolio_optimizer import backtest_optimized, compare_backtests, optimize_panel
r_opt = backtest_optimized(oos_detail, universe_frac=0.3, max_w=0.03,
                           group_neutral=True, neutral='cap', turnover_limit=0.3)
cmp   = compare_backtests(oos_detail, universe_frac=0.3, max_w=0.03, turnover_limit=0.3)
```

真实数据 A/B（冻结因子策略，400 只，OOS rank-IC=+0.085，扣 20bps）：

| 指标 | naive 等权 top-30% | 约束优化 | 变化 |
|---|---|---|---|
| 年化（净） | −20.9% | **+4.5%** | +25.3pp |
| 夏普 | −1.03 | **+0.31** | +1.34 |
| 最大回撤 | −67.3% | **−35.9%** | +31.3pp |
| 日均交易成本 | 0.00166 | **0.00060** | −64% |

> 本环境数据湖无行业分类、无市值快照、筹码/换手率湖为空（chip 因子为 NaN），故真实信号弱于 headline；优化器在弱信号下仍靠**权重上限+中性化+换手约束**把亏损组合扭转为盈利、回撤腰斩、成本减半。接入行业/市值数据后效果更显著。
> 换手权衡：本样本弱信号下换手越紧收益反而越高（0.15→+22.2% vs ∞→+19.2%），说明约束换手可抑制追逐噪声的频繁调仓。

自测：`python portfolio_optimizer.py --selftest`（约束/中性/换手断言全过）。

---

## 12. 因子图书馆模块（factor_zoo，新增 2026-07-24）

把分散的「因子定义 / 冻结权重 / 衰减监控 / 再冻结」收敛到一个**有状态的图书馆**，实现完整生命周期：

```
register ──▶ freeze(IS 期算 ICIR 权重, 锁定因子集) ──▶ monitor(滚动 IC 衰减监控)
                 ▲                                            │
                 │                                            ▼ 衰减占比 > 阈值
                 └────────────── refreeze(用近期 IS 重算权重)◀──┘
```

- 状态落盘 `factor_zoo_state.json`：每个因子的定义 + 生命周期元数据（IS_IC/ICIR/近期IC/历史IC/衰减标志/冻结权重/冻结次数），全局 `frozen_set` / `freeze_date` / `refreeze_count`。
- `freeze` / `refreeze` 复用 `frozen_gate_wfa.frozen_icir_weights`（IC>0 & ICIR>0 → 权重=ICIR）。
- `monitor` 复用向量化截面 rank-IC（与 `factor_decay_monitor` 同源口径）：近窗 IC < 历史 ×`DECAY_FRAC`(0.30) 或近窗转负 → 衰减；衰减因子占比 > `REFREEZE_THRESHOLD`(0.40) → 触发再冻结。
- `weights_vector(feat_cols)` 直接产出对齐的冻结权重数组，与现有冻结策略管线零摩擦对接。

接口：
```python
from factor_mining.factor_zoo import FactorZoo
zoo = FactorZoo()
zoo.register_many(items, families=families, source="mine_v2")
zoo.freeze(long, feat_cols, is_cut)
zoo.maybe_refreeze(long, feat_cols, is_cut)   # 衰减超阈值则自动重算权重
w = zoo.weights_vector(feat_cols)             # 对齐的冻结权重
```

真实数据实证（400 只，`FACTOR_ZOO_REPORT.md`）：

| 指标 | 值 |
|---|---|
| 因子总数 | 40 |
| IS 冻结（IC>0 & ICIR>0） | 21 / 40 |
| 衰减因子（近窗 IC 跌破基线） | 4（10%） |
| 再冻结触发 | 否（衰减占比 10% ≤ 40%） |
| 冻结点（IS 锁定） | 2021-10-23 |

> `chip_*` 系列因子 IC=None（本环境筹码湖为空，NaN 因子自动按中性 0 处理，其 IS 权重本就≈0），不影响其余 21 个有效因子的冻结与监控。

---

## 13. 信号融合器模块（signal_meta_learner，新增 2026-07-24，增补周换手 A/B）

把两条已有信号管线做**元学习融合**，回答「静态冻结 ICIR 因子合成（稳健/零重训） vs 动态 XGBoost WFA（自适应/每折重训）谁在哪种市场更准、能否动态融合」。

两条基准信号（同一 OOS 宇宙，公平对照）：
- **frozen**：`frozen_icir_weights` 在 IS 锁定因子集与 ICIR 权重，OOS 静态加权合成，**零重训**。
- **xgb**：`run_wfa` 的 WFA 融合概率，每折重训 XGBoost。

两个元学习融合器（均 walk-forward，无前视）：
- **M1 regime_blend（主，体制条件动态凸融合）**：逐日按当前市场体制（bull/bear/osc，MA120+回撤口径，见 `REGIME_TRAINING_PLAN.md`）取该体制下两信号的滚动 rank-IC，权重 `clip(IC_f/(IC_f+IC_x), 0.15, 0.85)` → 凸融合。直接落地「按体制切换因子/信号」哲学（信号级版本）。
- **M2 stacked（次，滚动岭回归元模型）**：以 `(frozen, xgb)` 为元特征，过去 ~252 交易日滚动岭回归拟合 `fwd_ret_1`，预测次日信号，作对照。

四路回测（20bps 扣费，150 只，OOS 2021-10 ~ 2025-10）。**关键发现：同一信号跑「每日再平衡」与「每周再平衡」两遍，周换手把年交易次数从 ~252 降到 ~52、年化交易成本 42.1%→12.9%（砍掉 69%），直接逆转收益符号——印证此前「日换手吃掉 ~42%/年」的假设。**

每日再平衡（freq=1，原行为）：

| 信号 | 年化 | 夏普 | OOS rank-IC |
|---|---|---|---|
| 冻结 ICIR（基准） | −20.3% | −0.97 | **+0.0826** |
| XGBoost WFA（基准） | −29.2% | −1.36 | +0.0140 |
| M1 体制融合（主） | −20.5% | −0.94 | +0.0808 |
| M2 岭回归（对照） | −15.3% | −0.63 | +0.0698 |

每周再平衡（freq=5，低本版必选项）：

| 信号 | 年化 | 夏普 | OOS rank-IC |
|---|---|---|---|
| 冻结 ICIR（基准） | **+9.5%** | 0.53 | +0.0826 |
| XGBoost WFA（基准） | +3.2% | 0.25 | +0.0140 |
| M1 体制融合（主） | **+11.2%** | 0.60 | +0.0808 |
| M2 岭回归（对照） | +1.5% | 0.17 | +0.0698 |

结论：
- 两基准信号 **frozen OOS rank-IC=+0.083 显著强于 xgb=+0.014**；元学习器均正确识别 frozen 为更优基信号，融合结果贴近 frozen、且都明显优于裸 xgb（周换手年化改善 +8.1pp~−1.7pp）。
- **周换手是低本版必选项**：日换手把所有信号拖成深度负收益（−15%~−29%），根源是 ~42%/年的换手成本；切到周换手后四路信号全部翻正（+1.5%~+11.2%），同信号年化普遍改善 +29.8pp（frozen）/ +31.8pp（M1），日均成本砍掉约 69%。
- 体制切换诊断：本数据**无体制出现 xgb 反超**，故 M1 自动收敛为「重 frozen、轻 xgb」——该逻辑已就位，一旦未来某体制 xgb IC 占优会自动切权（即 `REGIME_TRAINING_PLAN.md` 哲学的信号级落地）。
- 推荐生产信号：**M1 体制融合（主）周换手**（本样本年化最优 +11.2%、夏普 0.60）；更稳健/零重训成本方案回退 **frozen 周换手**；**勿直接上线裸 xgb**（最弱，且勿用日换手上线任何信号）。注意：此为弱 alpha 信号，绝对收益受 2021-2025 震荡市拖累，实战应接 §11 组合优化器做配置层优化。

自测：`python signal_meta_learner.py --selftest`（合成数据冒烟，验证融合/岭回归逻辑、体制切换生效）。

---

*文档由系统整合任务生成（2026-07-23 初版，2026-07-24 增补组合优化器 / 因子图书馆 / 信号融合器章节），覆盖数据湖→挖掘→训练→回测→门控→配置→信号融合全链路，所有数字均来自 `RESULT_*.md` / `PORTFOLIO_OPT_RESULTS.md` / `FACTOR_ZOO_REPORT.md` / `META_SIGNAL_RESULTS.md` 实证产出。*
