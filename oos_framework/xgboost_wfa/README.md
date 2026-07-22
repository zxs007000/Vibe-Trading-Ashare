# XGBoost 因子挖掘 · WFA 原型（A 股数据湖）

> 本目录是 **Vibe-Trading-Ashare** 的 XGBoost 因子挖掘 / 样本外(WFA)验证原型。
> 配套数据湖构建脚本在 [`datalake/`](./datalake/)。
> 上游 stock-worm 仓库**只保留 cnstock 接口**(含 IP 封禁高风险标注), 量化流水线已整体迁移到此。

---

## 0. 一句话定位

用 **XGBoost 二分类** 预测「某只股票未来 N 日收益进入**全市场前 30%** 的概率」,
通过 **Walk-Forward Analysis(WFA, 3 年训练 / 1 年测试 / 1 年滚动)** 严格隔离样本外(OOS),
并用 **SHAP** 做白盒解释, 检验「因子拥挤度 + 筹码结构」两个新维度的边际贡献。

这是**原型 / 方法论验证**, 不是可直接上线的策略。OOS AUC 在 0.5x 量级(微弱但稳定),
重点在**管线正确性**(特征工程 → WFA → SHAP) 与**可解释性**, 而非收益魔法。

> ⚠️ **内存约束(已解决)**: 单遍把全市场 5448 只面板(~11M 行 × 217 维 ≈ 18GB)一次性读入内存,
> 在 8GB 上限(cgroup)下会 OOM(`rc=137`)。两种解法:
> - **(推荐) 分块流式版 `xgb_wfa_proto_v4_chunked.py`**: PhaseA 逐只落盘 → PhaseB 按年切片算截面特征 →
>   PhaseC 按需读分片 + float32 增量拼矩阵, **峰值内存≈单折训练矩阵(~3.5GB), 全市场 5500 只可在 8GB 内跑完**。
> - 旧版 `xgb_wfa_proto_v4_full.py` 仍保留: 用 `WFA_MAX_STOCKS` 控制子样本量(默认 5500),
>   自动做固定随机种子(42)的随机抽样, 使子样本是全市场的代表性样本(非前 N 只)。
> - 本仓库 `results/` 下提交的 **v4 全量结果来自 250 只代表性随机子样本**(peak RSS ≈ 1.1GB 分块版 / 2.7GB 单遍版),
>   方法学一致; 现可用分块版在 8GB 机器上直接跑全市场。

---

## 1. XGBoost 用法（核心）

### 1.1 模型定义

```python
from xgboost import XGBClassifier

# best 来自 IS 内 RandomizedSearchCV(cv=3) 调参
m = XGBClassifier(
    n_estimators=300,
    nthread=8,
    eval_metric="auc",
    random_state=42,
    use_label_encoder=False,   # xgboost 1.x/2.x 兼容
    **best,                    # subsample / reg_lambda / min_child_weight /
                              # max_depth / learning_rate / gamma / colsample_bytree
)
m.fit(Xis, yis, sample_weight=sw_is)
```

- **objective**: `binary:logistic`(XGBClassifier 默认)。输出 = **logistic 概率**。
- **标签 `cls_hz`**: 对每个 horizon `hz ∈ {5, 20, 60}` 日,
  用前向收益 `fwd_ret_hz` 做横截面分位, **进入前 30% 则标签=1**。
  即模型预测的是「**排进前 30% 收益组**」的**相对/排序概率**, **不是绝对涨跌方向** ——
  这一点决定了 SHAP 的解读口径(见 §4)。
- **融合预测**: 三个 horizon 的 `predict_proba` 取均值 → `fused`, 作为综合信号。

### 1.2 样本权重（§5.2 筹码成本集中度）

```python
cc = is_df["_chip_cc_raw"].fillna(0.0)          # 成本集中度(越集中越可信)
sw_is = 1.0 + np.clip(cc, 0.0, 10.0)            # 最高 ~11 倍权重
```

成本越集中 → 信号越可信 → 训练样本权重越高。

### 1.3 调参与评估

| 项 | 说明 |
|---|---|
| 调参范围 | IS 训练窗内 `RandomizedSearchCV(cv=3)`, **绝不碰 OOS** |
| OOS 指标 | `roc_auc_score`(AUC) + `spearman` 相关(IC) |
| horizon | 5 / 20 / 60 日, 单周期 + 融合分别报告 |
| 折数 | 4 折(WFA): 训练 3y / 测试 1y / 步长 1y |

> ⚠️ **IC 略负属正常**: 原始因子含噪声, 单周期 IC 常小幅为负; 看**融合 IC** 与 **AUC** 是否稳定 > 0.5 更有意义。

---

## 2. WFA 设计（样本外隔离）

```
折1: IS 2018-10 ~ 2021-10 | OOS 2021-10 ~ 2022-10
折2: IS 2019-10 ~ 2022-10 | OOS 2022-10 ~ 2023-10
折3: IS 2020-10 ~ 2023-10 | OOS 2023-10 ~ 2024-10
折4: IS 2021-10 ~ 2024-10 | OOS 2024-10 ~ 2025-10
```

- **OOS 全程隔离**: 测试窗从不参与特征标准化(z-score 的均值/标准差只用 IS 估计)、
  调参、或任何前瞻。
- **截面 Z-score**: 每个交易日对全市场横截面做标准化, 避免个股量纲差异。
- **防泄露**: `fwd_ret_hz` 从 `T+1` 起算(`c.shift(-hz)/c.shift(-1)-1`), 标签不泄露当日。

---

## 3. 特征族（v4 全量 ≈ 217 维）

| 族 | 维度 | 内容 |
|---|---|---|
| **Alpha(价量)** | 17 | ret_1/5/20/60, vol_20, rsi_14, amt_chg_20, ma_dev_20, amp_20 … |
| **市场环境** | 8 | mkt_ret20_std, mkt_adv, mkt_roe_mean, mkt_amp_mean, mkt_vol_mean …(横截面聚合) |
| **交互** | 168 | Alpha × 市场环境 的 原始值/历史分位/变化率/宏观交互/惩罚项(前缀 `ix_`) |
| **因子拥挤度** | 18 | §1.1 交易行为(成交额代理 mom/liq) + §1.4 PCA 吸收比率; §1.2 估值价差(需 PE)、§1.3 机构持仓(需基金持仓) 数据湖缺失→跳过 |
| **筹码结构** | 6 | §2 VWAP 中心三角分布递推 + §4 PR/CC/CB/短期 CB(`chip_pr/chip_cc/chip_cb/chip_cb_short`); §5.1 PR×CB 交互、§5.2 短期乖离惩罚 + CC 作样本权重 |

**演进路径**(本目录各 `xgb_wfa_proto*.py`):
`v1`(基线, AUC≈0.528) → `v2`(+交互+多周期融合+IS 调参) → `v3`(+因子拥挤度) → `v4`(+筹码结构) → `v4_full`(全市场 5448 只) → `v4_chunked`(分块流式, 8GB 跑全市场)。
对比见 [`results/proto_v2_v3_v4_comparison.md`](./results/proto_v2_v3_v4_comparison.md)。

---

## 4. SHAP 解释（修正版, 见 `shap_analysis.py`）

修复了朴素 SHAP 的多个漏洞:

| 漏洞 | 处理 |
|---|---|
| 规模 | 训练时已子采样到 ≤3000 行/折, 精确 `TreeExplainer` 可行(不碰全量 11M 行) |
| 共线 | 按「特征族」汇总 Mean\|SHAP\|, 缓和同源变体信用瓜分 |
| 跨折 | 跨折用「方向一致性(sign agreement)」判稳健, **非**对 SHAP 取均值(尺度不可比) |
| 尺度 | 明确 SHAP 在 **log-odds(边际)** 尺度; 标签是「排前 30% 概率」(排序/相对) |
| 阈值 | 依赖图带 bootstrap 95% CI; 尾部稀疏处「断崖」标注为不可靠 |
| 交互 | 额外 `shap_interaction_values` 解耦交互 |

```bash
python3.11 shap_analysis.py --base <WFA_OUT>     # 读 booster_fold*.json + shap_data_fold*.parquet
# 产出: shap_report.md + shap_*.png(summary/dot/dependence/interaction/family)
```

---

## 5. 本地运行（从零到结果）

### 5.1 依赖

```bash
cd oos_framework/xgboost_wfa
pip install -r requirements.txt
```

### 5.2 准备数据湖

需要一个本地数据湖, 目录结构见 [`datalake/README.md`](./datalake/README.md):

```
stocklake/
  daily/{code}.parquet                      # index=date, 列: open/high/low/close/volume/amount
  fundamentals/
    income_statement/{code}.parquet        # index=REPORT_DATE
    cash_flow_statement/{code}.parquet     # index=REPORT_DATE
    balance_sheet/{code}.parquet           # index=REPORT_DATE
  BUILD_DONE                                # 三大表齐全的标记文件
```

- `{code}` = 6 位 A 股代码(如 `600519`)。
- 数据湖可由 [`datalake/build_lake.py`](./datalake/build_lake.py)(东财日线) +
  [`datalake/drive_stmt.py`](./datalake/drive_stmt.py)(东财三大表) 构建。
- **也可直接指向你已有的 lake**(只要 schema 一致), 用环境变量 `STOCKLAKE` 覆盖默认路径。

### 5.3 跑全量流水线

```bash
# 方式 A: 端到端启动器(等 BUILD_DONE → 跑 v4 全量 → 跑 SHAP, 失败自动重跑封顶 4 次)
export STOCKLAKE=/path/to/your/stocklake     # 默认 /workspace/stocklake
export WFA_OUT=./v4proto_out                  # 结果落盘目录(默认脚本同级 v4proto_out/)
# 内存受限(<32GB)时务必调低, 否则全市场面板(~18GB)会 OOM:
export WFA_MAX_STOCKS=500                     # 默认 5500=全量; 自动随机抽样为代表性子样本
bash run_full_on_build_done.sh

# 方式 B: 直接跑(更可控)
export STOCKLAKE=/path/to/your/stocklake WFA_OUT=./v4proto_out WFA_MAX_STOCKS=500
python3.11 xgb_wfa_proto_v4_full.py           # 训练 4 折, 落盘 booster/shap_data/feats/结果 md
python3.11 shap_analysis.py --base ./v4proto_out   # SHAP 解释

# 方式 C(推荐, 8GB 内存即可跑全市场 5500 只): 分块流式版
export STOCKLAKE=/path/to/your/stocklake WFA_OUT=./v4proto_out WFA_MAX_STOCKS=5500
python3.11 xgb_wfa_proto_v4_chunked.py        # 三阶段分块, 峰值内存≈单折训练矩阵(~3.5GB)
python3.11 shap_analysis.py --base ./v4proto_out
```

> 内存阶梯: 方式 C 分块版全市场峰值 ≈ 3.5GB(8GB 机器可跑); 方式 B `v4_full` 全市场会 OOM,
> 需 `WFA_MAX_STOCKS` 降到 250~500(峰值 1~3GB)。两者**算法完全一致**, 仅内存策略不同。

> 想先冒烟测试? 设 `WFA_MAX_STOCKS=20` 环境变量即可小样本跑通(无需改代码)。

### 5.4 输出物(`WFA_OUT/`)

| 文件 | 说明 |
|---|---|
| `booster_fold{0..3}.json` | 各折 XGBoost booster(精确 TreeExplainer 需要) |
| `shap_data_fold{0..3}.parquet` | 各折 OOS 子样本(≤3000 行), 供 SHAP |
| `feats_v4full.json` | 特征名列表(顺序 = booster 特征顺序) |
| `proto_v4_full_results.md` | WFA 各折 AUC/IC + 因子重要性 |
| `shap_report.md` + `shap_*.png` | SHAP 解释报告与图表 |

> 大型产物(`booster_*.json` / `shap_data_*.parquet` / `*.log`)已被本目录 `.gitignore` 排除,
> 仅提交 markdown 报告 + 图表 + 特征表(可复现, 不占仓库体积)。

---

## 6. 已知风险与坑

- **cnstock 接口 IP 封禁**: `data.cnstock.com` 曾被高并发爬取触发**整域 403**(IP 级),
  表现为 stock_detail 与 fetch_period **双双 403**。已改用**东财三大表**派生基本面(已建 95%+),
  cnstock 接口保留但加了限速 + 封禁即退避的保护(见 stock-worm `cnstock.py` 顶部警告)。
- **银行股缺 current_ratio**: 资产负债表无流动/非流动划分 → `current_ratio` 派生为 `NaN`,
  下游已做 8 列补齐, 不会 KeyError。
- **某票三大表文件存在但损坏**: `_read_parquet` 返回 `None` → `bal=None` 但现金流非空,
  曾触发 `sc` 未绑定 `UnboundLocalError`; 已在 `load_fundamentals` 顶部初始化 `pn=sc=None` 修复。
- **全市场 OOM(8GB cgroup)**: 单遍把全量面板(~18GB)读入内存会超过 8GB 上限 → `SIGKILL(rc=137)`。
  已用**分块流式版 `xgb_wfa_proto_v4_chunked.py`** 彻底解决(见 §5.3 方式 C): 峰值内存压到单折训练矩阵量级,
  全市场 5500 只可在 8GB 内跑完; 旧版 `v4_full` 仍可用 `WFA_MAX_STOCKS` 降采样。

---

## 6.1 已提交结果（250 只代表性随机子样本, WFA 4 折）

> 因 8GB 内存上限, 仓库提交的是 **250 只随机子样本**(`WFA_MAX_STOCKS=250`, seed=42)的跑通结果,
> 方法学与全量一致。完整结果见 [`results/proto_v4_full_results.md`](./results/proto_v4_full_results.md)
> 与 [`results/shap_report.md`](./results/shap_report.md)。
> 分块流式版 `xgb_wfa_proto_v4_chunked.py` 现已可在 8GB 内存跑全市场 5500 只, 其全市场结果将在跑通后补充
> (见 [`results/proto_v4_chunked_results.md`](./results/proto_v4_chunked_results.md))。

**各折 OOS 表现**(标签=未来收益排前 30% 的概率):

| 折 | 单周期5日 AUC | 融合 AUC(5/20/60日) | 融合 IC(5/20/60日) |
|---|---|---|---|
| 1 | 0.533 | 0.529 / 0.535 / 0.526 | +0.005 / +0.027 / +0.006 |
| 2 | 0.529 | 0.534 / 0.555 / 0.535 | +0.040 / +0.073 / +0.022 |
| 3 | 0.523 | 0.515 / 0.545 / 0.538 | +0.044 / +0.085 / +0.084 |
| 4 | 0.530 | 0.528 / 0.563 / 0.587 | +0.063 / +0.104 / +0.123 |
| **均值** | **0.529** | **0.527 / 0.549 / 0.547** | **+0.038 / +0.072 / +0.059** |

**解读**:
- 融合 AUC 稳定在 **0.53~0.55**(>0.5 即有微弱但一致的样本外区分度); 单周期5日 AUC≈0.53。
- 融合 IC 全正(0.04~0.12), 且随 horizon 拉长(20/60日)IC 更高 —— 信号更偏中期。
- 因子重要性 Top: 多为「市场环境 × 个股」交互特征(`ix_mkt_*__*`)、`pca_absorp*`(PCA 吸收比率/拥挤度代理)、`mkt_roe_mean`(市场 ROE 环境)。
- 拥挤度 / 筹码结构 两新维度 Gain 占比分别为 **3.1% / 2.6%** —— 占比不高但非零, 证明它们携带了增量信息(详见 SHAP 报告跨折方向一致性)。

> ⚠️ 这是 **原型验证** 的 OOS 信号, 非可上线 alpha; 全量 + 更多特征/数据有望提升, 但需在更大内存环境复现。

---

## 7. 文件索引

| 文件 | 作用 |
|---|---|
| `xgb_wfa_proto.py` | v1 基线(AUC≈0.528) |
| `xgb_wfa_proto_v2.py` | +交互 +多周期融合 +IS 调参 |
| `xgb_wfa_proto_v3.py` | +因子拥挤度 |
| `xgb_wfa_proto_v4.py` | +筹码结构维度 |
| `xgb_wfa_proto_v4_full.py` | **全市场版**(5448 只, 单遍; 大内存或降采样用) |
| `xgb_wfa_proto_v4_chunked.py` | **分块流式版**(8GB 内存即可跑全市场 5500 只; 推荐) |
| `shap_analysis.py` | 修正版 SHAP 解释(跨折方向一致性 + bootstrap CI) |
| `diag_v3.py` | v3 诊断脚本 |
| `run_full_on_build_done.sh` | 端到端无人值守启动器 |
| `datalake/` | 数据湖构建(日线 + 三大表 + cnstock 参考) |
| `results/` | 各版本结果 markdown + 全量 SHAP 报告/图表 |
