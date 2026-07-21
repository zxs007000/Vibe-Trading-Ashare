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

## 3. 特征族（v4 全量 = 208 维）

| 族 | 维度 | 内容 |
|---|---|---|
| **Alpha(价量)** | 17 | ret_1/5/20/60, vol_20, rsi_14, amt_chg_20, ma_dev_20, amp_20 … |
| **市场环境** | 8 | mkt_ret20_std, mkt_adv, mkt_roe_mean, mkt_amp_mean, mkt_vol_mean …(横截面聚合) |
| **交互** | 165 | Alpha × 市场环境 的 原始值/历史分位/变化率/宏观交互/惩罚项(前缀 `ix_`) |
| **因子拥挤度** | 18 | §1.1 交易行为(成交额代理 mom/liq) + §1.4 PCA 吸收比率; §1.2 估值价差(需 PE)、§1.3 机构持仓(需基金持仓) 数据湖缺失→跳过 |
| **筹码结构** | — | §2 VWAP 中心三角分布递推 + §4 PR/CC/CB/短期 CB(`chip_pr/chip_cc/chip_cb/chip_cb_short`); §5.1 PR×CB 交互、§5.2 短期乖离惩罚 + CC 作样本权重 |

**演进路径**(本目录各 `xgb_wfa_proto*.py`):
`v1`(基线, AUC≈0.528) → `v2`(+交互+多周期融合+IS 调参) → `v3`(+因子拥挤度) → `v4`(+筹码结构) → `v4_full`(全市场 5448 只)。
对比见 [`results/proto_v2_v3_v4_comparison.md`](./results/proto_v2_v3_v4_comparison.md)。

---

## 4. SHAP 解释（修正版, 见 `shap_analysis.py`）

修复了朴素 SHAP 的多个漏洞:

| 漏洞 | 处理 |
|---|---|
| 规模 | 训练时已子采样到 ≤3000 行/折, 精确 `TreeExplainer` 可行(不碰全量 9M 行) |
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
bash run_full_on_build_done.sh

# 方式 B: 直接跑(更可控)
export STOCKLAKE=/path/to/your/stocklake WFA_OUT=./v4proto_out
python3.11 xgb_wfa_proto_v4_full.py           # 训练 4 折, 落盘 booster/shap_data/feats/结果 md
python3.11 shap_analysis.py --base ./v4proto_out   # SHAP 解释
```

> 想先冒烟测试? 改 `xgb_wfa_proto_v4_full.py` 顶部 `MAX_STOCKS`(如 20)即可小样本跑通。

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

---

## 7. 文件索引

| 文件 | 作用 |
|---|---|
| `xgb_wfa_proto.py` | v1 基线(AUC≈0.528) |
| `xgb_wfa_proto_v2.py` | +交互 +多周期融合 +IS 调参 |
| `xgb_wfa_proto_v3.py` | +因子拥挤度 |
| `xgb_wfa_proto_v4.py` | +筹码结构维度 |
| `xgb_wfa_proto_v4_full.py` | **全市场版**(5448 只, 本目录主力) |
| `shap_analysis.py` | 修正版 SHAP 解释(跨折方向一致性 + bootstrap CI) |
| `diag_v3.py` | v3 诊断脚本 |
| `run_full_on_build_done.sh` | 端到端无人值守启动器 |
| `datalake/` | 数据湖构建(日线 + 三大表 + cnstock 参考) |
| `results/` | 各版本结果 markdown + 全量 SHAP 报告/图表 |
