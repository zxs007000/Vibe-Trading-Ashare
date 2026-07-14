# Regime-Conditioned Factor Training Plan

## 问题

WFA 回测暴露核心矛盾：因子在牛/熊/震荡市中 alpha 结构不同，但冻结因子时不做区分。

**实证依据（2026-07-14 WFA 结果）：**

- 全量 WFA（含退市股，底仓 50%）：2024 折 Sharpe -0.16（仅存活股 +0.115），退市股拖累
- 因子冻结用历史扩张窗，2021-2023 连熊的 IC 主导冻结 → 选出的因子在 2024 牛市失效
- 1 年窗真滚动下闸门拖累 -0.082（vs 2 年窗 -0.049），训练数据越短越不稳

## 目标

闸门不仅做仓位开关，更应**按市场状态切换因子全集**。牛市用牛市区训练的因子，熊市用熊市区训练的因子。

## 两阶段方案

### Phase 1: Regime IC 诊断（轻量，纯数据探索）

**目的：** 确认因子是否真的有 regime 依赖，避免基于"拍脑袋"做复杂实现

**方法：**
- 用 anti 闸门的 regime 判定（`_bear` / MA120 + DD 回撤 10%）把全量历史拆成牛/熊两段
- 对每个因子分段计算 rank_IC（均值、标准差、ICIR）
- 检查：是否有因子的 IC 符号在牛熊间反转（牛市正、熊市负，或反之）？
- 输出：`regime_ic_diagnosis.csv`，含每个因子的牛 IC、熊 IC、符号反转标记

**判据：**
- 若 ≥30% 因子有 regime 符号反转 → Phase 2 值得做
- 若因子 IC 只是量级变化（牛市大、熊市小，但符号同）→ regime 分训帮助有限，优先做基本面选股

### Phase 2: Rolling Regime-Conditioned WFA

**核心思路：** 在现有滚动 WFA 框架上，每折根据当前市场状态选匹配的因子冻结集

**流程：**
```
每折 fold k:
  1. split_date = 当年末（如 2023-12-31）
  2. 用 split_date 前的上证综指数据，判定测试期 regime（bull/bear/osc）
  3. 从历史 IC 中筛选 regime 匹配的观测期 → 只取同 regime 期的 IC
  4. 冻结：IS_IC > 0 且 IS_ICIR > 0（在匹配 regime 内）
  5. 用该 regime 的冻结因子跑历史回测 + OOS 测试
```

**Regime 判定（防前视）：**
- 每折仅用 split_date 前数据计算 MA120 位置和滚动回撤
- 上证综指 `load_market_index('sh000001')`
- 判定逻辑与 anti 闸门内 `_bear` 一致：净值破 MA120 或高点回撤 >10% → 熊

**因子切换：**
- 闸门实时监测 regime → 自动装载对应 regime 的冻结因子集
- bull 因子集用于牛市期选股，bear 因子集用于熊市期选股
- 震荡期：用全量因子（不筛选 regime）

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 每 regime 仅 4-6 年 IC 样本过薄 | 按月/季切 regime → 每状态 50-100 月观测点；可选"共享核心 + regime 增量"（全量冻结基础因子 + regime 额外加选） |
| 年份级 regime 分类模糊（2015 前牛后崩） | 用月频或季频 regime 标签替代年级 |
| 闸门与因子切换不同步（闸门调仓、因子切换不同频） | 因子切换频率与闸门再平衡频率对齐（都按 rebalance_dates 驱动） |
| 过度复杂，过拟合风险 | Phase 1 先验证因子确有 regime 反转；Phase 2 用 bootstrap/MC 检验稳定性 |

## 预期收益

- 2024 折从 -0.16 回升到正（牛市因子集取代熊市因子集）
- 闸门拖累缩小（正确的因子 + 正确的仓位）
- 历史 WFA 综合 Sharpe 从 0.615 向上改善

## 文件清单

- 诊断脚本：`regime_ic_diagnosis.py`（Phase 1）
- 引擎改动：`oos_engine.py` 新增 `regime_factor_sets` 字典 + `run_backtest` 动态切换
- WFA 脚本：扩写 `wfa_fullmarket_alldelisted.py` 接入 regime 条件
- 输出：`regime_ic_diagnosis.csv` / `wfa_regime_conditioned.json`
