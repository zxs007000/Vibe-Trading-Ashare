# 圆桌议题：Vibe-Trading-Ashare Phase 1-3 完成后的下一步优先方向

> 数据口径：市场状态实时（2026-07-08 20:13 拉取 `/dashboard/market-state`）；Phase 3 回测结果来自今日 18:53 运行的 `agent/backtest/verify_phase3.py`（40 只蓝筹、约 600 交易日、目标波动率 15%）；因子实现证据来自本轮 20:13 读取的 `sentiment_score.py` / `quality_roe_change.py` / `verify_phase3.py` 源码。
> 结论标注：研究辅助、非投资建议。

## 角色
🎙️ 主持人 · 👍 赞成者 · 👎 反对者 · 📊 数据党 · 📈 量化研究员 · 🏦 基本面分析师 · 🛡️ 风控官

## 第 1 轮 · 立场

**📊 数据党（先放真实数据）**
1. **实时市场状态**（20:13）：`state = pullback / 冲高回落`，置信 0.72，自 2026-06-19 起已 19 天；`close 4792 < ma60 4843`（ma60 仍在 ma250 4576 之上 → 中期多头排列但短期回落）；`ret5_pct = -3.8%`（近 5 日下跌）、`ret20_pct = +1.4%`（近 20 日仍微正）；年化波动 26%。→ 当前是"冲高后回落"的回调段，不是趋势市。
2. **Phase 3 回测结果**（今日 verify_phase3.py）：复合 Sharpe **0.520** / MaxDD **-14.8%** / Calmar 为正；单链最佳 **quality_momentum（vol-targeted）Sharpe 0.830**。
3. **情感因子实现证据**（读 `sentiment_score.py`）：`compute()` 注释明确写"回测时全期填入同一值"——40 只股票每天填同一个静态情感分 → 回测里情感维度的时间序列信息 = 0，只有截面一次性差异被利用。
4. **ROE 因子实现证据**（读 `quality_roe_change.py`）：`roe.diff(periods=1)` 作用在"季报前向填充"的 ROE 面板上 → 相邻日几乎恒为 0，仅季报切换日跳变 → 截面变化极小，解释了 Phase 2 里 ROE 精炼因子 IC≈0。

**👍 赞成者**：单链 0.83 真实、数据底座已通。建议优先 **① 产品化封装 quality_momentum(vol-targeted)**——确定性最高，且当前回调市正是波动率缩放产品最有用的时候，能立刻交付。

**👎 反对者**：0.83 是 40 只大蓝筹窄样本内的"最佳链"，没做 walk-forward / 样本外，可能过拟合；且市场状态是 pullback，动量因子在回调段易回撤，单押动量有尾部风险。

**📈 量化研究员**：情感 IC **0.233** / IR **1.33** 是今日 `verify_sentiment_factors.py` 实测的真 alpha。但当前回测把情感当静态截面量，把 IC 白白浪费了。方向 ② 把情感做成"当日截面"能兑现 IC，代价是变成 **live-only 信号**（前视偏差，无法回测验证）——这是"收益 vs 可验证性"的权衡。

**🏦 基本面分析师**：ROE 在 A 股长期有 alpha 但边际递减；`quality_roe_change` 日频 diff 设计有缺陷（季报填充后相邻日恒为 0）。方向 ③ 改月频 + 行业中位差是对的修法，但 ROE 方向天花板有限，优先级应低于情感。

**🛡️ 风控官**：当前 pullback，market_state 闸门应降仓（recommended 首项 `value_qlowvol` score 0.5 = neutral）。任何"下一步"都要尊重闸门。单链产品化若不加闸门，0.83 在回调段会回吐。事件信号（④）龙虎榜只有 live 无历史 API，补历史数据源不确定、成本高、且有信息滞后风险。

**🎙️ 主持人定调**：核心问题 = 在「回调市 + 已有单链 0.83 + 情感真 alpha 被浪费 + ROE 设计缺陷」四张牌下，下一步该把资源押在 **"确定性交付"** 还是 **"上限突破"**？

## 第 2 轮 · 交锋

**👎 反对者 反驳 赞成者**：0.83 是窄样本，quality_momentum 本质是"高质量 + 动量"，在 pullback 段动量会反过来伤人。先封装产品再发现样本外崩，不如先把情感 live 信号接上（真 IC 0.233）做增强。

**📊 数据党 补刀**：market_state 返回 pullback，recommended 首项 `value_qlowvol` 仅 0.5（neutral）。系统自己判断现在是"中性偏弱"，应半仓而非满仓。给单链加闸门后，回撤上限能进一步压住。

**📈 量化研究员 反驳 反对者**：情感 IC 0.233 是真，但"当日截面化"后你没法回测它，等于把可验证 alpha 变成黑箱 live 信号——产品化风险更高。更稳的路：① 先把质量动量波动率产品做扎实（含闸门），② 再用情感 live 信号做 **overlay**（小权重增强），不进核心回测。

**🏦 基本面 补充 ③**：ROE 改月频行业中位差是必要修，但只建议在因子库维护时顺手做，不单列成一轮重点——ROE 边际 alpha 已被验证有限（Phase 2 ROE 精炼 IC≈0）。

**🛡️ 风控官 收口**：四个方向里，④ 补事件历史成本/不确定性最高（龙虎榜无历史 API），① 确定性最高。建议 **①为主、②为 stage-2 overlay、③④顺手维护、④暂缓**。

## 第 3 轮 · 裁决

**结论 = 中性偏多（优先 ①，置信 75%）**

- **论据**：单链 0.83 是今日真实回测最佳；当前 pullback 市波动率缩放产品正当时；情感 IC 0.233 真但只能 live 验证，适合做 overlay 而非核心。
- **建议路径**：
  - **P1（本阶段）**：封装 `quality_momentum(vol-targeted)` 为独立可交付产品，**强制挂 market_state 闸门**（pullback 半仓），输出净值 / 回撤 / 夏普监控面板。
  - **P2（stage-2）**：情感"当日截面"信号做成 **live overlay**（小权重，不进回测核心），需实时新闻管道 + 截面 z-score。
  - **P3（维护）**：ROE 因子改月频行业中位差（修设计缺陷）。
  - **暂缓**：④ 事件历史数据（数据源缺失，ROI 低）。
- **主要风险**：0.83 窄样本过拟合；pullback 段动量回撤；情感 live 信号无回测背书。
- **后续验证**：对 ① 做 walk-forward 样本外（扩到全市场 500+ 只，分 2019-2021 训练 / 2022-2024 测试）；对 ② 先做 1 个月 live 纸面跟踪再决定是否加权。

## 数据附录
- 市场状态：2026-07-08 20:13 `GET /dashboard/market-state` → pullback / conf 0.72 / close 4792 / ma60 4843 / ma250 4576 / ret5 -3.8% / ret20 +1.4%。
- Phase 3 回测：`agent/backtest/verify_phase3.py`，40 只蓝筹、约 600 交易日、目标波动率 15%、63 天滚动窗、EMA(0.06) 平滑、杠杆 [0.1, 2.0]；复合 Sharpe 0.520 / MaxDD -14.8%；单链最佳 quality_momentum Sharpe 0.830。
- 因子源码（本轮 20:13 读取）：`agent/src/factors/zoo/sentiment/sentiment_score.py`（回测全期同值）、`agent/src/factors/zoo/fundamental/quality_roe_change.py`（日频 diff 对季报填充面板）、`agent/backtest/verify_phase3.py`（8 原始因子 + 4 增强链）。
- 情感 IC 0.233 / IR 1.33 来自今日 `verify_sentiment_factors.py`。

> ⚠️ 研究辅助，非投资建议。
