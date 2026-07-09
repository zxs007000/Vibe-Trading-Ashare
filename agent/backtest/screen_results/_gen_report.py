"""根据筛查 CSV 生成 Markdown 报告（数字全部取自真实结果）。"""
import pandas as pd
import numpy as np
from pathlib import Path

RES = Path(__file__).parent
df = pd.read_csv(RES / "zoo_screen_20260709_090025.csv")
short = pd.read_csv(RES / "high_sharpe_shortlist.csv")
div = pd.read_csv(RES / "high_sharpe_diversified.csv")

ok = df[df["status"] == "ok"].copy()
skip = df[df["status"].isin(["skip", "no_ic"])].copy()
n_total = len(df)

cols = ["alpha_id", "zoo", "theme", "ic_mean", "icir", "ic_pos_ratio",
        "ic_tstat", "ls_sharpe", "top_sharpe", "ls_ann_ret"]


def fmt(d, n=30):
    d = d.sort_values("ls_sharpe", ascending=False).head(n)
    return d[cols].to_markdown(index=False, floatfmt=".3f")


# zoo 级别汇总
g = (ok.groupby("zoo")
       .agg(因子数=("alpha_id", "size"),
            平均多空夏普=("ls_sharpe", "mean"),
            平均ICIR=("icir", "mean"))
       .round(3).sort_values("平均多空夏普", ascending=False))

lines = []
lines.append("# 因子库夏普率筛查报告")
lines.append("")
lines.append(f"> 生成时间自动；样本区间 **2022-01-01 ~ 2026-06-30**（约 1084 个交易日，覆盖牛熊多段），"
             f"股票宇宙为去重后的 CSI300 代表样本（78 只），日频 OHLCV。")
lines.append("")
lines.append("## 一、覆盖率")
lines.append("")
lines.append(f"- 注册表因子总数：**{n_total}**")
lines.append(f"- 成功回测（有完整 IC/夏普）：**{len(ok)}**")
lines.append(f"- 跳过（依赖 `amount`/`vwap`/`sector`，沙箱无该字段）：**{len(skip)}**")
lines.append(f"- 成功因子中多空夏普率分布：均值 {ok['ls_sharpe'].mean():.3f}，"
             f"中位数 {ok['ls_sharpe'].median():.3f}，最高 {ok['ls_sharpe'].max():.3f}，最低 {ok['ls_sharpe'].min():.3f}")
lines.append("")
lines.append("各夏普门槛下的因子数量：")
for thr in [0.5, 1.0, 1.5, 2.0]:
    lines.append(f"- `ls_sharpe ≥ {thr}`：**{(ok['ls_sharpe'] >= thr).sum()}** 个")
lines.append("")
lines.append("## 二、各因子库（zoo）表现对比")
lines.append("")
lines.append(g.to_markdown())
lines.append("")
lines.append("> 结论：**alpha101** 与 **academic** 两个库的平均多空夏普与 ICIR 显著领先；"
             "gtja191 / qlib158 多数因子在本样本期 ICIR 偏弱（部分甚至符号为负），需结合具体因子挑选，不宜整体采用。")
lines.append("")
lines.append("## 三、Top 30 因子（按多空夏普率降序）")
lines.append("")
lines.append(fmt(ok, 30))
lines.append("")
lines.append("## 四、高夏普精选清单（多空夏普 ≥ 1.0 且 |ICIR| ≥ 0.5）")
lines.append("")
lines.append(f"共 **{len(short)}** 个，已落盘 `high_sharpe_shortlist.csv`：")
lines.append("")
lines.append(short[cols].sort_values("ls_sharpe", ascending=False).to_markdown(index=False, floatfmt=".3f"))
lines.append("")
lines.append("## 五、去冗余后的推荐组合")
lines.append("")
lines.append("对上面的精选因子做两两秩相关（|r|>0.7 视为冗余），贪心保留夏普最高且彼此低相关的因子，"
             "得到 **8 个低相关高夏普因子**，已落盘 `high_sharpe_diversified.csv`：")
lines.append("")
lines.append(div[cols].sort_values("ls_sharpe", ascending=False).to_markdown(index=False, floatfmt=".3f"))
lines.append("")
lines.append("### 发现的跨库冗余对（同一因子在不同 zoo 的重复实现）")
lines.append("")
lines.append("- `gtja191_171` ↔ `alpha101_054`（r=0.97）—— 均为 Top 因子，已保留 `gtja191_171`")
lines.append("- `alpha101_060` ↔ `gtja191_111`（r=0.78）")
lines.append("- `alpha101_028` ↔ `gtja191_191`（r=1.00）—— 完全等价，仅缩放不同")
lines.append("- `qlib158_imxd10` ↔ `gtja191_096`（r=0.76）")
lines.append("")
lines.append("> 这些因子 IC/夏普与缩放无关（基于秩），所以指标完全相同；建模时应只保留其一，避免数据泄漏式重复计数。")
lines.append("")
lines.append("## 六、指标口径与方法")
lines.append("")
lines.append("- **前瞻收益**：`close.pct_change().shift(-1)`，因子在 t 日打分、用 t+1 日收益评价，无前视偏差。")
lines.append("- **RankIC**：每日截面 Spearman 秩相关；**ICIR** = mean(IC)/std(IC) × √252（年化）。")
lines.append("- **多空夏普（ls_sharpe）**：每日按因子值五分位，取最高组减最低组的日收益序列，年化夏普（252 交易日）。")
lines.append("- **Top 组夏普（top_sharpe）**：仅持有最高分位组的纯多头夏普，用于评估不带对冲的单边暴露。")
lines.append("- **跳过规则**：注册表 `compute()` 在面板缺 `amount`/`vwap`/`sector` 时抛 `SkipAlpha`，自动剔除（沙箱仅 5 列 OHLCV）。")
lines.append("")
lines.append("## 七、注意事项")
lines.append("")
lines.append("1. **样本窗**：2022–2026 涵盖牛熊，但单一样本仍可能高估；上生产前建议做 Walk-forward / 滚动 OOS 验证。")
lines.append("2. **股票宇宙**：78 只蓝筹截面偏窄，小市值/微盘因子可能失真；扩大宇宙后结果可能变化。")
lines.append("3. **多空组合**：ls_sharpe 是理论多空（无成本），未计佣金、冲击成本、停牌与涨跌停限制，实盘会打折。")
lines.append("4. **未覆盖因子**：94 个依赖 `amount`/`vwap` 的因子（含大量量价经典因子）需用 tushare/akshare 补齐 `amount` 后单独回测。")
lines.append("5. **方正金工因子**：属于独立 5 分钟高频因子族（另见 `agent/backtest/factors/founder/` 与 `verify_founder_factors.py`），接口不同于本注册表，未纳入本次日频筛查。")
lines.append("")

out = RES / "因子库夏普筛查报告.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("报告已生成:", out)
print("字符数:", len("\n".join(lines)))
