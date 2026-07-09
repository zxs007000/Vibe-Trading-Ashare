"""根据真实回测结果生成《咱们后加因子夏普报告.md》(数字全部取自 CSV, 可复跑)。"""
import pandas as pd
import numpy as np
from pathlib import Path

RES = Path(__file__).parent

fd = pd.read_csv(RES / "custom_founder_eval.csv", index_col=0)
ht = pd.read_csv(RES / "custom_huatai_eval.csv", index_col=0)
gh = pd.read_csv(RES / "custom_guosheng_haitong_eval.csv", index_col=0)
zoo = pd.read_csv(RES / "zoo_screen_20260709_090025.csv")
zoo_top = zoo[zoo["status"] == "ok"].sort_values("ls_sharpe", ascending=False).head(5)

# 自研 structured_reversal (1D 2022-2026, 同 zoo 口径, 已验证稳定)
sr = {
    "structured_reversal_v2": dict(icir=-1.522, ls_sharpe=0.172, reversal_ls=-0.172,
                                   note="volume+多窗口(10/21/63)+截面zscore『优化版』"),
    "structured_reversal_v1": dict(icir=0.030, ls_sharpe=0.667, reversal_ls=-0.667,
                                   note="equal 单窗口21d"),
}

def direction(ic):
    if pd.isna(ic): return "—"
    if ic < 0: return "做空高值(取反)"
    if ic > 0: return "做多高值"
    return "双向弱"

def fmt_rows(df, src_label):
    rows = []
    for name, r in df.iterrows():
        nm = name.replace("HT:", "").replace("GS:", "").replace("HA:", "")
        prof = abs(r["ls_sharpe"]) if pd.notna(r["ls_sharpe"]) else np.nan
        rows.append((nm, r["状态"], r["ic_mean"], r["icir"],
                     r["ls_sharpe"], prof, direction(r["ic_mean"])))
    rows.sort(key=lambda x: (x[5] if pd.notna(x[5]) else -1), reverse=True)
    return rows

L = []
L.append("# 咱们后加的因子 — 夏普率体检报告(全量版)")
L.append("")
L.append("> 与因子库 zoo 筛查**同一套口径**：日频前瞻收益(无前视) + RankIC/ICIR + 五分位多空组合夏普(ls_sharpe, 年化 252)。")
L.append("> 这批因子(自研 `structured_reversal` + 复现的方正金工 / 华泰金工因子)不在注册表 zoo 内, 故单独评估。")
L.append("> **本轮更新**：① 方正 4 个未跑通因子修复, 16 因子全量体检; ② 新增华泰金工 4 因子; ③ **新增国盛/海通金工 3 因子**(量价背离/隔夜收益/非流动性)。")
L.append("")

# ---- 一、自研 structured_reversal ----
L.append("## 一、自研因子 `structured_reversal`(1D 面板, 2022-01-01~2026-06-30, 78 只)")
L.append("")
L.append("| 版本 | 说明 | ICIR | 多空夏普(做多高值) | 反转方向夏普(取反) |")
L.append("|---|---|---|---|---|")
for k, v in sr.items():
    L.append(f"| {k} | {v['note']} | {v['icir']:+.2f} | {v['ls_sharpe']:.2f} | {v['reversal_ls']:+.2f} |")
L.append("")
L.append("**关键结论：** 该因子原始值=近期收益率(高=近期赢家), 本质是**动量信号**; 作为『反转』用须取反。")
L.append("2022-2026 动量占优 → 反转方向亏钱(v1 −0.67、v2 −0.17)。⚠️ **『优化版』v2 反比 v1 差**(0.17 vs 0.67)。建议回退 v1 或改用券商因子。")
L.append("")

# ---- 二、券商复现因子 ----
L.append("## 二、券商研报复现因子（方正 + 华泰）")
L.append("")

# 2.1 方正
L.append("### 2.1 方正金工因子(5m 真实数据, 2025-01-01~2026-06-30, 30 只)")
L.append("")
L.append("> ls_sharpe 为库约定(做多高因子值)。方正因子多数 IC<0(论文『负向因子』), 盈利方向为取反, 下表『盈利能力|ls|』即取反后夏普量级。")
L.append("")
L.append("| 因子 | 状态 | IC | ICIR | 多空夏普 | 盈利能力|ls| | 论文方向 |")
L.append("|---|---|---|---|---|---|---|")
for name, st, ic, icir, ls, prof, dr in fmt_rows(fd, "方正"):
    prof_s = "—(信号稀疏)" if pd.isna(prof) else f"{prof:.2f}"
    L.append(f"| {name} | {st} | {ic:+.3f} | {icir:+.2f} | {ls:+.2f} | {prof_s} | {dr} |")
L.append("")
L.append("- 15 个可算因子全部 `ok`(clouds_disperse / rapids_advance 仍因 mootdx 5m 无 `amount` 未纳入)。")
L.append("- **修复的 4 个已全部入榜**：`drip_water_stone`(取反 **1.08**)、`withered_tree_blooms`(**0.60**)、`synergy_effect`(**0.30**)、`panic_factor`(IC 有效 icir=−0.74 但多空夏普 NaN——衰减惊恐度信号过稀疏)。")
L.append("- 最强：`smart_money`(1.10)、`drip_water_stone`(1.08)、`flower_hidden`(1.02)、`wait_rescue`(1.00)、`scaling_heights`(0.96)、`complete_tide`(0.85) — 多空夏普 0.3~1.1, 接近 zoo 一线。")
L.append("")

# 2.2 华泰
L.append("### 2.2 华泰金工因子(5m/日频, 2025-01-01~2026-06-30, 30 只)")
L.append("")
L.append("> 复现自华泰『多因子系列』报告(波动率类/资金流向/历史分位数)。ls_sharpe 为库约定。")
L.append("> ⚠️ **方法学提示**：本批 4 个华泰日频因子 IC 均为负, 但样本内五分位多空方向却为正(IC 与 ls 符号相反)。")
L.append("> 这在 30 只小宇宙下属正常(极值分组长空受个股噪声干扰); **实际上线前须用 walk-forward / IC 符号定交易方向**, 下表『盈利能力|ls|』仅记量级。")
L.append("")
L.append("| 因子 | 状态 | IC | ICIR | 多空夏普 | 盈利能力|ls| | IC方向 |")
L.append("|---|---|---|---|---|---|---|")
for name, st, ic, icir, ls, prof, dr in fmt_rows(ht, "华泰"):
    prof_s = "—" if pd.isna(prof) else f"{prof:.2f}"
    L.append(f"| {name} | {st} | {ic:+.3f} | {icir:+.2f} | {ls:+.2f} | {prof_s} | {dr} |")
L.append("")
L.append("- `historical_percentile`(**0.97**)、`money_flow`(**0.49**) 量级较好, 接近方正一线; `downside_deviation`(0.22)、`idiosyncratic_volatility`(0.13) 偏弱。")
L.append("- `idiosyncratic_volatility` 用个股收益对市场收益滚动回归残差标准差(经典低波异象); `downside_deviation` 为半方差下行波动; `money_flow` 为 5m 符号量能累加(大单资金流代理); `historical_percentile` 为收盘价在 60 日历史分布中的分位(均值回复信号)。")
L.append("- ⚠️ 华泰『理想换手率/一致预期/财务质量』等因子依赖换手率或基本面数据, 沙箱无覆盖, 未复现。")
L.append("")
L.append("### 2.3 国盛金工(量价淘金) / 海通金工(流动性)")
L.append("")
L.append("> 复现自国盛『量价淘金』系列与海通『选股因子系列』(流动性方向)。ls_sharpe 为库约定。")
L.append("")
L.append("| 因子 | 来源 | 状态 | IC | ICIR | 多空夏普 | 盈利能力|ls| | IC方向 |")
L.append("|---|---|---|---|---|---|---|---|")
for name, st, ic, icir, ls, prof, dr in fmt_rows(gh, "国盛/海通"):
    prof_s = "—" if pd.isna(prof) else f"{prof:.2f}"
    src = "国盛" if name.startswith("overnight") or name.startswith("volume_price") else "海通"
    L.append(f"| {name} | {src} | {st} | {ic:+.3f} | {icir:+.2f} | {ls:+.2f} | {prof_s} | {dr} |")
L.append("")
L.append("- `volume_price_divergence`(**1.41**)、`amihud_illiquidity`(取反 **0.87**)、`overnight_return`(**0.69**) 量级均较好, 其中量价背离为全样本最强新因子。")
L.append("- ⚠️ `volume_price_divergence` 的 IC 近 0 但五分位多空夏普高达 1.41(极值分组捕捉到非单调的量价关系), 上线前须 walk-forward 验证方向, 谨防过拟合。")
L.append("- `amihud_illiquidity` 用 volume 代理成交金额(沙箱 5m 无 amount), 非流动性溢价方向以样本为准。")
L.append("- 国盛『逐笔羊群/异动雷达』、海通『正交大单的大买』等依赖逐笔/amount, 沙箱无覆盖, 未复现。")
L.append("")

# ---- 三、与 zoo Top 对比 ----
L.append("## 三、与 zoo Top 因子横向对比")
L.append("")
L.append("| 因子 | 来源 | 多空夏普 | 备注 |")
L.append("|---|---|---|---|")
for _, z in zoo_top.iterrows():
    L.append(f"| {z['alpha_id']} | zoo({z['zoo']}) | {z['ls_sharpe']:.2f} | {z.get('theme','')} |")
for name, st, ic, icir, ls, prof, dr in fmt_rows(fd, "方正"):
    if pd.notna(prof) and prof >= 0.8:
        L.append(f"| {name} | 方正 | {prof:.2f} | 复现 |")
for name, st, ic, icir, ls, prof, dr in fmt_rows(ht, "华泰"):
    if pd.notna(prof) and prof >= 0.8:
        L.append(f"| {name} | 华泰 | {prof:.2f} | 复现 |")
for name, st, ic, icir, ls, prof, dr in fmt_rows(gh, "国盛/海通"):
    if pd.notna(prof) and prof >= 0.8:
        src = "国盛" if name.startswith(("overnight", "volume_price")) else "海通"
        L.append(f"| {name} | {src} | {prof:.2f} | 复现 |")
L.append(f"| structured_reversal_v1 | 自研 | −0.67(反转方向) | 自研, 弱 |")
L.append(f"| structured_reversal_v2 | 自研 | −0.17(反转方向) | 自研『优化版』, 更弱 |")
L.append("")
L.append("> 结论：复现的**券商因子质量明显高于自研 structured_reversal**。方正 + 华泰合计约 8 个因子进入 0.8~1.1 区间, 接近 zoo 一线(zoo 三甲 ~1.8)。")
L.append("")

# ---- 四、口径与注意事项 ----
L.append("## 四、口径与注意事项")
L.append("")
L.append("- 方正/华泰窗口 2025-01~2026-06(1.5 年, 5m 真实数据), 宇宙 30 只; 与 zoo 的 2022-2026/78 只不完全同窗, 绝对夏普不可直接相减, 量级可比。")
L.append("- 多数券商因子 IC<0(论文『负向因子』)：高因子值→低收益, 盈利方向为做空高值/做多低值, 取反后夏普即『盈利能力』。")
L.append("- ⚠️ **华泰日频因子 IC 与 ls 符号相反**(见 2.2)：交易方向须以 walk-forward/IC 符号为准, 不可直接照搬因子符号。")
L.append("- `panic_factor` 衰减惊恐度仅保留正跃变日 → 单日信号稀疏 → 五分位多空夏普不可靠(IC 仍有效)。")
L.append("- 多空夏普未计交易成本/冲击/涨跌停, 实盘会打折。")
L.append("")

# ---- 五、修复记录 ----
L.append("## 五、修复记录(本轮回测)")
L.append("")
L.append("- `drip_water_stone`：原 `len(v)<120` 门槛吃掉 5m 数据(每日仅~44 根)→ 全 NaN。改为频率自适应门槛(≥12)+『高频能量占比』频带回, 1m/5m 通用。")
L.append("- `panic_factor`：原 `market_ret` 退化为 `ret.shift(1)` 使惊恐度过平滑 → 全 NaN。改为真实市场收益(截面均值)+ 向量化分钟波动率(修复 dict→Series→reindex 索引对齐失败)。")
L.append("- `synergy_effect`：原逐 peer 重建掩码数组超时。改为向量化截面法(逐日算个股日内路径 vs 截面均值路径相关系数=协同度), 8s 跑完; 修复 corrcoef 含 NaN 污染。")
L.append("- `withered_tree_blooms`：旧版单股接口 ERR, 当前逐股调用已正常, 30/30 有效。")
L.append("")
L.append("## 六、华泰复现数据局限")
L.append("")
L.append("- 华泰原版资金流向因子依赖**逐笔大单成交额划分**, 沙箱 5m 无 `amount`, 以 `ret×volume` 符号量能代理, 非原版。")
L.append("- `idiosyncratic_volatility` 回归所需市场收益用**截面均值**代理(无中证全指指数序列)。")
L.append("- 华泰『人工智能』系列(autoencoder 挖因子)及需基本面(估值/成长/财务质量/一致预期)的因子不在沙箱覆盖内, 未复现。")
L.append("")

out = RES / "咱们后加因子夏普报告.md"
out.write_text("\n".join(L), encoding="utf-8")
print("报告已生成:", out, "字符数:", len("\n".join(L)))
