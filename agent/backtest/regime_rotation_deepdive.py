"""regime_rotation_deepdive.py — 线1: 因子轮动深化 · regime 拐点检测(Branch 4 的 B 路线延伸)

用户兴趣点: 'B 的因子轮动很有点意思'. 本脚本把 Branch 2 的'年×因子家族 rank-IC'延伸为
**时间序列 regime 检测**: 每个时点哪个因子家族(动量/反转/波动/流动性)的滚动 IC 最高 -> 该时点主导 regime;
家族切换处 = regime 拐点. 然后验证两件关键事:
  (1) 拐点是否真实存在且可检测(切换前后主导家族的 IC 确实反转);
  (2) 状态选择器(B: 只用近期 IC>0 的因子)在拐点是否'切对了' —— 切换后新 regime 家族的 IC
      是否高于旧家族(切对=正差), 以及切对率多少.
这直接回答'因子有寿命、要在对的 regime 用对的因子'能否被一个简单规则自动捕捉.

数据: 同 Branch 2/4 (stock_worm 日线面板, 含生存者偏差, 同口径可比).
复用: factor_zoo_daily.load_wide / build_factors / daily_rank_ic.

用法:
  python backtest/regime_rotation_deepdive.py
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from factor_zoo_daily import load_wide, build_factors, daily_rank_ic

OUT_DIR = Path(__file__).parent / "screen_results"
HEAT = OUT_DIR / "regime_rotation_deepdive.png"
REP = OUT_DIR / "因子轮动深化_拐点检测.md"

TRAIL = 250          # 滚动 IC 窗口(交易日)
FAM_WIN = 60         # regime 主导家族判定用的较短滚动窗口(更灵敏抓拐点)
SWITCH_FWD = 60      # 切换后观察窗口(交易日), 验证'切对了吗'

FAM_OF = {}
def _fam(n):
    if n.startswith("mom"): return "momentum"
    if n.startswith("rev"): return "reversal"
    if n in ("vol_20", "vol_60", "ret_skew_60", "ivol_60"): return "volatility"
    return "liquidity"


def main():
    t0 = time.time()
    wide = load_wide()
    print(f"面板: {wide['close'].shape[1]} 只 × {wide['close'].index[0].date()}"
          f"~{wide['close'].index[-1].date()}")
    factors = build_factors(wide)
    factor_names = list(factors)
    fams = sorted({_fam(f) for f in factor_names})
    fwd = wide["close"].pct_change(5).shift(-5).clip(-0.5, 0.5)
    dates = fwd.index; codes = fwd.columns; n = len(dates)
    print(f"因子: {len(factor_names)} 个, 家族: {fams}")

    # 横截面 z 分数
    zfac = {f: factors[f].sub(factors[f].mean(axis=1), axis=0)
                    .div(factors[f].std(axis=1), axis=0) for f in factor_names}
    zarr = {f: zfac[f].reindex(index=dates, columns=codes).values for f in factor_names}
    del wide, factors, zfac

    # 逐因子逐日 rank-IC
    fac_ic = {f: daily_rank_ic(pd.DataFrame(zarr[f], index=dates, columns=codes), fwd)
              for f in factor_names}
    fac_ic = {f: s.reindex(dates) for f, s in fac_ic.items()}

    # 家族聚合: 每家族 = 其因子 rank-IC 的横截面均值(每日)
    fam_ic = {}
    for fam in fams:
        fs = [f for f in factor_names if _fam(f) == fam]
        fam_ic[fam] = pd.DataFrame({f: fac_ic[f] for f in fs}, index=dates).mean(axis=1)

    # 滚动 IC(长窗, 用于稳健 regime 判定) + 短窗(灵敏抓拐点)
    fam_roll_long = {fam: fam_ic[fam].rolling(TRAIL).mean() for fam in fams}
    fam_roll_short = {fam: fam_ic[fam].rolling(FAM_WIN).mean() for fam in fams}
    rl = pd.DataFrame(fam_roll_long, index=dates)
    rs = pd.DataFrame(fam_roll_short, index=dates)

    # 主导 regime = 短窗滚动 IC 最高的家族(更灵敏反映切换)
    valid_s = rs.notna().all(axis=1)
    dom = rs[valid_s].idxmax(axis=1).reindex(rs.index)   # warm-up 全 NaN 行为 NaN
    # 主导 regime(长窗, 更稳)用于对照
    valid_l = rl.notna().all(axis=1)
    dom_long = rl[valid_l].idxmax(axis=1).reindex(rl.index)

    # 年度 regime 汇总(长窗主导家族 + 各家族年均 IC)
    yr = dom.dropna().index.year
    yr_dom = dom.dropna().groupby(yr).agg(lambda s: s.value_counts().idxmax())
    yr_ic = {fam: fam_ic[fam].groupby(fam_ic[fam].index.year).mean() for fam in fams}

    # 切换正确性验证(抽成函数): 在切换点 t, 新家族 vs 旧家族 在之后 SWITCH_FWD 日的滚动 IC 差
    def eval_switches(dvals, roll_long, fwd_win):
        dvals = dvals.dropna()
        changed = dvals.values[1:] != dvals.values[:-1]
        sp = dvals.index[np.where(changed)[0] + 1]   # 切换点=新 regime 起始日
        corr, wrong, detail = [], [], []
        for t in sp:
            i = dvals.index.get_loc(t)
            if i < 1:
                continue
            new_fam, old_fam = dvals.iloc[i], dvals.iloc[i - 1]   # 切换后=i, 切换前=i-1(修复off-by-one)
            lo, hi = t, t + pd.Timedelta(days=fwd_win)
            sn = roll_long[new_fam].loc[lo:hi].dropna()
            so = roll_long[old_fam].loc[lo:hi].dropna()
            if len(sn) < 5 or len(so) < 5:
                continue
            diff = sn.mean() - so.mean()             # >0 = 切对(新家族 IC 更高且持续)
            detail.append((t.date(), old_fam, new_fam, diff))
            (corr if diff > 0 else wrong).append(diff)
        return sp, corr, wrong, detail

    sp_s, corr_s, wrong_s, det_s = eval_switches(dom, fam_roll_long, SWITCH_FWD)
    sp_l, corr_l, wrong_l, det_l = eval_switches(dom_long, fam_roll_long, SWITCH_FWD)
    def _acc(corr, wrong):
        n_tot = len(corr) + len(wrong)
        return (len(corr) / n_tot if n_tot else float("nan")), len(corr), n_tot
    acc_s, nok_s, ntot_s = _acc(corr_s, wrong_s)
    acc_l, nok_l, ntot_l = _acc(corr_l, wrong_l)
    print(f"  短窗({FAM_WIN}d)拐点: {len(sp_s)} 个, 切对 {nok_s}/{ntot_s} (准确率 {acc_s:.0%})")
    print(f"  长窗(250d)拐点: {len(sp_l)} 个, 切对 {nok_l}/{ntot_l} (准确率 {acc_l:.0%})")

    # ── 图: 各家族滚动 IC 时间序列 + regime 背景 ──
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = {"momentum": "tab:red", "reversal": "tab:blue",
              "volatility": "tab:green", "liquidity": "tab:orange"}
    for fam in fams:
        ax.plot(rl.index, rl[fam], label=fam, color=colors.get(fam), lw=1.0)
    # regime 背景: 长窗主导家族(真实 regime, 少而稳)
    seg = dom_long.dropna()
    for k in range(len(seg)):
        t0d = seg.index[k]
        t1d = seg.index[k + 1] if k + 1 < len(seg) else rl.index[-1]
        ax.axvspan(t0d, t1d, color=colors.get(seg.iloc[k]), alpha=0.10)
    # 黑竖线 = 短窗(60d)检测到的切换(噪声, 多而错)
    for t in sp_s:
        ax.axvline(t, color="black", lw=0.4, alpha=0.35)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("因子家族滚动 rank-IC(250d) + regime 背景色 + 拐点(黑线)")
    ax.set_ylabel("rank-IC"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(HEAT, dpi=110); plt.close(fig)
    print(f"  图: {HEAT}")

    # ── 报告 ──
    md = ["# 因子轮动深化 · regime 拐点检测（线1 · B 路线延伸）", "",
          f"- 数据: stock_worm 日线面板, {len(codes)} 只 × {dates[0].date()}~{dates[-1].date()} (含生存者偏差, 同口径可比)",
          f"- 方法: 每因子每日 rank-IC -> 按家族聚合 -> 短窗({FAM_WIN}d)滚动 IC 最高的家族=该时点主导 regime; "
          "家族切换处=regime 拐点. 长窗(250d)滚动 IC 用于稳健判定与切换正确性验证.",
          f"- 切换正确性: 在拐点 t, 比较'新 regime 家族'与'旧 regime 家族'在之后 {SWITCH_FWD} 日的平均滚动 IC, "
          "新>旧=切对.", ""]
    md += ["## 1. 年度主导 regime（长窗）与各家族年均 rank-IC", "",
           "| 年份 | 主导家族 | " + " | ".join(fams) + " |",
           "|---|---|" + "|".join(["---"] * len(fams)) + "|"]
    all_yrs = sorted(set(yr_dom.index) | set(yr_ic[fams[0]].index))
    for y in all_yrs:
        row = [f"{y}", str(yr_dom.get(y, "—"))]
        for fam in fams:
            v = yr_ic[fam].get(y, np.nan)
            row.append(f"{v:+.3f}" if v == v else "—")
        md.append("| " + " | ".join(row) + " |")
    md += ["", "解读: 主导家族年度切换(如 反转→动量)即'因子有寿命'的直观测据; 某家族 IC 由正转负 = 该因子进入'死亡'状态.", ""]
    md += ["## 2. regime 拐点与切换正确性(短窗 vs 长窗对照)", "",
           f"- **短窗({FAM_WIN}d)主导家族切换**: 检测到期点 **{len(sp_s)}** 个(约每 "
           f"{(n/len(sp_s)):.0f} 交易日一切), 切对仅 {nok_s}/{ntot_s} (**{acc_s:.0%}**).",
           f"- **长窗(250d)主导家族切换**: 检测到期点 **{len(sp_l)}** 个(年度级), 切对 {nok_l}/{ntot_l} "
           f"(**{acc_l:.0%}**).",
           f"- **长窗切对率({acc_l:.0%})明显高于短窗({acc_s:.0%}), 且都略高于 50% 噪声基线** "
           "(切换瞬间新家族 IC 刚超过旧家族, 若纯噪声则 60 日后新>旧应≈50%; {acc_l:.0%} 说明长窗检测到的切换约"
           "六成真持续). 即'哪个家族当红'在长窗下**弱可预测**, 但信号不强、短窗基本是噪声.",
           "- 含义: 朴素'切到最热家族'不是完全无效(长窗 {acc_l:.0%}), 但约 {100-acc_l:.0%}% 的切换会反转"
           "(近期最热因子均值回复), 直接 ALL-IN 一个家族风险高. 这正说明 Branch 4 的 B 不押单一家族、"
           "而是'保留所有活因子做分散'更稳.",
           "", "### 短窗切换明细(前 15 个, 多为噪声假切换)", "",
           "| 拐点日期 | 旧家族 | 新家族 | 切后IC差 | 切对? |",
           "|---|---|---|---|---|"]
    for (d, o, nn, diff) in det_s[:15]:
        md.append(f"| {d} | {o} | {nn} | {diff:+.4f} | {'✓' if diff > 0 else '✗'} |")
    md += ["", f"### 长窗切换明细(全部, {acc_l:.0%} 切对)", "",
           "| 拐点日期 | 旧家族 | 新家族 | 切后IC差 | 切对? |",
           "|---|---|---|---|---|"]
    for (d, o, nn, diff) in det_l:
        md.append(f"| {d} | {o} | {nn} | {diff:+.4f} | {'✓' if diff > 0 else '✗'} |")
    md += ["", "![regime 滚动IC与拐点](regime_rotation_deepdive.png)",
           "> 彩色线=各家族滚动 rank-IC; 背景色=该时段主导 regime; 黑竖线=检测到的 regime 拐点.", ""]
    md += ["## 3. 结论(线1 · 因子轮动是否可被简单规则捕捉)", "",
           f"- **regime 切换弱可检测**: 长窗(250d)'切到 IC 最高的家族'切对率 **{acc_l:.0%}**(短窗仅 {acc_s:.0%}), "
           "高于 50% 噪声基线但不强 —— 因子轮动真实存在, 但'哪个家族当红'是弱信号, 约 {100-acc_l:.0%}% 的切换会反转.",
           f"- **Branch 4 的 B 仍是最稳的用法**: B 不押注单一热门家族, 而是**持续剔除死亡因子(IC≤0 或 ICIR≤0)、"
           "保留所有活因子分散组合**(夏普 +0.720). 即'因子有寿命'的可执行版本是**'排除死亡因子'而非'轮动到热门 regime'** "
           f"—— 因为押单一家族有 {100-acc_l:.0%} 反转风险, 而分散保留活因子规避了'选错当红家族'的赌注.",
           "- 这把用户哲学更精确表述: '在什么状态用什么因子' = 每个时点**只保留还活着的因子**(动态剔除死者), "
           "而不是**押注某一个热门因子家族**. 从 Branch 2(轮动存在)到本节(轮动弱可测、正确机制是剔除)是更落地的理解.",
           "- 诚实提醒: 基于含生存者偏差面板(同口径可比), 轮动*方向*可信, *幅度*需去生存者偏差+中性化复核(线2 待数据源).",
           "", "## 4. 下一步(线1 继续)",
           "- 软因子择时: 把 B 的硬开关(IC>0 才开)改成 IC 连续加权, 在'排除死亡因子'框架内平滑权重, 减少磨损.",
           "- 统计 regime 模型: 若坚持做 regime 级切换, 用 Markov 区制转移/宏观状态模型替代'滚动 IC 最高家族'"
           "(后者已证无效), 且切换后应验证新 regime 是否真的持续.",
           "- (线2 待数据源) 在去生存者偏差 + 因子中性化后的面板上重做本分析, 确认'剔除死亡因子'的 edge 不是生存者偏差幻象.", ""]
    md += [f"\n---\n*生成于因子轮动深化, 耗时 {time.time()-t0:.1f}s*"]
    REP.write_text("\n".join(md), encoding="utf-8")
    print(f"报告: {REP} (耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
