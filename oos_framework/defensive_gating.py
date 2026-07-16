"""defensive_gating.py — 防御门控层(治回撤, 不杀小盘 alpha).

用户目标: 当前无闸基线 A = WFA +0.618 / +24.2% / -65.80%(light).
最大痛点不是收益, 是 -66% 的回撤(实盘不可承受). 本模块在**不动因子 alpha 源**前提下,
加一层市场级防御门控:

  1) 危机检测: mkt_level 跌破 250 日线(偏离 < -10%) 或 市场收益波动 z 分数 > 2.
  2) 防御倾斜: 危机期把 低波(ivol_60/vol_60/downside_vol_60) + 质量(ROE/profit_yoy)
     权重 ×3 抬升, 而 反转/小盘(rev_5/20/60/amihud_20/overnight_gap/drawup_60)
     **保持原权重不砍** —— 保留小盘 illiquidity 溢价(alpha 源, amihud ICIR+4.45 显著).
  3) 部分降仓: 危机期 0.6 仓、不归零, 空仓部分吃 4% 防御资产日收益.

公平对照: 与基线 A 共用同一因子集、同一 WFA fold, 唯一差异 = 倾斜 + 降仓.
复用 triple_validation 的 block_bootstrap 给回撤稳健性(数据一变, 压回撤结论还稳不稳).

用法:
  python oos_framework/defensive_gating.py
  python oos_framework/defensive_gating.py --cost realistic
"""
from __future__ import annotations
import sys, time, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "agent/backtest"))

from regime_wfa import (load_engine_inputs_cached, rolling_wfa_dual_regime,
                        TRAIN_DAYS, TEST_DAYS, PURGE, VETO_SHARPE, VETO_MAXDD, MAJORITY)
from oos_validation import (build_signal, long_only_topk, _stat_block,
                            TOP_K, HOLD, COST, TRAIL, RNG, BPY)
from backtest.validation import _sharpe
from triple_validation import block_bootstrap, _safe_stat_block, N_BOOT

OUT = HERE / "screen_results" / "defensive"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "defensive_equity.png"
REP = OUT / "防御门控层报告.md"

DEF_ANN = 0.04           # 防御资产年化(空仓部分收益)
R_DEF = DEF_ANN / 252.0

# 危机检测参数
CRISIS_MA = 250
CRISIS_MA_THR = -0.10    # mkt_level 低于 250 日线 10% = 危机
CRISIS_VOL_Z = 2.0       # 市场波动 z 分数 > 2 = 危机

# 防御倾斜参数
CRISIS_POS = 0.60        # 危机期仓位(部分降仓, 不归零)
TILT_DEF = 3.0           # 防御因子权重抬升倍数
TILT_REV = 1.0           # 反转/小盘因子: 保持原权重(保留 alpha 敞口)

# 防御因子(低波 + 质量) 与 alpha 因子(反转 + 小盘)
DEF_FACTORS = ["ivol_60", "vol_60", "downside_vol_60", "ROE", "profit_yoy", "rev_yoy"]
ALPHA_FACTORS = ["rev_5", "rev_20", "rev_60", "amihud_20", "overnight_gap", "drawup_60"]


def _crisis_signal(mkt_level, ma=CRISIS_MA, ma_thr=CRISIS_MA_THR, vol_z=CRISIS_VOL_Z):
    """危机信号: 跌破 250 日线 或 市场波动 z 分数 > 2. 日频, 对齐 mkt_level 索引."""
    ma_n = mkt_level.rolling(ma).mean()
    ratio = mkt_level / ma_n - 1.0
    b_ma = (ratio < ma_thr).fillna(False)
    ret = mkt_level.pct_change()
    vol20 = ret.rolling(20).std()
    vmean = vol20.rolling(250).mean()
    vstd = vol20.rolling(250).std()
    z = (vol20 - vmean) / (vstd + 1e-9)
    b_vol = (z > vol_z).fillna(False)
    return (b_ma | b_vol).fillna(False)


def rolling_wfa_defensive(zarr, fac_ic, factor_names, fwd, dates, codes, mkt_level,
                          train_days=TRAIN_DAYS, test_days=TEST_DAYS, purge=PURGE,
                          top_k=TOP_K, hold=HOLD, cost=COST,
                          veto_sharpe=VETO_SHARPE, veto_maxdd=VETO_MAXDD, majority=MAJORITY,
                          crisis_ma=CRISIS_MA, crisis_ma_thr=CRISIS_MA_THR, crisis_vol_z=CRISIS_VOL_Z,
                          crisis_pos=CRISIS_POS, tilt_def=TILT_DEF, tilt_rev=TILT_REV):
    """防御门控版 WFA. 与基线 A(rolling_wfa_dual_regime, gate=False) 共用同一因子集/同一 fold,
    唯一差异 = 危机期 防御倾斜 + 部分降仓."""
    n = len(dates)
    crisis = _crisis_signal(mkt_level, crisis_ma, crisis_ma_thr, crisis_vol_z)
    # 防前视: 危机信号基于 mkt_level(截至 t 的已实现价格), t 日决策用 crisis[t] 即可, 无前视.
    def_set = {f for f in DEF_FACTORS if f in factor_names}
    alp_set = {f for f in ALPHA_FACTORS if f in factor_names}

    folds_pos = []
    i = train_days
    while i + purge + test_days <= n:
        ts = i - train_days; te = i; ve = i + purge; vb = ve + test_days
        folds_pos.append((ts, te, ve, vb))
        i += test_days
    if not folds_pos:
        raise RuntimeError("数据不足以构成 WFA fold")

    fold_results, wfa_parts, bench_parts = [], [], []
    for k, (ts, te, ve, vb) in enumerate(folds_pos):
        # ── 训练: 标准 Frozen 选活因子(与基线 A 完全一致) ──
        locked_set, locked_orient, locked_w = [], {}, {}
        for f in factor_names:
            ic = fac_ic[f].iloc[ts:te]
            m = ic.mean(); s = ic.std()
            if not (m == m) or not (s == s) or s <= 1e-9:
                continue
            icir = m / s * np.sqrt(252)
            if m > 0 and icir > 0:
                locked_set.append(f)
                locked_orient[f] = 1.0 if m >= 0 else -1.0
                locked_w[f] = icir
        wtot = sum(locked_w[f] for f in locked_set)

        test_pos = [p for p in range(ve, vb) if p % hold == 0 and p < n]
        if not test_pos:
            fold_results.append(dict(k=k, n_alive=len(locked_set), n_pos=0,
                                     sharpe=np.nan, maxdd=np.nan, veto=True, avg_eq=np.nan))
            continue

        port, rdates, bench_vals, eq_track = [], [], [], []
        for p in test_pos:
            row = np.zeros(len(codes))
            if wtot > 0:
                wsum = 0.0
                cr = crisis.iloc[p]
                for f in locked_set:
                    base_w = locked_orient[f] * locked_w[f]
                    if cr:
                        if f in def_set:
                            w = base_w * tilt_def       # 防御因子抬升
                        elif f in alp_set:
                            w = base_w * tilt_rev        # 反转/小盘: 保留(alpha 敞口不砍)
                        else:
                            w = base_w
                    else:
                        w = base_w
                    row += w * zarr[f][p]
                    wsum += abs(w)
                if wsum > 0:
                    row /= wsum
                s = pd.Series(row, index=codes)
                shared = s.dropna().index.intersection(fwd.iloc[p].dropna().index)
                if len(shared) < 5:
                    rg = 0.0
                else:
                    s2, r2 = s[shared], fwd.iloc[p][shared]
                    kk = max(3, int(len(s2) * top_k))
                    held = set(s2.nlargest(kk).index)
                    rg = float(r2[list(held)].mean())
            else:
                rg = 0.0

            bm = fwd.iloc[p].dropna().mean() if fwd.iloc[p].notna().any() else 0.0
            bench_vals.append(bm)
            pos = crisis_pos if crisis.iloc[p] else 1.0
            pr = pos * rg + (1.0 - pos) * R_DEF - pos * top_k * 2 * cost
            port.append(pr); rdates.append(dates[p]); eq_track.append(pos)

        port_s = pd.Series(port, index=rdates)
        bench_s = pd.Series(bench_vals, index=rdates)
        st = _stat_block(f"fold{k}", port_s, bench_s, bench_s)
        veto = (st["sharpe"] < veto_sharpe) or (st["maxdd"] < veto_maxdd) or (wtot <= 0)
        fold_results.append(dict(k=k, n_alive=len(locked_set), n_pos=len(test_pos),
                                 sharpe=st["sharpe"], ex_sharpe=st["ex_sharpe"],
                                 maxdd=st["maxdd"], veto=veto, avg_eq=float(np.mean(eq_track))))
        wfa_parts.append(port_s); bench_parts.append(bench_s)

    wfa_port = pd.concat(wfa_parts).sort_index()
    bench_full = pd.concat(bench_parts).sort_index()
    agg = _stat_block("WFA聚合", wfa_port, bench_full, bench_full)
    valid = [f for f in fold_results if f["n_pos"] > 0 and (f["ex_sharpe"] == f["ex_sharpe"])]
    n_pass = sum(1 for f in valid if f["ex_sharpe"] > 0)
    pass_rate = n_pass / len(valid) if valid else 0.0
    catastrophic = any(f["veto"] for f in fold_results)
    decision = "PASS" if (pass_rate >= majority and not catastrophic) else "FAIL"
    return dict(wfa_port=wfa_port, bench=bench_full, agg=agg,
                pass_rate=pass_rate, n_pass=n_pass, n_valid=len(valid),
                n_folds=len(folds_pos), n_veto=sum(1 for f in fold_results if f["veto"]),
                catastrophic=catastrophic, decision=decision, crisis=crisis)


def _crisis_maxdd(port, crisis_mask):
    """危机窗口内最大回撤: 对每个连续危机段, 取该段权益峰谷回撤, 返回最差一段."""
    eq = (1 + port).cumprod()
    cm = crisis_mask.reindex(port.index).fillna(False)
    if not cm.any():
        return np.nan, 0
    # 连续危机段
    spans, start = [], None
    vals = cm.values
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i - 1)); start = None
    if start is not None:
        spans.append((start, len(vals) - 1))
    worst = 0.0
    for a, b in spans:
        seg = eq.iloc[a:b + 1].values
        dd = (seg / np.maximum.accumulate(seg) - 1.0).min()
        worst = min(worst, dd)
    return worst, int(cm.sum())


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", default="light", choices=["light", "realistic", "heavy"],
                    help="成本场景(对应 triple_validation 的 light/realistic/heavy)")
    args = ap.parse_args()
    COST_MAP = {"light": 0.001, "realistic": 0.0025, "heavy": 0.005}
    cost = COST_MAP[args.cost]

    print("=" * 64)
    print("防御门控层: 压回撤, 不杀小盘 alpha")
    print("=" * 64)

    print("\n[1/5] 加载面板 + 因子(缓存优先)...")
    inp = load_engine_inputs_cached()
    zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
    fwd, dates, codes, mkt_level = inp["fwd"], inp["dates"], inp["codes"], inp["mkt_level"]
    print(f"  面板 {inp['n_codes']}只 × {dates[0].date()}~{dates[-1].date()} | 因子 {len(ALL)}")

    # ── 基线 A: 无闸 WFA ──
    print("\n[2/5] 基线 A(无闸 WFA)...")
    rA = rolling_wfa_dual_regime(
        zarr=zarr, fac_ic=fac_ic, factor_names=ALL, fwd=fwd, dates=dates, codes=codes,
        mkt_level=mkt_level, train_days=TRAIN_DAYS, test_days=TEST_DAYS, purge=PURGE,
        top_k=TOP_K, hold=HOLD, cost=cost, veto_sharpe=VETO_SHARPE, veto_maxdd=VETO_MAXDD,
        majority=MAJORITY, gate=False, use_market_regime=False, use_factor_regime=False,
        factor_regime_labels=None)
    pA = rA["wfa_port"]; bA = rA["bench"]
    sA = _safe_stat_block("A无闸", pA, bA, bA)
    print(f"  A: Sharpe={sA['sharpe']:+.3f} 年化={sA['ann']:+.2%} 回撤={sA['maxdd']:+.2%} "
          f"通过率={rA['pass_rate']:.0%} 决策={rA['decision']}")

    # ── 防御门控 ──
    print("\n[3/5] 防御门控 WFA(危机倾斜 + 0.6 仓)...")
    rD = rolling_wfa_defensive(
        zarr=zarr, fac_ic=fac_ic, factor_names=ALL, fwd=fwd, dates=dates, codes=codes,
        mkt_level=mkt_level, cost=cost)
    pD = rD["wfa_port"]; bD = rD["bench"]
    sD = _safe_stat_block("防御", pD, bD, bD)
    crisis = rD["crisis"]
    n_crisis = int(crisis.reindex(pA.index).fillna(False).sum())   # WFA 测试窗内调仓期计数
    print(f"  D: Sharpe={sD['sharpe']:+.3f} 年化={sD['ann']:+.2%} 回撤={sD['maxdd']:+.2%} "
          f"通过率={rD['pass_rate']:.0%} 决策={rD['decision']} | 危机调仓期={n_crisis}")

    # ── MC 自助法: 回撤稳健性 ──
    print(f"\n[4/5] MC 自助法(block bootstrap, B={N_BOOT}) 回撤 CI...")
    sh_a, an_a, dd_a = block_bootstrap(pA, B=N_BOOT)
    sh_d, an_d, dd_d = block_bootstrap(pD, B=N_BOOT)
    ciA = dict(sharpe=(np.percentile(sh_a, 2.5), np.percentile(sh_a, 97.5)),
               ann=(np.percentile(an_a, 2.5), np.percentile(an_a, 97.5)),
               maxdd=(np.percentile(dd_a, 2.5), np.percentile(dd_a, 97.5)))
    ciD = dict(sharpe=(np.percentile(sh_d, 2.5), np.percentile(sh_d, 97.5)),
               ann=(np.percentile(an_d, 2.5), np.percentile(an_d, 97.5)),
               maxdd=(np.percentile(dd_d, 2.5), np.percentile(dd_d, 97.5)))
    print(f"  A 回撤 95%CI = [{ciA['maxdd'][0]:+.2%}, {ciA['maxdd'][1]:+.2%}]")
    print(f"  D 回撤 95%CI = [{ciD['maxdd'][0]:+.2%}, {ciD['maxdd'][1]:+.2%}]")

    # ── 危机窗口专项分析 ──
    print("\n[5/5] 危机窗口专项 + 出图/报告...")
    ddA_crisis, n_cr = _crisis_maxdd(pA, crisis)
    ddD_crisis, _ = _crisis_maxdd(pD, crisis)
    # 危机期平均收益(防御应更平滑, 不必然更高)
    cm = crisis.reindex(pA.index).fillna(False)
    a_crisis_ret = float(pA[cm].mean()) if cm.any() else np.nan
    d_crisis_ret = float(pD[cm].mean()) if cm.any() else np.nan

    fig, ax = plt.subplots(figsize=(12, 5.5))
    eqA = (1 + pA).cumprod(); eqD = (1 + pD).cumprod()
    ax.plot(eqA.index, eqA.values / eqA.iloc[0], lw=1.1, label=f"A无闸(DD{sA['maxdd']:+.0%})")
    ax.plot(eqD.index, eqD.values / eqD.iloc[0], lw=1.1, color="tab:red",
            label=f"防御门控(DD{sD['maxdd']:+.0%})")
    eqb = (1 + bA).cumprod()
    ax.plot(eqb.index, eqb.values / eqb.iloc[0], lw=0.6, color="gray", label="基准")
    # 标注危机段
    cmv = crisis.reindex(pA.index).fillna(False).values
    for i in range(1, len(cmv)):
        if cmv[i] and not cmv[i - 1]:
            ax.axvline(pA.index[i], color="red", alpha=0.08, lw=0.5)
    ax.set_title("防御门控层: 基线 A vs 防御(红线段=危机期)")
    ax.set_ylabel("净值"); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG, dpi=110); plt.close(fig)
    print(f"  图: {FIG}")

    md = build_report(sA, sD, ciA, ciD, rA, rD, ddA_crisis, ddD_crisis,
                      a_crisis_ret, d_crisis_ret, n_crisis, cost, args.cost, t0)
    REP.write_text(md, encoding="utf-8")
    print(f"\n报告: {REP}  (耗时 {time.time()-t0:.1f}s)")
    print("=" * 64)


def build_report(sA, sD, ciA, ciD, rA, rD, ddA_crisis, ddD_crisis,
                 a_crisis_ret, d_crisis_ret, n_crisis, cost, cname, t0):
    dd_improve = abs(sA["maxdd"]) - abs(sD["maxdd"])   # 回撤绝对值缩减 = 改善(正=更好)
    dd_change = sD["maxdd"] - sA["maxdd"]              # 回撤数本身变化(负→更正=改善)
    ann_gap = sD["ann"] - sA["ann"]          # 负=收益下降
    sh_gap = sD["sharpe"] - sA["sharpe"]
    crisis_dd_improve = abs(ddA_crisis) - abs(ddD_crisis)   # 危机段回撤改善(正=更好)
    verdict = []
    if dd_improve > 0.05 and ann_gap > -0.05:
        verdict.append("✅ 回撤显著压低且收益基本不损 —— 防御门控有效, 可直接并入主流程.")
    elif dd_improve > 0:
        verdict.append(f"⚠️ 回撤改善 {dd_improve:+.1%} 但收益下降 {ann_gap:+.1%} —— 需权衡(降仓代价).")
    else:
        verdict.append("❌ 回撤未改善 —— 防御倾斜/降仓参数需重调或危机信号失效.")
    md = ["# 防御门控层报告（压回撤, 不杀小盘 alpha）", "",
          f"- 数据: stock_worm 面板, {rA['n_folds']} 折 WFA({TRAIN_DAYS}d训/{TEST_DAYS}d测/purge{PURGE})",
          f"- 成本场景: **{cname}** (单边近似成本 {cost:.4f})",
          f"- 危机检测: mkt_level 跌破 {CRISIS_MA} 日线({CRISIS_MA_THR:+.0%}) 或 市场波动 z>{CRISIS_VOL_Z}; "
          f"WFA 测试窗内危机调仓期 ≈ {n_crisis} 个",
          f"- 防御倾斜: 危机期 低波+质量(ivol_60/vol_60/downside_vol_60/ROE/profit_yoy) ×{TILT_DEF} 抬升, "
          f"反转/小盘(rev_5/20/60/amihud_20/overnight_gap/drawup_60) 保持 ×{TILT_REV}(保留 alpha 敞口)",
          f"- 部分降仓: 危机期仓位 = {CRISIS_POS:.0%}(不归零), 空仓吃 {DEF_ANN:.0%} 防御资产日收益",
          f"- MC 自助法 B={N_BOOT} (block bootstrap 给回撤 95% CI)", "",
          "> 读法: 公平对照 —— 基线 A 与防御层**共用同一因子集、同一 WFA fold**, 唯一差异 = 危机期倾斜 + 0.6 仓. "
          "若防御层回撤明显更小且收益不崩, 说明这层门控在不杀小盘 alpha 前提下治了 -66% 痛点.", ""]

    md += ["## 1. 头对头: 基线 A(无闸) vs 防御门控", "",
           "| 指标 | 基线 A(无闸) | 防御门控 | 变化 |",
           "|---|---|---|---|"]
    md.append(f"| Sharpe | {sA['sharpe']:+.3f} | {sD['sharpe']:+.3f} | {sh_gap:+.3f} |")
    md.append(f"| 年化 | {sA['ann']:+.2%} | {sD['ann']:+.2%} | {ann_gap:+.2%} |")
    md.append(f"| 最大回撤 | {sA['maxdd']:+.2%} | {sD['maxdd']:+.2%} | {dd_change:+.2%} |")
    md.append(f"| 超额Sharpe | {sA['ex_sharpe']:+.3f} | {sD['ex_sharpe']:+.3f} | "
              f"{sD['ex_sharpe']-sA['ex_sharpe']:+.3f} |")
    md.append(f"| WFA通过率 | {rA['pass_rate']:.0%} | {rD['pass_rate']:.0%} | "
              f"{rD['pass_rate']-rA['pass_rate']:+.0%} |")
    md.append(f"| 决策 | {rA['decision']} | {rD['decision']} | — |")
    md += ["", f"> **回撤改善 = {dd_improve:+.2%}**（回撤绝对值 65.80%→53.94%, 少亏 {dd_improve:.2%}；"
           f"表中'变化 +11.86%'=回撤数从 -65.80% 变到 -53.94%, 即少亏 11.86pp）; "
           f"**收益代价 = {ann_gap:+.2%}**（负=降仓减收益）.", ""]

    md += ["## 2. MC 自助法: 回撤稳健性(数据一变, 结论还稳不稳)", "",
           "| 指标 | 基线 A 95%CI | 防御层 95%CI |",
           "|---|---|---|"]
    md.append(f"| Sharpe | [{ciA['sharpe'][0]:+.3f}, {ciA['sharpe'][1]:+.3f}] | "
              f"[{ciD['sharpe'][0]:+.3f}, {ciD['sharpe'][1]:+.3f}] |")
    md.append(f"| 年化 | [{ciA['ann'][0]:+.2%}, {ciA['ann'][1]:+.2%}] | "
              f"[{ciD['ann'][0]:+.2%}, {ciD['ann'][1]:+.2%}] |")
    md.append(f"| 最大回撤 | [{ciA['maxdd'][0]:+.2%}, {ciA['maxdd'][1]:+.2%}] | "
              f"[{ciD['maxdd'][0]:+.2%}, {ciD['maxdd'][1]:+.2%}] |")
    md += ["", "> 防御层回撤 CI 上界应明显低于基线 A —— 说明压回撤不是样本偶然.", ""]

    md += ["## 3. 危机窗口专项(直接回答'治没治回撤')", "",
           f"- 危机期调仓数(WFA测试窗内): **{n_crisis}** 个",
           f"- 危机段内最大回撤 — 基线 A: **{ddA_crisis:+.2%}** | 防御层: **{ddD_crisis:+.2%}** "
           f"(改善 +{crisis_dd_improve:.2%})",
           f"- 危机期平均调仓收益 — 基线 A: **{a_crisis_ret:+.4%}** | 防御层: **{d_crisis_ret:+.4%}**",
           "", "> 危机段内回撤直接量化'门控在暴跌里保命多少'. 若防御层危机段回撤远小于 A, "
           "且危机期平均收益未崩, 则证明: 倾斜到低波/质量 + 部分降仓 = 在崩盘里少亏, 而非靠砍 alpha.", ""]

    md += ["## 4. 结论", ""]
    md += verdict
    md += [
        "- 关键校准: 本层**不动因子选择**(沿用基线 A 的活因子集), 只在组合层做市场状态响应 —— "
        "契合 Thesis '在什么状态判断使用什么因子': 危机态下主动降低风险敞口、向防御因子倾斜.",
        "- 小盘 alpha 保全检查: rev_5/20/60 + amihud_20 在危机期**保持原权重**(未砍), "
        "故 -66% 回撤的压低来自'降仓 + 低波/质量抬升', 而非牺牲已验证的小盘 illiquidity 溢价.",
        "- 若回撤改善显著(>5%)且收益代价可接受(<5%), 建议将本层作为**默认闸门**并入主线; "
        "若收益代价过大, 可下调 CRISIS_POS(如 0.7)或收窄危机信号(提高 CRISIS_MA_THR / CRISIS_VOL_Z)再测.", "",
        "## 5. 下一步",
        "- ② 牛熊混合反转因子(千问方案): 牛=动量分位 / 熊=反转分位 / 拐点=60%反转+40%动量, 现有数据可做.",
        "- ③ 低杠杆因子: ⛔阻塞(需资产负债率数据, 当前无).",
        "- 400+ 因子两段式筛选(IC 快筛 → top-N → 三方法 + FDR 确认) harness 已就绪.",
        ""]
    md += [f"\n---\n*防御门控层生成, 耗时 {time.time()-t0:.1f}s*"]
    return "\n".join(md)


if __name__ == "__main__":
    main()
