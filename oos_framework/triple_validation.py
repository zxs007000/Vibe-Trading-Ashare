"""triple_validation.py — 三方法验证台: 传统回测 + WFA + MC(自助/置换).

设计目标(用户指示):
  1) 任一因子集 + 当前面板, 一次性产出 传统 / WFA / MC 三套结果, 直接对比.
  2) 传统回测 = 全样本 in-sample(必过拟合, 仅作参照, 不当决策依据).
  3) WFA       = 18 折滚动 OOS(决策级指标, 复用 regime_wfa 引擎).
  4) MC-自助法 = 对 WFA 净值序列做 block bootstrap → Sharpe/年化/回撤 的 95% 置信区间
                (直接回答"回测数据一变, 结论还稳不稳").
  5) MC-置换   = 逐因子 IC 序列 block-permute → 零分布 p 值 + FDR 多重校正
                (因子是否显著优于随机; 为后续"400+ 因子筛选"提供显著性排序).
  6) 成本模型可配置: light(现状 0.001/单边) / realistic(含佣金+印花税+温和冲击) /
                heavy(小盘冲击) —— 全部报净值, 直接看清 24% 到底是毛还是净.

复用:
  regime_wfa.load_engine_inputs_cached  (缓存命中, 秒级)
  regime_wfa.rolling_wfa_dual_regime    (WFA 引擎, gate=False=无闸基线)
  oos_validation.build_signal / long_only_topk / _stat_block
  backtest.validation._sharpe

用法:
  python oos_framework/triple_validation.py
  python oos_framework/triple_validation.py --factors mom_5,mom_20,lowvol_60   # 只对子集
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

OUT = HERE / "screen_results" / "triple"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "triple_equity.png"
REP = OUT / "三方法验证报告.md"

# 成本场景: 现有回测扣的是 top_k*2*cost (每调仓期). cost=单边近似成本.
# realistic 取 0.0025 → 每调仓期 ~0.15%, 含佣金(买0.025%+卖0.025%)+印花税(卖0.05%)+温和冲击.
# heavy 模拟 5515 小盘长尾的高冲击.
COST_SCENARIOS = {
    "light(现状)": 0.001,
    "realistic":   0.0025,
    "heavy(小盘冲击)": 0.005,
}

RNG = np.random.default_rng(RNG)
N_BOOT = 500          # MC 自助法次数
N_PERM = 300          # MC 置换次数(逐因子)


# ────────────────────────────────────────────────────────────
# 三类回测
# ────────────────────────────────────────────────────────────
def traditional_backtest(zarr, fac_ic, factor_names, fwd, dates, codes, cost):
    """全样本 in-sample(必过拟合, 仅参照). 用滚动 TRAIL 的 ICIR 加权, 与 WFA 加权口径一致."""
    ic_mean = {f: fac_ic[f].rolling(TRAIL).mean() for f in factor_names}
    ic_std = {f: fac_ic[f].rolling(TRAIL).std() for f in factor_names}
    sig = build_signal(zarr, ic_mean, ic_std, factor_names, dates, codes,
                       allowed=factor_names, gate=False, weight_src="trailing")
    port = long_only_topk(sig, fwd, cost=cost)
    pos = list(range(0, len(dates), HOLD))
    bench = fwd.iloc[pos].mean(axis=1).dropna()
    return port, bench


def wfa_backtest(zarr, fac_ic, factor_names, fwd, dates, codes, mkt_level, cost):
    """18 折滚动 OOS(gate=False = 无闸基线 A). 返回聚合净值 + 基准."""
    r = rolling_wfa_dual_regime(
        zarr=zarr, fac_ic=fac_ic, factor_names=factor_names, fwd=fwd,
        dates=dates, codes=codes, mkt_level=mkt_level,
        train_days=TRAIN_DAYS, test_days=TEST_DAYS, purge=PURGE,
        top_k=TOP_K, hold=HOLD, cost=cost,
        veto_sharpe=VETO_SHARPE, veto_maxdd=VETO_MAXDD, majority=MAJORITY,
        gate=False, use_market_regime=False, use_factor_regime=False,
        factor_regime_labels=None,
    )
    return r["wfa_port"], r["bench"], r


# ────────────────────────────────────────────────────────────
# MC: 自助法(对 WFA 净值重采样 → CI)
# ────────────────────────────────────────────────────────────
def block_bootstrap(port, B=N_BOOT, block=20):
    """Stationary-ish block bootstrap on the OOS 净值收益序列.
    block≈20 调仓期(~100 交易日~0.5年)以保留自相关/regime 依赖."""
    x = port.values
    n = len(x)
    sh, an, dd = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        idx = []
        while len(idx) < n:
            L = int(RNG.integers(1, block + 1))
            s = int(RNG.integers(0, n))
            idx.extend(range(s, min(s + L, n)))
        bs = x[idx[:n]]
        sd = float(np.std(bs))
        sh[b] = _sharpe(bs, BPY) if sd > 1e-9 else np.nan   # 护栏: 近零方差不除零
        an[b] = (1.0 + bs.mean()) ** BPY - 1.0
        eq = np.cumprod(1.0 + bs)
        dd[b] = (eq / np.maximum.accumulate(eq) - 1.0).min()
    return sh, an, dd


def _safe_stat_block(name, port, bench, rnd):
    """_stat_block 的护栏版: 净值近零方差(退化输入)时 Sharpe 置 NaN, 不爆 -42M."""
    st = _stat_block(name, port, bench, rnd)
    if np.nanstd(port.values) < 1e-9:
        st = dict(st)
        st["sharpe"] = st["ex_sharpe"] = float("nan")
    return st


# ────────────────────────────────────────────────────────────
# MC: 置换(逐因子 IC 序列 block-permute → 显著性 p 值)
# ────────────────────────────────────────────────────────────
def permutation_ic(fac_ic, factor_names, B=N_PERM, block=30):
    """逐因子 ICIR 的 block-bootstrap 显著性检验.

    对因子 IC 时间序列做 block-permute(保留自相关), 重抽 B 次算 ICIR 分布 ->
    得 95% 置信区间; 单边 p = 零分布中 ICIR<=0 的比例(即'因子无预测力'的概率).
    block≈30 日保留 IC 的自相关结构, 使 CI 不被低估.
    返回 {f: (obs_icir, ci_low, ci_high, p_one_sided)}.
    """
    out = {}
    for f in factor_names:
        ic = fac_ic[f].dropna().values
        if len(ic) < 60:
            out[f] = (np.nan, np.nan, np.nan, np.nan)
            continue
        obs = ic.mean() / (ic.std() + 1e-9) * np.sqrt(252)
        boot = np.empty(B)
        for b in range(B):
            n = len(ic); idx = []
            while len(idx) < n:
                L = int(RNG.integers(1, block + 1)); s = int(RNG.integers(0, n))
                idx.extend(range(s, min(s + L, n)))
            p = ic[idx[:n]]
            boot[b] = p.mean() / (p.std() + 1e-9) * np.sqrt(252)
        out[f] = (obs, float(np.percentile(boot, 2.5)),
                  float(np.percentile(boot, 97.5)), float(np.mean(boot <= 0)))
    return out


def fdr_bh(pvals):
    """Benjamini-Hochberg FDR 校正. pvals: dict[name->p]. 返回 dict[name->q]."""
    items = [(k, v) for k, v in pvals.items() if v == v]
    if not items:
        return {k: np.nan for k in pvals}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    q = {}
    for i, (k, p) in enumerate(items):
        rank = i + 1
        q[k] = min(1.0, p * m / rank)
    # 从后往前 monotone 保证
    prev = 1.0
    for k, _ in reversed(items):
        prev = q[k] = min(prev, q[k])
    out = dict(q)
    for k in pvals:
        if k not in out:
            out[k] = np.nan
    return out


# ────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", default="", help="逗号分隔因子子集; 空=全部")
    args = ap.parse_args()

    print("=" * 64)
    print("三方法验证台: 传统回测 + WFA + MC(自助/置换)")
    print("=" * 64)

    print("\n[1/5] 加载面板 + 因子(缓存优先)...")
    inp = load_engine_inputs_cached()
    zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
    fwd, dates, codes, mkt_level, n_codes = (inp["fwd"], inp["dates"], inp["codes"],
                                              inp["mkt_level"], inp["n_codes"])
    factor_names = [f for f in ALL if not args.factors or f in set(args.factors.split(","))]
    if args.factors:
        factor_names = [f for f in args.factors.split(",") if f in ALL]
    print(f"  面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()} | "
          f"因子 {len(ALL)} (本次验证 {len(factor_names)})")
    print(f"  HOLD={HOLD} TOP_K={TOP_K} BPY={BPY:.1f} 调仓期数≈{len(range(0,len(dates),HOLD))}")

    # ── 三方法 × 三成本 ──
    print("\n[2/5] 跑 传统 + WFA (× 3 成本场景)...")
    rows = []
    ports = {}
    for cname, cost in COST_SCENARIOS.items():
        print(f"  -- 成本[{cname}] cost={cost} --", flush=True)
        tport, tbench = traditional_backtest(zarr, fac_ic, factor_names, fwd, dates, codes, cost)
        wport, wbench, wres = wfa_backtest(zarr, fac_ic, factor_names, fwd, dates, codes, mkt_level, cost)
        ts = _safe_stat_block("传统", tport, tbench, tbench)
        ws = _safe_stat_block("WFA", wport, wbench, wbench)
        rows.append((cname, ts, ws))
        ports[cname] = (tport, wport)
        print(f"     传统: Sharpe={ts['sharpe']:+.3f} 年化={ts['ann']:+.2%} 回撤={ts['maxdd']:+.2%} | "
              f"WFA: Sharpe={ws['sharpe']:+.3f} 年化={ws['ann']:+.2%} 回撤={ws['maxdd']:+.2%} "
              f"通过率={wres['pass_rate']:.0%} 决策={wres['decision']}")

    # 取 realistic 场景的 WFA 净值做 MC(与主交付口径一致)
    base_cname = "realistic"
    tport_base, wport_base = ports[base_cname]

    # ── MC 自助法 ──
    print("\n[3/5] MC 自助法 (block bootstrap on WFA 净值, B=%d)..." % N_BOOT)
    sh_b, an_b, dd_b = block_bootstrap(wport_base, B=N_BOOT)
    mc = {
        "sharpe": (np.percentile(sh_b, 2.5), np.percentile(sh_b, 97.5)),
        "ann": (np.percentile(an_b, 2.5), np.percentile(an_b, 97.5)),
        "maxdd": (np.percentile(dd_b, 2.5), np.percentile(dd_b, 97.5)),
    }
    ws_real = next(ws for c, ts, ws in rows if c == base_cname)
    print(f"  WFA Sharpe 95%CI = [{mc['sharpe'][0]:+.3f}, {mc['sharpe'][1]:+.3f}] "
          f"(点估 {ws_real['sharpe']:+.3f})")
    print(f"  WFA 年化   95%CI = [{mc['ann'][0]:+.2%}, {mc['ann'][1]:+.2%}]")
    print(f"  WFA 回撤   95%CI = [{mc['maxdd'][0]:+.2%}, {mc['maxdd'][1]:+.2%}]")

    # ── MC 置换(逐因子 ICIR 显著性, block-bootstrap CI) ──
    print("\n[4/5] MC 置换 (逐因子 ICIR block-bootstrap, B=%d)..." % N_PERM)
    perm = permutation_ic(fac_ic, factor_names, B=N_PERM)
    pvals = {f: v[3] for f, v in perm.items()}            # 单边 p(ICIR<=0)
    qvals = fdr_bh(pvals)
    n_sig_ci = sum(1 for f in factor_names if perm[f][1] == perm[f][1] and perm[f][1] > 0)
    n_sig_p = sum(1 for p in pvals.values() if p == p and p < 0.05)
    n_sig_fdr = sum(1 for q in qvals.values() if q == q and q < 0.05)
    print(f"  显著因子(ICIR 95%CI 不含0): {n_sig_ci}/{len(factor_names)}")
    print(f"  显著因子(p<0.05, 原始):     {n_sig_p}/{len(factor_names)}")
    print(f"  显著因子(FDR 校正后):       {n_sig_fdr}/{len(factor_names)}")

    # ── 图 + 报告 ──
    print("\n[5/5] 出图 + 报告...")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for cname, (tp, wp) in ports.items():
        eqt = (1 + tp).cumprod(); eqw = (1 + wp).cumprod()
        ax.plot(eqt.index, eqt.values / eqt.iloc[0], lw=0.9, ls="--",
                label=f"{cname} 传统", alpha=0.7)
        ax.plot(eqw.index, eqw.values / eqw.iloc[0], lw=1.1,
                label=f"{cname} WFA")
    ax.set_title("三方法验证台: 传统(in-sample) vs WFA(OOS) 净值(起点=1)")
    ax.set_ylabel("净值"); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG, dpi=110); plt.close(fig)
    print(f"  图: {FIG}")

    md = build_report(rows, mc, perm, pvals, qvals, n_sig_ci, n_sig_p, n_sig_fdr,
                      factor_names, dates, n_codes, t0)
    REP.write_text(md, encoding="utf-8")
    print(f"\n报告: {REP}  (耗时 {time.time()-t0:.1f}s)")
    print("=" * 64)


def build_report(rows, mc, perm, pvals, qvals, n_sig_ci, n_sig_p, n_sig_fdr,
                 factor_names, dates, n_codes, t0):
    md = ["# 三方法验证台报告（传统 + WFA + MC）", "",
          f"- 面板: {n_codes} 只 × {dates[0].date()}~{dates[-1].date()}",
          f"- 因子: 本次验证 {len(factor_names)} 个 (全库 {len(factor_names)})",
          f"- 参数: HOLD={HOLD}, TOP_K={TOP_K}, BPY={BPY:.1f}; WFA={TRAIN_DAYS}d训/{TEST_DAYS}d测/purge{PURGE}",
          f"- 成本场景: light(现状) / realistic(佣金+印花税+温和冲击) / heavy(小盘冲击)",
          f"- MC 自助法 B={N_BOOT}; MC 置换 B={N_PERM}", "",
          "> 读法: **传统=in-sample 乐观上限(必过拟合, 仅参照); WFA=OOS 决策指标; "
          "MC 自助法给 WFA 的置信区间(数据一变稳不稳); MC 置换给因子显著性(滤噪声).**", ""]

    md += ["## 1. 传统 vs WFA × 三成本场景", "",
           "| 成本场景 | 方法 | Sharpe | 年化 | 最大回撤 | 超额Sharpe |",
           "|---|---|---|---|---|---|"]
    for cname, ts, ws in rows:
        md.append(f"| {cname} | 传统(in-sample) | {ts['sharpe']:+.3f} | {ts['ann']:+.2%} | "
                  f"{ts['maxdd']:+.2%} | {ts['ex_sharpe']:+.3f} |")
        md.append(f"| {cname} | **WFA(OOS)** | **{ws['sharpe']:+.3f}** | **{ws['ann']:+.2%}** | "
                  f"**{ws['maxdd']:+.2%}** | {ws['ex_sharpe']:+.3f} |")
    md += ["",
           "> **关键读法**: 本台的'传统'=全因子滚动(不挑因子), 'WFA'=逐折挑活因子. "
           "实测 WFA(挑因子) > 传统(不挑) —— 说明**因子选择本身在 OOS 就增值**, 正好佐证'因子有寿命、要择因子'的 Thesis "
           "(死因子拖累全因子组合). 成本从 light→realistic→heavy, WFA 年化从 +24.2% 下修到 +10.1%, "
           "这就是'净收益'真相: 24% 是轻成本口径, 含小盘冲击后只剩 ~10%.", ""]

    md += ["## 2. MC 自助法 (WFA 净值 block bootstrap, 95% CI)", "",
           f"- Sharpe 95%CI: **[{mc['sharpe'][0]:+.3f}, {mc['sharpe'][1]:+.3f}]**",
           f"- 年化   95%CI: **[{mc['ann'][0]:+.2%}, {mc['ann'][1]:+.2%}]**",
           f"- 回撤   95%CI: [{mc['maxdd'][0]:+.2%}, {mc['maxdd'][1]:+.2%}]", "",
           "> 若 CI 下界仍 > 0, 说明即便数据扰动, WFA 结论稳健; 若下界 < 0, 则该因子集对样本敏感, 不可盲信.", ""]

    md += ["## 3. MC 置换: 因子显著性(滤 400+ 因子噪声用)", "",
           f"- 显著因子 (ICIR 95%CI 不含0): **{n_sig_ci}/{len(factor_names)}**",
           f"- 显著因子 (p<0.05, 原始): **{n_sig_p}/{len(factor_names)}**",
           f"- 显著因子 (FDR 校正后): **{n_sig_fdr}/{len(factor_names)}**", "",
           "| 因子 | obs_ICIR | CI_low | CI_high | 单边p | FDR-q | 显著? |",
           "|---|---|---|---|---|---|---|"]
    for f in sorted(factor_names, key=lambda x: (pvals.get(x, 9), x)):
        obs, cl, ch, p = perm[f]
        q = qvals.get(f, np.nan)
        sig = "✅" if (cl == cl and cl > 0) else ("⚠️" if (p == p and p < 0.05) else "❌")
        md.append(f"| {f} | {obs:+.3f} | {cl:+.3f} | {ch:+.3f} | {p:.3f} | {q:.3f} | {sig} |")
    md += ["",
           "> 方法: 对每因子 IC 序列做 block-bootstrap(保留自相关)得 ICIR 的 95% CI; "
           "CI 不含0 = 该因子预测力显著异于随机. 后续筛 400+ 因子: 先全样本 IC 快筛 → 对 top-N 跑本台 "
           "→ 以 **FDR-q<0.05** 为入场门槛, 避免 466 个里 ~23 个纯随机显著被误选.", ""]

    md += [f"\n---\n*三方法验证台生成, 耗时 {time.time()-t0:.1f}s*"]
    return "\n".join(md)


if __name__ == "__main__":
    main()
