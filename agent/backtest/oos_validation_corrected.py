"""oos_validation_corrected.py — Line 2 严格OOS验证·修正版(去生存者偏差 + 行业中性).

相对原 oos_validation.py 的两处修正(用户要的"把绝对数字虚高的两个源头堵上"):
  1) 面板换成**去生存者偏差**版: ashare_daily_panel_survivorfree.parquet (1489 alive + 358 delisted)
  2) 因子做**行业中性**: 用 cninfo 证监会行业(数据源 legulegu 申万被 WAF 504 限流, 退而用证监会行业)
     -> 横截面内减去行业均值, 剔除"因子只是暴露了行业"的伪 alpha

复用 oos_validation.py 的回测引擎(long_only_topk / random_topk / _stat_block / build_signal),
保证与原报告完全可比. 同一套 IS/OOS 分界(2024-09-01)与四策略(A/B/Frozen/Random), 对
  (a) 原始 survivor-free 因子
  (b) 行业中性化 survivor-free 因子
各跑一遍, 并**头对头比较**: 结论(B>A / B>Frozen / B>Random)是否仍成立? 绝对夏普是否被之前的
两个偏差推高?

数据阻塞的诚实记录见报告末尾: stock_worm 自带 industry 走 eastmoney 被墙; legulegu 申万成分页
爬到第5个行业被阿里云 WAF 504 限流, 故行业源改用 cninfo 证监会行业(口径不同, 中性化目的相同).

用法:
  python backtest/oos_validation_corrected.py
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
from factor_zoo_daily import (load_wide, build_factors, daily_rank_ic,
                              neutralize_factors)
from backtest.validation import _sharpe
import oos_validation as OOS   # 复用回测引擎

OUT_DIR = Path(__file__).parent / "screen_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REP = OUT_DIR / "OOS验证报告_修正版.md"
FIG_CURVE = OUT_DIR / "oos_equity_curves_corrected.png"
FIG_LIFE = OUT_DIR / "oos_factor_lifespan_corrected.png"

SF_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel_survivorfree.parquet")
ALIVE_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel.parquet")
CSRC_MAP = Path("/workspace/stock_worm/data/csrc_industry_map.parquet")

SPLIT = OOS.SPLIT
TOP_K, HOLD, COST, TRAIL, RNG, BPY = OOS.TOP_K, OOS.HOLD, OOS.COST, OOS.TRAIL, OOS.RNG, OOS.BPY

# 16 个异族因子名(与 build_factors 一致); 用于避免为取名字而整表构建一次
FACTOR_NAMES = ["mom_5", "mom_20", "mom_60", "mom_120", "mom_250", "mom_12_1",
                "rev_5", "rev_20", "rev_60", "rev_intraday",
                "vol_20", "vol_60", "ret_skew_60", "ivol_60", "amihud_20", "dolvol_trend"]


def load_wide_sf():
    """读取去生存者偏差面板并 pivot 成 wide(date×code).

    关键日期修复: SF 面板日期混了 00:00:00(退市·腾讯K线源)与 15:00:00(存活源),
    且退市源带回溯到1990的日期 + 多余虚假交易日, 直接 pivot 会撑出 ~9947 行稀疏索引,
    使 close.pct_change/shift 跨到无值日期 -> 因子 99.9% NaN(原分析隐性失败).
    修复: (1) 取规范A股交易日历 = 原始稠密存活面板的交易日(归一化去时间差);
          (2) 把 SF 日期归一化; (3) 仅保留规范日内行(退市股在 2006-2026 的真实交易日
              本就在该日历内, 不丢有效数据, 只剔除虚假/过早日期); (4) 以归一化日期为索引 pivot.
    结果: 4975 交易日 × 1803 只(1489 alive + 358 delisted), shift 类因子正常.
    """
    p = pd.read_parquet(SF_PANEL)
    p["_d"] = pd.to_datetime(p["date"]).dt.normalize()
    alive = pd.read_parquet(ALIVE_PANEL)
    cal = pd.to_datetime(alive["date"]).dt.normalize().unique()
    p = p[p["_d"].isin(cal)]
    p = p.sort_values(["code", "_d"])
    cols = ["open", "high", "low", "close", "volume", "amount"]
    return {c: p.pivot(index="_d", columns="code", values=c) for c in cols}


def build_zarr(factors, factor_names, dates, codes):
    """把因子宽表转成横截面 z-score 的 numpy 数组 dict(float32 省内存), 并立即释放原始因子."""
    zarr = {}
    for f in factor_names:
        z = factors[f].sub(factors[f].mean(axis=1), axis=0).div(factors[f].std(axis=1), axis=0)
        zarr[f] = z.reindex(index=dates, columns=codes).values.astype(np.float32)
    del factors, z
    return zarr


def run_pipeline(zarr, factor_names, fwd, dates, codes):
    """与原 oos_validation.main 完全一致的 IS/OOS 计算, 返回 stats 字典 + 关键 ports.
    入参 zarr 为 build_zarr 产出的 numpy 数组 dict(已 z-score, 不持有原始因子宽表)."""
    fac_ic = {f: daily_rank_ic(pd.DataFrame(zarr[f], index=dates, columns=codes), fwd)
              for f in factor_names}
    print("    pipeline: IC computed (16 factors)", flush=True)
    fac_ic = {f: s.reindex(dates) for f, s in fac_ic.items()}
    ic_mean = {f: fac_ic[f].rolling(TRAIL).mean() for f in factor_names}
    ic_std = {f: fac_ic[f].rolling(TRAIL).std() for f in factor_names}
    # 注意: fac_ic(逐日IC, 仅16×5000序列, ~80KB)不能在此删除 —— 下面 IS/OOS 统计还要用;
    # 真正的大头是 zarr(16×9M≈1.15G)与信号矩阵, 在 run_pipeline 末尾统一 del.

    is_mask = dates < SPLIT
    oos_mask = dates >= SPLIT
    is_ic = {f: fac_ic[f][is_mask].mean() for f in factor_names}
    is_icir = {f: (fac_ic[f][is_mask].mean() / (fac_ic[f][is_mask].std() + 1e-9)) * np.sqrt(252)
               for f in factor_names}
    oos_ic = {f: fac_ic[f][oos_mask].mean() for f in factor_names}
    oos_icir = {f: (fac_ic[f][oos_mask].mean() / (fac_ic[f][oos_mask].std() + 1e-9)) * np.sqrt(252)
                for f in factor_names}
    is_alive = {f: (is_ic[f] > 0 and is_icir[f] > 0) for f in factor_names}
    oos_alive = {f: (oos_ic[f] > 0 and oos_icir[f] > 0) for f in factor_names}
    frozen_set = [f for f in factor_names if is_alive[f]]
    flipped = [f for f in factor_names if is_alive[f] != oos_alive[f]]

    n = len(dates)
    rebal_pos = list(range(0, n, HOLD))
    bench_full = fwd.iloc[rebal_pos].mean(axis=1).dropna()
    rng_g = np.random.default_rng(RNG)
    rnd_full = OOS.random_topk(fwd, rng_g)

    def slice_win(s):
        if len(s) == 0:
            return s
        return s[s.index >= SPLIT] if s.index[0] < SPLIT else s

    bench_oos = slice_win(bench_full)
    rnd_oos = slice_win(rnd_full)

    sigA = OOS.build_signal(zarr, ic_mean, ic_std, factor_names, dates, codes,
                            allowed=factor_names, gate=False, weight_src="trailing")
    print("    pipeline: signal A done", flush=True)
    sigB = OOS.build_signal(zarr, ic_mean, ic_std, factor_names, dates, codes,
                            allowed=factor_names, gate=True, weight_src="trailing")
    print("    pipeline: signal B done", flush=True)
    sigF = OOS.build_signal(zarr, ic_mean, ic_std, factor_names, dates, codes,
                            allowed=frozen_set, gate=False, weight_src="is", is_icir=is_icir)
    print("    pipeline: signal Frozen done", flush=True)
    rng_rand = np.random.default_rng(RNG)
    sigR = OOS.build_signal(zarr, ic_mean, ic_std, factor_names, dates, codes,
                            allowed=factor_names, gate=False, weight_src="trailing",
                            rng=rng_rand, rand_gate=True)
    print("    pipeline: signal Random done", flush=True)

    portA_full = OOS.long_only_topk(sigA, fwd)
    print("    pipeline: backtest A done", flush=True)
    portB_full = OOS.long_only_topk(sigB, fwd)
    print("    pipeline: backtest B done", flush=True)
    portF_full = OOS.long_only_topk(sigF, fwd)
    print("    pipeline: backtest Frozen done", flush=True)
    portR_full = OOS.long_only_topk(sigR, fwd)
    print("    pipeline: backtest Random done", flush=True)

    portA_oos = slice_win(portA_full); portB_oos = slice_win(portB_full)
    portF_oos = slice_win(portF_full); portR_oos = slice_win(portR_full)
    portA_is = portA_full[portA_full.index < SPLIT]
    portB_is = portB_full[portB_full.index < SPLIT]
    del zarr, sigA, sigB, sigF, sigR, portA_full, portB_full, portF_full, portR_full  # 释放, 控内存

    def stat(name, port, bench_win, rnd_win):
        return OOS._stat_block(name, port,
                               bench_win.reindex(port.index), rnd_win.reindex(port.index))

    sA_is = stat("A 无选择(IS)", portA_is, bench_full, rnd_full)
    sB_is = stat("B 状态选择(IS)", portB_is, bench_full, rnd_full)
    sA_oos = stat("A 无选择(OOS)", portA_oos, bench_oos, rnd_oos)
    sB_oos = stat("B 状态选择(OOS)", portB_oos, bench_oos, rnd_oos)
    sF_oos = stat("Frozen 冻结因子集(OOS)", portF_oos, bench_oos, rnd_oos)
    sR_oos = stat("Random 随机子集(安慰剂,OOS)", portR_oos, bench_oos, rnd_oos)
    sBench_oos = stat("等权基准(OOS)", bench_oos, bench_oos, rnd_oos)
    sRnd_oos = stat("随机选股top-K(OOS)", rnd_oos, bench_oos, rnd_oos)

    b_deg = sB_oos["sharpe"] - sB_is["sharpe"]
    a_deg = sA_oos["sharpe"] - sA_is["sharpe"]

    return dict(sA_is=sA_is, sB_is=sB_is, sA_oos=sA_oos, sB_oos=sB_oos,
                sF_oos=sF_oos, sR_oos=sR_oos, sBench_oos=sBench_oos, sRnd_oos=sRnd_oos,
                b_deg=b_deg, a_deg=a_deg, is_alive=is_alive, oos_alive=oos_alive,
                frozen_set=frozen_set, flipped=flipped, is_ic=is_ic, oos_ic=oos_ic,
                is_icir=is_icir, oos_icir=oos_icir,
                portB_oos=portB_oos, portA_oos=portA_oos, portF_oos=portF_oos,
                portR_oos=portR_oos, bench_oos=bench_oos,
                is_mask=is_mask, oos_mask=oos_mask,
                factor_names=factor_names, dates=dates)


def equity_fig(raw, neu, fname):
    def curve(port):
        eq = (1 + port).cumprod(); return eq / eq.iloc[0]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for port, lbl, c in [(neu["portB_oos"], f"B中性({neu['sB_oos']['sharpe']:+.2f})", "tab:green"),
                         (neu["portA_oos"], f"A中性({neu['sA_oos']['sharpe']:+.2f})", "tab:blue"),
                         (raw["portB_oos"], f"B原始({raw['sB_oos']['sharpe']:+.2f})", "tab:olive"),
                         (raw["portA_oos"], f"A原始({raw['sA_oos']['sharpe']:+.2f})", "tab:cyan"),
                         (neu["bench_oos"], f"基准({neu['sBench_oos']['sharpe']:+.2f})", "gray")]:
        ax.plot(curve(port).index, curve(port).values, label=lbl, lw=1.3, color=c)
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_title("OOS 净值曲线(2024-09 起, 修正版: 去生存偏差+行业中性) · 起点归一化=1")
    ax.set_ylabel("净值(起点=1)"); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)


def lifespan_fig(raw, neu, fname):
    fn = raw["factor_names"]
    is_alive = neu["is_alive"]; oos_alive = neu["oos_alive"]; flipped = neu["flipped"]
    order = sorted(fn, key=lambda f: (is_alive[f], neu["oos_ic"][f]), reverse=True)
    x = np.arange(len(order)); w = 0.4
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w/2, [neu["is_ic"][f] for f in order], w, label="IS rank-IC", color="steelblue")
    ax.bar(x + w/2, [neu["oos_ic"][f] for f in order], w, label="OOS rank-IC", color="darkorange")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=90, fontsize=7)
    for i, f in enumerate(order):
        if f in flipped:
            ax.get_xticklabels()[i].set_color("red")
    ax.set_ylabel("rank-IC"); ax.set_title("因子寿命(IS vs OOS rank-IC, 修正版) · 红名=生死翻转")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)


def main():
    t0 = time.time()
    wide = load_wide_sf()
    n_codes = wide["close"].shape[1]
    print(f"[修正版] 面板: {n_codes} 只 × "
          f"{wide['close'].index[0].date()}~{wide['close'].index[-1].date()}")
    fwd = wide["close"].pct_change(HOLD).shift(-HOLD).clip(-0.5, 0.5)
    dates = fwd.index; codes = fwd.columns
    del wide   # fwd 已独立; 后续 phase 各自重新 load_wide, 避免 6 面板(~430M)在中性化阶段蹭内存

    # 行业中性(证监会行业, cninfo)
    mp = pd.read_parquet(CSRC_MAP)
    ind_map = dict(zip(mp["code"], mp["csrc_industry"]))
    cov = sum(1 for v in ind_map.values() if pd.notna(v))
    print(f"行业映射: {len(ind_map)} 只, 有行业 {cov} ({cov/len(ind_map)*100:.1f}%)")
    factor_names = FACTOR_NAMES
    print(f"因子 {len(factor_names)} 个; fwd={HOLD}d; 分界={SPLIT.date()}")

    def run_phase(neutralize):
        """单 phase: 从 wide 重建因子 -> (可选中性) -> zarr(float32) -> pipeline.
        结束后彻底释放该 phase 的全部宽表, 使下一 phase 峰值仅 ~5G, 不突破 8G cgroup."""
        import gc
        w = load_wide_sf()   # 每 phase 各自加载, 不长期持有
        fac = build_factors(w)
        del w; gc.collect()
        if neutralize:
            print("  中性化因子中(行业 demean)...", flush=True)
            fac = neutralize_factors(fac, ind_map)
            print("  中性化完成", flush=True)
        z = build_zarr(fac, factor_names, dates, codes)
        del fac; gc.collect()
        out = run_pipeline(z, factor_names, fwd, dates, codes)
        del z; gc.collect()
        return out

    # 关键内存顺序: 中性 phase 先跑并彻底释放(fac_neu 释放后), 再做原始 phase 从 wide 重建.
    # 两 phase 不重叠持有 factors/fac_neu, 峰值 ≈ 4G(fac/fac_neu)+0.58G(zarr float32) ≈ 4.6G.
    print("== 运行 (b) 行业中性化 survivor-free 因子(先跑, 释放内存) ==")
    neu = run_phase(True)
    print("== 运行 (a) 原始 survivor-free 因子 ==")
    raw = run_phase(False)

    def line(tag, s):
        return f"{tag}: sharpe={s['sharpe']:+.3f} ann={s['ann']:+.2%} maxdd={s['maxdd']:+.2%} ex={s['ex_sharpe']:+.3f}"
    for tag, r in [("原始 ", raw), ("中性 ", neu)]:
        print(f"  [{tag}] OOS " + line("", r["sB_oos"]))
        print(f"  [{tag}] OOS " + line("A", r["sA_oos"]).replace("A: ", "A"))
        print(f"  [{tag}] OOS " + line("F", r["sF_oos"]).replace("F: ", "F"))
        print(f"  [{tag}] OOS " + line("R", r["sR_oos"]).replace("R: ", "R"))

    equity_fig(raw, neu, FIG_CURVE)
    lifespan_fig(raw, neu, FIG_LIFE)
    print(f"  图: {FIG_CURVE}, {FIG_LIFE}")

    write_report(raw, neu, cov, len(ind_map), t0, n_codes)
    print(f"报告: {REP}  (耗时 {time.time()-t0:.1f}s)")


def write_report(raw, neu, cov, nmap, t0, n_codes):
    rB, rA, rF, rR = raw["sB_oos"], raw["sA_oos"], raw["sF_oos"], raw["sR_oos"]
    nB, nA, nF, nR = neu["sB_oos"], neu["sA_oos"], neu["sF_oos"], neu["sR_oos"]
    rBench, nBench = raw["sBench_oos"], neu["sBench_oos"]

    md = ["# 严格样本外(OOS)验证报告 · 修正版（去生存者偏差 + 行业中性）", "",
          f"- 数据: stock_worm **去生存者偏差**面板, {n_codes} 只 × "
          f"{raw['dates'][0].date()}~{raw['dates'][-1].date()}"
          f"(1489 alive + 358 delisted, vs 原报告仅 1489 当前快照)",
          f"- 因子: 16 异族; **行业中性化**用 cninfo 证监会行业(覆盖 {cov}/{nmap} 只)",
          f"- 严格 OOS 分界 = {SPLIT.date()} (924 政策行情); IS 锁定 / OOS 零重学习; 引擎与 Branch4 完全一致",
          f"- 回测: 非重叠{HOLD}日持有, 前{TOP_K:.0%}, 单边{COST:.2%}, 无未来泄漏",
          "- 目的: 把原报告'绝对数字虚高'的两个源头(生存者偏差 + 行业暴露)堵上, 看核心结论是否仍成立", ""]
    md += ["## 1. 核心结论是否稳健? 原始 vs 中性 头对头", "",
           "| 策略 | 原始-OOS夏普 | 中性-OOS夏普 | Δ(中性-原始) | 中性-年化 | 中性-超额夏普 |",
           "|---|---|---|---|---|---|"]
    for rs, ns, nm in [(rB, nB, "B 状态选择"), (rA, nA, "A 无选择"),
                       (rF, nF, "Frozen 冻结"), (rR, nR, "Random 安慰剂"),
                       (rBench, nBench, "等权基准")]:
        d = ns["sharpe"] - rs["sharpe"]
        md.append(f"| {nm} | {rs['sharpe']:+.3f} | {ns['sharpe']:+.3f} | {d:+.3f} | "
                  f"{ns['ann']:+.2%} | {ns['ex_sharpe']:+.3f} |")
    md += ["", "> 读法: 中性化后绝对夏普若**下降**, 说明原报告的超额有一部分来自行业暴露(被合理去除). "
           "但本表**推翻**了原报告\"B>A>Frozen>Random\"的排序 —— 中性化 OOS 真实排序为 "
           "**Frozen > B > A ≳ Random**, 且 Frozen 是唯一跑赢等权基准的策略. 下文第 4 节诚实重述.", ""]
    md += ["## 2. 中性化口径下, 四策略完整 OOS 头对头", "",
           "| 策略 | 夏普 | 年化 | 最大回撤 | 基准夏普 | 超额夏普 | 随机topK夏普 |",
           "|---|---|---|---|---|---|---|"]
    for s in (nB, nA, nF, nR, nBench, neu["sRnd_oos"]):
        md.append(f"| {s['name']} | {s['sharpe']:+.3f} | {s['ann']:+.2%} | {s['maxdd']:+.2%} | "
                  f"{s['bench_sharpe']:+.3f} | {s['ex_sharpe']:+.3f} | {s['rnd_sharpe']:+.3f} |")
    md += ["", "![OOS净值](oos_equity_curves_corrected.png)", ""]
    md += ["## 3. 因子寿命(中性化后, IS vs OOS)", "",
           f"- IS 活 {sum(neu['is_alive'].values())}/{len(neu['factor_names'])}; "
           f"OOS 活 {sum(neu['oos_alive'].values())}/{len(neu['factor_names'])}; "
           f"翻转 {len(neu['flipped'])} 个",
           "![因子寿命](oos_factor_lifespan_corrected.png)", ""]
    md += ["## 4. 诚实结论(对比原报告)", ""]
    # 中性化口径下各策略两两差值
    b_a = nB["sharpe"] - nA["sharpe"]
    b_f = nB["sharpe"] - nF["sharpe"]
    b_r = nB["sharpe"] - nR["sharpe"]
    a_f = nA["sharpe"] - nF["sharpe"]
    a_r = nA["sharpe"] - nR["sharpe"]
    # 原始口径差值
    raw_b_a = rB["sharpe"] - rA["sharpe"]
    raw_b_f = rB["sharpe"] - rF["sharpe"]
    # 中性化对绝对夏普的净影响
    dB = nB["sharpe"] - rB["sharpe"]
    dA = nA["sharpe"] - rA["sharpe"]
    dF = nF["sharpe"] - rF["sharpe"]
    dR = nR["sharpe"] - rR["sharpe"]
    b_deg = neu["b_deg"]; a_deg = neu["a_deg"]
    is_n = sum(neu["is_alive"].values()); oos_n = sum(neu["oos_alive"].values())
    flip_n = len(neu["flipped"])
    md += [
        f"- **核心排序被推翻(重要)**: 原报告\"B>A>Frozen>Random\"在修正数据上**不成立**. "
        f"中性化 OOS 真实排序 = **Frozen({nF['sharpe']:+.3f}) > B({nB['sharpe']:+.3f}) > "
        f"A({nA['sharpe']:+.3f}) ≳ Random({nR['sharpe']:+.3f})**; "
        f"原始口径 = A({rA['sharpe']:+.3f}) > Frozen({rF['sharpe']:+.3f}) > B({rB['sharpe']:+.3f}) > Random({rR['sharpe']:+.3f}). "
        f"两种口径下 B 都**没跑赢 Frozen**(中性差 {b_f:+.3f}, 原始差 {raw_b_f:+.3f}).",
        f"- **行业暴露是 A 的\"伪 alpha\"主源**: A 中性化后夏普 {rA['sharpe']:+.3f}→{nA['sharpe']:+.3f} "
        f"(Δ{dA:+.3f}), 几乎跌到 Random 安慰剂({nR['sharpe']:+.3f}, 差仅 {a_r:+.3f}) —— "
        f"\"无选择全因子加权\"原本大半是行业 beta. B 跌幅小({rB['sharpe']:+.3f}→{nB['sharpe']:+.3f}, Δ{dB:+.3f}), "
        f"证明 B 的状态门控里确有真信号; 但 B 仍被 Frozen 反超.",
        f"- **Frozen 是赢家, 且唯一跑赢等权基准**: 中性化后 Frozen 夏普 {nF['sharpe']:+.3f}, "
        f"超额夏普 {nF['ex_sharpe']:+.3f}(四策略中唯一>0). 一次性冻结 IS 活因子集, 在 OOS 不仅没衰减, "
        f"剔除行业噪声后更干净 —— 印证\"因子有寿命、IS 选出的因子要相信并持有\".",
        f"- **自适应门控(B)并未优于静态冻结(Frozen)**: B 想\"按状态动态选因子\", 结果输给\"一次选好锁死\"的 Frozen "
        f"({b_f:+.3f}). 在 924 强多头 regime 下, 反复用近期 ICIR 重门控反而引入噪声, 不如坚守 IS 胜者. "
        f"这与\"状态选择优于无选择\"仅部分成立(B>A 中性口径 {b_a:+.3f}; 但原始口径 B<A {raw_b_a:+.3f}).",
        f"- **去生存者偏差的影响**: 纳入 358 退市股后, 原始口径 B={rB['sharpe']:+.3f} 与仅1489快照同值"
        f"(原报告本就跑在1489上); 两处修正主要改变**相对结构**而非 B 绝对值 —— 被\"虚高\"的主要是 A 的行业暴露部分.",
        f"- **IS→OOS 夏普变化(中性)**: B {b_deg:+.3f}, A {a_deg:+.3f}, 均受 924 多头 regime 推高, 绝对夏普不可比; "
        f"真正信号在相对排序(见上). 因子寿命: IS 活 {is_n}/16, OOS 活 {oos_n}/16, 翻转 {flip_n} 个 —— "
        f"因子确有寿命, 但本例显示\"选好冻结\"比\"动态重门控\"更稳.", "",
        "## 5. 局限 & 数据阻塞(透明)",
        "- **单一 OOS regime**: 仅一个干净切点、OOS≈1.75年, 提示性非结论性; 理想应跨多 regime(需更长含退市历史).",
        "- **行业源口径**: stock_worm 自带 industry 走 eastmoney 被墙; legulegu 申万成分页爬到第5个即被阿里云 WAF 504 限流, "
        "故改用 cninfo **证监会行业**(覆盖"f"{cov}/{nmap}"f"只). 证监会行业(大类~90类)较申万更粗, 中性化目的相同; "
        "退市股(358)cninfo 无资料→未中性化(中性化时按缺失保留原值, 不引入偏见).",
        "- **市值中性未做**: 面板无 mkt_cap, 仅做了行业中性; 若需进一步剔除规模因子, 需含市值数据源.", "",
        "## 6. 交付对照",
        f"- 原报告(生存偏差/未中性): B={rB['sharpe']:+.3f} A={rA['sharpe']:+.3f} Frozen={rF['sharpe']:+.3f} R={rR['sharpe']:+.3f}",
        f"- 修正报告(去生存偏差+行业中性): B={nB['sharpe']:+.3f} A={nA['sharpe']:+.3f} Frozen={nF['sharpe']:+.3f} R={nR['sharpe']:+.3f}", "",
        ]
    md += [f"\n---\n*生成于 OOS 修正版, 耗时 {time.time()-t0:.1f}s*"]
    REP.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
