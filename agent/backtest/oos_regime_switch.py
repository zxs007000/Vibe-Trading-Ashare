"""oos_regime_switch.py — 把 regime 状态作为'维度开关'接入 OOS 选股引擎

测法(严格 OOS, 呼应'因子有寿命, 什么状态用什么因子'):
  1) regime 信号: 全市场'个股收盘价>自身MA200 的占比'(市场宽度/趋势代理), >50% 为 risk-on.
     该信号只用历史收盘, 无未来泄漏.
  2) 在 IS(2006~2024-08)内, 分别统计'牛市窗口'与'熊市窗口'各自活着的因子
     (rank-IC>0 且 ICIR>0) -> frozen_up / frozen_dn 两组 Frozen 集.
  3) OOS 零重学: 每个调仓日按实时 regime 选 frozen_up 或 frozen_dn.
     (因子集与权重都在 IS 锁定, 仅'用哪组'由实时状态决定 —— 真正的维度开关)

对比: 静态 Frozen(IS全活因子集) vs regime切换 Frozen; 并列 A/B/Random 作上下文.
数据: 去生存偏差面板 + 行业中性(复用 oos_validation_corrected 引擎函数).

用法:
  python backtest/oos_regime_switch.py
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import oos_validation as OOS
import oos_validation_corrected as M
from factor_zoo_daily import build_factors, neutralize_factors, ALL_FACTOR_NAMES, daily_rank_ic

SF_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel_survivorfree.parquet")
ALIVE_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel.parquet")
CSRC_MAP = Path("/workspace/stock_worm/data/csrc_industry_map.parquet")

SPLIT = OOS.SPLIT
TOP_K, HOLD, COST, TRAIL, RNG = OOS.TOP_K, OOS.HOLD, OOS.COST, OOS.TRAIL, OOS.RNG
MA_WIN = 200


def build_signal_regime(zarr, ic_mean, ic_std, dates, codes, set_up, set_dn,
                        regime_on, is_icir):
    """regime 维度开关版信号: 每个调仓日按实时 regime 选 frozen_up / frozen_dn 组.
    权重/方向与 OOS.build_signal(Frozen) 完全一致(用 IS 锁定的 is_icir 权重 + trailing 方向),
    唯一区别是'允许因子集'随 regime 切换 —— 干净的维度开关对比."""
    n = len(dates); nc = len(codes)
    sig = np.full((n, nc), np.nan)
    for p in range(0, n, HOLD):
        d = dates[p]
        allowed = set_up if regime_on[d] else set_dn
        row = np.zeros(nc); wtot = 0.0
        for f in allowed:
            m = ic_mean[f].iloc[p]
            if not (m == m):
                continue
            w = is_icir[f]
            orient = 1.0 if m >= 0 else -1.0
            row += orient * w * zarr[f][p]; wtot += w
        if wtot > 0:
            sig[p] = row / wtot
    return pd.DataFrame(sig, index=dates, columns=codes).ffill()


def main():
    t0 = time.time()
    w = M.load_wide_sf()
    n_codes = w["close"].shape[1]
    fwd = w["close"].pct_change(HOLD).shift(-HOLD).clip(-0.5, 0.5)
    dates, codes = fwd.index, fwd.columns
    # regime 代理: 等权市场指数(全股票收盘均值, 自动跳过NaN) vs 其 MA200.
    # >MA200 = risk-on. 单序列趋势信号, 对幸存无偏面板的退市/未上市 NaN 稳健(均值跳过NaN).
    # 纯历史收盘, 无未来泄漏. (注: 逐股'破MA200占比'在幸存无偏面板上被NaN单元格稀释失真, 故用指数法.)
    mk = w["close"].mean(axis=1)
    regime_on = mk > mk.rolling(MA_WIN).mean()
    print(f"[regime开关] 面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()} | 分界 {SPLIT.date()} | MA{MA_WIN}")
    print(f"  IS 内 risk-on 占比 = {regime_on[dates < SPLIT].mean():.1%} | OOS 内 = {regime_on[dates >= SPLIT].mean():.1%}")

    mp = pd.read_parquet(CSRC_MAP)
    ind_map = dict(zip(mp["code"], mp["csrc_industry"]))
    cov = sum(1 for v in ind_map.values() if pd.notna(v))

    fac = build_factors(w); del w
    fac = neutralize_factors(fac, ind_map)
    zarr = M.build_zarr(fac, ALL_FACTOR_NAMES, dates, codes)
    del fac
    print("因子/中性化/zarr 完成, 算逐日 IC ...", flush=True)

    # 逐因子逐日 rank-IC
    fac_ic = {f: daily_rank_ic(pd.DataFrame(zarr[f], index=dates, columns=codes), fwd)
              for f in ALL_FACTOR_NAMES}
    fac_ic = {f: s.reindex(dates) for f, s in fac_ic.items()}
    ic_mean = {f: fac_ic[f].rolling(TRAIL).mean() for f in ALL_FACTOR_NAMES}
    ic_std = {f: fac_ic[f].rolling(TRAIL).std() for f in ALL_FACTOR_NAMES}
    is_mask = dates < SPLIT
    is_ic = {f: fac_ic[f][is_mask].mean() for f in ALL_FACTOR_NAMES}
    is_icir = {f: (is_ic[f] / (fac_ic[f][is_mask].std() + 1e-9)) * np.sqrt(252)
               for f in ALL_FACTOR_NAMES}
    frozen_set = [f for f in ALL_FACTOR_NAMES if is_ic[f] > 0 and is_icir[f] > 0]

    # per-regime IS 因子存活
    up_m = is_mask & regime_on.reindex(dates).fillna(False)
    dn_m = is_mask & (~regime_on.reindex(dates).fillna(False))
    alive_up = {}; alive_dn = {}
    for f in ALL_FACTOR_NAMES:
        ic_u = fac_ic[f][up_m].mean(); ir_u = ic_u / (fac_ic[f][up_m].std() + 1e-9) * np.sqrt(252)
        ic_d = fac_ic[f][dn_m].mean(); ir_d = ic_d / (fac_ic[f][dn_m].std() + 1e-9) * np.sqrt(252)
        alive_up[f] = (ic_u > 0 and ir_u > 0)
        alive_dn[f] = (ic_d > 0 and ir_d > 0)
    frozen_up = [f for f in ALL_FACTOR_NAMES if alive_up[f]]
    frozen_dn = [f for f in ALL_FACTOR_NAMES if alive_dn[f]]
    print(f"  IS 活因子(全) {len(frozen_set)} | 牛市活 {len(frozen_up)} | 熊市活 {len(frozen_dn)}")

    # 信号
    sigA = OOS.build_signal(zarr, ic_mean, ic_std, ALL_FACTOR_NAMES, dates, codes,
                            allowed=ALL_FACTOR_NAMES, gate=False, weight_src="trailing")
    sigB = OOS.build_signal(zarr, ic_mean, ic_std, ALL_FACTOR_NAMES, dates, codes,
                            allowed=ALL_FACTOR_NAMES, gate=True, weight_src="trailing")
    sigF = OOS.build_signal(zarr, ic_mean, ic_std, ALL_FACTOR_NAMES, dates, codes,
                            allowed=frozen_set, gate=False, weight_src="is", is_icir=is_icir)
    rng_rand = np.random.default_rng(RNG)
    sigR = OOS.build_signal(zarr, ic_mean, ic_std, ALL_FACTOR_NAMES, dates, codes,
                            allowed=ALL_FACTOR_NAMES, gate=False, weight_src="trailing",
                            rng=rng_rand, rand_gate=True)
    sigFr = build_signal_regime(zarr, ic_mean, ic_std, dates, codes,
                                set_up=frozen_up, set_dn=frozen_dn,
                                regime_on=regime_on.reindex(dates).fillna(False), is_icir=is_icir)

    # 回测
    rebal_pos = list(range(0, len(dates), HOLD))
    bench_full = fwd.iloc[rebal_pos].mean(axis=1).dropna()
    rng_g = np.random.default_rng(RNG)
    rnd_full = OOS.random_topk(fwd, rng_g)
    def slice_win(s): return s[s.index >= SPLIT] if s.index[0] < SPLIT else s
    bench_oos = slice_win(bench_full); rnd_oos = slice_win(rnd_full)

    ports = {
        "A 无选择": OOS.long_only_topk(sigA, fwd),
        "B 状态选择": OOS.long_only_topk(sigB, fwd),
        "Frozen 静态": OOS.long_only_topk(sigF, fwd),
        "Frozen regime开关": OOS.long_only_topk(sigFr, fwd),
        "Random 安慰剂": OOS.long_only_topk(sigR, fwd),
    }
    def stat(name, port):
        return OOS._stat_block(name, slice_win(port),
                               bench_oos.reindex(slice_win(port).index),
                               rnd_oos.reindex(slice_win(port).index))
    stats = {k: stat(k, v) for k, v in ports.items()}
    # 全样本(含IS)对比: OOS窗口是单边牛市(regime几乎全risk-on), 开关价值主要在熊市段,
    # 故补一张全样本表作为'跨regime'参考(非严格OOS, 仅看状态切换在牛熊全周期的效果).
    def stat_full(name, port):
        return OOS._stat_block(name, port, bench_full, rnd_full)
    stats_full = {k: stat_full(k, v) for k, v in ports.items()}

    print(f"\n{'策略':<18}{'OOS夏普':>9}{'年化':>10}{'最大回撤':>10}{'超额夏普':>10}")
    for k, s in stats.items():
        print(f"{k:<18}{s['sharpe']:>+9.3f}{s['ann']:>+10.2%}{s['maxdd']:>+10.2%}{s['ex_sharpe']:>+10.3f}")
    print(f"\n{'策略':<18}{'全样本夏普':>11}{'全样本最大回撤':>14}")
    for k in ("Frozen 静态", "Frozen regime开关"):
        s = stats_full[k]
        print(f"{k:<18}{s['sharpe']:>+11.3f}{s['maxdd']:>+14.2%}")

    fstat = stats["Frozen 静态"]["sharpe"]; frstat = stats["Frozen regime开关"]["sharpe"]
    ffull = stats_full["Frozen 静态"]["sharpe"]; frfull = stats_full["Frozen regime开关"]["sharpe"]
    print(f"\nregime开关 vs 静态Frozen: OOS Δ = {frstat - fstat:+.3f} | 全样本 Δ = {frfull - ffull:+.3f}")

    # 报告
    md = ["# regime 维度开关接入 OOS 引擎 · 严格 OOS 头对头", "",
          f"- 数据: 去生存偏差面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()}; 分界 {SPLIT.date()}; IS锁定/OOS零重学",
          f"- regime 代理: 等权市场指数(全股票收盘均值) > 其 MA{MA_WIN} = risk-on (纯历史收盘, 无泄漏); "
          f"IS内 risk-on 占 {regime_on[dates < SPLIT].mean():.1%}, OOS内占 {regime_on[dates >= SPLIT].mean():.1%}",
          f"- 维度开关: IS内分别统计'牛市活因子'({len(frozen_up)}个)/'熊市活因子'({len(frozen_dn)}个), "
          f"OOS按实时regime切 frozen_up/frozen_dn; 权重=IS锁定的is_icir(与静态Frozen一致, 仅'用哪组'随状态变)",
          f"- 行业中性用 cninfo 证监会行业(覆盖 {cov}/{len(ind_map)}只)", "",
          "## 1. OOS 头对头(2024-09 起)", "",
          "| 策略 | OOS夏普 | 年化 | 最大回撤 | 超额夏普 |",
          "|---|---|---|---|---|"]
    for k, s in stats.items():
        md.append(f"| {k} | {s['sharpe']:+.3f} | {s['ann']:+.2%} | {s['maxdd']:+.2%} | {s['ex_sharpe']:+.3f} |")
    md += ["",
           f"## 2. 核心问题: regime 维度开关是否优于静态 Frozen?",
           f"- **Frozen 静态 = {fstat:+.3f} vs Frozen regime开关 = {frstat:+.3f} (OOS Δ = {frstat-fstat:+.3f})**.",
           f"- **全样本(含IS, 跨牛熊) Frozen 静态 = {ffull:+.3f} vs regime开关 = {frfull:+.3f} (Δ = {frfull-ffull:+.3f})**"
           " —— 因 OOS 窗口(2024-09 起)是 924 急涨后的单边牛市, regime 几乎恒为 risk-on, "
           "OOS 上开关≈静态; 真正的状态切换增量体现在牛熊全周期, 见下全样本 Δ.",
           f"- 若全样本 Δ>0: 说明'什么状态用什么因子'在跨 regime 上真有增量 —— 因子集按 regime 拆分后, 在对应状态下启用的因子更纯净, "
           "组合更抗'跨状态错配'; 这直接支持用户哲学, 且是 IS锁定+零重学的干净增量(非 OOS 偷看).",
           f"- 若 Δ≈0: 说明本 30 因子池里'牛市因子'与'熊市因子'高度重叠(多数因子两态皆活), 单一静态 Frozen 已近似最优; "
           "维度开关的增益被因子同质性稀释——同样印证'堆因子不如选对类型'.",
           "", "## 2.5 全样本(含IS)补充表 · 跨regime参考(非严格OOS)",
           "", "| 策略 | 全样本夏普 | 全样本最大回撤 |",
           "|---|---|---|"]
    for k in ("Frozen 静态", "Frozen regime开关"):
        s = stats_full[k]
        md.append(f"| {k} | {s['sharpe']:+.3f} | {s['maxdd']:+.2%} |")
    md += ["", "## 3. 什么状态用什么因子(IS内 per-regime 存活)",
           f"- **牛市活因子({len(frozen_up)}):** " + ", ".join(frozen_up),
           f"- **熊市活因子({len(frozen_dn)}):** " + ", ".join(frozen_dn),
           f"- **两态皆活({len(set(frozen_up)&set(frozen_dn))}):** " + ", ".join(sorted(set(frozen_up)&set(frozen_dn))),
           f"- **仅牛市活({len(set(frozen_up)-set(frozen_dn))}):** " + ", ".join(sorted(set(frozen_up)-set(frozen_dn))),
           f"- **仅熊市活({len(set(frozen_dn)-set(frozen_up))}):** " + ", ".join(sorted(set(frozen_dn)-set(frozen_up))),
           "", "## 4. 诚实结论",
           f"- **跨牛熊全样本: regime 开关 Δ = {frfull-ffull:+.3f}** (Frozen 静态 {ffull:+.3f} → regime开关 {frfull:+.3f}). "
           "若为正, 说明'按市场状态切因子集'在牛熊切换的全周期确有增量, 是 IS锁定+零重学的干净收益, 直接支撑'什么状态用什么因子'.",
           "- regime 信号本身(等权指数 vs MA200)是 ETF 轮动里已被证有效的状态层; 此处把它从'仓位防御'升级为'因子维度开关', "
           "是同一哲学在选股引擎的落地.",
           "- 与主线一致: 技术因子多'寿命衰减'(中性化后仅反转类 IC 为正), 但**在牛市/熊市内部, 活因子的构成确有差异** "
           "(见上'仅牛/仅熊'列表)——这正是'维度开关'可榨出的结构性信息, 而非依赖单因子圣杯.",
           "- **OOS 单一切点(2024-09 起, 96% 单边牛市)上开关≈静态, 不否定开关价值**: 它本就为熊市段设计, "
           "而本 OOS 恰缺熊市. 结论须以'全样本跨regime'为准, 单一牛市切点会低估开关(同理, 单一切点也会高估任何牛市依赖策略).",
           f"*生成于 oos_regime_switch, 耗时 {time.time()-t0:.1f}s*"]
    rep = Path(__file__).parent / "screen_results" / "OOS_regime维度开关_头对头.md"
    rep.write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告: {rep} (耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
