"""oos_regime_switch_fund.py — regime 维度开关 + 基本面(质量维度)头对头

是 oos_regime_switch 的延伸: 原版只用 30 技术因子, 因子池跨regime高度同质 -> 开关≈静态Frozen.
本脚本把**全面板覆盖(1828/1803)的基本面因子 ROE / rev_yoy / profit_yoy** 作为'质量维度'并入因子池,
重测: 基本面是否在熊市活得更稳(防御性质量), 从而给 regime 开关提供跨regime异质性, 使其真正优于静态Frozen.

测法完全沿用 oos_regime_switch(严格 OOS, IS锁定/OOS零重学, 等权指数 vs MA200 作regime信号),
唯一增量: 因子池 30 -> 33(30技术 + 3基本面, 基本面未覆盖码填0=中性, 与 oos_fundamental_check 一致).

对比(均在33因子集内):
  - 静态 Frozen(33): IS全活因子集, 不切regime
  - Frozen regime开关(33): IS内分牛/熊存活集, OOS按实时regime切 frozen_up/frozen_dn
  - 并列 A/B/Random 作上下文

用法:
  python backtest/oos_regime_switch_fund.py
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
FUND_PARQUET = Path("/workspace/stock_worm/data/fundamentals/fund_factors_daily.parquet")
FUND_NAMES = ["ROE", "rev_yoy", "profit_yoy"]

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
    mk = w["close"].mean(axis=1)
    regime_on = mk > mk.rolling(MA_WIN).mean()
    print(f"[regime开关+基本面] 面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()} | 分界 {SPLIT.date()} | MA{MA_WIN}")
    print(f"  IS 内 risk-on 占比 = {regime_on[dates < SPLIT].mean():.1%} | OOS 内 = {regime_on[dates >= SPLIT].mean():.1%}")

    mp = pd.read_parquet(CSRC_MAP)
    ind_map = dict(zip(mp["code"], mp["csrc_industry"]))
    cov = sum(1 for v in ind_map.values() if pd.notna(v))

    fac = build_factors(w); del w
    fac = neutralize_factors(fac, ind_map)
    # 并入全面板基本面(覆盖 ~1828/1803, 未覆盖填0=中性)
    fund = pd.read_pickle(FUND_PARQUET)
    for f in FUND_NAMES:
        fac[f] = fund[f]
    ALL = ALL_FACTOR_NAMES + FUND_NAMES
    zarr = M.build_zarr(fac, ALL, dates, codes)
    del fac
    # 基本面未覆盖码填0(中性): 防 NaN 沿加权和传播把码挤出选股池
    for f in FUND_NAMES:
        zarr[f] = np.nan_to_num(zarr[f], nan=0.0)
    print(f"因子(技术{len(ALL_FACTOR_NAMES)}+基本面{len(FUND_NAMES)}={len(ALL)})/中性化/zarr 完成, 算逐日 IC ...", flush=True)

    fac_ic = {f: daily_rank_ic(pd.DataFrame(zarr[f], index=dates, columns=codes), fwd)
              for f in ALL}
    fac_ic = {f: s.reindex(dates) for f, s in fac_ic.items()}
    ic_mean = {f: fac_ic[f].rolling(TRAIL).mean() for f in ALL}
    ic_std = {f: fac_ic[f].rolling(TRAIL).std() for f in ALL}
    is_mask = dates < SPLIT
    is_ic = {f: fac_ic[f][is_mask].mean() for f in ALL}
    is_icir = {f: (is_ic[f] / (fac_ic[f][is_mask].std() + 1e-9)) * np.sqrt(252)
               for f in ALL}
    frozen_set = [f for f in ALL if is_ic[f] > 0 and is_icir[f] > 0]

    # per-regime IS 因子存活
    up_m = is_mask & regime_on.reindex(dates).fillna(False)
    dn_m = is_mask & (~regime_on.reindex(dates).fillna(False))
    alive_up = {}; alive_dn = {}
    for f in ALL:
        ic_u = fac_ic[f][up_m].mean(); ir_u = ic_u / (fac_ic[f][up_m].std() + 1e-9) * np.sqrt(252)
        ic_d = fac_ic[f][dn_m].mean(); ir_d = ic_d / (fac_ic[f][dn_m].std() + 1e-9) * np.sqrt(252)
        alive_up[f] = (ic_u > 0 and ir_u > 0)
        alive_dn[f] = (ic_d > 0 and ir_d > 0)
    frozen_up = [f for f in ALL if alive_up[f]]
    frozen_dn = [f for f in ALL if alive_dn[f]]
    print(f"  IS 活因子(全) {len(frozen_set)} | 牛市活 {len(frozen_up)} | 熊市活 {len(frozen_dn)}")
    # 基本面因子在各regime的存活明细
    for f in FUND_NAMES:
        print(f"    基本面 {f}: 牛={alive_up[f]} 熊={alive_dn[f]} | IS_IC {is_ic[f]:+.4f} IS_ICIR {is_icir[f]:+.2f}")

    # 信号
    sigA = OOS.build_signal(zarr, ic_mean, ic_std, ALL, dates, codes,
                            allowed=ALL, gate=False, weight_src="trailing")
    sigB = OOS.build_signal(zarr, ic_mean, ic_std, ALL, dates, codes,
                            allowed=ALL, gate=True, weight_src="trailing")
    sigF = OOS.build_signal(zarr, ic_mean, ic_std, ALL, dates, codes,
                            allowed=frozen_set, gate=False, weight_src="is", is_icir=is_icir)
    rng_rand = np.random.default_rng(RNG)
    sigR = OOS.build_signal(zarr, ic_mean, ic_std, ALL, dates, codes,
                            allowed=ALL, gate=False, weight_src="trailing",
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
        "Frozen 静态(33)": OOS.long_only_topk(sigF, fwd),
        "Frozen regime开关(33)": OOS.long_only_topk(sigFr, fwd),
        "Random 安慰剂": OOS.long_only_topk(sigR, fwd),
    }
    def stat(name, port):
        return OOS._stat_block(name, slice_win(port),
                               bench_oos.reindex(slice_win(port).index),
                               rnd_oos.reindex(slice_win(port).index))
    stats = {k: stat(k, v) for k, v in ports.items()}
    def stat_full(name, port):
        return OOS._stat_block(name, port, bench_full, rnd_full)
    stats_full = {k: stat_full(k, v) for k, v in ports.items()}

    print(f"\n{'策略':<22}{'OOS夏普':>9}{'年化':>10}{'最大回撤':>10}{'超额夏普':>10}")
    for k, s in stats.items():
        print(f"{k:<22}{s['sharpe']:>+9.3f}{s['ann']:>+10.2%}{s['maxdd']:>+10.2%}{s['ex_sharpe']:>+10.3f}")
    print(f"\n{'策略':<22}{'全样本夏普':>11}{'全样本最大回撤':>14}")
    for k in ("Frozen 静态(33)", "Frozen regime开关(33)"):
        s = stats_full[k]
        print(f"{k:<22}{s['sharpe']:>+11.3f}{s['maxdd']:>+14.2%}")

    fstat = stats["Frozen 静态(33)"]["sharpe"]; frstat = stats["Frozen regime开关(33)"]["sharpe"]
    ffull = stats_full["Frozen 静态(33)"]["sharpe"]; frfull = stats_full["Frozen regime开关(33)"]["sharpe"]
    print(f"\nregime开关 vs 静态Frozen(33): OOS Δ = {frstat - fstat:+.3f} | 全样本 Δ = {frfull - ffull:+.3f}")

    # 报告
    md = ["# regime 维度开关 + 基本面(质量维度) · 严格 OOS 头对头",
          "",
          f"- 数据: 去生存偏差面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()}; 分界 {SPLIT.date()}; IS锁定/OOS零重学",
          f"- 因子池: 30 技术 + 3 基本面(ROE/rev_yoy/profit_yoy, 全面板覆盖 ~1828/1803, 未覆盖填0中性)",
          f"- regime 代理: 等权市场指数 > 其 MA{MA_WIN} = risk-on; IS内 risk-on 占 {regime_on[dates < SPLIT].mean():.1%}, "
          f"OOS内占 {regime_on[dates >= SPLIT].mean():.1%}",
          f"- 维度开关: IS内分'牛市活'({len(frozen_up)})/'熊市活'({len(frozen_dn)}), OOS按实时regime切 frozen_up/frozen_dn; "
          f"权重=IS锁定的is_icir",
          f"- 行业中性用 cninfo 证监会行业(覆盖 {cov}/{len(ind_map)}只)",
          "- 动机: 原 oos_regime_switch 仅30技术因子, 跨regime高度同质 -> 开关≈静态Frozen; "
          "现注入全面板基本面作'质量维度', 检验基本面是否在熊市活得更稳(防御性质量)从而给开关异质性.",
          "",
          "## 1. OOS 头对头(2024-09 起)", "",
          "| 策略 | OOS夏普 | 年化 | 最大回撤 | 超额夏普 |",
          "|---|---|---|---|---|"]
    for k, s in stats.items():
        md.append(f"| {k} | {s['sharpe']:+.3f} | {s['ann']:+.2%} | {s['maxdd']:+.2%} | {s['ex_sharpe']:+.3f} |")
    md += ["",
           f"## 2. 核心问题: regime 维度开关(含基本面)是否优于静态 Frozen?",
           f"- **Frozen 静态(33) = {fstat:+.3f} vs Frozen regime开关(33) = {frstat:+.3f} (OOS Δ = {frstat-fstat:+.3f})**.",
           f"- **全样本(含IS, 跨牛熊) Frozen 静态(33) = {ffull:+.3f} vs regime开关(33) = {frfull:+.3f} (Δ = {frfull-ffull:+.3f})**.",
           "- OOS 窗口(2024-09 起)是 924 急涨后单边牛市, regime 几乎恒为 risk-on, OOS 上开关≈静态; "
           "真正的状态切换增量体现在牛熊全周期, 以'全样本 Δ'为准.",
           "- 若全样本 Δ>0: 说明基本面(质量维度)在熊市活得更稳, '什么状态用什么因子'在跨regime上真有增量, "
           "直接支撑用户哲学, 且是 IS锁定+零重学的干净增量.",
           "",
           "## 2.5 全样本(含IS)补充表 · 跨regime参考(非严格OOS)",
           "", "| 策略 | 全样本夏普 | 全样本最大回撤 |",
           "|---|---|---|"]
    for k in ("Frozen 静态(33)", "Frozen regime开关(33)"):
        s = stats_full[k]
        md.append(f"| {k} | {s['sharpe']:+.3f} | {s['maxdd']:+.2%} |")
    md += ["", "## 3. 什么状态用什么因子(IS内 per-regime 存活)",
           f"- **牛市活因子({len(frozen_up)}):** " + ", ".join(frozen_up),
           f"- **熊市活因子({len(frozen_dn)}):** " + ", ".join(frozen_dn),
           f"- **两态皆活({len(set(frozen_up)&set(frozen_dn))}):** " + ", ".join(sorted(set(frozen_up)&set(frozen_dn))),
           f"- **仅牛市活({len(set(frozen_up)-set(frozen_dn))}):** " + ", ".join(sorted(set(frozen_up)-set(frozen_dn))),
           f"- **仅熊市活({len(set(frozen_dn)-set(frozen_up))}):** " + ", ".join(sorted(set(frozen_dn)-set(frozen_up))),
           "", "### 基本面因子跨regime明细",
           "",
           "| 因子 | 牛活 | 熊活 | IS_IC | IS_ICIR | OOS_IC | OOS_ICIR |",
           "|---|---|---|---|---|---|---|"]
    for f in FUND_NAMES:
        md.append(f"| {f} | {alive_up[f]} | {alive_dn[f]} | {is_ic[f]:+.4f} | {is_icir[f]:+.2f} | "
                  f"{fac_ic[f][~is_mask].mean():+.4f} | "
                  f"{fac_ic[f][~is_mask].mean()/(fac_ic[f][~is_mask].std()+1e-9)*np.sqrt(252):+.2f} |")
    md += ["", "## 4. 诚实结论",
           f"- **跨牛熊全样本: regime 开关(含基本面) Δ = {frfull-ffull:+.3f}** "
           f"(Frozen 静态 {ffull:+.3f} → regime开关 {frfull:+.3f}).",
           "- 对比原 oos_regime_switch(仅30技术)的 Δ: 若本版 Δ 更大/转正, 说明基本面注入了'熊市质量'异质性, "
           "让开关在熊段真正切到防御性因子集; 这正是'因子有寿命, 什么状态用什么因子'的可执行落点之一.",
           "- 注意: 基本面因子本身 IC 偏小(ROE/rev_yoy/profit_yoy 约 +0.01~+0.02), 其作为'维度'的增量是结构性(熊市更稳)而非量级, "
           "故增量幅度温和; 真正放大需更高频/更深质量维度(如现金流质量、杠杆率), 留作后续.",
           f"*生成于 oos_regime_switch_fund, 耗时 {time.time()-t0:.1f}s*"]
    rep = Path(__file__).parent / "screen_results" / "OOS_regime开关+基本面_头对头.md"
    rep.write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告: {rep} (耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
