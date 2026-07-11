"""oos_regime_switch_ortho.py — regime 维度开关 + 状态正交因子(激活选股层开关)

是 oos_regime_switch / oos_regime_switch_fund 的延伸. 前两版证明:
  30 技术(同源同质) 与 +3 基本面(跨regime皆活) 都让开关 ≈ 静态 Frozen,
  根因是因子池跨 regime 缺乏异质性('什么状态用什么因子'无不同因子可切).

本脚本注入**状态正交因子**(factor_zoo_ortho: beta_60牛 / lowvol_60·lowivol_60·liq_stress_20·
distress_60 熊)后, 重测:
  regime 开关(38 = 30技术+3基本面+5正交) 是否真正压过 静态 Frozen(38)?
  -> 关键看新正交因子在 per-regime 存活检验里是否呈现'牛专/熊专', 从而给开关异质性.

测法严格 OOS, 与 oos_regime_switch 完全一致(IS锁定/OOS零重学, 等权指数 vs MA200 作regime信号).
对比(均在 38 因子集内):
  - 静态 Frozen(38): IS全活因子集, 不切regime
  - Frozen regime开关(38): IS内分牛/熊存活集, OOS按实时regime切 frozen_up/frozen_dn
  - 并列 A/B/Random 作上下文
并额外给出: (a) 纯正交因子(5)的 per-regime 存活明细; (b) 静态Frozen(38) vs 静态Frozen(33) 增量,
看正交因子本身是否在全样本贡献(无论开关与否).

用法:
  python backtest/oos_regime_switch_ortho.py
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
from factor_zoo_ortho import build_ortho_factors, ORTHO_NAMES, ORTHO_FAMILY

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
    """regime 维度开关版信号: 每个调仓日按实时 regime 选 frozen_up / frozen_dn 组."""
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
    print(f"[regime开关+正交因子] 面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()} | 分界 {SPLIT.date()} | MA{MA_WIN}")
    print(f"  IS 内 risk-on 占比 = {regime_on[dates < SPLIT].mean():.1%} | OOS 内 = {regime_on[dates >= SPLIT].mean():.1%}")

    mp = pd.read_parquet(CSRC_MAP)
    ind_map = dict(zip(mp["code"], mp["csrc_industry"]))
    cov = sum(1 for v in ind_map.values() if pd.notna(v))

    fac = build_factors(w)
    ortho = build_ortho_factors(w)        # 复用同一宽表(避免二次 load 撑内存)
    del w
    fac = neutralize_factors(fac, ind_map)
    # 并入全面板基本面(覆盖 ~1828/1803, 未覆盖填0中性)
    fund = pd.read_pickle(FUND_PARQUET)
    for f in FUND_NAMES:
        fac[f] = fund[f]
    # 并入状态正交因子(纯价格/成交额, 全面板覆盖)
    for f in ORTHO_NAMES:
        fac[f] = ortho[f]
    del ortho
    ALL = ALL_FACTOR_NAMES + FUND_NAMES + ORTHO_NAMES
    zarr = M.build_zarr(fac, ALL, dates, codes)
    del fac
    for f in FUND_NAMES:                       # 基本面未覆盖码填0(中性), 防 NaN 传播挤出选股池
        zarr[f] = np.nan_to_num(zarr[f], nan=0.0)
    print(f"因子(技术{len(ALL_FACTOR_NAMES)}+基本面{len(FUND_NAMES)}+正交{len(ORTHO_NAMES)}={len(ALL)})/中性化/zarr 完成, 算逐日 IC ...", flush=True)

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
    print(f"  仅牛活 {len(set(frozen_up)-set(frozen_dn))}: " + ", ".join(sorted(set(frozen_up)-set(frozen_dn))))
    print(f"  仅熊活 {len(set(frozen_dn)-set(frozen_up))}: " + ", ".join(sorted(set(frozen_dn)-set(frozen_up))))
    print("  正交因子 per-regime 明细:")
    for f in ORTHO_NAMES:
        print(f"    {f}({ORTHO_FAMILY[f]}): 牛={alive_up[f]} 熊={alive_dn[f]} | IS_IC {is_ic[f]:+.4f} IS_ICIR {is_icir[f]:+.2f}")

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

    rebal_pos = list(range(0, len(dates), HOLD))
    bench_full = fwd.iloc[rebal_pos].mean(axis=1).dropna()
    rng_g = np.random.default_rng(RNG)
    rnd_full = OOS.random_topk(fwd, rng_g)
    def slice_win(s): return s[s.index >= SPLIT] if s.index[0] < SPLIT else s
    bench_oos = slice_win(bench_full); rnd_oos = slice_win(rnd_full)

    ports = {
        "A 无选择": OOS.long_only_topk(sigA, fwd),
        "B 状态选择": OOS.long_only_topk(sigB, fwd),
        "Frozen 静态(38)": OOS.long_only_topk(sigF, fwd),
        "Frozen regime开关(38)": OOS.long_only_topk(sigFr, fwd),
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
    for k in ("Frozen 静态(38)", "Frozen regime开关(38)"):
        s = stats_full[k]
        print(f"{k:<22}{s['sharpe']:>+11.3f}{s['maxdd']:>+14.2%}")

    fstat = stats["Frozen 静态(38)"]["sharpe"]; frstat = stats["Frozen regime开关(38)"]["sharpe"]
    ffull = stats_full["Frozen 静态(38)"]["sharpe"]; frfull = stats_full["Frozen regime开关(38)"]["sharpe"]
    print(f"\nregime开关 vs 静态Frozen(38): OOS Δ = {frstat - fstat:+.3f} | 全样本 Δ = {frfull - ffull:+.3f}")
    # 正交因子本身的全样本增量(静态Frozen(38) vs 已知静态Frozen(33)=+0.495 全样本)
    print(f"  (参考: 静态Frozen(33) 全样本=+0.495; 本 静态Frozen(38) 全样本={ffull:+.3f} -> 正交边际={ffull-0.495:+.3f})")

    # 报告
    md = ["# regime 维度开关 + 状态正交因子 · 严格 OOS 头对头",
          "",
          f"- 数据: 去生存偏差面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()}; 分界 {SPLIT.date()}; IS锁定/OOS零重学",
          f"- 因子池: 30 技术 + 3 基本面(全面板~1828) + 5 状态正交(beta_60牛 / lowvol_60·lowivol_60·liq_stress_20·distress_60 熊), 共 {len(ALL)}",
          f"- regime 代理: 等权市场指数 > 其 MA{MA_WIN} = risk-on; IS内 risk-on 占 {regime_on[dates < SPLIT].mean():.1%}, "
          f"OOS内占 {regime_on[dates >= SPLIT].mean():.1%}",
          f"- 维度开关: IS内分'牛市活'({len(frozen_up)})/'熊市活'({len(frozen_dn)}), OOS按实时regime切 frozen_up/frozen_dn",
          f"- 行业中性用 cninfo 证监会行业(覆盖 {cov}/{len(ind_map)}只)",
          "- 动机: 前两版(仅30技术 / 30+3基本面)开关均≈静态Frozen, 因因子池跨regime同质(几乎全'两态皆活'). "
          "现注入**理论状态正交因子**, 检验其是否真呈现'牛专/熊专', 从而给开关异质性、激活选股层'什么状态用什么因子'.",
          "",
          "## 1. OOS 头对头(2024-09 起)", "",
          "| 策略 | OOS夏普 | 年化 | 最大回撤 | 超额夏普 |",
          "|---|---|---|---|---|"]
    for k, s in stats.items():
        md.append(f"| {k} | {s['sharpe']:+.3f} | {s['ann']:+.2%} | {s['maxdd']:+.2%} | {s['ex_sharpe']:+.3f} |")
    md += ["",
           f"## 2. 核心问题: regime 维度开关(含正交因子)是否优于静态 Frozen?",
           f"- **Frozen 静态(38) = {fstat:+.3f} vs Frozen regime开关(38) = {frstat:+.3f} (OOS Δ = {frstat-fstat:+.3f})**.",
           f"- **全样本(含IS, 跨牛熊) Frozen 静态(38) = {ffull:+.3f} vs regime开关(38) = {frfull:+.3f} (Δ = {frfull-ffull:+.3f})**.",
           "- OOS 窗口(2024-09 起)是单边牛市, regime 几乎恒 risk-on, OOS 上开关≈静态; 真正增量在牛熊全周期, 以'全样本 Δ'为准.",
           "- 若全样本 Δ>0 且明显大于前两版(30:+? / 33:-0.001): 说明正交因子注入了跨regime异质性, 开关在熊段切到防御集真有增量.",
           "",
           "## 2.5 全样本(含IS)补充表", "",
           "| 策略 | 全样本夏普 | 全样本最大回撤 |",
           "|---|---|---|"]
    for k in ("Frozen 静态(38)", "Frozen regime开关(38)"):
        s = stats_full[k]
        md.append(f"| {k} | {s['sharpe']:+.3f} | {s['maxdd']:+.2%} |")
    md += ["", "## 3. 状态正交因子的 per-regime 存活(关键诊断)",
           "",
           "| 因子 | 家族 | 牛活 | 熊活 | IS_IC | IS_ICIR | OOS_IC | OOS_ICIR | 状态特性 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for f in ORTHO_NAMES:
        char = "牛专" if (alive_up[f] and not alive_dn[f]) else ("熊专" if (alive_dn[f] and not alive_up[f]) else ("两态皆活" if (alive_up[f] and alive_dn[f]) else "两态皆死"))
        md.append(f"| {f} | {ORTHO_FAMILY[f]} | {alive_up[f]} | {alive_dn[f]} | {is_ic[f]:+.4f} | "
                  f"{is_icir[f]:+.2f} | {fac_ic[f][~is_mask].mean():+.4f} | "
                  f"{fac_ic[f][~is_mask].mean()/(fac_ic[f][~is_mask].std()+1e-9)*np.sqrt(252):+.2f} | {char} |")
    md += ["", "## 4. 什么状态用什么因子(IS内 per-regime 存活, 全集)",
           f"- **牛市活因子({len(frozen_up)}):** " + ", ".join(frozen_up),
           f"- **熊市活因子({len(frozen_dn)}):** " + ", ".join(frozen_dn),
           f"- **仅牛市活({len(set(frozen_up)-set(frozen_dn))}):** " + ", ".join(sorted(set(frozen_up)-set(frozen_dn))),
           f"- **仅熊市活({len(set(frozen_dn)-set(frozen_up))}):** " + ", ".join(sorted(set(frozen_dn)-set(frozen_up))),
           "", "## 5. 诚实结论",
           f"- **跨牛熊全样本: regime 开关(38) Δ = {frfull-ffull:+.3f}** (Frozen 静态 {ffull:+.3f} → regime开关 {frfull:+.3f}).",
           f"- **静态 Frozen(38) 全样本 = {ffull:+.3f}, 相对静态 Frozen(33)=+0.495 的正交边际 = {ffull-0.495:+.3f}** "
           "(正交因子本身是否在全样本贡献, 无论开关与否).",
           "- 判读: 若'仅牛活/仅熊活'名单里出现了正交因子(尤其 beta_60 进仅牛、lowvol/liq_stress/distress 进仅熊), "
           "说明注入成功——开关终于有'不同状态不同因子'可切; 此时若全样本 Δ>0, 即证实'状态正交因子'是激活选股层开关的钥匙.",
           "- 反之若正交因子仍'两态皆活', 则连价格构造的防御/暴露因子都在 A 股全样本上同质, 选股层开关需更彻底的 regime 划分(如波动率机制/流动性机制)而非因子层面.",
           f"*生成于 oos_regime_switch_ortho, 耗时 {time.time()-t0:.1f}s*"]
    rep = Path(__file__).parent / "screen_results" / "OOS_regime开关+正交因子_头对头.md"
    rep.write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告: {rep} (耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
