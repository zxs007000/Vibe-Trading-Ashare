"""oos_engine_prod.py — 生产版 OOS 选股引擎(质量层 + Frozen 落地)

把经严格 OOS 验证的两项结论落成**可交付的生产引擎**:
  (1) 基本面质量层: ROE / rev_yoy / profit_yoy(全面板覆盖~1828/1803, 未覆盖填0中性) —
      已证在中立口径下给 30 技术因子组合带来 +0.02~+0.06 夏普的干净增益(oos_fundamental_check).
  (2) 最优策略 = Frozen(冻结 IS 活因子集 + IS锁定 ICIR 加权):
      在 30 技术因子上, 中性化口径 Frozen 是四策略(A/B/Frozen/Random)里唯一跑赢等权基准的,
      且反复证优于动态门控(oos_validation_corrected / oos_regime_switch 系列).

引擎做法(严格 OOS, 无未来泄漏):
  - 因子选股集与权重全部在 IS(≤2024-09-01)锁定; OOS 零重学习, 仅用实时信号组合.
  - 输出**实际持仓**: 每个调仓日 top-K(前30%)股票(多维 DataFrame + CSV), 以及'最新一期的买入清单'.
  - 报告同时给: 全样本绩效 / 严格 OOS 段绩效(诚实) / 相对等权基准与随机top-K 的超额.

用法:
  python backtest/oos_engine_prod.py
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

import oos_validation as OOS
import oos_validation_corrected as M
from factor_zoo_daily import build_factors, neutralize_factors, ALL_FACTOR_NAMES, daily_rank_ic
from oos_wfa import rolling_wfa, wfa_fig, wfa_report_block, build_engine_inputs

SF_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel_survivorfree.parquet")
ALIVE_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel.parquet")
CSRC_MAP = Path("/workspace/stock_worm/data/csrc_industry_map.parquet")
FUND_PARQUET = Path("/workspace/stock_worm/data/fundamentals/fund_factors_daily.parquet")
FUND_NAMES = ["ROE", "rev_yoy", "profit_yoy"]

SPLIT = OOS.SPLIT
TOP_K, HOLD, COST, TRAIL, RNG = OOS.TOP_K, OOS.HOLD, OOS.COST, OOS.TRAIL, OOS.RNG

OUT_DIR = Path(__file__).parent / "screen_results"
HOLD_CSV = OUT_DIR / "OOS生产引擎_持仓.csv"
FIG = OUT_DIR / "OOS生产引擎_净值.png"
REP = OUT_DIR / "OOS生产引擎_报告.md"
WFA_FIG = OUT_DIR / "OOS生产引擎_WFA.png"


def backtest_with_holdings(signal_w, fwd_w, top_k=TOP_K, hold=HOLD, cost=COST):
    """同 OOS.long_only_topk, 但额外返回每调仓日持仓集合(用于落地实际持仓)."""
    dates = signal_w.index
    port, rdates, held_list = [], [], []
    for i in range(len(dates)):
        if i % hold != 0:
            continue
        d = dates[i]; s = signal_w.loc[d]; r = fwd_w.loc[d]
        shared = s.dropna().index.intersection(r.dropna().index)
        if len(shared) < 5:
            continue
        s, r = s[shared], r[shared]
        k = max(3, int(len(s) * top_k))
        held = set(s.nlargest(k).index)
        pr = r[list(held)].mean() - top_k * 2 * cost
        port.append(pr); rdates.append(d); held_list.append(held)
    return pd.Series(port, index=rdates), held_list


def main():
    t0 = time.time()
    inp = build_engine_inputs()
    zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
    fwd, dates, codes = inp["fwd"], inp["dates"], inp["codes"]
    n_codes, cov = inp["n_codes"], inp["cov"]
    ind_map = inp["ind_map"]
    print(f"[生产引擎] 面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()} | 分界 {SPLIT.date()}")
    print(f"因子 {len(ALL)}(技术{len(ALL_FACTOR_NAMES)}+基本面{len(FUND_NAMES)})/zarr 完成, 算逐日 IC ...", flush=True)

    ic_mean = {f: fac_ic[f].rolling(TRAIL).mean() for f in ALL}
    ic_std = {f: fac_ic[f].rolling(TRAIL).std() for f in ALL}
    is_mask = dates < SPLIT
    is_ic = {f: fac_ic[f][is_mask].mean() for f in ALL}
    is_icir = {f: (is_ic[f] / (fac_ic[f][is_mask].std() + 1e-9)) * np.sqrt(252)
               for f in ALL}
    # === IS 锁定: 因子集与权重只在 IS 决定, OOS 零重学习 ===
    frozen_set = [f for f in ALL if is_ic[f] > 0 and is_icir[f] > 0]
    print(f"  IS 活因子(锁定) {len(frozen_set)}/{len(ALL)}: {', '.join(frozen_set)}")

    # Frozen 信号(IS锁定 is_icir 加权, 与所有对比脚本一致)
    sigF = OOS.build_signal(zarr, ic_mean, ic_std, ALL, dates, codes,
                            allowed=frozen_set, gate=False, weight_src="is", is_icir=is_icir)
    rng_rand = np.random.default_rng(RNG)
    sigR = OOS.build_signal(zarr, ic_mean, ic_std, ALL, dates, codes,
                            allowed=ALL, gate=False, weight_src="trailing",
                            rng=rng_rand, rand_gate=True)

    # 回测(持仓 + 收益)
    portF, held_list = backtest_with_holdings(sigF, fwd)
    portR = OOS.long_only_topk(sigR, fwd)

    # 基准 / 随机
    rebal_pos = list(range(0, len(dates), HOLD))
    bench_full = fwd.iloc[rebal_pos].mean(axis=1).dropna()
    def slice_win(s):
        return s[s.index >= SPLIT] if (len(s) and s.index[0] < SPLIT) else s
    bench_oos = slice_win(bench_full); rnd_oos = slice_win(portR)

    sF = OOS._stat_block("Frozen 生产(全样本)", portF, bench_full, portR)
    sF_oos = OOS._stat_block("Frozen 生产(OOS)", slice_win(portF), bench_oos, rnd_oos)
    sR = OOS._stat_block("随机top-K基线", portR, bench_full, portR)

    # ── 滚动 WFA 验证(对齐知乎文章 AlgoXpert Stage II) ──
    wfa = rolling_wfa(zarr, fac_ic, ALL, fwd, dates, codes)
    wfa_fig(wfa, WFA_FIG)
    print(f"[WFA] 决策={wfa['decision']} 通过率={wfa['pass_rate']:.0%} "
          f"({wfa['n_pass']}/{wfa['n_valid']}) 灾难性否决={wfa['catastrophic']} "
          f"聚合Sharpe={wfa['agg']['sharpe']:+.3f} 超额={wfa['agg']['ex_sharpe']:+.3f}")

    # 持仓矩阵
    hmat = pd.DataFrame(0, index=portF.index, columns=codes, dtype=int)
    for d, held in zip(portF.index, held_list):
        hmat.loc[d, list(held)] = 1
    hmat.index.name = "rebal_date"
    hmat.to_csv(HOLD_CSV)

    # 最新一期买入清单
    last_d = portF.index[-1]; last_held = sorted(hmat.columns[hmat.loc[last_d] == 1])
    print(f"\n最新调仓日 {last_d.date()} 持仓 {len(last_held)} 只, 见 {HOLD_CSV}")

    # 净值曲线
    eq = (1 + portF).cumprod()
    eq_bench = (1 + bench_full).cumprod()
    plt.figure(figsize=(12, 5.5))
    plt.plot(eq.index, eq.values / eq.iloc[0], lw=1.3, label=f"Frozen生产(全{len(ALL)}因子, Sharpe {sF['sharpe']:+.2f})", color="tab:green")
    plt.plot(eq_bench.index, eq_bench.values / eq_bench.iloc[0], lw=1.0, label=f"等权基准(Sharpe {sF['bench_sharpe']:+.2f})", color="gray")
    axv = SPLIT
    plt.axvline(axv, color="k", ls="--", lw=0.8)
    plt.text(axv, 0.5, " OOS分界", rotation=90, fontsize=8, va="center")
    plt.title("OOS 生产引擎净值(全样本, 起点归一化=1)")
    plt.ylabel("净值"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG, dpi=110); plt.close()

    # 报告
    def row(name, s):
        return f"| {name} | {s['sharpe']:+.3f} | {s['ann']:+.2%} | {s['maxdd']:+.2%} | {s['cum']:+.1%} | {s['ex_sharpe']:+.3f} |"
    md = ["# OOS 生产引擎报告(质量层 + Frozen 落地)", "",
          f"- 数据: 去生存偏差面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()}; 分界 {SPLIT.date()}",
          f"- 因子: 30 技术(动量/反转/波动/特质波动/流动性/技术面/微观结构/量价) + 3 基本面质量层"
          f"(ROE/rev_yoy/profit_yoy, 全面板覆盖~1828/1803, 未覆盖填0中性)",
          f"- 行业中性: cninfo 证监会行业(覆盖 {cov}/{len(ind_map)}只)",
          f"- 策略 = Frozen: 因子集与权重在 IS(≤{SPLIT.date()})**锁定**, OOS 零重学习; IS活因子 {len(frozen_set)} 个, IS-ICIR 加权",
          f"- 回测: 非重叠{HOLD}日持有, 前{TOP_K:.0%}, 单边{COST:.2%}; 引擎与所有 OOS 验证脚本完全一致(无未来泄漏)",
          "- 落地内容: 实际持仓时间序列(CSV) + 最新买入清单 + 净值曲线 + 诚实绩效(全样本/OOS双口径)",
          "", "## 1. 绩效(双口径)", "",
          "| 口径 | 夏普 | 年化 | 最大回撤 | 累计 | 超额夏普 |",
          "|---|---|---|---|---|---|",
          row("Frozen 生产(全样本)", sF),
          row("Frozen 生产(OOS 2024-09起)", sF_oos),
          row("随机top-K基线", sR),
          "", "![净值](OOS生产引擎_净值.png)", "",
          "## 2. 最新一期买入清单",
          f"- 调仓日: **{last_d.date()}** | 持仓数: **{len(last_held)}** | 完整序列见 `OOS生产引擎_持仓.csv`",
          "", "```",
          ", ".join(last_held),
          "```", "",
          "## 3. 诚实结论 & 与主线一致性",
          f"- **质量层有效**: 3 基本面并入后, 中立口径下 33 因子组合相对 30 技术因子夏普 +0.02~+0.06"
          "(oos_fundamental_check), 本引擎即该结论的落地(Frozen + 质量层).",
          f"- **Frozen 是验证过的最优**: 中性化口径下 Frozen 是 A/B/Frozen/Random 四策略中唯一跑赢等权基准的"
          "(oos_validation_corrected); 反复证优于动态门控. 故生产引擎采用 Frozen, 不做状态动态门控.",
          f"- **因子集 IS 锁定 / OOS 零重学习**: 选股集与权重只用 ≤{SPLIT.date()} 数据决定, OOS 段仅用实时信号组合, 无未来泄漏; "
          "绝对夏普不可外推, 相对排序(跑赢等权基准)才是真信号.",
          "- **选股层不做 regime 因子开关**: 方向A已证, A股截面因子空间缺乏状态正交因子, 切因子集的开关≈静态Frozen无增量; "
          "regime 信号的价值在资产配置层(ETF轮动总闸), 不在截面选股层. 故本引擎'冻结IS胜者'而非'动态切因子'.",
          "- **风险提示**: 单一 OOS regime(2024-09起约96%为牛市)使 OOS 段绩效提示性非结论性; "
          "因子有寿命, 实盘应定期(如年度)用滚动 IS 窗口重冻结因子集与权重(本引擎已参数化, 改 SPLIT/TRAIL 即可重训).",
          ]

    md += wfa_report_block(wfa, n_codes, cov)

    md += [f"*生成于 OOS 生产引擎, 耗时 {time.time()-t0:.1f}s*"]
    REP.write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告: {REP}\n图: {FIG}\n持仓CSV: {HOLD_CSV}")
    print(f"\n[绩效] 全样本 Sharpe={sF['sharpe']:+.3f} 年化={sF['ann']:+.2%} 最大回撤={sF['maxdd']:+.2%} 超额={sF['ex_sharpe']:+.3f}")
    print(f"[绩效] OOS段 Sharpe={sF_oos['sharpe']:+.3f} 超额={sF_oos['ex_sharpe']:+.3f}")


if __name__ == "__main__":
    main()
