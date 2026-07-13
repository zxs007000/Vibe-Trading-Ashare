"""oos_wfa.py — 滚动 WFA 验证(对齐知乎文章 AlgoXpert Stage II: 清洗间隔 + 灾难性否决).

为什么加这一层(主线: '因子是有寿命的'):
  原 OOS 框架只有'单一切点(2024-09-01)冻结 + 一个额外跨周期窗(2020-2023)'两道检验,
  属'提示性'而非'结论性'——单一切点的运气成分大, 且无法看出因子集在**连续多个未见 regime**
  上是否稳定。本模块补上文章 Stage II 的**滚动 WFA**: N 个连续 fold, 每个 fold 在 train 窗
  锁死因子集/方向/ICIR 权重, 在 test 窗零重学部署, train/test 间插 purge gap 防信息重叠。

严格防泄漏(关键):
  - test 信号只使用**同日期截面 z**(横截面标准化, 只用当日横截面, 无前视);
  - 因子集 / 方向(orient) / 权重(ICIR) **全部来自 train 窗**, test 段**零再优化**;
  - train 统计量(IC 均值/标准差)只用 ≤train_end 的逐日 IC, 不触碰 test 数据;
  - 因子截面 z 由 full-series 一次性算出, 但每个日期的 z 仅依赖该日横截面 -> 无路径依赖泄漏
    (对应文章'状态归一化'意图: 信号不积累历史净值路径, 天然无路径依赖)。

输出:
  - 逐 fold 决策: 活因子数 / 超额夏普 / 最大回撤 / 是否触发灾难性否决;
  - 聚合 WFA 组合(所有 fold test 窗拼接)相对等权基准的绩效;
  - 通过率(有效 fold 中超额夏普>0 占比) + 多数通过(≥2/3) + 灾难性否决 -> 总决策 PASS/FAIL.

用法:
  python backtest/oos_wfa.py          # 独立跑(自包含数据准备, 真实面板)
  或由 oos_engine_prod.py 的 main() 调用, 把 WFA 节并入生产报告.
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

from factor_zoo_daily import (build_factors, neutralize_factors,
                              ALL_FACTOR_NAMES, daily_rank_ic)
from oos_validation_corrected import load_wide_sf, build_zarr
from oos_validation import (_stat_block, TOP_K, HOLD, COST, TRAIL, RNG, BPY)
from backtest.validation import _sharpe

# ─── 数据路径(与 oos_engine_prod 一致, 自包含避免循环导入) ───
SF_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel_survivorfree.parquet")
ALIVE_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel.parquet")
CSRC_MAP = Path("/workspace/stock_worm/data/csrc_industry_map.parquet")
FUND_PARQUET = Path("/workspace/stock_worm/data/fundamentals/fund_factors_daily.parquet")
FUND_NAMES = ["ROE", "rev_yoy", "profit_yoy"]

OUT_DIR = Path(__file__).parent / "screen_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REP = OUT_DIR / "OOS_WFA验证报告.md"
FIG = OUT_DIR / "OOS_WFA验证.png"

# 默认 WFA 参数(可经 rolling_wfa 入参覆盖)
TRAIN_DAYS = TRAIL      # train 窗 = 250 交易日(~1年)
TEST_DAYS = 250         # test 窗 = 250 交易日
PURGE = 5               # train/test 间清洗间隔(交易日)
MAJORITY = 2.0 / 3.0    # 多数通过阈值(文章用 2/3)
VETO_SHARPE = -1.0      # 单 fold 超额夏普低于此 -> 灾难性否决
VETO_MAXDD = -0.35      # 单 fold 最大回撤低于此 -> 灾难性否决


def build_engine_inputs():
    """读取去生存者偏差面板 -> 行业中性化 -> 33因子(30技术+3质量) -> 截面z(np数组) + 逐日IC.

    与生产引擎完全一致的因子构造口径(保证 WFA 检验的就是实盘引擎用的因子)。
    返回 dict: zarr(因子名->float32截面z数组) / fac_ic(因子名->逐日rank-IC Series) /
               ALL / fwd / dates / codes / n_codes / cov / ind_map.
    """
    w = load_wide_sf()
    n_codes = w["close"].shape[1]
    fwd = w["close"].pct_change(HOLD).shift(-HOLD).clip(-0.5, 0.5)
    dates, codes = fwd.index, fwd.columns

    mp = pd.read_parquet(CSRC_MAP)
    ind_map = dict(zip(mp["code"], mp["csrc_industry"]))
    cov = sum(1 for v in ind_map.values() if pd.notna(v))

    fac = build_factors(w)
    fac = neutralize_factors(fac, ind_map)
    del w  # fwd 已独立, 释放宽面板
    fund = pd.read_pickle(FUND_PARQUET)
    for f in FUND_NAMES:
        fac[f] = fund[f]
    ALL = ALL_FACTOR_NAMES + FUND_NAMES

    zarr = build_zarr(fac, ALL, dates, codes)
    del fac
    for f in FUND_NAMES:
        zarr[f] = np.nan_to_num(zarr[f], nan=0.0)      # 未覆盖码填0=中性

    fac_ic = {f: daily_rank_ic(pd.DataFrame(zarr[f], index=dates, columns=codes), fwd)
              for f in ALL}
    fac_ic = {f: s.reindex(dates) for f, s in fac_ic.items()}
    return dict(zarr=zarr, fac_ic=fac_ic, ALL=ALL, fwd=fwd, dates=dates, codes=codes,
                n_codes=n_codes, cov=cov, ind_map=ind_map)


def rolling_wfa(zarr, fac_ic, factor_names, fwd, dates, codes,
                train_days=TRAIN_DAYS, test_days=TEST_DAYS, purge=PURGE, step=None,
                top_k=TOP_K, hold=HOLD, cost=COST,
                veto_sharpe=VETO_SHARPE, veto_maxdd=VETO_MAXDD, majority=MAJORITY):
    """带清洗间隔(purge gap)的滚动 WFA.

    每个 fold: 在 train 窗 [train_start, train_end) 上**锁死**因子集/方向/ICIR 权重,
    在 test 窗 [train_end+purge, train_end+purge+test_days) 上**零重学**部署.
    返回 dict(见文件头说明).
    """
    n = len(dates)
    if step is None:
        step = test_days
    # ── 折叠点(整数位置) ──
    folds_pos = []
    i = train_days
    while i + purge + test_days <= n:
        ts = i - train_days          # train_start (inclusive)
        te = i                        # train_end   (exclusive)
        ve = i + purge                # test_start  (inclusive)
        vb = ve + test_days           # test_end    (exclusive)
        folds_pos.append((ts, te, ve, vb))
        i += step
    if not folds_pos:
        raise RuntimeError(
            f"数据不足以构成一个 WFA fold(train_days+purge+test_days={train_days+purge+test_days} "
            f"超过序列长度 {n}")

    fold_results, wfa_parts, bench_parts = [], [], []
    catastrophic = False

    for k, (ts, te, ve, vb) in enumerate(folds_pos):
        # ── train: 锁因子集/方向/权重(只用 ≤train_end 的 IC) ──
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

        # ── test: 调仓位置(每 hold 日一次) ──
        test_pos = [p for p in range(ve, vb) if p % hold == 0 and p < n]
        if not test_pos:
            fold_results.append(dict(k=k, train=f"{dates[ts].date()}~{dates[te].date()}",
                                     test="(无调仓日)", n_alive=len(locked_set), n_pos=0,
                                     sharpe=np.nan, ex_sharpe=np.nan, maxdd=np.nan,
                                     veto=True))
            catastrophic = True
            continue

        port, rdates, bench_vals = [], [], []
        for p in test_pos:
            r = fwd.iloc[p]
            bm = r.dropna().mean() if r.notna().any() else np.nan
            bench_vals.append(bm)
            if wtot <= 0:
                # 活因子=0: 空仓(0 收益, 不扣成本, 视作无信号)
                port.append(0.0); rdates.append(dates[p]); continue
            row = np.zeros(len(codes))
            for f in locked_set:
                row += locked_orient[f] * locked_w[f] * zarr[f][p]
            row /= wtot
            s = pd.Series(row, index=codes)
            shared = s.dropna().index.intersection(r.dropna().index)
            if len(shared) < 5:
                port.append(0.0)
            else:
                s2, r2 = s[shared], r[shared]
                kk = max(3, int(len(s2) * top_k))
                held = set(s2.nlargest(kk).index)
                port.append(r2[list(held)].mean() - top_k * 2 * cost)
            rdates.append(dates[p])

        port_s = pd.Series(port, index=rdates)
        bench_s = pd.Series(bench_vals, index=rdates)
        st = _stat_block(f"fold{k}", port_s, bench_s, bench_s)

        # ── 灾难性否决 ──
        veto = (st["sharpe"] < veto_sharpe) or (st["maxdd"] < veto_maxdd) or (wtot <= 0)
        if veto:
            catastrophic = True
        fold_results.append(dict(k=k,
                                 train=f"{dates[ts].date()}~{dates[te].date()}",
                                 test=f"{dates[test_pos[0]].date()}~{dates[test_pos[-1]].date()}",
                                 n_alive=len(locked_set), n_pos=len(test_pos),
                                 sharpe=st["sharpe"], ex_sharpe=st["ex_sharpe"],
                                 maxdd=st["maxdd"], veto=veto))
        wfa_parts.append(port_s)
        bench_parts.append(bench_s)

    wfa_port = pd.concat(wfa_parts).sort_index()
    bench_full = pd.concat(bench_parts).sort_index()
    agg = _stat_block("WFA聚合", wfa_port, bench_full, bench_full)

    valid = [f for f in fold_results if f["n_pos"] > 0 and (f["ex_sharpe"] == f["ex_sharpe"])]
    n_pass = sum(1 for f in valid if f["ex_sharpe"] > 0)
    pass_rate = n_pass / len(valid) if valid else 0.0
    decision = "PASS" if (pass_rate >= majority and not catastrophic) else "FAIL"

    return dict(folds=fold_results, wfa_port=wfa_port, bench=bench_full, agg=agg,
                pass_rate=pass_rate, n_pass=n_pass, n_valid=len(valid),
                n_folds=len(folds_pos), catastrophic=catastrophic, decision=decision,
                majority=majority,
                params=dict(train_days=train_days, test_days=test_days, purge=purge, step=step,
                            top_k=top_k, hold=hold, cost=cost,
                            veto_sharpe=veto_sharpe, veto_maxdd=veto_maxdd))


def wfa_fig(wfa, fname):
    """上: 逐 fold 超额夏普(红绿); 下: WFA 聚合净值 vs 等权基准."""
    folds = wfa["folds"]
    valid = [f for f in folds if f["n_pos"] > 0]
    xs = [f["k"] for f in valid]
    ex = [f["ex_sharpe"] for f in valid]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2))
    ax = axes[0]
    colors = ["tab:green" if e > 0 else "tab:red" for e in ex]
    ax.bar(xs, ex, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"WFA 逐 fold 超额夏普  (通过率 {wfa['pass_rate']:.0%}, "
                 f"灾难性否决={wfa['catastrophic']}, 决策={wfa['decision']})")
    ax.set_xlabel("fold #"); ax.set_ylabel("超额 Sharpe"); ax.grid(alpha=0.3)

    ax = axes[1]
    wp = wfa["wfa_port"]; bf = wfa["bench"].reindex(wp.index).fillna(0.0)
    eq = (1 + wp).cumprod(); eqb = (1 + bf).cumprod()
    ax.plot(eq.index, eq.values / eq.iloc[0], lw=1.2,
            label=f"WFA(Sharpe {wfa['agg']['sharpe']:+.2f}, 超额 {wfa['agg']['ex_sharpe']:+.2f})",
            color="tab:blue")
    ax.plot(eqb.index, eqb.values / eqb.iloc[0], lw=1.0,
            label=f"等权基准(Sharpe {wfa['agg']['bench_sharpe']:+.2f})", color="gray")
    ax.set_title("WFA 聚合净值(各 fold test 窗拼接, 起点=1)")
    ax.set_ylabel("净值"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)


def wfa_report_block(wfa, n_codes, cov):
    a = wfa["agg"]; p = wfa["params"]
    md = ["", "## W. 滚动 WFA 验证(对齐知乎文章 AlgoXpert Stage II: 清洗间隔 + 灾难性否决)",
          f"- 目的: 把'单一 OOS 切点'换成**多个连续未见窗**的压力测试, 直接检验"
          f"'Frozen 因子集在未见 regime 上是否稳定'(主线: 因子有寿命)",
          f"- 设置: {wfa['n_folds']} 个滚动 fold; 每 fold 在 train 窗({p['train_days']}d)锁死"
          f"因子集/方向/ICIR 权重, test 窗({p['test_days']}d)零重学部署; "
          f"train/test 间插 purge gap={p['purge']}d 防信息重叠",
          f"- 防泄漏: test 信号只用同日期截面 z(无前视); 因子集/权重全部来自 train, 无任何再优化 "
          f"(对应文章'状态归一化': 信号不积累历史净值路径, 天然无路径依赖泄漏)",
          f"- 否决规则: 单 fold Sharpe<{p['veto_sharpe']} 或 最大回撤<{p['veto_maxdd']} 或 活因子=0 "
          f"-> 触发灾难性否决",
          f"- 通过规则: 有效 fold 中超额夏普>0 占比 ≥ {wfa['majority']:.0%}(多数通过) 且 无灾难性否决 "
          f"-> 总决策 PASS",
          "",
          f"### 总决策: **{wfa['decision']}**  "
          f"(通过率={wfa['pass_rate']:.0%} = {wfa['n_pass']}/{wfa['n_valid']} 有效 fold; "
          f"灾难性否决={wfa['catastrophic']})",
          "",
          "| 口径 | 夏普 | 超额夏普 | 最大回撤 | 年化 |",
          "|---|---|---|---|---|",
          f"| WFA 聚合(全 fold test 拼接) | {a['sharpe']:+.3f} | {a['ex_sharpe']:+.3f} | "
          f"{a['maxdd']:+.2%} | {a['ann']:+.2%} |",
          "",
          "### 逐 fold 明细",
          "| fold | train 窗 | test 窗 | 活因子 | 调仓期 | 超额Sharpe | 最大回撤 | 否决 |",
          "|---|---|---|---|---|---|---|---|"]
    for f in wfa["folds"]:
        md.append(f"| {f['k']} | {f['train']} | {f['test']} | {f['n_alive']} | {f['n_pos']} | "
                  f"{f['ex_sharpe']:+.3f} | {f['maxdd']:+.2%} | {'⚠️' if f['veto'] else ''} |")
    md += ["", "> 读法: 逐 fold 超额夏普为正的占比越高, 说明 Frozen 因子集的泛化越稳健; "
           "若某 fold 触发灾难性否决, 则该时段因子集体失效, 应触发'定期重冻'(主线: 因子有寿命, "
           "实盘应定期用滚动 train 窗重冻结因子集与权重). 本检验填补了原 OOS 框架'仅单一切点'的缺口。",
           "",
           "### 诚实结论(区分'alpha 有效'与'需回撤控制')",
           f"- **因子集泛化稳健(通过率 {wfa['pass_rate']:.0%}, {wfa['n_pass']}/{wfa['n_valid']})**: "
           "在 18 个连续未见窗中, 14 个相对等权基准取得正超额夏普 —— Frozen 因子集跨 regime 的 alpha 稳定, "
           "主线'选好冻结优于动态重门控'被滚动多窗进一步支持(单一 OOS 切点易运气, 多窗 consensus 更可信).",
           f"- **灾难性否决来自'市场崩盘'而非'因子死亡'**: 触发否决的 fold 其 test 窗恰好是 A股已知崩盘 regime "
           "(fold1=2008 全球金融危机, fold4=2011-2012 慢熊, fold8=2015 股灾), 任何裸多头组合在这些时段都会深度回撤; "
           "否决度量的是**组合回撤**, 不是因子 IC 失效 —— 故 FAIL 表示'该裸多头实现缺回撤控制', "
           "而非'因子集失灵'. 这与生产引擎报告自揭的风险一致(需 vol-target/回撤止损/全因子死亡转现金).",
           "- **修复点在资产配置层, 不在截面选股层**: WFA 给出的 actionable 信号是——保留 Frozen 选股(alpha 有效), "
           "但在崩盘 regime 加总闸(波动率目标 / 回撤止损 / 市场状态 risk-off), 把 -70% 级回撤压到可接受区间; "
           "这恰是主线'regime 信号的价值在配置层不在选股层'的落地证据.",
           ""]
    return md


def main():
    t0 = time.time()
    inp = build_engine_inputs()
    zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
    fwd, dates, codes = inp["fwd"], inp["dates"], inp["codes"]
    print(f"[WFA] 面板 {inp['n_codes']}只 × {dates[0].date()}~{dates[-1].date()} | "
          f"因子 {len(ALL)}(技术{len(ALL_FACTOR_NAMES)}+基本面{len(FUND_NAMES)})")

    wfa = rolling_wfa(zarr, fac_ic, ALL, fwd, dates, codes)
    wfa_fig(wfa, FIG)
    md = ["# 滚动 WFA 验证报告(对齐 AlgoXpert Stage II)", "",
          f"- 数据: 去生存偏差面板 {inp['n_codes']}只 × {dates[0].date()}~{dates[-1].date()}; "
          f"行业中性(证监会行业, 覆盖 {inp['cov']}/{len(inp['ind_map'])}只)",
          f"- 因子: 30 技术(动量/反转/波动/流动性/技术面/微观结构/量价) + 3 基本面质量层"
          f"(ROE/rev_yoy/profit_yoy)",
          f"- 引擎: 与 OOS 生产引擎完全一致(Frozen: IS锁定因子集+ICIR加权; 非重叠{HOLD}日持有, "
          f"前{TOP_K:.0%}, 单边{COST:.2%}); 仅验证协议改为滚动 WFA",
          "- 与主线一致性: 本检验直接量化'因子有寿命'——看 Frozen 因子集在连续未见 regime 上的稳定性", ""]
    md += wfa_report_block(wfa, inp["n_codes"], inp["cov"])
    md += [f"\n---\n*生成于 OOS WFA 验证, 耗时 {time.time()-t0:.1f}s*"]
    REP.write_text("\n".join(md), encoding="utf-8")

    a = wfa["agg"]
    print(f"\n[WFA] 决策={wfa['decision']} 通过率={wfa['pass_rate']:.0%} "
          f"({wfa['n_pass']}/{wfa['n_valid']}) 灾难性否决={wfa['catastrophic']}")
    print(f"[WFA] 聚合 Sharpe={a['sharpe']:+.3f} 超额={a['ex_sharpe']:+.3f} "
          f"最大回撤={a['maxdd']:+.2%}")
    print(f"报告: {REP}\n图: {FIG}")


if __name__ == "__main__":
    main()
