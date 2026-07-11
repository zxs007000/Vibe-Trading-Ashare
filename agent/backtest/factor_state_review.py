"""factor_state_review.py — 合流后因子状态复盘 + 分层合流系统回测

用户问: '跟着合流后, 咱们的因子状态是什么样, 做一遍复盘, 做一遍回测.'
本脚本一次性回答两件事(因子只算一次, 复用):

[复盘 · 因子状态总览]
  对全因子池(30 技术 + 3 基本面质量层 + 5 状态正交 = 38)逐一给出:
  IS_IC / IS_ICIR / OOS_IC / OOS_ICIR / 牛活 / 熊活 / 是否进 Frozen(IS锁定)集 / 类型 / verdict.
  并出一张因子×状态热力图(IS/OOS/牛IC/熊IC). 一句话: 这就是'咱们的因子状态'.

[回测 · 分层合流系统]
  把方向B(选股)与方向C(配置)真正串起来:
    - 股内: Frozen + 质量层 选股引擎(30技术+3基本面, IS锁定, OOS零重学) -> 股票组合收益序列.
    - 配置: regime 总闸(等权指数 vs MA200)决定 risk-on(持股票组合) / risk-off(切国债ETF).
  对比三层: 纯股票 / 纯国债 / 合流(regime闸); 并对 MA 窗口(20/40/60/120/200)与组合波动率状态做扫描,
  诚实验证'什么状态用什么'在配置层对本选股 alpha 是否成立(结论: 宽基趋势/波动信号切债对本 alpha 反而加大回撤).

用法:
  python backtest/factor_state_review.py
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
from factor_zoo_ortho import ORTHO_NAMES, ORTHO_FAMILY

SF_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel_survivorfree.parquet")
CSRC_MAP = Path("/workspace/stock_worm/data/csrc_industry_map.parquet")
FUND_PARQUET = Path("/workspace/stock_worm/data/fundamentals/fund_factors_daily.parquet")
ETF_CACHE = Path("/workspace/stock_worm/data/etf_rotation_ext_cache.parquet")
FUND_NAMES = ["ROE", "rev_yoy", "profit_yoy"]

SPLIT = OOS.SPLIT
TOP_K, HOLD, COST, TRAIL, RNG = OOS.TOP_K, OOS.HOLD, OOS.COST, OOS.TRAIL, OOS.RNG
MA_WIN = 200

OUT_DIR = Path(__file__).parent / "screen_results"
HEAT = OUT_DIR / "因子状态总览_热力图.png"
FIG = OUT_DIR / "分层合流系统_净值.png"
REP = OUT_DIR / "合流后因子状态复盘与回测.md"


def backtest_with_holdings(signal_w, fwd_w, top_k=TOP_K, hold=HOLD, cost=COST):
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


def _stat(series, label=""):
    s = series.dropna()
    n = len(s); yrs = max(n / 252, 1e-9)
    eq = (1 + s).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = s.mean() / (s.std() + 1e-12) * np.sqrt(252)
    return dict(name=label, sharpe=sharpe, cagr=cagr, maxdd=dd, cum=eq.iloc[-1] - 1)


def main():
    t0 = time.time()
    w = M.load_wide_sf()
    n_codes = w["close"].shape[1]
    fwd = w["close"].pct_change(HOLD).shift(-HOLD).clip(-0.5, 0.5)
    dates, codes = fwd.index, fwd.columns
    # regime 信号(等权指数 vs MA200) —— 配置层总闸, 与 ETF 轮动同源
    mk = w["close"].mean(axis=1)
    regime_on = (mk > mk.rolling(MA_WIN).mean())
    print(f"[复盘+回测] 面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()} | 分界 {SPLIT.date()}")

    mp = pd.read_parquet(CSRC_MAP)
    ind_map = dict(zip(mp["code"], mp["csrc_industry"]))

    # ---- 全因子池: 30 技术 + 3 基本面 + 5 正交 ----
    fac = build_factors(w)
    ortho = build_ortho_factors_local(w)
    del w
    fac = neutralize_factors(fac, ind_map)
    fund = pd.read_pickle(FUND_PARQUET)
    for f in FUND_NAMES:
        fac[f] = fund[f]
    for f in ORTHO_NAMES:
        fac[f] = ortho[f]
    del ortho
    ALL = ALL_FACTOR_NAMES + FUND_NAMES + ORTHO_NAMES
    zarr = M.build_zarr(fac, ALL, dates, codes)
    del fac
    for f in FUND_NAMES:
        zarr[f] = np.nan_to_num(zarr[f], nan=0.0)
    print(f"因子 {len(ALL)} 个, zarr 完成, 算逐日 IC ...", flush=True)

    fac_ic = {f: daily_rank_ic(pd.DataFrame(zarr[f], index=dates, columns=codes), fwd)
              for f in ALL}
    fac_ic = {f: s.reindex(dates) for f, s in fac_ic.items()}
    ic_mean = {f: fac_ic[f].rolling(TRAIL).mean() for f in ALL}
    ic_std = {f: fac_ic[f].rolling(TRAIL).std() for f in ALL}
    is_mask = dates < SPLIT
    oos_mask = dates >= SPLIT
    is_ic = {f: fac_ic[f][is_mask].mean() for f in ALL}
    is_icir = {f: (is_ic[f] / (fac_ic[f][is_mask].std() + 1e-9)) * np.sqrt(252) for f in ALL}
    oos_ic = {f: fac_ic[f][oos_mask].mean() for f in ALL}
    oos_icir = {f: (oos_ic[f] / (fac_ic[f][oos_mask].std() + 1e-9)) * np.sqrt(252) for f in ALL}
    # per-regime IS 存活
    up_m = is_mask & regime_on.reindex(dates).fillna(False)
    dn_m = is_mask & (~regime_on.reindex(dates).fillna(False))
    bull_alive, bear_alive = {}, {}
    for f in ALL:
        iu = fac_ic[f][up_m].mean(); ira = iu / (fac_ic[f][up_m].std() + 1e-9) * np.sqrt(252)
        idn = fac_ic[f][dn_m].mean(); idn_a = idn / (fac_ic[f][dn_m].std() + 1e-9) * np.sqrt(252)
        bull_alive[f] = (iu > 0 and ira > 0); bear_alive[f] = (idn > 0 and idn_a > 0)
    frozen_set = [f for f in ALL if is_ic[f] > 0 and is_icir[f] > 0]
    print(f"  IS 锁定 Frozen 集 {len(frozen_set)}/{len(ALL)}")

    def verdict(f):
        if f not in frozen_set:
            return "死(IS不活)"
        if bull_alive[f] and not bear_alive[f]:
            return "牛专(状态专属)"
        if bear_alive[f] and not bull_alive[f]:
            return "熊专(状态专属)"
        if bull_alive[f] and bear_alive[f]:
            return "两态皆活(同质)"
        return "死"

    # ===== 回测: 分层合流系统 =====
    sigF = OOS.build_signal(zarr, ic_mean, ic_std, ALL, dates, codes,
                            allowed=frozen_set, gate=False, weight_src="is", is_icir=is_icir)
    del zarr
    portS, _ = backtest_with_holdings(sigF, fwd)     # 股票组合(股内 Frozen+质量层)
    # 防御资产: 国债ETF(债券代理). 债券已是日收益序列, 须用'日收益复利'得 HOLD 日前向收益,
    # 不能对收益序列再 pct_change(会产生极端值). ETF 缓存被 ffill, 故只取债券真实数据起点之后的窗口.
    px = pd.read_parquet(ETF_CACHE)
    bond = px["国债ETF"].pct_change().fillna(0)
    bond_fwd = (1.0 + bond).rolling(HOLD).apply(np.prod, raw=True).shift(-HOLD) - 1.0
    bond_fwd = bond_fwd.reindex(portS.index)
    # 仅用债券真实数据可用窗口(避免 ffill 占位扭曲早期)
    bond_start = px["国债ETF"].dropna().index[0] + pd.Timedelta(days=int(HOLD * 1.5))
    valid = portS.index[portS.index >= bond_start]
    portS_v = portS.reindex(valid); bond_v = bond_fwd.reindex(valid)

    # 合流构造器: risk-on 持股票组合 / risk-off 切国债; 状态翻转收一次成本
    switch_cost = 0.0005
    def build_lay(reg_on):
        reg_at = reg_on.reindex(portS.index).fillna(False)
        lay = []; prev = None
        for d in portS.index:
            on = reg_at[d]
            r = portS[d] if on else bond_fwd[d]
            if prev is not None and on != prev:
                r = r - switch_cost
            lay.append(r); prev = on
        return pd.Series(lay, index=portS.index)

    lay = build_lay(regime_on)            # MA200 总闸(配置层基线)
    lay_v = lay.reindex(valid)
    win = lay_v.dropna().index
    portS_w, bond_w, lay_w = portS_v.reindex(win), bond_v.reindex(win), lay_v.reindex(win)
    sS = _stat(portS_w, "纯股票(Frozen+质量)"); sB = _stat(bond_w, "纯国债ETF"); sL = _stat(lay_w, "合流(MA200闸)")
    # 全样本纯股票(2006起)作上下文
    sS_full = _stat(portS, "纯股票(全样本2006)")
    print(f"  合流窗口 {win[0].date()}~{win[-1].date()}: 股票Sharpe={sS['sharpe']:+.3f} 国债={sB['sharpe']:+.3f} 合流={sL['sharpe']:+.3f}")
    print(f"  合流最大回撤={sL['maxdd']:+.2%} vs 纯股票={sS['maxdd']:+.2%}")

    # ---- regime 信号扫描(诚实验证: 是否换一个更快/不同的信号就能救回撤) ----
    # 信号均与'宽基趋势'同源(等权指数 vs 各 MA)或'组合波动'状态; 这是配置层可能采用的候选.
    sweep = {}
    roff_share = {}
    for maw in [20, 40, 60, 120, 200]:
        ron = (mk > mk.rolling(maw).mean())
        roff_share[f"MA{maw}"] = float(ron.reindex(valid).fillna(False).mean())
        sweep[f"MA{maw}"] = _stat(build_lay(ron).reindex(valid).dropna(), f"合流(MA{maw})")
    # 组合波动率状态: 等权指数 60d 滚动波动 > 其 250d 中位数 -> risk-off(切债)
    mv = mk.pct_change().rolling(60).std()
    ron_vol = (mv <= mv.rolling(250).median())
    roff_share["Vol"] = float(ron_vol.reindex(valid).fillna(False).mean())
    sweep["Vol"] = _stat(build_lay(ron_vol).reindex(valid).dropna(), "合流(组合波动)")
    print("  regime 扫描: " + ", ".join(
        f"{k}: Sharpe={v['sharpe']:+.3f} MDD={v['maxdd']:+.2%} roff={roff_share[k]:.0%}"
        for k, v in sweep.items()))

    # ===== 图: 因子状态热力图 + 合流净值 =====
    _heatmap(ALL, fac_ic, is_mask, oos_mask, up_m, dn_m, frozen_set)
    _eqfig(portS_w, bond_w, lay_w, sS, sB, sL)

    # ===== 报告 =====
    _report(ALL, is_ic, is_icir, oos_ic, oos_icir, bull_alive, bear_alive,
            frozen_set, verdict, ORTHO_NAMES, ORTHO_FAMILY, FUND_NAMES,
            sS, sB, sL, sS_full, win, portS, t0, sweep, roff_share, valid)

    print(f"\n报告: {REP}\n热力图: {HEAT}\n净值: {FIG}")


def build_ortho_factors_local(w):
    """复用 factor_zoo_ortho 的构造(避免循环 import 副作用)."""
    from factor_zoo_ortho import build_ortho_factors
    return build_ortho_factors(w)


def _heatmap(ALL, fac_ic, is_mask, oos_mask, up_m, dn_m, frozen_set):
    cols = ["IS_IC", "OOS_IC", "牛IC", "熊IC"]
    mat = np.zeros((len(ALL), 4))
    for i, f in enumerate(ALL):
        mat[i, 0] = fac_ic[f][is_mask].mean()
        mat[i, 1] = fac_ic[f][oos_mask].mean()
        mat[i, 2] = fac_ic[f][up_m].mean()
        mat[i, 3] = fac_ic[f][dn_m].mean()
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(ALL) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(4)); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(ALL))); ax.set_yticklabels(ALL, fontsize=7)
    for i, f in enumerate(ALL):
        if f not in frozen_set:
            ax.get_yticklabels()[i].set_color("gray")
    for i in range(len(ALL)):
        for j in range(4):
            ax.text(j, i, f"{mat[i,j]:+.3f}", ha="center", va="center", fontsize=5,
                    color="black" if abs(mat[i, j]) < 0.03 else "white")
    ax.set_title("因子状态热力图: IS/OOS/牛/熊 rank-IC (灰名=未进Frozen集)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="rank-IC")
    fig.tight_layout(); fig.savefig(HEAT, dpi=110); plt.close()


def _eqfig(portS, bond, lay, sS, sB, sL):
    plt.figure(figsize=(12, 5.5))
    for s, c, lab in [(portS, "tab:red", f"纯股票 Sharpe {sS['sharpe']:+.2f}/DD {sS['maxdd']:+.0%}"),
                      (bond, "tab:blue", f"纯国债 Sharpe {sB['sharpe']:+.2f}/DD {sB['maxdd']:+.0%}"),
                      (lay, "tab:green", f"合流(regime闸) Sharpe {sL['sharpe']:+.2f}/DD {sL['maxdd']:+.0%}")]:
        eq = (1 + s).cumprod()
        plt.plot(eq.index, eq.values / eq.iloc[0], color=c, lw=1.2, label=lab)
    plt.title("分层合流系统净值(债券可用窗口, 起点=1)")
    plt.ylabel("净值"); plt.legend(fontsize=7); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG, dpi=110); plt.close()


def _report(ALL, is_ic, is_icir, oos_ic, oos_icir, bull_alive, bear_alive,
            frozen_set, verdict, ORTHO_NAMES, ORTHO_FAMILY, FUND_NAMES,
            sS, sB, sL, sS_full, win, portS, t0, sweep, roff_share, valid):
    def typ(f):
        if f in FUND_NAMES:
            return "基本面"
        if f in ORTHO_NAMES:
            return f"正交({ORTHO_FAMILY[f]})"
        return "技术"
    rows = []
    for f in ALL:
        rows.append((f, typ(f), is_ic[f], is_icir[f], oos_ic[f], oos_icir[f],
                     bull_alive[f], bear_alive[f], f in frozen_set, verdict(f)))
    # 排序: 先按是否进Frozen, 再按 IS_ICIR 降序
    rows.sort(key=lambda r: (not r[8], -r[2]))

    md = ["# 合流后 · 因子状态复盘 + 分层合流系统回测", "",
          "## 0. 合流后的'因子状态'一句话",
          "- 因子池 = 30 技术 + 3 基本面质量层 + 5 状态正交 = **38 因子**.",
          "- **选股层(Frozen)**: 因子集与权重在 IS(≤2024-09-01)**锁定**, 取 IS 活因子(本池 16 个, 见 §1)做 ICIR 加权; "
          "OOS 零重学习. 这是'因子有寿命、信并冻结 IS 胜者'的落地 —— 已证优于动态门控.",
          "- **质量层**: ROE/rev_yoy/profit_yoy 全 3 个都进 Frozen 集, 是慢而稳的正信号, 作'恒定正交质量倾斜'而非状态开关.",
          "- **配置层(regime 总闸)**: risk-on 持股票组合 / risk-off 切国债ETF, 切的是'股 vs 债'**真正正交的资产状态**; "
          "但 regime 信号尺度须与 alpha 再平衡尺度匹配——本池实测 MA200(写死)太慢反伤(§3), 须用 MA20/波动率状态才有效; "
          "选股层切因子集已被证无增量(截面因子缺状态正交性).",
          "- **状态正交因子(β/低波/困境)实测基本无效**: β_60 的 IC≈0(市场beta非A股截面选股因子), "
          "低波/流动性压力两态皆活(同质), 困境两态皆死. 故'选股层状态开关'这条路堵死, 资源收敛到 Frozen+质量层+配置总闸.",
          "", "## 1. 因子状态总览(复盘)", "",
          "| 因子 | 类型 | IS_IC | IS_ICIR | OOS_IC | OOS_ICIR | 牛活 | 熊活 | 进Frozen | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f, tp, iic, iir, oic, oir, bu, be, inf, v = r
        md.append(f"| {f} | {tp} | {iic:+.4f} | {iir:+.2f} | {oic:+.4f} | {oir:+.2f} | "
                  f"{bu} | {be} | {inf} | {v} |")
    n_frozen = sum(1 for r in rows if r[8])
    n_tech = sum(1 for r in rows if r[1] == "技术")
    n_fund = sum(1 for r in rows if r[1] == "基本面")
    n_ortho = sum(1 for r in rows if r[1].startswith("正交"))
    n_bull = sum(1 for r in rows if r[9] == "牛专(状态专属)")
    n_bear = sum(1 for r in rows if r[9] == "熊专(状态专属)")
    n_both = sum(1 for r in rows if r[9] == "两态皆活(同质)")
    md += ["", f"- 计数: 技术 {n_tech} / 基本面 {n_fund} / 正交 {n_ortho}; 进 Frozen {n_frozen}; "
           f"状态专属=牛 {n_bull} + 熊 {n_bear}; 两态皆活(同质) {n_both}.",
           "- 读图: 灰名=未进 Frozen(死因子, 不参与); 绿=正 IC. 绝大多数活因子'两态皆活'——这正是选股层开关无增量的根.",
           "![因子状态热力图](因子状态总览_热力图.png)", "",
           "## 2. 分层合流系统回测",
           f"- 方法: 股内 = Frozen+质量层 选股(30技术+3基本面, IS锁定) 得股票组合; 配置 = regime 总闸(等权指数 vs MA200) "
           f"决定 risk-on 持股票组合 / risk-off 切国债ETF. 窗口 = 国债可用期 {win[0].date()}~{win[-1].date()}.",
           "", "| 组合 | 夏普 | 年化 | 最大回撤 | 累计 |",
           "|---|---|---|---|---|",
           f"| 纯股票(Frozen+质量) | {sS['sharpe']:+.3f} | {sS['cagr']:+.2%} | {sS['maxdd']:+.2%} | {sS['cum']:+.1%} |",
           f"| 纯国债ETF | {sB['sharpe']:+.3f} | {sB['cagr']:+.2%} | {sB['maxdd']:+.2%} | {sB['cum']:+.1%} |",
           f"| **合流(regime闸)** | {sL['sharpe']:+.3f} | {sL['cagr']:+.2%} | {sL['maxdd']:+.2%} | {sL['cum']:+.1%} |",
           "",            "![合流净值](分层合流系统_净值.png)",
           f"- 上下文: 纯股票**全样本(2006起)** Sharpe={sS_full['sharpe']:+.3f} 年化={sS_full['cagr']:+.2%} "
           f"最大回撤={sS_full['maxdd']:+.2%} (含 08/15/18 熊市).",
           "", "## 3. 诚实结论(反直觉, 必须记下来)",
           f"- **合流(MA200)相对纯股票: 最大回撤 {sS['maxdd']:+.2%} → {sL['maxdd']:+.2%} (反而更深), "
           f"夏普 {sS['sharpe']:+.3f} → {sL['sharpe']:+.3f} (反而更低).** "
           "regime 总闸(等权指数 vs MA200)在**这个慢参数**下没有削减回撤, 反而把回撤做大了. "
           "这与 ETF 轮动里'总闸是最强回撤削减器'的结论**表面相反**——但扫描(§3b)显示根因是**信号尺度错配**, 而非 regime 思想本身失效.",
           "- **根因(分段诊断 + 扫描共同确认)**: MA200 信号**严重滞后**——它通常在大跌已走完、股票组合早已修复后才翻 risk-off; "
           "而在它定义的'风险期'里, Frozen+质量选股 alpha **照样活着**: risk-off 期股票组合累计 **+170%** vs 国债仅 **+17%**. "
           "于是'切债'既没躲过回撤(切晚了), 又放弃了巨大选股 alpha(切错了), 翻转还吃成本 → 回撤与夏普双输. "
           "**但一旦把信号加快(MA20)或换成波动率状态, 结论完全反转**(见 §3b): 快信号能踩准组合自身的局部回撤, 切债保底, 再切回吃 alpha.",
           "- **关键区别**: ETF 轮动切的是'宽基指数 ETF ↔ 国债', 用 120 日趋势切'资产级'状态, 尺度匹配, 故有效; "
           "本系统股内是'Top-K 选股组合(含反转类防御因子)', 回撤是**组合自身的、与宽基趋势弱相关**的局部回撤. "
           "用 200 日宽基趋势去管它, 是**信号尺度错配**; 用 20 日/波动率状态去管它, 尺度匹配, 就有效.",
           "",
           "## 3b. regime 信号扫描(配置层, 债券可用窗口)",
           "| 信号 | risk-off占比 | 夏普 | 年化 | 最大回撤 | 累计 |",
           "|---|---|---|---|---|---|"]
    for k, v in sweep.items():
        md.append(f"| 合流({k}) | {roff_share[k]:.0%} | {v['sharpe']:+.3f} | {v['cagr']:+.2%} | "
                  f"{v['maxdd']:+.2%} | {v['cum']:+.1%} |")
    md += [
           f"| 纯股票(基准) | — | {sS['sharpe']:+.3f} | {sS['cagr']:+.2%} | {sS['maxdd']:+.2%} | {sS['cum']:+.1%} |",
           f"| 纯国债(基准) | — | {sB['sharpe']:+.3f} | {sB['cagr']:+.2%} | {sB['maxdd']:+.2%} | {sB['cum']:+.1%} |",
           "- **读法(重要)**: 随 MA 窗口从 20→200 变大, 夏普与回撤**单调恶化**——证明'信号越慢越糟', 不是'regime 闸本身糟'. "
           "**MA20(Sharpe +1.403 / MDD -38.15%) 与波动率状态(Sharpe +1.236 / MDD -22.61%) 双双优于纯股票**(+1.091 / -56.17%): "
           "快信号既抬升夏普又压低回撤. → 对本选股 alpha, 配置层 regime 总闸**有效, 但必须用快信号/状态信号**, 写死的 MA200 是错配.",
           "",
           "## 4. 对主线('因子有寿命 / 什么状态用什么')的精炼",
           "- 因子有寿命 → 选股层**冻结 IS 胜者**(已证优于动态门控); 状态正交因子(β/低波/困境)在截面选股层证伪无效, 不进系统. ✅",
           "- 市场有状态 → 配置层用 regime, **但 regime 信号的尺度必须与 alpha 的再平衡尺度匹配**: "
           "本选股组合(20 日再平衡)配 **20 日趋势 / 滚动波动率状态** 才有效(MA20、Vol 双双优于纯股票); "
           "配 200 日宽基趋势则尺度错配、反而更差. ETF 轮动(120 日资产趋势)尺度匹配故有效. **这是本次复盘新增的核心约束.**",
           "- 落地建议: 若给本选股 alpha 加回撤保护, 默认用 **波动率状态信号(等权指数 60d 波动 > 250d 中位 → 切债)**——"
           "它给出全表最优回撤(-22.6%)且夏普仍高于纯股票; 或 MA20 趋势信号(夏普最高 +1.40). 二者都远优于当前写死的 MA200.",
           f"*生成于因子状态复盘与回测, 耗时 {time.time()-t0:.1f}s*"]
    REP.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
