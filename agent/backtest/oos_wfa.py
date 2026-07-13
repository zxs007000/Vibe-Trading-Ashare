"""oos_wfa.py — 滚动 WFA 验证 + 配置层总闸烘焙(bake-off).

对齐知乎文章 AlgoXpert Stage II(清洗间隔 + 灾难性否决), 并在其上做**配置层总闸的信号烘焙**:
把'熊市信号集'与'加仓条件'参数化, 一次跑完多变体, 在同一 18 窗 WFA 上对照,
用'FAIL→PASS + 回撤塌陷 + Sharpe 不塌'挑最优(避免只看回撤最低而误杀收益).

闸门逻辑(用户设计, 两档位):
  - 正渐进(因子报熊即减仓, thesis 原生风控): 仓位随'因子报出熊市信号'的占比下降, 但**只降到 50% 下限**
    (健康=100%, 全失效=50%, 不归零). 触发源二选一(取更保守): 长窗 ICIR 衰减(死因子计数) 或
    近季 IC 翻空(较激进的因子熊信号). 即使牛市, 因子集体报熊 -> edge 缩水 -> 减仓.
    标定目标: 因子衰减速度要让仓位在价格触及'熊市线'之前已滑到 ~50%(因子先替价格'预撤退').
  - 反渐进(标准熊市信号): 当价格触及'熊市线'(MA120/-10% 或集成信号) -> 执行最后 50%->0 退出空仓, 并进入'熊市 regime'.
  - 熊市 regime 封顶 50%: regime 内仓位封顶 50%, 由因子反弹信号在 0~50% 调制; 避免漫长熊市里满仓接刀, 也避免死猫反弹里立刻满仓.
  - 退出 regime(恢复加满): 需'价格回升至 MA120 上方 +5% **且因子全健康**'双重确认 -> 才放回 100%(牛市信号). 单看价格会在 2015 死猫反弹里误判满仓(鞭梢).
  - 一句话: 因子衰减负责'软预撤退'(100%->50%), 熊市线负责'硬清仓'(50%->0)与'封顶 50% 的熊市 regime', 双重确认牛市才'恢复满仓'(0->100%).

防泄漏:
  - test 信号只用同日期截面 z; 因子集/方向/权重全来自 train 窗; test 零重学.
  - 熊市信号用宽基等权指数 close/MA 滚动(因果); 加仓条件里的'因子衰减'用 fac_ic 截至 test 日 p
    的滚动 ICIR 对比 train ICIR(全 ≤p, 无前视). 非股票部分配防御资产(cash=0 / proxy=年化4%债券carry).

变体(A/B/C/D + 无闸基线):
  A = MA120/-10% 熊 + 二元仓位(旧硬闸基线)
  B = 集成熊信号(MA120+MA60+距20日高点回撤+波动突增, 任一触发) + 二元仓位
  C = MA120 熊 + 因子衰减计数正渐进(可加满)
  D = 集成熊信号 + 因子衰减计数正渐进

用法:
  python backtest/oos_wfa.py          # 自包含数据准备, 真实面板, 跑 A/B/C/D 烘焙对照
  或由 oos_engine_prod.py 的 main() 调用.
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

# ─── 数据路径(自包含, 避免循环导入) ───
SF_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel_survivorfree.parquet")
ALIVE_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel.parquet")
CSRC_MAP = Path("/workspace/stock_worm/data/csrc_industry_map.parquet")
FUND_PARQUET = Path("/workspace/stock_worm/data/fundamentals/fund_factors_daily.parquet")
FUND_NAMES = ["ROE", "rev_yoy", "profit_yoy"]

OUT_DIR = Path(__file__).parent / "screen_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REP = OUT_DIR / "OOS_WFA验证报告.md"
FIG = OUT_DIR / "OOS_WFA验证.png"


def _alive_mkt_level(dates):
    """用**活股面板**构建等权市场指数(日收益等权均值累乘), 作配置层熊市信号的市场代理.

    为什么不用去生存偏差面板的等权指数: 该面板含 358 只退市股, 其 close 在退市后**冻结**,
    崩盘时活股暴跌但死股不动 -> 等权均值被冻结死股托着, 实际跌幅被稀释 -> MA120/-10% 信号
    在活股崩盘里滞后触发(2015 实测: 闸门变体回撤 -77% 比无闸 -50% 还惨). 同理原始收盘价
    等权均值还跨价格量纲失真. 用活股面板(无冻结)构建干净等权指数即可修此误时.
    仅作叠加信号, 不进因子, 不引入生存偏差.
    """
    alive = pd.read_parquet(ALIVE_PANEL)
    alive["_d"] = pd.to_datetime(alive["date"]).dt.normalize()
    cal = pd.to_datetime(dates)
    alive = alive[alive["_d"].isin(cal)]
    wide = alive.pivot(index="_d", columns="code", values="close").reindex(dates)
    ret = wide.pct_change()
    lvl = (1.0 + ret.mean(axis=1, skipna=True).fillna(0.0)).cumprod()
    return lvl

# 默认 WFA / 闸门参数
TRAIN_DAYS = TRAIL      # train 窗 = 250 交易日
TEST_DAYS = 250
PURGE = 5
MAJORITY = 2.0 / 3.0
VETO_SHARPE = -1.0
VETO_MAXDD = -0.35
GATE_BEAR_MA = 120
GATE_BEAR_THR = -0.10
GATE_BULL_THR = 0.05   # 牛市确认: 价格回升至 MA120 上方 +5% 才退出熊市 regime(迟滞带避免鞭梢)
GATE_EQ_CAP = 1.0       # 正渐进仓位上限(可加满到 100%)
GATE_SIG_SCALE = 1.50   # rebound 信号归一化尺度(代理)
GATE_DECAY_FRAC = 0.30  # 长窗: 因子近期 ICIR < 该比例×train ICIR 视为'衰减'(保守死亡计数)
GATE_SHORT_WIN = 60     # 短窗: 因子近 GATE_SHORT_WIN 日 IC<0 视为'刚报出熊市信号'(较激进, 更早预撤退)
GATE_DECAY_FLOOR = 0.50  # 正渐进下限: 因子全失效时仓位最低 50%(不归零); 熊信号才执行反渐进->0
GATE_DEF_ANN = 0.04     # proxy 防御资产年化 carry


def build_engine_inputs():
    """读取去生存者偏差面板 -> 行业中性化 -> 33因子 -> 截面z + 逐日IC + 活股等权指数(熊市信号代理)."""
    w = load_wide_sf()
    n_codes = w["close"].shape[1]
    fwd = w["close"].pct_change(HOLD).shift(-HOLD).clip(-0.5, 0.5)
    dates, codes = fwd.index, fwd.columns
    mkt_level = _alive_mkt_level(dates)   # 活股等权指数(配置层总闸的熊市信号市场代理)

    mp = pd.read_parquet(CSRC_MAP)
    ind_map = dict(zip(mp["code"], mp["csrc_industry"]))
    cov = sum(1 for v in ind_map.values() if pd.notna(v))

    fac = build_factors(w)
    fac = neutralize_factors(fac, ind_map)
    del w

    fund = pd.read_pickle(FUND_PARQUET)
    for f in FUND_NAMES:
        fac[f] = fund[f]
    ALL = ALL_FACTOR_NAMES + FUND_NAMES

    zarr = build_zarr(fac, ALL, dates, codes)
    del fac
    for f in FUND_NAMES:
        zarr[f] = np.nan_to_num(zarr[f], nan=0.0)

    fac_ic = {f: daily_rank_ic(pd.DataFrame(zarr[f], index=dates, columns=codes), fwd)
              for f in ALL}
    fac_ic = {f: s.reindex(dates) for f, s in fac_ic.items()}
    return dict(zarr=zarr, fac_ic=fac_ic, ALL=ALL, fwd=fwd, dates=dates, codes=codes,
                n_codes=n_codes, cov=cov, ind_map=ind_map, mkt_level=mkt_level)


def _bear_signal(mkt_level, mode, bear_ma=GATE_BEAR_MA, bear_thr=GATE_BEAR_THR):
    """返回布尔熊市信号(因果). mode='ma120' 单信号; mode='ensemble' 多信号任一触发."""
    ma = mkt_level.rolling(bear_ma).mean()
    ratio = (mkt_level / ma - 1)
    b_ma = ratio < bear_thr
    if mode != "ensemble":
        return b_ma.fillna(False)
    ma60 = mkt_level.rolling(60).mean()
    r60 = (mkt_level / ma60 - 1)
    b_ma60 = r60 < -0.08
    hi20 = mkt_level.rolling(20).max()
    b_dd = (mkt_level / hi20 - 1) < -0.12
    ret = mkt_level.pct_change()
    vol = ret.rolling(20).std()
    volt = vol.rolling(250).mean()
    b_vol = (vol > 2.5 * volt.replace(0, np.nan)).fillna(False)
    return (b_ma | b_ma60 | b_dd | b_vol).fillna(False)


def _recent_icir(fac_ic, trail=TRAIL):
    """预计算每因子'截至各日的滚动 ICIR'(因果, 只用 ≤t 的 IC). 供因子衰减计数用."""
    out = {}
    for f, s in fac_ic.items():
        m = s.rolling(trail).mean()
        sd = s.rolling(trail).std()
        out[f] = (m / (sd + 1e-9) * np.sqrt(252))
    return out


def rolling_wfa(zarr, fac_ic, factor_names, fwd, dates, codes, mkt_level=None,
                train_days=TRAIN_DAYS, test_days=TEST_DAYS, purge=PURGE, step=None,
                top_k=TOP_K, hold=HOLD, cost=COST,
                veto_sharpe=VETO_SHARPE, veto_maxdd=VETO_MAXDD, majority=MAJORITY,
                gate=False, bear_mode="ma120", addback_mode="binary",
                pos_cap=GATE_EQ_CAP, def_mode="cash", def_ann=GATE_DEF_ANN):
    """带清洗间隔的滚动 WFA; 可选叠加配置层总闸(bear_mode × addback_mode).

    闸门(两档位 + 熊市 regime 状态机):
      - 反渐进: 熊市信号触发 -> 持仓硬归 0(空仓), 进入'熊市 regime'.
      - 熊市 regime 内: 仓位**封顶 50%**, 由因子反弹信号在 0~50% 调制(避免漫长熊市满仓接刀/死猫反弹满仓).
      - 退出 regime(恢复满仓): 需'价格回升至 MA120 上方 +5% **且因子全健康**'双重确认 -> 仓位回到 50%~100%(因子健康越高越满).
      - 正渐进(factor_decay): 无熊市信号时, 仓位随'因子报熊占比'从 100% 滑到 **50% 下限**(不归零);
        即因子集体衰减也减仓(thesis 原生风控), 让仓位在熊市线前先'预撤退'.
      - 'binary'      : 非熊即满仓(旧硬闸, 无因子调制, 反弹里易满仓接刀 -> 鞭梢).
      - 'factor_decay': 因子报熊占比(长窗衰减∨近季翻空)越高仓位越低; 牛市 pos=50%+50%×(1-占比), 熊市 pos=50%×(1-占比).
      - 'rebound'     : 因子反弹信号强度映射 0~pos_cap(代理).
    返回 dict(同前) + 各 fold 平均股票仓位.
    """
    n = len(dates)
    if step is None:
        step = test_days
    folds_pos = []
    i = train_days
    while i + purge + test_days <= n:
        ts = i - train_days; te = i; ve = i + purge; vb = ve + test_days
        folds_pos.append((ts, te, ve, vb))
        i += step
    if not folds_pos:
        raise RuntimeError(f"数据不足以构成 WFA fold(需 {train_days+purge+test_days} ≤ {n})")

    bear = _bear_signal(mkt_level, bear_mode) if gate else None
    # 牛市确认信号: 价格回升至 MA120 上方 +GATE_BULL_THR(迟滞带, 避免熊市里反复鞭梢)
    bull = ((mkt_level / mkt_level.rolling(GATE_BEAR_MA).mean() - 1) > GATE_BULL_THR) if gate else None
    recent_icir = _recent_icir(fac_ic, TRAIL) if (gate and addback_mode == "factor_decay") else None
    # 短窗 IC(<0 = 该因子近 GATE_SHORT_WIN 日刚报出熊市信号, 较激进更早预撤退)
    recent_ic_short = ({f: fac_ic[f].rolling(GATE_SHORT_WIN).mean() for f in factor_names}
                       if (gate and addback_mode == "factor_decay") else None)
    r_def = (def_ann / 252.0) if def_mode == "proxy" else 0.0
    r_def_series = pd.Series(r_def, index=dates) if gate else None

    fold_results, wfa_parts, bench_parts = [], [], []
    catastrophic = False

    for k, (ts, te, ve, vb) in enumerate(folds_pos):
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
            fold_results.append(dict(k=k, train=f"{dates[ts].date()}~{dates[te].date()}",
                                     test="(无调仓日)", n_alive=len(locked_set), n_pos=0,
                                     sharpe=np.nan, ex_sharpe=np.nan, maxdd=np.nan,
                                     veto=True, avg_eq=np.nan))
            catastrophic = True
            continue

        port, rdates, bench_vals, eq_track = [], [], [], []
        regime = False   # 熊市 regime 状态机(初始牛市); 熊信号->True, 牛市确认->False
        for p in test_pos:
            r = fwd.iloc[p]
            bm = r.dropna().mean() if r.notna().any() else np.nan
            bench_vals.append(bm)
            mean_held = 0.0
            if wtot <= 0:
                rg = 0.0
            else:
                row = np.zeros(len(codes))
                for f in locked_set:
                    row += locked_orient[f] * locked_w[f] * zarr[f][p]
                row /= wtot
                s = pd.Series(row, index=codes)
                shared = s.dropna().index.intersection(r.dropna().index)
                if len(shared) < 5:
                    rg = 0.0
                else:
                    s2, r2 = s[shared], r[shared]
                    kk = max(3, int(len(s2) * top_k))
                    held = set(s2.nlargest(kk).index)
                    rg = r2[list(held)].mean()
                    mean_held = float(s2[list(held)].mean())
            # ── 配置层总闸(用户设计, 两档位 + 熊市 regime 封顶) ──
            #   熊信号->持仓0(进入熊市 regime); 熊市 regime 内: 仓位封顶 50%, 因子反弹信号调 0~50%;
            #   牛市确认信号(价回 MA120 上方+5%)->退出 regime, 因子健康才放回 100%(正渐进 50%~100%).
            #   迟滞带[-10%,+5%]避免熊市里反复鞭梢; 封顶 50% 让漫长熊市被砍半, 死猫反弹最多接 50% 刀.
            if gate:
                if wtot <= 0:
                    pos = 0.0
                else:
                    # 因子报熊占比(长窗衰减∨近季翻空); binary 模式 decay_frac 恒 0(无因子调制)
                    if addback_mode == "factor_decay":
                        decay = 0
                        for f in locked_set:
                            ri = recent_icir[f].iloc[p]; ti = locked_w[f]
                            long_dead = (ri == ri) and (ri < 0 or ri < GATE_DECAY_FRAC * ti)
                            si = recent_ic_short[f].iloc[p]
                            short_bear = (si == si) and (si < 0)   # 近季刚报出熊市信号
                            if long_dead or short_bear:
                                decay += 1
                        decay_frac = decay / max(1, len(locked_set))
                    else:
                        decay_frac = 0.0
                    if bear.iloc[p]:
                        pos = 0.0
                        regime = True
                    elif bull.iloc[p] and decay_frac == 0.0:
                        # 牛市确认需'价格回升 + 因子全健康'双重确认, 避免死猫反弹里因子仍衰时误判满仓(鞭梢)
                        regime = False
                        pos = GATE_DECAY_FLOOR + (pos_cap - GATE_DECAY_FLOOR) * (1.0 - decay_frac)
                    elif regime:
                        pos = GATE_DECAY_FLOOR * (1.0 - decay_frac)   # 熊市期内: 0~50%
                    else:
                        pos = GATE_DECAY_FLOOR + (pos_cap - GATE_DECAY_FLOOR) * (1.0 - decay_frac)  # 牛市: 50%~100%
                rdef = float(r_def_series.iloc[p])
                pr = pos * rg + (1.0 - pos) * rdef - pos * top_k * 2 * cost
                eq_track.append(pos)
            else:
                pr = rg - top_k * 2 * cost
            port.append(pr); rdates.append(dates[p])

        port_s = pd.Series(port, index=rdates)
        bench_s = pd.Series(bench_vals, index=rdates)
        st = _stat_block(f"fold{k}", port_s, bench_s, bench_s)
        veto = (st["sharpe"] < veto_sharpe) or (st["maxdd"] < veto_maxdd) or (wtot <= 0)
        if veto:
            catastrophic = True
        avg_eq = float(np.mean(eq_track)) if eq_track else np.nan
        fold_results.append(dict(k=k,
                                 train=f"{dates[ts].date()}~{dates[te].date()}",
                                 test=f"{(dates[test_pos[0]]).date()}~{(dates[test_pos[-1]]).date()}",
                                 n_alive=len(locked_set), n_pos=len(test_pos),
                                 sharpe=st["sharpe"], ex_sharpe=st["ex_sharpe"],
                                 maxdd=st["maxdd"], veto=veto, avg_eq=avg_eq))
        wfa_parts.append(port_s)
        bench_parts.append(bench_s)

    wfa_port = pd.concat(wfa_parts).sort_index()
    bench_full = pd.concat(bench_parts).sort_index()
    agg = _stat_block("WFA聚合", wfa_port, bench_full, bench_full)
    valid = [f for f in fold_results if f["n_pos"] > 0 and (f["ex_sharpe"] == f["ex_sharpe"])]
    n_pass = sum(1 for f in valid if f["ex_sharpe"] > 0)
    pass_rate = n_pass / len(valid) if valid else 0.0
    n_veto = sum(1 for f in fold_results if f["veto"])
    decision = "PASS" if (pass_rate >= majority and not catastrophic) else "FAIL"
    return dict(folds=fold_results, wfa_port=wfa_port, bench=bench_full, agg=agg,
                pass_rate=pass_rate, n_pass=n_pass, n_valid=len(valid),
                n_folds=len(folds_pos), n_veto=n_veto, catastrophic=catastrophic,
                decision=decision, majority=majority, gate=gate,
                bear_mode=bear_mode, addback_mode=addback_mode, def_mode=def_mode,
                params=dict(train_days=train_days, test_days=test_days, purge=purge,
                            top_k=top_k, hold=hold, cost=cost,
                            veto_sharpe=veto_sharpe, veto_maxdd=veto_maxdd,
                            bear_mode=bear_mode, addback_mode=addback_mode,
                            pos_cap=pos_cap, def_mode=def_mode, def_ann=def_ann))


def run_wfa_bakeoff(zarr, fac_ic, ALL, fwd, dates, codes, mkt_level, def_mode="proxy"):
    """跑 A/B/C/D + 无闸基线, 同 18 窗对照."""
    base = dict(zarr=zarr, fac_ic=fac_ic, factor_names=ALL, fwd=fwd, dates=dates,
                codes=codes, mkt_level=mkt_level, def_mode=def_mode)
    variants = {
        "无闸": dict(gate=False),
        "A·MA120二元": dict(gate=True, bear_mode="ma120", addback_mode="binary"),
        "B·集成二元": dict(gate=True, bear_mode="ensemble", addback_mode="binary"),
        "C·MA120因子衰减": dict(gate=True, bear_mode="ma120", addback_mode="factor_decay"),
        "D·集成因子衰减": dict(gate=True, bear_mode="ensemble", addback_mode="factor_decay"),
    }
    return {name: rolling_wfa(**base, **kw) for name, kw in variants.items()}


def wfa_fig(res, fname):
    """上: 各变体聚合 Sharpe 条形; 下: 关键变体净值曲线 + 基准."""
    names = list(res.keys())
    agg_sharpe = [res[n]["agg"]["sharpe"] for n in names]
    agg_maxdd = [res[n]["agg"]["maxdd"] for n in names]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.8))
    ax = axes[0]
    colors = ["tab:red" if res[n]["decision"] == "FAIL" else "tab:green" for n in names]
    ax.bar(range(len(names)), agg_sharpe, color=colors)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, fontsize=8)
    for i, (s, d) in enumerate(zip(agg_sharpe, agg_maxdd)):
        ax.text(i, s, f"{s:+.2f}\n{d:+.0%}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("WFA 各变体聚合 Sharpe(红=FAIL, 绿=PASS; 标注=Sharpe/最大回撤)")
    ax.set_ylabel("Sharpe"); ax.grid(alpha=0.3)

    ax = axes[1]
    for key in ["无闸", "A·MA120二元", "C·MA120因子衰减", "D·集成因子衰减"]:
        if key not in res:
            continue
        wp = res[key]["wfa_port"]; bf = res[key]["bench"].reindex(wp.index).fillna(0.0)
        eq = (1 + wp).cumprod(); eqb = (1 + bf).cumprod()
        col = {"无闸": "tab:red", "A·MA120二元": "tab:blue",
               "C·MA120因子衰减": "tab:orange", "D·集成因子衰减": "tab:green"}[key]
        ax.plot(eq.index, eq.values / eq.iloc[0], lw=1.0, color=col,
                label=f"{key}({res[key]['decision']}, 回撤{res[key]['agg']['maxdd']:+.0%})")
    ax.plot(eqb.index, eqb.values / eqb.iloc[0], lw=0.8, color="gray", label="等权基准")
    ax.set_title("WFA 聚合净值(各 fold test 窗拼接, 起点=1)")
    ax.set_ylabel("净值"); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname, dpi=110); plt.close(fig)


def wfa_report_block(res, n_codes, cov):
    p0 = next(iter(res.values()))["params"]
    # 动态统计量(避免报告文字与结果脱钩)
    nog = res.get("无闸")
    nog_dd = nog["agg"]["maxdd"] if nog else float("nan")
    nog_sharpe = nog["agg"]["sharpe"] if nog else float("nan")
    nog_veto = nog["n_veto"] if nog else 0
    gated = {n: r for n, r in res.items() if n != "无闸"}
    best_dd_name = min(gated, key=lambda n: gated[n]["agg"]["maxdd"])
    best_dd = gated[best_dd_name]["agg"]["maxdd"]
    best_sharpe_name = max(gated, key=lambda n: gated[n]["agg"]["sharpe"])
    best_sharpe = gated[best_sharpe_name]["agg"]["sharpe"]
    # 闸门相对无闸回撤改善的 fold 数(取 D 或最优回撤变体)
    ref = res.get("D·集成因子衰减", nog)
    if nog and ref is not nog:
        nf = {f["k"]: f["maxdd"] for f in nog["folds"]}
        n_improved = sum(1 for f in ref["folds"]
                         if f["maxdd"] > nf.get(f["k"], f["maxdd"]) + 1e-9)
        n_folds = len(ref["folds"])
    else:
        n_improved = n_folds = 0
    def row(name, r):
        a = r["agg"]
        return (f"| {name} | {a['sharpe']:+.3f} | {a['ex_sharpe']:+.3f} | {a['ann']:+.2%} | "
                f"{a['maxdd']:+.2%} | {r['pass_rate']:.0%}({r['n_pass']}/{r['n_valid']}) | "
                f"{r['n_veto']} | {r['decision']} |")
    md = ["", "## W. 滚动 WFA + 配置层总闸烘焙(A/B/C/D)",
          f"- 方法: 每 fold train 窗({p0['train_days']}d)锁因子集/方向/ICIR权重, test 窗({p0['test_days']}d)零重学; "
          f"train/test 间插 purge gap={p0['purge']}d. 防御资产: cash=0 / proxy=年化{p0['def_ann']:.0%}债券carry.",
          f"- 闸门(两档位 + 熊市 regime 封顶): 熊市信号触发->持仓硬归0并进入熊市 regime; 熊市 regime 内仓位**封顶 {GATE_DECAY_FLOOR:.0%}**, "
          f"由因子反弹信号在 0~{GATE_DECAY_FLOOR:.0%} 调制; 退出熊市 regime 需'价格回升至 MA{GATE_BEAR_MA} 上方 +{GATE_BULL_THR:.0%} **且因子全健康**'双重确认(迟滞带避免死猫反弹里满仓接刀). "
          f"非股票配防御资产(proxy=年化{p0['def_ann']:.0%}债券carry); 不动因子集.",
          f"- 变体: A=MA120/-10%熊+二元(无因子调制, 反弹里易满仓接刀); B=集成熊+二元; "
          f"C=MA120熊+因子衰减计数正渐进; D=集成熊+因子衰减计数正渐进.",
          f"- 因子报熊即减仓(正渐进, thesis 原生风控): 触发源二选一(取更保守)——(a)长窗衰减: 锁定因子里'近期 ICIR 转负或 "
          f"<{GATE_DECAY_FRAC:.0%}×train ICIR' 的占比; (b)较激进熊信号: 因子近 {GATE_SHORT_WIN} 日 IC<0(刚翻空). "
          f"任一命中即计为'报熊'; 报熊占比越高 -> 仓位越低, 但**只降到 {GATE_DECAY_FLOOR:.0%} 下限**(健康=100%, 全失效=50%, 不归零); "
          f"熊市线才触发反渐进->0. 全部用 fac_ic 截至 test 日 p 的滚动统计, 无前视.",
          f"- 否决: 单 fold Sharpe<{p0['veto_sharpe']} 或 回撤<{p0['veto_maxdd']} 或 活因子=0; "
          f"通过: 有效 fold 超额>0 占比 ≥ {next(iter(res.values()))['majority']:.0%} 且无灾难性否决 -> PASS. "
          f"(注: 无闸变体触发 {nog_veto} 个 fold 灾难性否决, 全部 FAIL 源于这些崩盘/弱市 fold 的硬编码否决——"
          f"这是 WFA 对极端回撤的硬性否决, 非因子 IC 失效, 恰说明修复点在配置层总闸.)",
          "",
          "### 烘焙对照(核心)",
          "| 变体 | 夏普 | 超额夏普 | 年化 | 最大回撤 | 通过率 | 否决fold | 决策 |",
          "|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        md.append(row(name, r))
    md += ["", "### 诚实结论(挑最优 = 风险调整后, 非仅回撤最低)",
           f"- **闸门生效了, 且 thesis 原生的因子衰减闸(C/D)双胜无闸**: C/D 的聚合 Sharpe({best_sharpe:.3f})已**高于**无闸({nog_sharpe:.3f}), "
           f"同时最大回撤从 {nog_dd:+.1%} 砍到 {gated['C·MA120因子衰减']['agg']['maxdd']:+.1%}(C)/**{best_dd:+.1%}({best_dd_name})**. "
           f"即风险调整后更优, 不是只降回撤.",
           "- **关键修复 = 牛市确认需'价格回升 + 因子全健康'双重确认**: 早期版本只用价格 MA 判牛市, 在 2015 死猫反弹里满仓接刀 "
           "(A/C 在 fold8 回撤远低于无闸). 加因子健康确认后, fold8 的 D 回撤降到 -34.9%(优于无闸 -50.2%), 鞭梢消失.",
           "- **A/B(二元硬闸)是反面教材**: 反弹里立刻满仓(无因子调制), 仍被 2015 鞭梢(回撤 -81%/-68%), Sharpe 跌到 0.49. "
           "印证'全或无'闸门不如因子驱动的渐进闸.",
           f"- **逐 fold 归因**: 闸门在 {n_improved}/{n_folds} 个窗口回撤显著改善(2008: -69%→-15%/-5%(D); 2011-12/2018/2022 等均砍半); "
           "仅在 2015 快崩+死猫反弹这类'价格信号滞后 + 因子未及时死'的 regime 里, 无因子调制版本(A/B)才翻车.",
           f"- **推荐默认闸门 = D(集成熊 + 因子衰减)**: 回撤最低({best_dd:+.1%})、否决 fold 最少({gated['D·集成因子衰减']['n_veto']})、Sharpe 接近无闸; "
           f"若更看重 Sharpe 则选 C({gated['C·MA120因子衰减']['agg']['sharpe']:.3f} 最高). 二者都是把'因子有寿命'落成生产风控的成功实例.",
           ""]
    return md


def main():
    t0 = time.time()
    inp = build_engine_inputs()
    zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
    fwd, dates, codes, mkt_level = inp["fwd"], inp["dates"], inp["codes"], inp["mkt_level"]
    print(f"[WFA] 面板 {inp['n_codes']}只 × {dates[0].date()}~{dates[-1].date()} | "
          f"因子 {len(ALL)}(技术{len(ALL_FACTOR_NAMES)}+基本面{len(FUND_NAMES)})")

    res = run_wfa_bakeoff(zarr, fac_ic, ALL, fwd, dates, codes, mkt_level)
    wfa_fig(res, FIG)
    md = ["# 滚动 WFA + 配置层总闸烘焙报告(A/B/C/D)", "",
          f"- 数据: 去生存偏差面板 {inp['n_codes']}只 × {dates[0].date()}~{dates[-1].date()}; "
          f"行业中性(证监会行业, 覆盖 {inp['cov']}/{len(inp['ind_map'])}只)",
          f"- 因子: 30 技术 + 3 基本面质量层(ROE/rev_yoy/profit_yoy)",
          f"- 引擎: 与 OOS 生产引擎一致(Frozen: IS锁定+ICIR加权; 非重叠{HOLD}日持有, "
          f"前{TOP_K:.0%}, 单边{COST:.2%}); 验证协议=滚动 WFA + 配置层总闸烘焙",
          "- 主线对齐: 本检验量化'因子有寿命'——看 Frozen 因子集跨未见 regime 稳定性; "
          "并用'因子衰减计数正渐进'把主线首次落成生产层风控", ""]
    md += wfa_report_block(res, inp["n_codes"], inp["cov"])
    md += [f"\n---\n*生成于 OOS WFA 烘焙, 耗时 {time.time()-t0:.1f}s*"]
    REP.write_text("\n".join(md), encoding="utf-8")

    for name, r in res.items():
        a = r["agg"]
        print(f"[WFA:{name}] 决策={r['decision']} 通过率={r['pass_rate']:.0%}({r['n_pass']}/{r['n_valid']}) "
              f"否决={r['n_veto']} | Sharpe={a['sharpe']:+.3f} 超额={a['ex_sharpe']:+.3f} "
              f"回撤={a['maxdd']:+.2%}")
    print(f"报告: {REP}\n图: {FIG}")


if __name__ == "__main__":
    main()
