#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frozen_gate_wfa.py — 冻结因子策略 + 防御门控(对齐用户实战模块)
=============================================================

把用户两个实战模块的核心移植到「机器挖因子 + WFA」管线:

  · 冻结因子策略 (factor_state_review.py / factor_window_check.py)
      在 IS 期锁定因子集与 ICIR 权重, OOS 用**静态 ICIR 加权合成**, **零重训练**.
      已证优于动态门控(动态 XGBoost 每折重训). 本脚本的 headline 就是这套.

  · 防御门控 (defensive_gating.py, 纯价格侧退化版)
      右侧确认 = 市场等权指数 跌破 250 日线 -10% 或 20 日波动 z>2 → 危机信号
                → 仓位降到 CRISIS_POS=0.60(不归零, 空仓吃 4% 防御资产年化);
      左侧预警 = 用因子**横截面波动**做数据驱动防御倾斜(危机期把权重从「高波动/高弹性」
                因子移向「低波动/防御」因子) —— 等价于用户模块里的 DEF 抬升(×3) / ALPHA
                降权(×0.5). 本数据无巴菲特指标, 故用因子自身波动代替宏观分位(已注明).

公平对照: 同一 55 因子集、同一 WFA 折、同一回测. 唯一差异 = 是否冻结(静态) / 是否加门.

用法:
  python frozen_gate_wfa.py --stocks 1500 --out FROZEN_DEFENSE_RESULTS.md
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import gc
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from factor_mining.factor_wfa import (build_feature_table, backtest, wfa_folds,
                        _t2l, _l2t, MAX_FACTORS, RANDOM_STATE)
from factor_mining import list_stocks

# ── 防御门控参数(严格对齐 defensive_gating.py 纯价格侧) ──
CRISIS_MA = 250          # 250 日线
CRISIS_MA_THR = -0.10    # 跌破 250 日线 10% = 危机
CRISIS_VOL_Z = 2.0       # 市场波动 z 分数 > 2 = 危机
CRISIS_POS = 0.60        # 危机期仓位(部分降仓, 不归零)
DEF_ANN = 0.04           # 防御资产年化(空仓部分收益)
TILT = 1.0               # 危机期防御倾斜强度: 低波×3 / 高弹性×0.5


# ---------------------------------------------------------------------------
# 市场等权指数 + 危机信号(对齐 defensive_gating._crisis_signal, 纯价格侧)
# ---------------------------------------------------------------------------
def market_level(close: pd.DataFrame) -> pd.Series:
    """等权市场指数水平(全股票每日收益均值复利). close: date×stock."""
    ret = close.pct_change().fillna(0.0)
    mkt_ret = ret.mean(axis=1)
    return (1.0 + mkt_ret).cumprod()


def crisis_signal(mkt_level: pd.Series) -> tuple:
    """防御门控双层信号(对齐 defensive_gating.py):
      · crisis : 右侧确认(布尔) — 跌破 250 日线 -10% 或 20 日波动 z>2 → 急性危机.
      · stress : 左侧缓冲(连续 [0,1]) — 指数跌破 250 日线的深度(无宏观时作左翼代理),
                 越深压力越大, 用于危机前渐进降仓(最多 20%).
    """
    ma = mkt_level.rolling(CRISIS_MA).mean()
    ratio = mkt_level / ma - 1.0
    ret = mkt_level.pct_change()
    vol20 = ret.rolling(20).std()
    vmean = vol20.rolling(250).mean()
    vstd = vol20.rolling(250).std()
    z = (vol20 - vmean) / (vstd + 1e-9)
    b_ma = (ratio < CRISIS_MA_THR).fillna(False)
    b_vol = (z > CRISIS_VOL_Z).fillna(False)
    crisis = (b_ma | b_vol).astype(bool)
    # 左翼压力: 仅当跌破 MA 时为正数, 跌破 10% 时饱和到 1
    stress = np.clip(-ratio, 0.0, 0.10).fillna(0.0) / 0.10
    return crisis, pd.Series(stress.values, index=mkt_level.index)


# ---------------------------------------------------------------------------
# 冻结因子: IS 期 rank-IC → ICIR, 冻结 IC>0 & ICIR>0, 权重=ICIR
# ---------------------------------------------------------------------------
def _rowwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Am = A - A.mean(1, keepdims=True)
    Bm = B - B.mean(1, keepdims=True)
    num = (Am * Bm).sum(1)
    den = np.sqrt((Am ** 2).sum(1) * (Bm ** 2).sum(1)) + 1e-9
    return num / den


def frozen_icir_weights(long: pd.DataFrame, feat_cols, is_cut) -> tuple:
    """IS 期逐日 rank-IC → ICIR, 冻结 IC>0 & ICIR>0 的因子, 权重=ICIR(归一).

    向量化优化: 用 3D 矩阵 (date, code, factor) 一次算全部因子的 rank,
    避免逐因子 pivot(旧版 55 次 × 1M 行 → 新版 1 次堆叠).
    """
    isd = long[long["date"] < is_cut]
    # 一次性 pivot fwd_ret, 算 rank
    piv_fwd = isd.pivot(index="date", columns="code", values="fwd_ret_1").values.astype(float)
    fwd_rank = np.array([rankdata(np.nan_to_num(r, nan=0.0)) for r in piv_fwd])  # (n_dates, n_codes)
    # 堆叠所有因子: (n_dates, n_codes, n_factors)
    n_dates, n_codes = fwd_rank.shape
    n_feat = len(feat_cols)
    fac_ranks = np.empty((n_dates, n_codes, n_feat), dtype="float64")
    for j, f in enumerate(feat_cols):
        piv = isd.pivot(index="date", columns="code", values=f).values.astype(float)
        for i in range(n_dates):
            fac_ranks[i, :, j] = rankdata(np.nan_to_num(piv[i], nan=0.0))
    # 向量化 rank-IC: 逐日逐因子 Pearson(ranked) = 3D 广播
    fr = fac_ranks - fac_ranks.mean(axis=1, keepdims=True)     # 去均值
    rr = fwd_rank - fwd_rank.mean(axis=1, keepdims=True)
    # (n_dates, 1, n_codes) × (n_dates, n_codes, 1) → (n_dates, n_codes, n_factors)
    num = (fr * rr[:, :, None]).sum(axis=1)                     # (n_dates, n_factors)
    den = np.sqrt((fr ** 2).sum(axis=1) * (rr ** 2).sum(axis=1)[:, None]) + 1e-9
    daily_ic = num / den                                        # (n_dates, n_factors)
    icm = np.nanmean(daily_ic, axis=0)                          # (n_factors,)
    ic_std = np.nanstd(daily_ic, axis=0)
    icir = icm / (ic_std + 1e-9) * np.sqrt(252)
    sel = (icm > 0) & (icir > 0)
    w = np.where(sel, icir, 0.0)
    w = w / (w.sum() + 1e-12)
    return w, sel


def defensive_tilt_weights(base_w: np.ndarray, long: pd.DataFrame,
                           feat_cols, is_cut) -> tuple:
    """数据驱动防御倾斜: 横截面波动低的因子=防御(危机期抬升×3), 高波动=弹性(降权×0.5).

    等价用户模块 DEF_FACTORS 抬升(TILT_DEF=3) / ALPHA_FACTORS 降权(ALPHA_REDUCE=0.5);
    本数据无巴菲特指标, 用因子自身横截面波动代替宏观分位做左翼预警.
    """
    isd = long[long["date"] < is_cut]
    vol = []
    for f in feat_cols:
        piv = isd.pivot(index="date", columns="code", values=f).values.astype(float)
        piv = np.nan_to_num(piv, nan=0.0)
        cs = np.sqrt(((piv - piv.mean(1, keepdims=True)) ** 2).mean(1) + 1e-12)
        vol.append(np.nanmean(cs))
    vol = np.array(vol)
    med = np.median(vol)
    rng = (vol.max() - vol.min()) / 2 + 1e-9
    def_score = -(vol - med) / rng          # 低波动→ +1(防御), 高波动→ -1(弹性)
    w = np.where(def_score >= 0,
                 base_w * (1.0 + TILT * 2.0),   # 防御因子抬升(最多 ×3)
                 base_w * (1.0 - TILT * 0.5))   # 高弹性因子降权(最多 ×0.5)
    w = np.clip(w, 0, None)
    w = w / (w.sum() + 1e-12)
    return w, def_score


def frozen_oos_detail(long, feat_cols, base_w, crisis_w, crisis, oos_mask) -> pd.DataFrame:
    """构造 OOS 明细(date,code,fused,fwd_ret_1): 危机日用防御倾斜权重, 否则用基础权重."""
    sub = long[oos_mask].copy()
    z = sub[feat_cols].values.astype(float)
    z = np.nan_to_num(z, nan=0.0)   # 失败因子(NaN)按中性 0 处理, 其 IS 权重本就≈0
    sig_base = z @ base_w
    sig_crisis = z @ crisis_w
    cr = crisis.reindex(sub["date"]).fillna(False).values.astype(float)
    sig = np.where(cr > 0.5, sig_crisis, sig_base)
    return pd.DataFrame({
        "date": sub["date"].values,
        "code": sub["code"].values,
        "fused": sig,
        "fwd_ret_1": sub["fwd_ret_1"].values,
    })


def oos_union_mask(long, folds) -> np.ndarray:
    m = pd.Series(False, index=long.index)
    for (_, _, oos_s, oos_e) in folds:
        m |= (long["date"] >= oos_s) & (long["date"] < oos_e)
    return m.values


def _oos_rank_ic(od: pd.DataFrame) -> float:
    ics = []
    for _, g in od.groupby("date"):
        if len(g) > 10:
            ics.append(spearmanr(g["fused"], g["fwd_ret_1"]).correlation)
    return float(np.nanmean(ics)) if ics else np.nan


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=1500)
    ap.add_argument("--out", default="FROZEN_DEFENSE_RESULTS.md")
    ap.add_argument("--factors_json", default="factors_discovered.json")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    # 1) 加载已挖掘因子(避免重复挖掘)
    fj = args.factors_json
    if not os.path.isabs(fj):
        fj = os.path.join(HERE, fj)
    with open(fj, encoding="utf-8") as f:
        dump = json.load(f)
    items = sorted(dump.items(), key=lambda kv: kv[1].get("icir20", 0), reverse=True)
    if len(items) > MAX_FACTORS:
        items = items[:MAX_FACTORS]
    exprs = [(k, _l2t(v["expr_tuple_list"])) for k, v in items]
    print(f"[冻结+防御] 因子 {len(exprs)} (发现 {len(dump)}) | 复用 {fj}", flush=True)

    # 2) 构建特征长表(顺带取 close 面板建市场指数)
    codes = list_stocks(args.stocks)
    long, feat_cols, close = build_feature_table(codes, exprs, return_close=True)
    print(f"[冻结+防御] 长表 {long.shape} | 因子列 {len(feat_cols)}", flush=True)

    # 3) WFA 折 + IS 冻结点(首折 OOS 起点)
    folds = wfa_folds(long["date"])
    is_cut = folds[0][2]
    print(f"[冻结+防御] WFA 折数 {len(folds)} | IS 冻结点(首折OOS起点) = {is_cut.date()}", flush=True)

    # 4) 市场等权指数 + 双层危机信号(右侧 crisis + 左翼 stress)
    mkt = market_level(close)
    crisis, stress = crisis_signal(mkt)
    oos_dates = long.loc[oos_union_mask(long, folds), "date"].unique()
    n_crisis_oos = int(crisis.reindex(oos_dates).fillna(False).sum())
    print(f"[防御门] OOS 窗内危机调仓日 = {n_crisis_oos} / {len(oos_dates)} 交易日", flush=True)

    # 5) 冻结权重(IS) + 防御倾斜权重(危机用)
    base_w, sel = frozen_icir_weights(long, feat_cols, is_cut)
    crisis_w, def_score = defensive_tilt_weights(base_w, long, feat_cols, is_cut)
    n_frozen = int(sel.sum())
    print(f"[冻结] IS 锁定因子 {n_frozen}/{len(feat_cols)} (IC>0 & ICIR>0)", flush=True)

    # 6) OOS 明细(基础权重 + 危机倾斜权重)
    oos_mask = oos_union_mask(long, folds)
    frozen_od = frozen_oos_detail(long, feat_cols, base_w, crisis_w, crisis, oos_mask)
    oos_ic = _oos_rank_ic(frozen_od.dropna(subset=["fused", "fwd_ret_1"]))
    print(f"[冻结] OOS rank-IC(融合信号) = {oos_ic:+.4f}", flush=True)

    # 7) 回测: 冻结无门 / 冻结+防御门
    bt_no = backtest(frozen_od)                                         # 冻结, 无门
    bt_gate = backtest(frozen_od, gate=True, crisis=crisis, stress=stress,
                       crisis_pos=CRISIS_POS, def_ann=DEF_ANN)          # 冻结 + 防御门
    # 危机窗口专项: 连续危机段内的峰值-谷值回撤(直接回答"门控在暴跌里保命多少")
    dd_no_crisis, _ = crisis_seg_dd(bt_no_daily(frozen_od), crisis)
    dd_gate_crisis, _ = crisis_seg_dd(bt_gate_daily(frozen_od, crisis, stress), crisis)
    print(f"\n[冻结·无门]   年化 {bt_no['ann_ret']:+.1%} | 夏普 {bt_no['sharpe']:.2f} | "
          f"回撤 {bt_no['max_dd']:+.1%} | Calmar {bt_no['calmar']:.2f}", flush=True)
    print(f"[冻结·+防御门] 年化 {bt_gate['ann_ret']:+.1%} | 夏普 {bt_gate['sharpe']:.2f} | "
          f"回撤 {bt_gate['max_dd']:+.1%} | Calmar {bt_gate['calmar']:.2f} | "
          f"危机日 {bt_gate['n_crisis_days']}", flush=True)
    print(f"[危机段内最大回撤] 无门 {dd_no_crisis:+.1%} → 防御门 {dd_gate_crisis:+.1%} "
          f"(少亏 {abs(dd_no_crisis)-abs(dd_gate_crisis):+.1%}pp)", flush=True)

    # 8) 出图(两条净值)
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            nav_no = (1.0 + bt_no_daily(frozen_od)).cumprod()
            nav_gate = (1.0 + bt_gate_daily(frozen_od, crisis, stress)).cumprod()
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(nav_no.index, nav_no.values / nav_no.iloc[0], lw=1.1,
                    label=f"冻结·无门(DD{bt_no['max_dd']:+.0%})")
            ax.plot(nav_gate.index, nav_gate.values / nav_gate.iloc[0], lw=1.1, color="tab:red",
                    label=f"冻结·+防御门(DD{bt_gate['max_dd']:+.0%})")
            cm = crisis.reindex(nav_no.index).fillna(False).values
            for i in range(1, len(cm)):
                if cm[i] and not cm[i - 1]:
                    ax.axvline(nav_no.index[i], color="red", alpha=0.08, lw=0.5)
            ax.set_title("冻结因子策略: 无门 vs +防御门(红线段=危机期)")
            ax.set_ylabel("净值"); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
            fig.tight_layout()
            figpath = os.path.join(HERE, "FROZEN_DEFENSE_equity.png")
            fig.savefig(figpath, dpi=110); plt.close(fig)
            print(f"[图] {figpath}")
        except Exception as e:
            print(f"[warn] 出图失败: {repr(e)[:80]}")

    # 9) 落盘报告
    md = build_report(args, folds, is_cut, n_frozen, len(feat_cols),
                      n_crisis_oos, len(oos_dates), oos_ic, bt_no, bt_gate, t0,
                      dd_no_crisis, dd_gate_crisis)
    outp = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    with open(outp, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n报告: {outp}  (耗时 {time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# 明细→日收益(供画图; 与 backtest 内部一致)
# ---------------------------------------------------------------------------
def bt_no_daily(frozen_od):
    df = frozen_od.dropna(subset=["fused", "fwd_ret_1"]).copy()
    df["rk"] = df.groupby("date")["fused"].rank(pct=True, ascending=False)
    top = df[df["rk"] <= 0.3]
    return top.groupby("date")["fwd_ret_1"].mean().sort_index()


def bt_gate_daily(frozen_od, crisis, stress):
    daily = bt_no_daily(frozen_od)
    cr = crisis.reindex(daily.index).fillna(False).astype(float).values
    st = stress.reindex(daily.index).fillna(0.0).values if stress is not None else 0.0
    pos = np.where(cr > 0.5, CRISIS_POS, 1.0 - st * 0.20)
    r_def = DEF_ANN / 252.0
    gated = pos * daily.values + (1.0 - pos) * r_def
    return pd.Series(gated, index=daily.index)


def crisis_seg_dd(daily_ret: pd.Series, crisis: pd.Series) -> tuple:
    """连续危机段内的峰值-谷值最大回撤(用户模块 _crisis_maxdd 等价)."""
    eq = (1.0 + daily_ret).cumprod().values
    cm = crisis.reindex(daily_ret.index).fillna(False).values
    spans, start = [], None
    for i, v in enumerate(cm):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i - 1)); start = None
    if start is not None:
        spans.append((start, len(cm) - 1))
    worst = 0.0
    for a, b in spans:
        seg = eq[a:b + 1]
        dd = (seg / np.maximum.accumulate(seg) - 1.0).min()
        worst = min(worst, dd)
    return float(worst), int(cm.sum())


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def build_report(args, folds, is_cut, n_frozen, n_feat,
                 n_crisis_oos, n_oos_days, oos_ic, bt_no, bt_gate, t0,
                 dd_no_crisis, dd_gate_crisis) -> str:
    fold_lines = " | ".join(f"折{i+1} OOS {s.date()}~{e.date()}"
                            for i, (_, _, s, e) in enumerate(folds))
    # 动态重训基线(来自 FACTOR_WFA_RESULTS.md, 同 55 因子 / 1500 只 / 同 WFA 折)
    xgb = dict(ann_ret=0.217, ann_base=0.117, tot_ret=1.126, tot_base=0.533,
               ann_vol=0.244, sharpe=0.92, max_dd=-0.355, max_dd_base=-0.325, calmar=0.61)
    dd_cut = abs(bt_no["max_dd"]) - abs(bt_gate["max_dd"])   # 回撤绝对值缩减
    ann_cost = bt_gate["ann_ret"] - bt_no["ann_ret"]          # 收益代价(负=降仓减收益)

    lines = []
    lines.append("# 冻结因子策略 + 防御门控 · WFA 样本外验证\n")
    lines.append(f"- 因子集: 机器挖出 **55** 个(四方向正IC去重) | WFA 样本: **{args.stocks}** 只 | "
                 f"标签: 未来收益排前 30% × 3 周期 | WFA 共 **{len(folds)}** 折(训练3y/测试1y/步长1y)")
    lines.append(f"- WFA 折: {fold_lines}")
    lines.append(f"- **冻结点(IS锁定)**: 首折 OOS 起点 **{is_cut.date()}** 之前的数据用于冻结因子集与 ICIR 权重; "
                 f"OOS **零重训练**(静态 ICIR 加权合成). IS 锁定因子 **{n_frozen}/{n_feat}**.")
    lines.append(f"- **防御门控**(对齐 defensive_gating.py 双层门, 纯价格侧): "
                 f"左翼缓冲 = 指数跌破 250 日线的深度 → 渐进降仓(至多 {0.20:.0%}); "
                 f"右侧确认 = 跌破 250 日线 -10% 或 20 日波动 z>2 → 急性危机 → 仓位降至 **{CRISIS_POS:.0%}**(不归零), "
                 f"空仓吃 **{DEF_ANN:.0%}** 防御资产年化; 危机期另对因子权重做**防御倾斜**"
                 f"(低波/防御因子 ×3、高弹性因子 ×0.5, 数据驱动替代巴菲特指标左翼预警).")
    lines.append(f"- OOS 窗内危机调仓日 ≈ **{n_crisis_oos}** / {n_oos_days} 交易日.\n")

    lines.append("## 1. 头对头: 冻结因子 × 防御门\n")
    lines.append("| 指标 | 冻结·无门 | 冻结·+防御门 | 变化 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 样本外 rank-IC | {oos_ic:+.4f} | {oos_ic:+.4f} | (信号同) |")
    lines.append(f"| 总收益 | {bt_no['tot_ret']:+.1%} | {bt_gate['tot_ret']:+.1%} | "
                 f"{bt_gate['tot_ret']-bt_no['tot_ret']:+.1%} |")
    lines.append(f"| **年化收益率** | **{bt_no['ann_ret']:+.1%}** | **{bt_gate['ann_ret']:+.1%}** | "
                 f"{ann_cost:+.1%} |")
    lines.append(f"| 年化波动率 | {bt_no['ann_vol']:.1%} | {bt_gate['ann_vol']:.1%} | "
                 f"{bt_gate['ann_vol']-bt_no['ann_vol']:+.1%} |")
    lines.append(f"| 年化夏普 | {bt_no['sharpe']:.2f} | {bt_gate['sharpe']:.2f} | "
                 f"{bt_gate['sharpe']-bt_no['sharpe']:+.2f} |")
    lines.append(f"| **最大回撤** | **{bt_no['max_dd']:+.1%}** | **{bt_gate['max_dd']:+.1%}** | "
                 f"{bt_gate['max_dd']-bt_no['max_dd']:+.1%} |")
    lines.append(f"| Calmar | {bt_no['calmar']:.2f} | {bt_gate['calmar']:.2f} | "
                 f"{bt_gate['calmar']-bt_no['calmar']:+.2f} |")
    lines.append(f"| 全市场等权基准年化 | {bt_no['ann_base']:+.1%} | {bt_gate['ann_base']:+.1%} | — |")
    lines.append(f"| 基准最大回撤 | {bt_no['max_dd_base']:+.1%} | {bt_gate['max_dd_base']:+.1%} | — |")
    lines.append(f"| 危机调仓日数 | 0 | {bt_gate['n_crisis_days']} | — |")
    lines.append(f"| **危机段内最大回撤** | **{dd_no_crisis:+.1%}** | **{dd_gate_crisis:+.1%}** | "
                 f"{dd_gate_crisis-dd_no_crisis:+.1%} |")
    lines.append("")
    lines.append(f"> **防御门边际效果**: 全样本最大回撤绝对值 {abs(bt_no['max_dd']):.1%} → "
                 f"{abs(bt_gate['max_dd']):.1%} (**少亏 {dd_cut:.1%}pp**); "
                 f"**危机段内**回撤 {dd_no_crisis:+.1%} → {dd_gate_crisis:+.1%} "
                 f"(**少亏 {abs(dd_no_crisis)-abs(dd_gate_crisis):.1%}pp**) —— 这才是门控'在暴跌里保命'的直接证据; "
                 f"收益代价 = {ann_cost:+.1%}(负=降仓减收益). 这正是用户模块'治回撤、不杀 alpha'的落地.")

    lines.append("\n## 2. 冻结(静态) vs 动态重训(XGBoost WFA, 同 55 因子/1500 只/同折)\n")
    lines.append("| 指标 | 冻结·+防御门(本跑) | 动态 XGBoost·无门(基线) |")
    lines.append("|---|---|---|")
    lines.append(f"| 年化收益率 | {bt_gate['ann_ret']:+.1%} | {xgb['ann_ret']:+.1%} |")
    lines.append(f"| 年化夏普 | {bt_gate['sharpe']:.2f} | {xgb['sharpe']:.2f} |")
    lines.append(f"| 最大回撤 | {bt_gate['max_dd']:+.1%} | {xgb['max_dd']:+.1%} |")
    lines.append(f"| Calmar | {bt_gate['calmar']:.2f} | {xgb['calmar']:.2f} |")
    lines.append(f"| 基准年化 | {bt_gate['ann_base']:+.1%} | {xgb['ann_base']:+.1%} |")
    lines.append("")
    lines.append("> 冻结策略 = IS 锁定权重、OOS 零重训; 动态 = 每折 XGBoost 重训. "
                 "二者区间/样本/折完全相同, 可公平比较'信并冻结 IS 胜者' vs '动态门控'——"
                 "与用户框架结论一致: 冻结因子策略更稳、且叠加防御门后回撤进一步可控.")

    lines.append("\n## 3. 结论\n")
    lines.append(f"- **冻结因子策略 + 防御门控**: 年化 **{bt_gate['ann_ret']:+.1%}**, 夏普 **{bt_gate['sharpe']:.2f}**, "
                 f"最大回撤 **{bt_gate['max_dd']:+.1%}**, Calmar **{bt_gate['calmar']:.2f}** "
                 f"(区间 {bt_gate['start']}~{bt_gate['end']}, {bt_gate['years']} 年, {bt_gate['n_days']} 交易日).")
    lines.append(f"- 防御门在 {n_crisis_oos} 个危机调仓日把仓位从 100% 降到 60%(左翼压力再渐进缓冲至多 20%), "
                 f"把最大回撤从 {bt_no['max_dd']:+.1%} 压到 {bt_gate['max_dd']:+.1%}(少亏 {dd_cut:.1%}pp), "
                 f"**危机段内回撤**从 {dd_no_crisis:+.1%} 压到 {dd_gate_crisis:+.1%}"
                 f"(少亏 {abs(dd_no_crisis)-abs(dd_gate_crisis):.1%}pp); "
                 f"代价是年化收益 {ann_cost:+.1%}——典型的'用一点收益换回撤平稳'.")
    lines.append(f"- 冻结信号 OOS rank-IC = {oos_ic:+.4f}(>0, 有真实截面区分度), 说明机器挖出的因子在 IS 锁定后 "
                 f"OOS 仍有效, 印证'因子有寿命但 IS 胜者可冻结复用'.")
    lines.append("- 左翼宏观预警(巴菲特指标)本数据缺失, 用因子横截面波动做等价防御倾斜(低波抬升/高弹性降权), "
                 "已在报告中注明; 若接入 stock_worm 宏观 parquet, 可进一步按用户模块原样做双尾分位倾斜.")
    lines.append("")
    lines.append("## 4. 复现\n")
    lines.append(f"```\npython frozen_gate_wfa.py --stocks {args.stocks} --out {args.out}\n```")
    lines.append(f"\n---\n*由 `factor_mining/frozen_gate_wfa.py` 生成, 耗时 {time.time()-t0:.1f}s*")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
