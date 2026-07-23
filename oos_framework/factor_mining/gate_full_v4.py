#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5c · 完整双尾 + 持仓结构倾斜 (巴菲特左侧预警驱动, 价量波动防御度)
================================================================
在 gate_full_v3(上证基准+巴指左侧+因子衰减+敏感耦合)之上, 补上此前缺失的
"持仓结构抗风险调整":

  每日选股: top-30% by fused (alpha 保收益来源, 与裸多头/旧双尾完全一致)
  D̃_i = clip(1 + 0.5·zmean[ -vol_20, -ivol_60, -downside_vol_60, +chip_disp, -chip_conc90 ], 0, 2)
  w_i  = (1-tc)·等权 + tc·softmax(fused_i · D̃_i)        tc = min(tilt, TILT_CAP)

  · tc=0(巴指无预警) → 纯等权, 与裸多头/旧双尾基线零差异, 收益零损耗
  · tc>0(巴指预警)  → 向"高alpha×高防御"滑; D̃ 仅价量波动(快变量, 跟市场实时调)
  · 巴指不对称/阴跌钝化无碍: 它只管"何时开启倾斜", 倾斜本身用快变量
  · D̃ 无基本面/PIT泄漏(纯价量+chip成本分布)

对照: 裸多头(等权) / 倾斜(价量防御,无危机门) / 完整双尾+持仓倾斜(v4, 最终)
评判: 回撤系指标 + 危机段内回撤. 重点看"回撤降了, 年化是否守住".
"""
from __future__ import annotations
import os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OOS_ROOT = HERE.parent
sys.path.insert(0, str(OOS_ROOT))

from oos_engine import load_market_index
from defensive_gating import (
    _load_macro, _macro_gating, _crisis_signal,
    CRISIS_POS, DEF_ANN, MAX_POS_REDUCE, MAX_POS_REDUCE_DECAY, GATE_DECAY_FRAC)
from factor_mining.gate_backtest_v2 import daily_ret_top, metrics, fmt, apply_gate
from factor_mining.gate_full_v3 import daily_cross_ic, factor_decay_flag
from frozen_gate_wfa import crisis_seg_dd

OOS_PARQUET = "factor_mining/oos_detail_v2.parquet"
DEF_PANEL = "factor_mining/defense_panel.parquet"
RESULT_MD = "factor_mining/RESULT_v2_gate_full_v4.md"
TILT_CAP = 0.70          # 用户定: 倾斜上限 0.7
SENS_GATE = 0.5
DECAY_WIN = 60
DECAY_BASE_WIN = 250


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd and sd > 0 else s * 0.0


def tilted_daily(od: pd.DataFrame, defense: pd.DataFrame,
                 tilt: pd.Series, top_frac: float = 0.3) -> pd.Series:
    """持仓倾斜日收益: 选股按 fused(top30%), 权重=等权↔softmax(fused·D̃) 按 tc 混合."""
    df = od.dropna(subset=["fused", "fwd_ret_1"]).merge(
        defense, on=["date", "code"], how="inner")
    need = ["vol_20", "downside_vol_60", "chip_disp", "chip_conc90"]   # ivol_60 暂弃(面板计算bug已修, 待重算)
    df = df.dropna(subset=need)
    # 截面 z 分位 → 防御度 D̃
    signed = {
        "s1": -df["vol_20"], "s2": -df["downside_vol_60"],
        "s3": df["chip_disp"], "s4": -df["chip_conc90"],
    }
    zparts = []
    for k, v in signed.items():
        df[k] = v.groupby(df["date"]).transform(zscore)
        zparts.append(k)
    df["zmean"] = df[zparts].mean(axis=1)
    df["Dtilde"] = (1.0 + 0.5 * df["zmean"]).clip(0.0, 2.0)
    df["eff"] = df["fused"] * df["Dtilde"]
    df["rk"] = df.groupby("date")["fused"].rank(pct=True, ascending=False)

    tc_map = tilt.reindex(df["date"].unique()).fillna(0.0).to_dict()

    def _w(g):
        sel = g[g["rk"] <= top_frac]
        K = len(sel)
        if K == 0:
            return 0.0
        tc = min(float(tc_map.get(g.name, 0.0)), TILT_CAP)
        w_eq = np.full(K, 1.0 / K)
        e = sel["eff"].values.astype(float)
        e = e - e.max()
        wd = np.exp(e)
        wd = wd / wd.sum()
        w = (1.0 - tc) * w_eq + tc * wd
        return float(np.sum(w * sel["fwd_ret_1"].values))

    ret = df.groupby("date").apply(_w).sort_index()
    return ret


def main():
    t0 = time.time()
    od = pd.read_parquet(OOS_PARQUET)
    od["date"] = pd.to_datetime(od["date"])
    defense = pd.read_parquet(DEF_PANEL)
    defense["date"] = pd.to_datetime(defense["date"])
    port_dates = pd.DatetimeIndex(sorted(od["date"].unique()))
    print(f"[gate_v4] oos {od.shape} | defense {defense.shape} | "
          f"窗 {port_dates.min().date()}~{port_dates.max().date()}", flush=True)

    # ---- 上证综指 + 巴指 + 敏感耦合 + 因子衰减 (复用 v3 组件) ----
    warm = pd.date_range("2016-01-01", port_dates.max(), freq="B")
    mkt = load_market_index("sh000001", index=warm).dropna()
    buffett, m2 = _load_macro(mkt)
    tilt = _macro_gating(buffett, m2, mkt)
    crisis = _crisis_signal(mkt, tilt=tilt, sens_gate=SENS_GATE)
    ic = daily_cross_ic(od)
    decay = factor_decay_flag(ic)
    print(f"[gate_v4] tilt均值 {tilt.reindex(port_dates).fillna(0).mean():.2f} | "
          f"危机日 {int(crisis.reindex(port_dates).fillna(False).sum())} | "
          f"衰减日 {int(decay.reindex(port_dates).fillna(False).sum())}", flush=True)

    # ---- 三档组合 ----
    bare = daily_ret_top(od, top_frac=0.3)                       # 等权裸多头
    tilted = tilted_daily(od, defense, tilt, top_frac=0.3)       # 持仓倾斜(无危机门)
    idx = bare.index
    cv = crisis.reindex(idx).fillna(False).values.astype(float)
    dv = decay.reindex(idx).fillna(False).values.astype(float)
    r_def = DEF_ANN / 252.0

    def gate(daily):
        buf = np.maximum(tilt.reindex(idx).fillna(0).values * MAX_POS_REDUCE,
                         dv * MAX_POS_REDUCE_DECAY)
        pos = 1.0 - buf
        pos = np.where(cv > 0.5, np.minimum(pos, CRISIS_POS), pos)
        return pd.Series(pos * daily.values + (1 - pos) * r_def, index=idx)

    gated_tilted = gate(tilted)        # 最终: 倾斜 + 双尾危机门
    base = od.dropna(subset=["fwd_ret_1"]).groupby("date")["fwd_ret_1"].mean() \
             .reindex(idx).fillna(0.0)

    rows = {
        "裸多头(等权,无门)": metrics(bare),
        "持仓倾斜(价量防御,无危机门)": metrics(tilted),
        "完整双尾+持仓倾斜(v4)": metrics(gated_tilted),
        "基准(全市场等权)": metrics(base),
    }
    print("", flush=True)
    for k, m in rows.items():
        print(f"  {k:30s} {fmt(m)}", flush=True)

    dd_no, nseg = crisis_seg_dd(bare, crisis)
    dd_t, _ = crisis_seg_dd(gated_tilted, crisis)
    print(f"\n  危机段内最大回撤: 裸多头 {dd_no:+.1%} → v4 {dd_t:+.1%} "
          f"(少亏 {abs(dd_no)-abs(dd_t):+.1%}pp, 危机段 {nseg})", flush=True)

    lines = ["# P5c · 完整双尾 + 持仓结构倾斜 (巴菲特左侧驱动, 价量波动防御度)\n",
             f"- OOS: {idx.min().date()} ~ {idx.max().date()} ({len(idx)} 交易日)",
             f"- TILT_CAP={TILT_CAP} | D̃=clip(1+0.5·zmean[-vol_20,-downside_vol_60,"
             "+chip_disp,-chip_conc90](ivol_60暂弃: 面板计算bug已修待重算),0,2) | 权重=(1-tc)·等权+tc·softmax(fused·D̃)",
             "- tc=min(tilt,0.7); 巴指无预警→纯等权(收益零损耗); 预警→滑向高alpha×高防御",
             "- 危机门: 上证破位/波动→0.6 + 巴指缓冲≤20% + 因子衰减≤20% + 敏感耦合\n",
             "| 组合 | 年化 | 夏普 | Sortino | 最大回撤 | Calmar | CVaR5(日) | 总收益 |",
             "|---|---|---|---|---|---|---|---|"]
    for k, m in rows.items():
        lines.append(f"| {k} | {m['ann']:+.1%} | {m['sharpe']:.2f} | {m['sortino']:.2f} | "
                     f"{m['max_dd']:+.1%} | {m['calmar']:.2f} | {m['cvar5_daily']:+.2%} | {m['tot']:+.1%} |")
    lines.append(f"\n- **危机段内最大回撤**: 裸多头 **{dd_no:+.1%}** → v4 **{dd_t:+.1%}** "
                 f"(少亏 {abs(dd_no)-abs(dd_t):+.1%}pp)")
    lines.append(f"\n*耗时 {(time.time()-t0)/60:.1f}min · gate_full_v4.py*")
    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    out = pd.DataFrame({"bare": bare, "tilted": tilted, "gated_tilted": gated_tilted,
                        "tilt": tilt.reindex(idx).fillna(0.0),
                        "crisis": crisis.reindex(idx).fillna(False)})
    out.to_parquet("factor_mining/gate_full_daily_v4.parquet")
    print(f"\n[gate_v4] 结果 → {RESULT_MD} | {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
