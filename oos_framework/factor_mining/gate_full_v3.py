#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5b · 完整双尾防御门控 (对齐用户 defensive_gating.py 宏观三腿)
================================================================
在 oos_detail_v2(P4 全池 WFA 组合)上叠加用户原设计的完整防御, 复用真组件:
  ① 上证综指为基准(oos_engine.load_market_index sh000001) —— 替换 P5 的等权代理.
  ② 左侧巴菲特指数(defensive_gating._macro_gating): 5Y 分位双尾 → defensive_tilt[0,1].
     组合级只能施加"缓冲降仓≤20%"(结构倾斜需因子权重, 组合已融合故无法, 明确标注).
  ③ 因子预警(factor_decay 精神的组合级代理): 组合滚动截面 IC < 0.3×历史均值 → 判衰减
     → 额外降仓≤20%(与巴指缓冲取 max, 对齐原设计).
  ④ 左侧→右侧敏感耦合(_crisis_signal tilt=): 巴指预警时右侧危机线左移(更早触发).
右侧确认: 上证跌破250日线(-10%,敏感时-5%) 或 波动z>2(敏感时>1) → 降仓至 CRISIS_POS.
空仓吃防御资产 4% 年化(国债ETF sleeve).
对照: 裸多头 / P5等权门(v2) / 完整双尾(本脚本). 评判用回撤系指标.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OOS_ROOT = HERE.parent                       # oos_framework
sys.path.insert(0, str(OOS_ROOT))

from oos_engine import load_market_index      # 真实上证综指
from defensive_gating import (
    _load_macro, _macro_gating, _crisis_signal,
    CRISIS_POS, DEF_ANN, MAX_POS_REDUCE, MAX_POS_REDUCE_DECAY, GATE_DECAY_FRAC)
from factor_mining.gate_backtest_v2 import daily_ret_top, metrics, fmt

OOS_PARQUET = "factor_mining/oos_detail_v2.parquet"
RESULT_MD = "factor_mining/RESULT_v2_gate_full.md"
DECAY_WIN = 60          # 组合滚动 IC 窗口(因子预警腿代理)
DECAY_BASE_WIN = 250    # 历史 IC 均值窗口
SENS_GATE = 0.5         # 仅巴指极端预警区(tilt>0.5)才施加右侧敏感耦合(滤毛刺)


def daily_cross_ic(od: pd.DataFrame) -> pd.Series:
    """组合每日截面 IC: spearman(fused, fwd_ret_1). 作为因子健康度代理(factor_decay 精神)."""
    def _ic(g):
        if len(g) < 20:
            return np.nan
        return g["fused"].rank().corr(g["fwd_ret_1"].rank())
    return od.groupby("date").apply(_ic).sort_index()


def factor_decay_flag(ic: pd.Series) -> pd.Series:
    """因子预警: 近 DECAY_WIN 滚动 IC 均值 < GATE_DECAY_FRAC × 历史均值 → 衰减(判熊, 降仓)."""
    roll = ic.rolling(DECAY_WIN, min_periods=20).mean()
    base = ic.rolling(DECAY_BASE_WIN, min_periods=60).mean()
    flag = (roll < GATE_DECAY_FRAC * base) & (base > 0)
    return flag.fillna(False)


def main():
    t0 = time.time()
    od = pd.read_parquet(OOS_PARQUET)
    od["date"] = pd.to_datetime(od["date"])
    port_dates = pd.DatetimeIndex(sorted(od["date"].unique()))
    print(f"[gate_v3] oos_detail {od.shape} | 组合 {port_dates.min().date()}~{port_dates.max().date()}", flush=True)

    # ---- ① 上证综指(从 2016 起给 5Y 分位/250MA 预热), 对齐后切到组合窗 ----
    warm = pd.date_range("2016-01-01", port_dates.max(), freq="B")
    mkt = load_market_index("sh000001", index=warm)
    if mkt is None:
        raise RuntimeError("上证综指加载失败")
    mkt = mkt.dropna()
    print(f"[gate_v3] 上证综指 {mkt.index.min().date()}~{mkt.index.max().date()} ({len(mkt)}日)", flush=True)

    # ---- ② 巴指 + M2 → 左侧 defensive_tilt ----
    buffett, m2 = _load_macro(mkt)
    print(f"[gate_v3] 巴指 {'OK '+str(int(buffett.notna().sum()))+'日' if buffett is not None else '缺失'} | "
          f"M2 {'OK' if m2 is not None else '缺失'}", flush=True)
    tilt = _macro_gating(buffett, m2, mkt)      # [0,1], full range
    tilt_p = tilt.reindex(port_dates).fillna(0.0)
    print(f"[gate_v3] 巴指 tilt: 均值 {tilt_p.mean():.2f} | 预警日(>0.3) {int((tilt_p>0.3).sum())}/{len(tilt_p)}", flush=True)

    # ---- ④ 敏感耦合右侧危机信号(巴指 tilt 使阈值左移) ----
    crisis = _crisis_signal(mkt, tilt=tilt, sens_gate=SENS_GATE)
    crisis_base = _crisis_signal(mkt)           # 无耦合基准(看敏感耦合多触发几天)
    cr_p = crisis.reindex(port_dates).fillna(False)
    crb_p = crisis_base.reindex(port_dates).fillna(False)
    print(f"[gate_v3] 右侧危机日(敏感耦合) {int(cr_p.sum())} vs 无耦合 {int(crb_p.sum())} "
          f"(+{int(cr_p.sum())-int(crb_p.sum())}天提前/额外触发)", flush=True)

    # ---- ③ 因子预警腿(组合滚动 IC 衰减代理) ----
    ic = daily_cross_ic(od)
    decay = factor_decay_flag(ic).reindex(port_dates).fillna(False)
    print(f"[gate_v3] 因子衰减预警日 {int(decay.sum())}/{len(port_dates)}", flush=True)

    # ---- 组合日收益 + 分层门控 ----
    daily = daily_ret_top(od, top_frac=0.3)
    base = od.dropna(subset=["fwd_ret_1"]).groupby("date")["fwd_ret_1"].mean().reindex(daily.index).fillna(0.0)
    idx = daily.index
    tv = tilt.reindex(idx).fillna(0.0).values
    cv = crisis.reindex(idx).fillna(False).values.astype(float)
    dv = decay.reindex(idx).fillna(False).values.astype(float)

    # 仓位: 左侧缓冲(巴指 tilt) 与 因子衰减 取 max, 再被右侧危机确认压到 CRISIS_POS
    buf = np.maximum(tv * MAX_POS_REDUCE, dv * MAX_POS_REDUCE_DECAY)   # 缓冲降仓 ≤20%
    pos = 1.0 - buf
    pos = np.where(cv > 0.5, np.minimum(pos, CRISIS_POS), pos)         # 右侧确认降至0.6
    r_def = DEF_ANN / 252.0
    gated = pos * daily.values + (1.0 - pos) * r_def
    gated = pd.Series(gated, index=idx)

    # 对照: 仅右侧(无巴指无衰减) / 仅巴指左侧
    pos_r = np.where(cv > 0.5, CRISIS_POS, 1.0)
    only_right = pd.Series(pos_r * daily.values + (1 - pos_r) * r_def, index=idx)

    rows = {
        "裸多头(无门)": metrics(daily),
        "仅右侧确认(上证破位)": metrics(only_right),
        "完整双尾(上证+巴指+衰减+耦合)": metrics(gated),
        "基准(全市场等权)": metrics(base),
    }
    print("", flush=True)
    for k, m in rows.items():
        print(f"  {k:28s} {fmt(m)}", flush=True)

    # 危机段内回撤(核心防御 KPI)
    from factor_mining.gate_backtest_v2 import apply_gate  # reuse for compat
    from frozen_gate_wfa import crisis_seg_dd
    dd_no, nseg = crisis_seg_dd(daily, crisis)
    dd_full, _ = crisis_seg_dd(gated, crisis)
    print(f"\n  危机段内最大回撤: 裸多头 {dd_no:+.1%} → 完整双尾 {dd_full:+.1%} "
          f"(少亏 {abs(dd_no)-abs(dd_full):+.1%}pp, 危机段 {nseg})", flush=True)

    # ---- 落盘 ----
    lines = ["# P5b · 完整双尾防御门控 (上证基准 + 巴指左侧 + 因子预警 + 敏感耦合)\n",
             f"- OOS: {idx.min().date()} ~ {idx.max().date()} ({len(idx)} 交易日)",
             f"- 上证综指右侧危机日 {int(cr_p.sum())}(敏感耦合, 无耦合 {int(crb_p.sum())}) | "
             f"巴指预警日(tilt>0.3) {int((tilt_p>0.3).sum())} | 因子衰减日 {int(decay.sum())}",
             "- 门控腿: ①上证破位/波动右侧确认→降至0.6 ②巴指5Y分位左侧缓冲≤20% "
             "③因子衰减缓冲≤20% ④巴指预警→右侧阈值左移(更早)",
             "- ⚠️ 局限: 组合已融合, 巴指'结构倾斜(防御因子升权)'无法施加, 仅缓冲降仓; "
             "完整结构倾斜需在 WFA 因子层跑(defensive_gating 主流程)\n",
             "| 组合 | 年化 | 夏普 | Sortino | 最大回撤 | Calmar | CVaR5(日) | 总收益 |",
             "|---|---|---|---|---|---|---|---|"]
    for k, m in rows.items():
        lines.append(f"| {k} | {m['ann']:+.1%} | {m['sharpe']:.2f} | {m['sortino']:.2f} | "
                     f"{m['max_dd']:+.1%} | {m['calmar']:.2f} | {m['cvar5_daily']:+.2%} | {m['tot']:+.1%} |")
    lines.append(f"\n- **危机段内最大回撤**: 裸多头 **{dd_no:+.1%}** → 完整双尾 **{dd_full:+.1%}** "
                 f"(少亏 {abs(dd_no)-abs(dd_full):+.1%}pp)")
    lines.append(f"\n*耗时 {(time.time()-t0)/60:.1f}min · gate_full_v3.py*")
    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    out = pd.DataFrame({"daily_no": daily, "daily_full": gated, "daily_right": only_right,
                        "tilt": tilt.reindex(idx).fillna(0.0),
                        "crisis": crisis.reindex(idx).fillna(False),
                        "decay": decay.reindex(idx).fillna(False)})
    out.to_parquet("factor_mining/gate_full_daily_v3.parquet")
    print(f"\n[gate_v3] 结果 → {RESULT_MD} | {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
