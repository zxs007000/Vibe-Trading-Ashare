#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5 · 防御门控叠加回测 (三维度 v2 alpha)
==========================================
在 oos_detail_v2.parquet(P4 全池 WFA 样本外融合概率)之上叠加双层防御门控:
  · crisis(右侧确认): 市场等权指数跌破250日线-10% 或 20日波动z>2 → 仓位降至 0.60
  · stress(左侧缓冲): 跌破250日线深度 [0,1] → 渐进降仓最多 20%
  · 空仓部分吃防御资产 4% 年化(国债ETF sleeve)
对比: 无门控 vs 门控 vs 门控敏感性(crisis_pos 0.4/0.6/0.8), 输出危机段内回撤对比。
纯规则门(不衰减后盾), ML 概率闸后续可作为第三层叠加。
"""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

from factor_mining.base_data import load_panel
from factor_mining.factor_wfa import backtest
from factor_mining.frozen_gate_wfa import market_level, crisis_signal, crisis_seg_dd
from factor_mining.universe import load_universe

OOS_PARQUET = "factor_mining/oos_detail_v2.parquet"
RESULT_MD = "factor_mining/RESULT_v2_gate.md"


def daily_ret_top(oos_detail: pd.DataFrame, top_frac: float = 0.3) -> pd.Series:
    df = oos_detail.dropna(subset=["fused", "fwd_ret_1"]).copy()
    df["rk"] = df.groupby("date")["fused"].rank(pct=True, ascending=False)
    top = df[df["rk"] <= top_frac]
    return top.groupby("date")["fwd_ret_1"].mean().sort_index()


def apply_gate(daily: pd.Series, crisis: pd.Series, stress: pd.Series,
               crisis_pos: float = 0.60, def_ann: float = 0.04,
               max_pos_reduce: float = 0.20) -> pd.Series:
    cr = crisis.reindex(daily.index).fillna(False).astype(float).values
    st = stress.reindex(daily.index).fillna(0.0).values
    pos = np.where(cr > 0.5, crisis_pos, 1.0 - st * max_pos_reduce)
    r_def = def_ann / 252.0
    return pd.Series(pos * daily.values + (1.0 - pos) * r_def, index=daily.index)


def metrics(daily: pd.Series) -> dict:
    nav = (1.0 + daily).cumprod()
    n = len(daily)
    ann = (nav.iloc[-1]) ** (252.0 / n) - 1.0
    vol = daily.std() * np.sqrt(252)
    sharpe = (daily.mean() * 252) / vol if vol > 0 else np.nan
    dd = float((nav / nav.cummax() - 1.0).min())
    neg = daily[daily < 0]
    dvol = neg.std() * np.sqrt(252) if len(neg) > 3 else np.nan
    sortino = (daily.mean() * 252) / dvol if dvol and dvol > 0 else np.nan
    var5 = float(np.percentile(daily, 5))
    cvar5 = float(daily[daily <= var5].mean()) if (daily <= var5).any() else np.nan
    return {"ann": ann, "vol": vol, "sharpe": sharpe, "max_dd": dd,
            "calmar": ann / abs(dd) if dd < 0 else np.nan,
            "sortino": sortino, "cvar5_daily": cvar5,
            "tot": float(nav.iloc[-1] - 1.0)}


def fmt(m: dict) -> str:
    return (f"年化 {m['ann']:+.1%} | 夏普 {m['sharpe']:.2f} | Sortino {m['sortino']:.2f} | "
            f"回撤 {m['max_dd']:+.1%} | Calmar {m['calmar']:.2f} | CVaR5 {m['cvar5_daily']:+.2%}")


def main():
    t0 = time.time()
    # ---- 1) 载入 OOS 融合概率明细 ----
    od = pd.read_parquet(OOS_PARQUET)
    od["date"] = pd.to_datetime(od["date"])
    print(f"[gate_v2] oos_detail {od.shape} | {od['date'].min().date()}~{od['date'].max().date()}", flush=True)

    # ---- 2) 市场等权指数(2018 起, 保证 250 日 MA 预热) ----
    codes = load_universe()
    close = load_panel("close", codes=codes, start="2018-01-01")
    print(f"[gate_v2] close 面板 {close.shape}", flush=True)
    mkt = market_level(close)
    crisis, stress = crisis_signal(mkt)
    oos_dates = pd.DatetimeIndex(od["date"].unique()).sort_values()
    n_cr = int(crisis.reindex(oos_dates).fillna(False).sum())
    print(f"[gate_v2] OOS 窗内危机日 {n_cr}/{len(oos_dates)} "
          f"({n_cr/len(oos_dates):.0%})", flush=True)

    # ---- 3) 组合日收益 + 门控对比 ----
    daily = daily_ret_top(od, top_frac=0.3)
    base = od.dropna(subset=["fwd_ret_1"]).groupby("date")["fwd_ret_1"].mean().reindex(daily.index).fillna(0.0)

    rows = {}
    rows["无门控(裸多头)"] = metrics(daily)
    rows["基准(全市场等权)"] = metrics(base)
    for cp in (0.8, 0.6, 0.4):
        g = apply_gate(daily, crisis, stress, crisis_pos=cp)
        rows[f"门控 crisis_pos={cp}"] = metrics(g)

    # 危机段内回撤对比(核心防御指标)
    dd_no, n_seg = crisis_seg_dd(daily, crisis)
    g06 = apply_gate(daily, crisis, stress, crisis_pos=0.6)
    dd_gate, _ = crisis_seg_dd(g06, crisis)

    print("", flush=True)
    for k, m in rows.items():
        print(f"  {k:26s} {fmt(m)}", flush=True)
    print(f"\n  危机段内最大回撤: 无门 {dd_no:+.1%} → 门控0.6 {dd_gate:+.1%} "
          f"(少亏 {abs(dd_no)-abs(dd_gate):+.1%}pp, 危机段数 {n_seg})", flush=True)

    # ---- 4) 落盘 ----
    lines = ["# P5 · 防御门控叠加回测 (三维度 v2 alpha)\n",
             f"- OOS: {daily.index.min().date()} ~ {daily.index.max().date()} "
             f"({len(daily)} 交易日) | 危机日 {n_cr} ({n_cr/len(oos_dates):.0%})",
             "- 门控: crisis(250日线-10% 或 波动z>2)→降仓; stress 渐进降仓≤20%; 空仓吃 4% 年化\n",
             "| 组合 | 年化 | 夏普 | Sortino | 最大回撤 | Calmar | CVaR5(日) | 总收益 |",
             "|---|---|---|---|---|---|---|---|"]
    for k, m in rows.items():
        lines.append(f"| {k} | {m['ann']:+.1%} | {m['sharpe']:.2f} | {m['sortino']:.2f} | "
                     f"{m['max_dd']:+.1%} | {m['calmar']:.2f} | {m['cvar5_daily']:+.2%} | {m['tot']:+.1%} |")
    lines.append(f"\n- **危机段内最大回撤**: 无门 **{dd_no:+.1%}** → 门控0.6 **{dd_gate:+.1%}** "
                 f"(少亏 {abs(dd_no)-abs(dd_gate):+.1%}pp)")
    lines.append(f"\n*耗时 {(time.time()-t0)/60:.1f}min · gate_backtest_v2.py*")
    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[gate_v2] 结果已写入 {RESULT_MD} | {(time.time()-t0)/60:.1f}min", flush=True)

    # 门控净值序列落盘(供后续 ML 概率闸叠加对比)
    out = pd.DataFrame({"daily_no_gate": daily, "daily_gate06": g06,
                        "crisis": crisis.reindex(daily.index).fillna(False),
                        "stress": stress.reindex(daily.index).fillna(0.0)})
    out.to_parquet("factor_mining/gate_daily_v2.parquet")
    print("[gate_v2] 日收益序列落盘 gate_daily_v2.parquet", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
