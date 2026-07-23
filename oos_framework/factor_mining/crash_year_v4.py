#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5c · 股灾年/股灾段 专项对比 (完整双尾 v4 vs 裸多头 vs 仅倾斜)
==========================================================
设计本为极端情况准备, 平时巴指沉睡正常; 此脚本挑 OOS 窗内公认股灾段,
看双尾防御在真危机里是否显形。对比: 裸多头 / 持仓倾斜(无危机门) / 完整双尾v4。
每段附: 上证综指收益&最大回撤(背景) + 危机门激活天数占比。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OOS_ROOT = HERE.parent
sys.path.insert(0, str(OOS_ROOT))

from oos_engine import load_market_index
from factor_mining.gate_backtest_v2 import metrics, fmt

DAY = "factor_mining/gate_full_daily_v4.parquet"
OUT_MD = "factor_mining/RESULT_v2_crash_year.md"


def window_metrics(d: pd.DataFrame, s: pd.Series, start, end, label: str):
    seg = d.loc[start:end]
    if len(seg) == 0:
        return None
    mkt_seg = s.loc[start:end]
    nav = (1 + mkt_seg).cumprod()
    mkt_dd = float((nav / nav.cummax() - 1).min())
    mkt_ret = float(nav.iloc[-1] - 1)
    crisis_frac = float(seg["crisis"].mean()) if "crisis" in seg else 0.0
    rows = {
        "裸多头": metrics(seg["bare"]),
        "持仓倾斜(无门)": metrics(seg["tilted"]),
        "完整双尾v4": metrics(seg["gated_tilted"]),
    }
    return {"label": label, "n": len(seg), "mkt_ret": mkt_ret, "mkt_dd": mkt_dd,
            "crisis_frac": crisis_frac, "rows": rows}


def main():
    t0 = time.time()
    d = pd.read_parquet(DAY)
    d.index = pd.to_datetime(d.index)
    warm = pd.date_range("2016-01-01", d.index.max(), freq="B")
    mkt = load_market_index("sh000001", index=warm).dropna()
    mkt = mkt.pct_change().dropna()
    mkt = mkt.reindex(d.index).fillna(0.0)

    # ---- 股灾窗口 (OOS 窗 2021-10~2025-09 内公认危机段) ----
    windows = [
        ("2022 全年(慢熊)", "2022-01-01", "2022-12-31"),
        ("2024.1-2 微盘/DMA踩踏(最急)", "2024-01-02", "2024-02-29"),
        ("2024 全年(踩踏+9.24政策泵)", "2024-01-01", "2024-12-31"),
        ("2025(关税波动, 部分)", "2025-01-01", "2025-09-18"),
    ]
    results = []
    for lab, s, e in windows:
        r = window_metrics(d, mkt, s, e, lab)
        if r:
            results.append(r)

    # ---- 自动检测最深市场回撤段 (trough 前后) ----
    nav = (1 + mkt).cumprod()
    dd = nav / nav.cummax() - 1
    # 取回撤最深的那段: 从前期高点走到 trough
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    r_auto = window_metrics(d, mkt, str(peak.date()), str(trough.date()),
                            f"自动检测最深段({peak.date()}~{trough.date()})")
    if r_auto:
        results.append(r_auto)

    # ---- 打印 ----
    print(f"\n{'='*78}\n股灾段专项对比 (完整双尾 v4) | 窗 {d.index.min().date()}~{d.index.max().date()}\n{'='*78}")
    md = ["# P5c · 股灾年/股灾段专项对比 (完整双尾 v4)\n",
          f"- OOS 窗 {d.index.min().date()}~{d.index.max().date()} | 数据源 gate_full_daily_v4.parquet\n",
          "- 对比: 裸多头 / 持仓倾斜(价量防御,无危机门) / 完整双尾v4(倾斜+上证破位+巴指+衰减+耦合)\n",
          "- 每段附: 上证综指收益/最大回撤(背景) + 危机门激活天数占比\n",
          "| 股灾段 | 上证收益 | 上证回撤 | 危机门天占比 | 组合 | 年化 | 夏普 | 最大回撤 | 区间收益 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        print(f"\n### {r['label']}  (n={r['n']}日, 上证收益 {r['mkt_ret']:+.1%}, 上证回撤 {r['mkt_dd']:+.1%}, 危机门 {r['crisis_frac']:.0%})")
        md.append(f"\n### {r['label']}  (n={r['n']}, 上证 {r['mkt_ret']:+.1%} / 回撤 {r['mkt_dd']:+.1%}, 危机门激活 {r['crisis_frac']:.0%})\n")
        md.append("| 组合 | 年化 | 夏普 | 最大回撤 | 区间收益 |\n|---|---|---|---|---|")
        for k, m in r["rows"].items():
            line = (f"| {r['label']} | {r['mkt_ret']:+.1%} | {r['mkt_dd']:+.1%} | "
                    f"{r['crisis_frac']:.0%} | {k} | {m['ann']:+.1%} | {m['sharpe']:.2f} | "
                    f"{m['max_dd']:+.1%} | {m['tot']:+.1%} |")
            print(f"  {k:20s} 年化 {m['ann']:+.1%} | 夏普 {m['sharpe']:.2f} | 回撤 {m['max_dd']:+.1%} | 区间 {m['tot']:+.1%}")
            md.append(f"| {k} | {m['ann']:+.1%} | {m['sharpe']:.2f} | {m['max_dd']:+.1%} | {m['tot']:+.1%} |")
    md.append(f"\n*耗时 {(time.time()-t0):.1f}s · crash_year_v4.py*")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n[crash] 结果 → {OUT_MD} | {(time.time()-t0):.1f}s", flush=True)


if __name__ == "__main__":
    main()
