#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双尾防御门控 · 极端股灾抗压测试 (等权组合代理, 20年历史)
========================================================
不吃 alpha: 用等权全市场组合(load_panel close, codes=None)当"被保卫仓位",
门控=上证破位(右侧)+巴指左侧缓冲+敏感耦合(复用 defensive_gating 真组件)。
挑极端股灾年压门控抗压性: 2008/2015(全年+年中acute)/2016熔断 + 2022/2024 对照 + 全历史。
评判: 等权裸组合 vs 双尾门控组合 的区间收益/最大回撤/危机段内回撤。
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OOS_ROOT = HERE.parent
sys.path.insert(0, str(OOS_ROOT))

from oos_engine import load_market_index
from defensive_gating import (
    _load_macro, _macro_gating, _crisis_signal,
    CRISIS_POS, DEF_ANN, MAX_POS_REDUCE)
from factor_mining.base_data import load_panel
from factor_mining.gate_backtest_v2 import metrics, fmt

OUT_MD = "factor_mining/RESULT_crash_gate_hist.md"
START = "2005-01-01"


def gate_apply(port: pd.Series, crisis: pd.Series, tilt: pd.Series) -> pd.Series:
    idx = port.index
    cv = crisis.reindex(idx).fillna(False).values.astype(float)
    tv = tilt.reindex(idx).fillna(0.0).values
    buf = tv * MAX_POS_REDUCE
    pos = 1.0 - buf
    pos = np.where(cv > 0.5, np.minimum(pos, CRISIS_POS), pos)
    r_def = DEF_ANN / 252.0
    return pd.Series(pos * port.values + (1 - pos) * r_def, index=idx)


def main():
    t0 = time.time()
    # ---- 等权全市场组合(被保卫仓位) ----
    print("[crash] 加载全历史 close (等权组合)...", flush=True)
    close = load_panel("close", codes=None, start=START)
    close = close.sort_index()
    port = close.pct_change().mean(axis=1).dropna()   # 等权日收益
    print(f"[crash] 等权组合 {port.index.min().date()}~{port.index.max().date()} ({len(port)}日)", flush=True)

    # ---- 信号: 上证 + 巴指 ----
    warm = pd.date_range("2000-01-01", port.index.max(), freq="B")
    mkt = load_market_index("sh000001", index=warm).dropna()
    buffett, m2 = _load_macro(mkt)
    tilt = _macro_gating(buffett, m2, mkt)
    crisis = _crisis_signal(mkt, tilt=tilt, sens_gate=0.5)
    defended = gate_apply(port, crisis, tilt)
    print(f"[crash] 门控套用完成 | 危机门年均激活 {crisis.reindex(port.index).mean():.0%}", flush=True)

    # ---- 股灾窗口 ----
    windows = [
        ("2008 全球金融危机", "2008-01-01", "2008-12-31"),
        ("2015 股灾(全年)", "2015-01-01", "2015-12-31"),
        ("2015 年中acute(6-8月)", "2015-06-12", "2015-08-26"),
        ("2016 熔断", "2016-01-01", "2016-12-31"),
        ("2022 慢熊", "2022-01-01", "2022-12-31"),
        ("2024.1-2 微盘/DMA踩踏", "2024-01-02", "2024-02-29"),
        ("全历史 2005-2026", str(port.index.min().date()), str(port.index.max().date())),
    ]
    print(f"\n{'='*74}\n双尾防御门控 · 极端股灾抗压测试 (等权组合代理)\n{'='*74}")
    md = ["# 双尾防御门控 · 极端股灾抗压测试 (等权组合代理, 20年)\n",
          "- 等权全市场组合=被保卫仓位(透明代理, 不吃alpha); 门控=上证破位+巴指左侧缓冲+敏感耦合\n",
          "- 评判: 等权裸组合 vs 双尾门控 的区间收益/最大回撤\n",
          "| 股灾窗口 | 等权裸 年化 | 裸回撤 | 门控 年化 | 门控回撤 | 回撤削减 | 危机门激活 |",
          "|---|---|---|---|---|---|---|"]
    for lab, s, e in windows:
        b = port[s:e]
        g = defended[s:e]
        if len(b) < 20:
            continue
        mb, mg = metrics(b), metrics(g)
        cr = crisis.reindex(b.index).mean()
        print(f"\n### {lab} (n={len(b)}日)")
        print(f"  等权裸:  年化 {mb['ann']:+.1%} | 回撤 {mb['max_dd']:+.1%}")
        print(f"  双尾门控: 年化 {mg['ann']:+.1%} | 回撤 {mg['max_dd']:+.1%} | 危机门 {cr:.0%}")
        cut = abs(mb['max_dd']) - abs(mg['max_dd'])
        md.append(f"| {lab} | {mb['ann']:+.1%} | {mb['max_dd']:+.1%} | {mg['ann']:+.1%} | "
                  f"{mg['max_dd']:+.1%} | {cut:+.1%}pp | {cr:.0%} |")
    md.append(f"\n*耗时 {(time.time()-t0)/60:.1f}min · crash_gate_full_hist.py*")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n[crash] 结果 → {OUT_MD} | {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
