#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
connect_optimizer.py — 信号融合器 → 组合优化器 串联（A/B 实证）
==========================================================================

把 signal_meta_learner 产出的四路信号（frozen / xgb / regime_blend / stacked）
各自接入 portfolio_optimizer.backtest_optimized，与「朴素等权周换手」
（factor_wfa.backtest, freq=5）做 A/B，验证「组合层约束优化」在
周换手低本版基础上的增量价值。

口径对齐（保证公平）
--------------------
  · 两套回测都用 **周换手（freq=5）** + **20bps 扣费**。
  · 朴素 = 等权持有 top-30%；优化 = 约束优化（个股 ≤5% 上限 + 周频低换手
    + 可选行业中性）。基准线 = 全市场等权（两套一致）。

用法
----
  python connect_optimizer.py --stocks 150 --out PORTFOLIO_OPT_SIGNAL_RESULTS.md
  python connect_optimizer.py --recompute        # 强制重算信号（默认命中缓存则复用）
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from factor_mining.signal_meta_learner import (build_signals, fuse_and_eval,
                                               _l2t, list_stocks)
from factor_mining.portfolio_optimizer import backtest_optimized
from factor_mining.factor_wfa import backtest

CACHE = os.path.join(HERE, "meta_ods_cache.parquet")
SIGNALS = ["frozen", "xgb", "regime_blend", "stacked"]
NAME_MAP = {"frozen": "冻结 ICIR（基准）", "xgb": "XGBoost WFA（基准）",
            "regime_blend": "M1 体制融合（主）", "stacked": "M2 岭回归（对照）"}
HDR = "| 信号 | 朴素周换手 年化 | 优化周换手 年化 | 夏普 | 最大回撤 | Calmar | 日成本 | Δ年化 |"
SEP = "|---|---|---|---|---|---|---|---|"


# ===========================================================================
# 信号构建（带 parquet 缓存，避免每次重跑 ~5min WFA）
# ===========================================================================
def build_ods(stocks: int, recompute: bool = False) -> dict:
    """返回 {signal: oos_detail(date,code,fused,fwd_ret_1)}；命中缓存则复用。"""
    if (not recompute) and os.path.exists(CACHE):
        df = pd.read_parquet(CACHE)
        if len(df) and "signal" in df.columns:
            print(f"[connect] 命中信号缓存 {CACHE}（{len(df)} 行）", flush=True)
            return {s: df[df["signal"] == s][["date", "code", "fused", "fwd_ret_1"]]
                    .reset_index(drop=True) for s in SIGNALS}

    sel = json.load(open(os.path.join(HERE, "factors_v2_selected.json"), encoding="utf-8"))
    d3 = json.load(open(os.path.join(HERE, "factors_v2_3dim.json"), encoding="utf-8"))
    items = [(k, _l2t(d3[k]["expr_tuple_list"])) for k in sel if k in d3]
    codes = list_stocks(stocks)
    print(f"[connect] 重建信号（{stocks} 只）…", flush=True)
    m, regime, folds, is_cut, feat_cols, xgb_od, oos_ic_xgb, n_frozen = build_signals(codes, items)
    res = fuse_and_eval(m, regime, cost_bps=20.0)
    ods = res["ods"]
    cache = pd.concat([v.assign(signal=s) for s, v in ods.items()], ignore_index=True)
    cache.to_parquet(CACHE, index=False)
    print(f"[connect] 信号已缓存 {CACHE}（{len(cache)} 行）", flush=True)
    return ods


# ===========================================================================
# A/B 主流程
# ===========================================================================
def run_ab(ods: dict, cost_bps: float = 20.0, freq: int = 5,
           max_w: float = 0.05, turnover_limit: float = 0.3, group_neutral: bool = False):
    """对四路信号各跑 朴素周换手 vs 约束优化周换手，返回明细列表。"""
    rows = []
    for s in SIGNALS:
        od = ods[s]
        naive = backtest(od, top_frac=0.3, cost_bps=cost_bps, rebalance_freq=freq)
        opt = backtest_optimized(od, universe_frac=0.3, cost_bps=cost_bps,
                                 rebalance_freq=freq, max_w=max_w,
                                 turnover_limit=turnover_limit,
                                 group_neutral=group_neutral)
        rows.append((s, naive, opt))
        print(f"[connect] {NAME_MAP[s]}: 朴素 {naive['ann_ret']:+.1%} → 优化 "
              f"{opt['ann_ret']:+.1%}（sh {opt['sharpe']:.2f}, "
              f"Δ {opt['ann_ret'] - naive['ann_ret']:+.1%}）", flush=True)
    return rows


def run_sensitivity(ods: dict, cost_bps: float = 20.0, freq: int = 5):
    """对主信号 M1 做优化器参数灵敏度（隔离 选择效应 vs 换手约束效应）。"""
    od = ods["regime_blend"]
    cfgs = {
        "无约束（仅个股≤5%）": dict(max_w=0.05, turnover_limit=None),
        "轻约束（≤5% + 周换手限0.3）": dict(max_w=0.05, turnover_limit=0.3),
        "紧约束（≤3% + 周换手限0.2）": dict(max_w=0.03, turnover_limit=0.2),
    }
    out = {}
    for name, kw in cfgs.items():
        r = backtest_optimized(od, universe_frac=0.3, cost_bps=cost_bps,
                               rebalance_freq=freq, group_neutral=False, **kw)
        out[name] = r
        print(f"[connect] M1 灵敏度 [{name}]: 年化 {r['ann_ret']:+.1%} | "
              f"sh {r['sharpe']:.2f} | 成本 {r['avg_daily_cost']:.5f}", flush=True)
    return out


# ===========================================================================
# 报告
# ===========================================================================
def build_report(args, rows, sens, t0) -> str:
    lines = []
    lines.append("# 信号融合器 → 组合优化器 串联实证（周换手低本版）\n")
    lines.append(f"*生成耗时 {time.time()-t0:.1f}s, 样本 {args.stocks} 只, "
                 f"周换手 freq={args.freq}, 成本 {args.cost_bps:.0f}bps*\n")
    lines.append("- 把四路信号（frozen / xgb / M1 体制融合 / M2 岭回归）各自接入 "
                 "`portfolio_optimizer.backtest_optimized`，与「朴素等权周换手」做 A/B。")
    lines.append("- **优化器约束**：个股 ≤5% 上限（`max_w=0.05`）+ 周频换手的低换手 "
                 "+ 换手上限 `turnover_limit=0.3`；行业/市值中性本轮未接（缺行业元数据，"
                 "见文末「后续扩展」）。")
    lines.append("- 两套回测口径完全一致（周换手 + 20bps + 全市场等权基准），A/B 公平。\n")

    lines.append("## 一、四路信号：朴素周换手 vs 约束优化周换手\n")
    lines.append(HDR)
    lines.append(SEP)
    for s, naive, opt in rows:
        d_ann = opt["ann_ret"] - naive["ann_ret"]
        lines.append(
            f"| {NAME_MAP[s]} | {naive['ann_ret']:+.1%} | {opt['ann_ret']:+.1%} | "
            f"{opt['sharpe']:.2f} | {opt['max_dd']:+.1%} | {opt['calmar']:.2f} | "
            f"{opt['avg_daily_cost']:.5f} | {d_ann:+.1%} |")

    # 推荐信号
    best = max(rows, key=lambda r: r[2]["ann_ret"])
    bs, _, bo = best
    naive_bs = next(r[1] for r in rows if r[0] == bs)
    lines.append(f"\n> **推荐生产组合**：`{NAME_MAP[bs]} 周换手 + 约束优化` "
                 f"（年化 {bo['ann_ret']:+.1%}, 夏普 {bo['sharpe']:.2f}, "
                 f"最大回撤 {bo['max_dd']:+.1%}）；对比同一信号朴素周换手 "
                 f"{naive_bs['ann_ret']:+.1%}，组合层增量 "
                 f"{bo['ann_ret']-naive_bs['ann_ret']:+.1%}。")

    lines.append("\n## 二、M1 主信号：优化器参数灵敏度\n")
    lines.append("- 验证「选股集中（softmax+个股上限）」与「换手约束」各自的贡献。\n")
    lines.append("| 配置 | 年化 | 夏普 | 最大回撤 | Calmar | 日成本 |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in sens.items():
        lines.append(f"| {name} | {r['ann_ret']:+.1%} | {r['sharpe']:.2f} | "
                     f"{r['max_dd']:+.1%} | {r['calmar']:.2f} | {r['avg_daily_cost']:.5f} |")

    lines.append("\n## 三、结论\n")
    # 计算整体增量
    avg_naive = np.mean([r[1]["ann_ret"] for r in rows])
    avg_opt = np.mean([r[2]["ann_ret"] for r in rows])
    best_naive = max(rows, key=lambda r: r[1]["ann_ret"])
    lines.append(f"- 周换手低本版上叠加约束优化：四路信号平均年化 "
                 f"{avg_naive:+.1%} → {avg_opt:+.1%}（Δ {avg_opt-avg_naive:+.1%}）。")
    naive_bs = next(r[1] for r in rows if r[0] == bs)
    lines.append(f"- 朴素周换手最优为 **{NAME_MAP[best_naive[0]]} "
                 f"({best_naive[1]['ann_ret']:+.1%})**；约束优化后最优为 "
                 f"**{NAME_MAP[bs]} ({bo['ann_ret']:+.1%})**，夏普 {bo['sharpe']:.2f}"
                 f"（对比同一信号朴素 {naive_bs['ann_ret']:+.1%}，"
                 f"Δ {bo['ann_ret']-naive_bs['ann_ret']:+.1%}）。")
    lines.append("- 约束优化在「弱 alpha + 周换手低本版」上的价值主要来自："
                 "① 个股 ≤5% 上限抑制单票黑天鹅；② 换手上限进一步压低调仓日冲击成本；"
                 "③ softmax 倾斜把权重适度集中于高信号名（相对等权）。")
    lines.append("- 注意：此为弱 alpha 信号，绝对收益受 2021-2025 震荡市拖累；"
                 "行业/市值中性、风险模型（方差项最优化）未接入，是下一步提升点。")

    lines.append("\n## 四、后续扩展\n")
    lines.append("- 接入 SW 行业 / 申万一级 + 市值面板 → 开启 `group_neutral=True`，"
                 "做规模/行业中性，降低风格漂移。")
    lines.append("- 接入协方差估计 → 用 `optimize_day_exact`（scipy SLSQP）带方差项风险最优化。")
    lines.append("- 防御门控（`gate=True`）叠加组合层，危机期自动降仓。")
    return "\n".join(lines)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=150)
    ap.add_argument("--out", default="PORTFOLIO_OPT_SIGNAL_RESULTS.md")
    ap.add_argument("--cost_bps", type=float, default=20.0)
    ap.add_argument("--freq", type=int, default=5, help="再平衡频率（5≈周换手）")
    ap.add_argument("--max_w", type=float, default=0.05)
    ap.add_argument("--turnover_limit", type=float, default=0.3)
    ap.add_argument("--recompute", action="store_true", help="强制重算信号（忽略缓存）")
    args = ap.parse_args()

    t0 = time.time()
    ods = build_ods(args.stocks, recompute=args.recompute)
    rows = run_ab(ods, cost_bps=args.cost_bps, freq=args.freq,
                  max_w=args.max_w, turnover_limit=args.turnover_limit)
    sens = run_sensitivity(ods, cost_bps=args.cost_bps, freq=args.freq)

    md = build_report(args, rows, sens, t0)
    outp = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    open(outp, "w", encoding="utf-8").write(md)
    print(f"\n报告: {outp}  (耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
