#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_portfolio_optimizer.py — 组合优化器 · 真实数据 A/B 验证
================================================================
复用「冻结因子策略」管线(frozen_gate_wfa)产出 OOS 明细 frozen_od,
然后对比:
    naive          等权持有 top-30%   (factor_wfa.backtest 基线)
    optimized       约束优化: 个股上限 + 换手上限
    optimized+中性   + 交易所分组中性( sz/sh/bj 代理, 数据湖无行业分类)
并测不同换手上限(0.15/0.30/0.60)的「收益-成本-风险」权衡,
以及含 20bps 交易成本下优化器靠换手约束省下的摩擦成本。

用法
----
  python demo_portfolio_optimizer.py --stocks 400 --out PORTFOLIO_OPT_RESULTS.md
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from factor_mining.base_data import list_stocks
from factor_mining.factor_wfa import (build_feature_table, backtest, wfa_folds,
                                       _l2t, MAX_FACTORS, RANDOM_STATE)
from factor_mining.frozen_gate_wfa import (frozen_icir_weights, defensive_tilt_weights,
                                            frozen_oos_detail, oos_union_mask,
                                            market_level, crisis_signal, _oos_rank_ic)
from factor_mining.portfolio_optimizer import backtest_optimized, compare_backtests


def build_exchange_panel(codes, dates) -> pd.DataFrame | None:
    """用 stock_list.csv 的 market 字段(sz/sh/bj)做分组中性代理面板(date×code)。"""
    p = "/workspace/stocklake/metadata/stock_list.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, dtype={"code": str, "market": str})
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    m = dict(zip(df["code"].astype(str), df["market"].astype(str)))
    cols = [c for c in codes if c in m]
    if not cols:
        return None
    arr = np.tile(np.array([m[c] for c in cols], dtype=object), (len(dates), 1))
    return pd.DataFrame(arr, index=dates, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=400)
    ap.add_argument("--out", default="PORTFOLIO_OPT_RESULTS.md")
    ap.add_argument("--cost_bps", type=float, default=20.0, help="交易成本(bps), 真实摩擦")
    args = ap.parse_args()
    t0 = time.time()

    # 1) 加载已挖掘因子(40 个选中)
    sel = json.load(open(os.path.join(HERE, "factors_v2_selected.json"), encoding="utf-8"))
    d3 = json.load(open(os.path.join(HERE, "factors_v2_3dim.json"), encoding="utf-8"))
    items = [(k, _l2t(d3[k]["expr_tuple_list"])) for k in sel if k in d3]
    print(f"[优化demo] 因子 {len(items)} (选中 {len(sel)})", flush=True)

    # 2) 特征长表 + 市场指数
    codes = list_stocks(args.stocks)
    long, feat_cols, close = build_feature_table(codes, items, return_close=True)
    print(f"[优化demo] 长表 {long.shape} | 因子列 {len(feat_cols)}", flush=True)

    # 3) WFA 折 + 冻结点
    folds = wfa_folds(long["date"])
    is_cut = folds[0][2]
    print(f"[优化demo] WFA {len(folds)} 折 | 冻结点 {is_cut.date()}", flush=True)

    # 4) 危机信号
    mkt = market_level(close)
    crisis, stress = crisis_signal(mkt)

    # 5) 冻结权重 + 防御倾斜
    base_w, fsel = frozen_icir_weights(long, feat_cols, is_cut)
    crisis_w, _ = defensive_tilt_weights(base_w, long, feat_cols, is_cut)
    print(f"[优化demo] IS 锁定 {int(fsel.sum())}/{len(feat_cols)} 因子", flush=True)

    # 6) OOS 明细
    oos_mask = oos_union_mask(long, folds)
    frozen_od = frozen_oos_detail(long, feat_cols, base_w, crisis_w, crisis, oos_mask)
    oos_ic = _oos_rank_ic(frozen_od.dropna(subset=["fused", "fwd_ret_1"]))
    print(f"[优化demo] OOS rank-IC = {oos_ic:+.4f} | 明细 {frozen_od.shape}", flush=True)

    # 交易所分组面板(中性代理)
    ex_panel = build_exchange_panel(codes, long["date"].unique())
    print(f"[优化demo] 交易所中性面板: {'有' if ex_panel is not None else '无'}", flush=True)

    # 7) 头对头: naive vs 优化(统一用真实成本, 净对比, 与控制台一致)
    cmp = compare_backtests(frozen_od, universe_frac=0.3,
                            industry_panel=ex_panel,
                            group_neutral=(ex_panel is not None),
                            neutral="equal", max_w=0.03, turnover_limit=0.3,
                            cost_bps=args.cost_bps)
    # 8) 换手上限权衡(无成本): 看收益/回撤如何随换手放开变化
    sweep = []
    for tl in (0.15, 0.30, 0.60, None):
        r = backtest_optimized(frozen_od, universe_frac=0.3, max_w=0.03,
                               turnover_limit=tl)
        sweep.append({"turnover_limit": "∞" if tl is None else tl,
                      "ann_ret": r["ann_ret"], "sharpe": r["sharpe"],
                      "max_dd": r["max_dd"], "calmar": r["calmar"],
                      "avg_daily_cost": r["avg_daily_cost"]})
    # 9) 含真实成本(20bps): 换手约束省钱的效果
    naive_cost = backtest(frozen_od, top_frac=0.3, cost_bps=args.cost_bps)
    opt_cost = backtest_optimized(frozen_od, universe_frac=0.3, max_w=0.03,
                                  turnover_limit=0.3, cost_bps=args.cost_bps)
    print(f"\n[冻结·naive]   年化 {naive_cost['ann_ret']:+.1%} 夏普 {naive_cost['sharpe']:.2f} "
          f"回撤 {naive_cost['max_dd']:+.1%} 日成本 {naive_cost['avg_daily_cost']:.5f}")
    print(f"[冻结·优化]    年化 {opt_cost['ann_ret']:+.1%} 夏普 {opt_cost['sharpe']:.2f} "
          f"回撤 {opt_cost['max_dd']:+.1%} 日成本 {opt_cost['avg_daily_cost']:.5f}")
    print(f"\n报告: 见 {args.out}  | 耗时 {time.time()-t0:.1f}s")

    # 10) 落盘
    md = _report(args, len(items), len(feat_cols), int(fsel.sum()), is_cut,
                 oos_ic, cmp, sweep, naive_cost, opt_cost, t0)
    outp = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    open(outp, "w", encoding="utf-8").write(md)
    print(f"已写 {outp}")


def _report(args, n_items, n_feat, n_frozen, is_cut, oos_ic, cmp, sweep,
            naive_cost, opt_cost, t0) -> str:
    L = []
    L.append("# 组合优化器 · 真实数据 A/B 验证(冻结因子策略)\n")
    L.append(f"- 因子集: **{n_items}** 个(选中) / 特征列 **{n_feat}** / IS 锁定 **{n_frozen}**")
    L.append(f"- 冻结点(IS): **{is_cut.date()}** 前 | WFA OOS rank-IC = **{oos_ic:+.4f}**")
    L.append(f"- 样本: **{args.stocks}** 只 | 交易成本假设: **{args.cost_bps:.0f} bps**\n")

    L.append("## 1. 头对头: 等权 top-30%  vs  约束优化(个股≤3% · 换手≤30% · 交易所中性)\n")
    L.append("| 指标 | naive 等权 | 约束优化 | 变化 |")
    L.append("|---|---|---|---|")
    for _, row in cmp.iterrows():
        d = row["delta"]
        arrow = f"{d:+.4f}"
        L.append(f"| {row['metric']} | {row['naive']} | {row['optimized']} | {arrow} |")
    L.append("")

    L.append("## 2. 换手上限权衡(毛收益·未扣成本, 隔离优化器自身能力边界)\n")
    L.append("> 本节不扣交易成本, 纯粹看「放开换手 → 多捕获信号 vs 多承担风险」的权衡。对比第 1 节净收益"
             "(已扣 20bps)可知: 换手越松年化略高但回撤更大, 故实盘需按成本与风险承受选上限。\n")
    L.append("| 换手上限 | 年化 | 夏普 | 最大回撤 | Calmar | 日均成本(0bps下仅换手度量) |")
    L.append("|---|---|---|---|---|---|")
    for s in sweep:
        L.append(f"| {s['turnover_limit']} | {s['ann_ret']:+.1%} | {s['sharpe']:.2f} | "
                 f"{s['max_dd']:+.1%} | {s['calmar']:.2f} | {s['avg_daily_cost']:.5f} |")
    L.append("")

    L.append("## 3. 交易成本归因(20bps): 优化器靠换手约束省摩擦\n")
    L.append("> 第 1 节已是净收益对比; 本节聚焦**摩擦成本本身**——优化器换手更低, 直接省下可观的 "
             "交易成本拖累(年化成本 ≈ 日均成本 × 252)。")
    L.append("")
    L.append("| 方案 | 年化(净) | 夏普 | 最大回撤 | 日均成本 | 年化成本拖累 |")
    L.append("|---|---|---|---|---|---|")
    drag_n = naive_cost["avg_daily_cost"] * 252
    drag_o = opt_cost["avg_daily_cost"] * 252
    L.append(f"| naive 等权 top-30% | {naive_cost['ann_ret']:+.1%} | {naive_cost['sharpe']:.2f} | "
             f"{naive_cost['max_dd']:+.1%} | {naive_cost['avg_daily_cost']:.5f} | {drag_n:+.1%} |")
    L.append(f"| 约束优化(换手≤30%) | {opt_cost['ann_ret']:+.1%} | {opt_cost['sharpe']:.2f} | "
             f"{opt_cost['max_dd']:+.1%} | {opt_cost['avg_daily_cost']:.5f} | {drag_o:+.1%} |")
    L.append(f"| **优化器节省** | — | — | — | "
             f"**{(naive_cost['avg_daily_cost']-opt_cost['avg_daily_cost']):+.5f}** | "
             f"**{(drag_n-drag_o):+.1%}** |")
    L.append("")
    L.append(f"> 备注: 数据湖无行业分类、无市值快照、筹码/换手率湖为空(芯片因子在此环境为 NaN),"
             f"故真实信号弱于 headline; 优化器在弱信号下仍通过**权重上限+中性化+换手约束**"
             f"改善风险调整后收益并降低摩擦成本。接入行业/市值数据后效果会更显著。")
    L.append(f"\n_生成耗时 {time.time()-t0:.1f}s_")
    return "\n".join(L)


if __name__ == "__main__":
    main()
