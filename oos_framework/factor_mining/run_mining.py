#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 第四章四大方向统一编排脚本
=========================================
一键跑通报告第四章的全部四个因子挖掘进阶方向, 并产出结果摘要:
  方向1  算子 + 变量网格搜索      (grid_search)
  方向2  遗传规划因子工厂          (genetic_programming.evolve, NSGA-II 多目标)
  方向3  LLM + MCTS 公式化挖掘     (llm_mcts.MCTSAgent, 默认本地启发式, 可挂 LLM)
  方向4  微观结构挖掘(Level-2)     (microstructure, 合成 tick 演示)

用法
----
  python run_mining.py                 # 默认小样本(快速验收)
  python run_mining.py --stocks 400   # 更大样本
  python run_mining.py --out results.md

说明: 数据来自 /workspace/stocklake(daily 层)。方向4 需真实 Level-2 才具实战意义,
这里用合成 tick 演示流水线; 方向3 若设置 OPENAI_API_KEY 会自动启用 LLM 反馈闭环。
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_mining import (
    list_stocks, load_base_data, derive_variables, forward_returns,
    grid_search, evolve, MCTSAgent, microstructure_features, demo_microstructure,
)
from factor_mining.llm_mcts import _make_llm_client


def _hdr(t):
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


def run(n_stocks: int, out_path: str | None):
    t_all = time.time()
    codes = list_stocks(n_stocks)
    data = derive_variables(load_base_data(codes))
    fwd = forward_returns(data["close"])
    print(f"样本: {len(codes)} 只 | 变量池: {len(data)} | 交易日: {len(data['close'])}")

    md = []
    md.append(f"# 因子挖掘第四章 · 四大方向运行结果\n")
    md.append(f"- 样本: **{len(codes)} 只** A 股 | 变量池: {len(data)} | "
              f"区间: {data['close'].index.min().date()} ~ {data['close'].index.max().date()}\n")

    # ---- 方向1: 网格搜索 ----
    _hdr("方向1 · 算子 + 变量网格搜索")
    t = time.time()
    gs = grid_search(data, fwd, top_k=10)
    md.append("## 方向1 · 算子 + 变量网格搜索\n")
    md.append(f"正 IC 候选 Top{len(gs)} (耗时 {time.time()-t:.1f}s):\n")
    md.append("| 因子表达式 | IC(20d) | ICIR(20d) | 换手率 |")
    md.append("|---|---|---|---|")
    for r in gs:
        print(f"  {r['expr']:<44} ic20={r['ic20']:+.3f} icir20={r['icir20']:+.2f} to={r['turnover']:.3f}")
        md.append(f"| `{r['expr']}` | {r['ic20']:+.3f} | {r['icir20']:+.2f} | {r['turnover']:.3f} |")

    # ---- 方向2: 遗传规划 ----
    _hdr("方向2 · 遗传规划因子工厂 (NSGA-II 多目标)")
    t = time.time()
    gp_res = evolve(data, fwd, pop_size=30, generations=12, verbose=False)
    md.append("\n## 方向2 · 遗传规划因子工厂 (NSGA-II 多目标)\n")
    md.append(f"Pareto 前沿规模: **{len(gp_res['pareto'])}** | 最优单体: `{gp_res['best']}` | "
              f"耗时 {time.time()-t:.1f}s\n")
    md.append("| 因子表达式(Pareto) | ICIR(20d) | 换手率 | IC(20d) |")
    md.append("|---|---|---|---|")
    for r in gp_res["pareto"][:8]:
        print(f"  {r['expr']:<46} icir20={r['icir20']:+.2f} to={r['turnover']:.3f} ic20={r['ic20']:+.3f}")
        md.append(f"| `{r['expr']}` | {r['icir20']:+.2f} | {r['turnover']:.3f} | {r['ic20']:+.3f} |")

    # ---- 方向3: LLM + MCTS ----
    _hdr("方向3 · LLM + MCTS 公式化挖掘")
    llm = _make_llm_client()
    print(f"  LLM 钩子: {'已启用(OPENAI_API_KEY 已设置)' if llm else '未配置 → 本地启发式 MCTS'}")
    t = time.time()
    agent = MCTSAgent(data, fwd, max_depth=2, llm=llm, seed=7)
    mcts = agent.search(iterations=120, verbose=False)
    md.append("\n## 方向3 · LLM + MCTS 公式化挖掘\n")
    md.append(f"LLM 钩子: **{'启用' if llm else '未配置(本地启发式)'}** | "
              f"候选数: {len(mcts)} | 耗时 {time.time()-t:.1f}s\n")
    md.append("| 因子表达式(MCTS) | ICIR(20d) | IC(20d) | 换手率 | 访问次数 |")
    md.append("|---|---|---|---|---|")
    for c in mcts[:8]:
        print(f"  {c['expr']:<46} icir={c['icir']:+.2f} ic20={c['ic20']:+.3f} to={c['turnover']:.3f}")
        md.append(f"| `{c['expr']}` | {c['icir']:+.2f} | {c['ic20']:+.3f} | {c['turnover']:.3f} | {c['visits']} |")
    if llm:
        ok, added = agent.llm_refine([c["expr"] for c in mcts])
        if added:
            md.append("\nLLM 反馈闭环新增候选:\n")
            for a in added:
                md.append(f"- `{a['expr']}` icir={a['icir']:+.2f}")

    # ---- 方向4: 微观结构 ----
    _hdr("方向4 · 微观结构挖掘 (Level-2 / Tick, 合成数据演示)")
    t = time.time()
    mdf = demo_microstructure(n_stocks=6, ticks_per_stock=2000)
    md.append("\n## 方向4 · 微观结构挖掘 (Level-2 / Tick, 合成数据演示)\n")
    md.append(f"合成 {len(mdf)} 只股票 tick 的微观结构特征 (耗时 {time.time()-t:.1f}s)。"
              f"**接入真实 Level-2 时需替换 demo 的 tick 输入, 特征函数可直接复用。**\n")
    feats = [c for c in mdf.columns if c != "stock_id"]
    md.append("| 股票 | " + " | ".join(feats) + " |")
    md.append("|---|" + "|".join(["---"] * len(feats)) + "|")
    for _, row in mdf.iterrows():
        md.append("| " + row["stock_id"] + " | " + " | ".join(f"{row[c]:.4f}" for c in feats) + " |")

    md.append(f"\n---\n*总耗时 {time.time()-t_all:.1f}s · 由 `factor_mining/run_mining.py` 自动生成*")

    full = "\n".join(md)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full)
        print(f"\n结果已写入: {out_path}")
    return full


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=150)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run(args.stocks, args.out)
