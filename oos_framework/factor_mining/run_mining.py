#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 第六章六大方向统一编排脚本
=========================================
一键跑通因子挖掘的全部六个进阶方向, 并产出结果摘要:
  方向1  算子 + 变量网格搜索       (grid_search)
  方向2  遗传规划因子工厂           (genetic_programming.evolve, NSGA-II 多目标)
  方向3  LLM + MCTS 公式化挖掘      (llm_mcts.MCTSAgent, 默认本地启发式, 可挂 LLM)
  方向4  微观结构挖掘(Level-2)      (microstructure, 合成 tick 演示)
  方向5  因子 → XGBoost WFA 验证    (factor_wfa: 四方向挖出的因子当特征, XGBoost 做
                                    Walk-Forward 样本外验证 + 样本外滚动回测)
  方向6  冻结因子策略 + 防御门控     (frozen_gate_wfa: IS 锁定因子集 + ICIR 加权, OOS 零重学习,
                                    危机降仓; 与方向5 回测的防御门控同源)

用法
----
  python run_mining.py                  # 默认小样本(快速验收方向1~4)
  python run_mining.py --stocks 400     # 更大样本
  python run_mining.py --xgb            # 额外跑方向5+6(XGBoost WFA, 小样本验收)
  python run_mining.py --xgb --mine-xgb 150 --stocks-xgb 400
  python run_mining.py --out results.md

说明: 数据来自 /workspace/stocklake(daily 层)。方向4 需真实 Level-2 才具实战意义,
这里用合成 tick 演示流水线; 方向3 若设置 OPENAI_API_KEY 会自动启用 LLM 反馈闭环;
方向5/6 默认不跑(重计算), 用 --xgb 开启, 全量运行请直接调用 factor_wfa.py / frozen_gate_wfa.py。
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
from factor_mining import (
    collect_factors, build_feature_table, run_wfa, backtest, write_results,
)
from factor_mining import (
    market_level, crisis_signal,
)


def _hdr(t):
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


def run(n_stocks: int, out_path: str | None,
        run_xgb: bool = False, mine_xgb: int = 120, stocks_xgb: int = 400):
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

    # ---- 方向5 + 方向6: XGBoost WFA 验证 & 冻结因子防御门控 ----
    # 重计算, 默认跳过; 用 --xgb 开启(小样本验收)。全量运行请直接调 factor_wfa.py。
    if run_xgb:
        _hdr("方向5 · 因子 → XGBoost WFA 验证 (--xgb)")
        t = time.time()
        seen = collect_factors(mine_xgb)
        # 按 ICIR 截取验证用因子(对齐 factor_wfa.MAX_FACTORS, 这里小样本直接全用)
        items = sorted(seen.items(), key=lambda kv: kv[1]["icir20"], reverse=True)
        from factor_mining.factor_wfa import MAX_FACTORS
        if len(items) > MAX_FACTORS:
            items = items[:MAX_FACTORS]
        exprs = [(k, v["expr_tuple"]) for k, v in items]
        from collections import Counter
        dc = Counter(v["dir"] for v in seen.values())
        print(f"  发现因子(正IC去重): {len(seen)} | 方向1 {dc.get(1,0)} / "
              f"方向2 {dc.get(2,0)} / 方向3 {dc.get(3,0)} | 用于验证: {len(exprs)}")
        codes = list_stocks(stocks_xgb)
        long, feat_cols, close = build_feature_table(codes, exprs, return_close=True)
        rows, imp_acc, folds, oos_detail = run_wfa(long, feat_cols)
        bt = backtest(oos_detail)
        full, rec, mean = write_results(
            None,
            {"mine": mine_xgb, "stocks": stocks_xgb, "n_total": len(seen)},
            rows, imp_acc, folds, len(exprs), bt=bt,
        )
        md.append("\n## 方向5 · 因子 → XGBoost WFA 验证\n")
        md.append(f"发现因子(正IC去重): **{len(seen)}** (方向1 {dc.get(1,0)} / "
                  f"方向2 {dc.get(2,0)} / 方向3 {dc.get(3,0)}) | 用于验证: **{len(exprs)}** | "
                  f"折数: {len(folds)} | 耗时 {time.time()-t:.1f}s\n")
        cols = ["fold", "auc_single5", "ic_single5", "auc_fuse_5", "auc_fuse_20",
                "auc_fuse_60", "ic_fuse_5", "ic_fuse_20", "ic_fuse_60"]
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in rec.iterrows():
            md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        md.append("| **均值** | " + " | ".join(
            f"**{mean.get(c, '')}**" for c in cols if c != "fold") + " |")
        if bt:
            md.append(f"\n**样本外回测** ({bt['start']}~{bt['end']}, {bt['years']}y): "
                      f"年化 **{bt['ann_ret']:+.1%}** vs 基准 {bt['ann_base']:+.1%} | "
                      f"夏普 {bt['sharpe']:.2f} | 最大回撤 **{bt['max_dd']:+.1%}**\n")
        else:
            md.append("\n(样本不足, 未产出 OOS 回测)\n")

        _hdr("方向6 · 冻结因子策略 + 防御门控 (--xgb)")
        md.append("\n## 方向6 · 冻结因子策略 + 防御门控\n")
        # 用 close 面板建市场等权指数 + 双层危机信号(右侧 crisis + 左翼 stress),
        # 复用方向5 的 OOS 融合概率做带防御门控的回测对照。
        mkt = market_level(close)
        crisis, stress = crisis_signal(mkt)
        if oos_detail is not None and len(oos_detail) > 0:
            bt_gate = backtest(oos_detail, gate=True, crisis=crisis, stress=stress,
                               crisis_pos=0.60, def_ann=0.04, max_pos_reduce=0.20)
            if bt_gate:
                no_gate = (f"年化 {bt['ann_ret']:+.1%} / 最大回撤 {bt['max_dd']:+.1%}"
                           if bt else "N/A(无门控明细)")
                md.append(f"方向5 组合 OOS {no_gate} → "
                          f"叠加防御门控后 年化 {bt_gate['ann_ret']:+.1%} / "
                          f"最大回撤 **{bt_gate['max_dd']:+.1%}** "
                          f"(危机降仓天数 {bt_gate['n_crisis_days']})\n")
                md.append("> 冻结因子集在 IS 锁定、OOS 零重学习; 防御门控与方向5 回测同源, "
                          "全量冻结因子方案见 `factor_mining/frozen_gate_wfa.py`。\n")
            else:
                md.append("(防御门控回测未产出)\n")
        else:
            md.append("(方向5 无 OOS 明细, 跳过; 全量请直接运行 `frozen_gate_wfa.py`)\n")
    else:
        _hdr("方向5/6 · 因子 → XGBoost WFA 验证 / 冻结因子防御门控 (未运行)")
        print("  [跳过] 方向5/6 重计算, 用 --xgb 开启 (全量请直接调 factor_wfa.py / frozen_gate_wfa.py)")
        md.append("\n## 方向5/6 · 因子 → XGBoost WFA 验证 / 冻结因子防御门控\n")
        md.append("> 默认未运行(重计算)。开启: `python run_mining.py --xgb`; "
                  "全量: `python factor_wfa.py all` / `python frozen_gate_wfa.py`。\n")

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
    ap.add_argument("--xgb", action="store_true",
                    help="额外跑方向5+6: 因子→XGBoost WFA 验证 + 冻结因子防御门控(重计算)")
    ap.add_argument("--mine-xgb", type=int, default=120,
                    help="方向5/6 的因子挖掘样本数(默认120, 小样本验收)")
    ap.add_argument("--stocks-xgb", type=int, default=400,
                    help="方向5/6 的 WFA 验证样本数(默认400, 全量请直接调 factor_wfa.py)")
    args = ap.parse_args()
    run(args.stocks, args.out, run_xgb=args.xgb,
        mine_xgb=args.mine_xgb, stocks_xgb=args.stocks_xgb)
