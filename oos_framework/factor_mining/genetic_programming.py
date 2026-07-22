#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 方向2: 遗传规划因子工厂(Genetic Programming)
=============================================================
对应报告第四章 4.2 与代码示例文档第 2 节(DEAP 版)。
自动生成 Alpha 表达式树, 通过变异/交叉/选择进化, 引入多目标惩罚(高收益但高换手/高回撤的畸形因子下修)。

工程化改造(不依赖 DEAP, 零额外安装):
  - 用 sklearn 随机数实现表达式树种群、tournament 选择、子树变异/交叉
  - 多目标: 最大化 ICIR + 最小化换手(对应文档 weights=(1.0,-1.0)), 输出 Pareto 前沿
  - 适应度评估复用 evaluate 模块的真实 Rank-IC / 换手率
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .operators import evaluate_expr, random_expr, expr_to_str, all_leaves
from .evaluate import evaluate_factor, turnover, factor_valid_ratio


def _fitness(expr, data, fwd_dict, horizons=(20,)):
    """返回 (icir20, turnover, ic20); 无效/退化(零预测力)表达式给强惩罚。"""
    try:
        factor = evaluate_expr(expr, data)
        valid = factor_valid_ratio(factor)
        if valid < 0.8:
            return (-1.0, 100.0, 0.0)
        ev = evaluate_factor(factor, fwd_dict, horizons)
        e = ev[20]
        to = turnover(factor)
        ic = float(e["ic"])
        # 退化过滤: |IC| 过小(近乎零预测力) 或 换手≈0(近常数) 一律下修,
        # 防止「IC 均值≈0 但方差极小 → 虚假超高 ICIR」骗过多目标排序。
        if abs(ic) < 0.004 or to < 0.001:
            return (-1.0, 100.0, ic)
        return (float(e["icir"]), float(to), ic)
    except Exception:
        return (-1.0, 100.0, 0.0)


def _non_dominated(pop, fits):
    """轻量 NSGA-II 式非支配排序: 返回 Pareto 前沿索引(最大化 icir, 最小化 turnover)。"""
    n = len(pop)
    front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # j 支配 i: j 在两目标都不差且至少一目标更优
            ji = (fits[j][0] >= fits[i][0] and fits[j][1] <= fits[i][1]
                  and (fits[j][0] > fits[i][0] or fits[j][1] < fits[i][1]))
            if ji:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def _mutate(expr, vars, rng, max_depth=3):
    """随机替换一个子树为新的随机子树(对应文档 gp.mutUniform)。"""
    if rng.random() < 0.3 or not isinstance(expr, tuple) or len(expr) < 2:
        return random_expr(vars, max_depth, rng)
    pos = rng.integers(1, len(expr))
    new = list(expr)
    new[pos] = _mutate(expr[pos], vars, rng, max_depth - 1) if isinstance(expr[pos], tuple) else random_expr(vars, max_depth - 1, rng)
    return tuple(new)


def _crossover(a, b, rng):
    """单点子树交叉(对应文档 gp.cxOnePoint)。"""
    def _pick(e):
        nodes = []
        def _walk(x, path=()):
            if isinstance(x, tuple):
                nodes.append(x)
                for i, c in enumerate(x[1:]):
                    _walk(c, path + (i + 1,))
        _walk(e)
        return nodes
    na, nb = _pick(a), _pick(b)
    if not na or not nb:
        return a, b
    ta = na[rng.integers(0, len(na))]
    tb = nb[rng.integers(0, len(nb))]
    # 用 tb 替换 a 中第一个出现的 ta 形状子树(简化处理: 用 tb 替换 a 的随机节点)
    def _replace(e, target, repl, rng):
        if e == target:
            return repl
        if isinstance(e, tuple):
            return tuple([e[0]] + [_replace(c, target, repl, rng) for c in e[1:]])
        return e
    child = _replace(a, ta, tb, rng)
    return child, b


def evolve(data, fwd_dict, pop_size=40, generations=20, seed=42,
           vars=None, max_depth=3, top_k=10, verbose=True):
    """
    进化因子表达式种群。

    返回
    ----
    dict: {
      'pareto': [{'expr','icir20','turnover','ic20'}...] (Pareto 前沿, 按 icir 降序),
      'best'  : 最优单体(按 icir),
      'history': 每代最优 icir 序列
    }
    """
    rng = np.random.default_rng(seed)
    vars = vars or [v for v in data if v != "close"]
    pop = [random_expr(vars, max_depth, rng) for _ in range(pop_size)]
    history = []
    for gen in range(generations):
        fits = [_fitness(ind, data, fwd_dict) for ind in pop]
        front = _non_dominated(pop, fits)
        # 精英: 保留 Pareto 前沿
        elites = [pop[i] for i in front]
        # 记录当代最优 icir
        best_icir = max((f[0] for f in fits), default=-1)
        history.append(best_icir)
        if verbose:
            print(f"  [GP] gen {gen+1:>2}/{generations}  最优ICIR={best_icir:+.3f}  "
                  f"Pareto前沿={len(elites)}")
        # 育种种群
        children = list(elites)
        while len(children) < pop_size:
            i, j = rng.integers(0, len(pop)), rng.integers(0, len(pop))
            # tournament: 选 icir 高者作父代
            pa = pop[i] if fits[i][0] >= fits[j][0] else pop[j]
            pb = pop[rng.integers(0, len(pop))]
            c1, _ = _crossover(pa, pb, rng)
            c1 = _mutate(c1, vars, rng, max_depth)
            children.append(c1)
        pop = children[:pop_size]
    # 终代评估
    fits = [_fitness(ind, data, fwd_dict) for ind in pop]
    front = _non_dominated(pop, fits)
    pareto = sorted(
        [{"expr": expr_to_str(pop[i]), "icir20": round(fits[i][0], 3),
          "turnover": round(fits[i][1], 3), "ic20": round(fits[i][2], 4)}
         for i in front],
        key=lambda r: r["icir20"], reverse=True)
    best = max(range(len(pop)), key=lambda i: fits[i][0])
    return {"pareto": pareto[:top_k], "best": expr_to_str(pop[best]),
            "history": [round(x, 3) for x in history]}


if __name__ == "__main__":
    from .base_data import load_base_data, forward_returns, list_stocks, derive_variables
    codes = list_stocks(200)
    data = derive_variables(load_base_data(codes))
    fwd = forward_returns(data["close"])
    res = evolve(data, fwd, pop_size=30, generations=12, verbose=True)
    print("\nPareto 前沿 Top:")
    for r in res["pareto"]:
        print(f"  {r['expr']:<46} icir20={r['icir20']:+.2f} to={r['turnover']:.3f} ic20={r['ic20']:+.3f}")
