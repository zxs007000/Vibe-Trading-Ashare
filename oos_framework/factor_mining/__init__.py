#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 因子挖掘进阶工具箱（报告第四章实现）
==================================================
提供四大方向的可运行实现:
  1. 算子 + 变量网格搜索      -> grid_search.grid_search
  2. 遗传规划因子工厂          -> genetic_programming.evolve
  3. LLM + MCTS 公式化挖掘     -> llm_mcts.MCTSAgent (默认本地启发式, 可插 LLM)
  4. 微观结构挖掘(Level-2)     -> microstructure (骨架 + 合成数据演示)

共用底座: base_data(数据湖面板) / operators(算子+表达式引擎) / evaluate(IC/ICIR)
"""
from .base_data import (load_base_data, load_panel, list_stocks,
                        forward_returns, derive_variables)
from .operators import evaluate_expr, random_expr, expr_to_str, TS_OPS, CS_OPS, WINDOWS
from .evaluate import evaluate_factor, ic_series, turnover, factor_valid_ratio
from .grid_search import grid_search
from .genetic_programming import evolve
from .llm_mcts import MCTSAgent
from .microstructure import microstructure_features, demo_microstructure

__all__ = [
    "load_base_data", "load_panel", "list_stocks", "forward_returns", "derive_variables",
    "evaluate_expr", "random_expr", "expr_to_str", "TS_OPS", "CS_OPS", "WINDOWS",
    "evaluate_factor", "ic_series", "turnover", "factor_valid_ratio",
    "grid_search", "evolve", "MCTSAgent", "microstructure_features", "demo_microstructure",
]
