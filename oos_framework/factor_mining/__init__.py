#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 因子挖掘进阶工具箱（报告第四章实现）
==================================================
提供六大方向的可运行实现:
  1. 算子 + 变量网格搜索      -> grid_search.grid_search
  2. 遗传规划因子工厂          -> genetic_programming.evolve
  3. LLM + MCTS 公式化挖掘     -> llm_mcts.MCTSAgent (默认本地启发式, 可插 LLM)
  4. 微观结构挖掘(Level-2)     -> microstructure (骨架 + 合成数据演示)
  5. 因子 → XGBoost WFA 验证   -> factor_wfa (把四方向挖出的因子当特征, XGBoost 做
                                 Walk-Forward 样本外验证 + 样本外滚动回测, 含防御门控)
  6. 冻结因子策略 + 防御门控    -> frozen_gate_wfa (IS 锁定因子集 + ICIR 加权, OOS 零重学习,
                                 危机降仓; 与 factor_wfa.backtest 的防御门控同源)

共用底座: base_data(数据湖面板) / operators(算子+表达式引擎) / evaluate(IC/ICIR)
"""
from .base_data import (load_base_data, load_panel, list_stocks,
                        forward_returns, derive_variables, turnover_available,
                        load_turnover, load_float_shares)
from .operators import evaluate_expr, random_expr, expr_to_str, TS_OPS, CS_OPS, WINDOWS
from .evaluate import evaluate_factor, ic_series, turnover, factor_valid_ratio
from .grid_search import grid_search
from .genetic_programming import evolve
from .llm_mcts import MCTSAgent
from .microstructure import microstructure_features, demo_microstructure
from .factor_wfa import (
    collect_factors, build_feature_table, run_wfa, backtest, write_results, wfa_folds,
)
from .frozen_gate_wfa import (
    market_level, crisis_signal, frozen_icir_weights, defensive_tilt_weights,
    frozen_oos_detail, oos_union_mask, bt_no_daily, bt_gate_daily, crisis_seg_dd,
    build_report,
)

__all__ = [
    "load_base_data", "load_panel", "list_stocks", "forward_returns", "derive_variables",
    "turnover_available", "load_turnover", "load_float_shares",
    "evaluate_expr", "random_expr", "expr_to_str", "TS_OPS", "CS_OPS", "WINDOWS",
    "evaluate_factor", "ic_series", "turnover", "factor_valid_ratio",
    "grid_search", "evolve", "MCTSAgent", "microstructure_features", "demo_microstructure",
    "collect_factors", "build_feature_table", "run_wfa", "backtest", "write_results", "wfa_folds",
    "market_level", "crisis_signal", "frozen_icir_weights", "defensive_tilt_weights",
    "frozen_oos_detail", "oos_union_mask", "bt_no_daily", "bt_gate_daily",
    "crisis_seg_dd", "build_report",
]
