#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 方向1: 算子 + 变量网格搜索（Numba 加速版）
=============================================================
对应报告第四章 4.1 与代码示例文档第 1 节。
在「基础变量 × 时序算子 × 窗口 × 可选截面算子」的高维网格上枚举因子表达式,
逐一对齐 WFA 目标的 IC/ICIR, 输出 Top 候选。

与代码示例文档的差异(工程化改造):
  - 输入从 {var: 1D array} 升级为 {var: 日期×股票 面板}, 截面算子(cs_rank/cs_zscore)天然生效
  - 评估指标从「非空率」升级为「Rank-IC / ICIR」(报告第四章核心指标)
  - 时序均值/排名走 Numba JIT(见 operators.ts_mean_nb / ts_rank_nb)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .operators import TS_OPS, CS_OPS, evaluate_expr
from .evaluate import evaluate_factor, turnover, factor_valid_ratio

# 默认搜索空间
DEFAULT_TS = ("ts_mean", "ts_rank", "ts_std", "ts_delta", "ts_pct", "ts_zscore")
DEFAULT_CS = (None, "cs_rank", "cs_zscore")
DEFAULT_WINDOWS = (5, 10, 20, 60)
DEFAULT_HORIZONS = (5, 20, 60)


def grid_search(data: dict[str, pd.DataFrame], fwd_dict: dict[int, pd.DataFrame],
                ts_ops=DEFAULT_TS, cs_ops=DEFAULT_CS, windows=DEFAULT_WINDOWS,
                horizons=DEFAULT_HORIZONS, min_valid: float = 0.9,
                require_ic_pos: bool = True, top_k: int = 25) -> list[dict]:
    """
    网格搜索因子表达式。

    参数
    ----
    data      : {field: 面板}, 含 'close' 用于向前收益目标
    fwd_dict  : forward_returns(close) 的输出
    min_valid : 最小非空覆盖率
    top_k     : 返回的 Top 候选数

    返回
    ----
    list of dict: {expr(可读串), ic5, ic20, ic60, icir20, turnover, valid}
                  按 icir20 降序
    """
    records = []
    # 变量池: 除 'close' 外的所有基础变量(close 仍作为目标, 不作为因子输入以避免目标泄漏)
    vars_pool = [v for v in data if v != "close"]
    for var in vars_pool:
        for op in ts_ops:
            for w in windows:
                base = ("ts", op, ("var", var), w)
                for cs in cs_ops:
                    try:
                        factor = evaluate_expr(base, data)
                        if cs:
                            factor = CS_OPS[cs](factor)
                        valid = factor_valid_ratio(factor)
                        if valid < min_valid:
                            continue
                        ev = evaluate_factor(factor, fwd_dict, horizons)
                        e20 = ev[20]
                        if require_ic_pos and e20["ic"] <= 0:
                            continue
                        expr_str = expr_to_str_wrap(base, cs)
                        records.append({
                            "expr": expr_str,
                            "ic5": round(ev[5]["ic"], 4),
                            "ic20": round(e20["ic"], 4),
                            "ic60": round(ev[60]["ic"], 4),
                            "icir20": round(e20["icir"], 3),
                            "turnover": round(turnover(factor), 4),
                            "valid": round(valid, 3),
                        })
                    except Exception:
                        continue
    records.sort(key=lambda r: r["icir20"], reverse=True)
    return records[:top_k]


def expr_to_str_wrap(base, cs) -> str:
    from .operators import expr_to_str
    s = expr_to_str(base)
    return f"{cs}({s})" if cs else s


if __name__ == "__main__":
    from .base_data import load_base_data, forward_returns, list_stocks
    codes = list_stocks(300)
    data = load_base_data(codes)
    fwd = forward_returns(data["close"])
    top = grid_search(data, fwd, top_k=15)
    print(f"网格搜索候选 Top{len(top)} (样本 {len(codes)} 只):")
    for r in top:
        print(f"  {r['expr']:<32} ic20={r['ic20']:+.3f} icir20={r['icir20']:+.2f} to={r['turnover']:.3f}")
