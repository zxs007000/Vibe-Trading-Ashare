#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 算子库 + 因子表达式引擎
=========================================
实现报告第四章「算子+变量网格搜索 / 遗传规划 / LLM+MCTS」共用的:
  1. 时序算子(TS, 沿日期轴逐股计算) — 含 Numba JIT 加速版本(ts_mean / ts_rank, 见代码示例文档)
  2. 截面算子(CS, 沿股票轴逐日计算) — rank / zscore
  3. 二元数据算子(BIN) — 时序相关 ts_corr
  4. 因子表达式 DSL(嵌套元组) 与求值器 evaluate_expr / 随机生成 random_expr / 反序列化 expr_to_str

统一表达式表示(供 GP 与 MCTS 复用):
  叶子 : ('var', name) | ('const', value)
  TS   : ('ts', op_name, child, window)
  CS   : ('cs', op_name, child)
  BIN  : ('bin', op_name, child1, child2)
所有算子输入/输出均为「日期×股票」面板(pandas DataFrame)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit

# ───────────────────────── Numba 加速时序算子(1D, 参考代码示例文档) ─────────────────────────
@njit
def ts_mean_nb(arr, window):
    """时序滚动均值（Numba 加速）。"""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            out[i] = np.mean(arr[i - window + 1:i + 1])
    return out


@njit
def ts_rank_nb(arr, window):
    """时序滚动排名百分比（Numba 加速, 当前值在窗口内所处的分位）。"""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < window - 1:
            out[i] = np.nan
        else:
            wd = arr[i - window + 1:i + 1]
            out[i] = np.sum(wd <= arr[i]) / window
    return out


def _roll_panel(panel: pd.DataFrame, w: int, kernel) -> pd.DataFrame:
    """将 1D Numba 核逐列应用到面板（编译一次, 多列复用）。"""
    vals = panel.values.astype(np.float64)
    res = np.empty_like(vals, dtype=np.float64)
    for j in range(vals.shape[1]):
        res[:, j] = kernel(vals[:, j], w)
    return pd.DataFrame(res, index=panel.index, columns=panel.columns)


# ───────────────────────── 时序算子(TS) 注册表 ─────────────────────────
# 部分算子直接用 pandas 向量化(已 C 加速); 最重的 mean/rank 走 Numba。
def _ts_mean(p, w):
    return _roll_panel(p, w, ts_mean_nb)

def _ts_rank(p, w):
    return _roll_panel(p, w, ts_rank_nb)

def _ts_std(p, w):
    return p.rolling(w, min_periods=max(2, w // 2)).std()

def _ts_delta(p, w):
    return p - p.shift(w)

def _ts_pct(p, w):
    return p / p.shift(w) - 1.0

def _ts_min(p, w):
    return p.rolling(w, min_periods=max(2, w // 2)).min()

def _ts_max(p, w):
    return p.rolling(w, min_periods=max(2, w // 2)).max()

def _ts_zscore(p, w):
    m = p.rolling(w, min_periods=max(2, w // 2)).mean()
    s = p.rolling(w, min_periods=max(2, w // 2)).std()
    return (p - m) / s

TS_OPS = {
    "ts_mean": _ts_mean,
    "ts_rank": _ts_rank,
    "ts_std": _ts_std,
    "ts_delta": _ts_delta,
    "ts_pct": _ts_pct,
    "ts_min": _ts_min,
    "ts_max": _ts_max,
    "ts_zscore": _ts_zscore,
}

# ───────────────────────── 截面算子(CS, 逐日跨股票) ─────────────────────────
def _cs_rank(p):
    return p.rank(pct=True, axis=1)

def _cs_zscore(p):
    return (p - p.mean(axis=1, skipna=True)) / p.std(axis=1, skipna=True)

def _cs_mean(p):
    return p.mean(axis=1, skipna=True).to_frame()

def _cs_std(p):
    return p.std(axis=1, skipna=True).to_frame()

CS_OPS = {
    "cs_rank": _cs_rank,
    "cs_zscore": _cs_zscore,
}

# ───────────────────────── 二元数据算子(BIN) ─────────────────────────
def _bin_corr(a, b):
    # 时序相关: 逐股滚动相关
    return a.rolling(20, min_periods=10).corr(b)

def _bin_sub(a, b):
    return a - b

def _bin_div(a, b):
    return a / (b.replace(0, np.nan))

def _bin_mul(a, b):
    return a * b

BIN_OPS = {
    "bin_corr": _bin_corr,
    "bin_sub": _bin_sub,
    "bin_div": _bin_div,
    "bin_mul": _bin_mul,
}

WINDOWS = [5, 10, 20, 60]


# ───────────────────────── 因子表达式求值 / 生成 / 序列化 ─────────────────────────
def evaluate_expr(expr, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """递归求值因子表达式, 返回「日期×股票」面板。"""
    t = expr[0]
    if t == "var":
        return data[expr[1]]
    if t == "const":
        base = next(iter(data.values()))
        return pd.DataFrame(np.full(base.shape, expr[1]), index=base.index, columns=base.columns)
    if t == "ts":
        _, op, child, w = expr
        return TS_OPS[op](evaluate_expr(child, data), w)
    if t == "cs":
        _, op, child = expr
        return CS_OPS[op](evaluate_expr(child, data))
    if t == "bin":
        _, op, c1, c2 = expr
        return BIN_OPS[op](evaluate_expr(c1, data), evaluate_expr(c2, data))
    raise ValueError(f"未知表达式节点: {expr}")


def expr_to_str(expr) -> str:
    """将表达式转为可读公式字符串。"""
    t = expr[0]
    if t == "var":
        return expr[1]
    if t == "const":
        return str(expr[1])
    if t == "ts":
        return f"{expr[1]}({expr_to_str(expr[2])},{expr[3]})"
    if t == "cs":
        return f"{expr[1]}({expr_to_str(expr[2])})"
    if t == "bin":
        op = expr[1]
        op = op[4:] if op.startswith("bin_") else op  # bin_sub -> sub
        return f"{expr_to_str(expr[2])} {op} {expr_to_str(expr[3])}"
    return "?"


def random_expr(vars: list[str], max_depth: int = 3, rng: np.random.Generator | None = None,
                p_leaf: float = 0.35):
    """随机生成因子表达式树（供遗传规划 / MCTS 初始化与变异）。"""
    rng = rng or np.random.default_rng()
    if max_depth <= 0 or rng.random() < p_leaf:
        return ("var", str(rng.choice(vars)))
    kind = rng.choice(["ts", "cs", "bin"])
    if kind == "ts":
        op = str(rng.choice(list(TS_OPS)))
        w = int(rng.choice(WINDOWS))
        return ("ts", op, random_expr(vars, max_depth - 1, rng), w)
    if kind == "cs":
        op = str(rng.choice(list(CS_OPS)))
        return ("cs", op, random_expr(vars, max_depth - 1, rng))
    # bin
    op = str(rng.choice(list(BIN_OPS)))
    return ("bin", op, random_expr(vars, max_depth - 1, rng), random_expr(vars, max_depth - 1, rng))


def all_leaves(expr) -> list:
    """收集表达式所有叶子变量名（用于相关性/多样性判断）。"""
    t = expr[0]
    if t == "var":
        return [expr[1]]
    if t == "const":
        return []
    if t in ("ts", "cs"):
        return all_leaves(expr[2])  # expr[2] 是子表达式, 不是窗口
    if t == "bin":
        return all_leaves(expr[2]) + all_leaves(expr[3])
    return []
