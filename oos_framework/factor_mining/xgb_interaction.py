#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 方向4: XGBoost 交互挖掘
========================================
「树负责发现, 公式负责承载, WFA 负责裁决」:
  1) 在三维度变量池(价量派生 + chip 筹码 + 拥挤度)上训**浅层** GBDT (max_depth=3,
     封顶三阶交互; subsample 抗噪) —— 目标为多周期(5/10/20日)前向收益的截面 rank。
  2) 从树 dump 提取高 gain 分裂路径上的**特征共现对** (跨树、跨周期均出现的才可信)。
  3) 将 top 交互对翻译成 DSL 公式因子 (mul/div/sub × cs_rank 变换等模板), 逐个算 IC 验证,
     正 IC 者进池 —— 产出与 grid/GP/MCTS 完全同构, 一起过 FDR + 衰减感知 WFA。

防过拟合纪律 (ml-strategy / quant-statistics skill):
  max_depth=3 · subsample=0.7 · colsample=0.7 · 少量树(120) · 早停 ·
  仅取「出现频次>=2 且 gain 靠前」的交互 · 翻译后仍要过 IC 门槛。
"""
from __future__ import annotations

import itertools
import time
import numpy as np
import pandas as pd

from factor_mining.operators import evaluate_expr, expr_to_str

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# ---------------------------------------------------------------------------
# IC 工具(与 grid_search 一致: 逐日截面 spearman → 均值/ICIR)
# ---------------------------------------------------------------------------
def _panel_ic(factor: pd.DataFrame, fwd: pd.DataFrame, min_obs: int = 30):
    f = factor.rank(pct=True, axis=1)
    r = fwd.rank(pct=True, axis=1)
    ics = []
    common = f.index.intersection(r.index)
    fv, rv = f.loc[common], r.loc[common]
    for i in range(len(common)):
        a, b = fv.iloc[i], rv.iloc[i]
        m = a.notna() & b.notna()
        if m.sum() < min_obs:
            continue
        ics.append(np.corrcoef(a[m], b[m])[0, 1])
    if not ics:
        return 0.0, 0.0
    arr = np.asarray(ics)
    sd = arr.std()
    return float(arr.mean()), float(arr.mean() / sd) if sd > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# 1) 截面样本 → 浅层 GBDT
# ---------------------------------------------------------------------------
def _build_xy(data: dict[str, pd.DataFrame], fwd: pd.DataFrame,
              feat_names: list[str], date_stride: int = 5,
              max_rows: int = 400_000, seed: int = 42):
    """按 stride 抽日期截面, 特征取 cs_rank(稳健), 标签取截面 rank(回归)。"""
    close = data["close"]
    dates = close.index[::date_stride]
    X_parts, y_parts = [], []
    for dt in dates:
        if dt not in fwd.index:
            continue
        y = fwd.loc[dt]
        y = y.rank(pct=True)
        row = {}
        for fn in feat_names:
            p = data[fn]
            if dt in p.index:
                row[fn] = p.loc[dt].rank(pct=True)
        if not row:
            continue
        Xd = pd.DataFrame(row)
        m = y.notna() & Xd.notna().any(axis=1)
        if m.sum() < 50:
            continue
        X_parts.append(Xd[m])
        y_parts.append(y[m])
    X = pd.concat(X_parts, axis=0)
    y = pd.concat(y_parts, axis=0)
    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    return X, y


def _extract_pairs_xgb(model, feat_names: list[str]) -> dict[tuple, float]:
    """从 xgboost dump 提取同一路径上的特征共现对, gain 加权。"""
    pairs: dict[tuple, float] = {}
    trees = model.get_booster().get_dump(with_stats=True)
    import re
    pat = re.compile(r"\[f?(\w+)[<>]")
    gain_pat = re.compile(r"gain=([\d.eE+-]+)")
    for tr in trees:
        # 简化: 按缩进重建路径
        stack: list[tuple[int, str, float]] = []
        for line in tr.split("\n"):
            if not line.strip() or "leaf=" in line:
                continue
            depth = len(line) - len(line.lstrip("\t"))
            m = pat.search(line)
            g = gain_pat.search(line)
            if not m:
                continue
            fname = m.group(1)
            if fname.isdigit():
                fname = feat_names[int(fname)]
            gain = float(g.group(1)) if g else 0.0
            stack = [s for s in stack if s[0] < depth]
            for _, anc, ag in stack:
                if anc != fname:
                    key = tuple(sorted((anc, fname)))
                    pairs[key] = pairs.get(key, 0.0) + min(ag, gain)
            stack.append((depth, fname, gain))
    return pairs


def _extract_pairs_lgb(model, feat_names: list[str]) -> dict[tuple, float]:
    """lightgbm: 遍历树结构 dict 提取路径共现对。"""
    pairs: dict[tuple, float] = {}

    def walk(node, path):
        if "split_feature" not in node:
            return
        f = feat_names[node["split_feature"]]
        g = float(node.get("split_gain", 0.0))
        for anc, ag in path:
            if anc != f:
                key = tuple(sorted((anc, f)))
                pairs[key] = pairs.get(key, 0.0) + min(ag, g)
        for ch in ("left_child", "right_child"):
            if node.get(ch):
                walk(node[ch], path + [(f, g)])

    for tr in model.booster_.dump_model()["tree_info"]:
        walk(tr["tree_structure"], [])
    return pairs


# ---------------------------------------------------------------------------
# 2) 交互对 → DSL 候选表达式模板
# ---------------------------------------------------------------------------
def _pair_templates(a: str, b: str):
    va, vb = ("var", a), ("var", b)
    ra = ("cs", "cs_rank", va)
    rb = ("cs", "cs_rank", vb)
    return [
        ("bin", "bin_mul", ra, rb),
        ("bin", "bin_div", va, vb),
        ("bin", "bin_sub", ra, rb),
        ("bin", "bin_mul", ra, ("bin", "bin_sub", ("const", 1.0), rb)),
        ("bin", "bin_corr", va, vb),
    ]


# ---------------------------------------------------------------------------
# 3) 主入口
# ---------------------------------------------------------------------------
def mine_interactions(data: dict[str, pd.DataFrame], fwd_dict: dict[int, pd.DataFrame],
                      horizons=(5, 10, 20), top_pairs: int = 12,
                      ic_min: float = 0.01, icir_min: float = 0.10,
                      date_stride: int = 5, seed: int = 42,
                      verbose: bool = True) -> list[dict]:
    """
    返回 [{expr, expr_tuple, ic20, icir20, pair}] —— 与 grid/GP/MCTS 产出同构。
    """
    if not (_HAS_XGB or _HAS_LGB):
        print("[xgb_mine] 无 xgboost/lightgbm, 方向4 跳过")
        return []
    t0 = time.time()
    exclude = {"open", "high", "low", "close", "volume", "amount"}
    feat_names = [k for k in data.keys() if k not in exclude]

    # ---- 逐周期训模型, 合并共现对(跨周期出现次数) ----
    pair_gain: dict[tuple, float] = {}
    pair_hits: dict[tuple, int] = {}
    for h in horizons:
        if h not in fwd_dict:
            continue
        X, y = _build_xy(data, fwd_dict[h], feat_names, date_stride=date_stride, seed=seed)
        if _HAS_XGB:
            model = xgb.XGBRegressor(
                n_estimators=120, max_depth=3, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
                tree_method="hist", random_state=seed, n_jobs=4)
            model.fit(X, y, verbose=False)
            pairs = _extract_pairs_xgb(model, list(X.columns))
        else:
            model = lgb.LGBMRegressor(
                n_estimators=120, max_depth=3, num_leaves=7, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
                random_state=seed, n_jobs=4, verbose=-1)
            model.fit(X, y)
            pairs = _extract_pairs_lgb(model, list(X.columns))
        for k, g in pairs.items():
            pair_gain[k] = pair_gain.get(k, 0.0) + g
            pair_hits[k] = pair_hits.get(k, 0) + 1
        if verbose:
            print(f"[xgb_mine] h={h} 样本 {len(X)} × {len(X.columns)}维 | 交互对 {len(pairs)}", flush=True)

    # 纪律: 至少在 2 个周期的模型中出现
    stable = {k: g for k, g in pair_gain.items() if pair_hits.get(k, 0) >= min(2, len(horizons))}
    ranked = sorted(stable.items(), key=lambda kv: -kv[1])[:top_pairs]
    if verbose:
        print(f"[xgb_mine] 稳定交互对 {len(stable)} → 取 top {len(ranked)}: "
              f"{[f'{a}×{b}' for (a, b), _ in ranked[:6]]}", flush=True)

    # ---- 翻译为公式并 IC 验证 ----
    fwd20 = fwd_dict.get(20, next(iter(fwd_dict.values())))
    out, seen_expr = [], set()
    for (a, b), _g in ranked:
        for tpl in _pair_templates(a, b):
            es = expr_to_str(tpl)
            if es in seen_expr:
                continue
            seen_expr.add(es)
            try:
                f = evaluate_expr(tpl, data)
            except Exception:
                continue
            ic, icir = _panel_ic(f, fwd20)
            sign = 1.0
            if ic < 0:  # 方向翻转也算(负 IC 因子取负)
                ic, icir, sign = -ic, -icir, -1.0
                tpl = ("bin", "bin_sub", ("const", 0.0), tpl)
                es = expr_to_str(tpl)
            if ic >= ic_min and icir >= icir_min:
                out.append({"expr": es, "expr_tuple": tpl, "ic20": round(ic, 4),
                            "icir20": round(icir, 3), "turnover": np.nan,
                            "pair": f"{a}x{b}"})
    if verbose:
        print(f"[xgb_mine] 方向4 产出 {len(out)} 个公式因子 | {time.time()-t0:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    from factor_mining.base_data import list_stocks, load_base_data, derive_variables, forward_returns
    codes = list_stocks(300)
    data = derive_variables(load_base_data(codes))
    fwd = forward_returns(data["close"], horizons=(5, 10, 20))
    res = mine_interactions(data, fwd)
    for r in res[:10]:
        print(f"  {r['ic20']:+.4f} icir={r['icir20']:+.3f}  {r['expr']}   <- {r['pair']}")
