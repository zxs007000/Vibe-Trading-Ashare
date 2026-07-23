#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · P3 衰减感知因子筛选 (Decay-Aware Screening)
============================================================
位于 collect_factors(四方向挖矿) 与 build_feature_table_chunked(WFA 特征表) 之间,
对 100+ 候选因子做**衰减感知**预筛选, 只把「跨折稳定 + 近期折加权 IC 强」的因子送进 WFA。

为什么需要这一层(治"450 因子远期行、近期衰减"的病)
---------------------------------------------------
- 候选因子用整样本 IC 排序会偏向"历史辉煌但已衰减"的因子(未来信息泄露式的乐观)。
- 正确做法: 把历史切成若干块(默认按年, 近似 WFA 折), 计算每块 IC,
  用**指数衰减权重**(近期块权重高)聚合 → decayed_ic; 只认 decayed_ic 正且稳健的。
- 多重检验: 挖 100+ 因子必撞假阳性, 用 **Benjamini-Hochberg FDR**(q<0.10) 校正,
  只保留经多重检验仍显著者。
- 方向4(XGBoost 交互挖掘)的 chip 交互因子额外加权(用户: 三维度 + chip 强度大),
  作为 tie-break, 不绕过 FDR。

输出: select_factors() 返回 [(name, expr_tuple), ...] (精简后的入选因子), 供下游直接用。
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, false_discovery_control, norm

from factor_mining.base_data import list_stocks, load_base_data, derive_variables, forward_returns
from factor_mining.operators import evaluate_expr
from factor_mining.universe import load_universe

# 衰减窗(近似 WFA 折, 按年切; 每块内部用 trailing 口径避免前视)
DECAY_BLOCKS = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]
DECAY_HALFLIFE = 2.0   # 块指数衰减半衰期(2 年): 2025 权重最高
FDR_Q = 0.10            # Benjamini-Hochberg 阈值


def _block_ic(factor: pd.DataFrame, fwd: pd.DataFrame, sub_dates) -> float:
    """在 sub_dates(某历史块)内算 Rank-IC 均值; 样本不足返回 NaN。"""
    idx = factor.index.intersection(sub_dates)
    if len(idx) < 30:
        return np.nan
    f = factor.loc[idx].rank(pct=True, axis=1)
    r = fwd.reindex(idx).rank(pct=True, axis=1)
    ics = []
    for d in idx:
        a, b = f.loc[d], r.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < 30:
            continue
        rho = spearmanr(a[m], b[m]).correlation
        if np.isfinite(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else np.nan


def screen_factors(seen: dict, n_sub: int = 400, fwd_horizon: int = 20,
                   keep: int = 60, blocking_threshold: float = 0.006,
                   verbose: bool = True) -> list[tuple]:
    """
    seen: collect_factors 返回的 {str: {expr_tuple, dir, ic20, icir20, ...}}
    返回入选 [(name, expr_tuple), ...] (按 decayed_ic 降序)。
    """
    t0 = time.time()
    uni = load_universe()
    codes_all = list_stocks()
    if uni:
        rng = np.random.default_rng(7)
        codes = sorted(rng.choice(uni, min(n_sub, len(uni)), replace=False).tolist())
    else:
        codes = codes_all[:n_sub]
    data = derive_variables(load_base_data(codes))
    fwd_full = forward_returns(data["close"], horizons=(fwd_horizon,))[fwd_horizon]
    blocks = {b: pd.Timestamp(f"{b}-01-01") for b in DECAY_BLOCKS}
    weights = {b: 0.5 ** (len(DECAY_BLOCKS) - i - 1) / DECAY_HALFLIFE
               for i, b in enumerate(DECAY_BLOCKS)}
    wsum = sum(weights.values())

    rows = []
    for name, meta in seen.items():
        try:
            fac = evaluate_expr(meta["expr_tuple"], data)
        except Exception:
            continue
        block_ics = []
        for b in DECAY_BLOCKS:
            sub = fwd_full.index[fwd_full.index.year == int(b)]
            ic = _block_ic(fac, fwd_full, sub)
            block_ics.append(ic)
        w = np.array([weights[b] for b in DECAY_BLOCKS])
        ics = np.array([np.nan if not np.isfinite(x) else x for x in block_ics])
        decayed = float(np.nansum(ics * w) / (wsum if wsum else 1.0))
        n_valid = int(np.sum(np.isfinite(ics)))
        # 近期折(最近2块)IC 均值, 治衰减
        recent = float(np.nanmean(ics[-2:])) if n_valid >= 2 else np.nan
        rows.append({"name": name, "tuple": meta["expr_tuple"], "dir": meta["dir"],
                     "decayed_ic": decayed, "recent_ic": recent, "n_valid": n_valid,
                     "block_ics": ics})
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    # FDR 校正: decayed_ic 正态近似 z = ic·sqrt(有效样本天数) → 双尾 p 值 → BH
    # 有效天数近似: 每块约 240 交易日, IC 标准误 ≈ 1/sqrt(总天数)
    n_days = df["n_valid"].clip(lower=1) * 240
    df["z"] = df["decayed_ic"] * np.sqrt(n_days)
    df["p"] = 2.0 * norm.sf(df["z"].abs().values)   # 双尾 p 值
    try:
        qvals = false_discovery_control(df["p"].values, method="bh")
    except Exception:
        qvals = np.where(df["decayed_ic"].abs() >= blocking_threshold, 0.05, 0.5)
    df["q"] = qvals
    sig = df[(df["q"] < FDR_Q) & (df["decayed_ic"] > blocking_threshold) & (df["n_valid"] >= 3)].copy()
    # 方向4(XGBoost chip 交互) tie-break 加权: decayed_ic 乘 1.15(不绕过 FDR)
    sig["score"] = sig["decayed_ic"] * np.where(sig["dir"] == 4, 1.15, 1.0)
    sig = sig.sort_values("score", ascending=False)
    chosen = sig.head(keep)
    out = [(r["name"], r["tuple"]) for _, r in chosen.iterrows()]
    if verbose:
        print(f"[screen] 候选 {len(df)} → FDR显著 {int((df['q']<FDR_Q).sum())} → "
              f"decayed>阈值 {int((df['decayed_ic']>blocking_threshold).sum())} → 入选 {len(out)} "
              f"| {time.time()-t0:.0f}s", flush=True)
        print(f"[screen] 入选方向分布: {dict(chosen['dir'].value_counts())}", flush=True)
        for _, r in chosen.head(12).iterrows():
            print(f"   dir{r['dir']} decayed_ic={r['decayed_ic']:+.4f} recent={r['recent_ic']:+.4f} "
                  f"{r['name'][:80]}", flush=True)
    return out


if __name__ == "__main__":
    import json
    from factor_mining.factor_wfa import _l2t
    d = json.load(open("factor_mining/factors_v2_3dim.json", encoding="utf-8"))
    seen = {k: {"expr_tuple": _l2t(v["expr_tuple_list"]), "dir": v["dir"],
                "ic20": v["ic20"], "icir20": v["icir20"]} for k, v in d.items()}
    sel = screen_factors(seen, n_sub=400)
    print(f"\n入选 {len(sel)} 个, 前 5:")
    for n, t in sel[:5]:
        print(" ", n)
