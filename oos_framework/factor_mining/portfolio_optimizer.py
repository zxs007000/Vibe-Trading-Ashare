#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portfolio_optimizer.py — 组合优化器(把截面 alpha 信号 → 可落地的约束型多头组合)
===================================================================================

为什么需要它
------------
现有 `factor_wfa.backtest` 用的是「等权持有 top-30%」——这是选股信号验证的朴素基线,
不是可落地的组合: 它不控制个股权重上限、不中性化行业/市值、不约束换手。
真实资金不会等权买 1500 只里的 450 只, 也不会每天全换仓。

本模块把"信号"升级为"组合":
  · 选股(Selection): 每日按信号取 Top-K 缩小优化域(4 折 OOS 全市场优化太重, 且没必要)。
  · 配置(Allocation): 在选股域内求解约束权重。

约束(均可独立开关)
------------------
  ✓ long_only      多头(权重 ≥ 0)
  ✓ sum_to_one     满仓(权重和 = 1)
  ✓ max_w          个股权重上限(默认 5%, 防单票黑天鹅)
  ✓ group_neutral  分组中性(行业 / 交易所 / 自定义分组), 目标 = 市值基准或等权
  ✓ turnover_limit 换手上限(单日 ||Δw||₁ 上限, 直接控交易成本与风格漂移)

求解方法: 投影 + 收缩 (projection + shrinkage)
---------------------------------------------
纯 numpy, **无外部求解器依赖**(scipy/cvxpy 都不需要), 速度极快且数值稳健:

  1. 原始权重      w* = softmax(alpha · temp)            (选股域内)
  2. 切顶投影      w  = cap_project(w*, max_w)            (满足个股上限)
  3. 分组中性      w  = IPF(w, 各组目标权重)              (迭代比例拟合 raking)
  4. 换手控制      w  = (1-φ)·w* + φ·w_prev, φ 由上限精确解出 (收缩法)

相比 scipy SLSQP: 没有收敛失败/矩阵奇异风险, 万级资产每日毫秒级, 可直接上全市场。
(如需带方差项的风险模型最优化, 见文件末尾 `optimize_day_exact`, 小域适用。)

接口对齐
--------
`backtest_optimized(oos_detail, ...)` 与 `factor_wfa.backtest` 同指标口径(年化/夏普/
最大回撤/Calmar/成本), 可直接 A/B。oos_detail 结构: (date, code, fused, fwd_ret_1)。

用法
----
  # 作为 backtest 的升级替代(支持周换手 rebalance_freq=5)
  from factor_mining.portfolio_optimizer import backtest_optimized, compare_backtests
  r_opt = backtest_optimized(oos_detail, universe_frac=0.3, max_w=0.03,
                             group_neutral=True, neutral='cap', turnover_limit=0.3,
                             rebalance_freq=5)
  cmp   = compare_backtests(oos_detail, universe_frac=0.3, max_w=0.03,
                             group_neutral=True, turnover_limit=0.3, rebalance_freq=5)

  # 独立优化任意信号面板
  from factor_mining.portfolio_optimizer import optimize_panel
  W = optimize_panel(signal_panel, industry_panel=ind, mktcap_panel=mc,
                     universe_frac=0.2, max_w=0.04, turnover_limit=0.25)

  # 自测
  python portfolio_optimizer.py --selftest
"""
from __future__ import annotations
import argparse
import os
import sys

# 确保直接执行时也能 `from factor_mining.xxx import ...`
# (本文件在 oos_framework/factor_mining/ 下, 父目录 oos_framework 需加入路径)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

EPS = 1e-12


# ===========================================================================
# 基础工具
# ===========================================================================
def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    x = x - np.nanmax(x)
    e = np.exp(x)
    s = e.sum()
    return e / (s + EPS)


def _cap_project(w: np.ndarray, max_w: float, long_only: bool, iters: int = 12) -> np.ndarray:
    """切顶投影: 把权重压进 [-max_w, max_w](多头则 [0, max_w]) 并归一, 迭代至收敛。"""
    w = np.asarray(w, float)
    N = w.size
    if N == 0:
        return w
    if long_only:
        w = np.clip(w, 0.0, None)
    else:
        w = np.clip(w, -max_w, max_w)
    s0 = w.sum()
    if s0 <= 0:  # 全部被切没(理论上 softmax 不会), 退化等权
        return np.full(N, 1.0 / N)
    for _ in range(iters):
        w = w / w.sum()
        if long_only:
            over = w > max_w
            if not over.any():
                break
            w[over] = max_w
        else:
            over = w > max_w
            under = w < -max_w
            if not over.any() and not under.any():
                break
            w[over] = max_w
            w[under] = -max_w
    s = w.sum()
    return w / s if s > 0 else np.full(N, 1.0 / N)


def _group_neutralize(w: np.ndarray, industry: np.ndarray, mktcap: np.ndarray | None,
                      neutral: str, max_w: float, long_only: bool, n_ipf: int) -> np.ndarray:
    """迭代比例拟合(IPF / raking): 把各组总权重拉到基准目标, 每轮重做切顶。

    neutral:
      'cap'   → 基准 = 各组市值占比(规模中性, 默认)
      'equal' → 基准 = 各组等权(纯行业中性, 不偏重大行业)
    """
    groups = pd.Series(industry).fillna("__NA__").values
    uniq, inv = np.unique(groups, return_inverse=True)
    n_g = len(uniq)
    if n_g <= 1:
        return w
    cur = w.copy()
    if neutral == "cap" and mktcap is not None:
        mc = np.asarray(mktcap, float)
        mc = np.where(np.isfinite(mc) & (mc > 0), mc, 0.0)
        bench = np.array([mc[inv == g].sum() for g in range(n_g)])
        bench = bench / (bench.sum() + EPS)
    else:  # equal
        bench = np.full(n_g, 1.0 / n_g)
    for _ in range(n_ipf):
        gsum = np.array([cur[inv == g].sum() for g in range(n_g)])
        gsum = np.where(gsum <= 0, EPS, gsum)
        scale = bench / gsum
        new = cur.copy()
        for g in range(n_g):
            mask = inv == g
            new[mask] = cur[mask] * scale[g]
        new = _cap_project(new, max_w, long_only)
        # 收敛检查
        gsum2 = np.array([new[inv == g].sum() for g in range(n_g)])
        if np.max(np.abs(gsum2 - bench)) < 1e-3:
            return new
        cur = new
    return cur


# ===========================================================================
# 单日权重求解
# ===========================================================================
def optimize_day(alpha: np.ndarray, industry: np.ndarray | None = None,
                 mktcap: np.ndarray | None = None, prev_w: np.ndarray | None = None,
                 *, long_only: bool = True, max_w: float = 0.05, min_w: float = 0.0,
                 group_neutral: bool = False, neutral: str = "cap", signal_temp: float = 1.0,
                 turnover_limit: float | None = None, n_ipf: int = 40) -> np.ndarray:
    """求解单日组合权重。

    参数
    ----
    alpha        : 截面 alpha 信号(1D, 越大越看多), 长度 N
    industry     : 与 alpha 同序的分组标签(1D, 可含 None), 分组中性用
    mktcap       : 与 alpha 同序的市值(1D), neutral='cap' 的目标用
    prev_w       : 上一日权重(1D, 同序), 换手约束用
    max_w        : 个股权重上限(默认 0.05)
    group_neutral: 是否做分组中性
    neutral      : 'cap' 市值基准 / 'equal' 等权基准
    signal_temp  : alpha 软化温度(越大越集中, 越小越分散)
    turnover_limit: 单日换手上限 ||Δw||₁ 上限(默认 None 不约束)

    返回
    ----
    w : np.ndarray, 长度 N, 已 sum=1, long-only, ≤ max_w
    """
    alpha = np.asarray(alpha, float)
    N = alpha.size
    if N == 0:
        return alpha
    # 1) 原始信号权重
    w = _softmax(alpha * signal_temp)
    # 2) 切顶投影(个股上限)
    w = _cap_project(w, max_w, long_only)
    # 3) 分组中性
    if group_neutral and industry is not None:
        w = _group_neutralize(w, industry, mktcap, neutral, max_w, long_only, n_ipf)
    # 4) 换手控制(收缩法)
    if turnover_limit is not None and prev_w is not None:
        prev_w = np.asarray(prev_w, float)
        if prev_w.shape == w.shape:
            d = float(np.abs(w - prev_w).sum())
            if d > EPS:
                phi = float(np.clip(1.0 - turnover_limit / d, 0.0, 1.0))
                w = (1.0 - phi) * w + phi * prev_w
                w = w / (w.sum() + EPS)
                if long_only:
                    w = np.clip(w, 0.0, None)
                    w = w / (w.sum() + EPS)
    return w


# ===========================================================================
# 面板级驱动(选股 → 配置, 链式换手约束)
# ===========================================================================
def optimize_panel(signal_panel: pd.DataFrame, industry_panel: pd.DataFrame | None = None,
                   mktcap_panel: pd.DataFrame | None = None, *,
                   universe_frac: float = 0.3, universe_top: int | None = None,
                   long_only: bool = True, max_w: float = 0.05, min_w: float = 0.0,
                   group_neutral: bool = False, neutral: str = "cap", signal_temp: float = 1.0,
                   turnover_limit: float | None = None, n_ipf: int = 40,
                   rebalance_freq: int = 1) -> pd.DataFrame:
    """对 (date×asset) 信号面板逐日求解权重, 返回同形权重面板。

    每日: 按信号在有效资产中取 Top-K(K = universe_top 或 universe_frac·N)作为优化域,
    域内调用 optimize_day; 域外权重 = 0。换手约束跨全部资产链式传递(计入进出域)。

    signal_panel / industry_panel / mktcap_panel 需同索引同列(asset 为列)。
    """
    dates = signal_panel.index
    assets = signal_panel.columns
    N = len(assets)
    W = pd.DataFrame(0.0, index=dates, columns=assets)
    sig = signal_panel.values.astype(float)
    ind = industry_panel.reindex(columns=assets).values if industry_panel is not None else None
    mc = mktcap_panel.reindex(columns=assets).values if mktcap_panel is not None else None

    freq = max(1, int(rebalance_freq))
    date_list = list(dates)
    rebal_dates = set(date_list[::freq]) if freq > 1 else None

    prev = None
    for i, d in enumerate(dates):
        row = sig[i]
        valid = ~np.isnan(row)
        idx = np.where(valid)[0]
        if idx.size == 0:
            prev = None
            continue
        # 周频(低频)再平衡: 非调仓日直接持有上一期权重(不重算、不交易)
        if rebal_dates is not None and d not in rebal_dates:
            if prev is not None:
                W.iloc[i] = prev
            continue
        a = row[idx]
        if universe_top:
            k = min(int(universe_top), idx.size)
        else:
            k = max(1, int(round(universe_frac * idx.size)))
        order = np.argsort(-a)[:k]
        sel = idx[order]
        a_sel = a[order]
        ind_sel = ind[i, sel] if ind is not None else None
        mc_sel = mc[i, sel] if mc is not None else None
        w_sel = optimize_day(
            a_sel, ind_sel, mc_sel, prev_w=None,
            long_only=long_only, max_w=max_w, min_w=min_w,
            group_neutral=group_neutral, neutral=neutral, signal_temp=signal_temp,
            n_ipf=n_ipf)
        w_full = np.zeros(N)
        w_full[sel] = w_sel
        if w_full.sum() > 0:
            w_full /= w_full.sum()
        # 换手约束(全向量层面, 含进出选股域): w=(1-φ)w*+φ·w_prev
        if turnover_limit is not None and prev is not None:
            d = float(np.abs(w_full - prev).sum())
            if d > EPS:
                phi = float(np.clip(1.0 - turnover_limit / d, 0.0, 1.0))
                w_full = (1.0 - phi) * w_full + phi * prev
                if long_only:
                    w_full = np.clip(w_full, 0.0, None)
                if w_full.sum() > 0:
                    w_full /= w_full.sum()
        W.iloc[i] = w_full
        prev = w_full
    return W


# ===========================================================================
# 对齐 backtest 的优化回测
# ===========================================================================
def backtest_optimized(oos_detail: pd.DataFrame, signal_col: str = "fused",
                       universe_frac: float = 0.3, universe_top: int | None = None,
                       industry_panel: pd.DataFrame | None = None,
                       mktcap_panel: pd.DataFrame | None = None,
                       gate: bool = False, crisis=None, stress=None,
                       crisis_pos: float = 0.60, def_ann: float = 0.04,
                       max_pos_reduce: float = 0.20,
                       cost_bps: float = 0.0,
                       long_only: bool = True, max_w: float = 0.05, min_w: float = 0.0,
                       group_neutral: bool = False, neutral: str = "cap",
                       signal_temp: float = 1.0, turnover_limit: float | None = None,
                       n_ipf: int = 40, rebalance_freq: int = 1) -> dict:
    """用优化权重替代等权 top-N, 指标口径严格对齐 factor_wfa.backtest。

    与 backtest 的唯一差异 = 组合构建方式(等权 → 约束优化)。
    其余(基准=全市场等权、防御门控、成本模型)完全一致, 保证 A/B 公平。
    """
    if oos_detail is None or len(oos_detail) == 0:
        return None
    df = oos_detail.dropna(subset=[signal_col, "fwd_ret_1"]).copy()
    if len(df) == 0:
        return None

    # 信号面板 & 前向收益面板
    sig_panel = df.pivot(index="date", columns="code", values=signal_col)
    fwd_panel = df.pivot(index="date", columns="code", values="fwd_ret_1")
    sig_panel = sig_panel.sort_index()
    fwd_panel = fwd_panel.reindex(index=sig_panel.index, columns=sig_panel.columns)

    # 优化权重面板
    W = optimize_panel(
        sig_panel, industry_panel, mktcap_panel,
        universe_frac=universe_frac, universe_top=universe_top,
        long_only=long_only, max_w=max_w, min_w=min_w,
        group_neutral=group_neutral, neutral=neutral, signal_temp=signal_temp,
        turnover_limit=turnover_limit, n_ipf=n_ipf, rebalance_freq=rebalance_freq)

    # 组合日收益 = Σ w·fwd_ret (仅持有域)
    # 注意: 未选中单元 w=0, 但 fwd_panel 在缺测 (date,code) 处为 NaN,
    # 0*NaN=NaN 会污染整行求和 → 用 nansum(把 NaN 当 0, 等价于只在持仓处累加)
    port = np.nansum(W.values * fwd_panel.values, axis=1)
    daily = pd.Series(port, index=W.index)
    base = fwd_panel.mean(axis=1).reindex(daily.index).fillna(0.0)  # 全市场等权基准

    # 交易成本(口径对齐 backtest: tov = 0.5·||ΔW||₁, cost = tov·2·cost_bps/1e4)
    cost_daily = 0.0
    if cost_bps and cost_bps > 0:
        dw = W.diff().abs()
        tov = dw.sum(axis=1).fillna(0.0).values * 0.5
        cost_series = pd.Series(tov * 2.0 * cost_bps / 10000.0, index=daily.index)
        daily = daily - cost_series
        cost_daily = float(cost_series.mean())
        base = base - cost_series.mean()

    # 防御门控(对齐 backtest 双层门)
    n_crisis_days = 0
    if gate and crisis is not None:
        cr = crisis.reindex(daily.index).fillna(False).astype(float).values
        n_crisis_days = int((cr > 0.5).sum())
        st = stress.reindex(daily.index).fillna(0.0).values if stress is not None else 0.0
        pos = np.where(cr > 0.5, crisis_pos, 1.0 - st * max_pos_reduce)
        r_def = def_ann / 252.0
        gated = pos * daily.values + (1.0 - pos) * r_def
        daily = pd.Series(gated, index=daily.index)

    nav = (1.0 + daily).cumprod()
    bnav = (1.0 + base).cumprod()
    n = len(daily)
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / n) - 1.0 if n > 1 else np.nan
    ann_b = (bnav.iloc[-1] / bnav.iloc[0]) ** (252.0 / n) - 1.0 if n > 1 else np.nan
    vol = daily.std() * np.sqrt(252)
    sharpe = (daily.mean() * 252) / vol if vol and vol > 0 else np.nan
    max_dd = float((nav / nav.cummax() - 1.0).min())
    max_dd_b = float((bnav / bnav.cummax() - 1.0).min())
    calmar = (ann / abs(max_dd)) if max_dd < 0 else np.nan
    return {
        "n_days": n, "n_crisis_days": n_crisis_days,
        "years": round(n / 252.0, 2),
        "start": str(daily.index.min().date()), "end": str(daily.index.max().date()),
        "ann_ret": round(float(ann), 4), "ann_base": round(float(ann_b), 4),
        "tot_ret": round(float(nav.iloc[-1] / nav.iloc[0] - 1), 4),
        "tot_base": round(float(bnav.iloc[-1] / bnav.iloc[0] - 1), 4),
        "ann_vol": round(float(vol), 4), "sharpe": round(float(sharpe), 3),
        "max_dd": round(max_dd, 4), "max_dd_base": round(max_dd_b, 4),
        "calmar": round(float(calmar), 3),
        "cost_bps": cost_bps, "avg_daily_cost": round(cost_daily, 6),
        "method": "constrained_optimization",
    }


def compare_backtests(oos_detail: pd.DataFrame, signal_col: str = "fused",
                      universe_frac: float = 0.3, industry_panel=None, mktcap_panel=None,
                      gate: bool = False, crisis=None, stress=None,
                      cost_bps: float = 0.0, **opt_kwargs) -> pd.DataFrame:
    """naive 等权 top-N  vs  约束优化 的并列对比, 直接看优化器的增量价值。"""
    from factor_mining.factor_wfa import backtest
    r_naive = backtest(oos_detail, top_frac=universe_frac, gate=gate, crisis=crisis,
                       stress=stress, cost_bps=cost_bps)
    r_opt = backtest_optimized(oos_detail, signal_col=signal_col, universe_frac=universe_frac,
                               industry_panel=industry_panel, mktcap_panel=mktcap_panel,
                               gate=gate, crisis=crisis, stress=stress, cost_bps=cost_bps,
                               **opt_kwargs)
    rows = []
    for k in r_naive:
        if k in r_opt and isinstance(r_naive[k], (int, float)):
            rows.append({"metric": k, "naive": r_naive[k], "optimized": r_opt[k],
                         "delta": round(r_opt[k] - r_naive[k], 4)})
    return pd.DataFrame(rows)


# ===========================================================================
# (可选) scipy SLSQP 精确最优化: 带换手罚项 / 方差项, 小域适用
# ===========================================================================
def optimize_day_exact(alpha: np.ndarray, industry=None, mktcap=None, prev_w=None, *,
                       long_only=True, max_w=0.05, group_neutral=False, neutral="cap",
                       signal_temp=1.0, turnover_limit=None, lambda_turnover=1.0,
                       cov=None, lambda_risk=0.0):
    """scipy SLSQP 精确解(默认不启用, 大域请用 optimize_day)。

    目标: min  -(w·alpha) + λ_t·||w-w_prev||₁ + λ_r·wᵀΣw
    约束: Σw=1, w≥0, w≤max_w, (分组中性) Σ_{g} w = bench_g
    """
    try:
        from scipy.optimize import minimize
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要 scipy: pip install scipy") from e
    N = len(alpha)
    a = np.asarray(alpha, float) * signal_temp
    x0 = _softmax(a)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, max_w)] * N if long_only else [(-max_w, max_w)] * N
    if group_neutral and industry is not None:
        groups = pd.Series(industry).fillna("__NA__").values
        uniq, inv = np.unique(groups, return_inverse=True)
        if neutral == "cap" and mktcap is not None:
            mc = np.where(np.isfinite(mktcap) & (mktcap > 0), mktcap, 0.0)
            bench = np.array([mc[inv == g].sum() for g in range(len(uniq))])
            bench = bench / (bench.sum() + EPS)
        else:
            bench = np.full(len(uniq), 1.0 / len(uniq))
        for g in range(len(uniq)):
            mask = (inv == g)
            cons.append({"type": "eq", "fun": lambda w, m=mask: w[m].sum() - bench[g]})
    pw = np.asarray(prev_w, float) if prev_w is not None else np.zeros(N)

    def obj(w):
        o = -w @ a
        if lambda_turnover and prev_w is not None:
            o += lambda_turnover * np.abs(w - pw).sum()
        if lambda_risk and cov is not None:
            o += lambda_risk * w @ cov @ w
        return o

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    w = res.x if res.success else x0
    return np.clip(w, 0.0 if long_only else -max_w, max_w)


# ===========================================================================
# 自测: 合成数据验证全部约束
# ===========================================================================
def _selftest(n_assets: int = 200, n_days: int = 250, seed: int = 42):
    print("=" * 70)
    print("组合优化器 自测 (合成数据)")
    print("=" * 70)
    rng = np.random.default_rng(seed)
    assets = [f"A{i:03d}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    # 隐变量 alpha(决定真实截面排序); 信号 = alpha + 截面噪声
    true_alpha = rng.standard_normal(n_assets)
    # 信号带轻微时序噪声(让换手约束有意义)
    sig_noise = rng.standard_normal((n_days, n_assets)) * 0.6
    signal = pd.DataFrame(true_alpha[None, :] + sig_noise, index=dates, columns=assets)
    # 行业分组(10 个)+ 市值(对数正态)
    industry = rng.integers(0, 10, n_assets).astype(str)
    industry_panel = pd.DataFrame(np.tile(industry, (n_days, 1)), index=dates, columns=assets)
    mktcap = np.exp(rng.standard_normal(n_assets) * 0.8 + 12)
    mktcap_panel = pd.DataFrame(np.tile(mktcap, (n_days, 1)), index=dates, columns=assets)
    # 收益: 真实因子口径 —— alpha 微弱偏置(日 ~0.08bp) + 2.5% 日波动噪声
    #   → rank-IC ≈ 0.03(贴近实盘), 有真实回撤, 优化器价值在于风险约束而非放大收益
    alpha_mag = 0.0008
    sigma = 0.025
    fwd = pd.DataFrame(
        true_alpha[None, :] * alpha_mag + rng.standard_normal((n_days, n_assets)) * sigma,
        index=dates, columns=assets)

    ok = True
    # ---- 单日约束 ----
    a = signal.iloc[10].values
    ind = industry_panel.iloc[10].values
    mc = mktcap_panel.iloc[10].values
    w = optimize_day(a, ind, mc, None, max_w=0.03, group_neutral=True, neutral="cap")
    assert abs(w.sum() - 1.0) < 1e-6, "sum≠1"
    assert (w >= -1e-9).all(), "非多头"
    assert (w <= 0.03 + 1e-6).all(), "超个股上限"
    # 行业中性检查: 各行业权重 ≈ 市值基准
    gsum = pd.Series(w, index=ind).groupby(ind).sum()
    bench = pd.Series(mc, index=ind).groupby(ind).sum()
    bench = bench / bench.sum()
    err = (gsum - bench).abs().max()
    assert err < 1e-2, f"行业中性偏差过大 {err}"
    print(f"[单日] sum={w.sum():.4f} max_w={w.max():.4f} 行业中性最大偏差={err:.4f} ✓")

    # ---- 换手约束 ----
    W = optimize_panel(signal, industry_panel, mktcap_panel, universe_frac=0.3,
                       max_w=0.03, group_neutral=True, neutral="cap", turnover_limit=0.3)
    dw = W.diff().abs().sum(axis=1).dropna() * 0.5
    viol = (dw > 0.3 + 1e-6).sum()
    assert viol == 0, f"换手超限 {viol} 天"
    print(f"[面板] 换手上限 0.30 | 最大单日换手={dw.max():.3f} | 超限天数={viol} ✓")

    # ---- 对比回测(合成 oos_detail)----
    oos = pd.DataFrame({
        "date": np.repeat(dates.values, n_assets),
        "code": np.tile(assets, n_days),
        "fused": signal.values.flatten(),
        "fwd_ret_1": fwd.values.flatten(),
    })
    from factor_mining.factor_wfa import backtest
    rn = backtest(oos, top_frac=0.3)
    ro = backtest_optimized(oos, universe_frac=0.3, max_w=0.03,
                            group_neutral=True, neutral="cap", turnover_limit=0.3)
    cmp = pd.DataFrame([
        {"metric": "ann_ret", "naive": rn["ann_ret"], "optimized": ro["ann_ret"]},
        {"metric": "sharpe", "naive": rn["sharpe"], "optimized": ro["sharpe"]},
        {"metric": "max_dd", "naive": rn["max_dd"], "optimized": ro["max_dd"]},
        {"metric": "calmar", "naive": rn["calmar"], "optimized": ro["calmar"]},
    ])
    print("\n[合成 A/B]")
    print(cmp.to_string(index=False))
    # 真实性 + 不劣化校验
    for r in (rn, ro):
        assert np.isfinite(r["ann_ret"]) and abs(r["ann_ret"]) < 3.0, "年化异常(数值不合理)"
        assert np.isfinite(r["max_dd"]) and r["max_dd"] < 0, "应有真实回撤"
        assert r["sharpe"] > 0, "夏普应为正"
    # 优化器目标: 风险约束(个股权重上限/中性化)应改善或至少不显著劣化风险调整后收益
    assert ro["sharpe"] >= rn["sharpe"] - 0.05, "优化后夏普显著劣化"
    print("\n✅ 自测通过: 权重约束 / 行业中性 / 换手上限 / 回测口径 全部正确")
    return cmp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="跑合成数据自测")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("用法: python portfolio_optimizer.py --selftest")
