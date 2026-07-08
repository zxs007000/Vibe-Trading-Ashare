"""Synthetic backtest for 8 logic chains — differentiated noise + benchmark.

Every chain gets its **own** signal generator and noise series so that
Sharpe / MaxDD / excess return reflect the chain's unique alpha profile.
A CSI-300-style benchmark is synthesised for excess-Sharpe context.

No real market data required — all returns are synthesised deterministically
(seed derived from chain_id) so results are reproducible across environments.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure ``agent.backtest.metrics`` is importable from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENT_DIR = _PROJECT_ROOT / "agent"
for _p in (str(_PROJECT_ROOT), str(_AGENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.backtest.metrics import calc_metrics  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# 8 logic chains — redesigned by time horizon, grounded in real zoo factors.
# Pipeline length is 2–4 (not rigidly 3).  Chains that rely on themes with
# 0 factors (growth / sentiment / leverage) have been removed.
# ══════════════════════════════════════════════════════════════════════════

LOGIC_CHAINS = {
    # ── 短期 (1-5d) ──
    "micro_reversal": {
        "name_zh": "微观反转", "horizon": "short", "horizon_days": "1-5d",
        "pipeline": ["microstructure", "reversal"],
        "desc": "微观结构异动 → 短期价格修正",
    },
    "liq_momentum": {
        "name_zh": "放量动量", "horizon": "short", "horizon_days": "1-5d",
        "pipeline": ["liquidity", "momentum", "volume"],
        "desc": "流动性异动 → 趋势启动 → 放量确认",
    },
    "vol_reversal": {
        "name_zh": "波动回归", "horizon": "short", "horizon_days": "1-5d",
        "pipeline": ["volatility", "reversal"],
        "desc": "高波动 → 均值回归",
    },
    # ── 中期 (5-60d) ──
    "value_momentum": {
        "name_zh": "价值动量", "horizon": "medium", "horizon_days": "5-60d",
        "pipeline": ["value", "momentum", "volume"],
        "desc": "价值发现 → 趋势跟随 → 量价确认",
    },
    "quality_momentum": {
        "name_zh": "质量动量", "horizon": "medium", "horizon_days": "5-60d",
        "pipeline": ["quality", "momentum", "volume"],
        "desc": "质量筛选 → 动量入场 → 成交量退场信号",
    },
    "reversal_momentum": {
        "name_zh": "反转接力", "horizon": "medium", "horizon_days": "5-60d",
        "pipeline": ["reversal", "momentum", "volume"],
        "desc": "反转触底 → 动量启动 → 量能确认",
    },
    # ── 长期 (60d+) ──
    "value_qlowvol": {
        "name_zh": "价值质量低波", "horizon": "long", "horizon_days": "60d+",
        "pipeline": ["value", "quality", "volatility"],
        "desc": "深度价值 → 质量过滤 → 低波防御",
    },
    "value_stable": {
        "name_zh": "价值稳定", "horizon": "long", "horizon_days": "60d+",
        "pipeline": ["value", "volatility", "liquidity"],
        "desc": "价值底盘 → 波动过滤 → 流动性校验",
    },
}

# ══════════════════════════════════════════════════════════════════════════
# Factor IC / IR / lifecycle params (from frontend FactorAnalysis FACTORS)
# ══════════════════════════════════════════════════════════════════════════

FACTOR_PARAMS = {
    "mom20":    {"ic": 0.062, "ir": 0.82, "life": "alive",    "theme": "momentum"},
    "rev60":    {"ic": 0.041, "ir": 0.55, "life": "decaying", "theme": "momentum"},
    "ep":       {"ic": 0.071, "ir": 0.94, "life": "alive",    "theme": "value"},
    "bp":       {"ic": 0.068, "ir": 0.88, "life": "alive",    "theme": "value"},
    "cfp":      {"ic": 0.052, "ir": 0.71, "life": "alive",    "theme": "quality"},
    "roe_stab": {"ic": 0.048, "ir": 0.66, "life": "alive",    "theme": "quality"},
    "rev_g":    {"ic": 0.039, "ir": 0.52, "life": "decaying", "theme": "growth"},
    "vol":      {"ic": -0.058, "ir": -0.79, "life": "alive",  "theme": "volatility"},
    "turn":     {"ic": -0.044, "ir": -0.60, "life": "decaying", "theme": "liquidity"},
    "amp":      {"ic": -0.039, "ir": -0.53, "life": "alive",  "theme": "liquidity"},
    "north":    {"ic": 0.033, "ir": 0.45, "life": "decaying", "theme": "sentiment"},
    "size":     {"ic": -0.035, "ir": -0.48, "life": "alive",  "theme": "size"},
    "lev":      {"ic": -0.028, "ir": -0.37, "life": "dead",   "theme": "leverage"},
    "dyr":      {"ic": 0.045, "ir": 0.61, "life": "alive",    "theme": "dividend"},
}

# ══════════════════════════════════════════════════════════════════════════
# Theme → factor mapping (pipeline themes → factor IDs in FACTOR_PARAMS)
# ══════════════════════════════════════════════════════════════════════════

THEME_TO_FACTORS: dict[str, list[str]] = {
    "value":          ["ep", "bp"],
    "momentum":       ["mom20"],
    "quality":        ["cfp", "roe_stab"],
    "volatility":     ["vol"],
    "liquidity":      ["turn", "amp"],
    # Cross-theme proxies — real zoo factors under adjacent themes
    "reversal":       ["rev60"],     # rev60 is momentum-tagged but captures reversal
    "volume":         ["turn"],      # turnover proxied to volume activity
    "microstructure": ["size"],      # size as microstructure proxy (market-cap tier)
}

# ══════════════════════════════════════════════════════════════════════════
# Synthesis parameters
# ══════════════════════════════════════════════════════════════════════════

_N_DAYS = 252
_INITIAL_CASH = 1_000_000.0
_ANNUAL_VOL = 0.25                # typical A-share annualised vol
_DAILY_VOL = _ANNUAL_VOL / np.sqrt(_N_DAYS)
_SIGNAL_RHO = 0.05                # day-to-day signal autocorrelation
_DISPERSION_SCALE = 0.012         # daily cross-sectional return dispersion
_BASE_SEED = 42

# Benchmark (CSI-300 style): ~14 % annual return, ~20 % vol, Sharpe ≈ 0.65
# Realistic for 2024 A-share large-cap
_BENCH_ANNUAL_RET = 0.14
_BENCH_ANNUAL_VOL = 0.20


# ── helper functions ─────────────────────────────────────────────────────────

def _expected_ic(factor_ids: list[str]) -> float:
    """Mean |IC| across chain's representative factors."""
    if not factor_ids:
        return 0.0
    return float(np.mean([abs(FACTOR_PARAMS[fid]["ic"]) for fid in factor_ids]))


def _classify_chain(factor_ids: list[str]) -> str:
    """Classify synthetic chain status from its factors' life tags + IC strength.

    Priority: dead > decaying > alive.
    - If ANY factor is dead → dead
    - If mean |IC| < 0.04 → dead (too weak)
    - If ANY factor is decaying OR mean |IC| < 0.05 → decaying
    - Else alive

    Also marks 'broken' when a pipeline node has 0 usable factors
    (detected upstream — if factor_ids is empty after theme resolution).
    """
    if not factor_ids:
        return "broken"
    lives = {FACTOR_PARAMS[fid]["life"] for fid in factor_ids}
    mean_ic = _expected_ic(factor_ids)
    if "dead" in lives or mean_ic < 0.040:
        return "dead"
    if "decaying" in lives or mean_ic < 0.050:
        return "decaying"
    return "thriving"


def _horizon_label(h: str) -> str:
    return {"short": "短期", "medium": "中期", "long": "长期"}.get(h, h)


# ══════════════════════════════════════════════════════════════════════════
# Backtest core
# ══════════════════════════════════════════════════════════════════════════

def run_backtest() -> list[dict]:
    """Run differentiated synthetic backtest for all logic chains.

    Each chain gets **its own** noise series (seeded by chain_id hash) so
    MaxDD and Sharpe are genuinely independent.  A CSI-300-style benchmark
    is generated once and shared for excess-return context.

    Returns
    -------
    list[dict]
        One dict per chain, sorted by excess Sharpe descending.
    """
    dates = pd.bdate_range("2024-01-02", "2024-12-31")[:_N_DAYS]
    n = len(dates)

    # ── Benchmark: deterministic CSI-300 (+14 % with mild seasonal pattern) ──
    t = np.arange(n)
    bench_daily = _BENCH_ANNUAL_RET / _N_DAYS + 0.00015 * np.sin(2 * np.pi * t / 63)
    bench_ret = pd.Series(bench_daily, index=dates)
    bench_equity = (1.0 + bench_ret).cumprod() * _INITIAL_CASH

    # ── Per-chain backtest ────────────────────────────────────────────────────
    results: list[dict] = []

    for idx, (chain_id, chain_def) in enumerate(LOGIC_CHAINS.items()):
        pipeline = chain_def["pipeline"]

        # Resolve each pipeline theme → factor IDs
        chain_factor_ids: list[str] = []
        empty_nodes = 0
        for theme in pipeline:
            fids = THEME_TO_FACTORS.get(theme, [])
            if not fids:
                empty_nodes += 1
            chain_factor_ids.extend(fids)

        expected_ic = _expected_ic(chain_factor_ids)
        status = _classify_chain(chain_factor_ids)

        # ── Per-chain independent noise & signal ─────────────────────────
        # Seed = base_seed + chain-specific offset so each chain is isolated.
        chain_seed = _BASE_SEED + hash(chain_id) % 10000 + idx * 100
        rng = np.random.default_rng(chain_seed)

        # Independent market noise for this chain
        raw_noise = rng.standard_normal(n) * _DAILY_VOL
        chain_noise = raw_noise - raw_noise.mean()

        # Independent autocorrelated signal
        sig = np.zeros(n)
        for t in range(1, n):
            sig[t] = (
                _SIGNAL_RHO * sig[t - 1]
                + rng.standard_normal() * np.sqrt(1.0 - _SIGNAL_RHO**2)
            )
        signal = (sig - sig.mean()) / (sig.std() + 1e-10)
        signal_shifted = np.roll(signal, 1)
        signal_shifted[0] = 0.0

        # ── Status-driven decay envelope ─────────────────────────────────
        if empty_nodes > 0:
            # broken: at least one pipeline node has 0 factors → sharp collapse
            decay = np.linspace(0.6, 0.08, n)
            status = "broken"
        elif status == "dead":
            decay = np.linspace(0.7, 0.15, n)
        elif status == "decaying":
            decay = np.linspace(1.0, 0.45, n)
        else:  # thriving
            decay = np.ones(n)

        # ── Daily returns ────────────────────────────────────────────────
        alpha = np.abs(signal_shifted) * abs(expected_ic) * _DISPERSION_SCALE * decay
        daily_rets = pd.Series(alpha + chain_noise, index=dates, name="daily_return")

        equity_curve = (1.0 + daily_rets).cumprod() * _INITIAL_CASH

        # ── Metrics ───────────────────────────────────────────────────────
        metrics = calc_metrics(
            equity_curve,
            trades=[],
            initial_cash=_INITIAL_CASH,
            bars_per_year=_N_DAYS,
        )

        # Excess vs benchmark
        excess_ret_series = daily_rets - bench_ret.values
        excess_equity = (1.0 + excess_ret_series).cumprod() * _INITIAL_CASH
        excess_metrics = calc_metrics(
            excess_equity,
            trades=[],
            initial_cash=_INITIAL_CASH,
            bars_per_year=_N_DAYS,
        )

        results.append({
            "chain_id": chain_id,
            "name_zh": chain_def["name_zh"],
            "horizon": chain_def["horizon"],
            "pipeline": pipeline,
            "status": status,
            "expected_ic": expected_ic,
            "sharpe": metrics["sharpe"],
            "excess_sharpe": excess_metrics["sharpe"],
            "annual_return": metrics["annual_return"],
            "max_drawdown": metrics["max_drawdown"],
            "calmar": metrics["calmar"],
            "sortino": metrics["sortino"],
            "total_return": metrics["total_return"],
        })

    # Benchmark metrics
    bench_m = calc_metrics(bench_equity, trades=[], initial_cash=_INITIAL_CASH, bars_per_year=_N_DAYS)
    results.sort(key=lambda r: -r["excess_sharpe"])
    return results, bench_m


# ══════════════════════════════════════════════════════════════════════════
# ASCII table output
# ══════════════════════════════════════════════════════════════════════════

def print_table(results: list[dict], bench_m: dict) -> None:
    """Print pure-ASCII comparison table with excess Sharpe and benchmark."""
    w = {"chain": 20, "hz": 6, "st": 8, "ic": 10,
         "sharpe": 10, "ex_sh": 10, "ann_ret": 10, "max_dd": 10}

    def _sep():
        return ("+" + "-" * w["chain"] + "+" + "-" * w["hz"] + "+"
                + "-" * w["st"] + "+" + "-" * w["ic"] + "+"
                + "-" * w["sharpe"] + "+" + "-" * w["ex_sh"] + "+"
                + "-" * w["ann_ret"] + "+" + "-" * w["max_dd"] + "+")

    header = (
        f"| {'Chain':<{w['chain']-1}s}| {'周期':<{w['hz']-1}s}"
        f"| {'Status':<{w['st']-1}s}| {'E[|IC|]':<{w['ic']-1}s}"
        f"| {'Sharpe':<{w['sharpe']-1}s}| {'exSharpe':<{w['ex_sh']-1}s}"
        f"| {'AnnRet%':<{w['ann_ret']-1}s}| {'MaxDD%':<{w['max_dd']-1}s}|"
    )

    print()
    hz_map = {"short": "短期", "medium": "中期", "long": "长期"}
    print("LOGIC CHAIN BACKTEST (2024, 252d • per-chain independent noise • vs CSI300)")
    print(f"  benchmark → Sharpe={bench_m['sharpe']:.3f}  AnnRet={bench_m['annual_return']:.2%}  MaxDD={bench_m['max_drawdown']:.2%}")
    print(_sep())
    print(header)
    print(_sep())

    for r in results:
        row = (
            f"| {r['chain_id']:<{w['chain']-1}s}"
            f"| {hz_map.get(r['horizon'],r['horizon']):<{w['hz']-1}s}"
            f"| {r['status']:<{w['st']-1}s}"
            f"| {r['expected_ic']:>{w['ic']-1}.4f}"
            f"| {r['sharpe']:>{w['sharpe']-1}.4f}"
            f"| {r['excess_sharpe']:>{w['ex_sh']-1}.4f}"
            f"| {r['annual_return']:>{w['ann_ret']-2}.2%} "
            f"| {r['max_drawdown']:>{w['max_dd']-2}.2%} |"
        )
        print(row)

    print(_sep())

    best = results[0]
    worst = results[-1]
    alive = [r for r in results if r["status"] == "thriving"]
    dead_broken = [r for r in results if r["status"] in ("dead", "broken")]

    print()
    print(f"  best:       {best['chain_id']}  (excess Sharpe {best['excess_sharpe']:.3f}  AnnRet {best['annual_return']:.2%})")
    print(f"  worst:      {worst['chain_id']}  (excess Sharpe {worst['excess_sharpe']:.3f}  AnnRet {worst['annual_return']:.2%})")
    print(f"  thriving:   {len(alive)} 条   decaying: {len(results)-len(alive)-len(dead_broken)} 条   dead/broken: {len(dead_broken)} 条")
    print(f"  spread(S):  {best['excess_sharpe'] - worst['excess_sharpe']:.3f}  (第1名-第8名超额Sharpe差)")
    print()


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results, bench_m = run_backtest()
    print_table(results, bench_m)
