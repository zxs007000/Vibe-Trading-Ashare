"""Logic chain life assessment engine.

Evaluates 8 logic chains (each = 3-stage pipeline of factor themes) across
3 dimensions:

  1. **Node health** (weight 0.30): alive factor ratio per theme node.
  2. **Crossover effectiveness** (weight 0.35): Granger-like IC significance
     between adjacent pipeline stages.
  3. **Chain composite alpha** (weight 0.35): rolling IC & IR of the
     equal-weight chain combination.

Each chain is classified as: thriving / decaying / broken / dead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.factors.factor_analysis_core import compute_ic_series
from src.factors.rolling_ic import assess_lifecycle

logger = logging.getLogger(__name__)

# =============================================================================
# 9 logic chains — 8 pure-factor chains + 1 index-guarded strategy chain.
# Pipeline length = 2–4 stages (not rigidly 3).
# Every chain is grounded in real data; themes with 0 factors are avoided.
# "index_guard_value" is the first market-context-aware chain — it adds an
# index-level gate (MA250) before factor selection.
# =============================================================================

LOGIC_CHAINS: dict[str, dict[str, Any]] = {
    # ── 短期 (1-5 天)：交易型,捕捉瞬时 alpha ──
    "micro_reversal": {
        "name_zh": "微观反转",
        "horizon": "short",
        "horizon_days": "1-5d",
        "pipeline": ["microstructure", "reversal"],
        "desc": "微观结构异动 → 短期价格修正",
    },
    "liq_momentum": {
        "name_zh": "放量动量",
        "horizon": "short",
        "horizon_days": "1-5d",
        "pipeline": ["liquidity", "momentum", "volume"],
        "desc": "流动性异动 → 趋势启动 → 放量确认",
    },
    "vol_reversal": {
        "name_zh": "波动回归",
        "horizon": "short",
        "horizon_days": "1-5d",
        "pipeline": ["volatility", "reversal"],
        "desc": "高波动 → 均值回归",
    },

    # ── 中期 (5-60 天)：趋势型,跟随主升浪 ──
    "value_momentum": {
        "name_zh": "价值动量",
        "horizon": "medium",
        "horizon_days": "5-60d",
        "pipeline": ["value", "momentum", "volume"],
        "desc": "价值发现 → 趋势跟随 → 量价确认",
    },
    "quality_momentum": {
        "name_zh": "质量动量",
        "horizon": "medium",
        "horizon_days": "5-60d",
        "pipeline": ["quality", "momentum", "volume"],
        "desc": "质量筛选 → 动量入场 → 成交量退场信号",
    },
    "reversal_momentum": {
        "name_zh": "反转接力",
        "horizon": "medium",
        "horizon_days": "5-60d",
        "pipeline": ["reversal", "momentum", "volume"],
        "desc": "反转触底 → 动量启动 → 量能确认",
    },

    # ── 长期 (60 天+)：配置型,稳健底仓 ──
    "value_qlowvol": {
        "name_zh": "价值质量低波",
        "horizon": "long",
        "horizon_days": "60d+",
        "pipeline": ["value", "quality", "volatility"],
        "desc": "深度价值 → 质量过滤 → 低波防御",
    },
    "value_stable": {
        "name_zh": "价值稳定",
        "horizon": "long",
        "horizon_days": "60d+",
        "pipeline": ["value", "volatility", "liquidity"],
        "desc": "价值底盘 → 波动过滤 → 流动性校验",
    },
}

# =============================================================================
# Chain → preferred market states (mirrors market_state.py CHAIN_STATE_PREFERENCE)
# =============================================================================

CHAIN_STATE_PREFERENCE: dict[str, dict[str, Any]] = {
    "micro_reversal": {
        "preferred": [], "avoid": ["strong_bull", "strong_bear", "grind_up",
            "grind_down", "bounce", "pullback", "wide_range", "tight_range"],
        "note": "已降级:L3数据缺失,cord/klen均为日频K线代理",
    },
    "liq_momentum": {
        "preferred": ["bounce", "strong_bull", "grind_up"],
        "avoid": ["tight_range", "grind_down", "pullback"],
    },
    "vol_reversal": {
        "preferred": ["wide_range", "grind_down", "bounce"],
        "avoid": ["strong_bull", "strong_bear", "tight_range"],
    },
    "value_momentum": {
        "preferred": ["strong_bull", "grind_up", "bounce"],
        "avoid": ["strong_bear", "tight_range", "wide_range"],
    },
    "quality_momentum": {
        "preferred": ["strong_bull", "grind_up", "bounce"],
        "avoid": ["strong_bear", "tight_range", "pullback"],
    },
    "reversal_momentum": {
        "preferred": ["bounce", "grind_down", "wide_range"],
        "avoid": ["strong_bull", "strong_bear"],
    },
    "value_qlowvol": {
        "preferred": ["strong_bull", "grind_up", "tight_range"],
        "avoid": ["strong_bear", "wide_range"],
    },
    "value_stable": {
        "preferred": ["grind_up", "tight_range", "strong_bull"],
        "avoid": ["strong_bear", "wide_range"],
    },
}


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class CrossResult:
    """Crossover IC significance test result for one adjacent theme pair."""

    pair: tuple[str, str]
    """The (theme_a, theme_b) pair tested."""

    valid: bool
    """True if crossover IC t-test is significant (|t| > 2, p < 0.05)."""

    ic_mean: float
    """Mean of the crossover IC series."""

    t_stat: float
    """t-statistic from one-sample t-test (H0: IC == 0)."""

    p_value: float
    """p-value of the t-test."""


@dataclass
class CompositeResult:
    """Rolling-IC summary for the chain's equal-weight factor combination."""

    chain_ir: float
    """Mean rolling IR of the chain composite."""

    half_life: float
    """Days from 12-month peak IC to half-peak (inf if never decays)."""

    ic_mean: float
    """Mean IC of the chain composite across the full period."""


@dataclass
class ChainHealth:
    """Complete three-dimension assessment for one logic chain."""

    chain_id: str
    """Logic chain key (e.g. 'value_momentum')."""

    name_zh: str
    """Chinese display name."""

    pipeline: list[str]
    """Ordered theme pipeline."""

    node_health: dict[str, float]
    """Per-theme alive factor ratio."""

    avg_node_health: float
    """Mean node health across the pipeline (dimension 1, weight 0.30)."""

    cross_results: list[CrossResult]
    """Crossover IC results for each adjacent pair."""

    all_cross_valid: bool
    """True when every adjacent pair passes the crossover t-test."""

    composite: CompositeResult
    """Chain-level rolling IR and half-life (dimension 3, weight 0.35)."""

    classification: str = field(default="")
    """Final classification: thriving / decaying / broken / dead."""

    score: float = field(default=0.0)
    """Weighted composite score for debugging / ranking."""


# =============================================================================
# Dimension 1: node health
# =============================================================================


def _node_health(
    theme: str,
    registry: Any,
    panel: dict[str, pd.DataFrame],
    return_df: pd.DataFrame,
    window: int = 60,
) -> float:
    """Compute alive factor ratio for a single theme node.

    Iterates all factors registered under *theme*, computes each via
    ``registry.compute()``, and calls ``assess_lifecycle()`` (from
    ``rolling_ic.py``) to check liveness.

    Args:
        theme: Theme label (e.g. "momentum", "value").
        registry: ``Registry`` instance with ``.list(theme=...)`` and ``.compute()``.
        panel: OHLCV panel dict fed to ``registry.compute()``.
        return_df: Forward-return matrix (index=date, columns=code).
        window: Rolling window for lifecycle assessment.

    Returns:
        Ratio of alive factors: count(alive) / max(total, 1).
        Returns 0.0 when the theme has zero registered factors
        (e.g. "growth", "sentiment", "leverage" may be empty).
    """
    alpha_ids = registry.list(theme=theme)
    if not alpha_ids:
        return 0.0

    alive_count = 0
    for alpha_id in alpha_ids:
        try:
            factor_df = registry.compute(alpha_id, panel)
        except Exception:
            logger.debug("_node_health: skip %s (compute failed)", alpha_id)
            continue
        try:
            lifecycle = assess_lifecycle(alpha_id, factor_df, return_df, window=window)
        except Exception:
            logger.debug("_node_health: skip %s (lifecycle failed)", alpha_id)
            continue
        if lifecycle.status == "alive":
            alive_count += 1

    return alive_count / max(len(alpha_ids), 1)


# =============================================================================
# Dimension 2: crossover IC effectiveness
# =============================================================================

# Minimum number of IC observations required for a valid t-test.
_MIN_IC_OBS_CROSSOVER = 20
# Significance thresholds.
_CROSS_T_STAT_THRESHOLD = 2.0
_CROSS_P_VALUE_THRESHOLD = 0.05


def _crossover_ic(
    theme_a: str,
    theme_b: str,  # currently reserved for pipeline context; IC is vs returns
    registry: Any,
    panel: dict[str, pd.DataFrame],
    return_df: pd.DataFrame,
    window: int = 60,
) -> CrossResult:
    """Test whether theme A's top alive factor can Granger-predict returns.

    Uses Spearman correlation (IC) between theme A's factor values and
    forward returns (``return_df``).  If the mean IC is significantly
    different from zero (|t| > 2 and p < 0.05), the crossover is valid.

    When multiple alive factors exist in *theme_a*, the one with the
    highest absolute mean IC is selected.

    Args:
        theme_a: Source theme (pipeline stage N).
        theme_b: Target theme (pipeline stage N+1) — kept for extensibility.
        registry: ``Registry`` instance.
        panel: OHLCV panel dict.
        return_df: Forward-return matrix.
        window: Rolling window for lifecycle filtering.

    Returns:
        CrossResult with validity flag and test statistics.
    """
    alpha_ids = registry.list(theme=theme_a)

    # -- Find the best alive factor in theme A -------------------------------
    best_alpha_id: str | None = None
    best_factor_df: pd.DataFrame | None = None
    best_abs_ic_mean: float = -1.0

    for alpha_id in alpha_ids:
        try:
            factor_df = registry.compute(alpha_id, panel)
        except Exception:
            continue
        try:
            lifecycle = assess_lifecycle(alpha_id, factor_df, return_df, window=window)
        except Exception:
            continue
        if lifecycle.status != "alive":
            continue

        ic = compute_ic_series(factor_df, return_df)
        if ic.empty or len(ic) < _MIN_IC_OBS_CROSSOVER:
            continue
        abs_mean = abs(float(ic.mean()))
        if abs_mean > best_abs_ic_mean:
            best_abs_ic_mean = abs_mean
            best_alpha_id = alpha_id
            best_factor_df = factor_df

    if best_alpha_id is None or best_factor_df is None:
        return CrossResult(
            pair=(theme_a, theme_b),
            valid=False,
            ic_mean=0.0,
            t_stat=0.0,
            p_value=1.0,
        )

    # -- t-test on IC sequence -----------------------------------------------
    ic = compute_ic_series(best_factor_df, return_df)
    if ic.empty or len(ic) < _MIN_IC_OBS_CROSSOVER:
        return CrossResult(
            pair=(theme_a, theme_b),
            valid=False,
            ic_mean=float(ic.mean()) if not ic.empty else 0.0,
            t_stat=0.0,
            p_value=1.0,
        )

    ic_clean = ic.dropna()
    if len(ic_clean) < _MIN_IC_OBS_CROSSOVER:
        return CrossResult(
            pair=(theme_a, theme_b),
            valid=False,
            ic_mean=float(ic.mean()),
            t_stat=0.0,
            p_value=1.0,
        )

    t_stat, p_value = stats.ttest_1samp(ic_clean.values, 0.0)
    valid = abs(t_stat) > _CROSS_T_STAT_THRESHOLD and float(p_value) < _CROSS_P_VALUE_THRESHOLD

    return CrossResult(
        pair=(theme_a, theme_b),
        valid=valid,
        ic_mean=float(ic.mean()),
        t_stat=float(t_stat),
        p_value=float(p_value),
    )


# =============================================================================
# Dimension 3: chain composite alpha
# =============================================================================

_IR_WINDOW = 60  # rolling window (trading days) for chain-level IR
_IR_MIN_PERIODS = 20  # minimum observations per rolling window
_HALF_LIFE_LOOKBACK = 252  # 12-month lookback for peak → half decay


def _half_life_days(ic_series: pd.Series, lookback: int = _HALF_LIFE_LOOKBACK) -> float:
    """Days from rolling-IC peak (within *lookback*) to the first date
    where IC falls below ``peak / 2``.

    Returns ``float('inf')`` when IC never drops to half, or when fewer
    than 2 observations are available.
    """
    if len(ic_series) < 2:
        return float("inf")
    recent = ic_series.iloc[-lookback:] if len(ic_series) > lookback else ic_series
    if recent.empty or recent.isna().all():
        return float("inf")

    peak_val = float(recent.max())
    peak_idx = recent.idxmax()
    if peak_val <= 0 or pd.isna(peak_idx):
        return float("inf")

    half_val = peak_val / 2.0
    post_peak = recent[recent.index >= peak_idx]
    below_half = post_peak[post_peak <= half_val]
    if below_half.empty:
        return float("inf")

    half_idx = below_half.index[0]
    return float((half_idx - peak_idx).days)


def _chain_composite_alpha(
    chain: dict[str, Any],
    registry: Any,
    panel: dict[str, pd.DataFrame],
    return_df: pd.DataFrame,
    window: int = 60,
) -> CompositeResult:
    """Build an equal-weight composite of each pipeline node's top-1 alive
    factor, then compute rolling IC/IR and half-life.

    Args:
        chain: One entry from ``LOGIC_CHAINS`` (has key "pipeline").
        registry: ``Registry`` instance.
        panel: OHLCV panel dict.
        return_df: Forward-return matrix.
        window: Rolling window for IC/IR (default 60).

    Returns:
        CompositeResult with chain IR, half-life, and mean IC.
    """
    pipeline: list[str] = chain["pipeline"]
    selected: dict[str, pd.DataFrame] = {}  # theme → factor_df

    for theme in pipeline:
        alpha_ids = registry.list(theme=theme)
        best_alpha_id: str | None = None
        best_factor_df: pd.DataFrame | None = None
        best_abs_ic_mean: float = -1.0

        for alpha_id in alpha_ids:
            try:
                factor_df = registry.compute(alpha_id, panel)
            except Exception:
                continue
            try:
                lifecycle = assess_lifecycle(alpha_id, factor_df, return_df, window=window)
            except Exception:
                continue
            if lifecycle.status != "alive":
                continue

            ic = compute_ic_series(factor_df, return_df)
            if ic.empty:
                continue
            abs_mean = abs(float(ic.mean()))
            if abs_mean > best_abs_ic_mean:
                best_abs_ic_mean = abs_mean
                best_alpha_id = alpha_id
                best_factor_df = factor_df

        if best_alpha_id is not None and best_factor_df is not None:
            selected[theme] = best_factor_df

    if not selected:
        return CompositeResult(chain_ir=0.0, half_life=float("inf"), ic_mean=0.0)

    # -- Equal-weight blend --------------------------------------------------
    # Align all selected factor DataFrames to common index/columns, then
    # average (rank-normalise first for robustness).
    aligned: list[pd.DataFrame] = []
    for factor_df in selected.values():
        # Rank-normalise across stocks each day (0–1 scale).
        ranked = factor_df.rank(axis=1, method="average", pct=True)
        aligned.append(ranked)

    # Reindex union — only dates/codes present in ALL factors survive.
    common = aligned[0]
    for other in aligned[1:]:
        common_dates = common.index.intersection(other.index)
        common_codes = common.columns.intersection(other.columns)
        if len(common_dates) == 0 or len(common_codes) == 0:
            return CompositeResult(chain_ir=0.0, half_life=float("inf"), ic_mean=0.0)
        common = common.loc[common_dates, common_codes]
        other = other.loc[common_dates, common_codes]

    composite_df = sum(
        df.loc[common.index, common.columns] for df in aligned
    ) / len(aligned)

    # -- Rolling IC ----------------------------------------------------------
    ic = compute_ic_series(composite_df, return_df)
    if ic.empty or len(ic) < _IR_MIN_PERIODS:
        return CompositeResult(chain_ir=0.0, half_life=float("inf"), ic_mean=0.0)

    ic_mean = float(ic.mean())

    # Rolling IR
    roll_mean = ic.rolling(window=_IR_WINDOW, min_periods=_IR_MIN_PERIODS).mean()
    roll_std = ic.rolling(window=_IR_WINDOW, min_periods=_IR_MIN_PERIODS).std()
    roll_ir = (roll_mean / roll_std.where(roll_std > 0)).replace([np.inf, -np.inf], np.nan)

    chain_ir = float(roll_ir.mean()) if not roll_ir.empty else 0.0

    # Half-life from peak IC
    roll_mean_clean = roll_mean.dropna()
    half_life = _half_life_days(roll_mean_clean) if not roll_mean_clean.empty else float("inf")

    return CompositeResult(
        chain_ir=chain_ir,
        half_life=half_life,
        ic_mean=ic_mean,
    )


# =============================================================================
# Classification
# =============================================================================


def classify(health: ChainHealth) -> str:
    """Map a ``ChainHealth`` to a four-state classification.

    Decision matrix (evaluated top-down, first match wins)::

        thriving : node_health >= 0.6 AND all_cross_valid AND chain_IR >= 0.25
        decaying : node_health in [0.3, 0.6) OR chain_IR in [0.1, 0.25)
                   (conduction still valid)
        broken   : any adjacent crossover broken (p > 0.05)
                   OR any single node_health == 0
        dead     : node_health < 0.3 OR chain_IR < 0.1

    Args:
        health: Fully-populated ``ChainHealth`` instance.

    Returns:
        One of ``"thriving"``, ``"decaying"``, ``"broken"``, ``"dead"``.
    """
    nh = health.avg_node_health
    ir = health.composite.chain_ir
    any_zero_node = any(v == 0.0 for v in health.node_health.values())

    # thriving
    if nh >= 0.6 and health.all_cross_valid and ir >= 0.25:
        return "thriving"

    # broken
    if not health.all_cross_valid or any_zero_node:
        return "broken"

    # dead (before decaying, since these are hard thresholds)
    if nh < 0.3 or ir < 0.1:
        return "dead"

    # decaying
    if (0.3 <= nh < 0.6) or (0.1 <= ir < 0.25):
        return "decaying"

    # Fallback — shouldn't normally be reached.
    return "dead"


# =============================================================================
# Top-level assessment
# =============================================================================

# Dimension weights (must sum to 1.0).
_W_NODE = 0.30
_W_CROSS = 0.35
_W_COMPOSITE = 0.35


def _cross_score(valid: bool) -> float:
    """Map crossover validity to a 0–1 score."""
    return 1.0 if valid else 0.0


def _ir_score(chain_ir: float) -> float:
    """Saturate chain IR into [0, 1] (IR >= 0.5 → 1.0)."""
    return min(max(chain_ir / 0.5, 0.0), 1.0)


def assess_all_chains(
    registry: Any,
    panel: dict[str, pd.DataFrame],
    return_df: pd.DataFrame,
    window: int = 60,
) -> list[ChainHealth]:
    """Run the full three-dimension evaluation for every logic chain.

    Args:
        registry: ``Registry`` instance with ``.list()`` and ``.compute()``.
        panel: OHLCV panel dict.
        return_df: Forward-return matrix.
        window: Rolling window size (trading days) for lifecycle and IC.

    Returns:
        List of ``ChainHealth`` results, one per entry in ``LOGIC_CHAINS``,
        sorted by weighted score descending.
    """
    results: list[ChainHealth] = []

    for chain_id, chain_def in LOGIC_CHAINS.items():
        pipeline: list[str] = chain_def["pipeline"]

        # --- Dimension 1: node health ---------------------------------------
        node_health_map: dict[str, float] = {}
        for theme in pipeline:
            node_health_map[theme] = _node_health(theme, registry, panel, return_df, window=window)
        avg_nh = sum(node_health_map.values()) / max(len(pipeline), 1)

        # --- Dimension 2: crossover effectiveness ---------------------------
        cross_results: list[CrossResult] = []
        for i in range(len(pipeline) - 1):
            theme_a = pipeline[i]
            theme_b = pipeline[i + 1]
            cr = _crossover_ic(theme_a, theme_b, registry, panel, return_df, window=window)
            cross_results.append(cr)
        all_valid = all(cr.valid for cr in cross_results) if cross_results else True

        # --- Dimension 3: chain composite alpha -----------------------------
        composite = _chain_composite_alpha(chain_def, registry, panel, return_df, window=window)

        # --- Classification -------------------------------------------------
        health = ChainHealth(
            chain_id=chain_id,
            name_zh=chain_def["name_zh"],
            pipeline=pipeline,
            node_health=node_health_map,
            avg_node_health=avg_nh,
            cross_results=cross_results,
            all_cross_valid=all_valid,
            composite=composite,
        )
        health.classification = classify(health)

        # Weighted composite score (for ranking only).
        cross_ratio = (
            sum(1.0 for cr in cross_results if cr.valid) / max(len(cross_results), 1)
            if cross_results
            else 1.0
        )
        health.score = (
            _W_NODE * avg_nh
            + _W_CROSS * cross_ratio
            + _W_COMPOSITE * _ir_score(composite.chain_ir)
        )

        results.append(health)

    results.sort(key=lambda h: -h.score)
    return results


# =============================================================================
# Pretty-print / reporting helpers
# =============================================================================

_STATUS_ICONS: dict[str, str] = {
    "thriving": "THR",
    "decaying": "DEC",
    "broken": "BRK",
    "dead": "DED",
}


def print_chain_report(results: list[ChainHealth]) -> None:
    """Human-readable summary table for all chain results."""
    header = (
        f"{'Chain':<22s} {'Status':>5s}  {'Score':>6s}  "
        f"{'NodeH':>6s}  {'Cross':>5s}  {'IR':>7s}  {'HalfLife':>10s}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for h in results:
        status_tag = _STATUS_ICONS.get(h.classification, "???")
        cross_tag = "OK" if h.all_cross_valid else "!!"
        hl_str = f"{h.composite.half_life:.0f}d" if np.isfinite(h.composite.half_life) else "inf"
        print(
            f"{h.chain_id:<22s} {status_tag:>5s}  {h.score:>6.3f}  "
            f"{h.avg_node_health:>6.3f}  {cross_tag:>5s}  "
            f"{h.composite.chain_ir:>7.3f}  {hl_str:>10s}"
        )
    print(sep)


# =============================================================================
# Test / demo section
# =============================================================================

if __name__ == "__main__":
    import sys

    # ------------------------------------------------------------------
    # 1. Build synthetic panel + return data
    # ------------------------------------------------------------------
    np.random.seed(42)
    N_DAYS = 500
    N_STOCKS = 80
    dates = pd.bdate_range("2021-01-04", periods=N_DAYS)
    codes = [f"STK_{i:03d}" for i in range(N_STOCKS)]

    # Random-walk prices
    log_rets = pd.DataFrame(
        np.random.randn(N_DAYS, N_STOCKS) * 0.015,
        index=dates,
        columns=codes,
    )
    prices = (1 + log_rets).cumprod() * 100.0

    panel: dict[str, pd.DataFrame] = {
        "close": prices,
        "open": prices * 0.998,
        "high": prices * 1.012,
        "low": prices * 0.988,
        "volume": pd.DataFrame(
            np.random.rand(N_DAYS, N_STOCKS) * 1_000_000,
            index=dates,
            columns=codes,
        ),
    }

    # Forward 5-day returns
    return_df = prices.pct_change(5).shift(-5)

    # ------------------------------------------------------------------
    # 2. Build a synthetic Registry
    # ------------------------------------------------------------------
    # Map each theme to a small set of "alpha IDs" and synthetic factor values.
    # Themes with data → realistic alive factors.
    # Themes without data (growth, sentiment, leverage) → empty list.

    THEME_ALPHAS: dict[str, list[str]] = {
        "momentum": ["momentum_rsi", "momentum_macd"],
        "reversal": ["reversal_bb", "reversal_ma"],
        "volume": ["volume_vwap"],
        "volatility": ["volatility_atr", "volatility_bbw"],
        "quality": ["quality_roe"],
        "value": ["value_pe", "value_pb"],
        "liquidity": ["liquidity_to"],
        "microstructure": ["micro_spread"],
        "sentiment": [],  # empty → node_health == 0
        "growth": [],  # empty → node_health == 0
        "leverage": [],  # empty → node_health == 0
    }

    # Generate synthetic factor DataFrames with controlled IC properties.
    # "alive" factors get a moderate signal component; "dead" ones are pure noise.

    def _make_factor(ic_target: float = 0.04) -> pd.DataFrame:
        """Create a factor df with mean |IC| ≈ ic_target."""
        noise = pd.DataFrame(
            np.random.randn(N_DAYS, N_STOCKS) * 0.5,
            index=dates,
            columns=codes,
        )
        # Inject signal by adding a fraction of the forward return
        signal = return_df.fillna(0.0) * np.random.uniform(0.5, 1.5)
        return noise + signal * (ic_target / 0.02)  # rough scaling

    FACTOR_CACHE: dict[str, pd.DataFrame] = {}

    def _get_factor(alpha_id: str) -> pd.DataFrame:
        if alpha_id not in FACTOR_CACHE:
            # Diversify IC targets to get a mix of alive / decaying / dead.
            tag = hash(alpha_id) % 100
            if tag < 40:
                target = np.random.uniform(0.03, 0.08)  # alive
            elif tag < 70:
                target = np.random.uniform(0.008, 0.025)  # decaying
            else:
                target = np.random.uniform(-0.005, 0.008)  # dead
            FACTOR_CACHE[alpha_id] = _make_factor(ic_target=target)
        return FACTOR_CACHE[alpha_id]

    class _MockAlpha:
        __slots__ = ("id", "zoo", "module_path", "meta")

        def __init__(self, aid: str, zoo: str, themes: list[str]):
            self.id = aid
            self.zoo = zoo
            self.module_path = f"src.factors.zoo.{zoo}.{aid}"
            self.meta = {"theme": themes}

    # Figure out which zoo each alpha comes from (deterministic).
    _ZOO_LIST = ["academic", "alpha101", "gtja191", "qlib158"]

    class _MockRegistry:
        """Minimal stub that implements .list() and .compute()."""

        def list(
            self,
            zoo: str | None = None,
            theme: str | None = None,
            universe: str | None = None,
        ) -> list[str]:
            if theme is not None and theme in THEME_ALPHAS:
                ids = list(THEME_ALPHAS[theme])
            else:
                ids = sorted({aid for aids in THEME_ALPHAS.values() for aid in aids})
            if zoo is not None:
                # filter by deterministic zoo assignment
                ids = [aid for aid in ids if _ZOO_LIST[hash(aid) % len(_ZOO_LIST)] == zoo]
            return sorted(ids)

        def compute(self, alpha_id: str, panel_: Any) -> pd.DataFrame:
            if alpha_id not in THEME_ALPHAS.get(self._theme_of(alpha_id), []):
                raise KeyError(alpha_id)
            return _get_factor(alpha_id)

        def get(self, alpha_id: str):
            theme = self._theme_of(alpha_id)
            return _MockAlpha(alpha_id, _ZOO_LIST[hash(alpha_id) % len(_ZOO_LIST)], [theme])

        @staticmethod
        def _theme_of(alpha_id: str) -> str:
            for t, ids in THEME_ALPHAS.items():
                if alpha_id in ids:
                    return t
            return "momentum"

    mock_registry = _MockRegistry()

    # ------------------------------------------------------------------
    # 3. Run assessment & print
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Logic Chain Life Assessment — 逻辑链寿命三维评估引擎")
    print("=" * 80)
    print(f"  dates : {dates[0].date()} → {dates[-1].date()}  ({N_DAYS} bars)")
    print(f"  stocks: {N_STOCKS}  |  chains: {len(LOGIC_CHAINS)}")
    print("=" * 80)

    results = assess_all_chains(mock_registry, panel, return_df, window=60)
    print_chain_report(results)

    # ------------------------------------------------------------------
    # 4. Detailed breakdown for each chain
    # ------------------------------------------------------------------
    for h in results:
        print(f"\n{'─' * 72}")
        print(f"  [{h.chain_id}]  {h.name_zh}  —  {h.classification.upper()}")
        print(f"  pipeline: {' → '.join(h.pipeline)}")
        print(f"  node health:  {h.node_health}")
        print(f"  avg health:   {h.avg_node_health:.3f}  (weight {_W_NODE:.0%})")
        for cr in h.cross_results:
            status = "VALID" if cr.valid else "BROKEN"
            print(
                f"  cross {cr.pair[0]:>16s}→{cr.pair[1]:<16s}  "
                f"ic={cr.ic_mean:+.4f}  t={cr.t_stat:+.2f}  p={cr.p_value:.4f}  [{status}]"
            )
        print(
            f"  composite IR: {h.composite.chain_ir:.4f}  |  "
            f"mean IC: {h.composite.ic_mean:+.4f}  |  "
            f"half-life: {h.composite.half_life:.0f}d"
            if np.isfinite(h.composite.half_life)
            else f"  composite IR: {h.composite.chain_ir:.4f}  |  "
            f"mean IC: {h.composite.ic_mean:+.4f}  |  half-life: inf"
        )
        print(f"  classification: {h.classification}  |  score: {h.score:.3f}")

    # ------------------------------------------------------------------
    # 5. Summary counts
    # ------------------------------------------------------------------
    counts: dict[str, int] = {}
    for h in results:
        counts[h.classification] = counts.get(h.classification, 0) + 1
    print(f"\n{'=' * 80}")
    print("  Summary — by classification:")
    for state in ["thriving", "decaying", "broken", "dead"]:
        print(f"    {state:<12s}: {counts.get(state, 0)} chains")
    print("=" * 80 + "\n")

    sys.exit(0)
