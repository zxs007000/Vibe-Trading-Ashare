"""Market-state classification engine — the decision layer above logic chains.

Classifies the current A-share market into one of 8 micro-states using
CSI-300 price, moving averages, volatility, and MACD.  Each state maps to
a recommended set of logic chains via ``CHAIN_STATE_PREFERENCE``.

Architecture
------------
* Fetch CSI-300 daily data (akshare, with fallback to cached JSON).
* Compute 8-state rules on the last 60 bars.
* Output: current state, confidence, state duration, recommended chains.
* Historical mode: replay state transitions over any date range.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 8 market states — rules ordered by priority (first match wins)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StateRule:
    """Deterministic rule for one market state."""
    name: str
    label_zh: str
    priority: int          # lower = checked first
    description: str
    # Quantitative threshold
    ma60_above_ma250: Optional[bool] = None   # MA60 vs MA250 position
    price_above_ma60: Optional[bool] = None    # price vs MA60
    price_vs_ma250_pct: Optional[float] = None  # price relative to MA250 (%)
    vol20_range: Optional[tuple[float, float]] = None  # (low, high) annualised
    macd_trend_days: Optional[tuple[int, int]] = None  # consecutive MACD>0 days
    ret20_range: Optional[tuple[float, float]] = None   # 20d return (%)
    ret5_range: Optional[tuple[float, float]] = None    # 5d return (%)


STATE_RULES: list[StateRule] = [
    # priority 1: 单边上涨 — strongest trend first
    StateRule(
        name="strong_bull", label_zh="单边上涨",
        priority=1,
        description="MA60>MA250, 价格在MA60上方, 20日涨>3%, MACD持续为正15天+",
        ma60_above_ma250=True, price_above_ma60=True,
        ret20_range=(3.0, 100.0), macd_trend_days=(15, 999),
    ),
    # priority 2: 单边下跌 — strongest bear
    StateRule(
        name="strong_bear", label_zh="单边下跌",
        priority=2,
        description="MA60<MA250, 价格在MA60下方, 20日跌>3%, MACD持续为负15天+",
        ma60_above_ma250=False, price_above_ma60=False,
        ret20_range=(-100.0, -3.0), macd_trend_days=(15, 999),
    ),
    # priority 3: 冲高回落 (MA60>MA250 but price recently broke below MA60)
    StateRule(
        name="pullback", label_zh="冲高回落",
        priority=3,
        description="MA60>MA250 但价格跌破MA60, 且5日前尚在MA60上方",
        ma60_above_ma250=True, price_above_ma60=False,
        ret5_range=(-100.0, 0.0),
    ),
    # priority 4: 反弹 (MA60<MA250 but price recently broke above MA60)
    StateRule(
        name="bounce", label_zh="反弹",
        priority=4,
        description="MA60<MA250 但价格上穿MA60, 短期修复行情",
        ma60_above_ma250=False, price_above_ma60=True,
        ret5_range=(0.0, 100.0),
    ),
    # priority 5: 震荡上行 — trend exists but not strong
    StateRule(
        name="grind_up", label_zh="震荡上行",
        priority=5,
        description="MA60>MA250, 价格在MA60上方, 但趋势不够强(<3%/20d)",
        ma60_above_ma250=True, price_above_ma60=True,
        ret20_range=(0.0, 3.0),
    ),
    # priority 6: 震荡下行
    StateRule(
        name="grind_down", label_zh="震荡下行",
        priority=6,
        description="MA60<MA250, 价格在MA60下方, 但跌幅不够大(>-3%/20d)",
        ma60_above_ma250=False, price_above_ma60=False,
        ret20_range=(-3.0, 0.0),
    ),
    # priority 7: 宽幅震荡 — high vol, no clear direction
    StateRule(
        name="wide_range", label_zh="宽幅震荡",
        priority=7,
        description="价格在MA250±5%内, 20日波动>20%, 无明显趋势",
        price_vs_ma250_pct=(-5.0, 5.0), vol20_range=(20.0, 999.0),
    ),
    # priority 8: 窄幅盘整 — catch-all
    StateRule(
        name="tight_range", label_zh="窄幅盘整",
        priority=8,
        description="价格在MA250±3%内, 20日波动<15%, 缩量等待方向",
        price_vs_ma250_pct=(-3.0, 3.0), vol20_range=(0.0, 15.0),
    ),
]


def _check_rule(rule: StateRule, row: dict) -> bool:
    """Test a single StateRule against one row of computed features."""
    if rule.ma60_above_ma250 is not None:
        if row["ma60_above_ma250"] != rule.ma60_above_ma250:
            return False
    if rule.price_above_ma60 is not None:
        if row["price_above_ma60"] != rule.price_above_ma60:
            return False
    if rule.price_vs_ma250_pct is not None:
        lo, hi = rule.price_vs_ma250_pct
        if not (lo <= row["price_vs_ma250_pct"] <= hi):
            return False
    if rule.vol20_range is not None:
        lo, hi = rule.vol20_range
        if not (lo <= row["vol20_annual_pct"] <= hi):
            return False
    if rule.macd_trend_days is not None:
        lo, hi = rule.macd_trend_days
        if not (lo <= row["macd_consec_days"] <= hi):
            return False
    if rule.ret20_range is not None:
        lo, hi = rule.ret20_range
        if not (lo <= row["ret20_pct"] <= hi):
            return False
    if rule.ret5_range is not None:
        lo, hi = rule.ret5_range
        if not (lo <= row["ret5_pct"] <= hi):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Chain → preferred market states
# ══════════════════════════════════════════════════════════════════════════════

# Each chain declares which states it thrives in and which it should avoid.
# This is derived from the narrative: "what breaks this chain?"

CHAIN_STATE_PREFERENCE: dict[str, dict[str, Any]] = {
    "value_qlowvol": {
        "preferred": ["strong_bull", "grind_up", "tight_range"],
        "avoid":      ["strong_bear", "wide_range"],
        "note":       "低波防御在单边下跌和宽幅震荡中失效(流动性枯竭或波动率聚集)",
    },
    "value_momentum": {
        "preferred": ["strong_bull", "grind_up", "bounce"],
        "avoid":      ["strong_bear", "tight_range", "wide_range"],
        "note":       "动量需要趋势,盘整和宽幅震荡中动量IC衰减",
    },
    "value_stable": {
        "preferred": ["grind_up", "tight_range", "strong_bull"],
        "avoid":      ["strong_bear", "wide_range"],
        "note":       "防御底仓逻辑,仅在稳定或温和上涨中适用",
    },
    "quality_momentum": {
        "preferred": ["strong_bull", "grind_up", "bounce"],
        "avoid":      ["strong_bear", "tight_range", "pullback"],
        "note":       "冲高回落中质量股易被错杀",
    },
    "reversal_momentum": {
        "preferred": ["bounce", "grind_down", "wide_range"],
        "avoid":      ["strong_bull", "strong_bear"],
        "note":       "强趋势中反转信号被压制,适合转折/震荡环境",
    },
    "vol_reversal": {
        "preferred": ["wide_range", "grind_down", "bounce"],
        "avoid":      ["strong_bull", "strong_bear", "tight_range"],
        "note":       "需要足够波动才能回归,低波盘整和强趋势都不适用",
    },
    "liq_momentum": {
        "preferred": ["bounce", "strong_bull", "grind_up"],
        "avoid":      ["tight_range", "grind_down", "pullback"],
        "note":       "放量需要资金驱动,缩量环境量能信号噪音化",
    },
    # ↓ micro_reversal downgraded — no L3 order-book data
    "micro_reversal": {
        "preferred": [],
        "avoid":      ["strong_bull", "strong_bear", "grind_up", "grind_down",
                        "bounce", "pullback", "wide_range", "tight_range"],
        "note":       "已降级:缺少L3逐笔数据,cord/klen均为日频K线形态代理,非真正微观结构alpha。待L3数据接入后激活。",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Feature computation
# ══════════════════════════════════════════════════════════════════════════════

def _was_above_ma60_5d_ago(close: pd.Series, features: pd.DataFrame) -> bool:
    """Check if price was above MA60 five trading days ago (for pullback detection)."""
    if len(features) < 6:
        return False
    # Look back 5 rows in the feature frame
    row_5d = features.iloc[-6]
    return bool(row_5d["price_above_ma60"])


def compute_features(close: pd.Series, volume: pd.Series | None = None,
                     window: int = 250) -> pd.DataFrame:
    """Compute all features needed for state classification.

    Parameters
    ----------
    close : pd.Series (datetime index)
        CSI-300 daily closing prices.
    volume : pd.Series | None
        Optional volume for vol-weighted metrics.
    window : int
        Lookback window for MA/VOL computation (default 250 ≈ 1 year).

    Returns
    -------
    pd.DataFrame with columns: close, ma60, ma250, ret5_pct, ret20_pct,
    vol20_annual_pct, macd_dif, macd_signal, macd_consec_days,
    ma60_above_ma250, price_above_ma60, price_vs_ma250_pct.
    """
    df = pd.DataFrame({"close": close}, index=close.index)
    df["ma60"] = close.rolling(60, min_periods=20).mean()
    df["ma250"] = close.rolling(250, min_periods=60).mean()

    # Returns
    df["ret5_pct"] = close.pct_change(5).fillna(0.0) * 100
    df["ret20_pct"] = close.pct_change(20).fillna(0.0) * 100

    # Volatility (annualised %)
    df["vol20_annual_pct"] = (
        close.pct_change().rolling(20).std().fillna(0.0) * np.sqrt(252) * 100
    )

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_dif"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal
    df["macd_above_zero"] = macd >= 0

    # Consecutive MACD direction days
    consec = []
    cnt = 0
    for v in df["macd_above_zero"]:
        cnt = cnt + 1 if v else 0
        consec.append(cnt if v else -cnt)
    df["macd_consec_days"] = consec

    # Boolean features
    df["ma60_above_ma250"] = df["ma60"] >= df["ma250"]
    df["price_above_ma60"] = close >= df["ma60"]
    df["price_above_ma60_5d_ago"] = df["price_above_ma60"].shift(5).fillna(False)
    df["price_vs_ma250_pct"] = ((close - df["ma250"]) / df["ma250"].replace(0, 1)) * 100

    return df.tail(window)


# ══════════════════════════════════════════════════════════════════════════════
# State classification
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StateResult:
    """Output of a single market-state classification."""
    state: str
    label_zh: str
    confidence: float           # 0-1, how well it matches the rule
    since_date: str             # ISO date when this state started
    duration_days: int          # how long this state has lasted
    recommended_chains: list[dict]  # [{chain_id, name_zh, match_score}]
    feature_snapshot: dict      # key feature values at classification point


def classify(close: pd.Series, volume: pd.Series | None = None) -> StateResult:
    """Classify the current market state from CSI-300 data.

    Parameters
    ----------
    close : pd.Series
        CSI-300 daily close.  Must have at least 250 rows.
    volume : pd.Series | None

    Returns
    -------
    StateResult
    """
    features = compute_features(close, volume)
    last = features.iloc[-1].to_dict()

    # Try rules in priority order
    for rule in sorted(STATE_RULES, key=lambda r: r.priority):
        if _check_rule(rule, last):
            # Special: pullback must confirm price was above MA60 5 days ago
            if rule.name == "pullback":
                if not last.get("price_above_ma60_5d_ago", False):
                    continue  # not a true pullback, try next rule

            # Special: bounce must confirm price was below MA60 5 days ago
            if rule.name == "bounce":
                if last.get("price_above_ma60_5d_ago", True):
                    continue  # price was already above MA60, not a bounce
            # Compute confidence: how well each sub-condition is met
            matches = 0
            total = 0
            if rule.ma60_above_ma250 is not None:
                total += 1
                if last["ma60_above_ma250"] == rule.ma60_above_ma250:
                    matches += 1
            if rule.price_above_ma60 is not None:
                total += 1
                if last["price_above_ma60"] == rule.price_above_ma60:
                    matches += 1
            if rule.ret20_range is not None:
                total += 1
                lo, hi = rule.ret20_range
                v = last["ret20_pct"]
                matches += (1.0 - abs(v - (lo + hi) / 2) / max(abs(hi - lo) / 2, 0.01))
                matches = max(0, min(matches, 1))
            if rule.ret5_range is not None:
                total += 1
            if rule.macd_trend_days is not None:
                total += 1
            if rule.vol20_range is not None:
                total += 1
            if rule.price_vs_ma250_pct is not None:
                total += 1

            confidence = round(matches / max(total, 1), 3) if total > 0 else 0.5

            # Duration: how long since state change
            duration = 1
            for i in range(len(features) - 2, max(0, len(features) - 60), -1):
                prev_row = features.iloc[i].to_dict()
                if not _check_rule(rule, prev_row):
                    break
                duration += 1
            since_date = str(features.index[-duration].date()) if duration > 0 else str(
                features.index[-1].date()
            )

            # Recommended chains
            recommended = _recommend_chains(rule.name)

            return StateResult(
                state=rule.name,
                label_zh=rule.label_zh,
                confidence=confidence,
                since_date=since_date,
                duration_days=duration,
                recommended_chains=recommended,
                feature_snapshot={
                    "close": round(last["close"], 2),
                    "ma60": round(last["ma60"], 2),
                    "ma250": round(last["ma250"], 2),
                    "ret20_pct": round(last["ret20_pct"], 2),
                    "ret5_pct": round(last["ret5_pct"], 2),
                    "vol20_annual_pct": round(last["vol20_annual_pct"], 1),
                    "macd_hist": round(last["macd_hist"], 4),
                    "macd_consec_days": int(last["macd_consec_days"]),
                },
            )

    # Fallback (should never happen with tight_range as catch-all)
    return StateResult(
        state="tight_range", label_zh="窄幅盘整",
        confidence=0.3, since_date=str(features.index[-1].date()),
        duration_days=1, recommended_chains=[], feature_snapshot={},
    )


def _recommend_chains(state_name: str) -> list[dict]:
    """Rank chains by preference for this state."""
    scored = []
    for chain_id, pref in CHAIN_STATE_PREFERENCE.items():
        if state_name in pref["preferred"]:
            scored.append({"chain_id": chain_id, "name_zh": _chain_name(chain_id),
                           "match": "preferred", "score": 1.0})
        elif state_name in pref["avoid"]:
            scored.append({"chain_id": chain_id, "name_zh": _chain_name(chain_id),
                           "match": "avoid", "score": 0.0})
        else:
            scored.append({"chain_id": chain_id, "name_zh": _chain_name(chain_id),
                           "match": "neutral", "score": 0.5})
    scored.sort(key=lambda x: -x["score"])
    return scored


def _chain_name(chain_id: str) -> str:
    names = {
        "value_qlowvol":    "价值质量低波",
        "value_momentum":   "价值动量",
        "value_stable":     "价值稳定",
        "quality_momentum": "质量动量",
        "reversal_momentum":"反转接力",
        "vol_reversal":     "波动回归",
        "liq_momentum":     "放量动量",
        "micro_reversal":   "微观反转(已降级)",
    }
    return names.get(chain_id, chain_id)


# ══════════════════════════════════════════════════════════════════════════════
# Historical replay
# ══════════════════════════════════════════════════════════════════════════════

def state_history(close: pd.Series, min_duration: int = 5) -> pd.DataFrame:
    """Replay state transitions over the full history.

    Parameters
    ----------
    close : pd.Series
    min_duration : int
        Merge states lasting fewer than min_duration bars into the
        preceding state (noise reduction).

    Returns
    -------
    pd.DataFrame with columns: date, state, label_zh, confidence
    """
    features = compute_features(close)
    results = []
    for i in range(len(features)):
        row = features.iloc[i].to_dict()
        for rule in sorted(STATE_RULES, key=lambda r: r.priority):
            if _check_rule(rule, row):
                results.append({
                    "date": features.index[i],
                    "state": rule.name,
                    "label_zh": rule.label_zh,
                })
                break
        else:
            results.append({
                "date": features.index[i],
                "state": "tight_range",
                "label_zh": "窄幅盘整",
            })

    df = pd.DataFrame(results).set_index("date")

    # Merge short-duration states
    if min_duration > 1 and len(df) > min_duration:
        state_runs = (df["state"] != df["state"].shift()).cumsum()
        for run_id in state_runs.unique():
            mask = state_runs == run_id
            if mask.sum() < min_duration:
                prev_state = df["state"].shift().loc[mask.idxmax()]
                df.loc[mask, "state"] = prev_state
                prev_label = df["label_zh"].shift().loc[mask.idxmax()]
                df.loc[mask, "label_zh"] = prev_label

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Data fetching
# ══════════════════════════════════════════════════════════════════════════════

_CACHE_FILE = Path(__file__).resolve().parent / "_csi300_cache.parquet"


def fetch_csi300(force_refresh: bool = False) -> pd.Series:
    """Fetch CSI-300 daily close from akshare, with local parquet cache.

    Parameters
    ----------
    force_refresh : bool
        If True, bypass cache.

    Returns
    -------
    pd.Series with datetime index and name='close'.
    """
    if not force_refresh and _CACHE_FILE.exists():
        cache_age = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(
            _CACHE_FILE.stat().st_mtime))
        if cache_age < timedelta(hours=6):
            df = pd.read_parquet(_CACHE_FILE)
            if not df.empty and "close" in df.columns:
                return df["close"]

    try:
        import akshare as ak

        raw = ak.stock_zh_index_daily(symbol="sh000300")
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date").sort_index()
        raw["close"].to_frame().to_parquet(_CACHE_FILE)
        logger.info("CSI-300 data refreshed from akshare, %d rows", len(raw))
        return raw["close"]
    except Exception as exc:
        logger.warning("akshare failed (%s), trying cache fallback", exc)
        if _CACHE_FILE.exists():
            df = pd.read_parquet(_CACHE_FILE)
            return df["close"]
        raise RuntimeError("No CSI-300 data available — akshare failed and no cache") from exc


# ══════════════════════════════════════════════════════════════════════════════
# Quick API
# ══════════════════════════════════════════════════════════════════════════════

def current_state() -> StateResult:
    """One-liner: fetch data + classify current state."""
    close = fetch_csi300()
    return classify(close)


# ══════════════════════════════════════════════════════════════════════════════
# __main__ — demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 72)
    print("  MARKET STATE ENGINE — CSI-300 real-time classification")
    print("=" * 72)

    close = fetch_csi300()
    print(f"\n  Data: {close.index[0].date()} ~ {close.index[-1].date()} ({len(close)} rows)")

    # ── Current state ───────────────────────────────────────────────────
    result = classify(close)
    print(f"\n  ▶ CURRENT STATE: {result.label_zh} ({result.state})")
    print(f"    Confidence: {result.confidence:.0%}")
    print(f"    Since:      {result.since_date} ({result.duration_days} days)")
    print(f"    Snapshot:   close={result.feature_snapshot['close']:.0f}  "
          f"MA60={result.feature_snapshot['ma60']:.0f}  "
          f"MA250={result.feature_snapshot['ma250']:.0f}")
    print(f"    Vol20: {result.feature_snapshot['vol20_annual_pct']:.1f}%  "
          f"Ret20: {result.feature_snapshot['ret20_pct']:+.1f}%  "
          f"Ret5: {result.feature_snapshot['ret5_pct']:+.1f}%")

    # ── Recommended chains ──────────────────────────────────────────────
    preferred = [c for c in result.recommended_chains if c["match"] == "preferred"]
    neutral  = [c for c in result.recommended_chains if c["match"] == "neutral"]
    avoided  = [c for c in result.recommended_chains if c["match"] == "avoid"]

    print(f"\n  ▶ RECOMMENDED CHAINS")
    if preferred:
        for c in preferred:
            print(f"    ✅ {c['name_zh']:<16} ({c['chain_id']})")
    if neutral:
        for c in neutral:
            print(f"    ➖ {c['name_zh']:<16} ({c['chain_id']}) — neutral")
    if avoided:
        print(f"\n  ▶ AVOID IN THIS STATE")
        for c in avoided:
            print(f"    ❌ {c['name_zh']:<16} ({c['chain_id']})")

    # ── Historical states ───────────────────────────────────────────────
    print(f"\n  ▶ HISTORICAL STATE TRANSITIONS (last 12 months)")
    history = state_history(close, min_duration=5)
    recent = history.loc[history.index >= (history.index[-1] - pd.DateOffset(months=12))]
    transitions = recent[recent["state"] != recent["state"].shift()]
    for _, row in transitions.iterrows():
        ts = str(row.name.date())
        print(f"    {ts}  →  {row['label_zh']}")

    # ── State distribution ──────────────────────────────────────────────
    print(f"\n  ▶ STATE DISTRIBUTION (full history)")
    dist = history["label_zh"].value_counts()
    for state_label, count in dist.items():
        pct = count / len(history) * 100
        bar = "█" * int(pct / 2)
        print(f"    {state_label:<8} {pct:>5.1f}%  {bar}")

    print()
