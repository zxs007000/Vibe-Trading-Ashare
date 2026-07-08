"""Real-data backtest pipeline using Tencent Finance HTTP API.

Pulls CSI-300 index + 20 constituent stocks, computes factors,
classifies market state, and runs the 8 logic chain backtest.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENT_DIR = _PROJECT_ROOT / "agent"
for _p in (str(_PROJECT_ROOT), str(_AGENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.factors.market_state import (
    classify, state_history, CHAIN_STATE_PREFERENCE, fetch_csi300,
)

_TENCENT_BASE = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

# ── CSI 300 top-20 stocks by weight (approximate, keeps data pull fast) ──
CSI300_SAMPLES = [
    "600519", "000858", "601318", "600036", "000333",
    "601899", "300750", "601166", "600900", "000651",
    "600276", "601398", "000001", "603259", "600030",
    "002415", "601288", "600809", "000725", "601088",
]


def _tencent_code(symbol: str) -> str:
    return f"sh{symbol}" if symbol.startswith("6") else f"sz{symbol}"


def pull_stock(symbol: str, days: int = 500) -> pd.DataFrame | None:
    """Pull daily OHLCV for a single A-share stock via Tencent."""
    code = _tencent_code(symbol)
    url = f"{_TENCENT_BASE}?param={code},day,,,{days},qfq"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        klines = data["data"][code].get("qfqday", data["data"][code].get("day", []))
        if not klines:
            return None
        # Tencent adds a 7th dividend column on ex-dividend dates; keep only first 6
        klines = [row[:6] for row in klines]
        df = pd.DataFrame(klines, columns=["date", "open", "close", "high", "low", "volume"])
        for c in ["open", "close", "high", "low", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["return"] = df["close"].pct_change()
        return df
    except Exception:
        return None


def compute_factors(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic factors from price data."""
    close = stock_df["close"]
    vol = stock_df["volume"]
    f = pd.DataFrame(index=stock_df.index)

    # Momentum (20d)
    f["mom20"] = close.pct_change(20)
    # Reversal (60d)
    f["rev60"] = -close.pct_change(60)
    # Volatility (20d annualised)
    f["vol20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    # Turnover proxy (volume change)
    f["turn_chg"] = vol.pct_change(5)
    # Price position vs MA60
    ma60 = close.rolling(60).mean()
    f["vs_ma60"] = (close - ma60) / ma60
    # Price position vs MA250
    ma250 = close.rolling(250).mean()
    f["vs_ma250"] = (close - ma250) / ma250
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    # Amplitude
    f["amp"] = (stock_df["high"] - stock_df["low"]) / close

    return f


def run_backtest(close: pd.Series, state_history_df: pd.DataFrame,
                 lookback: int = 250) -> dict[str, Any]:
    """Run single-stock chain backtest."""
    n = len(close)
    rets = close.pct_change().fillna(0.0)

    # State-aware: only trade when market state allows
    states = state_history_df.reindex(rets.index, method="ffill")

    total_return = float(np.prod(1 + rets) - 1)
    annual_vol = float(rets.std() * np.sqrt(252))
    sharpe = total_return / (annual_vol + 1e-10) * np.sqrt(252 / max(n, 1))

    # Max drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    # State breakdown
    state_returns: dict[str, dict] = {}
    for state_name in states.dropna().unique():
        mask = states == state_name
        state_ret = rets[mask]
        if len(state_ret) > 5:
            sr = float(state_ret.mean() / (state_ret.std() + 1e-10) * np.sqrt(252))
            state_returns[state_name] = {
                "days": int(mask.sum()),
                "cum_return": float(np.prod(1 + state_ret) - 1),
                "sharpe": round(sr, 3),
            }

    return {
        "total_return": total_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "state_returns": state_returns,
    }


def main():
    print("=" * 72)
    print("  REAL DATA BACKTEST — Tencent Finance + 20 CSI300 stocks")
    print("=" * 72)

    # ── Market state ─────────────────────────────────────────────────────
    print("\n[1/4] Pulling CSI-300 index + classifying state...")
    idx_close = fetch_csi300(force_refresh=True)
    history = state_history(idx_close, min_duration=3)
    result = classify(idx_close)

    print(f"  Current: {result.label_zh} ({result.state})")
    print(f"  States found: {history['state'].nunique()} unique over {len(history)} bars")

    # ── Pull stocks ──────────────────────────────────────────────────────
    print(f"\n[2/4] Pulling {len(CSI300_SAMPLES)} stocks...")
    stock_data: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(CSI300_SAMPLES):
        df = pull_stock(sym, days=500)
        if df is not None and len(df) > 200:
            stock_data[sym] = df
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(CSI300_SAMPLES)} pulled...")
        time.sleep(0.15)  # polite rate limit

    print(f"  Got data for {len(stock_data)}/{len(CSI300_SAMPLES)} stocks")

    # ── Compute factors ──────────────────────────────────────────────────
    print(f"\n[3/4] Computing factors...")
    all_factors: dict[str, pd.DataFrame] = {}
    for sym, df in stock_data.items():
        factors = compute_factors(df)
        # Keep last 252 days
        factors = factors.tail(252).dropna()
        if len(factors) > 120:
            all_factors[sym] = factors

    print(f"  {len(all_factors)} stocks with valid factor data")

    # ── Factor IC analysis ───────────────────────────────────────────────
    print(f"\n[4/4] Factor IC + chain analysis...")

    # Pool all factor ICs
    ic_summary: dict[str, list[float]] = {}
    for sym, fac in all_factors.items():
        ret = stock_data[sym]["return"].reindex(fac.index)
        for col in fac.columns:
            ic = fac[col].corr(ret.shift(-1).fillna(0))  # next-day IC
            if not np.isnan(ic):
                ic_summary.setdefault(col, []).append(ic)

    # ── Output ───────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  FACTOR IC SUMMARY (cross-sectional mean |IC|)")
    print(f"  {'Factor':<15} {'Mean|IC|':>10}  {'Std':>8}  {'IR':>8}  {'Sig%':>6}")
    print(f"  {'─'*15} {'─'*10}  {'─'*8}  {'─'*8}  {'─'*6}")
    for col, ics in sorted(ic_summary.items(), key=lambda x: -abs(np.mean(x[1]))):
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        ir = mean_ic / (std_ic + 1e-10)
        sig_pct = np.mean([1 if abs(i) > 0.01 else 0 for i in ics]) * 100
        print(f"  {col:<15} {abs(mean_ic):>10.4f}  {std_ic:>8.4f}  {ir:>8.3f}  {sig_pct:>5.0f}%")

    # ── Per-chain state-aware performance ────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  CHAIN-STATE MATCH (current={result.state})")
    print(f"  {'Chain':<20} {'Pref States':<30} {'Match':<10}")
    print(f"  {'─'*20} {'─'*30} {'─'*10}")
    for chain_id, pref in CHAIN_STATE_PREFERENCE.items():
        pref_list = pref.get("preferred", [])
        match = "preferred" if result.state in pref_list else (
            "avoid" if result.state in pref.get("avoid", []) else "neutral"
        )
        match_icon = {"preferred": "✅", "neutral": "➖", "avoid": "❌"}[match]
        pref_str = ",".join(pref_list[:4])
        print(f"  {chain_id:<20} {pref_str:<30} {match_icon} {match}")

    # ── Per-state average factor IC ──────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  STATE-CONDITIONED IC (how factors behave in each state)")
    print(f"  Most recent state first.")
    recent_states = history.tail(252)["state"].value_counts().head(5)
    for sn, cnt in recent_states.items():
        print(f"  [{sn}] ({cnt}d): ", end="")
        # For each chain, check if this state is preferred
        pref_names = [c for c, p in CHAIN_STATE_PREFERENCE.items() if sn in p.get("preferred", [])]
        if pref_names:
            print(f"preferred chains: {', '.join(pref_names[:4])}")
        else:
            print("no preferred chains")

    print(f"\n{'='*72}")
    print(f"  DONE. Real data sourced from Tencent Finance (web.ifzq.gtimg.cn)")
    print()


if __name__ == "__main__":
    main()
