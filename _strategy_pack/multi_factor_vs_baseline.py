"""Real-data multi-factor backtest: N stocks × M chains → baseline comparison.

Tests the hypothesis: can multi-factor / multi-chain diversification close the
gap between single-chain Sharpe (~0.49) and baseline Sharpe (~0.95 / ~1.3 ex-2008)?
"""

from __future__ import annotations

import sys, time
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

from agent.backtest.metrics import calc_metrics

TENCENT = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
INITIAL = 1_000_000

# 50 CSI300 stocks — broad sector coverage
STOCKS_50 = [
    "600519","000858","601318","600036","000333","601899","300750","601166",
    "600900","000651","600276","601398","000001","603259","600030","002415",
    "601288","600809","000725","601088","601012","002714","000002","600887",
    "601857","600028","688981","002475","300059","601688","000063","300124",
    "600585","600309","600436","002594","601225","603288","002304","000568",
    "601995","600570","002352","300015","601066","600104","601628","000776",
    "300498","002230",
]


def pull(code: str, days: int = 1500) -> pd.DataFrame | None:
    """Pull OHLCV from Tencent, return df with date-indexed close + return."""
    try:
        url = f"{TENCENT}?param={code},day,,,{days},qfq"
        r = requests.get(url, timeout=10)
        d = r.json()["data"][code]
        k = d.get("qfqday", d.get("day", []))
        k = [x[:6] for x in k]
        df = pd.DataFrame(k, columns=["date","open","close","high","low","volume"])
        for c in ["open","close","high","low","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["return"] = df["close"].pct_change()
        return df
    except Exception:
        return None


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    f = pd.DataFrame(index=df.index)
    f["mom20"] = c.pct_change(20)
    f["rev60"] = -c.pct_change(60)
    f["vs_ma250"] = (c - c.rolling(250).mean()) / c.rolling(250).mean()
    f["vs_ma60"] = (c - c.rolling(60).mean()) / c.rolling(60).mean()
    f["vol20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    f["turn_chg"] = df["volume"].pct_change(5).fillna(0)
    f["amp"] = (df["high"] - df["low"]) / c
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    return f


def main():
    print("="*72)
    print("  MULTI-FACTOR vs BASELINE: 50 stocks × 8 factors")
    print("  (Real data from Tencent Finance, ~4yr daily)")
    print("="*72)

    # ── Pull all stocks ──────────────────────────────────────────────
    print("\n[1] Pulling 50 CSI-300 stocks...")
    stocks = {}
    for i, sym in enumerate(STOCKS_50):
        df = pull(f"sh{sym}" if sym.startswith("6") else f"sz{sym}", days=1500)
        if df is not None and len(df) > 250:
            stocks[sym] = df
        if (i+1) % 10 == 0:
            print(f"    {i+1}/50 done, {len(stocks)} valid")
        time.sleep(0.1)
    print(f"    Total: {len(stocks)} stocks with valid data")

    # ── Factor IC per stock ──────────────────────────────────────────
    print("\n[2] Computing factor ICs (next-day predictive)...")
    all_ics = {}
    for sym, df in stocks.items():
        fac = compute_factors(df).dropna()
        fut = df["return"].shift(-1)
        for col in fac.columns:
            ic = fac[col].corr(fut)
            if not np.isnan(ic):
                all_ics.setdefault(col, []).append(ic)

    print(f"    Factor cross-sectional |IC|:")
    factor_ic = {}
    for col, ics in sorted(all_ics.items(), key=lambda x: -abs(np.mean(x[1]))):
        mean_ic = abs(np.mean(ics))
        factor_ic[col] = mean_ic
        print(f"    {col:<15} |IC|={mean_ic:.4f}  (n={len(ics)})")

    # ── Multi-factor portfolio: daily cross-sectional long-short ───────
    print(f"\n[3] Building cross-sectional factor portfolio...")
    # Align all stocks to common dates, keep ~4yr window
    common = stocks[list(stocks.keys())[0]].index
    for df in stocks.values():
        common = common.intersection(df.index)
    common = common[-500:]

    # Precompute factor DataFrames for all stocks (once, not per day)
    print(f"    Precomputing factors for {len(stocks)} stocks × {len(common)} days...")
    precomputed: dict[str, pd.DataFrame] = {}
    for sym, df in stocks.items():
        precomputed[sym] = compute_factors(df).reindex(common)

    # For each factor, each day: rank 50 stocks, long top 10 / short bottom 10
    # Equal weight across 8 factors → combined daily return

    TOP_N = 10  # top 10 stocks per factor
    daily_rets_by_factor = {col: [] for col in factor_ic}
    dates_with_data = []

    for i, date in enumerate(common):
        if i == len(common) - 1:
            break
        next_date = common[i + 1]
        factor_rets = {}
        all_good = True

        for col in factor_ic:
            scores = {}
            for sym, df in stocks.items():
                fac = precomputed[sym]
                if col in fac.columns and date in fac.index:
                    v = fac[col].loc[date]
                    if not np.isnan(v):
                        scores[sym] = v

            if len(scores) < TOP_N * 2:
                all_good = False
                break

            # Sort by factor score (higher = better)
            ranked = sorted(scores.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999)
            ranked = [(s, v) for s, v in ranked if not np.isnan(v)]
            longs = [s for s, _ in ranked[-TOP_N:]]
            shorts = [s for s, _ in ranked[:TOP_N]]

            # Next-day return: average of longs minus average of shorts
            long_ret = np.mean([stocks[s]["return"].get(next_date, 0) for s in longs])
            short_ret = np.mean([stocks[s]["return"].get(next_date, 0) for s in shorts])
            factor_rets[col] = long_ret - short_ret

        if len(factor_rets) == len(factor_ic):
            for col, ret in factor_rets.items():
                daily_rets_by_factor[col].append(ret)
            dates_with_data.append(next_date)
        elif i % 50 == 0:
            pass  # skip rare incomplete days silently

    # Equal-weight across factors
    combined_ret = np.zeros(len(dates_with_data))
    for col in daily_rets_by_factor:
        arr = np.array(daily_rets_by_factor[col])
        if len(arr) > 0:
            combined_ret += arr / len(factor_ic)
    combined_ret = pd.Series(combined_ret, index=dates_with_data)
    equity = (1.0 + combined_ret).cumprod() * INITIAL
    m = calc_metrics(equity, trades=[], initial_cash=INITIAL, bars_per_year=252)

    # ── Baseline comparison ──────────────────────────────────────────
    print(f"\n[4] Results")
    n_stocks = len(stocks)
    n_factors = len(factor_ic)
    # Theoretical max Sharpe from IC diversification
    # IR ≈ sqrt(N_factors * N_stocks_eff) * mean_IC / vol
    mean_ic = np.mean(list(factor_ic.values()))
    theory_sharpe = mean_ic * np.sqrt(n_factors) * np.sqrt(min(n_stocks, 30)) / 0.25

    print(f"    Stocks used:    {n_stocks}")
    print(f"    Factors used:   {n_factors}")
    print(f"    Mean |IC|:      {mean_ic:.4f}")
    print(f"    Theory Sharpe:  {theory_sharpe:.3f} (IF factors fully orthogonal)")
    print(f"    Realized:       {m['sharpe']:.3f}")
    print(f"    ──────────────────────────────────────")
    print(f"    Baseline(20yr): Sharpe 0.95 / MaxDD -25%")
    print(f"    Baseline(ex-08): Sharpe ~1.3  / MaxDD <5%")

    gap = m["sharpe"] / 0.95
    if gap >= 1.0:
        verdict = "✅ 追平基线 (full period)"
    elif gap >= 0.7:
        verdict = f"⚠️ 接近基线({gap:.0%}),但差{0.95-m['sharpe']:.2f}"
    else:
        verdict = f"❌ 差距较大({gap:.0%}),需要更多正交因子或更多股票"

    print(f"\n    Verdict: {verdict}")
    print(f"    AnnRet: {m['annual_return']:.2%}  MaxDD: {m['max_drawdown']:.2%}  "
          f"Vol: {float(daily_port_ret.std()*np.sqrt(252)):.1%}")
    print()


if __name__ == "__main__":
    main()
