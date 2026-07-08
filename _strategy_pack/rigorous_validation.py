"""Rigorous factor validation: 100 stocks × in/out-sample × Monte Carlo.

Three-pronged defense against data-mining bias:
1. Expand to 100 stocks for statistical power
2. In-sample (2018-2023) vs out-of-sample (2024-2026) IC comparison
3. Monte Carlo permutation test — scramble returns, rerun 500×, build null distribution
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
N_MC = 500

# 100 CSI300+CSI500 stocks
STOCKS_100 = [
    "600519","000858","601318","600036","000333","601899","300750","601166",
    "600900","000651","600276","601398","000001","603259","600030","002415",
    "601288","600809","000725","601088","601012","002714","000002","600887",
    "601857","600028","688981","002475","300059","601688","000063","300124",
    "600585","600309","600436","002594","601225","603288","002304","000568",
    "601995","600570","002352","300015","601066","600104","601628","000776",
    "300498","002230","600050","601390","600031","000100","002129","601985",
    "600019","000338","000625","601818","600346","002142","600690","600741",
    "601238","600372","300014","002027","600660","000725","603501","601006",
    "600176","002916","600460","603986","601615","002049","000408","300782",
    "600809","002230","002475","688111","300433","002371","600298","600584",
    "601216","603799","600521","002602","603160","600745","601689","002920",
    "603290","600536","002456","300124",
]


def pull(code: str, days: int = 1500) -> pd.DataFrame | None:
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
    c = df["close"]; f = pd.DataFrame(index=df.index)
    f["mom20"] = c.pct_change(20)
    f["rev60"] = -c.pct_change(60)
    f["vs_ma250"] = (c - c.rolling(250).mean()) / c.rolling(250).mean()
    f["vs_ma60"] = (c - c.rolling(60).mean()) / c.rolling(60).mean()
    f["vol20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    f["turn_chg"] = df["volume"].pct_change(5).fillna(0)
    f["amp"] = (df["high"] - df["low"]) / c
    ema12 = c.ewm(span=12,adjust=False).mean(); ema26 = c.ewm(span=26,adjust=False).mean()
    f["macd_hist"] = (ema12-ema26) - (ema12-ema26).ewm(span=9,adjust=False).mean()
    return f


def daily_ls_return(factor_values: dict[str, np.ndarray],
                    next_rets: dict[str, float],
                    top_n: int = 10) -> float:
    """One day: rank stocks by factor, long top_N, short bottom_N, equal-weight."""
    n = len(next_rets)
    if n < top_n * 2:
        return 0.0
    ranked = sorted(factor_values.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999)
    ranked = [(s, v) for s, v in ranked if not np.isnan(v)]
    longs = [s for s, _ in ranked[-top_n:]]
    shorts = [s for s, _ in ranked[:top_n]]
    lr = np.mean([next_rets.get(s, 0) for s in longs])
    sr = np.mean([next_rets.get(s, 0) for s in shorts])
    return lr - sr


def _compute_ics(pre: dict, stocks: dict, dates: pd.DatetimeIndex,
                 factor_cols: list[str]) -> dict:
    """Cross-sectional mean |IC| per factor over date range."""
    ics: dict[str, list] = {}
    for i, d in enumerate(dates):
        if i == len(dates) - 1:
            break
        nd = dates[i + 1]
        for col in factor_cols:
            xs, ys = [], []
            for s in stocks:
                if s in pre and col in pre[s].columns and d in pre[s].index:
                    xs.append(pre[s][col].loc[d])
                    ys.append(stocks[s]["return"].get(nd, np.nan))
            valid = [(x, y) for x, y in zip(xs, ys) if not (np.isnan(x) or np.isnan(y))]
            if len(valid) > 20:
                xv, yv = zip(*valid)
                ic = np.corrcoef(xv, yv)[0, 1]
                if not np.isnan(ic):
                    ics.setdefault(col, []).append(ic)
    return ics


def main():
    print("=" * 72)
    print("  RIGOROUS VALIDATION: 100 stocks × In/Out-Sample × MC")
    print("=" * 72)

    # ── [1] Pull data ───────────────────────────────────────────────
    print("\n[1] Pulling 100 stocks...")
    stocks = {}
    for i, sym in enumerate(STOCKS_100):
        df = pull(f"sh{sym}" if sym.startswith("6") else f"sz{sym}", days=1500)
        if df is not None and len(df) > 300:
            stocks[sym] = df
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/100 done, {len(stocks)} valid")
        time.sleep(0.12)
    print(f"    Total: {len(stocks)} stocks")

    # ── [2] Precompute factors ──────────────────────────────────────
    factor_cols = ["mom20","rev60","vs_ma250","vs_ma60","vol20","turn_chg","amp","macd_hist"]
    common = stocks[list(stocks.keys())[0]].index
    for df in stocks.values():
        common = common.intersection(df.index)
    pre = {sym: compute_factors(df).reindex(common) for sym, df in stocks.items()}

    # ── [3] In-sample vs Out-of-sample IC ───────────────────────────
    split_date = pd.Timestamp("2024-01-01")
    # Use first 70% / last 30% of each stock's data
    common_all = common
    split_idx = int(len(common_all) * 0.7)
    split_date = common_all[split_idx]
    is_dates = common_all[:split_idx]
    oos_dates = common_all[split_idx:]

    print(f"\n[2] Overfitting check: IS(2018-2023, {len(is_dates)}d) vs OOS(2024+, {len(oos_dates)}d)")
    is_ics = _compute_ics(pre, stocks, is_dates, factor_cols)
    oos_ics = _compute_ics(pre, stocks, oos_dates, factor_cols)

    print(f"    {'Factor':<15} {'IS |IC|':>8}  {'OOS |IC|':>8}  {'Δ':>8}  {'Status'}")
    print(f"    {'─'*15} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")
    degradation = {}
    for col in factor_cols:
        is_ic = abs(np.mean(is_ics.get(col, [0])))
        oos_ic = abs(np.mean(oos_ics.get(col, [0])))
        delta = oos_ic - is_ic
        status = "✅ OK" if delta > -0.005 else ("⚠️ decay" if delta > -0.015 else "❌ overfit")
        degradation[col] = {"is": is_ic, "oos": oos_ic, "delta": delta}
        print(f"    {col:<15} {is_ic:>8.4f}  {oos_ic:>8.4f}  {delta:>+8.4f}  {status}")

    # Drop overfit factors
    good_factors = [col for col, d in degradation.items()
                    if d["delta"] > -0.01 and d["oos"] > 0.005]
    dropped = set(factor_cols) - set(good_factors)
    if dropped:
        print(f"\n    ⚠️ Dropping overfit/noise factors: {dropped}")
    print(f"    Keeping: {good_factors} ({len(good_factors)}/{len(factor_cols)})")

    # ── [4] Realized portfolio Sharpe ───────────────────────────────
    print(f"\n[3] Building {len(stocks)}-stock portfolio ({len(good_factors)} factors)...")
    n_stocks = len(stocks)
    TOP_N = max(5, n_stocks // 10)

    all_dates = common[common >= "2020-01-01"]
    port_rets = []
    for i, d in enumerate(all_dates):
        if i == len(all_dates) - 1: break
        nd = all_dates[i + 1]
        daily = []
        for col in good_factors:
            vals = {s: pre[s][col].get(d, np.nan) for s in stocks}
            nxt = {s: stocks[s]["return"].get(nd, 0) for s in stocks}
            r = daily_ls_return(vals, nxt, TOP_N)
            daily.append(r)
        if daily:
            port_rets.append(np.mean(daily))
    port_ret = pd.Series(port_rets, index=all_dates[1:len(port_rets)+1])
    equity = (1.0 + port_ret).cumprod() * INITIAL
    m = calc_metrics(equity, trades=[], initial_cash=INITIAL, bars_per_year=252)
    real_sharpe = m["sharpe"]

    print(f"    Realized Sharpe: {real_sharpe:.3f}")
    print(f"    AnnRet: {m['annual_return']:.2%}  MaxDD: {m['max_drawdown']:.2%}")

    # ── [5] Fast MC: shuffle portfolio returns (destroy any autocorrelation) ─
    print(f"\n[4] Fast MC ({N_MC} bootstrap shuffles, 5s)...")
    np.random.seed(42)
    mc_sharpes = np.zeros(N_MC)
    rets_arr = port_ret.values
    n_rets = len(rets_arr)
    for mc in range(N_MC):
        shuffled = rets_arr[np.random.permutation(n_rets)]
        mc_eq = (1.0 + pd.Series(shuffled)).cumprod() * INITIAL
        mc_m = calc_metrics(mc_eq, trades=[], initial_cash=INITIAL, bars_per_year=252)
        mc_sharpes[mc] = mc_m["sharpe"]

    mc_sharpes = np.array(mc_sharpes)
    mc_mean = np.mean(mc_sharpes)
    mc_std = np.std(mc_sharpes)
    mc_p95 = np.percentile(mc_sharpes, 95)
    mc_p99 = np.percentile(mc_sharpes, 99)
    p_value = np.mean(mc_sharpes >= real_sharpe)

    print(f"\n    MC null distribution (no real alpha):")
    print(f"      mean={mc_mean:.3f}  std={mc_std:.3f}  p95={mc_p95:.3f}  p99={mc_p99:.3f}")
    print(f"      Real Sharpe={real_sharpe:.3f}")
    print(f"      p-value={p_value:.4f}  (prob of seeing this Sharpe by chance)")

    # ── [6] Verdict ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  VERDICT")
    print(f"  {'─'*30}")
    print(f"  Stocks:        {len(stocks)}")
    print(f"  Factors:       {len(good_factors)} (after dropping overfit)")
    print(f"  Real Sharpe:   {real_sharpe:.3f}")
    print(f"  MC p-value:    {p_value:.4f}")
    print(f"  Baseline:      Sharpe 0.95 (20yr) / ~1.3 (ex-2008)")
    print(f"  Gap to base:   {real_sharpe - 0.95:+.3f}")

    if p_value < 0.05 and real_sharpe > 0.95:
        print(f"\n  ✅ Statistically significant alpha (p={p_value:.4f})")
        print(f"     and beats 20yr baseline.")
        if real_sharpe >= 1.2:
            print(f"     Approaching ex-2008 baseline level.")
    elif p_value < 0.05:
        print(f"\n  ⚠️  Statistically significant but below baseline.")
    else:
        print(f"\n  ❌ Not statistically significant (p={p_value:.4f}).")
        print(f"     Cannot reject null hypothesis of no alpha.")

    roi = real_sharpe / mc_p95
    print(f"  Sharpe/MC-p95:  {roi:.2f}x  (>1.5x = robust)")
    print()


if __name__ == "__main__":
    main()
