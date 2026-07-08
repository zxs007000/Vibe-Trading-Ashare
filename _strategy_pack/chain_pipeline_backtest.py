"""Logic Chain Pipeline Backtest — sequential filtering, not equal-weight.

Each chain = 2–3 stage pipeline (value→momentum→volume etc.).
At each stage, the stock universe shrinks — factors are FILTERS, not ranks.

Uses zoo factors (academic + alpha101) mapped to chain themes.
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

STOCKS = [
    "600519","000858","601318","600036","000333","601899","300750","601166",
    "600900","000651","600276","601398","000001","603259","600030","002415",
    "601288","600809","000725","601088","601012","002714","000002","600887",
    "601857","600028","688981","002475","300059","601688","000063","300124",
    "600585","600309","600436","002594","601225","603288","002304","000568",
    "601995","600570","002352","300015","601066","600104","601628","000776",
    "300498","002230",
]

# ══════════════════════════════════════════════════════════════════════
# Logic chain definitions — each theme maps to zoo/derived factor functions
# ══════════════════════════════════════════════════════════════════════

def factor_value(c: pd.Series) -> pd.Series:
    """Value: inverse 252d return (＝ academic HML), + vs MA250."""
    return (-c.pct_change(252) + (c - c.rolling(250).mean()) / c.rolling(250).mean()).fillna(0)

def factor_momentum(c: pd.Series) -> pd.Series:
    """Momentum: 20d return + 52w high proximity (＝ carhart_mom + academic)."""
    h52 = c.rolling(252).max()
    mom = c.pct_change(20) + 0.5 * ((c - h52) / h52.replace(0,1))
    return mom.fillna(0)

def factor_quality(c: pd.Series, v: pd.Series) -> pd.Series:
    """Quality: inverse vol-growth(CMA) + low-vol(RMW) + small-cap(SMB)."""
    vol60 = v.rolling(60).mean()
    vol_g = -np.log(vol60 / vol60.shift(60).replace(0,1))
    std60 = c.pct_change().rolling(60).std()
    dv = -np.log(v.rolling(60).mean() * c.rolling(60).mean() + 1)
    return (vol_g - std60 + dv).fillna(0)

def factor_volatility(c: pd.Series) -> pd.Series:
    """Volatility: inverse 60d return skewness (retskew) + inverse 20d vol."""
    r = c.pct_change()
    skew = r.rolling(60).skew()
    vol20 = r.rolling(20).std() * np.sqrt(252)
    return (-skew - vol20).fillna(0)

def factor_liquidity(c: pd.Series, v: pd.Series) -> pd.Series:
    """Liquidity: Amihud illiq + volume change."""
    r = c.pct_change().abs()
    illiq = (r / (c * v + 1)).rolling(21).mean()
    vchg = v.pct_change(5)
    return (illiq + 0.1 * vchg).fillna(0)

def factor_reversal(c: pd.Series) -> pd.Series:
    """Reversal: 5d + 60d inverse return."""
    return (-c.pct_change(5) - 0.5 * c.pct_change(60)).fillna(0)

def factor_volume(c: pd.Series, v: pd.Series) -> pd.Series:
    """Volume: 5d volume change + volume/MA ratio."""
    vma = v.rolling(20).mean()
    return (v.pct_change(5) + (v - vma) / vma.replace(0,1)).fillna(0)

def factor_microstructure(df: pd.DataFrame) -> pd.Series:
    """Microstructure: amplitude + high-low position."""
    c, h, l = df["close"], df["high"], df["low"]
    amp = (h - l) / c
    pos = (c - l) / (h - l + 1e-10)  # close position within daily range
    return (amp + pos).fillna(0)

# ══════════════════════════════════════════════════════════════════════
# Chain pipeline: sequential filtering
# ══════════════════════════════════════════════════════════════════════

CHAINS = {
    "value_momentum":    ["value", "momentum", "volume"],
    "value_qlowvol":     ["value", "quality", "volatility"],
    "value_stable":      ["value", "volatility", "liquidity"],
    "quality_momentum":  ["quality", "momentum", "volume"],
    "reversal_momentum": ["reversal", "momentum", "volume"],
    "vol_reversal":      ["volatility", "reversal"],
    "liq_momentum":      ["liquidity", "momentum", "volume"],
    "micro_reversal":    ["microstructure", "reversal"],
}

CHAIN_NAMES = {
    "value_momentum":"价值动量","value_qlowvol":"价值质量低波",
    "value_stable":"价值稳定","quality_momentum":"质量动量",
    "reversal_momentum":"反转接力","vol_reversal":"波动回归",
    "liq_momentum":"放量动量","micro_reversal":"微观反转",
}

FACTOR_FUNC = {
    "value":          lambda df: factor_value(df["close"]),
    "momentum":       lambda df: factor_momentum(df["close"]),
    "quality":        lambda df: factor_quality(df["close"], df["volume"]),
    "volatility":     lambda df: factor_volatility(df["close"]),
    "liquidity":      lambda df: factor_liquidity(df["close"], df["volume"]),
    "reversal":       lambda df: factor_reversal(df["close"]),
    "volume":         lambda df: factor_volume(df["close"], df["volume"]),
    "microstructure": lambda df: factor_microstructure(df),
}


def pull(code: str, days=1500) -> pd.DataFrame | None:
    try:
        u=f"{TENCENT}?param={code},day,,,{days},qfq"
        r=requests.get(u,timeout=10)
        d=r.json()["data"][code]
        k=d.get("qfqday",d.get("day",[])); k=[x[:6] for x in k]
        df=pd.DataFrame(k,columns=["date","open","close","high","low","volume"])
        for c in ["open","close","high","low","volume"]:
            df[c]=pd.to_numeric(df[c],errors="coerce")
        df["date"]=pd.to_datetime(df["date"]); df=df.set_index("date").sort_index()
        df["return"]=df["close"].pct_change()
        return df
    except: return None


def pipeline_chain(chain_id: str, stocks_data: dict, precomputed: dict,
                   common: pd.DatetimeIndex, top_final: int = 10) -> tuple:
    """Run one chain as a sequential pipeline on all stocks.

    Returns (daily_rets, daily_counts) where:
      - daily_rets: list of next-day returns for the final selected stocks
      - daily_counts: how many stocks survived each stage per day
    """
    stages = CHAINS[chain_id]
    daily_rets = []
    stage_counts = {s: [] for s in stages}

    for i, d in enumerate(common):
        if i == len(common) - 1: break
        nd = common[i + 1]

        pool = set(stocks_data.keys())  # start with all stocks
        for stage_idx, stage in enumerate(stages):
            if not pool: break
            # Score pool by this stage's factor
            scores = {}
            for sym in pool:
                if sym in precomputed and stage in precomputed[sym]:
                    v = precomputed[sym][stage].get(d, np.nan)
                    if not np.isnan(v): scores[sym] = v
            if not scores: break

            # Keep top 40% at each stage
            keep_n = max(2, int(len(scores) * 0.40))
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            pool = {s for s, _ in ranked[:keep_n]}
            stage_counts[stage].append(len(pool))

        if not pool: continue

        # Final pool: equal-weight return
        nxt_rets = [stocks_data[s]["return"].get(nd, 0) for s in pool]
        daily_rets.append(np.mean(nxt_rets))

    return daily_rets, stage_counts


def main():
    print("=" * 72)
    print("  LOGIC CHAIN PIPELINE BACKTEST (sequential filtering)")
    print("  Zoo factors mapped to chain themes, 50 stocks, real data")
    print("=" * 72)

    # ── Pull stocks ──────────────────────────────────────────────────
    print("\n[1] Pulling 50 stocks...")
    stocks_data = {}
    for i, sym in enumerate(STOCKS):
        df = pull(f"sh{sym}" if sym.startswith("6") else f"sz{sym}")
        if df is not None and len(df) > 500:
            stocks_data[sym] = df
        if (i+1) % 10 == 0: print(f"    {i+1}/50, {len(stocks_data)} valid")
        time.sleep(0.12)
    print(f"    {len(stocks_data)} stocks")

    # ── Precompute factors ───────────────────────────────────────────
    common = stocks_data[list(stocks_data.keys())[0]].index
    for df in stocks_data.values():
        common = common.intersection(df.index)
    common = common[-600:]  # ~2.5yr

    print(f"\n[2] Precomputing 8 theme factors for {len(stocks_data)} stocks...")
    pre = {}
    for sym, df in stocks_data.items():
        pf = pd.DataFrame(index=df.index)
        for theme, fn in FACTOR_FUNC.items():
            pf[theme] = fn(df)
        pre[sym] = pf.reindex(common)
    print(f"    Done. Common window: {common[0].date()} ~ {common[-1].date()}")

    # ── Build CSI300 baseline for comparison ─────────────────────────
    idx = pull("sh000300")
    idx_ret = idx["return"].reindex(common).dropna()
    idx_eq = (1.0 + idx_ret).cumprod() * INITIAL
    idx_m = calc_metrics(idx_eq, trades=[], initial_cash=INITIAL, bars_per_year=252)

    # ── Run chains ───────────────────────────────────────────────────
    print(f"\n[3] Running 8 logic chain pipelines...")
    results = {}
    for cid in CHAINS:
        rets, counts = pipeline_chain(cid, stocks_data, pre, common, top_final=10)
        if len(rets) < 30: continue
        ret_s = pd.Series(rets)
        eq = (1.0 + ret_s).cumprod() * INITIAL
        m = calc_metrics(eq, trades=[], initial_cash=INITIAL, bars_per_year=252)
        avg_pool = np.mean([np.mean(c) for c in counts.values() if c]) if counts else 0
        results[cid] = {
            "sharpe": m["sharpe"], "ann_ret": m["annual_return"],
            "max_dd": m["max_drawdown"], "avg_pool": int(avg_pool),
            "n_days": len(rets),
        }

    # ── Print ────────────────────────────────────────────────────────
    print(f"\n    CSI300: Sharpe={idx_m['sharpe']:.3f}  "
          f"AnnRet={idx_m['annual_return']:.2%}  MaxDD={idx_m['max_drawdown']:.2%}")
    print(f"    {'Chain':<20} {'Sharpe':>8} {'AnnRet':>8} {'MaxDD':>8} {'Pool':>5} {'vsCSI':>8}")
    print(f"    {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*5} {'─'*8}")

    for cid in sorted(results, key=lambda c: -results[c]["sharpe"]):
        r = results[cid]
        vs = r["sharpe"] - idx_m["sharpe"]
        name = CHAIN_NAMES.get(cid, cid)
        print(f"    {name:<20} {r['sharpe']:>8.3f} {r['ann_ret']:>7.2%} "
              f"{r['max_dd']:>7.2%} {r['avg_pool']:>5.0f} {vs:>+7.3f}")

    best = max(results, key=lambda c: results[c]["sharpe"])
    print(f"\n    Top chain: {CHAIN_NAMES[best]} (pool ~{results[best]['avg_pool']} stocks/stage)")
    print(f"    Baseline(20yr): Sharpe 0.95 | 常态: Sharpe ~1.3")
    print()


if __name__ == "__main__":
    main()
