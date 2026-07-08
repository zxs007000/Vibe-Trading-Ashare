"""verify_structured_reversal.py — 月度多空回测

50×CSI300 股票, 每月末按 structured_reversal(volume) 分10组,
long 第1组(超卖)/short 第10组(强趋势), 等权月度换仓。
输出: Sharpe/MaxDD/IR + 逐组IC, 对照论文基准 2.54。
"""

from __future__ import annotations

import sys, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from numpy.lib.stride_tricks import sliding_window_view

_PROJECT = Path(__file__).resolve().parents[2]
_AGENT = _PROJECT / "agent"
sys.path.insert(0, str(_PROJECT)); sys.path.insert(0, str(_AGENT))

TENCENT = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
INITIAL = 1_000_000

SYMS = [
    "600519","000858","601318","600036","000333","601899","300750","601166",
    "600900","000651","600276","601398","000001","603259","600030","002415",
    "601288","600809","000725","601088","601012","002714","000002","600887",
    "601857","600028","688981","002475","300059","601688","000063","300124",
    "600585","600309","600436","002594","601225","603288","002304","000568",
    "601995","600570","002352","300015","601066","600104","601628","000776",
    "300498","002230",
]


def pull(code: str, days: int = 1500) -> pd.DataFrame | None:
    try:
        r = requests.get(f"{TENCENT}?param={code},day,,,{days},qfq", timeout=10)
        d = r.json()["data"][code]
        k = d.get("qfqday", d.get("day", [])); k = [x[:6] for x in k]
        df = pd.DataFrame(k, columns=["date","open","close","high","low","volume"])
        for c in ["open","close","high","low","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["return"] = df["close"].pct_change()
        return df
    except Exception:
        return None


def structured_reversal(bars, method="volume", window=21, halflife=5.0):
    close = pd.to_numeric(bars["close"], errors="coerce")
    vol = pd.to_numeric(bars.get("volume", pd.Series(np.nan, index=bars.index)),
                        errors="coerce").astype(float)
    log_ret = np.log(close / close.shift(1)).to_numpy(dtype=float)
    vol_arr = vol.to_numpy(dtype=float)
    n = len(log_ret); out = np.full(n, np.nan, dtype=float)
    if n < window: return pd.Series(out, index=bars.index)
    if method == "equal":
        W = np.ones(window)
        out[window-1:] = sliding_window_view(log_ret, window).dot(W) / W.sum()
    elif method == "time":
        ages = (window-1) - np.arange(window)
        W = np.exp(-ages * np.log(2) / halflife)
        out[window-1:] = sliding_window_view(log_ret, window).dot(W) / W.sum()
    else:
        R = sliding_window_view(log_ret, window); V = sliding_window_view(vol_arr, window)
        ws = np.nansum(R*V, axis=1); wt = np.nansum(V, axis=1)
        out[window-1:] = np.where((wt>0)&np.isfinite(wt), ws/wt, np.nan)
    return pd.Series(out, index=bars.index)


def metrics(r: pd.Series) -> tuple:
    r = r.dropna(); n = len(r)
    if n < 5: return (0, 0, 0)
    eq = (1+r).cumprod(); dd = float((eq-eq.cummax()).div(eq.cummax()).min())
    ann = float(eq.iloc[-1]**(252/n)-1); vol = float(r.std()*np.sqrt(252))
    return (ann/(vol+1e-10), ann, dd)


def main():
    print("=" * 68)
    print("  VERIFY: structured_reversal 月度多空回测")
    print("  50 stocks × 10 groups × monthly rebalance")
    print("=" * 68)

    # ── Pull stocks ──────────────────────────────────────────────────
    print("\n[1] Pulling 50 stocks...")
    stocks = {}
    for i, s in enumerate(SYMS):
        df = pull(f"sh{s}" if s.startswith("6") else f"sz{s}")
        if df is not None and len(df) > 500: stocks[s] = df
        if (i+1) % 10 == 0: print(f"    {i+1}/50, {len(stocks)} valid")
        time.sleep(0.12)
    print(f"    {len(stocks)} stocks")

    # ── Compute structured_reversal (volume) for all ──────────────────
    common = stocks[list(stocks.keys())[0]].index
    for d in stocks.values(): common = common.intersection(d.index)
    common = pd.DatetimeIndex(sorted(set(common[-600:])))

    print(f"\n[2] Computing structured_reversal(volume, w=21)...")
    rev_factors = {}
    for sym, df in stocks.items():
        rev_factors[sym] = structured_reversal(df, method="volume", window=21).reindex(common)
    print(f"    Done. {common[0].date()} ~ {common[-1].date()}")

    # ── Monthly group backtest ────────────────────────────────────────
    print(f"\n[3] Monthly long-short (10 groups)...")
    months = pd.Series(1, index=common).resample("ME").last().index
    month_keys = sorted(months)
    ls_rets = []
    group_rets: dict[int, list] = {g: [] for g in range(10)}
    group_ics = []

    for mi in range(len(month_keys)-1):
        m_start = month_keys[mi]
        m_end   = month_keys[mi+1]

        # Signal at month start
        signals = {}
        for sym in stocks:
            v = rev_factors[sym].get(m_start, np.nan)
            if not np.isnan(v): signals[sym] = v

        if len(signals) < 20: continue
        ranked = sorted(signals.items(), key=lambda x: -x[1])  # higher = more momentum
        n_stocks = len(ranked)
        group_size = max(2, n_stocks // 10)

        # Future return for this month
        future_rets = {}
        for code, _ in ranked:
            rr = stocks[code]["return"].loc[m_start:m_end].mean()
            future_rets[code] = rr

        # Assign to groups — G0=最高因子(强势), G9=最低因子(弱势)
        for g in range(10):
            start = g * group_size
            end = (g+1) * group_size if g < 9 else n_stocks
            g_stocks = [s for s, _ in ranked[start:end]]
            g_ret = np.mean([future_rets.get(s, 0) for s in g_stocks])
            group_rets[g].append(g_ret)

        # Long-short: G0(强势) minus G9(弱势)
        if group_rets[0] and group_rets[9]:
            ls_rets.append(group_rets[0][-1] - group_rets[9][-1])

        # IC: factor value vs future return
        xs = [v for _, v in ranked]
        ys = [future_rets.get(c, np.nan) for c, _ in ranked]
        valid = [(x, y) for x, y in zip(xs, ys) if not np.isnan(y)]
        if len(valid) > 15:
            ic = np.corrcoef([v[0] for v in valid], [v[1] for v in valid])[0,1]
            if not np.isnan(ic): group_ics.append(ic)

    # ── Results ───────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  RESULTS ({len(month_keys)} months)")
    print(f"{'='*68}")

    if ls_rets:
        ls = pd.Series(ls_rets)
        sh, ann, dd = metrics(ls)
        ic_mean = np.mean(group_ics)
        ic_ir = ic_mean / (np.std(group_ics) + 1e-10)
        print(f"  Long-Short(G0强势-G9弱势): {ann:.1%}/yr  Sharpe{sh:.3f}  MaxDD{dd:.1%}")
        print(f"  Mean IC: {ic_mean:.4f}  IR: {ic_ir:.2f}")
        print(f"  Months: {len(ls_rets)}  ({sum(1 for r in list(ls_rets) if r>0)} positive)")

    print(f"\n  ── Per-group returns ──")
    print(f"  {'Group':<8} {'AnnRet':>8} {'Sharpe':>7} {'IC_vs_next':>10}")
    print(f"  {'─'*8} {'─'*8} {'─'*7} {'─'*10}")
    for g in range(10):
        gr = pd.Series(group_rets[g])
        gsh, gann, gdd = metrics(gr)
        gic = np.corrcoef(range(10), [np.mean(group_rets[i]) if group_rets[i] else 0 for i in range(10)])[0,1]
        tag = " ← LONG(强势)" if g == 0 else (" ← SHORT(弱势)" if g == 9 else "")
        print(f"  G{g:<7} {gann:>7.1%}  {gsh:>+7.3f}  {'':>10}{tag}")

    print(f"\n  Reference: paper Sharpe 2.54 (monthly, multi-stock cross-section)")
    print(f"  Note: our monthly long-short uses {len(stocks)} stocks vs paper's universe")


if __name__ == "__main__":
    main()
