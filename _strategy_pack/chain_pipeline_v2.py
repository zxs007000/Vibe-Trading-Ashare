"""Logic Chain Pipeline v2: 100 stocks × deeper factors × wider filter."""

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

def pull(code: str, days=1500) -> pd.DataFrame | None:
    try:
        u=f"{TENCENT}?param={code},day,,,{days},qfq"
        r=requests.get(u,timeout=10)
        d=r.json()["data"][code]
        k=d.get("qfqday",d.get("day",[])); k=[x[:6] for x in k]
        df=pd.DataFrame(k,columns=["date","open","close","high","low","volume"])
        for c in ["open","close","high","low","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
        df["date"]=pd.to_datetime(df["date"]); df=df.set_index("date").sort_index()
        df["return"]=df["close"].pct_change(); return df
    except: return None

# ══════════════════════════════════════════════════════════════════════
# Deep factors — academic zoo + alpha101 inspired, multi-window
# ══════════════════════════════════════════════════════════════════════

def f_value(c: pd.Series) -> pd.Series:
    """HML(-252d) + vsMA250."""
    return (-c.pct_change(252) + (c-c.rolling(250).mean())/c.rolling(250).mean()).fillna(0)

def f_momentum(c: pd.Series) -> pd.Series:
    """20d return + 52w-high proximity."""
    h52=c.rolling(252).max()
    return (c.pct_change(20)+(c-h52)/h52.replace(0,1)).fillna(0)

def f_quality(c: pd.Series, v: pd.Series) -> pd.Series:
    """-vol_growth(CMA) + -std60(RMW)."""
    v60=v.rolling(60).mean(); dv=-np.log(v60/v60.shift(60).replace(0,1))
    std60=c.pct_change().rolling(60).std()
    return (dv-std60).fillna(0)

def f_volatility(c: pd.Series) -> pd.Series:
    """-ret_skew60 - vol20."""
    r=c.pct_change(); skew=r.rolling(60).skew()
    vol20=r.rolling(20).std()*np.sqrt(252)
    return (-skew-vol20*0.1).fillna(0)

def f_liquidity(c: pd.Series, v: pd.Series) -> pd.Series:
    """-Amihud illiq + volume_break."""
    r=c.pct_change().abs(); illiq=(r/(c*v+1)).rolling(21).mean()
    v_break=v/v.rolling(20).mean()-1
    return (-illiq+v_break).fillna(0)

def f_reversal(c: pd.Series) -> pd.Series:
    """-ret5 - 0.5*ret60."""
    return (-c.pct_change(5)-0.5*c.pct_change(60)).fillna(0)

def f_volume(c: pd.Series, v: pd.Series) -> pd.Series:
    """Volume change + volume/MA20."""
    vma=v.rolling(20).mean()
    return (v.pct_change(5)+(v-vma)/vma.replace(0,1)).fillna(0)

def f_micro(c: pd.Series, h: pd.Series, l: pd.Series) -> pd.Series:
    """Amplitude + close_position."""
    return ((h-l)/c+(c-l)/(h-l+1e-10)).fillna(0)

# ══════════════════════════════════════════════════════════════════════
# Chain config
# ══════════════════════════════════════════════════════════════════════

CHAINS = {
    "value_momentum":    ["value","momentum","volume"],
    "value_qlowvol":     ["value","quality","volatility"],
    "value_stable":      ["value","volatility","liquidity"],
    "quality_momentum":  ["quality","momentum","volume"],
    "reversal_momentum": ["reversal","momentum","volume"],
    "vol_reversal":      ["volatility","reversal"],
    "liq_momentum":      ["liquidity","momentum","volume"],
    "micro_reversal":    ["microstructure","reversal"],
}
CNAME={"value_momentum":"价值动量","value_qlowvol":"价值质量低波",
       "value_stable":"价值稳定","quality_momentum":"质量动量",
       "reversal_momentum":"反转接力","vol_reversal":"波动回归",
       "liq_momentum":"放量动量","micro_reversal":"微观反转"}

FACTORS = {
    "value": lambda df: f_value(df["close"]),
    "momentum": lambda df: f_momentum(df["close"]),
    "quality": lambda df: f_quality(df["close"],df["volume"]),
    "volatility": lambda df: f_volatility(df["close"]),
    "liquidity": lambda df: f_liquidity(df["close"],df["volume"]),
    "reversal": lambda df: f_reversal(df["close"]),
    "volume": lambda df: f_volume(df["close"],df["volume"]),
    "microstructure": lambda df: f_micro(df["close"],df["high"],df["low"]),
}

# ══════════════════════════════════════════════════════════════════════
# Pipeline: sequential filtering, wider gates (KEEP=0.60 per stage)
# ══════════════════════════════════════════════════════════════════════

def pipeline_chain(cid: str, stocks: dict, pre: dict,
                   common: pd.DatetimeIndex, keep_frac: float = 0.60) -> list:
    stages = CHAINS[cid]; rets = []
    for i, d in enumerate(common):
        if i == len(common)-1: break
        nd = common[i+1]
        pool = set(stocks.keys())
        for stage in stages:
            scores = {s: pre[s][stage].get(d,np.nan) for s in pool}
            scores = {s: v for s, v in scores.items() if not np.isnan(v)}
            if len(scores) < 3: break
            keep = max(2, int(len(scores) * keep_frac))
            pool = {s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:keep]}
        if len(pool) >= 2:
            nxt = [stocks[s]["return"].get(nd,0) for s in pool]
            rets.append(np.mean(nxt))
    return rets


def main():
    print("=" * 72)
    print("  PIPELINE v2: 100 stocks × deep factors × 60% gates")
    print("=" * 72)

    # ── Stocks ──────────────────────────────────────────────────────
    print("\n[1] Pulling 100 stocks...")
    SYMS = [
        "600519","000858","601318","600036","000333","601899","300750","601166",
        "600900","000651","600276","601398","000001","603259","600030","002415",
        "601288","600809","000725","601088","601012","002714","000002","600887",
        "601857","600028","688981","002475","300059","601688","000063","300124",
        "600585","600309","600436","002594","601225","603288","002304","000568",
        "601995","600570","002352","300015","601066","600104","601628","000776",
        "300498","002230","600050","601390","600031","000100","002129","601985",
        "600019","000338","000625","601818","600346","002142","600690","600741",
        "601238","600372","300014","002027","600660","002456","603501","601006",
        "600176","002916","600460","603986","601615","002049","000408","300782",
        "600809","600584","601216","603799","600521","002602","603160","600745",
        "601689","002920","603290","600536","000568","002371","002475","300433",
        "688111","600298","300124","002049",
    ]
    stocks = {}
    for i, s in enumerate(SYMS):
        df = pull(f"sh{s}" if s.startswith("6") else f"sz{s}")
        if df is not None and len(df) > 500: stocks[s] = df
        if (i+1) % 20 == 0: print(f"    {i+1}/{len(SYMS)}, {len(stocks)} valid")
        time.sleep(0.10)
    print(f"    {len(stocks)} stocks")

    # ── Precompute factors ──────────────────────────────────────────
    common = stocks[list(stocks.keys())[0]].index
    for d in stocks.values(): common = common.intersection(d.index)
    common = common[-600:]

    print(f"\n[2] Precomputing 8 deep factors ({len(stocks)} stocks)...")
    pre = {}
    for sym, df in stocks.items():
        pf = pd.DataFrame(index=df.index)
        for th, fn in FACTORS.items(): pf[th] = fn(df)
        pre[sym] = pf.reindex(common)
    print(f"    Done. {common[0].date()} ~ {common[-1].date()} ({len(common)}d)")

    # ── CSI300 ──────────────────────────────────────────────────────
    idx = pull("sh000300")
    idx_r = idx["return"].reindex(common).dropna()
    idx_e = (1.0 + idx_r).cumprod() * INITIAL
    idx_m = calc_metrics(idx_e, trades=[], initial_cash=INITIAL, bars_per_year=252)

    # ── Chains ──────────────────────────────────────────────────────
    print(f"\n[3] Running 8 chains (50% gate per stage)...")
    results = {}
    for cid in CHAINS:
        rets = pipeline_chain(cid, stocks, pre, common, keep_frac=0.50)
        if len(rets) < 30: continue
        rs = pd.Series(rets)
        eq = (1.0 + rs).cumprod() * INITIAL
        m = calc_metrics(eq, trades=[], initial_cash=INITIAL, bars_per_year=252)
        results[cid] = {"sharpe": m["sharpe"], "ann_ret": m["annual_return"],
                        "max_dd": m["max_drawdown"], "n": len(rets)}

    # ── Print ───────────────────────────────────────────────────────
    print(f"\n    CSI300: Sharpe={idx_m['sharpe']:.3f}  "
          f"{idx_m['annual_return']:.1%}  DD={idx_m['max_drawdown']:.1%}")
    print(f"    Baseline(20yr)=0.95 | 常态=1.3 | 股灾DD=-25%")
    print(f"    {'Chain':<20} {'Sharpe':>7} {'AnnRet':>7} {'MaxDD':>7} {'vsCSI':>7} {'vBline':>7}")
    print(f"    {'─'*20} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for cid in sorted(results, key=lambda c: -results[c]["sharpe"]):
        r = results[cid]
        print(f"    {CNAME[cid]:<20} {r['sharpe']:>7.3f} {r['ann_ret']:>6.1%} "
              f"{r['max_dd']:>6.1%} {r['sharpe']-idx_m['sharpe']:>+7.3f} "
              f"{r['sharpe']/0.95:>6.0%}")

    best = max(results, key=lambda c: results[c]["sharpe"])
    print(f"\n    Best: {CNAME[best]} (Sharpe {results[best]['sharpe']:.3f})")
    print()


if __name__ == "__main__":
    main()
