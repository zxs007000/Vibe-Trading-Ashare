"""Phase 2: 财务/情绪因子精炼 + IC 归因 + 复合回测。

新增因子：
  fundamental_quality_roe_change  — ROE QoQ 变化率
  fundamental_quality_roe_pct     — ROE 截面分位
  sentiment_divergence            — 新闻情感分歧度
"""

from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENT_DIR = _PROJECT_ROOT / "agent"
for _p in (str(_PROJECT_ROOT), str(_AGENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.backtest.loaders.financial_loader import fetch_fundamentals
from agent.backtest.metrics import calc_metrics
from agent.src.factors.registry import Registry

INITIAL = 1_000_000

# ── Data pull ──────────────────────────────────────
def pull(code: str) -> pd.DataFrame | None:
    from stcok_worm import tencent
    rows = tencent.get_kline(code)
    if not rows: return None
    df = pd.DataFrame(rows, columns=["date","open","close","high","low","volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["return"] = df["close"].pct_change()
    return df

# ── 8 original factors ─────────────────────────────
def f_value(c): return (-c.pct_change(252)+(c-c.rolling(250).mean())/c.rolling(250).mean()).fillna(0)
def f_momentum(c): h52=c.rolling(252).max(); return (c.pct_change(20)+(c-h52)/h52.replace(0,1)).fillna(0)
def f_quality(c,v): v60=v.rolling(60).mean(); dv=-np.log(v60/v60.shift(60).replace(0,1)); return (dv-c.pct_change().rolling(60).std()).fillna(0)
def f_volatility(c): r=c.pct_change(); return (-r.rolling(60).skew()-r.rolling(20).std()*np.sqrt(252)*0.1).fillna(0)
def f_liquidity(c,v): r=c.pct_change().abs(); return (-(r/(c*v+1)).rolling(21).mean()+v/v.rolling(20).mean()-1).fillna(0)
def f_reversal(c): return (-c.pct_change(5)-0.5*c.pct_change(60)).fillna(0)
def f_volume(c,v): vma=v.rolling(20).mean(); return (v.pct_change(5)+(v-vma)/vma.replace(0,1)).fillna(0)
def f_micro(c,h,l): return ((h-l)/c+(c-l)/(h-l+1e-10)).fillna(0)

FACTORS_ORIG = {
    "value":          lambda df: f_value(df["close"]),
    "momentum":       lambda df: f_momentum(df["close"]),
    "quality":        lambda df: f_quality(df["close"], df["volume"]),
    "volatility":     lambda df: f_volatility(df["close"]),
    "liquidity":      lambda df: f_liquidity(df["close"], df["volume"]),
    "reversal":       lambda df: f_reversal(df["close"]),
    "volume":         lambda df: f_volume(df["close"], df["volume"]),
    "microstructure": lambda df: f_micro(df["close"], df["high"], df["low"]),
}

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


def pipeline_chain(cid, stocks, pre, common, chains_dict, keep_frac=0.50):
    stages = chains_dict[cid]; rets = []
    for i, d in enumerate(common):
        if i == len(common)-1: break
        nd = common[i+1]; pool = set(stocks.keys())
        for stage in stages:
            scores = {s: pre[s][stage].get(d, np.nan) for s in pool}
            scores = {s: v for s, v in scores.items() if not np.isnan(v)}
            if len(scores) < 3: break
            keep = max(2, int(len(scores) * keep_frac))
            pool = {s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:keep]}
        if len(pool) >= 2:
            rets.append(np.mean([stocks[s]["return"].get(nd, 0) for s in pool]))
    return rets


def calc_ic(factor_series: dict[str, pd.Series], fwd_rets: dict[str, pd.Series],
            common_dates: pd.DatetimeIndex) -> dict[str, dict]:
    """计算每个因子的 IC (Rank IC) 和 IR。factor_series 是截面快照（symbol→value），fwd_rets 是按日期的前瞻收益."""
    results = {}
    for fname, fser in factor_series.items():
        ic_list = []
        for d in common_dates:
            scores = {}
            rets_fwd = {}
            for sym in fser.index:
                if sym in fwd_rets and d in fwd_rets[sym].index:
                    s = fser.get(sym, np.nan)
                    r = fwd_rets[sym].get(d, np.nan)
                    if not np.isnan(s) and not np.isnan(r):
                        scores[sym] = s
                        rets_fwd[sym] = r
            if len(scores) < 5: continue
            s_rank = pd.Series(scores).rank()
            r_rank = pd.Series(rets_fwd).rank()
            ic_list.append(s_rank.corr(r_rank))
        if ic_list:
            ic_arr = np.array(ic_list)
            results[fname] = {
                "ic_mean": float(np.mean(ic_arr)),
                "ic_std": float(np.std(ic_arr)),
                "ir": float(np.mean(ic_arr) / np.std(ic_arr)) if np.std(ic_arr) > 1e-10 else 0.0,
                "ic_t": float(np.mean(ic_arr) / (np.std(ic_arr)/np.sqrt(len(ic_arr)))) if len(ic_arr)>1 and np.std(ic_arr)>0 else 0.0,
                "n_days": len(ic_list),
            }
    return results


def main():
    print("=" * 72)
    print("  Phase 2: 财务/情绪因子精炼 + IC 归因")
    print("=" * 72)

    # ── Stocks ──────────────────────────────────────
    SYMS = [
        "600519","000858","601318","600036","000333","601899","300750","601166",
        "600900","000651","600276","601398","000001","603259","600030","002415",
        "601288","600809","000725","601088","601012","002714","000002","600887",
        "601857","600028","601688","300059","600585","600309","600436","002594",
        "601225","603288","002304","000568","601066","600104","000776","300498",
    ]
    print(f"\n[1] Loading {len(SYMS)} stocks...")
    stocks = {}
    for i, s in enumerate(SYMS):
        df = pull(s)
        if df is not None and len(df) > 500: stocks[s] = df
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{len(SYMS)}, {len(stocks)} valid")
        time.sleep(0.08)
    print(f"    Got {len(stocks)} stocks")

    common = stocks[list(stocks.keys())[0]].index
    for d in stocks.values(): common = common.intersection(d.index)
    common = common[-600:]
    codes = list(stocks.keys())

    # ── Original 8 factors ─────────────────────────
    print(f"\n[2] Computing 8 original factors...")
    pre = {}
    for sym, df in stocks.items():
        pf = pd.DataFrame(index=df.index)
        for th, fn in FACTORS_ORIG.items(): pf[th] = fn(df)
        pre[sym] = pf.reindex(common)

    # ── Financial factors (stock_worm → extras) ────
    print(f"\n[3] Fetching financial data (stock_worm, 8 periods)...")
    fin_raw = fetch_fundamentals(codes, use_cache=False, prefer="stock_worm", periods=8)
    print(f"    Got {len(fin_raw)} stocks with financial data")

    # Build ROE panel (forward-filled)
    roe_panel = pd.DataFrame(index=common, columns=codes, dtype=float)
    bvps_panel = pd.DataFrame(index=common, columns=codes, dtype=float)
    accruals_panel = pd.DataFrame(index=common, columns=codes, dtype=float)

    for code in codes:
        fdf = fin_raw.get(code)
        if fdf is None or fdf.empty: continue
        fdf = fdf.copy()
        fdf["report_date"] = pd.to_datetime(fdf["report_date"])
        fdf = fdf.sort_values("report_date")
        for col, panel in [("roe", roe_panel), ("bvps", bvps_panel), ("accruals", accruals_panel)]:
            if col not in fdf.columns: continue
            vals = fdf.set_index("report_date")[col].dropna()
            mapped = vals.reindex(common, method="ffill")
            if code in panel.columns:
                panel[code] = mapped

    # ── Compute refined financial factors ──────────
    print(f"\n[4] Computing refined financial factors...")

    # ROE change (QoQ)
    roe_change_panel = roe_panel.diff(periods=1) / roe_panel.shift(1).abs().replace(0, np.nan)
    roe_change_panel = roe_change_panel.clip(-1, 1).fillna(0)

    # ROE percentile (cross-sectional rank)
    roe_pct_panel = pd.DataFrame(index=common, columns=codes, dtype=float)
    for i, dt in enumerate(common):
        row = roe_panel.loc[dt].dropna()
        if len(row) < 3: continue
        ranks = row.rank(pct=True)
        for col in codes:
            if col in ranks.index:
                roe_pct_panel.loc[dt, col] = ranks[col]
    roe_pct_panel = roe_pct_panel.fillna(0.5)

    # ── Sentiment factors ──────────────────────────
    print(f"\n[5] Computing sentiment factors (stock_worm news)...")
    sentiment_scores = {}
    sentiment_divs = {}
    from stcok_worm import news as sw_news
    from stcok_worm.sentiment.analyzers.dictionary import DictionaryAnalyzer
    analyzer = DictionaryAnalyzer()

    for code in codes:
        try:
            articles = sw_news.stock_news(code, page_size=20)
            if articles:
                scores = [analyzer.analyze(a.get("title",""))["sentiment"] for a in articles]
                sentiment_scores[code] = np.mean(scores)
                sentiment_divs[code] = np.std(scores) if len(scores)>=2 else 0.0
            else:
                sentiment_scores[code] = 0.0
                sentiment_divs[code] = 0.0
        except Exception:
            sentiment_scores[code] = 0.0
            sentiment_divs[code] = 0.0
        time.sleep(0.05)

    sent_score_panel = pd.DataFrame(0.0, index=common, columns=codes)
    sent_div_panel = pd.DataFrame(0.0, index=common, columns=codes)
    for code in codes:
        if code in sent_score_panel.columns:
            sent_score_panel[code] = sentiment_scores.get(code, 0.0)
            sent_div_panel[code] = sentiment_divs.get(code, 0.0)

    # ── IC Attribution ─────────────────────────────
    print(f"\n[6] IC Attribution...")
    fwd_rets_5d = {sym: stocks[sym]["close"].pct_change(5).shift(-5) for sym in codes}

    # Prepare factor series for IC
    factor_series = {}

    # Original 8 factors (use latest day)
    for fname in FACTORS_ORIG:
        latest = {}
        for sym in codes:
            ser = pre[sym][fname].dropna()
            if not ser.empty:
                latest[sym] = ser.iloc[-1]
        if latest:
            factor_series[fname] = pd.Series(latest)

    # Refined financial
    for label, panel in [
        ("roe_change", roe_change_panel),
        ("roe_pct", roe_pct_panel),
    ]:
        latest = panel.iloc[-1].dropna()
        if len(latest) >= 3:
            factor_series[label] = latest

    # Sentiment
    for label, panel in [
        ("sent_score", sent_score_panel),
        ("sent_div", sent_div_panel),
    ]:
        latest = panel.iloc[-1].dropna()
        if len(latest) >= 3:
            factor_series[label] = latest

    ic_results = calc_ic(factor_series, fwd_rets_5d, common[-30:])

    print(f"\n    {'Factor':<20} {'IC_mean':>8} {'IC_std':>8} {'IR':>8} {'t-stat':>8} {'n_days':>7}")
    print(f"    {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    for fname in sorted(ic_results, key=lambda f: -abs(ic_results[f]["ic_mean"])):
        r = ic_results[fname]
        print(f"    {fname:<20} {r['ic_mean']:>8.4f} {r['ic_std']:>8.4f} "
              f"{r['ir']:>8.4f} {r['ic_t']:>8.2f} {r['n_days']:>7d}")

    # ── Composite backtest ─────────────────────────
    print(f"\n[7] Composite backtest (8 original chains)...")

    # Build enhanced precomputed factors: orig 8 + new financial/sentiment
    pre_enhanced = {}
    for sym in codes:
        pf = pre[sym].copy()
        # Add refined financial as new factor columns
        if sym in roe_change_panel.columns:
            pf["roe_change"] = roe_change_panel[sym]
        if sym in roe_pct_panel.columns:
            pf["roe_pct"] = roe_pct_panel[sym]
        if sym in sent_score_panel.columns:
            pf["sent_score"] = sent_score_panel[sym]
        if sym in sent_div_panel.columns:
            pf["sent_div"] = sent_div_panel[sym]
        pre_enhanced[sym] = pf

    # Enhanced chains: original + roe_change + roe_pct chains
    ENHANCED_CHAINS = dict(CHAINS)  # copy original
    ENHANCED_CHAINS["roe_quality"] = ["roe_change", "roe_pct", "quality"]
    ENHANCED_CHAINS["roe_value"] = ["roe_pct", "value", "volatility"]
    ENHANCED_CHAINS["sent_momentum"] = ["sent_score", "momentum", "volume"]
    ENHANCED_CHAINS["sent_quality"] = ["sent_score", "sent_div", "quality"]

    results_orig = {}
    results_enhanced = {}

    for cid in CHAINS:
        rets = pipeline_chain(cid, stocks, pre, common, CHAINS, keep_frac=0.50)
        if len(rets) >= 30:
            eq = (1.0 + pd.Series(rets)).cumprod() * INITIAL
            m = calc_metrics(eq, trades=[], initial_cash=INITIAL, bars_per_year=252)
            results_orig[cid] = m

    for cid in ENHANCED_CHAINS:
        rets = pipeline_chain(cid, stocks, pre_enhanced, common, ENHANCED_CHAINS, keep_frac=0.50)
        if len(rets) >= 30:
            eq = (1.0 + pd.Series(rets)).cumprod() * INITIAL
            m = calc_metrics(eq, trades=[], initial_cash=INITIAL, bars_per_year=252)
            results_enhanced[cid] = m

    print(f"\n    ═══ Original 8 chains ═══")
    print(f"    {'Chain':<20} {'Sharpe':>7} {'AnnRet':>7} {'MaxDD':>7} {'Calmar':>7}")
    for cid in sorted(results_orig, key=lambda c: -results_orig[c]["sharpe"]):
        m = results_orig[cid]
        print(f"    {cid:<20} {m['sharpe']:>7.3f} {m['annual_return']:>6.1%} "
              f"{m['max_drawdown']:>6.1%} {m['calmar']:>7.2f}")

    print(f"\n    ═══ Enhanced chains (with refined financial/sentiment) ═══")
    for cid in sorted(results_enhanced, key=lambda c: -results_enhanced[c]["sharpe"]):
        m = results_enhanced[cid]
        is_new = " ★" if cid not in CHAINS else ""
        print(f"    {cid:<20} {m['sharpe']:>7.3f} {m['annual_return']:>6.1%} "
              f"{m['max_drawdown']:>6.1%} {m['calmar']:>7.2f}{is_new}")

    # ── Composite comparison ───────────────────────
    # Build composite return series
    orig_rets = {}
    for cid in CHAINS:
        rets = pipeline_chain(cid, stocks, pre, common, CHAINS, keep_frac=0.50)
        if len(rets) >= 30:
            orig_rets[cid] = pd.Series(rets)

    enh_rets = {}
    for cid in ENHANCED_CHAINS:
        rets = pipeline_chain(cid, stocks, pre_enhanced, common, ENHANCED_CHAINS, keep_frac=0.50)
        if len(rets) >= 30:
            enh_rets[cid] = pd.Series(rets)

    if orig_rets:
        df_orig = pd.DataFrame(orig_rets)
        comp_orig = df_orig.mean(axis=1)
        eq_orig = (1.0 + comp_orig.fillna(0)).cumprod() * INITIAL
        m_orig = calc_metrics(eq_orig, trades=[], initial_cash=INITIAL, bars_per_year=252)

    if enh_rets:
        df_enh = pd.DataFrame(enh_rets)
        comp_enh = df_enh.mean(axis=1)
        eq_enh = (1.0 + comp_enh.fillna(0)).cumprod() * INITIAL
        m_enh = calc_metrics(eq_enh, trades=[], initial_cash=INITIAL, bars_per_year=252)

    print(f"\n    ═══ Composite (等权全部链) ═══")
    if 'm_orig' in dir():
        print(f"    Original:   Sharpe={m_orig['sharpe']:.3f}  "
              f"AnnRet={m_orig['annual_return']:.1%}  MaxDD={m_orig['max_drawdown']:.1%}  "
              f"Calmar={m_orig['calmar']:.2f}")
    if 'm_enh' in dir():
        print(f"    Enhanced:   Sharpe={m_enh['sharpe']:.3f}  "
              f"AnnRet={m_enh['annual_return']:.1%}  MaxDD={m_enh['max_drawdown']:.1%}  "
              f"Calmar={m_enh['calmar']:.2f}")
        if 'm_orig' in dir():
            ds = m_enh["sharpe"] - m_orig["sharpe"]
            print(f"    Δ Sharpe: {ds:+.3f}")

    print()


if __name__ == "__main__":
    main()
