"""Phase 3: Volatility Targeting — 目标波动率 15% 仓位缩放。

无外部 API 依赖。基于 Phase 2 的全套因子 + 12 条链，
对比 ungated vs volatility-targeted 的 Sharpe/MaxDD/Calmar。
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

INITIAL = 1_000_000
TARGET_VOL = 0.15  # 年化目标波动率 15%
VOL_WINDOW = 63     # 63 天滚动窗口 (~1 季度)
MAX_LEVERAGE = 2.0
MIN_LEVERAGE = 0.1


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


def run_pipeline(chains_dict, stocks, pre, common, keep_frac=0.50):
    """返回 {chain_id: pd.Series of daily returns}."""
    results = {}
    for cid, stages in chains_dict.items():
        rets = []
        for i, d in enumerate(common):
            if i == len(common)-1: break
            nd = common[i+1]; pool = set(stocks.keys())
            for stage in stages:
                if stage not in next(iter(pre.values())): break
                scores = {s: pre[s][stage].get(d, np.nan) for s in pool}
                scores = {s: v for s, v in scores.items() if not np.isnan(v)}
                if len(scores) < 3: pool = set(); break
                keep = max(2, int(len(scores) * keep_frac))
                pool = {s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:keep]}
            if len(pool) >= 2:
                rets.append(np.mean([stocks[s]["return"].get(nd, 0) for s in pool]))
        if len(rets) >= 30:
            results[cid] = pd.Series(rets, index=common[:len(rets)])
    return results


def apply_vol_targeting(chain_rets: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """对每条链的日收益序列做波动率缩放。

    pos_weight_t = TARGET_VOL / trailing_annualized_vol_t
    夹在 [MIN_LEVERAGE, MAX_LEVERAGE]，平滑用 ema(0.94)。
    """
    targeted = {}
    for cid, daily_rets in chain_rets.items():
        rolled = daily_rets.rolling(VOL_WINDOW, min_periods=20).std() * np.sqrt(252)
        raw_w = TARGET_VOL / rolled.replace(0, np.nan)
        raw_w = raw_w.clip(MIN_LEVERAGE, MAX_LEVERAGE).fillna(1.0)
        # EMA 平滑避免日间剧烈跳跃
        w = raw_w.ewm(alpha=0.06).mean()
        scaled = daily_rets * w.shift(1).fillna(1.0)
        targeted[cid] = scaled.dropna()
    return targeted


def build_composite(chain_rets: dict[str, pd.Series]) -> pd.Series:
    """等权复合所有链的日收益."""
    df = pd.DataFrame(chain_rets)
    return df.mean(axis=1).dropna()


def equity_curve(rets: pd.Series) -> pd.Series:
    return (1.0 + rets).cumprod() * INITIAL


def calc_stats(rets: pd.Series) -> dict:
    eq = (1.0 + rets).cumprod() * INITIAL
    return calc_metrics(eq, trades=[], initial_cash=INITIAL, bars_per_year=252)


# ════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  Phase 3: Volatility Targeting (target=15% ann)")
    print("=" * 72)

    SYMS = [
        "600519","000858","601318","600036","000333","601899","300750","601166",
        "600900","000651","600276","601398","000001","603259","600030","002415",
        "601288","600809","000725","601088","601012","002714","000002","600887",
        "601857","600028","601688","300059","600585","600309","600436","002594",
        "601225","603288","002304","000568","601066","600104","000776","300498",
    ]

    # 1. Pull stocks
    print(f"\n[1] Loading {len(SYMS)} stocks...")
    stocks = {}
    for i, s in enumerate(SYMS):
        df = pull(s)
        if df is not None and len(df) > 500:
            stocks[s] = df
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{len(SYMS)}, {len(stocks)} valid")
        time.sleep(0.08)
    print(f"    Got {len(stocks)} stocks")

    common = stocks[list(stocks.keys())[0]].index
    for d in stocks.values():
        common = common.intersection(d.index)
    common = common[-600:]
    codes = list(stocks.keys())

    # 2. Original 8 factors
    print(f"\n[2] Computing 8 original factors...")
    pre = {}
    for sym, df in stocks.items():
        pf = pd.DataFrame(index=df.index)
        for th, fn in FACTORS_ORIG.items():
            pf[th] = fn(df)
        pre[sym] = pf.reindex(common)

    # 3. Financial data
    print(f"\n[3] Fetching financial data...")
    fin_raw = fetch_fundamentals(codes, use_cache=False, prefer="stock_worm", periods=8)

    roe_panel = pd.DataFrame(index=common, columns=codes, dtype=float)
    for code in codes:
        fdf = fin_raw.get(code)
        if fdf is None or fdf.empty: continue
        fdf = fdf.copy()
        fdf["report_date"] = pd.to_datetime(fdf["report_date"])
        fdf = fdf.sort_values("report_date")
        if "roe" in fdf.columns and code in roe_panel.columns:
            vals = fdf.set_index("report_date")["roe"].dropna()
            roe_panel[code] = vals.reindex(common, method="ffill")

    roe_change_panel = (roe_panel.diff(1) / roe_panel.shift(1).abs().replace(0, np.nan)).clip(-1,1).fillna(0)
    roe_pct_panel = 0.5 * pd.DataFrame(np.ones_like(roe_panel, dtype=float), index=common, columns=codes)
    for i in range(len(common)):
        row = roe_panel.iloc[i].dropna()
        if len(row) < 3: continue
        ranks = row.rank(pct=True)
        for col in codes:
            if col in ranks.index:
                roe_pct_panel.iloc[i, codes.index(col)] = ranks[col]

    # 4. Sentiment
    print(f"\n[4] Computing sentiment factors...")
    from stcok_worm import news as sw_news
    from stcok_worm.sentiment.analyzers.dictionary import DictionaryAnalyzer
    analyzer = DictionaryAnalyzer()

    sentiment_scores = {}
    for code in codes:
        try:
            articles = sw_news.stock_news(code, page_size=20)
            scores = [analyzer.analyze(a.get("title",""))["sentiment"] for a in articles] if articles else []
            sentiment_scores[code] = np.mean(scores) if scores else 0.0
        except Exception:
            sentiment_scores[code] = 0.0
        time.sleep(0.05)

    sent_score_panel = pd.DataFrame(0.0, index=common, columns=codes)
    for code in codes:
        if code in sent_score_panel.columns:
            sent_score_panel[code] = sentiment_scores.get(code, 0.0)

    # 5. Enhanced factors
    print(f"\n[5] Building enhanced factor panels...")
    pre_enhanced = {}
    for sym in codes:
        pf = pre[sym].copy()
        if sym in roe_change_panel.columns: pf["roe_change"] = roe_change_panel[sym]
        if sym in roe_pct_panel.columns: pf["roe_pct"] = roe_pct_panel[sym]
        if sym in sent_score_panel.columns: pf["sent_score"] = sent_score_panel[sym]
        pre_enhanced[sym] = pf

    ENHANCED_CHAINS = dict(CHAINS)
    ENHANCED_CHAINS["roe_quality"] = ["roe_change","roe_pct","quality"]
    ENHANCED_CHAINS["roe_value"] = ["roe_pct","value","volatility"]
    ENHANCED_CHAINS["sent_momentum"] = ["sent_score","momentum","volume"]
    ENHANCED_CHAINS["sent_quality"] = ["sent_score","quality"]

    # 6. Run pipeline (ungated)
    print(f"\n[6] Running pipeline (ungated)...")
    all_rets = run_pipeline(ENHANCED_CHAINS, stocks, pre_enhanced, common)

    # 7. Apply vol targeting
    print(f"\n[7] Applying volatility targeting (target_vol={TARGET_VOL:.0%})...")
    vol_rets = apply_vol_targeting(all_rets)

    # 8. Per-chain comparison
    print(f"\n    ═══ Per-Chain Comparison ═══")
    print(f"    {'Chain':<20} {'Ungated':^25} {'Vol-Targeted':^25} {'Δ':>7}")
    print(f"    {'':20} {'Sharpe':>6} {'AnnRet':>7} {'MaxDD':>7} {'Sharpe':>6} {'AnnRet':>7} {'MaxDD':>7} {'Sharpe':>7}")

    chain_deltas = []
    for cid in sorted(all_rets, key=lambda c: -calc_stats(all_rets[c])["sharpe"]):
        m_u = calc_stats(all_rets[cid])
        m_v = calc_stats(vol_rets.get(cid, pd.Series(dtype=float)))
        ds = m_v["sharpe"] - m_u["sharpe"]
        chain_deltas.append(ds)
        print(f"    {cid:<20} {m_u['sharpe']:>6.3f} {m_u['annual_return']:>6.1%} {m_u['max_drawdown']:>6.1%}  "
              f"{m_v['sharpe']:>6.3f} {m_v['annual_return']:>6.1%} {m_v['max_drawdown']:>6.1%}  "
              f"{ds:>+7.3f}")

    # 9. Composite comparison
    print(f"\n    ═══ Composite (等权全部链) ═══")
    comp_u = build_composite(all_rets)
    comp_v = build_composite(vol_rets)

    m_u = calc_stats(comp_u)
    m_v = calc_stats(comp_v)
    ds = m_v["sharpe"] - m_u["sharpe"]

    print(f"    Ungated:      Sharpe={m_u['sharpe']:.3f}  AnnRet={m_u['annual_return']:.1%}  "
          f"MaxDD={m_u['max_drawdown']:.1%}  Calmar={m_u['calmar']:.2f}")
    print(f"    Vol-Targeted: Sharpe={m_v['sharpe']:.3f}  AnnRet={m_v['annual_return']:.1%}  "
          f"MaxDD={m_v['max_drawdown']:.1%}  Calmar={m_v['calmar']:.2f}")
    print(f"    Δ Sharpe:     {ds:+.3f}")
    print(f"    Mean Δ across {len(chain_deltas)} chains: {np.mean(chain_deltas):+.3f} "
          f"(median: {np.median(chain_deltas):+.3f})")

    # 10. Vol targeting stats
    print(f"\n    ═══ Vol Targeting Diagnostics ═══")
    all_w = []
    for cid, rets in all_rets.items():
        rolled = rets.rolling(VOL_WINDOW, min_periods=20).std() * np.sqrt(252)
        raw_w = (TARGET_VOL / rolled.replace(0, np.nan)).clip(MIN_LEVERAGE, MAX_LEVERAGE)
        w = raw_w.ewm(alpha=0.06).mean()
        all_w.append(w)
    if all_w:
        w_panel = pd.concat(all_w, axis=1)
        avg_w = w_panel.mean(axis=1).dropna()
        print(f"    Avg weight: {avg_w.mean():.2f}  (range: {avg_w.min():.2f}-{avg_w.max():.2f})")
        print(f"    Pct time >1.5x: {100*(avg_w>1.5).mean():.1f}%  "
              f">1.0x: {100*(avg_w>1.0).mean():.1f}%  "
              f"<0.5x: {100*(avg_w<0.5).mean():.1f}%")
        print(f"    Comp trailing vol: {m_u.get('annual_vol',np.nan):.1f}% → "
              f"{m_v.get('annual_vol',np.nan):.1f}% (target: {TARGET_VOL:.0%})")

    print()


if __name__ == "__main__":
    main()
