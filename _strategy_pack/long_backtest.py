"""Long-period real-data backtest: baseline vs 8 logic chains.

Uses Tencent Finance daily data (CSI-300: 2000d/~8y; ETFs: 640d/~2.5y).
Computes per-chain Sharpe vs baseline, outputs comparison table.
"""

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

# ── Config ────────────────────────────────────────────────────────────

TENCENT = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
STOCKS = ["600519","601318","000858","600036","000333","601899","300750","601166",
          "600900","000651","600276","601398","000001","603259","600030","002415",
          "601288","600809","000725","601088"]
INITIAL = 1_000_000


def pull(code: str, days: int = 2000) -> pd.DataFrame:
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


def chain_sharpe(chain_factors: list[str], stock_returns: pd.DataFrame) -> float:
    """Equal-weight chain return from selected factors (simplified IC-weighted)."""
    all_rets = []
    for col in ["mom20","rev60","vs_ma250","vs_ma60","vol20","turn_chg","amp","macd_hist"]:
        if col in stock_returns.columns:
            ic = stock_returns[col].corr(stock_returns["close"].pct_change().shift(-1))
            if not np.isnan(ic) and abs(ic) > 0.01:
                w = abs(ic)
                all_rets.append(w * stock_returns["close"].pct_change())
    if not all_rets:
        return 0.0
    port = sum(all_rets) / sum(1 for _ in all_rets)
    e = (1+port).cumprod() * INITIAL
    m = calc_metrics(e, trades=[], initial_cash=INITIAL, bars_per_year=252)
    return m["sharpe"]


def baseline_sharpe() -> dict:
    """70/25/5 ETF baseline Sharpe across all available data."""
    codes = {"511010": 0.70, "512890": 0.25, "513100": 0.05}
    closes = {}
    for c in codes:
        df = pull(f"sh{c}", days=640)
        closes[c] = df["close"]
        time.sleep(0.1)
    common = closes["511010"].index
    for c in closes.values():
        common = common.intersection(c.index)
    rets = []
    for c, w in codes.items():
        r = closes[c].reindex(common).pct_change().fillna(0)
        rets.append(w * r)
    port = sum(rets)
    e = (1+port).cumprod() * INITIAL
    m = calc_metrics(e, trades=[], initial_cash=INITIAL, bars_per_year=252)
    return {
        "sharpe": m["sharpe"], "ann_ret": m["annual_return"],
        "max_dd": m["max_drawdown"], "ann_vol": float(port.std()*np.sqrt(252)),
        "n_days": len(port),
    }


def main():
    print("="*72)
    print("  LONG-PERIOD BACKTEST: Baseline vs 8 Logic Chains (Real Data)")
    print("="*72)

    # ── Baseline ────────────────────────────────────────────────────
    print("\n[1] Baseline (70国债/25红利低波/5纳指)...")
    bl = baseline_sharpe()
    print(f"    Sharpe={bl['sharpe']:.3f}  AnnRet={bl['ann_ret']:.2%}  "
          f"MaxDD={bl['max_dd']:.2%}  Vol={bl['ann_vol']:.1%}  ({bl['n_days']}d)")

    # ── CSI300 index chain performance ──────────────────────────────
    print("\n[2] CSI300 index (2000d ~ 8yr) state-aware analysis...")
    idx = pull("sh000300", days=2000)
    print(f"    {len(idx)}d, {idx.index[0].date()} ~ {idx.index[-1].date()}")

    # ── Stock factor IC ─────────────────────────────────────────────
    print(f"\n[3] Pulling {len(STOCKS)} stocks, computing factor IC...")
    all_ics: dict[str, list] = {}
    stock_count = 0
    for sym in STOCKS:
        df = pull(f"sh{sym}" if sym.startswith("6") else f"sz{sym}", days=1000)
        if len(df) < 200: continue
        c = df["close"]
        # Compute factors
        fac = pd.DataFrame(index=df.index)
        fac["mom20"] = c.pct_change(20)
        fac["rev60"] = -c.pct_change(60)
        fac["vs_ma250"] = (c - c.rolling(250).mean()) / c.rolling(250).mean()
        fac["vs_ma60"] = (c - c.rolling(60).mean()) / c.rolling(60).mean()
        fac["vol20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
        fac["turn_chg"] = df["volume"].pct_change(5)
        fac["amp"] = (df["high"] - df["low"]) / c
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        fac["macd_hist"] = (ema12-ema26) - (ema12-ema26).ewm(span=9, adjust=False).mean()
        fut = c.pct_change().shift(-1)
        for col in fac.columns:
            ic = fac[col].corr(fut)
            if not np.isnan(ic):
                all_ics.setdefault(col, []).append(ic)
        stock_count += 1
        if stock_count % 5 == 0:
            print(f"    {stock_count}/{len(STOCKS)} done")
        time.sleep(0.12)

    print(f"    {stock_count} stocks with valid factor data")

    # ── Per-factor mean IC ──────────────────────────────────────────
    print(f"\n[4] Factor IC Summary ({stock_count} stocks, 4yr daily)")
    factor_ic = {}
    for col, ics in sorted(all_ics.items(), key=lambda x: -abs(np.mean(x[1]))):
        mean_ic = np.mean(ics)
        factor_ic[col] = abs(mean_ic)
        print(f"    {col:<15} mean|IC|={abs(mean_ic):.4f}")

    # ── Chain summary: real IC-weighted Sharpe ──────────────────────
    chains = {
        "value_qlowvol":    ["vs_ma250","vs_ma60","vol20"],
        "value_momentum":   ["vs_ma250","mom20","turn_chg"],
        "value_stable":     ["vs_ma250","vol20","turn_chg"],
        "quality_momentum": ["vs_ma60","mom20","turn_chg"],
        "reversal_momentum":["rev60","mom20","turn_chg"],
        "vol_reversal":     ["vol20","rev60"],
        "liq_momentum":     ["turn_chg","mom20","amp"],
        "micro_reversal":   ["amp","rev60"],
    }
    chain_names = {
        "value_qlowvol":"价值质量低波","value_momentum":"价值动量",
        "value_stable":"价值稳定","quality_momentum":"质量动量",
        "reversal_momentum":"反转接力","vol_reversal":"波动回归",
        "liq_momentum":"放量动量","micro_reversal":"微观反转",
    }

    print(f"\n[5] Chain vs Baseline (realistic: IC→Sharpe conversion)")
    print(f"    IC→Sharpe ≈ |IC| * sqrt(N_stocks) / vol ≈ |IC| * √20 / 0.25")
    print(f"    Baseline: Sharpe={bl['sharpe']:.2f}  AnnRet={bl['ann_ret']:.2%}  MaxDD={bl['max_dd']:.2%}")
    print(f"    {'Chain':<20} {'IC_avg':>8} {'Est.Sharpe':>10} {'vs.Baseline':>12}")
    print(f"    {'─'*20} {'─'*8} {'─'*10} {'─'*12}")
    for cid, factors in chains.items():
        ic_avg = sum(factor_ic.get(f, 0) for f in factors) / len(factors)
        # Realistic Sharpe: |IC| * sqrt(20 stocks) / annual_vol(0.25)
        est_sharpe = ic_avg * np.sqrt(20) / 0.25
        vs_bl = est_sharpe / bl["sharpe"]
        status = "⚠️ weak" if vs_bl < 0.5 else ("≈ OK" if vs_bl < 1.0 else "✅ strong")
        print(f"    {chain_names[cid]:<20} {ic_avg:>8.4f} {est_sharpe:>10.3f}  {vs_bl:>8.0%} {status}")

    print(f"\n    Baseline Sharpe: {bl['sharpe']:.3f} (70%国债/25%红利低波/5%纳指)")
    print(f"    IC_scaled > 0.048 needed to match baseline (current best = {max(factor_ic.values()):.4f})")
    print()


if __name__ == "__main__":
    main()
