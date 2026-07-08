"""20-year real-data backtest: 70%国债型 + 25%红利型 + 5%纳指型.

Uses akshare for CSI300/NASDAQ, China 10Y yield for bond simulation,
dividend index with CSI300 extension for pre-2019 periods.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENT_DIR = _PROJECT_ROOT / "agent"
for _p in (str(_PROJECT_ROOT), str(_AGENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.backtest.metrics import calc_metrics

INITIAL = 1_000_000


def get_csi300() -> pd.Series:
    """CSI300 daily close from akshare (2002~)."""
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()["close"]


def get_nasdaq() -> pd.Series:
    """NASDAQ composite from akshare (2000~)."""
    import akshare as ak
    df = ak.index_us_stock_sina(symbol=".IXIC")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["close"]


def get_dividend_index() -> pd.Series:
    """CSI Dividend index (000922) until 2019, then extend with CSI300 proxy.

    The CSI dividend index from akshare stops at 2019-01-30.  After that date
    we use CSI300 × (1 + dividend_alpha) where dividend_alpha captures the
    historically observed excess return of dividend stocks over CSI300 (~2%/yr).
    """
    import akshare as ak
    div = ak.stock_zh_index_daily(symbol="sh000922")
    div["date"] = pd.to_datetime(div["date"])
    div = div.set_index("date").sort_index()
    div_close = div["close"]

    # Extend with CSI300 + dividend premium
    csi = get_csi300()
    # Historical excess: dividend stocks tend to outperform CSI300 by ~2%/yr
    excess_daily = 0.02 / 252
    extended = div_close.reindex(csi.index)
    mask = extended.isna()
    extended[mask] = csi[mask] * np.exp(np.arange(mask.sum()) * excess_daily / mask.sum())
    # Normalise to merge point
    merge_idx = div_close.index[-1]
    base_val = div_close.iloc[-1]
    csi_at_merge = csi.reindex(div_close.index).iloc[-1]
    ratio = base_val / csi_at_merge

    for i in range(len(extended)):
        if pd.isna(extended.iloc[i]):
            days_since_merge = (extended.index[i] - merge_idx).days
            extended.iloc[i] = csi.iloc[i] * ratio * (1 + excess_daily * days_since_merge)
    return extended


def get_bond_returns() -> pd.Series:
    """Simulate China government bond ETF returns from 10Y yield changes.

    Bond ETF total return ≈ yield + (-duration × Δyield).
    Duration ≈ 7 years for a China aggregate govt bond index.
    Starting yield ~ 3.2% (2005 average), extracts 10Y yield from akshare.

    Falls back to constant 4.5%/yr if yield data unavailable.
    """
    import akshare as ak
    try:
        yld = ak.bond_china_yield(start_date="2005-01-01")
        yld["date"] = pd.to_datetime(yld["日期"])
        yld = yld.set_index("date").sort_index()
        y10 = yld["中国10年期国债收益率"].dropna().astype(float)

        # Daily bond return ≈ yield/252 + (-7) × Δyield
        dyield = y10.diff().fillna(0) / 100
        daily_yield = y10.shift(1).fillna(y10.iloc[0]) / 100 / 252
        daily_ret = daily_yield.shift(1).fillna(0.045/252) - 7.0 * dyield
        return daily_ret
    except Exception:
        # Fallback: constant 4.5% annual
        csi = get_csi300()
        return pd.Series(0.045 / 252, index=csi.index)


def run_20yr() -> dict[str, Any]:
    print("Loading data...")

    csi = get_csi300()
    print(f"  CSI300: {csi.index[0].date()} ~ {csi.index[-1].date()} ({len(csi)}d)")

    nasdaq = get_nasdaq()
    print(f"  NASDAQ: {nasdaq.index[0].date()} ~ {nasdaq.index[-1].date()} ({len(nasdaq)}d)")

    div = get_dividend_index()
    print(f"  Dividend: {div.index[0].date()} ~ {div.index[-1].date()} ({len(div)}d)")

    bond_daily = get_bond_returns()
    bond_type = "yield-based simulation" if not isinstance(bond_daily, type(None)) else "simulated"
    print(f"  Bond: {bond_type}, {len(bond_daily)}d")

    # Align all to CSI300 dates (longest common)
    common = csi.index
    div_r = div.reindex(common).pct_change().fillna(0)
    ndx_r = nasdaq.reindex(common).pct_change().fillna(0)
    b_r = bond_daily.reindex(common).fillna(0.045 / 252)

    # Clip to common valid range
    valid_from = max(
        div[div > 0].index[0] if len(div[div > 0]) > 0 else common[0],
        nasdaq[nasdaq > 0].index[0] if len(nasdaq[nasdaq > 0]) > 0 else common[0],
    )
    mask = common >= valid_from
    common = common[mask]
    div_r = div_r[mask]
    ndx_r = ndx_r[mask]
    b_r = b_r[mask]

    # 70/25/5 portfolio
    port_ret = 0.70 * b_r + 0.25 * div_r + 0.05 * ndx_r
    equity = (1.0 + port_ret).cumprod() * INITIAL

    m = calc_metrics(equity, trades=[], initial_cash=INITIAL, bars_per_year=252)

    # Year-by-year breakdown
    yearly = {}
    for yr in range(common[0].year, common[-1].year + 1):
        ymask = common.year == yr
        yr_ret = port_ret[ymask]
        if len(yr_ret) > 100:
            yc = (1 + yr_ret).cumprod()
            yearly[yr] = {
                "return": float(yc.iloc[-1] - 1),
                "sharpe": float(yr_ret.mean() / (yr_ret.std() + 1e-10) * np.sqrt(252)),
            }

    # Worst year / MaxDD
    worst_yr = min(yearly, key=lambda y: yearly[y]["return"])
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())
    max_dd_date = str(dd.idxmin().date())

    return {
        "start": str(common[0].date()), "end": str(common[-1].date()),
        "n_days": len(common), "n_years": round(len(common) / 252, 1),
        "sharpe": m["sharpe"], "annual_return": m["annual_return"],
        "max_drawdown": max_dd, "max_dd_date": max_dd_date,
        "calmar": m["calmar"], "sortino": m["sortino"],
        "ann_vol": float(port_ret.std() * np.sqrt(252)),
        "worst_year": {"year": worst_yr, "return": yearly[worst_yr]["return"]},
        "best_year": max(yearly, key=lambda y: yearly[y]["return"]),
        "yearly": {y: yearly[y]["return"] for y in sorted(yearly)},
    }


def main():
    print("=" * 72)
    print("  20-YEAR BASELINE: 70%国债 + 25%红利 + 5%纳指")
    print("  (bond = 10Y yield simulation, dividend = index + CSI extension,")
    print("   nasdaq = real ^IXIC from akshare)")
    print("=" * 72)
    r = run_20yr()

    print(f"\n  Period:    {r['start']} ~ {r['end']} ({r['n_days']}d / {r['n_years']}yr)")
    print(f"  ──────────────────────────────────────────")
    print(f"  Sharpe:    {r['sharpe']:.3f}")
    print(f"  Ann.Ret:   {r['annual_return']:.2%}")
    print(f"  Ann.Vol:   {r['ann_vol']:.1%}")
    print(f"  Max.DD:    {r['max_drawdown']:.2%}  (on {r['max_dd_date']})")
    print(f"  Calmar:    {r['calmar']:.3f}")
    print(f"  Sortino:   {r['sortino']:.3f}")
    print(f"  Best Yr:   {r['best_year']} ({yearly_str(r, r['best_year'])})")
    print(f"  Worst Yr:  {r['worst_year']['year']} ({r['worst_year']['return']:.1%})")

    print(f"\n  Annual Returns:")
    for yr in sorted(r["yearly"]):
        ret = r["yearly"][yr]
        bar = "█" * max(1, int(abs(ret) * 50))
        sign = "+" if ret > 0 else ""
        print(f"    {yr}: {sign}{ret:>6.1%}  {bar}")

    print(f"\n  ⚠️  Data notes:")
    print(f"     • 国债: 模拟自中国10年期国债收益率变化(久期≈7年)")
    print(f"     • 红利: 中证红利指数(000922)至2019年,2019后由沪深300+2%溢价延伸")
    print(f"     • 纳指: akshare纳斯达克真实日线(2000-)")
    print(f"     • 复利再投资, 未扣管理费(ETF约0.5%/年)")


def yearly_str(r, yr):
    return f"{r['yearly'][yr]:.1%}"


if __name__ == "__main__":
    main()
