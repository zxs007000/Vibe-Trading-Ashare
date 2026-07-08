"""Baseline Portfolio — 70%国债ETF + 25%低波红利ETF + 5%纳指ETF.

A benchmark strategy with ZERO factor dependency.  If any logic chain
cannot beat this baseline, it should not be deployed.

ETFs:
  511010 — 国债ETF (China govt bond)
  512890 — 红利低波ETF (CSI dividend low-vol)
  513100 — 纳指100ETF (NASDAQ-100)

Method: fixed-weight rebalancing, no timing, no factor signals.
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

# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════

WEIGHTS = {"511010": 0.70, "512890": 0.25, "513100": 0.05}
NAMES   = {"511010": "国债ETF", "512890": "红利低波ETF", "513100": "纳指ETF"}
_TENCENT = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_INITIAL = 1_000_000


def pull_etf(code: str, days: int = 600) -> pd.Series:
    """Pull ETF close price from Tencent Finance."""
    url = f"{_TENCENT}?param=sh{code},day,,,{days},qfq"
    r = requests.get(url, timeout=10)
    d = r.json()["data"][f"sh{code}"]
    k = d.get("qfqday", d.get("day", []))
    k = [x[:6] for x in k]
    df = pd.DataFrame(k, columns=["date", "open", "close", "high", "low", "volume"])
    for c in ["close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()["close"]


def run() -> dict[str, Any]:
    """Run baseline backtest and return metrics."""
    closes = {}
    for code in WEIGHTS:
        closes[code] = pull_etf(code)
        time.sleep(0.1)

    common = closes["511010"].index
    for c in closes.values():
        common = common.intersection(c.index)

    rets = []
    for code, w in WEIGHTS.items():
        r = closes[code].reindex(common).pct_change().fillna(0.0)
        rets.append(w * r)

    port_ret = pd.Series(
        sum(r for r in rets).values, index=common, name="daily_return"
    )
    equity = (1.0 + port_ret).cumprod() * _INITIAL

    m = calc_metrics(equity, trades=[], initial_cash=_INITIAL, bars_per_year=252)

    return {
        "sharpe": m["sharpe"],
        "annual_return": m["annual_return"],
        "max_drawdown": m["max_drawdown"],
        "calmar": m["calmar"],
        "sortino": m["sortino"],
        "total_return": m["total_return"],
        "ann_vol": float(port_ret.std() * np.sqrt(252)),
        "n_days": len(port_ret),
        "start_date": str(common[0].date()),
        "end_date": str(common[-1].date()),
        "composition": WEIGHTS,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  BASELINE: 70%国债 + 25%红利低波 + 5%纳指")
    print("=" * 60)
    r = run()
    print(f"  Period:  {r['start_date']} ~ {r['end_date']} ({r['n_days']}d)")
    print(f"  Sharpe:  {r['sharpe']:.3f}")
    print(f"  AnnRet:  {r['annual_return']:.2%}")
    print(f"  AnnVol:  {r['ann_vol']:.1%}")
    print(f"  MaxDD:   {r['max_drawdown']:.2%}")
    print(f"  Calmar:  {r['calmar']:.3f}")
    print(f"  Sortino: {r['sortino']:.3f}")
    print(f"  Composition: 国债{r['composition']['511010']:.0%} "
          f"红利低波{r['composition']['512890']:.0%} "
          f"纳指{r['composition']['513100']:.0%}")
    print()
    print("  This is the PASS/FAIL line for any logic chain.")
    print("  If your chain Sharpe < this, the chain adds no value over")
    print("  a zero-thinking ETF allocation.")
    print()
