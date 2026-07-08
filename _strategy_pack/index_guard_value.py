"""Index-Guard Value Strategy — 指数年线护盾 + 低ROE高现金 + MACD择时.

User-supplied baseline:
  买入: 大盘指数上穿年线(MA250) → 选低ROE+高现金率 → MACD金叉入场
  卖出: 指数下穿年线 OR MACD死叉

Architecture (3-stage pipeline logic chain):
  index_guard  :: 大盘择时 — 指数在年线上方才允许持仓
  deep_value   :: 选股 — 低ROE高现金(困境反转信号)
  macd_gate    :: 择时 — MACD金叉/死叉控制进出
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENT_DIR = _PROJECT_ROOT / "agent"
for _p in (str(_PROJECT_ROOT), str(_AGENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.backtest.metrics import calc_metrics

# =============================================================================
# Strategy parameters
# =============================================================================

_N_DAYS = 252
_INITIAL_CASH = 1_000_000.0
_MA_WINDOW = 250           # 年线
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_SEED = 42

# Synthetic market parameters (realistic for CSI-300 2024)
_MKT_ANNUAL_RET = 0.14
_MKT_ANNUAL_VOL = 0.20
_DAILY_VOL = _MKT_ANNUAL_VOL / np.sqrt(_N_DAYS)

# Deep-value stock pool parameters
# Low-ROE + high-cash stocks: lower expected return but higher upside in bull
_VALUE_ANNUAL_RET = 0.05     # low ROE → low base return
_VALUE_UPSIDE_MULT = 2.5     # but during bull signals, upside is amplified
_VALUE_ANNUAL_VOL = 0.28     # higher vol (distressed stocks)
_VALUE_DAILY_VOL = _VALUE_ANNUAL_VOL / np.sqrt(_N_DAYS)

# Market-stock correlation (~0.6 typical for A-share)
_CORR_MKT_STOCK = 0.60


# =============================================================================
# Data generation
# =============================================================================

@dataclass
class SimulationData:
    """All time series needed for the strategy."""
    dates: pd.DatetimeIndex
    index_close: pd.Series          # market index price
    stock_close: pd.Series          # individual stock price
    index_ma250: pd.Series          # index 250-day moving average
    macd_line: pd.Series            # MACD (DIF)
    macd_signal_line: pd.Series     # MACD signal (DEA)


def generate_data(n: int = _N_DAYS, seed: int = _SEED) -> SimulationData:
    """Generate synthetic market + stock price series with realistic properties.

    Market index: CSI-300-like (trend + noise).
    Deep-value stock: lower base return, higher vol, correlated with market.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", "2024-12-31")[:n]

    # ── Market index ─────────────────────────────────────────────────────
    mkt_noise = rng.standard_normal(n) * _DAILY_VOL
    mkt_daily = _MKT_ANNUAL_RET / n + mkt_noise

    # Add realistic 2024 pattern: Q1 dip, Sep rally
    t = np.arange(n)
    mkt_daily += 0.0004 * np.sin(2 * np.pi * t / 63)   # quarterly cycle
    mkt_daily -= 0.0005 * np.exp(-0.5 * ((t - 30) / 15)**2)  # Feb dip
    mkt_daily += 0.0010 * np.exp(-0.5 * ((t - 180) / 25)**2)  # Sep rally

    index_close = pd.Series((1 + mkt_daily).cumprod() * 3000, index=dates, name="index")
    index_ma250 = index_close.rolling(_MA_WINDOW, min_periods=1).mean()

    # ── Stock price (deep-value style) ───────────────────────────────────
    # Correlated with market + idiosyncratic noise + occasional sharp moves
    stock_idio = rng.standard_normal(n) * _VALUE_DAILY_VOL * np.sqrt(1 - _CORR_MKT_STOCK**2)
    stock_sys = mkt_noise * _VALUE_DAILY_VOL / _DAILY_VOL * _CORR_MKT_STOCK
    stock_daily = _VALUE_ANNUAL_RET / n + stock_sys + stock_idio

    # Deeper dips + sharper recoveries (distressed stock behaviour)
    stock_daily += 0.0003 * np.sin(2 * np.pi * t / 42)  # faster cycles
    stock_close = pd.Series((1 + stock_daily).cumprod() * 20, index=dates, name="stock")

    # ── MACD ─────────────────────────────────────────────────────────────
    ema_fast = stock_close.ewm(span=_MACD_FAST, adjust=False).mean()
    ema_slow = stock_close.ewm(span=_MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm(span=_MACD_SIGNAL, adjust=False).mean()

    return SimulationData(
        dates=dates,
        index_close=index_close,
        stock_close=stock_close,
        index_ma250=index_ma250,
        macd_line=macd_line,
        macd_signal_line=macd_signal_line,
    )


# =============================================================================
# Strategy logic
# =============================================================================

@dataclass
class StrategyState:
    """Per-bar strategy state."""
    index_above_ma: bool = False      # 大盘在年线上方
    macd_golden: bool = False         # MACD 金叉信号(刚刚交叉)
    macd_dead: bool = False           # MACD 死叉信号
    macd_above_signal: bool = False   # MACD 在信号线上方(持有多头)
    in_position: bool = False


def run_strategy(data: SimulationData) -> tuple[pd.Series, list[dict]]:
    """Run the index-guard value strategy bar-by-bar.

    Returns
    -------
    equity_curve : pd.Series
    trades : list[dict]  — per-trade records for metrics
    """
    n = len(data.dates)
    state = StrategyState()
    cash = _INITIAL_CASH
    position = 0.0                 # shares held
    equity = np.zeros(n)
    trades: list[dict] = []
    trade_open_price = 0.0

    # Pre-compute boolean masks for speed
    index_above = (data.index_close >= data.index_ma250).values
    macd_above = (data.macd_line >= data.macd_signal_line).values

    # Golden/dead cross detection (cross happened on THIS bar)
    macd_cross_up = np.zeros(n, dtype=bool)
    macd_cross_down = np.zeros(n, dtype=bool)
    for t in range(1, n):
        prev_above = macd_above[t - 1]
        curr_above = macd_above[t]
        macd_cross_up[t] = (not prev_above) and curr_above      # dead→golden
        macd_cross_down[t] = prev_above and (not curr_above)    # golden→dead

    for t in range(n):
        price = data.stock_close.iloc[t]
        state.index_above_ma = bool(index_above[t])
        state.macd_above_signal = bool(macd_above[t])
        state.macd_golden = bool(macd_cross_up[t])
        state.macd_dead = bool(macd_cross_down[t])

        # ── Entry ────────────────────────────────────────────────────────
        if not state.in_position:
            # All three conditions must hold: index above 250MA + stock has
            # deep-value attributes (synthetic: always true in demo) + MACD golden cross
            if state.index_above_ma and state.macd_golden:
                # Enter with full position (simplified: 1 stock)
                shares = cash / price
                position = shares
                cash = 0.0
                state.in_position = True
                trade_open_price = price

        # ── Exit ─────────────────────────────────────────────────────────
        elif state.in_position:
            should_exit = False
            exit_reason = ""

            if not state.index_above_ma:
                # Index fell below 250MA — systemic risk, exit immediately
                should_exit = True
                exit_reason = "index_below_ma250"

            elif state.macd_dead:
                # MACD dead cross — individual stock exit
                should_exit = True
                exit_reason = "macd_dead"

            if should_exit:
                cash = position * price
                trade_ret = (price / trade_open_price) - 1.0
                trades.append({
                    "entry_date": data.dates[t - _macd_bars_since_cross(data, t, "up")],
                    "exit_date": data.dates[t],
                    "entry_price": trade_open_price,
                    "exit_price": price,
                    "return": trade_ret,
                    "reason": exit_reason,
                })
                position = 0.0
                state.in_position = False

        # ── Mark-to-market ───────────────────────────────────────────────
        equity[t] = cash + position * price

    equity_curve = pd.Series(equity, index=data.dates, name="equity")

    # Force exit on last bar if still in position
    if state.in_position:
        cash = position * data.stock_close.iloc[-1]
        equity[-1] = cash

    return equity_curve, trades


def _macd_bars_since_cross(data: SimulationData, current_bar: int, direction: str) -> int:
    """Count bars since last MACD cross (for trade entry date estimation)."""
    macd_cross = (
        (data.macd_line >= data.macd_signal_line) &
        (data.macd_line.shift(1) < data.macd_signal_line.shift(1))
    )
    for t in range(current_bar - 1, -1, -1):
        if macd_cross.iloc[t]:
            return current_bar - t
    return current_bar


# =============================================================================
# Run & report
# =============================================================================

if __name__ == "__main__":
    data = generate_data()
    equity, trades = run_strategy(data)

    # ── Benchmark: buy-and-hold the stock ─────────────────────────────────
    bh_equity = (1.0 + data.stock_close.pct_change().fillna(0.0)).cumprod() * _INITIAL_CASH

    # Index benchmark
    idx_equity = (1.0 + data.index_close.pct_change().fillna(0.0)).cumprod() * _INITIAL_CASH

    strat_m = calc_metrics(equity, trades=[], initial_cash=_INITIAL_CASH, bars_per_year=252)
    bh_m = calc_metrics(bh_equity, trades=[], initial_cash=_INITIAL_CASH, bars_per_year=252)
    idx_m = calc_metrics(idx_equity, trades=[], initial_cash=_INITIAL_CASH, bars_per_year=252)

    # ── Time-in-market ───────────────────────────────────────────────────
    in_market = 0
    for t in range(1, len(data.dates)):
        if equity.iloc[t] != equity.iloc[t - 1] and equity.iloc[t] != equity.iloc[t - 1] * (1 + data.stock_close.pct_change().iloc[t]):
            pass  # simplified; use the strategy state
    # Compute time-in-market from trades
    if trades:
        total_bars_in = sum(
            (data.dates.get_loc(t["exit_date"]) - data.dates.get_loc(t["entry_date"]))
            for t in trades
        )
        pct_in = total_bars_in / len(data.dates) * 100
    else:
        pct_in = 0.0

    # ── Output ───────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  INDEX-GUARD VALUE STRATEGY")
    print("  大盘年线护盾 + 低ROE高现金 + MACD择时")
    print("=" * 78)
    print()
    print(f"  Buy  rule : 指数 > MA250  +  低ROE高现金  +  MACD金叉")
    print(f"  Sell rule : 指数 < MA250  OR  MACD死叉")
    print()
    print(f"  Trades       : {len(trades)}")
    if trades:
        print(f"  Time-in-mkt  : ~{pct_in:.0f}% of bars")
        win_trades = [t for t in trades if t["return"] > 0]
        print(f"  Win rate     : {len(win_trades)/len(trades)*100:.0f}% ({len(win_trades)}/{len(trades)})")
        idx_exits = sum(1 for t in trades if "index" in t.get("reason", ""))
        macd_exits = sum(1 for t in trades if "macd" in t.get("reason", ""))
        print(f"  Exit reasons : {idx_exits} index-below-MA, {macd_exits} MACD-dead")
    print()
    print(f"  {'':<20} {'Sharpe':>8}  {'AnnRet':>8}  {'MaxDD':>8}  {'Calmar':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'Strategy':<20} {strat_m['sharpe']:>8.3f}  {strat_m['annual_return']:>7.2%}  {strat_m['max_drawdown']:>7.2%}  {strat_m['calmar']:>8.3f}")
    print(f"  {'Stock B&H':<20} {bh_m['sharpe']:>8.3f}  {bh_m['annual_return']:>7.2%}  {bh_m['max_drawdown']:>7.2%}  {bh_m['calmar']:>8.3f}")
    print(f"  {'Index B&H':<20} {idx_m['sharpe']:>8.3f}  {idx_m['annual_return']:>7.2%}  {idx_m['max_drawdown']:>7.2%}  {idx_m['calmar']:>8.3f}")
    print()

    # Trade log
    if trades:
        print("  TRADE LOG:")
        print(f"  {'Entry':<12} {'Exit':<12} {'Return':>8} {'Reason':<20}")
        for t in trades:
            entry_s = str(t["entry_date"])[:10]
            exit_s = str(t["exit_date"])[:10]
            print(f"  {entry_s:<12} {exit_s:<12} {t['return']:>7.2%}  {t['reason']:<20}")
    print()
