"""Phase 1: Market State Gated Backtest

三层闸门架构：
  L1 — 链选择 (CHAIN_STATE_PREFERENCE): strong_bear 关 momentum 链
  L2 — 信号缩放: avoid态仓位 20%, neutral 60%, preferred 100%
  L3 — 仓位上限: 最强空头环境整体仓位上限

对比 ungated vs gated 的 Sharpe / MaxDD / Calmar。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── 路径 ──
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENT_DIR = _PROJECT_ROOT / "agent"
for _p in (str(_PROJECT_ROOT), str(_AGENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.backtest.metrics import calc_metrics

INITIAL = 1_000_000

# ══════════════════════════════════════════════════════════════════════
# Data fetching
# ══════════════════════════════════════════════════════════════════════

def _pull_tencent(code: str, days=1500) -> pd.DataFrame | None:
    """腾讯行情 API (fallback)."""
    import requests
    try:
        u = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},qfq"
        r = requests.get(u, timeout=10)
        d = r.json()["data"][code]
        k = d.get("qfqday", d.get("day", []))
        k = [x[:6] for x in k]
        df = pd.DataFrame(k, columns=["date", "open", "close", "high", "low", "volume"])
        for c in ["open", "close", "high", "low", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["return"] = df["close"].pct_change()
        return df
    except Exception:
        return None


def pull(code: str, days=1500) -> pd.DataFrame | None:
    """stock_worm tencent K线（已修复前缀兼容）."""
    try:
        from stcok_worm import tencent as sw_tx
        rows = sw_tx.get_kline(code)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["return"] = df["close"].pct_change()
        return df
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# 8 Factors (same as chain_pipeline_v2.py)
# ══════════════════════════════════════════════════════════════════════

def f_value(c: pd.Series) -> pd.Series:
    return (-c.pct_change(252) + (c - c.rolling(250).mean()) / c.rolling(250).mean()).fillna(0)

def f_momentum(c: pd.Series) -> pd.Series:
    h52 = c.rolling(252).max()
    return (c.pct_change(20) + (c - h52) / h52.replace(0, 1)).fillna(0)

def f_quality(c: pd.Series, v: pd.Series) -> pd.Series:
    v60 = v.rolling(60).mean()
    dv = -np.log(v60 / v60.shift(60).replace(0, 1))
    std60 = c.pct_change().rolling(60).std()
    return (dv - std60).fillna(0)

def f_volatility(c: pd.Series) -> pd.Series:
    r = c.pct_change()
    skew = r.rolling(60).skew()
    vol20 = r.rolling(20).std() * np.sqrt(252)
    return (-skew - vol20 * 0.1).fillna(0)

def f_liquidity(c: pd.Series, v: pd.Series) -> pd.Series:
    r = c.pct_change().abs()
    illiq = (r / (c * v + 1)).rolling(21).mean()
    v_break = v / v.rolling(20).mean() - 1
    return (-illiq + v_break).fillna(0)

def f_reversal(c: pd.Series) -> pd.Series:
    return (-c.pct_change(5) - 0.5 * c.pct_change(60)).fillna(0)

def f_volume(c: pd.Series, v: pd.Series) -> pd.Series:
    vma = v.rolling(20).mean()
    return (v.pct_change(5) + (v - vma) / vma.replace(0, 1)).fillna(0)

def f_micro(c: pd.Series, h: pd.Series, l: pd.Series) -> pd.Series:
    return ((h - l) / c + (c - l) / (h - l + 1e-10)).fillna(0)


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

FACTORS = {
    "value": lambda df: f_value(df["close"]),
    "momentum": lambda df: f_momentum(df["close"]),
    "quality": lambda df: f_quality(df["close"], df["volume"]),
    "volatility": lambda df: f_volatility(df["close"]),
    "liquidity": lambda df: f_liquidity(df["close"], df["volume"]),
    "reversal": lambda df: f_reversal(df["close"]),
    "volume": lambda df: f_volume(df["close"], df["volume"]),
    "microstructure": lambda df: f_micro(df["close"], df["high"], df["low"]),
}


# ══════════════════════════════════════════════════════════════════════
# Market State Gate — L1/L2/L3
# ══════════════════════════════════════════════════════════════════════

# L1: 链级别 preference (来自 market_state.py CHAIN_STATE_PREFERENCE)
CHAIN_STATE_PREFERENCE: dict[str, dict[str, list[str]]] = {
    "value_qlowvol":    {"preferred": ["strong_bull", "grind_up", "tight_range"],
                         "avoid":      ["strong_bear", "wide_range"]},
    "value_momentum":   {"preferred": ["strong_bull", "grind_up", "bounce"],
                         "avoid":      ["strong_bear", "tight_range", "wide_range"]},
    "value_stable":     {"preferred": ["grind_up", "tight_range", "strong_bull"],
                         "avoid":      ["strong_bear", "wide_range"]},
    "quality_momentum": {"preferred": ["strong_bull", "grind_up", "bounce"],
                         "avoid":      ["strong_bear", "tight_range", "pullback"]},
    "reversal_momentum":{"preferred": ["bounce", "grind_down", "wide_range"],
                         "avoid":      ["strong_bull", "strong_bear"]},
    "vol_reversal":     {"preferred": ["wide_range", "grind_down", "bounce"],
                         "avoid":      ["strong_bull", "strong_bear", "tight_range"]},
    "liq_momentum":     {"preferred": ["bounce", "strong_bull", "grind_up"],
                         "avoid":      ["tight_range", "grind_down", "pullback"]},
    "micro_reversal":   {"preferred": [],
                         "avoid":      ["strong_bull", "strong_bear", "grind_up", "grind_down",
                                        "pullback", "bounce", "tight_range", "wide_range"]},
}

# L2/L3: 仓位缩放 (apply to ALL chains, on top of L1 filtering)
STATE_POSITION_CAP: dict[str, float] = {
    "strong_bull":  1.0,   # 满仓 — 最强趋势
    "grind_up":     0.8,   # 80% — 温和上涨
    "bounce":       0.6,   # 60% — 反弹修复
    "tight_range":  0.4,   # 40% — 窄幅盘整
    "pullback":     0.3,   # 30% — 冲高回落
    "wide_range":   0.3,   # 30% — 宽幅震荡
    "grind_down":   0.2,   # 20% — 震荡下行
    "strong_bear":  0.1,   # 10% — 单边下跌
}


def classify_market_state_simple(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """简化版 8 态分类（无需 market_state.py 完整依赖）.

    返回 DataFrame，index=日期，columns=[state, label_zh]。
    基于 MA60/MA250 + ret20 + vol20 + MACD 连续天数判定。
    """
    ma60 = close.rolling(60).mean()
    ma250 = close.rolling(250).mean()
    ret5 = close.pct_change(5) * 100
    ret20 = close.pct_change(20) * 100
    r = close.pct_change()
    vol20 = r.rolling(20).std() * np.sqrt(252) * 100

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    signal = dif.ewm(span=9, adjust=False).mean()
    macd_positive = dif > signal
    macd_consec = macd_positive.astype(int).groupby(
        (~macd_positive).cumsum()
    ).cumsum()
    macd_consec_neg = (~macd_positive).astype(int).groupby(
        (macd_positive).cumsum()
    ).cumsum()

    results = []
    for i in range(250, len(close)):
        dt = close.index[i]
        c = close.iloc[i]
        m60 = ma60.iloc[i]
        m250 = ma250.iloc[i]
        r5 = ret5.iloc[i]
        r20 = ret20.iloc[i]
        v20 = vol20.iloc[i]
        mc = macd_consec.iloc[i]

        # 优先匹配
        if (pd.notna(m60) and pd.notna(m250) and m60 > m250
                and c > m60 and r20 > 3 and mc >= 15):
            state = "strong_bull"
        elif (pd.notna(m60) and pd.notna(m250) and m60 < m250
                and c < m60 and r20 < -3 and macd_consec_neg.iloc[i] >= 15):
            state = "strong_bear"
        elif (pd.notna(m60) and pd.notna(m250) and m60 > m250
                and c < m60):
            # 检查 5 天前是否在 MA60 上方
            if i >= 5 and close.iloc[i - 5] > ma60.iloc[i - 5]:
                state = "pullback"
            else:
                state = "grind_down"  # fallback
        elif (pd.notna(m60) and pd.notna(m250) and m60 < m250
                and c > m60 and r5 > 0):
            state = "bounce"
        elif (pd.notna(m60) and pd.notna(m250) and m60 > m250
                and c > m60 and 0 < r20 <= 3):
            state = "grind_up"
        elif (pd.notna(m60) and pd.notna(m250) and m60 < m250
                and c < m60 and -3 < r20 <= 0):
            state = "grind_down"
        elif (pd.notna(m250) and abs(c - m250) / m250 < 0.05
                and v20 > 20):
            state = "wide_range"
        else:
            state = "tight_range"

        results.append({"date": dt, "state": state})

    df = pd.DataFrame(results).set_index("date")
    if df.empty:
        return df
    # 最短持续 5 天（平滑切换）
    df["state"] = _smooth_state(df["state"], min_duration=5)
    return df


def _smooth_state(states: pd.Series, min_duration: int = 5) -> pd.Series:
    """强制每种状态至少持续 min_duration 天."""
    if len(states) < 2:
        return states
    result = states.copy()
    run_start = 0
    for i in range(1, len(states)):
        if states.iloc[i] != states.iloc[i - 1]:
            run_len = i - run_start
            if run_len < min_duration:
                # 太短，回填前一个状态
                result.iloc[run_start:i] = states.iloc[i - 1]
            run_start = i
    return result


# ══════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════

def pipeline_chain(
    cid: str, stocks: dict, pre: dict,
    common: pd.DatetimeIndex, keep_frac: float = 0.50,
) -> list:
    """原始管道（无 gate），返回 daily return list."""
    stages = CHAINS[cid]
    rets = []
    for i, d in enumerate(common):
        if i == len(common) - 1:
            break
        nd = common[i + 1]
        pool = set(stocks.keys())
        for stage in stages:
            scores = {s: pre[s][stage].get(d, np.nan) for s in pool}
            scores = {s: v for s, v in scores.items() if not np.isnan(v)}
            if len(scores) < 3:
                break
            keep = max(2, int(len(scores) * keep_frac))
            pool = {s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:keep]}
        if len(pool) >= 2:
            nxt = [stocks[s]["return"].get(nd, 0) for s in pool]
            rets.append(np.mean(nxt))
    return rets


def pipeline_chain_gated(
    cid: str, stocks: dict, pre: dict,
    common: pd.DatetimeIndex, states: pd.DataFrame,
    keep_frac: float = 0.50,
) -> list:
    """带市场状态闸门的管道。

    L1: 链级 preference — avoid 态跳过该链（返回当日 0 收益）
    L2: 仓位缩放 — 按 STATE_POSITION_CAP 缩放每日收益
    """
    stages = CHAINS[cid]
    prefs = CHAIN_STATE_PREFERENCE.get(cid, {})
    avoid_states = set(prefs.get("avoid", []))

    rets = []
    state_history = []
    for i, d in enumerate(common):
        if i == len(common) - 1:
            break
        nd = common[i + 1]

        # 获取当日市场状态
        state_row = states.loc[states.index <= d]
        if state_row.empty:
            state_today = "tight_range"  # 默认
        else:
            state_today = state_row.iloc[-1]["state"]

        # L1: avoid 态跳过
        if state_today in avoid_states:
            rets.append(0.0)
            state_history.append((d, state_today))
            continue

        # 管道筛选
        pool = set(stocks.keys())
        for stage in stages:
            scores = {s: pre[s][stage].get(d, np.nan) for s in pool}
            scores = {s: v for s, v in scores.items() if not np.isnan(v)}
            if len(scores) < 3:
                break
            keep = max(2, int(len(scores) * keep_frac))
            pool = {s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:keep]}

        if len(pool) >= 2:
            nxt = [stocks[s]["return"].get(nd, 0) for s in pool]
            raw_ret = np.mean(nxt)
            # L2: 仓位缩放
            cap = STATE_POSITION_CAP.get(state_today, 1.0)
            rets.append(raw_ret * cap)
        else:
            rets.append(0.0)

        state_history.append((d, state_today))

    return rets


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  Phase 1: Market State Gated Backtest")
    print("  对比 ungated vs gated (L1链过滤 + L2仓位缩放)")
    print("=" * 72)

    # ── Stock pool ──────────────────────────────────
    SYMS = [
        "600519", "000858", "601318", "600036", "000333", "601899", "300750", "601166",
        "600900", "000651", "600276", "601398", "000001", "603259", "600030", "002415",
        "601288", "600809", "000725", "601088", "601012", "002714", "000002", "600887",
        "601857", "600028", "601688", "300059", "600585", "600309", "600436", "002594",
        "601225", "603288", "002304", "000568", "601066", "600104", "000776", "300498",
    ]
    print(f"\n[1] Loading {len(SYMS)} stocks...")
    stocks = {}
    for i, s in enumerate(SYMS):
        df = pull(s)
        if df is not None and len(df) > 500:
            stocks[s] = df
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(SYMS)}, {len(stocks)} valid")
        time.sleep(0.05)
    print(f"    Got {len(stocks)} stocks")

    # ── Common date range ────────────────────────────
    common = stocks[list(stocks.keys())[0]].index
    for d in stocks.values():
        common = common.intersection(d.index)
    common = common[-600:]
    print(f"    Date range: {common[0].date()} ~ {common[-1].date()} ({len(common)} days)")

    # ── Precompute factors ───────────────────────────
    print(f"\n[2] Computing 8 factors...")
    pre = {}
    for sym, df in stocks.items():
        pf = pd.DataFrame(index=df.index)
        for th, fn in FACTORS.items():
            pf[th] = fn(df)
        pre[sym] = pf.reindex(common)
    print("    Done.")

    # ── CSI300 + Market State ────────────────────────
    print(f"\n[3] CSI300 data + market state classification...")
    idx = _pull_tencent("sh000300")
    if idx is None:
        print("    ERROR: cannot fetch CSI300")
        return

    idx_r = idx["return"].reindex(common).dropna()
    idx_e = (1.0 + idx_r).cumprod() * INITIAL
    idx_m = calc_metrics(idx_e, trades=[], initial_cash=INITIAL, bars_per_year=252)

    # Market state classification
    idx_close = idx["close"].reindex(common).dropna()
    states = classify_market_state_simple(idx_close)
    print(f"    States classified: {len(states)} days")
    state_counts = states["state"].value_counts()
    for s, c in state_counts.items():
        print(f"      {s}: {c} days ({STATE_POSITION_CAP.get(s, 0):.0%})")

    # ── Run ungated ──────────────────────────────────
    print(f"\n[4] Running ungated chains...")
    ungated = {}
    for cid in CHAINS:
        rets = pipeline_chain(cid, stocks, pre, common, keep_frac=0.50)
        if len(rets) < 30:
            continue
        rs = pd.Series(rets)
        eq = (1.0 + rs).cumprod() * INITIAL
        m = calc_metrics(eq, trades=[], initial_cash=INITIAL, bars_per_year=252)
        ungated[cid] = {
            "sharpe": m["sharpe"], "ann_ret": m["annual_return"],
            "max_dd": m["max_drawdown"], "calmar": m["calmar"],
            "n": len(rets), "ret_series": rs,
        }

    # ── Run gated ────────────────────────────────────
    print(f"\n[5] Running gated chains (L1+L2)...")
    gated = {}
    for cid in CHAINS:
        rets = pipeline_chain_gated(cid, stocks, pre, common, states, keep_frac=0.50)
        if len(rets) < 30:
            continue
        rs = pd.Series(rets)
        eq = (1.0 + rs).cumprod() * INITIAL
        m = calc_metrics(eq, trades=[], initial_cash=INITIAL, bars_per_year=252)
        gated[cid] = {
            "sharpe": m["sharpe"], "ann_ret": m["annual_return"],
            "max_dd": m["max_drawdown"], "calmar": m["calmar"],
            "n": len(rets), "ret_series": rs,
        }

    # ── Composite (equal-weight all chains) ──────────
    print(f"\n[6] Composite (equal-weight all chains)...")
    if ungated:
        # 对齐所有链的 daily return
        comp_u = pd.DataFrame({cid: d["ret_series"] for cid, d in ungated.items()})
        comp_u_mean = comp_u.mean(axis=1)
        eq_u = (1.0 + comp_u_mean.fillna(0)).cumprod() * INITIAL
        m_u = calc_metrics(eq_u, trades=[], initial_cash=INITIAL, bars_per_year=252)

    if gated:
        comp_g = pd.DataFrame({cid: d["ret_series"] for cid, d in gated.items()})
        comp_g_mean = comp_g.mean(axis=1)
        eq_g = (1.0 + comp_g_mean.fillna(0)).cumprod() * INITIAL
        m_g = calc_metrics(eq_g, trades=[], initial_cash=INITIAL, bars_per_year=252)

    # ── Print Report ─────────────────────────────────
    print(f"\n{'='*72}")
    print("  RESULTS")
    print(f"{'='*72}")
    print(f"\n  CSI300: Sharpe={idx_m['sharpe']:.3f}  "
          f"AnnRet={idx_m['annual_return']:.1%}  MaxDD={idx_m['max_drawdown']:.1%}")

    print(f"\n  ═══ Per-Chain Comparison ═══")
    print(f"  {'Chain':<20} {'Ungated':>15} {'Gated':>15} {'Delta':>10}")
    print(f"  {'':20} {'Sharpe':>7} {'MaxDD':>7} {'Sharpe':>7} {'MaxDD':>7} {'Sharpe':>7}")

    all_chains = sorted(set(list(ungated.keys()) + list(gated.keys())))
    for cid in all_chains:
        u = ungated.get(cid, {})
        g = gated.get(cid, {})
        us = u.get("sharpe", float("nan"))
        ud = u.get("max_dd", float("nan"))
        gs = g.get("sharpe", float("nan"))
        gd = g.get("max_dd", float("nan"))
        ds = gs - us if not np.isnan(us) and not np.isnan(gs) else float("nan")
        print(f"  {cid:<20} {us:>7.3f} {ud:>6.1%} {gs:>7.3f} {gd:>6.1%} "
              f"{ds:>+7.3f}")

    # ── Composite Summary ─────────────────────────────
    print(f"\n  ═══ Composite (等权全部链) ═══")
    if 'm_u' in dir():
        print(f"  Ungated:    Sharpe={m_u['sharpe']:.3f}  "
              f"AnnRet={m_u['annual_return']:.1%}  MaxDD={m_u['max_drawdown']:.1%}  "
              f"Calmar={m_u['calmar']:.2f}")
    if 'm_g' in dir():
        print(f"  Gated:      Sharpe={m_g['sharpe']:.3f}  "
              f"AnnRet={m_g['annual_return']:.1%}  MaxDD={m_g['max_drawdown']:.1%}  "
              f"Calmar={m_g['calmar']:.2f}")
        if 'm_u' in dir():
            ds = m_g["sharpe"] - m_u["sharpe"]
            dd = m_g["max_drawdown"] - m_u["max_drawdown"]
            dc = m_g["calmar"] - m_u["calmar"]
            print(f"  Δ (Gated - Ungated): Sharpe {ds:+.3f}  "
                  f"MaxDD {dd:+.1%}  Calmar {dc:+.2f}")

    # ── State exposure ─────────────────────────────────
    if gated:
        print(f"\n  ═══ Gate Activity ═══")
        gate_days = 0
        for cid, g in gated.items():
            rets = g["ret_series"]
            zeros = (rets == 0).sum()
            total = len(rets)
            gate_days = max(gate_days, zeros)
            if zeros > 0:
                print(f"    {cid}: {zeros}/{total} days gated ({zeros/total:.0%})")

    print(f"\n  ═══ State Distribution ═══")
    for s, c in state_counts.items():
        cap = STATE_POSITION_CAP.get(s, 1.0)
        print(f"    {s:15s}: {c:4d}d cap={cap:.0%}")

    print()


if __name__ == "__main__":
    main()
