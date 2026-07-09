"""Amihud 非流动性因子 (Haitong 海通金工 / Amihud 2002).

海通金工『选股因子系列』(逐笔大单/流动性方向)大量使用非流动性因子。
经典 Amihud(2002): ILLIQ = (1/D) Σ |r_t| / V_t, V_t 为日成交金额。
高非流动性(缺深度)个股有流动性溢价, 未来收益更高(正向因子, IC>0)。

数据需求: 日频收益率 + 成交金额(沙箱 5m 无 amount, 以 volume 代理 V_t)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["amihud_illiquidity", "amihud_illiquidity_batch"]


def amihud_illiquidity(daily_bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """单股票 Amihud 非流动性(volume 代理)。"""
    close = pd.to_numeric(daily_bars["close"], errors="coerce")
    ret = close.pct_change().abs()
    vol = pd.to_numeric(daily_bars.get("volume", 0), errors="coerce").replace(0, np.nan)
    illiq = ret / vol
    factor = illiq.rolling(window, min_periods=5).mean()
    factor.name = "amihud_illiquidity"
    return factor


def amihud_illiquidity_batch(stocks_daily: dict[str, pd.DataFrame], window: int = 20) -> dict[str, pd.Series]:
    return {c: amihud_illiquidity(b, window) for c, b in stocks_daily.items()}
