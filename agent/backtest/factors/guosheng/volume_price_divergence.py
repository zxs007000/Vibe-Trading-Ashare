"""量价背离因子 (Guosheng 国盛金工『量价淘金』系列).

国盛『量价淘金』核心思想: 同一交易行为(量价关系)可从多视角刻画。
本实现取日收益与日成交量的滚动相关性作为『量价配合/背离』因子:
  - 高相关 = 放量确认涨跌(趋势健康)
  - 低/负相关 = 价量背离(放量不涨或缩量涨, 警示)

数据需求: 日频 close + volume。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["volume_price_divergence", "volume_price_divergence_batch"]


def volume_price_divergence(daily_bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """单股票量价滚动相关性。"""
    close = pd.to_numeric(daily_bars["close"], errors="coerce")
    vol = pd.to_numeric(daily_bars.get("volume", 0), errors="coerce")
    ret = close.pct_change()
    corr = ret.rolling(window, min_periods=10).corr(vol)
    corr.name = "volume_price_divergence"
    return corr


def volume_price_divergence_batch(stocks_daily: dict[str, pd.DataFrame], window: int = 20) -> dict[str, pd.Series]:
    return {c: volume_price_divergence(b, window) for c, b in stocks_daily.items()}
