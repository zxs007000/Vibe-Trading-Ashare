"""隔夜收益因子 (Guosheng 国盛金工『量价淘金/隔夜因子』研究).

国盛金工研究: 传统隔夜收益(今开/昨收-1)是弱动量, 但经变换后是有效选股因子,
捕捉隔夜信息(业绩/新闻/美股)的定价。本实现取隔夜收益过去 20 日均值作为因子。

数据需求: 日频 open/close(由 5m resample 得到; open=当日首根5m, prev_close=前交易日末根5m)。
原始研究(国盛): 隔夜收益经绝对值/符号处理后 IC 显著。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["overnight_return", "overnight_return_batch"]


def overnight_return(daily_bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """单股票隔夜收益(前收盘价 → 今开盘的跳空)。"""
    open_ = pd.to_numeric(daily_bars["open"], errors="coerce")
    close = pd.to_numeric(daily_bars["close"], errors="coerce")
    # 交易日对齐: 用前一交易日收盘作基准(避免周末跳空算到周一)
    close_clean = close.dropna()
    open_clean = open_.dropna()
    prev_close = close_clean.shift(1)
    on = open_clean / prev_close - 1.0
    factor = on.rolling(window, min_periods=5).mean().reindex(close.index)
    factor.name = "overnight_return"
    return factor


def overnight_return_batch(stocks_daily: dict[str, pd.DataFrame], window: int = 20) -> dict[str, pd.Series]:
    return {c: overnight_return(b, window) for c, b in stocks_daily.items()}
