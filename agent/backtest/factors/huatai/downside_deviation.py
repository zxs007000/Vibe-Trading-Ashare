"""下行波动率因子 (Huatai 华泰多因子系列之六：波动率类因子).

华泰定义: 仅对『下行收益』(日收益<0)计算标准差(半方差, downside deviation),
刻画个股的向下风险暴露。高下行波动率个股未来收益更低(负向因子, IC<0)。

数据需求: 日频收益率。
原始回测(华泰): 下行波动率 IC 为负, 与特异波动率同源但更聚焦尾部风险。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["downside_deviation", "downside_deviation_batch"]


def downside_deviation(daily_bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """单股票下行波动率(半方差)。"""
    close = pd.to_numeric(daily_bars["close"], errors="coerce")
    ret = close.pct_change()
    down = np.minimum(ret, 0.0)
    dd = down.rolling(window, min_periods=10).apply(
        lambda x: float(np.sqrt(np.nanmean(np.square(x)))), raw=True)
    dd.name = "downside_deviation"
    return dd


def downside_deviation_batch(stocks_daily: dict[str, pd.DataFrame],
                             window: int = 20) -> dict[str, pd.Series]:
    return {c: downside_deviation(b, window) for c, b in stocks_daily.items()}
