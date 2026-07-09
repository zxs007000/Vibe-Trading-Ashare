"""历史分位数因子 (Huatai 华泰多因子系列之十三：历史分位数因子).

华泰定义: 将因子(或价格)当前值在其自身历史分布中的分位数作为因子值,
捕捉『相对历史所处位置』。应用于收盘价时, 当前价处于历史高位 → 后续均值回复 → 负向。

本实现: 当前收盘价在【过去 window 日收盘价】分布中的分位排名
(= 历史中低于当前价的比例), 取值 [0,1], 越高=近期越贵。

数据需求: 日频收盘价。
原始回测(华泰): 历史分位数因子多为负向(高位→回落)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["historical_percentile", "historical_percentile_batch"]


def historical_percentile(daily_bars: pd.DataFrame, window: int = 60) -> pd.Series:
    """单股票价格历史分位数(均值回复信号)。"""
    close = pd.to_numeric(daily_bars["close"], errors="coerce")
    # 去掉周末/停牌 NaN(日历日索引会使滚动窗口整体失效), 在交易日上算分位数
    close_clean = close.dropna()

    def _pr(x):
        if np.isnan(x).any():
            return np.nan
        cur = x[-1]
        return float((x < cur).mean())  # 历史中低于当前价的比例

    hp = close_clean.rolling(window, min_periods=20).apply(_pr, raw=True)
    hp = hp.reindex(close.index)
    hp.name = "historical_percentile"
    return hp


def historical_percentile_batch(stocks_daily: dict[str, pd.DataFrame],
                                window: int = 60) -> dict[str, pd.Series]:
    return {c: historical_percentile(b, window) for c, b in stocks_daily.items()}
