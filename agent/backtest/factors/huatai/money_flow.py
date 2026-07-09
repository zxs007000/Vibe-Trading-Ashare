"""资金流向因子 (Huatai 华泰多因子系列之七：资金流向因子).

华泰定义: 用日内主动买卖金额刻画『资金流向』(大单净主动买入额 − 净主动卖出额)。
无逐笔/成交额(amount)时, 以 5 分钟 K 线做代理: 每根 bar 的资金流 = 收益率方向 × 成交量
(符号化量能, 类 OBV / Chaikin 资金流), 日度累加后滚动平滑。

逻辑: 价格上涨伴随放量 = 主动资金净流入; 反之为流出。净流入个股后续往往有超额。
(注: 此为 amount 缺失下的 5m 代理版, 非华泰原版逐笔大单划分。)

数据需求: 分钟频 OHLCV(本实现用 5m)。
原始回测(华泰): 资金流向因子 IC 多为正(流入预示上涨), 但具体方向以样本为准。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["money_flow", "money_flow_batch"]


def money_flow(minute_bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """单股票日内资金流(5m 代理)。

    每根 5m bar: mf_t = (close_t − close_{t-1}) × volume_t  (带符号的量能)
    日度累加 → 滚动均值平滑。
    """
    close = pd.to_numeric(minute_bars["close"], errors="coerce")
    vol = pd.to_numeric(minute_bars["volume"], errors="coerce")
    ret = close.diff()
    mf = (ret * vol).fillna(0.0)
    dates = close.index.normalize()
    daily = mf.groupby(dates).sum()
    factor = daily.rolling(window, min_periods=5).mean()
    factor.name = "money_flow"
    return factor


def money_flow_batch(stocks_minute: dict[str, pd.DataFrame], window: int = 20) -> dict[str, pd.Series]:
    return {c: money_flow(b, window) for c, b in stocks_minute.items()}
