"""完整潮汐因子 (Complete Tide, 多因子选股系列之二).

研报:《个股成交量的潮汐变化及"潮汐"因子构建》
发布: 2022-05-08

核心逻辑:
  日内成交量的"潮汐"过程: 顶峰(最大邻域成交量)→涨潮(前段最小)→退潮(后段最小)。
  从涨潮到退潮的价格变动速率反映投资者买卖意愿强度。

计算步骤:
  1. 邻域成交量: 第t分钟及前后4分钟(共9分钟)成交量之和
  2. 顶峰时刻 t = argmax(邻域成交量)
     涨潮时刻 m = argmin(邻域成交量[5:t-1]), 价格 Cm
     退潮时刻 n = argmin(邻域成交量[t+1:233]), 价格 Cn
  3. 全潮汐速率 = (Cn - Cm) / Cm / (n - m)
  4. 强弱拆分: Vm < Vn → 涨潮为强势; Vm > Vn → 退潮为强势
     强势半潮汐速率 = (Ct-Cm)/Cm/(t-m) 或 (Cn-Ct)/Ct/(n-t)
  5. 完整潮汐 = (强势半潮汐 + 稳定弱势半潮汐) / 2
     强势半潮汐 = 过去20天均值
     稳定弱势半潮汐 = (过去20天均值 + 标准差) / 2

数据需求: 分钟频成交量+收盘价
原始回测: RankIC -7.90%, ICIR -4.13, 多空27.09%, IR 3.08
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["complete_tide", "complete_tide_batch"]


def _daily_tide(close: np.ndarray, vol: np.ndarray) -> tuple[float, float]:
    """单日潮汐计算。

    Returns:
        (strong_half_tide_rate, weak_half_tide_rate)
    """
    n = len(close)
    if n < 30:
        return (np.nan, np.nan)

    # 邻域成交量: 9分钟窗口求和, 中心对齐
    w = 4  # 前后各4分钟
    nbr_vol = np.full(n, np.nan)
    for i in range(w, n - w):
        nbr_vol[i] = np.sum(vol[i - w:i + w + 1])

    valid = ~np.isnan(nbr_vol)
    if valid.sum() < 20:
        return (np.nan, np.nan)

    # 顶峰时刻(邻域成交量最大)
    t = np.nanargmax(nbr_vol)
    if t < w + 5 or t > n - w - 5:
        return (np.nan, np.nan)

    # 涨潮时刻: 5~t-1 中邻域成交量最小
    if t - 5 < 5:
        return (np.nan, np.nan)
    m = 5 + np.nanargmin(nbr_vol[5:t])
    # 退潮时刻: t+1~233 中邻域成交量最小
    end = min(233, n - w)
    if t + 1 >= end:
        return (np.nan, np.nan)
    n_idx = t + 1 + np.nanargmin(nbr_vol[t + 1:end])

    Cm, Ct, Cn = close[m], close[t], close[n_idx]
    Vm, Vn = nbr_vol[m], nbr_vol[n_idx]

    if Cm <= 0 or Cn <= 0 or Ct <= 0:
        return (np.nan, np.nan)

    # 强弱判断
    if Vm < Vn:
        # 涨潮为强势
        strong_rate = (Ct - Cm) / Cm / (t - m) if t > m else np.nan
        weak_rate = (Cn - Ct) / Ct / (n_idx - t) if n_idx > t else np.nan
    else:
        # 退潮为强势
        strong_rate = (Cn - Ct) / Ct / (n_idx - t) if n_idx > t else np.nan
        weak_rate = (Ct - Cm) / Cm / (t - m) if t > m else np.nan

    return (strong_rate, weak_rate)


def complete_tide(minute_bars: pd.DataFrame) -> pd.Series:
    """计算完整潮汐因子日序列。

    Args:
        minute_bars: 分钟K线, index=DatetimeIndex, 含 close/volume

    Returns:
        pd.Series(index=date), 因子值
    """
    if isinstance(minute_bars.index, pd.DatetimeIndex):
        idx = minute_bars.index
    else:
        idx = pd.to_datetime(minute_bars["date"])
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    dates = idx[mask].normalize()
    close = pd.to_numeric(minute_bars["close"], errors="coerce").to_numpy()[mask]
    vol = pd.to_numeric(minute_bars["volume"], errors="coerce").to_numpy()[mask]

    df = pd.DataFrame({"close": close, "vol": vol}, index=dates)
    strong_list, weak_list = [], []
    for d, g in df.groupby(level=0):
        s, w = _daily_tide(g["close"].values, g["vol"].values)
        strong_list.append((d, s)); weak_list.append((d, w))

    strong_s = pd.Series(dict(strong_list))
    weak_s = pd.Series(dict(weak_list))

    window = 20
    # 强势半潮汐 = 过去20天均值
    strong_factor = strong_s.rolling(window, min_periods=5).mean()
    # 稳定弱势半潮汐 = (均值 + 标准差) / 2
    weak_mean = weak_s.rolling(window, min_periods=5).mean()
    weak_std = weak_s.rolling(window, min_periods=5).std()
    weak_factor = (weak_mean + weak_std) / 2

    factor = (strong_factor + weak_factor) / 2
    factor.name = "complete_tide"
    return factor


def _filter_trading_hours(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """过滤交易时段。"""
    if isinstance(bars.index, pd.DatetimeIndex):
        idx = bars.index
    else:
        idx = pd.to_datetime(bars["date"])
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    return idx[mask]


def complete_tide_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """批量计算。"""
    return {code: complete_tide(bars) for code, bars in stocks_minute.items()}


if __name__ == "__main__":
    np.random.seed(42)
    n = 240 * 10
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="5min")
    # 过滤交易时段
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    idx = idx[mask]
    n = len(idx)
    close = 10 + np.cumsum(np.random.randn(n) * 0.002)
    vol = np.random.exponential(100000, n)
    bars = pd.DataFrame({"close": close, "volume": vol}, index=idx)
    f = complete_tide(bars)
    print(f"完整潮汐: 有效值={f.notna().sum()}/{len(f)}, 均值={f.mean():.6f}")
