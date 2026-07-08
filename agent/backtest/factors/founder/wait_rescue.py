"""待著而救因子 (Wait and Rescue, 多因子选股系列之十一).

研报:《大单成交后的跟随效应与"待著而救"因子》
发布: 2023-06-12

核心逻辑:
  大单成交后,如果普通投资者大量跟随买入→反应过度→未来回落。
  如果未产生大量跟随→反应不足→未来有超额收益。

计算步骤:
  1. 取当日成交量最大的10个分钟("海量时刻")
  2. 相邻间隔>5分钟才保留为独立"优势时刻"(≤5分钟视为跟随交易剔除)
  3. 对每个优势时刻t,其后5分钟成交量总和 / t时刻成交量 = 跟随系数
  4. 日跟随系数 = 日内所有跟随系数均值
  5. 待著而救 = (过去20天均值 + 标准差)  (负向因子)

数据需求: 分钟频成交量
原始回测: RankIC约-7%, 多空年化~33%
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["wait_rescue", "wait_rescue_batch"]


def _daily_wait_rescue(vol: np.ndarray) -> float:
    """单日待著而救因子(日跟随系数)。"""
    n = len(vol)
    if n < 30:
        return np.nan

    # 成交量最大的10个分钟
    top10 = np.argsort(-vol)[:10]
    top10_sorted = np.sort(top10)

    # 筛选独立优势时刻: 相邻间隔>5分钟
    advantage_times = [top10_sorted[0]]
    for t in top10_sorted[1:]:
        if t - advantage_times[-1] > 5:
            advantage_times.append(t)

    if len(advantage_times) < 2:
        return np.nan

    # 跟随系数: 每个优势时刻后5分钟成交量之和 / 该时刻成交量
    follow_coeffs = []
    for t in advantage_times:
        end = min(t + 6, n)
        follow_vol = np.sum(vol[t + 1:end])
        base_vol = vol[t]
        if base_vol > 0:
            follow_coeffs.append(follow_vol / base_vol)

    if len(follow_coeffs) < 2:
        return np.nan

    return float(np.mean(follow_coeffs))


def wait_rescue(minute_bars: pd.DataFrame) -> pd.Series:
    """计算待著而救因子日序列。"""
    if isinstance(minute_bars.index, pd.DatetimeIndex):
        idx = minute_bars.index
    else:
        idx = pd.to_datetime(minute_bars["date"])
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    dates = idx[mask].normalize()
    vol = pd.to_numeric(minute_bars["volume"], errors="coerce").to_numpy()[mask]

    df = pd.DataFrame({"v": vol}, index=dates)
    daily_list = []
    for d, g in df.groupby(level=0):
        daily_list.append((d, _daily_wait_rescue(g["v"].values)))

    daily_s = pd.Series(dict(daily_list))
    window = 20
    mean_f = daily_s.rolling(window, min_periods=5).mean()
    std_f = daily_s.rolling(window, min_periods=5).std()
    factor = mean_f + std_f  # 注意: 原文是相加不是等权
    factor.name = "wait_rescue"
    return factor


def wait_rescue_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    return {code: wait_rescue(bars) for code, bars in stocks_minute.items()}


if __name__ == "__main__":
    np.random.seed(42)
    n = 240 * 10
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="5min")
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    idx = idx[mask]; n = len(idx)
    v = np.random.exponential(100000, n)
    # 注入大单
    for _ in range(20):
        i = np.random.randint(10, n - 10)
        v[i] *= 8
    bars = pd.DataFrame({"volume": v}, index=idx)
    f = wait_rescue(bars)
    print(f"待著而救: 有效值={f.notna().sum()}/{len(f)}, 均值={f.mean():.4f}")
