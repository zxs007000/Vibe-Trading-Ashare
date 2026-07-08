"""飞蛾扑火因子 (Moth to Flame, 多因子选股系列之六).

研报:《个股股价跳跃及其对振幅因子的改进》
发布: 2022-09-22

核心逻辑:
  分钟跳跃度(泰勒展开残项)衡量股价"跳跃"程度。
  跳跃度大→吸引博彩偏好投资者(飞蛾扑火)→反应过度→未来反转。

计算步骤:
  1. 单利收益率: r_simple(t) = P_t/P_{t-1} - 1
  2. 连续复利收益率: r_log(t) = ln(P_t/P_{t-1})
  3. 单复利差: diff(t) = r_simple(t) - r_log(t)  (泰勒一阶残项)
  4. 泰勒二阶残项: resid(t) = 2*diff(t) - r_log(t)²
  5. 日跳跃度 = mean(resid(t))  (日内所有分钟泰勒残项均值)
  6. 月跳跃度 = (过去20天均值 + 标准差) / 2  (负向因子)

数据需求: 分钟频收盘价
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["moth_to_flame", "moth_to_flame_batch"]


def _daily_jump(close: np.ndarray) -> float:
    """单日跳跃度。"""
    n = len(close)
    if n < 30:
        return np.nan

    # 注意: close 可能有0或负值, 先过滤
    close = close[close > 0]
    if len(close) < 30:
        return np.nan

    r_simple = close[1:] / close[:-1] - 1
    r_log = np.log(close[1:] / close[:-1])

    diff = r_simple - r_log
    resid = 2 * diff - r_log ** 2

    return float(np.mean(resid))


def moth_to_flame(minute_bars: pd.DataFrame) -> pd.Series:
    """计算飞蛾扑火因子日序列。"""
    if isinstance(minute_bars.index, pd.DatetimeIndex):
        idx = minute_bars.index
    else:
        idx = pd.to_datetime(minute_bars["date"])
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    dates = idx[mask].normalize()
    close = pd.to_numeric(minute_bars["close"], errors="coerce").to_numpy()[mask]

    df = pd.DataFrame({"c": close}, index=dates)
    jump_list = []
    for d, g in df.groupby(level=0):
        jump_list.append((d, _daily_jump(g["c"].values)))

    jump_s = pd.Series(dict(jump_list))
    window = 20
    mean_f = jump_s.rolling(window, min_periods=5).mean()
    std_f = jump_s.rolling(window, min_periods=5).std()
    factor = (mean_f + std_f) / 2
    factor.name = "moth_to_flame"
    return factor


def moth_to_flame_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    return {code: moth_to_flame(bars) for code, bars in stocks_minute.items()}


if __name__ == "__main__":
    np.random.seed(42)
    n = 240 * 10
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="5min")
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    idx = idx[mask]; n = len(idx)
    c = 10 + np.cumsum(np.random.randn(n) * 0.002)
    # 注入几个跳跃
    for i in [50, 120, 200]:
        c[i:] *= 1.003
    bars = pd.DataFrame({"close": c}, index=idx)
    f = moth_to_flame(bars)
    print(f"飞蛾扑火: 有效值={f.notna().sum()}/{len(f)}, 均值={f.mean():.8f}")
