"""适度冒险因子 (Moderate Risk, 多因子选股系列之一).

研报:《成交量激增时刻蕴含的alpha信息》
发布: 2022-04

核心逻辑:
  成交量激增时刻(放量)后的5分钟("耀眼5分钟")的波动率和收益率,
  如果适度(不极端), 说明投资者理性 → 未来收益高。

计算步骤:
  1. 激增时刻: 分钟成交量差值 > mean + std
  2. 耀眼5分钟: 激增时刻及后4分钟
  3. 耀眼波动率: 耀眼5分钟内分钟收益率的标准差
  4. 适度日耀眼波动率: |日耀眼波动率 - 截面均值|
  5. 月均/月稳耀眼波动率: 过去20天均值/标准差
  6. 月耀眼波动率 = (月均 + 月稳) / 2
  7. 同理用收益率代替波动率 → 月耀眼收益率
  8. 适度冒险 = (月耀眼波动率 + 月耀眼收益率) / 2

数据需求: 分钟频 OHLCV
原始回测: Rank IC -6%~-7%, 多空年化 ~20%
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["moderate_risk", "moderate_risk_batch"]


def _daily_moderate_risk(close: np.ndarray, vol: np.ndarray) -> tuple[float, float]:
    """单日适度冒险因子(耀眼波动率, 耀眼收益率)。

    Returns:
        (daily_vol_factor, daily_ret_factor)
    """
    n = len(close)
    if n < 30:
        return (np.nan, np.nan)

    # 分钟收益率
    rets = np.diff(np.log(close))
    # 成交量差值
    vol_diff = np.diff(vol.astype(float))

    # 激增时刻: vol_diff > mean + std
    mu, sd = np.nanmean(vol_diff), np.nanstd(vol_diff)
    if sd < 1e-10 or np.isnan(sd):
        return (np.nan, np.nan)
    surge_mask = vol_diff > (mu + sd)
    surge_idx = np.where(surge_mask)[0]

    if len(surge_idx) == 0:
        return (0.0, 0.0)  # 无激增, 因子中性

    # 耀眼5分钟: 激增时刻及后4分钟
    dazzling_vol = []  # 耀眼波动率
    dazzling_ret = []  # 耀眼收益率
    for idx in surge_idx:
        end = min(idx + 5, len(rets))
        segment = rets[idx:end]
        if len(segment) >= 2:
            dazzling_vol.append(np.std(segment))
            dazzling_ret.append(np.mean(segment))

    if not dazzling_vol:
        return (0.0, 0.0)

    return (float(np.mean(dazzling_vol)), float(np.mean(dazzling_ret)))


def moderate_risk(minute_bars: pd.DataFrame) -> pd.Series:
    """计算适度冒险因子日序列(耀眼波动率+耀眼收益率等权)。

    Args:
        minute_bars: 分钟K线, index=DatetimeIndex, 含 close/volume

    Returns:
        pd.Series(index=date), 日频因子值
    """
    if isinstance(minute_bars.index, pd.DatetimeIndex):
        idx = minute_bars.index
    else:
        idx = pd.to_datetime(minute_bars["date"])

    # 过滤交易时段: 9:30-11:30, 13:00-15:00
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    idx = idx[mask]

    dates = idx.normalize()
    close = pd.to_numeric(minute_bars["close"], errors="coerce").to_numpy()[mask]
    vol = pd.to_numeric(minute_bars["volume"], errors="coerce").to_numpy()[mask]

    df = pd.DataFrame({"close": close, "vol": vol}, index=dates)
    vol_results, ret_results = {}, {}
    for d, g in df.groupby(level=0):
        v, r = _daily_moderate_risk(g["close"].values, g["vol"].values)
        vol_results[d] = v
        ret_results[d] = r

    vol_s = pd.Series(vol_results)
    ret_s = pd.Series(ret_results)

    # 月均 + 月稳 → 等权合成
    window = 20
    vol_mean = vol_s.rolling(window, min_periods=5).mean()
    vol_std = vol_s.rolling(window, min_periods=5).std()
    monthly_vol = (vol_mean + vol_std) / 2

    ret_mean = ret_s.rolling(window, min_periods=5).mean()
    ret_std = ret_s.rolling(window, min_periods=5).std()
    monthly_ret = (ret_mean + ret_std) / 2

    factor = (monthly_vol + monthly_ret) / 2
    factor.name = "moderate_risk"
    return factor


def moderate_risk_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """批量计算。"""
    return {code: moderate_risk(bars) for code, bars in stocks_minute.items()}


if __name__ == "__main__":
    np.random.seed(42)
    n = 240 * 5  # 5天
    close = 10 + np.cumsum(np.random.randn(n) * 0.001)
    vol = np.random.exponential(100000, n).astype(float)
    # 注入激增时刻
    for _ in range(10):
        idx = np.random.randint(1, n-5)
        vol[idx] *= 5

    bars = pd.DataFrame(
        {"close": close, "volume": vol},
        index=pd.date_range("2025-01-01 09:30", periods=n, freq="1min")
    )
    f = moderate_risk(bars)
    print(f"适度冒险因子冒烟测试:")
    print(f"  天数: {len(f)}, 有效: {f.notna().sum()}")
    print(f"  均值: {f.mean():.6f}, 标准差: {f.std():.6f}")
