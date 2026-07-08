"""勇攀高峰因子 (Scaling Heights, 多因子选股系列之三).

研报:《个股波动率的变动及"勇攀高峰"因子构建》
发布: 2022-05-30

核心逻辑:
  高波动时段的风险补偿(收益波动比与更优波动率的协方差)越高,未来收益越高。

计算步骤:
  1. 更优波动率: σ_opt(t) = [std(t-4~t的20个OHLC价格) / mean(同)]²
  2. 收益波动比: RAR(t) = ret(t) / σ_opt(t)
  3. 异常高波动时段: σ_opt >= mean + std 的分钟
  4. 勇攀高峰 = cov(RAR, σ_opt) 在异常高波动时段 → 月均+月稳等权
  5. 灾后重建 = cov(RAR, σ_opt) 在全天 → 月均+月稳等权 (负向)

数据需求: 分钟频OHLC
原始回测: 勇攀高峰 RankIC 5.62%, ICIR 4.47, 多空19.76%, IR 3.45
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["scaling_heights", "scaling_heights_batch"]


def _daily_scaling_heights(open_, high, low, close) -> tuple[float, float]:
    """单日勇攀高峰+灾后重建。

    Returns:
        (climb_cov, rebuild_cov)  勇攀高峰协方差, 灾后重建协方差
    """
    n = len(close)
    if n < 30:
        return (np.nan, np.nan)

    rets = np.diff(np.log(close))
    # 更优波动率: 5分钟窗口内20个OHLC价格
    w = 5
    sigma_opt = np.full(n, np.nan)
    for i in range(w - 1, n):
        prices = np.stack([open_[i-w+1:i+1], high[i-w+1:i+1],
                           low[i-w+1:i+1], close[i-w+1:i+1]]).ravel()
        mu, sd = np.mean(prices), np.std(prices)
        if mu > 0:
            sigma_opt[i] = (sd / mu) ** 2

    # RAR
    rar = np.full(n, np.nan)
    valid = sigma_opt > 1e-15
    rar[1:][valid[1:]] = rets[valid[1:]] / sigma_opt[1:][valid[1:]]

    # 全天协方差(灾后重建)
    valid_all = valid & np.isfinite(rar)
    if valid_all.sum() < 20:
        return (np.nan, np.nan)
    rebuild_cov = np.cov(rar[valid_all], sigma_opt[valid_all])[0, 1]

    # 异常高波动时段协方差(勇攀高峰)
    mu_s, sd_s = np.nanmean(sigma_opt), np.nanstd(sigma_opt)
    abnormal = valid_all & (sigma_opt >= mu_s + sd_s)
    if abnormal.sum() < 5:
        climb_cov = rebuild_cov  # 退化
    else:
        climb_cov = np.cov(rar[abnormal], sigma_opt[abnormal])[0, 1]

    return (float(climb_cov), float(rebuild_cov))


def scaling_heights(minute_bars: pd.DataFrame) -> pd.Series:
    """计算勇攀高峰因子日序列。"""
    if isinstance(minute_bars.index, pd.DatetimeIndex):
        idx = minute_bars.index
    else:
        idx = pd.to_datetime(minute_bars["date"])
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    dates = idx[mask].normalize()

    o = pd.to_numeric(minute_bars["open"], errors="coerce").to_numpy()[mask]
    h = pd.to_numeric(minute_bars["high"], errors="coerce").to_numpy()[mask]
    l = pd.to_numeric(minute_bars["low"], errors="coerce").to_numpy()[mask]
    c = pd.to_numeric(minute_bars["close"], errors="coerce").to_numpy()[mask]

    df = pd.DataFrame({"o": o, "h": h, "l": l, "c": c}, index=dates)
    climb_list, rebuild_list = [], []
    for d, g in df.groupby(level=0):
        cl, rb = _daily_scaling_heights(g["o"].values, g["h"].values,
                                         g["l"].values, g["c"].values)
        climb_list.append((d, cl)); rebuild_list.append((d, rb))

    climb_s = pd.Series(dict(climb_list))
    rebuild_s = pd.Series(dict(rebuild_list))

    window = 20
    climb_mean = climb_s.rolling(window, min_periods=5).mean()
    climb_std = climb_s.rolling(window, min_periods=5).std()
    factor = (climb_mean + climb_std) / 2
    factor.name = "scaling_heights"
    return factor


def scaling_heights_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    return {code: scaling_heights(bars) for code, bars in stocks_minute.items()}


if __name__ == "__main__":
    np.random.seed(42)
    n = 240 * 10
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="5min")
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    idx = idx[mask]; n = len(idx)
    c = 10 + np.cumsum(np.random.randn(n) * 0.002)
    o = c + np.random.randn(n) * 0.001
    h = np.maximum(o, c) + np.random.rand(n) * 0.005
    l = np.minimum(o, c) - np.random.rand(n) * 0.005
    bars = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)
    f = scaling_heights(bars)
    print(f"勇攀高峰: 有效值={f.notna().sum()}/{len(f)}, 均值={f.mean():.6f}")
