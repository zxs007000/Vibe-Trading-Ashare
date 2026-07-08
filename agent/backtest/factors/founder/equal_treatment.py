"""一视同仁因子 (Equal Treatment, 多因子选股系列之十八).

研报:《成交量激增与骤降时刻的对称性与"一视同仁"因子构建》
发布: 2024-05-23

核心逻辑:
  放量(激增)和缩量(骤降)都蕴含信息。如果投资者对两者反应不对称→反应过度。

计算步骤:
  1. 成交量差值: ΔV(t) = V(t) - V(t-1)
  2. 激增时刻: ΔV > mean + std
     骤降时刻: ΔV < mean - std
  3. 波动公平度 BDGP = |mean(5min_std(ret)在激增时刻) - mean(5min_std(ret)在骤降时刻)|
  4. 收益公平度 SYGP = |mean(ret在激增时刻) - mean(ret在骤降时刻)|
  5. 日修正收益: ret_bdgp = 日收益 × BDGP; ret_sygp = 日收益 × SYGP
  6. 一视同仁 = (过去20天ret_bdgp均值 + 过去20天ret_sygp均值) / 2  (负向)

数据需求: 分钟频成交量+收盘价
原始回测: RankIC -7.39%, ICIR -4.09, 多空31.36%, IR 3.49
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["equal_treatment", "equal_treatment_batch"]


def _rolling_std(arr, w=5):
    from numpy.lib.stride_tricks import sliding_window_view
    if len(arr) < w:
        return np.full(len(arr), np.nan)
    sw = sliding_window_view(arr, w)
    out = np.full(len(arr), np.nan)
    out[w-1:] = np.std(sw, axis=1)
    return out


def _daily_equal_treatment(close, vol):
    """单日一视同仁: 返回(ret_bdgp, ret_sygp)。"""
    n = len(close)
    if n < 30:
        return (np.nan, np.nan)

    rets = np.diff(np.log(close))
    dvol = np.diff(vol.astype(float))

    mu, sd = np.nanmean(dvol), np.nanstd(dvol)
    if sd < 1e-10:
        return (np.nan, np.nan)

    surge = dvol > (mu + sd)   # 激增时刻
    plunge = dvol < (mu - sd)  # 骤降时刻

    if surge.sum() < 2 or plunge.sum() < 2:
        return (0.0, 0.0)

    # 5分钟滚动波动率
    vol5 = _rolling_std(rets, 5)

    # 波动公平度
    surge_vol = np.nanmean(vol5[surge])
    plunge_vol = np.nanmean(vol5[plunge])
    bdgp = abs(surge_vol - plunge_vol) if not (np.isnan(surge_vol) or np.isnan(plunge_vol)) else 0

    # 收益公平度
    surge_ret = np.mean(rets[surge])
    plunge_ret = np.mean(rets[plunge])
    sygp = abs(surge_ret - plunge_ret)

    # 日收益率
    daily_ret = np.log(close[-1] / close[0])

    return (daily_ret * bdgp, daily_ret * sygp)


def equal_treatment(minute_bars: pd.DataFrame) -> pd.Series:
    """计算一视同仁因子日序列。"""
    if isinstance(minute_bars.index, pd.DatetimeIndex):
        idx = minute_bars.index
    else:
        idx = pd.to_datetime(minute_bars["date"])
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    dates = idx[mask].normalize()
    c = pd.to_numeric(minute_bars["close"], errors="coerce").to_numpy()[mask]
    v = pd.to_numeric(minute_bars["volume"], errors="coerce").to_numpy()[mask]

    df = pd.DataFrame({"c": c, "v": v}, index=dates)
    bdgp_list, sygp_list = [], []
    for d, g in df.groupby(level=0):
        b, s = _daily_equal_treatment(g["c"].values, g["v"].values)
        bdgp_list.append((d, b)); sygp_list.append((d, s))

    bdgp_s = pd.Series(dict(bdgp_list))
    sygp_s = pd.Series(dict(sygp_list))

    window = 20
    bdgp_f = bdgp_s.rolling(window, min_periods=5).mean()
    sygp_f = sygp_s.rolling(window, min_periods=5).mean()

    factor = (bdgp_f + sygp_f) / 2
    factor.name = "equal_treatment"
    return factor


def equal_treatment_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    return {code: equal_treatment(bars) for code, bars in stocks_minute.items()}


if __name__ == "__main__":
    np.random.seed(42)
    n = 240 * 10
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="5min")
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    idx = idx[mask]; n = len(idx)
    c = 10 + np.cumsum(np.random.randn(n) * 0.002)
    v = np.random.exponential(100000, n)
    # 注入激增和骤降
    for _ in range(15):
        i = np.random.randint(1, n - 1)
        v[i] *= 5
    for _ in range(15):
        i = np.random.randint(1, n - 1)
        v[i] *= 0.1
    bars = pd.DataFrame({"close": c, "volume": v}, index=idx)
    f = equal_treatment(bars)
    print(f"一视同仁: 有效值={f.notna().sum()}/{len(f)}, 均值={f.mean():.8f}")
