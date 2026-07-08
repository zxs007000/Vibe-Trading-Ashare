"""云开雾散因子 (Clouds Disperse, 多因子选股系列之五).

研报:《波动率的波动率与投资者模糊性厌恶》
发布: 2022-08-04

核心逻辑:
  波动率的波动率(模糊性)反映投资者不确定性。
  模糊性高时成交金额/成交量与模糊性的关联→反映投资者反应。

计算步骤:
  1. 分钟波动率: vol(t) = std(ret(t-4~t))
  2. 模糊性: ambig(t) = std(vol(t-4~t))
  3. 起雾时刻: ambig > 当日均值
  4. 模糊关联度 = corr(ambig, 分钟成交金额), 月均+月稳
  5. 模糊金额比 = 起雾时刻成交金额均值 / 全部成交金额均值, 月均+月稳
  6. 修正模糊价差 = 模糊金额比 - 模糊数量比, 对<0部分用过去10天std调整
  7. 云开雾散 = (模糊关联度 + 模糊金额比 + 修正模糊价差) / 3

数据需求: 分钟频收盘价+成交金额+成交量
原始回测: RankIC -9.81%, ICIR -4.48, 多空30.89%, IR 3.29
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["clouds_disperse", "clouds_disperse_batch"]


def _rolling_std(arr, w=5):
    """滑动窗口标准差。"""
    from numpy.lib.stride_tricks import sliding_window_view
    if len(arr) < w:
        return np.full(len(arr), np.nan)
    sw = sliding_window_view(arr, w)
    out = np.full(len(arr), np.nan)
    out[w-1:] = np.std(sw, axis=1)
    return out


def _daily_clouds(close, vol, amount):
    """单日云开雾散三个子指标。"""
    n = len(close)
    if n < 30:
        return (np.nan, np.nan, np.nan)

    rets = np.diff(np.log(close))
    # 分钟波动率(5分钟窗口)
    vol5 = _rolling_std(rets, 5)
    # 模糊性 = 波动率的波动率
    ambig = _rolling_std(vol5, 5)

    valid = ~np.isnan(ambig)
    if valid.sum() < 15:
        return (np.nan, np.nan, np.nan)

    ambig_v = ambig[valid]
    amt_v = amount[1:][valid]  # 成交金额对齐rets
    vol_v = vol[1:][valid]

    # 模糊关联度: corr(ambig, amt)
    if len(ambig_v) > 5 and np.std(amt_v) > 0:
        corr_fa = np.corrcoef(ambig_v, amt_v)[0, 1]
        if np.isnan(corr_fa): corr_fa = 0
    else:
        corr_fa = 0

    # 起雾时刻
    mu_ambig = np.mean(ambig_v)
    fog_mask = ambig_v > mu_ambig

    # 模糊金额比
    if fog_mask.sum() > 0 and len(amt_v) > 0:
        amt_ratio = np.mean(amt_v[fog_mask]) / (np.mean(amt_v) + 1e-10)
    else:
        amt_ratio = 0

    # 模糊数量比
    if fog_mask.sum() > 0 and len(vol_v) > 0:
        vol_ratio = np.mean(vol_v[fog_mask]) / (np.mean(vol_v) + 1e-10)
    else:
        vol_ratio = 0

    # 修正模糊价差
    spread = amt_ratio - vol_ratio

    return (float(corr_fa), float(amt_ratio), float(spread))


def clouds_disperse(minute_bars: pd.DataFrame) -> pd.Series:
    """计算云开雾散因子日序列。"""
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
    # 成交金额: 优先用amount列, 没有则用 close*volume 近似
    if "amount" in minute_bars.columns:
        a = pd.to_numeric(minute_bars["amount"], errors="coerce").to_numpy()[mask]
    else:
        a = c * v

    df = pd.DataFrame({"c": c, "v": v, "a": a}, index=dates)
    corr_list, amt_list, spread_list = [], [], []
    for d, g in df.groupby(level=0):
        cr, ar, sp = _daily_clouds(g["c"].values, g["v"].values, g["a"].values)
        corr_list.append((d, cr)); amt_list.append((d, ar)); spread_list.append((d, sp))

    corr_s = pd.Series(dict(corr_list))
    amt_s = pd.Series(dict(amt_list))
    spread_s = pd.Series(dict(spread_list))

    window = 20
    # 模糊关联度: 月均+月稳
    corr_f = (corr_s.rolling(window, min_periods=5).mean() +
              corr_s.rolling(window, min_periods=5).std()) / 2
    # 模糊金额比: 月均+月稳
    amt_f = (amt_s.rolling(window, min_periods=5).mean() +
             amt_s.rolling(window, min_periods=5).std()) / 2
    # 修正模糊价差: 对<0部分用过去10天std调整
    spread_std10 = spread_s.rolling(10, min_periods=3).std()
    spread_adj = spread_s.where(spread_s >= 0, spread_s * spread_std10)
    spread_f = spread_adj.rolling(window, min_periods=5).mean()

    factor = (corr_f + amt_f + spread_f) / 3
    factor.name = "clouds_disperse"
    return factor


def clouds_disperse_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    return {code: clouds_disperse(bars) for code, bars in stocks_minute.items()}


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
    bars = pd.DataFrame({"close": c, "volume": v}, index=idx)
    f = clouds_disperse(bars)
    print(f"云开雾散: 有效值={f.notna().sum()}/{len(f)}, 均值={f.mean():.6f}")
