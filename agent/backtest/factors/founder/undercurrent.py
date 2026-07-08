"""暗流涌动因子 (Undercurrent, 多因子选股系列之二十三).

研报:《个股日内成交量分布特征与日内流动性弹性刻画》
发布: 2025-08-27

核心逻辑:
  (1) 成交量分布熵值: 与市场节奏"不同步但偏离不太大"的股票超额收益更好
  (2) 日内流动性弹性: 成交量激增时价格反应"适度"的股票更好

子因子1: 成交量分布熵值
  1. 日内240分钟等分为48个5分钟区间
  2. rel_t = volume_stock_t / volume_market_t (需全市场量)
  3. p_k = Σ_{t∈区间k} rel_t / Σ rel_t  (k=1..48, Σp=1)
  4. H = -Σ p_k·log2(p_k)  (香农熵)
  5. H_demean = |H − mean_cross_section|

子因子2: 日内流动性弹性
  1. V5_t = mean(volume_{t-5}..volume_{t-1})
  2. 激增时刻: volume_t > 2×V5_t
  3. swing_t = (high_t − low_t)/open_t
  4. 敏感系数 = spike_mean/normal_mean ; 弹性系数 = 1 − 敏感系数
  5. E_demean = |弹性系数 − mean_cross_section|

合成: 暗流涌动 = (成交量熵值 + 流动性弹性) / 2  (负向因子, IC=-7.65%)

数据需求: 分钟级OHLCV + 全市场分钟成交量(rel_t计算)
原始回测: RankIC -7.65%, ICIR -4.44, 多空29.17%, IR 3.49
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["undercurrent", "undercurrent_batch"]


def _daily_entropy(rel_volume: np.ndarray) -> float:
    """单日成交量分布熵(48区间)。"""
    n = len(rel_volume)
    if n < 48:
        return np.nan
    total = np.sum(rel_volume)
    if total <= 0:
        return np.nan

    # 48个区间
    seg = n // 48
    p = []
    for k in range(48):
        start = k * seg
        end = (k + 1) * seg if k < 47 else n
        p_k = np.sum(rel_volume[start:end]) / total
        if p_k > 0:
            p.append(p_k)

    if len(p) < 10:
        return np.nan
    H = -np.sum([pk * np.log2(pk) for pk in p])
    return float(H)


def _daily_elasticity(close, high, low, vol):
    """单日流动性弹性。"""
    n = len(close)
    if n < 20:
        return np.nan

    # 激增时刻
    V5 = pd.Series(vol).rolling(5).mean().to_numpy()
    spike = vol > 2 * V5
    spike[np.isnan(V5)] = False

    swing = (high - low) / close
    swing = np.nan_to_num(swing, nan=0)

    spike_mean = np.mean(swing[spike]) if spike.sum() > 0 else 0
    normal_mean = np.mean(swing[~spike]) if (~spike).sum() > 0 else 0

    if normal_mean < 1e-10:
        return np.nan

    sensitivity = spike_mean / normal_mean
    elasticity = 1 - sensitivity
    return float(elasticity)


def undercurrent(minute_bars: pd.DataFrame, market_volume: np.ndarray | None = None) -> pd.Series:
    """计算暗流涌动因子日序列(单股票)。

    Args:
        minute_bars: 分钟K线, 含 close/high/low/volume
        market_volume: 可选, 全市场分钟成交量总和(用于rel_t). 无则用个股自身近似
    """
    if isinstance(minute_bars.index, pd.DatetimeIndex):
        idx = minute_bars.index
    else:
        idx = pd.to_datetime(minute_bars["date"])
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    dates = idx[mask].normalize()
    c = pd.to_numeric(minute_bars["close"], errors="coerce").to_numpy()[mask]
    h = pd.to_numeric(minute_bars["high"], errors="coerce").to_numpy()[mask]
    l = pd.to_numeric(minute_bars["low"], errors="coerce").to_numpy()[mask]
    v = pd.to_numeric(minute_bars["volume"], errors="coerce").to_numpy()[mask]

    # 相对成交量
    if market_volume is not None and len(market_volume) == len(v):
        rel_v = v / (market_volume + 1e-10)
    else:
        # 退化: 用个股自身相对日内均值
        rel_v = v / (np.mean(v) + 1e-10)

    df = pd.DataFrame({"c": c, "h": h, "l": l, "v": v, "rel": rel_v}, index=dates)

    entropy_list, elastic_list = [], []
    for d, g in df.groupby(level=0):
        H = _daily_entropy(g["rel"].values)
        E = _daily_elasticity(g["c"].values, g["h"].values, g["l"].values, g["v"].values)
        entropy_list.append((d, H)); elastic_list.append((d, E))

    H_s = pd.Series(dict(entropy_list))
    E_s = pd.Series(dict(elastic_list))

    # 月频: (月均+月稳)/2
    window = 20
    H_f = (H_s.rolling(window, min_periods=5).mean() + H_s.rolling(window, min_periods=5).std()) / 2
    E_f = (E_s.rolling(window, min_periods=5).mean() + E_s.rolling(window, min_periods=5).std()) / 2

    factor = (H_f + E_f) / 2
    factor.name = "undercurrent"
    return factor


def undercurrent_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """批量计算 + 均值距离化(截面z-score绝对值)。

    注意: 成交量熵值需要全市场分钟成交量。此处用所有传入股票
    的分钟成交量之和作为 market_volume 近似(需所有股票对齐时间)。
    """
    # 计算全市场分钟成交量(对齐到每个股票的时间索引)
    first_code = list(stocks_minute.keys())[0]
    base_idx = None
    market_sum = None
    for code, bars in stocks_minute.items():
        if isinstance(bars.index, pd.DatetimeIndex):
            bidx = bars.index
        else:
            bidx = pd.to_datetime(bars["date"])
        t = bidx.time
        bmask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
                ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
        v = pd.to_numeric(bars["volume"], errors="coerce").to_numpy()[bmask]
        s = pd.Series(v, index=bidx[bmask])
        if market_sum is None:
            market_sum = s
            base_idx = bidx[bmask]
        else:
            market_sum = market_sum.add(s, fill_value=0)

    out = {}
    for code, bars in stocks_minute.items():
        mv = market_sum.values if market_sum is not None else None
        f = undercurrent(bars, mv)
        out[code] = f

    # 均值距离化
    dates = None
    for s in out.values():
        if dates is None:
            dates = s.dropna().index
        else:
            dates = dates.intersection(s.dropna().index)

    result = {}
    for d in dates:
        vals = {}
        for code, s in out.items():
            v = s.get(d, np.nan)
            if not np.isnan(v):
                vals[code] = v
        if len(vals) < 5:
            continue
        arr = np.array(list(vals.values()))
        mu, sd = arr.mean(), arr.std()
        if sd < 1e-10:
            continue
        for code, v in vals.items():
            result.setdefault(code, {})[d] = abs((v - mu) / sd)

    final = {}
    for code, dmap in result.items():
        final[code] = pd.Series(dmap)

    return final


if __name__ == "__main__":
    np.random.seed(42)
    n = 240 * 10
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="5min")
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    idx = idx[mask]; n = len(idx)
    c = 10 + np.cumsum(np.random.randn(n) * 0.002)
    h = np.maximum(c, c + np.random.rand(n) * 0.005)
    l = np.minimum(c, c - np.random.rand(n) * 0.005)
    v = np.random.exponential(100000, n)
    bars = pd.DataFrame({"close": c, "high": h, "low": l, "volume": v}, index=idx)
    f = undercurrent(bars)
    print(f"暗流涌动: 有效={f.notna().sum()}/{len(f)}, 均值={f.mean():.6f}")
