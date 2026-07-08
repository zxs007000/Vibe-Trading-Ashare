"""激流勇进因子 (Rapids Advance, 多因子选股系列之十九).

研报:《个股交易放量期间的买入强度刻画与"激流勇进"因子构建》
发布: 2024-08-29

核心逻辑:
  放量上涨长期跑输(已验证); 但放量下跌时多空博弈激烈,
  此时"买入力量"更强的股票后续表现更好。
  用放量下跌时段的[成交额占比 − 成交量占比]刻画买方愿付单价(买入强度)。

计算步骤:
  1. 邻域成交量 V_t^nb = sum(volume_{t-4}..volume_t) (5分钟窗口)
  2. 放量: V_t^nb > V_{t-1}^nb ;  下跌: close_t/close_{t-5} - 1 < 0
  3. cond = 放量 AND 下跌
  4. Factor_raw = amount_flxd/amount_total − vol_flxd/vol_total
  5. Factor_demean = |Factor_raw − mean_cross_section|  (均值距离化)
  6. 月频 = rolling(20).mean().shift(1)  (正向因子, IC=+8%)

数据需求: 分钟级OHLCV+成交额(无amount时用close*volume近似)
原始回测: RankIC +8.00%, ICIR 4.30, 多空38.94%, IR 4.30
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["rapids_advance", "rapids_advance_batch"]


def _daily_rapids(close, vol, amount):
    """单日激流勇进原始因子值。"""
    n = len(close)
    if n < 20:
        return np.nan

    # 邻域成交量(5分钟窗口)
    V_nb = pd.Series(vol).rolling(5).sum().to_numpy()
    # 放量: 邻域成交量递增
    vol_up = np.zeros(n, dtype=bool)
    vol_up[1:] = V_nb[1:] > V_nb[:-1]
    # 下跌: 5分钟收益率<0
    ret5 = np.zeros(n)
    ret5[5:] = close[5:] / close[:-5] - 1
    down = ret5 < 0

    # 条件: 放量且下跌 (剔开盘前5分钟和收盘后3分钟)
    cond = vol_up & down & (np.arange(n) >= 5) & (np.arange(n) < n - 3)
    if cond.sum() < 3:
        return 0.0  # 中性

    amount_flxd = np.sum(amount[cond])
    amount_total = np.sum(amount)
    vol_flxd = np.sum(vol[cond])
    vol_total = np.sum(vol)

    if amount_total <= 0 or vol_total <= 0:
        return 0.0

    return float(amount_flxd / amount_total - vol_flxd / vol_total)


def rapids_advance(minute_bars: pd.DataFrame) -> pd.Series:
    """计算激流勇进因子日序列(单股票)。"""
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
    if "amount" in minute_bars.columns:
        a = pd.to_numeric(minute_bars["amount"], errors="coerce").to_numpy()[mask]
    else:
        a = c * v

    df = pd.DataFrame({"c": c, "v": v, "a": a}, index=dates)
    raw_list = []
    for d, g in df.groupby(level=0):
        raw_list.append((d, _daily_rapids(g["c"].values, g["v"].values, g["a"].values)))

    raw_s = pd.Series(dict(raw_list))
    # 月频: 20日均值, shift(1)防未来函数
    factor = raw_s.rolling(20, min_periods=5).mean().shift(1)
    factor.name = "rapids_advance"
    return factor


def rapids_advance_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """批量计算 + 均值距离化(截面z-score绝对值)。"""
    raw = {code: rapids_advance(bars) for code, bars in stocks_minute.items()}
    dates = None
    for s in raw.values():
        if dates is None:
            dates = s.dropna().index
        else:
            dates = dates.intersection(s.dropna().index)

    result = {}
    for d in dates:
        vals = {}
        for code, s in raw.items():
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

    out = {}
    for code, dmap in result.items():
        out[code] = pd.Series(dmap)

    return out


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
    # 注入放量下跌
    for _ in range(20):
        i = np.random.randint(10, n-10)
        v[i] *= 5
        c[i:] *= 0.999
    bars = pd.DataFrame({"close": c, "volume": v}, index=idx)
    f = rapids_advance(bars)
    print(f"激流勇进: 有效={f.notna().sum()}/{len(f)}, 均值={f.mean():.6f}")
