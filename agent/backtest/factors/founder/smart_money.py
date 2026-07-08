"""聪明钱因子 (Smart Money, 聆听高频世界的声音系列三).

研报:《跟踪聪明钱,从分钟行情数据到选股因子》
发布: 2016-07-08, 魏建榕/高子剑

核心逻辑:
  聪明钱(机构/大户)在分钟级别用更少成交量推动更大价格变动。
  通过 S = |R| / √V 识别聪明钱交易时段,取成交量前20%。
  聪明钱交易的 VWAP 与全量 VWAP 的比值 Q 反映聪明钱情绪:
    Q > 1 → 聪明钱在高位买入 → 看跌
    Q < 1 → 聪明钱在低位买入 → 看涨

改进版 S3 (2020): S = |R| / ln(V), IC=-0.050, IR=3.74

数据需求: 分钟频 OHLCV
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["smart_money", "smart_money_batch"]


def _daily_smart_money(close: np.ndarray, vol: np.ndarray, beta: float = 0.5) -> float:
    """单日聪明钱因子 Q。

    Args:
        close: 分钟收盘价序列
        vol:   分钟成交量序列
        beta:  S = |R| / V^beta 的指数, 0.5=原始, 0.1=改进, ln(V)=S3

    Returns:
        Q 因子值, NaN 若数据不足。
    """
    n = len(close)
    if n < 20:
        return np.nan

    # 分钟收益率
    rets = np.diff(np.log(close))
    vols = vol[1:].astype(float)

    # 过滤无效值
    mask = (vols > 0) & np.isfinite(rets)
    rets = rets[mask]; vols = vols[mask]
    if len(rets) < 15:
        return np.nan

    # 聪明度指标 S
    if beta < 0:  # S3: ln(V)
        s = np.abs(rets) / np.log(vols + 1)
    else:
        s = np.abs(rets) / np.power(vols, beta)

    # 按S排序,取成交量累积前20%为聪明钱
    order = np.argsort(-s)
    cum_vol = np.cumsum(vols[order])
    total_vol = cum_vol[-1]
    if total_vol < 1e-10:
        return np.nan
    threshold = 0.2 * total_vol
    # 找到累计成交量首次达到20%的位置,这些就是聪明钱交易
    smart_count = np.searchsorted(cum_vol, threshold) + 1
    smart_count = min(smart_count, len(order))
    smart_idx = order[:smart_count]

    # 聪明钱VWAP vs 全量VWAP
    # VWAP = Σ(价格×量) / Σ(量), 用close近似价格
    prices = close[1:][mask]
    smart_vwap = np.sum(prices[smart_idx] * vols[smart_idx]) / np.sum(vols[smart_idx])
    total_vwap = np.sum(prices * vols) / np.sum(vols)

    if total_vwap < 1e-10:
        return np.nan
    return float(smart_vwap / total_vwap)


def smart_money(
    minute_bars: pd.DataFrame,
    beta: float = 0.5,
) -> pd.Series:
    """计算聪明钱因子日序列。

    Args:
        minute_bars: 分钟K线, index=DatetimeIndex, 含 close/volume
        beta: 0.5=原始, 0.1=改进, -1=S3(ln版本)

    Returns:
        pd.Series(index=date), Q因子值
    """
    if isinstance(minute_bars.index, pd.DatetimeIndex):
        idx = minute_bars.index
    else:
        idx = pd.to_datetime(minute_bars["date"])

    # 过滤交易时段
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    idx = idx[mask]
    dates = idx.normalize()
    close = pd.to_numeric(minute_bars["close"], errors="coerce").to_numpy()[mask]
    vol = pd.to_numeric(minute_bars["volume"], errors="coerce").to_numpy()[mask]

    results = {}
    df = pd.DataFrame({"close": close, "vol": vol}, index=dates)
    for d, g in df.groupby(level=0):
        results[d] = _daily_smart_money(g["close"].values, g["vol"].values, beta)

    s = pd.Series(results)
    s.name = f"smart_money_beta{beta}"
    return s


def smart_money_batch(
    stocks_minute: dict[str, pd.DataFrame],
    beta: float = 0.5,
    window: int = 20,
) -> dict[str, pd.Series]:
    """批量计算,取过去 window 天均值。"""
    out = {}
    for code, bars in stocks_minute.items():
        daily = smart_money(bars, beta=beta)
        out[code] = daily.rolling(window, min_periods=5).mean()
    return out


if __name__ == "__main__":
    np.random.seed(42)
    # 合成: 聪明钱在低位买入的股票 → Q < 1
    n = 240
    close = np.cumsum(np.random.randn(n) * 0.001) + 10
    vol = np.random.exponential(100000, n)
    # 前20%高S时刻集中在价格低位
    low_idx = np.argsort(close)[:40]
    vol[low_idx] *= 3  # 低位放量
    close[low_idx] *= 0.998  # 低位有收益

    bars = pd.DataFrame(
        {"close": close, "volume": vol},
        index=pd.date_range("2025-01-01 09:30", periods=n, freq="1min")
    )
    q = smart_money(bars, beta=0.5)
    print(f"聪明钱因子冒烟测试:")
    print(f"  Q = {q.iloc[0]:.4f}  (低位放量应 < 1)")

    # 纯随机对比
    vol_r = np.random.exponential(100000, n)
    bars_r = pd.DataFrame(
        {"close": close, "volume": vol_r},
        index=pd.date_range("2025-01-01 09:30", periods=n, freq="1min")
    )
    q_r = smart_money(bars_r, beta=0.5)
    print(f"  随机量 Q = {q_r.iloc[0]:.4f}")
    print(f"  低位放量Q < 随机Q: {q.iloc[0] < q_r.iloc[0]}")
