"""滴水穿石因子 (Founder Factor #16, 多因子选股系列之二十四).

研报:《个股日内成交量周期性节奏刻画与"滴水穿石"因子构建》
发布: 2025-12-16

核心逻辑:
  成交量序列在 2-5 分钟周期上的能量占比越高,说明日内交易节奏越规律,
  聪明资金在规律性节奏中布局 → 未来收益越高。

计算步骤:
  1. 剔除集合竞价(开盘1分钟、收盘前3分钟)
  2. IQR 限幅去脉冲: clip(vol, median-3*IQR, median+3*IQR)
  3. 去均值 + Hann 窗
  4. rFFT → 功率谱 P(f) = |FFT|²
  5. band_power = Σ P[48:]  (2-5分钟周期, 240点rFFT的第48-120点)
     total_power = Σ P[1:]  (排除直流分量)
  6. 因子 = band_power / total_power

数据需求: 分钟频成交量(1分钟K线, 每日240根)
原始回测: 6年IC>0.1, 多空年化42%, 多头年化20%
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["drip_water_stone", "drip_water_stone_batch"]


def _daily_factor(vol_minute: np.ndarray) -> float:
    """单日滴水穿石因子值。

    Args:
        vol_minute: 某日分钟成交量序列(长度应接近240)。
                    已剔除集合竞价的分钟。

    Returns:
        2-5分钟周期能量占比, NaN 若数据不足。
    """
    v = np.asarray(vol_minute, dtype=float)
    # 剔除 NaN
    v = v[~np.isnan(v)]
    # 注: 论文用 1 分钟 K 线(每日~240 根); 5m 数据每日仅~44 根,
    # 故放宽门槛(5m 代理版)。1m 数据下 44→236 同样通过。
    if len(v) < 12:
        return np.nan

    # Step 1: IQR 限幅
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    med = np.median(v)
    lower, upper = med - 3 * iqr, med + 3 * iqr
    v = np.clip(v, lower, upper)

    # Step 2: 去均值 + Hann 窗
    v = v - v.mean()
    w = np.hanning(len(v))
    v = v * w

    # Step 3: rFFT + 功率谱
    fft_cf = np.fft.rfft(v)
    p_f = np.square(np.abs(fft_cf))

    # Step 4: 频带能量比(对频率分辨率自适应)
    # 论文原意: 取 2-5 分钟周期能量占比 = 1m 频谱 upper~60% 非直流段。
    # 用「相对频带」映射: band_start = 0.4*(NFFT-1), band_end = NFFT-1,
    # 对 1m(240根)复现 [48,120] 区间; 对 5m(~44根)即「日内快节奏能量占比」代理。
    nfft = len(p_f)
    band_start = max(1, int(round(0.40 * (nfft - 1))))
    band_end = nfft - 1

    total_power = np.sum(p_f[1:])
    if total_power < 1e-20:
        return np.nan

    band_power = np.sum(p_f[band_start:band_end + 1])
    return float(band_power / total_power)


def drip_water_stone(
    minute_bars: pd.DataFrame,
    date_col: str = "date",
    vol_col: str = "volume",
) -> pd.Series:
    """计算滴水穿石因子日序列。

    Args:
        minute_bars: 分钟K线 DataFrame, 至少含日期列和成交量列。
        date_col:    日期列名(或 index 为 datetime)。
        vol_col:     成交量列名。

    Returns:
        pd.Series(index=date), 每日的滴水穿石因子值。
    """
    if isinstance(minute_bars.index, pd.DatetimeIndex):
        dates = minute_bars.index.normalize()
    else:
        dates = pd.to_datetime(minute_bars[date_col]).dt.normalize()

    vol = pd.to_numeric(minute_bars[vol_col], errors="coerce").to_numpy()

    results = {}
    for d, mask in pd.Series(vol, index=dates).groupby(level=0):
        v = mask.values
        # 剔除开盘1分钟和收盘前3分钟
        if len(v) > 4:
            v = v[1:-3]
        results[d] = _daily_factor(v)

    s = pd.Series(results)
    s.name = "drip_water_stone"
    return s


def drip_water_stone_batch(
    stocks_minute: dict[str, pd.DataFrame],
    window: int = 20,
) -> dict[str, pd.Series]:
    """批量计算,取过去 window 天的均值作为月频因子。

    Args:
        stocks_minute: {code: 分钟K线 DataFrame}
        window:        滚动平均窗口(交易日)

    Returns:
        {code: pd.Series} 日频因子经 window 日均值平滑
    """
    out = {}
    for code, bars in stocks_minute.items():
        daily = drip_water_stone(bars)
        out[code] = daily.rolling(window, min_periods=5).mean()
    return out


if __name__ == "__main__":
    # 冒烟测试: 合成分钟数据
    np.random.seed(42)
    days = 30
    minutes = 240
    dates = pd.date_range("2025-01-01", periods=days * minutes, freq="1min")
    # 合成有周期性的成交量
    t = np.arange(minutes)
    base = 100000 + 50000 * np.sin(2 * np.pi * t / 3)  # 3分钟周期
    noise = np.random.randn(days * minutes) * 20000
    vol = np.tile(base, days) + noise
    bars = pd.DataFrame({"volume": vol}, index=dates)

    factor = drip_water_stone(bars)
    print(f"滴水穿石因子冒烟测试:")
    print(f"  天数: {len(factor)}, 有效值: {factor.notna().sum()}")
    print(f"  均值: {factor.mean():.4f}, 标准差: {factor.std():.4f}")
    print(f"  范围: [{factor.min():.4f}, {factor.max():.4f}]")

    # 合成有强3分钟周期的 vs 随机的
    vol_random = np.random.exponential(100000, days * minutes)
    bars_r = pd.DataFrame({"volume": vol_random}, index=dates)
    factor_r = drip_water_stone(bars_r)
    print(f"\n  有周期性(3min): 因子均值={factor.mean():.4f}")
    print(f"  纯随机:         因子均值={factor_r.mean():.4f}")
    print(f"  → 有周期性的应更高: {factor.mean() > factor_r.mean()}")
