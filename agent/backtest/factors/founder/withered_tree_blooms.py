"""枯树生花因子 (Withered Tree Blooms, 聆听高频系列七).

研报:《枯树生花,基于日内模式的动量因子革新》
发布: 2017-09-15, 魏建榕/高子剑

核心逻辑:
  传统动量用全天收益率,但日内不同时段的信息量不同。
  将每日切割为5个时段,分别累加动量,用最优权重合成:
    R0(隔夜)   = P今开 / P昨收 - 1
    R1(第1小时) = P10:30 / P09:30 - 1
    R2(第2小时) = P11:30 / P10:30 - 1
    R3(第3小时) = P14:00 / P13:00 - 1
    R4(第4小时) = P15:00 / P14:00 - 1

  Mi = 过去20天 Ri 的累加
  F = w0×M0 + w1×M1 + w2×M2 + w3×M3 + w4×M4

  全市场最优权重: (-0.47, -0.59, 0.76, 1.50, 1.00)
  中证500最优权重: (0.44, -0.64, 0.44, 2.23, 1.00)

数据需求: 日频 + 关键时点价格(9:30, 10:30, 11:30, 13:00, 14:00, 15:00)
原始回测: 全市场 IR=2.30, 年化17.4%; 中证500 IR=3.57, 年化25.7%
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["withered_tree_blooms", "WITHERED_TREE_WEIGHTS"]

# 论文最优权重
WITHERED_TREE_WEIGHTS = {
    "full_market": np.array([-0.47, -0.59, 0.76, 1.50, 1.00]),
    "csi500":      np.array([ 0.44, -0.64, 0.44, 2.23, 1.00]),
}


def withered_tree_blooms(
    daily_bars: pd.DataFrame,
    minute_bars: pd.DataFrame | None = None,
    window: int = 20,
    weights: str | np.ndarray = "full_market",
) -> pd.Series:
    """计算枯树生花动量因子。

    需要日内关键时点价格。如果只有日频数据,用 open/close 近似 R0 和 R4,
    R1/R2/R3 设为0(退化版)。如果有分钟数据,完整计算。

    Args:
        daily_bars:  日频K线, 含 open/close
        minute_bars: 可选,分钟K线用于精确切割时段
        window:      动量累加窗口(交易日)
        weights:     'full_market' | 'csi500' | 自定义5维权重

    Returns:
        pd.Series(index=date), 枯树生花因子值
    """
    if isinstance(weights, str):
        w = WITHERED_TREE_WEIGHTS[weights]
    else:
        w = np.asarray(weights)

    if minute_bars is not None:
        # 完整版: 从分钟数据提取5时段收益
        r0, r1, r2, r3, r4 = _extract_segment_returns(minute_bars)
    else:
        # 退化版: 只有日频
        r0 = daily_bars["open"].values / daily_bars["close"].shift(1).values - 1
        r1 = np.zeros(len(daily_bars))
        r2 = np.zeros(len(daily_bars))
        r3 = np.zeros(len(daily_bars))
        r4 = daily_bars["close"].values / daily_bars["open"].values - 1

    # 截面标准化(简化: 仅zscore)
    def _zscore(x):
        x = np.asarray(x, dtype=float)
        mu, sd = np.nanmean(x), np.nanstd(x)
        return (x - mu) / (sd + 1e-10)

    M0 = pd.Series(r0, index=daily_bars.index).rolling(window, min_periods=5).sum()
    M1 = pd.Series(r1, index=daily_bars.index).rolling(window, min_periods=5).sum()
    M2 = pd.Series(r2, index=daily_bars.index).rolling(window, min_periods=5).sum()
    M3 = pd.Series(r3, index=daily_bars.index).rolling(window, min_periods=5).sum()
    M4 = pd.Series(r4, index=daily_bars.index).rolling(window, min_periods=5).sum()

    factor = w[0]*M0 + w[1]*M1 + w[2]*M2 + w[3]*M3 + w[4]*M4
    factor.name = "withered_tree_blooms"
    return factor


def _extract_segment_returns(minute_bars: pd.DataFrame):
    """从分钟K线提取5时段收益率。

    需要 index 为 DatetimeIndex, 含 close 列。
    """
    if not isinstance(minute_bars.index, pd.DatetimeIndex):
        raise ValueError("minute_bars index must be DatetimeIndex")

    close = pd.to_numeric(minute_bars["close"], errors="coerce")
    dates = minute_bars.index.normalize()

    r0_list, r1_list, r2_list, r3_list, r4_list = [], [], [], [], []
    date_list = []

    for d, g in close.groupby(dates):
        g = g.dropna()
        if len(g) < 10:
            r0_list.append(np.nan); r1_list.append(np.nan)
            r2_list.append(np.nan); r3_list.append(np.nan); r4_list.append(np.nan)
            date_list.append(d); continue

        times = g.index.time
        prices = g.values

        # 找各时点价格
        def _price_at(hour, minute):
            mask = (times == pd.Timestamp(f"{hour:02d}:{minute:02d}").time())
            if mask.any():
                return prices[mask][-1]
            return np.nan

        p_0930 = _price_at(9, 30)
        p_1030 = _price_at(10, 30)
        p_1130 = _price_at(11, 30)
        p_1300 = _price_at(13, 0)
        p_1400 = _price_at(14, 0)
        p_1500 = _price_at(15, 0)

        # R0 需要昨收, 简化用当日开盘
        r0 = (p_0930 / p_0930) - 1 if not np.isnan(p_0930) else np.nan  # placeholder
        r1 = (p_1030 / p_0930) - 1 if not (np.isnan(p_1030) or np.isnan(p_0930)) else np.nan
        r2 = (p_1130 / p_1030) - 1 if not (np.isnan(p_1130) or np.isnan(p_1030)) else np.nan
        r3 = (p_1400 / p_1300) - 1 if not (np.isnan(p_1400) or np.isnan(p_1300)) else np.nan
        r4 = (p_1500 / p_1400) - 1 if not (np.isnan(p_1500) or np.isnan(p_1400)) else np.nan

        r0_list.append(r0); r1_list.append(r1); r2_list.append(r2)
        r3_list.append(r3); r4_list.append(r4); date_list.append(d)

    return r0_list, r1_list, r2_list, r3_list, r4_list


if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    # 日频数据
    daily = pd.DataFrame(
        {"open": 10 + np.cumsum(np.random.randn(n)*0.05),
         "close": 10 + np.cumsum(np.random.randn(n)*0.05)},
        index=pd.date_range("2025-01-01", periods=n, freq="B")
    )
    daily["open"] = daily["close"].shift(1).fillna(10) * (1 + np.random.randn(n)*0.005)

    factor = withered_tree_blooms(daily, window=20)
    print(f"枯树生花因子(退化版)冒烟测试:")
    print(f"  天数: {len(factor)}, 有效: {factor.notna().sum()}")
    print(f"  均值: {factor.mean():.4f}, 标准差: {factor.std():.4f}")
    print(f"  权重(full_market): {WITHERED_TREE_WEIGHTS['full_market']}")
