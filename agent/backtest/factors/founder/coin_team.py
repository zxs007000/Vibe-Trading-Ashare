"""球队硬币因子 (Coin & Team, 多因子选股系列之四).

研报:《个股动量效应的识别及"球队硬币"因子构建》
发布: 2022-06-11

核心逻辑:
  Moskowitz(2021) "可知性"概念:
  - 硬币型(高可知性): 波动率低+换手率下降 → 预期反转,但实际反转 → ×(-1)
  - 球队型(低可知性): 波动率高+换手率上升 → 预期动量,但实际动量 → 保持

  三维度分解: 日间反转 + 日内反转 + 隔夜反转
  每个维度根据波动率/换手率决定是否翻转方向

计算(日频版):
  日间收益 = close/prev_close - 1
  日内收益 = close/open - 1
  隔夜收益 = open/prev_close - 1

  波动翻转: 过去20天波动率 < 截面均值 → ×(-1)
  换手翻转: 换手率变化 > 市场均值的为球队型(保持), < 为硬币型(×(-1))

  修正日间 = (波动翻转 + 换手翻转) / 2
  修正日内 = 同理
  修正隔夜 = 同理(引入均值距离化)
  球队硬币 = (修正日间 + 修正日内 + 修正隔夜) / 3

数据需求: 日频 OHLCV + 换手率
原始回测: Rank IC -9.67%, ICIR -4.73, 多空年化 39.69%, IR 3.95
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["coin_team", "coin_team_batch"]


def coin_team(
    daily_bars: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """计算球队硬币因子(单只标的, 需截面信息由调用方提供)。

    Args:
        daily_bars: 日频K线, 含 open/close/high/low/volume/turnover(换手率)
        window:     滚动窗口

    Returns:
        pd.Series(index=date), 因子值
    """
    close = pd.to_numeric(daily_bars["close"], errors="coerce")
    open_ = pd.to_numeric(daily_bars["open"], errors="coerce")
    if "turnover" in daily_bars.columns:
        turnover = pd.to_numeric(daily_bars["turnover"], errors="coerce")
    else:
        # 退化: 用 volume/mean(volume) 近似换手率
        vol = pd.to_numeric(daily_bars.get("volume", 0), errors="coerce")
        turnover = vol / vol.rolling(window, min_periods=5).mean()

    # 三维度收益
    prev_close = close.shift(1)
    r_inter = close / prev_close - 1          # 日间
    r_intra = close / open_ - 1               # 日内
    r_overnight = open_ / prev_close - 1      # 隔夜

    # 波动率
    vol_inter = r_inter.rolling(window, min_periods=5).std()
    vol_intra = r_intra.rolling(window, min_periods=5).std()
    vol_overnight = r_overnight.rolling(window, min_periods=5).std()

    # 换手率变化
    turnover_chg = turnover.diff()

    # 反转动量(过去window天累加)
    mom_inter = r_inter.rolling(window, min_periods=5).sum()
    mom_intra = r_intra.rolling(window, min_periods=5).sum()
    mom_overnight = r_overnight.rolling(window, min_periods=5).sum()

    # 波动翻转: 波动率低于均值 → 硬币型 → ×(-1)
    # (截面均值由调用方处理, 这里用自身时序均值近似)
    vol_inter_mean = vol_inter.rolling(window*3, min_periods=window).mean()
    vol_intra_mean = vol_intra.rolling(window*3, min_periods=window).mean()
    vol_overnight_mean = vol_overnight.rolling(window*3, min_periods=window).mean()

    flip_inter_vol = np.where(vol_inter < vol_inter_mean, -1, 1)
    flip_intra_vol = np.where(vol_intra < vol_intra_mean, -1, 1)
    flip_overnight_vol = np.where(vol_overnight < vol_overnight_mean, -1, 1)

    # 换手翻转: 换手率变化低于均值 → 硬币型 → ×(-1)
    turnover_chg_mean = turnover_chg.rolling(window*3, min_periods=window).mean()
    flip_inter_turn = np.where(turnover_chg < turnover_chg_mean, -1, 1)

    # 修正三维度
    modified_inter = mom_inter * (flip_inter_vol + flip_inter_turn) / 2
    modified_intra = mom_intra * flip_intra_vol
    # 隔夜引入均值距离化
    overnight_mean = r_overnight.rolling(window*3, min_periods=window).mean()
    modified_overnight = (mom_overnight * flip_overnight_vol *
                          (1 + np.abs(r_overnight - overnight_mean)))

    factor = (modified_inter + modified_intra + modified_overnight) / 3
    factor = pd.Series(factor, index=daily_bars.index)
    factor.name = "coin_team"
    return factor


def coin_team_batch(
    stocks_daily: dict[str, pd.DataFrame],
    window: int = 20,
) -> dict[str, pd.Series]:
    """批量计算(注意: 完整版需要截面均值, 这里用自身时序近似)。"""
    return {code: coin_team(bars, window) for code, bars in stocks_daily.items()}


if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 10 + np.cumsum(np.random.randn(n) * 0.02)
    open_ = close + np.random.randn(n) * 0.01
    vol = np.random.randint(100000, 500000, n).astype(float)

    bars = pd.DataFrame({"open": open_, "close": close, "volume": vol}, index=idx)
    f = coin_team(bars, window=20)
    print(f"球队硬币因子冒烟测试:")
    print(f"  天数: {len(f)}, 有效: {f.notna().sum()}")
    print(f"  均值: {f.mean():.4f}, 标准差: {f.std():.4f}")
