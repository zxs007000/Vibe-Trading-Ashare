"""草木皆兵因子 (草木皆兵 / Panicked, 多因子选股系列之八).

研报:《显著效应、极端收益扭曲决策权重和"草木皆兵"因子》
发布: 2022-12-13

核心逻辑:
  投资者对极端收益(偏离市场)过度反应(守株待兔/草木皆兵心理),
  导致近期"过度惊恐下跌"的股票未来补涨。

计算步骤:
  0. 市场基准 m_t = 中证全指日收益率
  1. 惊恐度 S_t = |r_t − m_t| / (|r_t| + |m_t| + 0.1)   ∈ [0,1)
  2. 日波动率 σ_t = std(全天每分钟收益率)  [分钟频]
  3. 零售比 R_t = 个人交易额 / 总成交额  [需逐笔, 此处用换手率近似]
  4. 衰减惊恐度 S_t^dec = S_t − mean(S_{t-1}, S_{t-2}), 仅保留>0
  5. 加权决策分 = S_t^dec × σ_t × R_t × r_t

  月频合成: 月均(惊恐收益) + 月稳(惊恐波动) 等权
  负向因子: 高因子值(过度惊恐)→未来补涨→因子值低者未来收益高

数据需求: 日频r_t + 中证全指m_t + 分钟频σ_t + 零售比R_t(逐笔/近似)
原始回测: RankIC -8.90%, ICIR -4.54, 多空32.50%, IR 3.92
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["panic_factor", "panic_factor_batch"]


def _daily_sigma(close_minute: np.ndarray) -> float:
    """日内分钟频波动率。"""
    if len(close_minute) < 10:
        return np.nan
    rets = np.diff(np.log(close_minute))
    rets = rets[np.isfinite(rets)]
    if len(rets) < 5:
        return np.nan
    return float(np.std(rets))


def panic_factor(
    daily_bars: pd.DataFrame,
    minute_bars: pd.DataFrame | None = None,
    market_ret: pd.Series | None = None,
) -> pd.Series:
    """计算草木皆兵因子(单股票)。

    Args:
        daily_bars: 日频, 含 close/open/high/low/volume
        minute_bars: 可选, 分钟频(用于σ_t), 无则用日频波动近似
        market_ret: 可选, 中证全指日收益率序列(index对齐daily_bars)

    Returns:
        pd.Series(index=date), 因子值
    """
    close = pd.to_numeric(daily_bars["close"], errors="coerce")
    ret = close.pct_change()

    # 市场基准
    if market_ret is not None:
        m = market_ret.reindex(ret.index)
    else:
        # 退化: 用全市场近似(个股滞后收益)
        m = ret.shift(1)

    # 惊恐度
    dev = (ret - m).abs()
    base = ret.abs() + m.abs() + 0.1
    S = dev / base

    # 日波动率(分钟频优先)
    if minute_bars is not None:
        s = pd.Series(np.log(pd.to_numeric(minute_bars["close"], errors="coerce")),
                      index=minute_bars.index)
        nd = s.index.normalize()
        intraday_ret = s.groupby(nd).diff()
        sigma_daily = intraday_ret.groupby(nd).std().reindex(ret.index)
    else:
        # 退化: 日频波动率
        sigma_daily = ret.rolling(20, min_periods=5).std()

    # 零售比(近似: 高换手+小市值倾向零售参与度高, 用换手率z-score近似)
    if "turnover" in daily_bars.columns:
        R = pd.to_numeric(daily_bars["turnover"], errors="coerce")
    else:
        vol = pd.to_numeric(daily_bars.get("volume", 0), errors="coerce")
        R = (vol / vol.rolling(20, min_periods=5).mean()).clip(0, 3)

    # 衰减惊恐度
    S_dec = S - S.shift(1).rolling(2).mean()
    S_dec = S_dec.where(S_dec > 0, np.nan)

    # 加权决策分
    weighted_score = S_dec * sigma_daily * R * ret

    # 月频合成: 月均+月稳
    window = 20
    mean_f = weighted_score.rolling(window, min_periods=5).mean()
    std_f = weighted_score.rolling(window, min_periods=5).std()
    factor = (mean_f + std_f) / 2
    factor.name = "panic_factor"
    return factor


def panic_factor_batch(
    stocks_daily: dict[str, pd.DataFrame],
    stocks_minute: dict[str, pd.DataFrame] | None = None,
    market_ret: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """批量计算。"""
    out = {}
    for code, daily in stocks_daily.items():
        minute = stocks_minute.get(code) if stocks_minute else None
        out[code] = panic_factor(daily, minute, market_ret)
    return out


if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 10 + np.cumsum(np.random.randn(n) * 0.02)
    open_ = close + np.random.randn(n) * 0.01
    vol = np.random.randint(100000, 500000, n).astype(float)
    daily = pd.DataFrame({"open": open_, "close": close, "volume": vol}, index=idx)
    f = panic_factor(daily)
    print(f"草木皆兵: 有效={f.notna().sum()}/{len(f)}, 均值={f.mean():.6f}")
