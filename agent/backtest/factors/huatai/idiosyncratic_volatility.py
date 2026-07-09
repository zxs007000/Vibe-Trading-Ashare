"""特异波动率因子 (Huatai 华泰多因子系列之六：波动率类因子).

华泰定义: 对个股日收益率与市场日收益率做时间序列回归(窗口通常 20 交易日),
取回归残差的标准差作为『特异性波动率』(idiosyncratic volatility, IVOL)。
高 IVOL 个股未来收益更低(负向因子, IC<0), 是资产定价中经典的风险溢价异象。

数据需求: 日频收益率 + 市场收益率(截面均值代理)。
原始回测(华泰): 特异波动率 IC 显著为负, 多空组合稳定。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["idiosyncratic_volatility", "idiosyncratic_volatility_batch"]


def idiosyncratic_volatility(daily_bars: pd.DataFrame, market_ret: pd.Series | None = None,
                             window: int = 20) -> pd.Series:
    """单股票特异波动率。

    Args:
        daily_bars: 日频, 含 close
        market_ret: 市场日收益率序列(对齐 daily_bars.index); 缺省用个股滞后收益退化
        window:     回归窗口(交易日)
    Returns:
        pd.Series(index=date), 残差标准差(滚动)
    """
    close = pd.to_numeric(daily_bars["close"], errors="coerce")
    ret = close.pct_change()
    if market_ret is not None:
        m = market_ret.reindex(ret.index)
    else:
        m = ret.shift(1)

    # 去掉周末/停牌 NaN(日历日索引会使滚动窗口整体失效), 在交易日上做回归
    df = pd.DataFrame({"ret": ret, "m": m}).dropna()
    xs = df["m"].to_numpy(dtype=float)
    ys = df["ret"].to_numpy(dtype=float)
    n = len(xs)
    out = np.full(n, np.nan)
    for i in range(window, n):
        xw = xs[i - window:i]
        yw = ys[i - window:i]
        xb = xw - xw.mean()
        yb = yw - yw.mean()
        denom = xb.dot(xb)
        if denom < 1e-12:
            continue
        beta = xb.dot(yb) / denom
        resid = yw - beta * xw
        out[i] = float(np.std(resid))
    s = pd.Series(out, index=df.index).reindex(ret.index)
    s.name = "idiosyncratic_volatility"
    return s


def idiosyncratic_volatility_batch(stocks_daily: dict[str, pd.DataFrame],
                                   market_ret: pd.Series | None = None,
                                   window: int = 20) -> dict[str, pd.Series]:
    return {c: idiosyncratic_volatility(b, market_ret, window) for c, b in stocks_daily.items()}
