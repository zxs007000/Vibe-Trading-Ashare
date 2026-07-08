"""结构化反转因子 (Structured Reversal Factor).

实现《高频因子（二）：结构化反转因子》的三个变体，对 bar 频率透明
(日线 freq=9 / 分钟线 freq=0,2,3 均可)：

  - equal  : 等权反转  = Σ log(C_t-i+1 / C_t-i)  (即基础反转的窗口版)
  - time   : 时间加权  = Σ w_i·r_i,  w_i 按半衰期指数衰减(近期 bar 更重)
  - volume : 成交量加权 = Σ w_i·r_i,  w_i ∝ volume_i  (交投越活跃越易反转)

约定：因子值越低 = 近期跌得越多 = 反转信号越强(做多)。
因此在多空组合里，做多因子最低分组(超卖)、做空最高分组(强势)。

数据接口：输入为单只标的的 OHLCV DataFrame(index=datetime, 含 close/volume)。
不负责取数，取数在 verify 脚本里通过 stock_worm.mootdx_source 完成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["structured_reversal", "METHODS"]


# 三种加权方式
METHODS = ("equal", "time", "volume")


def _rolling_wsum(log_ret: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """固定权重下的滑动窗口加权和。

    Args:
        log_ret: 1D 每根 bar 的对数收益(可能含一个前置 NaN)。
        weight:  长度 = window 的权重向量(近期 bar = 最后一个元素)。

    Returns:
        长度 len(log_ret)-window+1 的加权和序列。
    """
    if len(log_ret) < len(weight):
        return np.array([])
    R = sliding_window_view(log_ret, len(weight))          # (T, window)
    return (R * weight).sum(axis=1)


def structured_reversal(
    bars: pd.DataFrame,
    method: str = "volume",
    window: int = 21,
    halflife: float | None = 5.0,
) -> pd.Series:
    """计算单只标的的结构化反转因子时间序列。

    Args:
        bars:     index=datetime, 至少含 'close' 与 'volume' 列的 DataFrame，
                 任何 bar 频率均可(日线或分钟线)。
        method:   'equal' | 'time' | 'volume'。
        window:   回看窗口(以 bar 计)。日线默认 21≈1 个月交易日。
        halflife: 'time' 方法的半衰期(以 bar 计)。其余方法忽略。

    Returns:
        pd.Series(index 同 bars)。窗口填满前为 NaN。
        值越低 → 近期跌幅越大 → 反转越强。
    """
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}, expected one of {METHODS}")

    close = pd.to_numeric(bars["close"], errors="coerce")
    if "volume" in bars:
        vol = pd.to_numeric(bars["volume"], errors="coerce").astype(float)
    else:
        vol = pd.Series(np.nan, index=bars.index, dtype=float)

    log_ret = np.log(close / close.shift(1)).to_numpy(dtype=float)
    vol_arr = vol.to_numpy(dtype=float)

    n = len(log_ret)
    out = np.full(n, np.nan, dtype=float)

    if n < window:
        return pd.Series(out, index=bars.index, name=f"rev_{method}")

    if method == "equal":
        W = np.ones(window, dtype=float)
        wsum = _rolling_wsum(log_ret, W)
        out[window - 1:] = wsum / W.sum()

    elif method == "time":
        hl = float(halflife) if halflife else float(window) / 2.0
        # 近期 bar 权重=1；每往前一根按半衰期衰减。
        # ages[p]=0 表示最近, ages[p]=window-1 表示最旧。
        ages = (window - 1) - np.arange(window)
        W = np.exp(-ages * np.log(2.0) / hl)
        wsum = _rolling_wsum(log_ret, W)
        out[window - 1:] = wsum / W.sum()

    else:  # volume：每根 bar 的权重取该窗口自身的成交量
        R = sliding_window_view(log_ret, window)            # (T, window)
        V = sliding_window_view(vol_arr, window)
        with np.errstate(invalid="ignore", divide="ignore"):
            wsum = np.nansum(R * V, axis=1)
            wtot = np.nansum(V, axis=1)
        vals = np.where((wtot > 0) & np.isfinite(wtot), wsum / wtot, np.nan)
        out[window - 1:] = vals

    return pd.Series(out, index=bars.index, name=f"rev_{method}")


def compute_all(
    bars: pd.DataFrame,
    methods=("equal", "time", "volume"),
    window: int = 21,
    halflife: float | None = 5.0,
) -> pd.DataFrame:
    """一次计算多只方法，返回合并的 DataFrame(列 = rev_<method>)。"""
    out = {}
    for m in methods:
        out[f"rev_{m}"] = structured_reversal(bars, method=m, window=window, halflife=halflife)
    return pd.DataFrame(out)
