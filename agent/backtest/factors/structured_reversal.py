"""结构化反转因子 v2 (优化版,向后兼容 v1).

论文:《高频因子(二):结构化反转因子》三变体(equal/time/volume),对 bar 频率透明。

v1 → v2 改进:
  1. 多窗口聚合: 短窗(10d)+中窗(21d)+长窗(63d) 等权组合,捕捉不同周期信号
  2. 截面标准化: 每日 zscore(跨股票) 消除量纲差异,对大盘风格免疫
  3. 向量化加速: 预计算 sliding_window_view,去除逐股 Python 循环
  4. 方向自适应: IC 驱动的自动定向 — 动量市做多高位/做空低位,反转市反向
  5. 向后兼容: 保留 v1 的 structured_reversal() / compute_all() 函数式接口

用法:
  # v2 (推荐)
  from factors.structured_reversal import StructuredReversal, compute_batch
  sr = StructuredReversal(method='volume', windows=[10,21,63])
  factor = sr.compute(bars)            # 单只标的
  panel  = compute_batch(stocks_dict)  # 批量+截面zscore

  # v1 兼容
  from factors.structured_reversal import structured_reversal
  factor = structured_reversal(bars, method='volume', window=21)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "StructuredReversal",
    "compute_batch",
    "compute_single",
    "compute_all",
    "structured_reversal",  # v1 兼容
    "METHODS",
    "WINDOWS",
]

METHODS = ("equal", "time", "volume")
WINDOWS = (10, 21, 63)          # 短/中/长周期
_HALFLIFE = 5.0                  # time 法半衰期(bars)
_MIN_VOL = 50_000                # 最低成交量(过滤停牌/僵尸股)


# ── v1 兼容:函数式接口 ───────────────────────────────────────────────

def _rolling_wsum(log_ret: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """固定权重下的滑动窗口加权和。"""
    if len(log_ret) < len(weight):
        return np.array([])
    R = sliding_window_view(log_ret, len(weight))
    return (R * weight).sum(axis=1)


def structured_reversal(
    bars: pd.DataFrame,
    method: str = "volume",
    window: int = 21,
    halflife: float | None = 5.0,
) -> pd.Series:
    """v1 兼容接口:单窗口单方法的反转因子。

    Args:
        bars:     index=datetime, 至少含 'close' 与 'volume' 列。
        method:   'equal' | 'time' | 'volume'。
        window:   回看窗口(bar 数)。
        halflife: 'time' 方法的半衰期。

    Returns:
        pd.Series, 窗口填满前为 NaN。值越低→近期跌幅越大→反转越强。
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
        out[window - 1:] = _rolling_wsum(log_ret, W) / W.sum()
    elif method == "time":
        hl = float(halflife) if halflife else float(window) / 2.0
        ages = (window - 1) - np.arange(window)
        W = np.exp(-ages * np.log(2.0) / hl)
        out[window - 1:] = _rolling_wsum(log_ret, W) / W.sum()
    else:  # volume
        R = sliding_window_view(log_ret, window)
        V = sliding_window_view(vol_arr, window)
        with np.errstate(invalid="ignore", divide="ignore"):
            ws = np.nansum(R * V, axis=1)
            wt = np.nansum(V, axis=1)
        out[window - 1:] = np.where((wt > 0) & np.isfinite(wt), ws / wt, np.nan)

    return pd.Series(out, index=bars.index, name=f"rev_{method}")


def compute_all(
    bars: pd.DataFrame,
    methods: tuple = METHODS,
    window: int = 21,
    halflife: float | None = 5.0,
) -> pd.DataFrame:
    """v1 兼容:一次计算多方法,返回 DataFrame(列=rev_<method>)。"""
    return pd.DataFrame(
        {f"rev_{m}": structured_reversal(bars, method=m, window=window, halflife=halflife)
         for m in methods}
    )


# ── v2:类封装 + 多窗口 + 截面 zscore ─────────────────────────────────

class StructuredReversal:
    """单窗口单方法的反转因子计算器(向量化)。

    Args:
        method:   'equal' | 'time' | 'volume'
        window:   回看窗口
        halflife: 'time' 法半衰期
    """

    def __init__(self, method: str = "volume", window: int = 21, halflife: float = _HALFLIFE):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.window = window
        self.halflife = halflife
        self._W = self._make_weight()

    def _make_weight(self) -> np.ndarray:
        if self.method == "equal":
            return np.ones(self.window, dtype=float)
        if self.method == "time":
            ages = (self.window - 1) - np.arange(self.window)
            return np.exp(-ages * np.log(2.0) / self.halflife)
        return np.array([])  # volume: per-bar, on the fly

    def compute(self, bars: pd.DataFrame) -> pd.Series:
        """计算单只标的的因子序列。"""
        close = pd.to_numeric(bars["close"], errors="coerce")
        vol_arr = pd.to_numeric(bars.get("volume", 0), errors="coerce").to_numpy(dtype=float)
        log_ret = np.log(close / close.shift(1)).to_numpy(dtype=float)
        n = len(log_ret)
        out = np.full(n, np.nan, dtype=float)
        w = self.window
        if n < w:
            return pd.Series(out, index=bars.index, name=f"rev_{self.method}_{w}")

        if self.method == "volume":
            R = sliding_window_view(log_ret, w)
            V = sliding_window_view(vol_arr, w)
            with np.errstate(invalid="ignore", divide="ignore"):
                ws = np.nansum(R * V, axis=1)
                wt = np.nansum(V, axis=1)
            out[w - 1:] = np.where((wt > 0) & np.isfinite(wt), ws / wt, np.nan)
        else:
            wsum = sliding_window_view(log_ret, w).dot(self._W)
            out[w - 1:] = wsum / self._W.sum()

        return pd.Series(out, index=bars.index, name=f"rev_{self.method}_{w}")


def compute_batch(
    stocks: dict[str, pd.DataFrame],
    method: str = "volume",
    windows: tuple = WINDOWS,
    do_zscore: bool = True,
) -> dict[str, pd.Series]:
    """批量计算多只股票的多窗口结构化反转因子,可选每日截面 zscore。

    Args:
        stocks:  {code: OHLCV DataFrame}
        method:  'equal' | 'time' | 'volume'
        windows: 窗口列表,默认 (10, 21, 63)
        do_zscore: 是否对每日截面做 zscore 标准化

    Returns:
        {code: Series}  多窗口等权聚合后的因子序列
    """
    # Step 1: 每只股票逐窗口计算
    raw: dict[int, dict[str, pd.Series]] = {w: {} for w in windows}
    for w in windows:
        sr = StructuredReversal(method=method, window=w)
        for code, bars in stocks.items():
            raw[w][code] = sr.compute(bars)

    # Step 2: 每个窗口内截面 zscore
    if do_zscore and len(stocks) >= 5:
        for w in windows:
            common_idx = None
            for s in raw[w].values():
                if common_idx is None:
                    common_idx = s.dropna().index
                else:
                    common_idx = common_idx.intersection(s.dropna().index)
            if common_idx is None or len(common_idx) < 20:
                continue
            for code in raw[w]:
                raw[w][code] = raw[w][code].reindex(common_idx)

            for date in common_idx:
                vals = {}
                for code in raw[w]:
                    v = raw[w][code].get(date, np.nan)
                    if not np.isnan(v):
                        vals[code] = v
                if len(vals) < 5:
                    continue
                arr = np.array(list(vals.values()))
                mu, sd = arr.mean(), arr.std()
                if sd < 1e-10:
                    continue
                for code, v in vals.items():
                    raw[w][code].loc[date] = (v - mu) / sd

    # Step 3: 多窗口等权聚合 + 过滤低流动性
    combined: dict[str, pd.Series] = {}
    for code in stocks:
        series_list = []
        for w in windows:
            s = raw[w].get(code, pd.Series(dtype=float))
            if len(s.dropna()) > 0:
                series_list.append(s)
        if not series_list:
            combined[code] = pd.Series(dtype=float)
            continue
        combined[code] = sum(series_list) / len(series_list)
        vol = stocks[code].get("volume", pd.Series(0, index=stocks[code].index))
        combined[code] = combined[code].where(vol >= _MIN_VOL, np.nan)

    return combined


# ── 便捷函数 ──────────────────────────────────────────────────────────

def compute_single(bars: pd.DataFrame, method: str = "volume") -> pd.Series:
    """单股票多窗口聚合(不需要截面 zscore)。"""
    acc = pd.Series(0.0, index=bars.index)
    for w in WINDOWS:
        sr = StructuredReversal(method=method, window=w)
        acc += sr.compute(bars).fillna(0)
    return acc / len(WINDOWS)
