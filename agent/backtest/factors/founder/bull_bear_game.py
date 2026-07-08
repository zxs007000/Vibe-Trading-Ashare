"""多空博弈因子 (Bull-Bear Game, 多因子选股系列之十三).

研报:《股票日内多空博弈激烈程度度量与"多空博弈"因子构建》
发布: 2023-11-29

核心逻辑:
  用"收益率方向"给"成交量/振幅"排序加权,衡量多空博弈激烈程度。
  本质是成交量(或振幅)与收益率的秩相关: 放量发生在大涨端 vs 大跌端。

计算步骤:
  对每天N分钟:
  1. 按收益率r_t从小到大排序, 升序排名p_t ∈ {1,...,N}
  2. 日因子 = Σ_t (2·p_t − N − 1) × w_t  ∝ Spearman(w, r)
     w_t = 成交量v_t 或 振幅a_t 或 相对位置pos_t

  三个基础因子:
  - 成交量博弈-收益率: 排序键r_t, 加权v_t
  - 成交量博弈-相对位置: 排序键pos_t, 加权v_t
  - 振幅博弈: 排序键r_t, 加权a_t

  成交量博弈 = 0.5×(收益率变体) + 0.5×(相对位置变体)
  多空博弈 = 0.5×成交量博弈 + 0.5×振幅博弈

  月频合成: 截面z-score绝对值(均值距离化)→月均+月稳等权

数据需求: 分钟频OHLCV (无需tick/市场基准/截面相关性, 仅月频截面z-score)
原始回测: RankIC -9.73%, ICIR -5.51, 多空40.12%, IR 4.51
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["bull_bear_game", "bull_bear_game_batch"]


def _daily_game(close, high, low, vol):
    """单日多空博弈三个基础因子。

    Returns:
        (vol_game_ret, vol_game_pos, amp_game)
    """
    n = len(close)
    if n < 20:
        return (np.nan, np.nan, np.nan)

    # 分钟收益率 (5分钟窗口近似)
    rets = np.diff(np.log(close))
    r = rets
    # 日内相对位置: 距当日最低涨幅 + 距当日最高跌幅
    cum_max = np.maximum.accumulate(close)
    cum_min = np.minimum.accumulate(close)
    pos_up = close / cum_min - 1
    pos_down = cum_max / close - 1
    pos = 0.5 * pos_up + 0.5 * pos_down
    pos = pos[1:]  # 对齐rets

    # 振幅
    amp = (high - low) / close
    amp = amp[1:]

    v = vol[1:]

    valid = np.isfinite(r) & np.isfinite(v) & np.isfinite(pos) & np.isfinite(amp)
    if valid.sum() < 20:
        return (np.nan, np.nan, np.nan)

    r_v = r[valid]; v_v = v[valid]; pos_v = pos[valid]; amp_v = amp[valid]

    # 排序位置加权
    def _rank_weighted(key, weight):
        # 按key升序排名
        order = np.argsort(key)
        ranked_key = key[order]
        ranked_w = weight[order]
        # 升序位置 p_t (1-indexed)
        p = np.arange(1, len(key) + 1)
        # 日因子 = Σ (2p - N - 1) * w
        contrib = np.sum((2 * p - len(key) - 1) * ranked_w)
        return contrib

    vol_game_ret = _rank_weighted(r_v, v_v)
    vol_game_pos = _rank_weighted(pos_v, v_v)
    amp_game = _rank_weighted(r_v, amp_v)

    return (float(vol_game_ret), float(vol_game_pos), float(amp_game))


def bull_bear_game(minute_bars: pd.DataFrame) -> pd.Series:
    """计算多空博弈因子日序列(单股票)。"""
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

    df = pd.DataFrame({"c": c, "h": h, "l": l, "v": v}, index=dates)
    ret_list, pos_list, amp_list = [], [], []
    for d, g in df.groupby(level=0):
        r, p, a = _daily_game(g["c"].values, g["h"].values, g["l"].values, g["v"].values)
        ret_list.append((d, r)); pos_list.append((d, p)); amp_list.append((d, a))

    ret_s = pd.Series(dict(ret_list))
    pos_s = pd.Series(dict(pos_list))
    amp_s = pd.Series(dict(amp_list))

    # 成交量博弈 = 0.5*(收益率变体) + 0.5*(相对位置变体)
    vol_game = 0.5 * ret_s + 0.5 * pos_s
    # 多空博弈 = 0.5*成交量博弈 + 0.5*振幅博弈
    raw_factor = 0.5 * vol_game + 0.5 * amp_s

    # 月频合成: 月均+月稳等权
    window = 20
    mean_f = raw_factor.rolling(window, min_periods=5).mean()
    std_f = raw_factor.rolling(window, min_periods=5).std()
    factor = (mean_f + std_f) / 2
    factor.name = "bull_bear_game"
    return factor


def bull_bear_game_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """批量计算 + 截面z-score(均值距离化)。"""
    raw = {code: bull_bear_game(bars) for code, bars in stocks_minute.items()}
    # 截面z-score绝对值(均值距离化)
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

    # 转回Series
    out = {}
    for code, dmap in result.items():
        s = pd.Series(dmap)
        # 月频: 再取20天均值
        out[code] = s.rolling(20, min_periods=5).mean()

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
    o = c + np.random.randn(n) * 0.001
    h = np.maximum(o, c) + np.random.rand(n) * 0.005
    l = np.minimum(o, c) - np.random.rand(n) * 0.005
    v = np.random.exponential(100000, n)
    bars = pd.DataFrame({"close": c, "high": h, "low": l, "volume": v}, index=idx)
    f = bull_bear_game(bars)
    print(f"多空博弈: 有效={f.notna().sum()}/{len(f)}, 均值={f.mean():.6f}")
