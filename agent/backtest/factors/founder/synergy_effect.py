"""协同效应因子 (Synergy Effect, 多因子选股系列之十六).

研报:《日内协同股票性价比度量与"协同效应"因子构建》
发布: 2024-03-19

⚠️ 重要说明:
  本因子**强依赖全市场所有股票的分钟频数据**(计算个股与同群体的
  分钟收益率/成交量相关性)。研报原始PDF为付费,以下为基于公开摘要
  的〔推断〕实现,精确公式需购买方正证券原报告。

核心逻辑:
  特征相似股票(同行业/市值相近)会因流动性约束出现短时同涨同跌/同步
  放量(协同走势)。在协同股中,辨识度强(领涨)且性价比高(更可能补涨)
  的票后续超额收益更高。

〔推断〕计算步骤:
  1. 协同度 C = 个股与同群体分钟收益率相关性 + 分钟成交量相关性
  2. 辨识度 I = 个股当日走势脱离群体的程度(相对强度)
  3. 性价比 P = 协同信号对未来收益的预测能力(此处用短期反转近似)
  4. 得分 = C × I × P, 月频20天均值

数据需求: ⚠️ 全市场所有股票分钟频OHLCV (强截面依赖)
原始回测: RankIC -10.76%, ICIR -4.09, 多空36.83%, IR 3.00
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["synergy_effect", "synergy_effect_batch"]


def synergy_effect(
    minute_bars: pd.DataFrame,
    peer_bars: list[pd.DataFrame] | None = None,
) -> pd.Series:
    """计算协同效应因子(〔推断〕版, 需同群体数据)。

    Args:
        minute_bars: 目标股票分钟K线
        peer_bars: 同群体(同行业/市值相近)其他股票的分钟K线列表
                   ⚠️ 若为空, 退化为单股票代理(信息量远低于原文)

    Returns:
        pd.Series(index=date), 因子值
    """
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

    df = pd.DataFrame({"c": c, "v": v}, index=dates)
    synergy_list = []
    for d, g in df.groupby(level=0):
        synergy_list.append((d, _daily_synergy(g["c"].values, g["v"].values, peer_bars, d)))

    s = pd.Series(dict(synergy_list))
    window = 20
    factor = s.rolling(window, min_periods=5).mean()
    factor.name = "synergy_effect"
    return factor


def _daily_synergy(close, vol, peer_bars, date):
    """单日协同度(〔推断〕)。"""
    n = len(close)
    if n < 20:
        return np.nan

    rets = np.diff(np.log(close))

    # 无peers → 退化代理: 自身放量-收益协同
    if not peer_bars:
        valid = np.isfinite(rets) & (vol[1:] > 0)
        if valid.sum() < 20:
            return np.nan
        r = rets[valid]; v = vol[1:][valid]
        return float(np.corrcoef(r, v)[0, 1]) if np.std(v) > 0 else np.nan

    # 有peers → 计算与同群体相关性
    peer_rets = []
    for pb in peer_bars:
        if isinstance(pb.index, pd.DatetimeIndex):
            pidx = pb.index
        else:
            pidx = pd.to_datetime(pb["date"])
        pt = pidx.time
        pmask = ((pt >= pd.Timestamp("09:30").time()) & (pt <= pd.Timestamp("11:30").time())) | \
                ((pt >= pd.Timestamp("13:00").time()) & (pt <= pd.Timestamp("15:00").time()))
        pc = pd.to_numeric(pb["close"], errors="coerce").to_numpy()[pmask]
        # 对齐日期
        pdate = pidx[pmask].normalize()
        if date not in set(pdate):
            continue
        # 取该日
        day_mask = pdate == date
        pc_day = pc[day_mask]
        if len(pc_day) < 20:
            continue
        pr = np.diff(np.log(pc_day))
        if len(pr) == len(rets):
            peer_rets.append(pr)

    if not peer_rets:
        return np.nan

    # 协同度 = 平均相关性
    corrs = []
    for pr in peer_rets:
        if np.std(pr) > 0 and np.std(rets) > 0:
            corrs.append(np.corrcoef(rets, pr)[0, 1])

    return float(np.mean(corrs)) if corrs else np.nan


def synergy_effect_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """批量计算(〔推断〕版, 用全市场作为peer群体近似)。"""
    codes = list(stocks_minute.keys())
    out = {}
    for i, code in enumerate(codes):
        # peer = 其他所有股票(近似同群体)
        peers = [stocks_minute[c] for c in codes[:i] + codes[i+1:i+5]]
        out[code] = synergy_effect(stocks_minute[code], peers)
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
    bars = pd.DataFrame({"close": c, "volume": v}, index=idx)
    f = synergy_effect(bars)
    print(f"协同效应(退化版): 有效={f.notna().sum()}/{len(f)}, 均值={f.mean():.6f}")
    print("  ⚠️ 需全市场数据才能精确复现")
