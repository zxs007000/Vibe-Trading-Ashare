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


def synergy_effect_batch(stocks_minute: dict[str, pd.DataFrame], window: int = 20) -> dict[str, pd.Series]:
    """批量计算(向量化截面版, 用全宇宙作同群体近似)。

    协同度 C = 个股日内分钟收益率 与 截面均值日内收益率 的逐日相关系数(真协同度量)。
    辨识度 I = 个股日收益 相对 截面均值 的标准化偏离。
    性价比 P = 短期反转(近5日日收益取负)。
    得分 = C × I × P, 月频(window=20)滚动均值。

    ⚠ 原研报需全市场+付费公式;此处为〔推断〕截面代理, 且已向量化:
       仅 O(交易日 × 股票 × 分钟) 一次构建, 不再逐 peer 重建掩码(原版超时)。
    """
    codes = list(stocks_minute.keys())

    # 1) 每只股票: 日内分钟收益率 pivot(date × time)
    pivots = {}
    for code in codes:
        bars = stocks_minute[code]
        c = pd.to_numeric(bars["close"], errors="coerce")
        dates = c.index.normalize()
        times = c.index.time
        r = c.groupby(dates).transform(lambda s: s.pct_change())
        df = pd.DataFrame({"v": r.values, "d": dates, "t": times})
        pivots[code] = df.pivot(index="d", columns="t", values="v")

    # 2) 日收益代理(日内收益率均值) + 截面均值
    daily_ret = pd.DataFrame({code: pivots[code].mean(axis=1, skipna=True) for code in codes})
    mret = daily_ret.mean(axis=1)

    # 3) 逐日协同度 C = corr(个股日内路径, 截面均值日内路径)
    all_dates = sorted(set().union(*[set(p.index) for p in pivots.values()]))
    C = pd.DataFrame(index=all_dates, columns=codes, dtype=float)
    for d in all_dates:
        cols = []
        for code in codes:
            piv = pivots[code]
            if d in piv.index:
                cols.append(piv.loc[d].values.astype(float))
            else:
                cols.append(np.full(piv.shape[1], np.nan))
        M = np.array(cols, dtype=float).T  # time × stocks
        mean_vec = np.nanmean(M, axis=1)
        if np.nanstd(mean_vec) < 1e-12:
            continue
        for j, code in enumerate(codes):
            x = M[:, j]
            ok = np.isfinite(x) & np.isfinite(mean_vec)
            if ok.sum() > 10 and np.nanstd(x[ok]) > 1e-12:
                C.loc[d, code] = float(np.corrcoef(x[ok], mean_vec[ok])[0, 1])
            else:
                C.loc[d, code] = np.nan

    # 4) 辨识度 I = (个股日收益 - 截面均值) 截面zscore
    I = daily_ret.sub(mret, axis=0)
    I = (I - I.mean(axis=0)).div(I.std(axis=0).replace(0, np.nan))

    # 5) 性价比 P = 短期反转(近5日日收益取负)
    P = -daily_ret.rolling(5, min_periods=3).mean()

    # 6) 合成 + 滚动
    score = C * I * P
    factor = score.rolling(window, min_periods=5).mean()
    return {code: factor[code] for code in codes}


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
