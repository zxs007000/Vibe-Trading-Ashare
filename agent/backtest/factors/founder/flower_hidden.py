"""花隐林间因子 (Flower Hidden in Forest, 多因子选股系列之十).

研报:《推动个股价格变化的因素分解与"花隐林间"因子》
发布: 2023-03-27

核心逻辑:
  将价格变化分解为: 市场推动力/个股突发信息/中长期基本面/噪声。
  用成交量增量对收益率的回归t值衡量信息流入的平稳度。

计算步骤:
  1. 对每分钟: y=ret(t), x=[ΔV(t),ΔV(t-1),...,ΔV(t-5)] 做线性回归
  2. 提取: 6个系数t值(t0~t5), 截距t值(t_int), F值
  3. 朝没晨雾 = std(t1,t2,t3,t4,t5), 过去20天均值 (越小越好)
  4. 午蔽古木 = |t_int|, 按F值截面均值翻转, 过去20天均值 (越小越好)
  5. 夜眠霜路 = corr(个股t_int, 截面所有股票t_int)的绝对值均值, 过去20天 (越大越好)
  6. 花隐林间 = (朝没晨雾 + 午蔽古木 + 夜眠霜路) / 3

数据需求: 分钟频收盘价+成交量
原始回测: RankIC -9.34%, ICIR -5.69, 多空32.39%, IR 4.46
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["flower_hidden", "flower_hidden_batch"]


def _daily_flower_hidden(close, vol):
    """单日花隐林间三个子指标(不含截面部分, 截面在batch中处理)。

    Returns:
        (morning_fog_std, noon_tree_abs_t, night_dew_t_int)
    """
    n = len(close)
    if n < 30:
        return (np.nan, np.nan, np.nan)

    rets = np.diff(np.log(close))
    dvol = np.diff(vol.astype(float))  # 成交量增量

    # 构建回归: 用滚动窗口, y=ret(t), x=[dV(t),...,dV(t-5)]
    # 对每个t, 用t-lookback~t的样本做OLS
    w = 6       # 6个滞后(当前+前5)
    lookback = 30  # 回归窗口
    t_values = []
    t_int_list = []
    f_list = []

    for i in range(w + lookback, len(rets)):
        # 构建样本矩阵
        Y = rets[i - lookback + 1:i + 1]
        X = np.column_stack([np.ones(lookback)] +
                            [dvol[i - lookback + 1 - lag:i + 1 - lag] for lag in range(w)])
        # 去NaN
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(Y)
        if valid.sum() < lookback // 2:
            continue
        X_v = X[valid]; Y_v = Y[valid]
        if np.std(X_v[:, 1]) < 1e-15:
            continue
        try:
            beta, _, _, _ = np.linalg.lstsq(X_v, Y_v, rcond=None)
            # 残差
            resid = Y_v - X_v @ beta
            sigma2 = np.sum(resid ** 2) / (len(Y_v) - w - 1) if len(Y_v) > w + 1 else 1
            XtX_inv = np.linalg.inv(X_v.T @ X_v + 1e-10 * np.eye(w + 1))
            se = np.sqrt(np.diag(XtX_inv) * sigma2)
            t_all = beta / np.where(se > 1e-15, se, 1)
            t_values.append(t_all[1:])  # t0~t5
            t_int_list.append(t_all[0])  # 截距t值
            f_list.append(np.sum(t_all[1:] ** 2))
        except Exception:
            continue

    if len(t_values) < 10:
        return (np.nan, np.nan, np.nan)

    t_arr = np.array(t_values)
    # 朝没晨雾 = std(t1~t5) 均值
    morning_fog = np.mean(np.std(t_arr[:, 1:], axis=1))
    # 午蔽古木 = |t_int| 均值
    noon_tree = np.mean(np.abs(t_int_list))
    # 夜眠霜路需要截面, 单日返回t_int均值作为近似
    night_dew = np.mean(t_int_list)

    return (float(morning_fog), float(noon_tree), float(night_dew))


def flower_hidden(minute_bars: pd.DataFrame) -> pd.Series:
    """计算花隐林间因子日序列(单股票版, 截面部分简化)。"""
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
    fog_list, tree_list, dew_list = [], [], []
    for d, g in df.groupby(level=0):
        fg, tr, dw = _daily_flower_hidden(g["c"].values, g["v"].values)
        fog_list.append((d, fg)); tree_list.append((d, tr)); dew_list.append((d, dw))

    fog_s = pd.Series(dict(fog_list))
    tree_s = pd.Series(dict(tree_list))
    dew_s = pd.Series(dict(dew_list))

    window = 20
    fog_f = fog_s.rolling(window, min_periods=5).mean()
    tree_f = tree_s.rolling(window, min_periods=5).mean()
    dew_f = dew_s.rolling(window, min_periods=5).mean()

    factor = (fog_f + tree_f + dew_f) / 3
    factor.name = "flower_hidden"
    return factor


def flower_hidden_batch(stocks_minute: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    return {code: flower_hidden(bars) for code, bars in stocks_minute.items()}


if __name__ == "__main__":
    np.random.seed(42)
    n = 240 * 10
    idx = pd.date_range("2025-01-01 09:30", periods=n, freq="5min")
    t = idx.time
    mask = ((t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("11:30").time())) | \
           ((t >= pd.Timestamp("13:00").time()) & (t <= pd.Timestamp("15:00").time()))
    idx = idx[mask]; n = len(idx)
    c = 10 + np.cumsum(np.random.randn(n) * 0.002)
    v = np.random.exponential(100000, n).astype(float)
    bars = pd.DataFrame({"close": c, "volume": v}, index=idx)
    f = flower_hidden(bars)
    print(f"花隐林间: 有效值={f.notna().sum()}/{len(f)}, 均值={f.mean():.6f}")
