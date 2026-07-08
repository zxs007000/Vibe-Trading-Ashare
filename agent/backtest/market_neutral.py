"""市场中性对冲框架（量化账户独立赛道）。

战略背景：基线(70/25/5)只作基准/及格线，量化单独开账户、纯做量化。
本框架回答「如何让独立量化策略过基线门槛(年化>5.7% & Sharpe>1.37 & MaxDD<5%)」——
核心杠杆之一是 **对冲掉市场β**，把纯因子策略动辄 −20% 的回撤压进个位数。

流程：
  1. 截面打分 → 每期做多得分最高一档、做空最低一档（多空组合）。
  2. 用 CSI300 收益对组合收益做 β 对冲（OLS 全样本 / 滚动窗口 / 直接减指数）。
  3. 对冲后净值 → 复用 backtest.metrics.calc_metrics 算 Sharpe/MaxDD。
  4. 与基线门槛比对，给出「过/不过线」裁决（淘汰无用功）。

依赖：价格用 stock_worm(通达信优先)，指数对冲用 000300.SH 同源。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest.metrics import calc_metrics

logger = logging.getLogger(__name__)

# 基线门槛（来自 STRATEGY_SUMMARY 553天同周期真实数据：纯基线 5.7%/1.37/−2.8%）
# 常态 MaxDD<5%，这里用 −5% 作保守红线。
BASELINE_HURDLE = {
    "annual_return": 0.057,
    "sharpe": 1.37,
    "max_drawdown": -0.05,
}


def _aligned_close(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """从 {code: df(含close)} 拼成 close 面板，按日期交集合并。"""
    close = pd.DataFrame({c: df["close"] for c, df in prices.items() if df is not None and not df.empty})
    close = close.sort_index()
    return close


def long_short_returns(
    prices: Dict[str, pd.DataFrame],
    score: pd.DataFrame,
    *,
    rebalance: str = "M",
    n_long: int = 10,
    n_short: int = 10,
    long_only: bool = False,
) -> pd.Series:
    """构建组合每日收益序列。

    Args:
        prices: {code: DataFrame(含 close)}，index=datetime。
        score: 截面打分面板，index=datetime(与价格对齐或超集)，columns=code；
               数值越高越做多。
        rebalance: 再平衡频率（'ME'月/'W'周/'Q'季）。
        n_long/n_short: 多/空档位数（按当期可用股票数上限）。
        long_only: True=只做多得分最高一档（演示纯多头因子组合的回撤，
                   再用对冲压下来）；False=多空对冲组合。

    Returns:
        每日组合收益 Series，index=交易日。
    """
    close = _aligned_close(prices)
    if close.empty:
        return pd.Series(dtype=float)
    # 打分对齐到价格交易日，前向填充（因子在调仓日已知）
    score_al = score.reindex(close.index).ffill().fillna(0.0)
    # 仅保留两边都有数据的股票
    common = [c for c in close.columns if c in score_al.columns]
    if not common:
        return pd.Series(dtype=float)
    close = close[common]
    score_al = score_al[common]

    daily_ret = close.pct_change(fill_method=None).fillna(0.0)

    # 再平衡日（用 ME 避免 'M' 弃用警告）
    freq = rebalance if rebalance in ("W", "Q") else "ME"
    rb_dates = close.resample(freq).last().index
    rb_dates = [d for d in rb_dates if d in close.index]

    port = pd.Series(0.0, index=close.index)
    for i, t in enumerate(rb_dates):
        t_next = rb_dates[i + 1] if i + 1 < len(rb_dates) else close.index[-1]
        seg = close.loc[t:t_next]
        if len(seg) < 2:
            continue
        # 用 t 时刻打分排序
        s = score_al.loc[t].sort_values(ascending=False)
        avail = s.index.tolist()
        n_l = min(n_long, max(1, len(avail) // 3))
        long_set = avail[:n_l]
        if long_only:
            # 纯多头：只做多最高一档
            seg_ret = daily_ret.loc[seg.index]
            port.loc[seg.index] = seg_ret[long_set].mean(axis=1)
        else:
            n_s = min(n_short, max(1, len(avail) // 3))
            short_set = avail[-n_s:]
            seg_ret = daily_ret.loc[seg.index]
            long_ret = seg_ret[long_set].mean(axis=1)
            short_ret = seg_ret[short_set].mean(axis=1)
            port.loc[seg.index] = long_ret - short_ret
    return port


def _index_returns(code: str, start: str, end: str) -> pd.Series:
    """取对冲基准（默认 CSI300）每日收益。

    指数行情走 akshare（无 token，东方财富指数接口），比 stock_worm 对指数更稳。
    """
    import akshare as ak

    # 000300.SH / 000300.XSHG → sh000300
    sym = "sh000300"
    if "000300" not in code and "300" in code:
        sym = "sh000300"
    try:
        df = ak.stock_zh_index_daily(symbol=sym)
    except Exception as exc:
        logger.warning("_index_returns akshare failed for %s: %s", code, exc)
        return pd.Series(dtype=float)
    if df is None or df.empty or "date" not in df.columns:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start:end]
    if "close" not in df.columns or df.empty:
        return pd.Series(dtype=float)
    return df["close"].pct_change(fill_method=None).fillna(0.0).rename("index_ret")


def hedge_portfolio(
    gross_ret: pd.Series,
    index_ret: Optional[pd.Series] = None,
    *,
    method: str = "ols",
    window: int = 60,
) -> tuple[pd.Series, float]:
    """对多空组合做 β 对冲，返回净收益序列与估计 β。

    method:
      'ols'      全样本 OLS beta（默认，稳健）
      'rolling'  滚动 window 日窗口估计 beta（自适应）
      'subtract' 直接减指数收益（beta=1，最简单市场中性）
    """
    if method == "subtract" or index_ret is None:
        if index_ret is None:
            logger.warning("hedge_portfolio: 无指数收益，跳过对冲")
            return gross_ret, 0.0
        net = gross_ret - index_ret.reindex(gross_ret.index).fillna(0.0)
        return net, 1.0

    idx = index_ret.reindex(gross_ret.index).fillna(0.0)
    aligned = pd.concat([gross_ret, idx], axis=1).dropna()
    if len(aligned) < 30:
        return gross_ret, 0.0
    g, x = aligned.iloc[:, 0], aligned.iloc[:, 1]

    if method == "rolling":
        beta = g.rolling(window).cov(x) / x.rolling(window).var()
        beta = beta.fillna(method="bfill").fillna(0.0)
        net = g - beta * x
        return net, float(beta.mean())

    # OLS 全样本
    beta = float(np.cov(g, x)[0, 1] / (np.var(x) + 1e-12))
    net = g - beta * x
    return net, beta


def run_market_neutral(
    prices: Dict[str, pd.DataFrame],
    score: pd.DataFrame,
    *,
    index_code: str = "000300.SH",
    start: Optional[str] = None,
    end: Optional[str] = None,
    rebalance: str = "M",
    n_long: int = 10,
    n_short: int = 10,
    long_only: bool = False,
    hedge: str = "ols",
    initial_cash: float = 1_000_000.0,
) -> Dict:
    """端到端运行市场中性回测。

    Returns:
        dict: {gross_returns, net_returns, nav, beta, gross_metrics, net_metrics,
               baseline_verdict}
    """
    gross = long_short_returns(
        prices, score, rebalance=rebalance, n_long=n_long, n_short=n_short,
        long_only=long_only,
    )
    if gross.empty:
        return {"error": "empty gross returns (check prices/score)"}

    s, e = start or str(gross.index[0].date()), end or str(gross.index[-1].date())
    idx_ret = _index_returns(index_code, s, e)
    net, beta = hedge_portfolio(gross, idx_ret, method=hedge)

    nav = (1.0 + net.fillna(0.0)).cumprod() * initial_cash
    gross_nav = (1.0 + gross.fillna(0.0)).cumprod() * initial_cash

    net_metrics = calc_metrics(nav, trades=[], initial_cash=initial_cash, bars_per_year=252)
    gross_metrics = calc_metrics(
        gross_nav, trades=[], initial_cash=initial_cash, bars_per_year=252
    )

    return {
        "gross_returns": gross,
        "net_returns": net,
        "nav": nav,
        "beta": beta,
        "gross_metrics": gross_metrics,
        "net_metrics": net_metrics,
        "baseline_verdict": baseline_verdict(net_metrics),
        "index_code": index_code,
        "hedge_method": hedge,
    }


def baseline_verdict(metrics: Dict) -> Dict:
    """与基线门槛比对，给出「过/不过线」裁决。"""
    ar = float(metrics.get("annual_return", 0.0) or 0.0)
    sh = float(metrics.get("sharpe", 0.0) or 0.0)
    mdd = float(metrics.get("max_drawdown", 0.0) or 0.0)
    fails = []
    if ar < BASELINE_HURDLE["annual_return"]:
        fails.append(
            f"年化 {ar:.1%} < 基线 {BASELINE_HURDLE['annual_return']:.1%}"
        )
    if sh < BASELINE_HURDLE["sharpe"]:
        fails.append(f"Sharpe {sh:.2f} < 基线 {BASELINE_HURDLE['sharpe']:.2f}")
    if mdd < BASELINE_HURDLE["max_drawdown"]:
        fails.append(f"最大回撤 {mdd:.1%} < 红线 {BASELINE_HURDLE['max_drawdown']:.1%}")
    return {
        "pass": len(fails) == 0,
        "fails": fails,
        "metrics": {"annual_return": ar, "sharpe": sh, "max_drawdown": mdd},
    }


if __name__ == "__main__":
    print("market_neutral framework loaded. BASELINE_HURDLE =", BASELINE_HURDLE)
