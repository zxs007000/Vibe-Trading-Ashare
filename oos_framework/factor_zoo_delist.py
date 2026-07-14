# -*- coding: utf-8 -*-
"""退市风险因子族 — 实时退市/困境风险评分, 用作选股宇宙过滤。

经济动机:
    策略在 2024 折被真实持有的即将退市/困境股拖累(含退市股 Sharpe -0.16 vs 仅存活 +0.115)。
    这些股票在退市前已有明确可观测的困境信号: 股价逼近 1 元面值退市线、年内深跌、成交枯竭、
    以及监管重罚(立案/处罚/谴责, 领先于 ST/退市)。把这些信号合成一个 0~1 的退市风险分,
    在回测选股时剔除高风险股, 直接隔离"困境股敞口"对策略的拖拽。

因子清单:
    f_delist_risk      合成退市风险分(0~1, 越高越危险) — 主用(作过滤)
    f_delist_faceval   面值退市风险(股价逼近 1 元)
    f_delist_distress  困境深跌(年内回撤)
    f_delist_illiq     流动性/规模枯竭风险(成交额截面底部)
    f_delist_reg       监管重罚衰减风险(领先 ST/退市)

使用方式:
    不在 IC 因子池里参与选股 —— 风险因子预期收益为负, 不会被 run_backtest 的正向冻结逻辑
    (IS_IC>0 且 IS_ICIR>0) 选中。而是作为 run_backtest 的 delist_risk 过滤矩阵, 选股前剔除
    risk>=阈值 的股票。这样避免对困境股产生正 alpha 暴露。
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# 复用监管事件因子族的事件加载与衰减逻辑
try:
    from factor_zoo_regulatory import _load_events, _trailing_decay
except Exception:  # pragma: no cover
    _load_events = None
    _trailing_decay = None


def _safe_rolling_max(close: pd.DataFrame, win: int) -> pd.DataFrame:
    return close.rolling(win, min_periods=max(20, win // 5)).max()


def _build_reg_risk(dates, codes) -> pd.DataFrame:
    """监管重罚衰减风险: severity>=3 的事件, 在事件后 250 日按半衰期 120 衰减。

    返回 0~1 矩阵; 无事件/无数据则全 0。
    """
    empty = pd.DataFrame(0.0, index=dates, columns=codes)
    if _load_events is None or _trailing_decay is None:
        return empty
    events = _load_events()
    if events is None or events.empty:
        return empty
    ev = events[events["severity"] >= 3].copy()
    if ev.empty:
        return empty
    reg = _trailing_decay(ev, dates, codes, half_life=120, window=250)
    # 严重度上限约 5, 多次事件可叠加; 截断到 [0, 5] 再归一
    return reg.clip(0, 5) / 5.0


def build_delist_risk_factors(panel: dict, industry=None) -> dict:
    """返回退市风险因子字典。

    panel 需含 'close'(必需) 与 'amount'(可选, 用于流动性风险)。
    返回 {f_delist_risk, f_delist_faceval, f_delist_distress, f_delist_illiq, f_delist_reg}。
    """
    close = panel.get("close")
    if close is None:
        logger.warning("退市风险: panel 无 close, 跳过")
        return {}
    amount = panel.get("amount")
    dates = close.index
    codes = list(close.columns)

    # 1) 面值退市风险: A股连续 20 日均价 < 1 元强制退市; 设 1.2 元预警缓冲
    #    close<0.5 -> 1.0 ; close=1.2 -> 0 ; 之间线性
    fv = ((1.2 - close).clip(lower=0) / 0.7).clip(0, 1)

    # 2) 困境深跌: 年内(252d)自高点回撤。50% 回撤 -> 0, 80% -> 1
    hi = _safe_rolling_max(close, 252)
    dd1y = (1 - close / hi).clip(lower=0)
    dist = ((dd1y - 0.5) / 0.3).clip(0, 1)

    # 3) 流动性/规模枯竭: 60 日均成交额截面底部 5%
    illiq = pd.DataFrame(0.0, index=dates, columns=codes)
    if amount is not None:
        amt = amount.rolling(60, min_periods=20).mean().values
        q05 = np.nanquantile(amt, 0.05, axis=1)  # 每日截面 5% 分位
        illiq_arr = (amt < q05[:, None]).astype(float)
        illiq = pd.DataFrame(illiq_arr, index=dates, columns=codes).fillna(0.0)

    # 4) 监管重罚风险(领先 ST/退市)
    reg = _build_reg_risk(dates, codes)

    # 合成: 加权(保守, 任一强信号主导)。阈值默认 0.5 由 run_backtest 控制
    risk = (0.50 * fv + 0.50 * dist + 0.35 * illiq + 0.60 * reg).clip(0, 1)

    return {
        "f_delist_risk": risk,
        "f_delist_faceval": fv,
        "f_delist_distress": dist,
        "f_delist_illiq": illiq,
        "f_delist_reg": reg,
    }
