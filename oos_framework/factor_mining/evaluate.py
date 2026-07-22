#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 因子评估(IC / ICIR / 换手率)
============================================
报告第四章强调以「IC / ICIR」作为因子筛选核心指标, 并对高换手畸形因子施加惩罚。
本模块提供:
  - ic_series       : 逐日横截面 Rank-IC(Spearman) 序列
  - evaluate_factor : 多 horizon 的 IC / ICIR 汇总
  - turnover        : 横截面Rank换手率(衡量交易磨损)
  - nan-oriented     : 用 pairwise 删除处理缺失, 避免整日被 NaN 拖垮
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from numba import njit

warnings.filterwarnings("ignore")  # 评估期常值窗口的 NaN 噪声


@njit
def _rank_ic_nb(F, R, min_obs=20):
    """逐日横截面 Rank-IC（Numba 向量化, 替代 Python 循环, 提速 ~20×）。"""
    T, N = F.shape
    out = np.empty(T)
    for i in range(T):
        sa = F[i]
        sb = R[i]
        # 逐日 pairwise 掩码
        cnt = 0
        ma = 0.0
        mb = 0.0
        for j in range(N):
            if not np.isnan(sa[j]) and not np.isnan(sb[j]):
                ma += sa[j]
                mb += sb[j]
                cnt += 1
        if cnt < min_obs:
            out[i] = np.nan
            continue
        ma /= cnt
        mb /= cnt
        num = 0.0
        da = 0.0
        db = 0.0
        for j in range(N):
            if not np.isnan(sa[j]) and not np.isnan(sb[j]):
                aa = sa[j] - ma
                bb = sb[j] - mb
                num += aa * bb
                da += aa * aa
                db += bb * bb
        if da == 0.0 or db == 0.0:
            out[i] = np.nan
            continue
        out[i] = num / np.sqrt(da * db)
    return out


def ic_series(factor: pd.DataFrame, fwd: pd.DataFrame) -> np.ndarray:
    """
    逐日计算横截面 Rank-IC(对因子与向前收益先做截面排名, 再做 Pearson = Spearman)。
    返回长度 = 交易日数的 1D 数组, 无效日记为 NaN。
    """
    f = factor.rank(pct=True, axis=1).values.astype(np.float64)
    r = fwd.rank(pct=True, axis=1).values.astype(np.float64)
    return _rank_ic_nb(f, r)


def evaluate_factor(factor: pd.DataFrame, fwd_dict: dict[int, pd.DataFrame],
                    horizons=(5, 20, 60)) -> dict:
    """
    在多个 horizon 上评估因子。
    返回 {h: {'ic':均值IC, 'icir':ICIR, 'n':有效天数}}。
    """
    res = {}
    for h in horizons:
        ic = ic_series(factor, fwd_dict[h])
        ic = ic[~np.isnan(ic)]
        if ic.size < 10:
            res[h] = {"ic": 0.0, "icir": 0.0, "n": 0}
        else:
            sd = ic.std()
            res[h] = {"ic": float(ic.mean()), "icir": float(ic.mean() / sd) if sd > 0 else 0.0,
                      "n": int(ic.size)}
    return res


def turnover(factor: pd.DataFrame) -> float:
    """横截面 Rank 换手率: 逐日 Rank 变化绝对值的截面均值, 再对时间取均值。"""
    r = factor.rank(pct=True, axis=1)
    d = r.diff().abs().mean(axis=1)
    d = d[~d.isna()]
    return float(d.mean()) if len(d) else 0.0


def factor_valid_ratio(factor: pd.DataFrame) -> float:
    """非空覆盖率(截面平均)。"""
    return float(1.0 - factor.isna().mean().mean())
