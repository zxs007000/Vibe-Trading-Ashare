"""最小化 validation 工具：OOS 框架仅用到 _sharpe。

（原项目 backtest/validation.py 还含蒙特卡洛/Bootstrap/Walk-Forward，
但依赖 backtest.models，本框架不需要，故只保留 _sharpe 以保证可独立运行。）
"""
from __future__ import annotations
import numpy as np


def _sharpe(returns, bars_per_year: int = 252) -> float:
    """年化夏普比率。returns 为日收益序列(numpy array 或 pandas Series)。"""
    std = np.asarray(returns, dtype=float).std()
    return float(np.asarray(returns, dtype=float).mean() / (std + 1e-10) * np.sqrt(bars_per_year))
