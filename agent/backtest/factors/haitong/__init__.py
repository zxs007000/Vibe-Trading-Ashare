"""海通证券金融工程(海通金工)因子复现子集 — 流动性方向.

复现自海通金工『选股因子系列研究』(非流动性/逐笔大单方向)。
选取可在 5m/日频 OHLCV 上复现的流动性类因子(逐笔大单需 amount, 沙箱无, 未复现)。
"""
from .amihud_illiquidity import amihud_illiquidity, amihud_illiquidity_batch

__all__ = [
    "amihud_illiquidity", "amihud_illiquidity_batch",
]
