"""华泰证券金融工程(华泰金工)多因子选股系列 — 复现子集.

复现自华泰证券研究所『多因子系列』研究报告(2016–, quant-wiki 合集):
  - 系列之六  波动率类因子   → idiosyncratic_volatility / downside_deviation
  - 系列之七  资金流向因子   → money_flow (5m 代理版)
  - 系列之十三 历史分位数因子 → historical_percentile

选取原则: 与现有 zoo / 方正因子正交(非反转/量价/微观结构),
       聚焦波动率、订单流、均值回复三类行为/风险维度。

注: 华泰『人工智能』系列(-autoencoder 挖因子)及需基本面(估值/成长/财务质量/
一致预期)的因子依赖财务数据, 不在本沙箱覆盖范围内, 暂未复现。
"""
from .idiosyncratic_volatility import idiosyncratic_volatility, idiosyncratic_volatility_batch
from .downside_deviation import downside_deviation, downside_deviation_batch
from .money_flow import money_flow, money_flow_batch
from .historical_percentile import historical_percentile, historical_percentile_batch

__all__ = [
    "idiosyncratic_volatility", "idiosyncratic_volatility_batch",
    "downside_deviation", "downside_deviation_batch",
    "money_flow", "money_flow_batch",
    "historical_percentile", "historical_percentile_batch",
]
