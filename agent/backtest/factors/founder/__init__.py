"""方正金工多因子选股系列因子复现.

复现自方正证券研究所金融工程团队"多因子选股系列研究"及
"聆听高频世界的声音"系列研报。

已复现因子(5个):
  1. 滴水穿石 (drip_water_stone)     — FFT频谱分析, IC>0.1, 多空42%
  2. 聪明钱   (smart_money)           — S指标+VWAP比值, IR=3.74
  3. 枯树生花 (withered_tree_blooms)  — 日内5时段动量, IR=2.30
  4. 适度冒险 (moderate_risk)         — 激增时刻+耀眼波动率, IC=-6%
  5. 球队硬币 (coin_team)             — 三维反转+可知性, IC=-9.67%

待复现因子(公式不完整, 需原始PDF):
  - 完整潮汐、勇攀高峰、云开雾散、飞蛾扑火、草木皆兵
  - 水中行舟、花隐林间、待著而救、多空博弈、协同效应
  - 一视同仁、激流勇进、暗流涌动
"""

from .drip_water_stone import drip_water_stone, drip_water_stone_batch
from .smart_money import smart_money, smart_money_batch
from .withered_tree_blooms import withered_tree_blooms, WITHERED_TREE_WEIGHTS
from .moderate_risk import moderate_risk, moderate_risk_batch
from .coin_team import coin_team, coin_team_batch

__all__ = [
    "drip_water_stone", "drip_water_stone_batch",
    "smart_money", "smart_money_batch",
    "withered_tree_blooms", "WITHERED_TREE_WEIGHTS",
    "moderate_risk", "moderate_risk_batch",
    "coin_team", "coin_team_batch",
]
