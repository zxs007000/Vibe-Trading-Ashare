"""方正金工多因子选股系列因子复现.

复现自方正证券研究所金融工程团队"多因子选股系列研究"及
"聆听高频世界的声音"系列研报。

已复现因子(12个):
  1. 滴水穿石 (drip_water_stone)    — 系列廿四, FFT频谱, IC>0.1, 多空42%
  2. 聪明钱   (smart_money)          — 高频系列三, S指标+VWAP, IR=3.74
  3. 枯树生花 (withered_tree_blooms) — 高频系列七, 5时段动量, IR=2.30
  4. 适度冒险 (moderate_risk)        — 系列一, 激增时刻, IC=-6%
  5. 球队硬币 (coin_team)            — 系列四, 三维反转, IC=-9.67%
  6. 完整潮汐 (complete_tide)        — 系列二, 邻域成交量, IC=-7.90%
  7. 勇攀高峰 (scaling_heights)      — 系列三, 更优波动率, IC=5.62%
  8. 云开雾散 (clouds_disperse)      — 系列五, 模糊性, IC=-9.81%
  9. 飞蛾扑火 (moth_to_flame)        — 系列六, 泰勒跳跃度, 负向
  10.花隐林间 (flower_hidden)        — 系列十, 回归t值, IC=-9.34%
  11.待著而救 (wait_rescue)          — 系列十一, 大单跟随, IC~-7%
  12.一视同仁 (equal_treatment)      — 系列十八, 放缩对称, IC=-7.39%

待复现(公式不完整, 需原始PDF):
  - 草木皆兵(系列八)、多空博弈(系列十三)、协同效应(系列十六)
  - 激流勇进(系列十九)、暗流涌动(系列廿三)
"""

from .drip_water_stone import drip_water_stone, drip_water_stone_batch
from .smart_money import smart_money, smart_money_batch
from .withered_tree_blooms import withered_tree_blooms, WITHERED_TREE_WEIGHTS
from .moderate_risk import moderate_risk, moderate_risk_batch
from .coin_team import coin_team, coin_team_batch
from .complete_tide import complete_tide, complete_tide_batch
from .scaling_heights import scaling_heights, scaling_heights_batch
from .clouds_disperse import clouds_disperse, clouds_disperse_batch
from .moth_to_flame import moth_to_flame, moth_to_flame_batch
from .flower_hidden import flower_hidden, flower_hidden_batch
from .wait_rescue import wait_rescue, wait_rescue_batch
from .equal_treatment import equal_treatment, equal_treatment_batch

__all__ = [
    "drip_water_stone", "drip_water_stone_batch",
    "smart_money", "smart_money_batch",
    "withered_tree_blooms", "WITHERED_TREE_WEIGHTS",
    "moderate_risk", "moderate_risk_batch",
    "coin_team", "coin_team_batch",
    "complete_tide", "complete_tide_batch",
    "scaling_heights", "scaling_heights_batch",
    "clouds_disperse", "clouds_disperse_batch",
    "moth_to_flame", "moth_to_flame_batch",
    "flower_hidden", "flower_hidden_batch",
    "wait_rescue", "wait_rescue_batch",
    "equal_treatment", "equal_treatment_batch",
]
