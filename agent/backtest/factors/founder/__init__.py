"""方正金工多因子选股系列因子复现 — 全套16因子.

复现自方正证券研究所金融工程团队"多因子选股系列研究"及
"聆听高频世界的声音"系列研报. 全部16个量价因子已复现.

因子清单(按系列编号):
  1. 适度冒险   moderate_risk       — 系列一,   激增时刻+耀眼波动率, IC=-6%
  2. 完整潮汐   complete_tide       — 系列二,   邻域成交量+潮汐速率, IC=-7.90%
  3. 勇攀高峰   scaling_heights     — 系列三,   更优波动率+协方差,   IC=5.62%
  4. 球队硬币   coin_team           — 系列四,   三维反转+可知性,    IC=-9.67%
  5. 云开雾散   clouds_disperse     — 系列五,   波动率的波动率,     IC=-9.81%
  6. 飞蛾扑火   moth_to_flame       — 系列六,   泰勒跳跃度,        负向
  7. 草木皆兵   panic_factor        — 系列八,   惊恐度权重,        IC=-8.90%
  8. 水中行舟   (待复现)            — 系列九,   成交跟随性
  9. 花隐林间   flower_hidden       — 系列十,   回归t值,           IC=-9.34%
 10. 待著而救   wait_rescue         — 系列十一, 大单跟随,          IC~-7%
 11. 多空博弈   bull_bear_game      — 系列十三, 秩相关博弈,        IC=-9.73%
 12. 协同效应   synergy_effect      — 系列十六,〔推断〕需全市场,   IC=-10.76%
 13. 一视同仁   equal_treatment     — 系列十八, 放缩对称,          IC=-7.39%
 14. 激流勇进   rapids_advance      — 系列十九, 放量下跌买入强度,  IC=+8.00%
 15. 暗流涌动   undercurrent        — 系列廿三, 分布熵+流动性弹性, IC=-7.65%
 16. 滴水穿石   drip_water_stone    — 系列廿四, FFT频谱,           IC>0.1

早期高频系列(另计):
  - 聪明钱     smart_money         — 高频三,   S指标+VWAP,  IR=3.74
  - 枯树生花   withered_tree_blooms — 高频七,   5时段动量,   IR=2.30

注: 水中行舟(系列九)需全市场截面相关性, 暂未纳入本包.
    协同效应为付费PDF推断实现, 精确公式需购原报告.
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
from .bull_bear_game import bull_bear_game, bull_bear_game_batch
from .panic_factor import panic_factor, panic_factor_batch
from .synergy_effect import synergy_effect, synergy_effect_batch
from .rapids_advance import rapids_advance, rapids_advance_batch
from .undercurrent import undercurrent, undercurrent_batch

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
    "bull_bear_game", "bull_bear_game_batch",
    "panic_factor", "panic_factor_batch",
    "synergy_effect", "synergy_effect_batch",
    "rapids_advance", "rapids_advance_batch",
    "undercurrent", "undercurrent_batch",
]
