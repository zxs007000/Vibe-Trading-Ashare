"""factor_zoo_ortho.py — 状态正交因子(专为激活选股层 regime 维度开关而构造).

动机(oos_regime_switch 的诚实结论): 30 技术 + 3 基本面因子在 IS 内几乎'两态皆活'
(仅 1 牛活 + 2 熊活), 因子池跨 regime 同质 -> regime 开关 ≈ 静态 Frozen.
要让'什么状态用什么因子'在选股层真正生效, 需要**真正状态正交的因子**:
  在一种市场状态(牛)出生、在另一种状态(熊)死亡(或显著弱化).

本模块用纯价格/成交额构造一批理论上状态正交的候选因子(无新数据源):
  - beta_60        : 60日滚动 beta 至等权市场. 经济含义=高 beta; 牛市骑趋势、熊市被杀 -> 偏牛.
  - lowvol_60      : -60日总波动. 低波防御, 熊市异常抗跌(低波异象在下跌段最强) -> 偏熊.
  - lowivol_60     : -60日特质波动(剔市场 beta 后的残差波动). 同低波但控住 beta -> 偏熊.
  - liq_stress_20  : 流动性压力 = 近期 Amihud 相对其 60 日均值的增幅; 流动性枯竭是典型的熊市/危机信号 -> 偏熊.
  - distress_60    : 60日下行波动(下行标准差). 越高越"受创", 熊市更高 -> 偏熊.

注: 因子符号由引擎按 IS mean-IC 自动定向(orient), 故此处只需保证经济内容正确, 不用手调正负.
这些因子是否与理论一致、是否真'状态正交', 由 oos_regime_switch_ortho 的 per-regime 存活检验实证.

用法:
  from factor_zoo_ortho import build_ortho_factors, ORTHO_NAMES
"""
from __future__ import annotations
import numpy as np, pandas as pd


ORTHO_NAMES = ["beta_60", "lowvol_60", "lowivol_60", "liq_stress_20", "distress_60"]

# 类型标注(用于报告分组)
ORTHO_FAMILY = {
    "beta_60": "市场暴露",
    "lowvol_60": "防御低波",
    "lowivol_60": "防御低波",
    "liq_stress_20": "流动性压力",
    "distress_60": "困境下行",
}


def build_ortho_factors(wide):
    """返回 dict: name -> date×code DataFrame(因子值, 未标准化).

    所有因子基于 wide 的 close/amount 计算, 与 factor_zoo_daily.build_factors 同口径面板,
    故可直接并入同一 fac dict 一起中性化 / zarr.
    """
    close = wide["close"]
    ret = close.pct_change()
    amount = wide["amount"]
    mkt = ret.mean(axis=1)                      # 等权市场日收益(与 regime 信号同源, 无泄漏)
    f = {}

    # ---- beta_60: 60日滚动 beta 至等权市场 ----
    # 用显式滚动协方差公式(向量化, 避免 DataFrame.rolling.cov(Series) 不广播的 NaN bug):
    #   cov_t(ret, mkt) = E[ret*mkt]_w - E[ret]_w * E[mkt]_w
    mkt_var = mkt.rolling(60).var()
    e_ret = ret.rolling(60).mean()
    e_mkt = mkt.rolling(60).mean()
    e_retmkt = ret.mul(mkt, axis=0).rolling(60).mean()   # 注意: DataFrame*Series 默认按列对齐->全NaN, 须 axis=0 按行(日期)广播
    cov_rm = e_retmkt - e_ret.mul(e_mkt, axis=0)
    beta_60 = cov_rm.div(mkt_var, axis=0)   # 须 .div(axis=0) 按日期广播, / 默认按列对齐->全NaN
    f["beta_60"] = beta_60

    # ---- lowvol_60: -60日总波动(低波防御) ----
    f["lowvol_60"] = -ret.rolling(60).std()

    # ---- lowivol_60: -60日特质波动(残差波动, 控住 beta) ----
    resid = ret - beta_60.mul(mkt, axis=0)
    f["lowivol_60"] = -resid.rolling(60).std()

    # ---- liq_stress_20: 流动性压力(Amihud 近期相对 trailing 增幅) ----
    amihud = (ret.abs() / amount).rolling(20).mean()      # 近期流动性(越高越不流动)
    liq_trail = amihud.rolling(60).mean()
    f["liq_stress_20"] = amihud / liq_trail - 1           # >0 = 流动性正在枯竭(压力升)

    # ---- distress_60: 60日下行波动(困境程度) ----
    f["distress_60"] = ret.clip(upper=0).rolling(60).std()

    # 统一 float32 省内存 + 去极端无穷
    out = {}
    for k, v in f.items():
        out[k] = v.astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return out
