#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 方向4: 微观结构挖掘(Level-2 / Tick)
===================================================
对应报告第四章 4.4「微观结构挖掘(Level-2/Tick)」与代码示例文档第 4 节。

现状说明(重要)
--------------
本仓库的数据湖 stocklake 目前只有日频 daily 层(开高低收量额)。Level-2 / Tick(逐笔成交、
十档委托、逐笔委托) 需要**另行接入**(券商 Level-2 落地 / 第三方快照)。
因此本模块:
  - `microstructure_features(tick)` : **真实可运行** 的微观结构特征计算函数,
    输入为「逐笔成交」面板(列: ts, price, volume, side), 输出横截面日频特征;
  - `demo_microstructure()`      : **合成数据** 演示, 用泊松到达 + 随机游走生成 tick,
    跑通整条流水线, 让第四章四个方向都能本地复现;
  - 接入真实数据时, 只需把 daily 层之外的 tick 读取逻辑补上, 直接替换 demo 的输入。

特征一览(均对齐代码示例文档与 A 股微观结构研究共识)
  - trade_count_autocorr : 成交量自相关(知情交易持续度)
  - single_trade_intercept: 单笔成交回归截距(大单占比代理)
  - vp_euclidean         : 量价轨迹欧氏距离(量价背离强度)
  - realized_spread      : 实现价差(流动性成本代理)
  - order_imbalance      : 买卖方向不平衡(主力净流入代理)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 真实特征函数(输入为「逐笔成交」DataFrame)
# ---------------------------------------------------------------------------
def microstructure_features(tick: pd.DataFrame) -> dict[str, float]:
    """
    从单只股票的逐笔成交序列计算当日微观结构特征。

    参数
    ----
    tick : DataFrame, 必须包含列 [ts, price, volume, side]
           - ts     : 成交时间戳
           - price  : 成交价
           - volume : 成交量(股)
           - side   : 方向, +1 买 / -1 卖(或 1 主买 / 0 主卖)

    返回
    ----
    dict : 各微观结构特征(标量)
    """
    if tick is None or len(tick) < 10:
        return {}

    tick = tick.sort_values("ts").reset_index(drop=True)
    price = tick["price"].to_numpy(dtype=float)
    vol = tick["volume"].to_numpy(dtype=float)
    side = tick["side"].to_numpy(dtype=float)
    n = len(tick)

    # 1) 成交量自相关(滞后 1): 知情交易往往持续, 自相关偏高
    dvol = vol - vol.mean()
    if np.std(dvol) > 1e-9:
        trade_count_autocorr = float(np.corrcoef(dvol[:-1], dvol[1:])[0, 1])
    else:
        trade_count_autocorr = 0.0

    # 2) 单笔成交回归截距: 用 log(volume) 对 log(price_move) 回归, 截距捕捉「大单」结构
    pchg = np.diff(price)
    pchg = np.concatenate([[0.0], pchg])
    mask = (vol > 0) & (np.abs(pchg) > 1e-9)
    if mask.sum() > 5:
        x = np.log(vol[mask])
        y = np.abs(pchg[mask])
        y = np.log(y + 1e-12)
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        single_trade_intercept = float(coef[1])
    else:
        single_trade_intercept = 0.0

    # 3) 量价轨迹欧氏距离: 归一化价、量序列的逐点欧氏距离, 衡量背离
    pn = (price - price.min()) / (price.max() - price.min() + 1e-12)
    vn = (vol - vol.min()) / (vol.max() - vol.min() + 1e-12)
    vp_euclidean = float(np.sqrt(np.sum((pn - vn) ** 2)) / n)

    # 4) 实现价差: 相邻成交价绝对变化的中值(隐含摩擦成本)
    realized_spread = float(np.median(np.abs(np.diff(price))))

    # 5) 买卖方向不平衡: 主买量 / 总成交量
    buy_vol = vol[side > 0].sum()
    total_vol = vol.sum()
    order_imbalance = float(buy_vol / total_vol) if total_vol > 0 else 0.0

    return {
        "trade_count_autocorr": round(trade_count_autocorr, 4),
        "single_trade_intercept": round(single_trade_intercept, 4),
        "vp_euclidean": round(vp_euclidean, 4),
        "realized_spread": round(realized_spread, 6),
        "order_imbalance": round(order_imbalance, 4),
    }


# ---------------------------------------------------------------------------
# 合成 tick 演示(无真实数据时跑通流水线)
# ---------------------------------------------------------------------------
def _synthetic_tick(seed: int = 0, n: int = 5000, start_price: float = 10.0):
    """泊松到达 + 带漂移的随机游走生成逐笔成交。仅用于演示, 不构成任何价格发现结论。"""
    rng = np.random.default_rng(seed)
    # 到达间隔(指数分布 ~ 泊松过程), 单位秒
    dt = rng.exponential(0.5, size=n).cumsum()
    drift = rng.normal(0, 0.0008, size=n).cumsum()
    price = start_price * np.exp(drift)
    volume = rng.lognormal(mean=6.0, sigma=0.8, size=n)  # 单笔股数, 长尾(大单存在)
    # 方向: 价格上行多为买, 下行多为卖, 加噪声
    side = np.where(rng.normal(drift, 0.01) > 0, 1, -1)
    # 注入「知情交易持续」特征: 部分时段买卖同向聚集
    block = rng.integers(0, 2, size=n)
    side = np.where(block == 0, side, -side)
    return pd.DataFrame({"ts": dt, "price": price, "volume": volume, "side": side})


def demo_microstructure(n_stocks: int = 5, ticks_per_stock: int = 3000, seed: int = 42):
    """
    合成多只股票 tick, 分别计算微观结构特征, 返回聚合 DataFrame。

    这是「方向 4」的本地可复现入口; 接入真实 Level-2 时, 把 _synthetic_tick
    换成真实 tick 读取即可, microstructure_features 无需改动。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_stocks):
        tick = _synthetic_tick(seed=int(rng.integers(0, 1_000_000)), n=ticks_per_stock,
                               start_price=float(rng.uniform(5, 50)))
        feats = microstructure_features(tick)
        feats["stock_id"] = f"SYN{i:03d}"
        rows.append(feats)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    print("微观结构特征(合成 tick 演示, 真实数据需另行接入 Level-2):")
    df = demo_microstructure(n_stocks=6, ticks_per_stock=2000)
    print(df.to_string(index=False))
