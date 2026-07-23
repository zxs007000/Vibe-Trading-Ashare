# -*- coding: utf-8 -*-
"""
factor_mining · 筹码结构特征 (Chip / Cost-Distribution Structure)
=================================================================
基于「准确历史换手率」(B1 层) + 日线 OHLCV, 用**衰减成本分布模型**逐股还原持仓成本结构,
产出供 XGBoost 选股用的「筹码结构」维度特征。

为什么这俩缺一不可
------------------
- 旧快照法用 成交量/当前流通股本 反推换手率, 在限售股解禁日会因流通股本突变而系统性低估早期换手,
  进而污染成本分布。B1 层 akshare 直接给交易所披露的真实当日换手率, 对解禁免疫。
- 本模块进一步用 当日换手率 反推「逐日流通股本」= volume / (turnover/100), 注入成本分布做衰减,
  因此即使没有逐日流通股本历史, 成本分布也对解禁天然稳健(解禁日换手率骤降 → 反推流通股本骤升 →
  当日注入筹码更多 → 成本峰自然下移)。

成本分布模型 (移动筹码分布 / 无限衰减)
--------------------------------------
对每只股票维护一个长度 n_bins 的价格分桶直方图 H:
  每日 t:
    1) 衰减: H = H * (1 - turnover_t/100)            # 当日成交比例的老筹码离场
    2) 注入: 在 [low_t, high_t] 上按三角分布(峰在 (open_t+close_t)/2)注入当日 volume_t
    3) 由 H 计算成本结构特征
价格分桶用「全历史 1%~99% 分位」的固定区间, 保证 H 跨日累积在同一价格轴上有效。

关键特征 (均为比率/归一量, 对 qfq 调整乘子不变, 故可直接与 qfq 日线湖混合)
---------------------------------------------------------------------------
  chip_profit_ratio : 获利盘比例 = 成本 <= 收盘价的筹码占比   (0~1)
  chip_cost_dev     : 平均成本偏离 = (close - 平均成本)/平均成本 (正=整体获利)
  chip_conc70       : 70% 成本集中度 = 中央70%筹码价格跨度 / 全区间 (越小越集中)
  chip_conc90       : 90% 成本集中度 = 中央90%筹码价格跨度 / 全区间
  chip_disp         : 成本离散度 = 成本分布标准差 / 平均成本 (变异系数)
  chip_skew         : 成本分布偏度 (负=筹码堆在低价区/套牢盘重)

内存: 逐股循环, 单股仅持有一个 n_bins 数组, 适配 8GB cgroup。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_mining.base_data import (
    list_stocks, load_panel, load_turnover, turnover_available, DEFAULT_START,
)

CHIP_FIELDS = [
    "chip_profit_ratio", "chip_cost_dev", "chip_conc70",
    "chip_conc90", "chip_disp", "chip_skew",
]


def _price_at_percentile(cum: np.ndarray, centers: np.ndarray, q: float) -> float:
    """在累积分布 cum(0~1) 上线性插值求分位 q 对应的价格。"""
    idx = np.searchsorted(cum, q)
    if idx <= 0:
        return centers[0]
    if idx >= len(centers):
        return centers[-1]
    c0, c1 = cum[idx - 1], cum[idx]
    if c1 - c0 < 1e-12:
        return centers[idx]
    frac = (q - c0) / (c1 - c0)
    return centers[idx - 1] + frac * (centers[idx] - centers[idx - 1])


def chip_single(open_, high, low, close, volume, turnover,
                n_bins: int = 100, warmup: int = 20) -> pd.DataFrame:
    """
    单只股票的成本分布特征。输入均为长度 T 的 1D 数组(已按日期对齐, 无 NaN 在首尾)。
    返回 date-indexed DataFrame, 列见 CHIP_FIELDS; warmup 天前为 NaN。
    """
    T = len(close)
    out = {k: np.full(T, np.nan, dtype="float32") for k in CHIP_FIELDS}

    # 固定价格分桶区间 (全历史 1%~99% 分位, 避免极端值撑宽)
    pmin = np.nanpercentile(low, 1)
    pmax = np.nanpercentile(high, 99)
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return pd.DataFrame(out)
    edges = np.linspace(pmin, pmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    span = pmax - pmin

    H = np.zeros(n_bins, dtype="float64")

    for t in range(T):
        to = turnover[t]
        if not np.isfinite(to) or to <= 0:
            # 无换手率(停牌/缺失): 仅衰减, 不注入
            H *= (1.0 - 0.0)
        else:
            decay = max(0.0, 1.0 - min(to, 100.0) / 100.0)
            H = H * decay
            lo, hi = low[t], high[t]
            if hi > lo and np.isfinite(lo) and np.isfinite(hi):
                # 三角权重: 在 [lo,hi] 内, 峰在 mid, 边缘为 0
                mid = 0.5 * (open_[t] + close[t])
                d = np.abs(centers - mid) / (hi - lo)
                w = np.where((centers >= lo) & (centers <= hi), np.maximum(0.0, 1.0 - d), 0.0)
                s = w.sum()
                if s > 0:
                    H = H + volume[t] * w / s

        if t < warmup:
            continue
        total = H.sum()
        if total <= 1e-12:
            continue
        cost_avg = float((centers * H).sum() / total)
        c = close[t]
        prof = float(H[centers <= c].sum() / total)
        # 分位
        cum = np.cumsum(H) / total
        p15 = _price_at_percentile(cum, centers, 0.15)
        p85 = _price_at_percentile(cum, centers, 0.85)
        p05 = _price_at_percentile(cum, centers, 0.05)
        p95 = _price_at_percentile(cum, centers, 0.95)
        var = float(((centers - cost_avg) ** 2 * H).sum() / total)
        m3 = float(((centers - cost_avg) ** 3 * H).sum() / total)
        sd = var ** 0.5
        skew = m3 / (var ** 1.5 + 1e-12)

        out["chip_profit_ratio"][t] = prof
        out["chip_cost_dev"][t] = (c - cost_avg) / (cost_avg + 1e-12)
        out["chip_conc70"][t] = (p85 - p15) / span
        out["chip_conc90"][t] = (p95 - p05) / span
        out["chip_disp"][t] = sd / (cost_avg + 1e-12)
        out["chip_skew"][t] = skew

    return pd.DataFrame(out)


def build_chip_panels(codes: list[str] | None = None, n_bins: int = 100,
                      start: str | None = DEFAULT_START,
                      end: str | None = None) -> dict[str, pd.DataFrame]:
    """
    对 codes 逐股计算筹码结构特征面板, 返回 {chip_field: date×stock DataFrame}。

    turnover 湖缺失则抛 RuntimeError(调用方应先用 turnover_available() 判断)。
    单只股票某特征在全期缺失(如退市早/无换手率)则该列该股票为全 NaN, 由 XGBoost 原生处理。
    """
    if not turnover_available():
        raise RuntimeError("turnover 湖不存在, 请先运行 build_turnover_lake.py (B1)")
    if codes is None:
        codes = list_stocks()

    # 一次性加载所需面板
    o = load_panel("open", codes=codes, start=start, end=end)
    h = load_panel("high", codes=codes, start=start, end=end)
    l = load_panel("low", codes=codes, start=start, end=end)
    c = load_panel("close", codes=codes, start=start, end=end)
    v = load_panel("volume", codes=codes, start=start, end=end)
    to = load_turnover(codes=codes, start=start, end=end)

    panels: dict[str, list] = {k: [] for k in CHIP_FIELDS}
    idx_list: list[pd.Index] = []
    code_list: list[str] = []

    for s in codes:
        if s not in c.columns or s not in to.columns:
            # 无 OHLCV 或无情手率(退市/抓取失败) → 跳过, 下游 XGBoost 不缺此股
            continue
        # 取该股各序列, 按日期内连接对齐 (OHLCV 与 turnover 可能交易日不完全一致)
        df = pd.DataFrame({
            "open": o[s], "high": h[s], "low": l[s],
            "close": c[s], "volume": v[s], "turnover": to[s],
        }).dropna(subset=["close", "open", "high", "low", "volume"])
        if len(df) < 60:
            continue
        res = chip_single(
            df["open"].to_numpy(dtype="float64"),
            df["high"].to_numpy(dtype="float64"),
            df["low"].to_numpy(dtype="float64"),
            df["close"].to_numpy(dtype="float64"),
            df["volume"].to_numpy(dtype="float64"),
            df["turnover"].to_numpy(dtype="float64"),
            n_bins=n_bins,
        )
        res = res.set_index(df.index)
        for k in CHIP_FIELDS:
            panels[k].append(res[k].rename(s))
        idx_list.append(res.index)
        code_list.append(s)

    if not code_list:
        raise RuntimeError("没有可计算筹码结构的股票(检查数据对齐)")

    out = {}
    for k in CHIP_FIELDS:
        out[k] = pd.concat(panels[k], axis=1).sort_index()
        out[k].columns = code_list
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    codes = list_stocks(50)
    chips = build_chip_panels(codes)
    print(f"[chip] {len(codes)} 只 | 特征: {list(chips)} | {time.time()-t0:.1f}s")
    for k, p in chips.items():
        print(f"  {k}: shape={p.shape} mean={np.nanmean(p.values):.4f}")
