#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 数据底座
================================
从 /workspace/stocklake 加载「日期 × 股票」面板，供算子库 / 网格搜索 / 遗传规划 / MCTS 共用。

设计要点（承接 8GB cgroup 实战经验）:
- 单字段面板(date×stock, float32) 在 5448 只全市场下仅 ~42MB，远小于 XGBoost 的 218 维特征矩阵，
  因此因子挖掘阶段可放心加载较大样本（默认 400 只，全市场亦可）。
- 读取采用「逐只读列 → 拼 DataFrame → 按日期对齐」的方式，避免一次性读入整库。
"""
from __future__ import annotations
import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # 抑制 ts_std 在常值窗口产生的 NaN 噪声告警

LAKE = os.environ.get("STOCKLAKE", "/workspace/stocklake")
DAILY = os.path.join(LAKE, "daily")

# 数据湖 daily 层提供的原始基础变量（算子库即围绕它们组合）
BASE_FIELDS = ["open", "high", "low", "close", "volume", "amount"]
# 默认回测起始（与全市场 WFA 一致）
DEFAULT_START = "2018-10-23"


def list_stocks(limit: int | None = None) -> list[str]:
    """列出数据湖中可用的股票代码（按文件名排序）。"""
    fs = sorted(glob.glob(os.path.join(DAILY, "*.parquet")))
    codes = [os.path.basename(f)[:6] for f in fs]  # 000001.parquet -> 000001
    if limit:
        codes = codes[:limit]
    return codes


def load_panel(field: str, codes: list[str] | None = None, start: str | None = DEFAULT_START,
               end: str | None = None, dtype=np.float32) -> pd.DataFrame:
    """
    加载某字段的「日期 × 股票」面板。

    参数
    ----
    field : 'open'|'high'|'low'|'close'|'volume'|'amount'
    codes : 股票代码列表；None 表示全部
    start/end : 'YYYY-MM-DD' 字符串过滤
    """
    if field not in BASE_FIELDS:
        raise ValueError(f"field 必须是 {BASE_FIELDS} 之一, 收到 {field!r}")
    if codes is None:
        codes = list_stocks()
    series = {}
    for c in codes:
        f = os.path.join(DAILY, f"{c}.parquet")
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f, columns=["date", field])
        s = df.set_index("date")[field]
        s.index = pd.to_datetime(s.index)  # 统一为 Timestamp, 避免 date/Timestamp 混用
        series[c] = s
    if not series:
        raise RuntimeError("没有任何股票数据被加载, 检查 STOCKLAKE 路径")
    panel = pd.DataFrame(series)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    if start:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if end:
        panel = panel[panel.index <= pd.Timestamp(end)]
    return panel.astype(dtype)


def load_base_data(codes: list[str] | None = None, fields: list[str] | None = None,
                   start: str | None = DEFAULT_START, end: str | None = None) -> dict[str, pd.DataFrame]:
    """一次性加载多个基础变量面板, 返回 {field: panel}。"""
    fields = fields or BASE_FIELDS
    return {f: load_panel(f, codes=codes, start=start, end=end) for f in fields}


def forward_returns(close_panel: pd.DataFrame, horizons=(5, 20, 60)) -> dict[int, pd.DataFrame]:
    """
    计算各 horizon 的向前收益率面板（t 时刻持有 h 日的收益）。
    因子在 t 日的值用于预测该向前收益, 是 IC 计算的标准目标。
    """
    out = {}
    for h in horizons:
        out[h] = close_panel.shift(-h) / close_panel - 1.0
    return out


def derive_variables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    从 6 个基础字段派生标准 alpha 变量池(对应报告「将 218 个特征作为原始变量」的思路)。
    返回 {原名: 面板} ∪ {派生名: 面板}, 派生变量包括:
      收益类   : ret_1/5/10/20, log_ret_1
      波动类   : vol_20(已实现波动), amp_1(日振幅), range_20(20日波幅)
      量价类   : vrate_5/20(量比), arate_5(额比), dist_high_20/dist_low_20(距高低点)
    这些变量叠加时序/截面算子后, 才能产生有截面区分度的因子(纯价格水位 IC≈0)。
    """
    out = dict(data)
    close, high, low, open_, volume, amount = (
        data["close"], data["high"], data["low"], data["open"], data["volume"], data["amount"])
    r1 = close.pct_change(1)
    out["ret_1"] = r1
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_20"] = close.pct_change(20)
    out["log_ret_1"] = np.log(close / close.shift(1))
    out["vol_20"] = r1.rolling(20, min_periods=10).std()
    out["amp_1"] = (high - low) / close
    out["range_20"] = (high.rolling(20, min_periods=10).max() - low.rolling(20, min_periods=10).min()) / close
    out["vrate_5"] = volume / volume.rolling(5, min_periods=3).mean()
    out["vrate_20"] = volume / volume.rolling(20, min_periods=10).mean()
    out["arate_5"] = amount / amount.rolling(5, min_periods=3).mean()
    out["dist_high_20"] = close / close.rolling(20, min_periods=10).max()
    out["dist_low_20"] = close / close.rolling(20, min_periods=10).min()
    return out


if __name__ == "__main__":
    codes = list_stocks(400)
    data = load_base_data(codes)
    fwd = forward_returns(data["close"])
    print(f"样本: {len(codes)} 只, 交易日: {len(data['close'])}")
    print(f"面板形状: {data['close'].shape}, 向前收益(20日)非空率: {1-fwd[20].isna().mean().mean():.3f}")
