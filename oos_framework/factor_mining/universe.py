#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 股票池过滤 (Universe)
=====================================
按用户拍板的三条规则过滤全市场股票池:
  1. 剔 ST/*ST/退市股  —— 按股票名称匹配(需 metadata/stock_names.parquet, 缺失时警告跳过)
  2. 剔次新股          —— 日线历史不足 MIN_LISTED_DAYS(默认 250 交易日 ≈ 1年)
  3. 剔低流动性        —— 近 LIQ_WINDOW 日均成交额 < MIN_AVG_AMOUNT(默认 3000 万元)

设计
----
- 静态过滤(构建期一次): 用于 collect_factors / build_feature_table 的入池名单。
- 名称表由 stockworm 侧 fetch_stock_names.py 落盘(akshare 全 A 名单快照)。
  注意: 名称是当前快照, 历史 ST 状态无 PIT。对挖因子影响有限(ST 股权重本就低),
  严格 PIT 版需接历史 ST 记录, 留作后续增强。
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

from factor_mining.base_data import LAKE, DAILY, list_stocks, load_panel

STOCK_NAMES = os.path.join(LAKE, "metadata", "stock_names.parquet")
UNIVERSE_META = os.path.join(LAKE, "metadata", "universe_stats.json")

MIN_LISTED_DAYS = 250          # 次新: 至少 250 个交易日
LIQ_WINDOW = 60                # 流动性: 近 60 日
MIN_AVG_AMOUNT = 3e7           # 日均成交额 >= 3000 万元


def _load_names() -> pd.Series | None:
    """code -> name 映射; 缺失返回 None。"""
    if not os.path.exists(STOCK_NAMES):
        return None
    df = pd.read_parquet(STOCK_NAMES)
    if "code" in df.columns:
        df = df.set_index("code")
    df.index = df.index.astype(str).str[:6]
    col = "name" if "name" in df.columns else df.columns[0]
    return df[col].astype(str)


def build_universe(min_listed_days: int = MIN_LISTED_DAYS,
                   liq_window: int = LIQ_WINDOW,
                   min_avg_amount: float = MIN_AVG_AMOUNT,
                   save_stats: bool = True,
                   verbose: bool = True) -> list[str]:
    """
    返回过滤后的股票代码列表, 并落 metadata/universe_stats.json 记录各环节剔除数。
    """
    codes = list_stocks()
    n0 = len(codes)

    # ---- 1) ST 过滤 (按名称) ----
    names = _load_names()
    st_removed: list[str] = []
    if names is not None:
        keep = []
        for c in codes:
            nm = names.get(c, "")
            if ("ST" in nm.upper()) or ("退" in nm):
                st_removed.append(c)
            else:
                keep.append(c)
        codes = keep
    elif verbose:
        print("[universe] 警告: 无 stock_names.parquet, 跳过 ST 过滤 (先运行 fetch_stock_names.py)")

    # ---- 2)+3) 次新 & 流动性: 只需 amount 面板 ----
    amt = load_panel("amount", codes=codes, start=None)
    listed_days = amt.notna().sum(axis=0)
    new_removed = listed_days[listed_days < min_listed_days].index.tolist()

    recent_amt = amt.tail(liq_window).mean(axis=0)
    illiq_removed = recent_amt[recent_amt < min_avg_amount].index.tolist()
    illiq_removed = [c for c in illiq_removed if c not in set(new_removed)]

    removed = set(new_removed) | set(illiq_removed)
    final = [c for c in codes if c not in removed]

    stats = {
        "total": n0,
        "st_removed": len(st_removed),
        "new_removed": len(new_removed),
        "illiq_removed": len(illiq_removed),
        "final": len(final),
        "params": {"min_listed_days": min_listed_days, "liq_window": liq_window,
                   "min_avg_amount": min_avg_amount},
    }
    if verbose:
        print(f"[universe] 全市场 {n0} → 剔ST {len(st_removed)} → 剔次新 {len(new_removed)}"
              f" → 剔低流动性 {len(illiq_removed)} → 终池 {len(final)}")
    if save_stats:
        os.makedirs(os.path.dirname(UNIVERSE_META), exist_ok=True)
        with open(UNIVERSE_META, "w", encoding="utf-8") as f:
            json.dump({**stats, "codes": final}, f, ensure_ascii=False)
    return final


def load_universe() -> list[str] | None:
    """读取上次 build_universe 落盘的名单; 缺失返回 None。"""
    if not os.path.exists(UNIVERSE_META):
        return None
    with open(UNIVERSE_META, encoding="utf-8") as f:
        return json.load(f).get("codes")


if __name__ == "__main__":
    build_universe()
