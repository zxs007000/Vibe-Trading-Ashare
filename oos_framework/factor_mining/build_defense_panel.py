#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5c · 防御面板预计算 (持仓倾斜用, 仅价量波动 + chip, 无基本面/PIT泄漏)
=========================================================================
从基础面板算每只票每日的防御度成分, 落盘 factor_mining/defense_panel.parquet
供 gate_full_v4 快速加载(避免每次重跑 derive_variables).

D̃ 成分(截面 z 分位后组合):
  -vol_20        (已实现波动, 低=防御)
  -ivol_60       (特质波动, 低=防御)  = 市场模型残差滚动std
  -downside_vol_60 (下行波动, 低=防御)
  +chip_disp     (筹码分散, 高=稳定)
  -chip_conc90   (筹码集中, 低=防御)
均来自价格/成交量/chip成本分布, 无任何财报PIT对齐问题.
"""
from __future__ import annotations
import os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OOS_ROOT = HERE.parent
sys.path.insert(0, str(OOS_ROOT))

from factor_mining.base_data import load_panel, load_chip_panels_from_lake
from factor_mining.universe import load_universe

OUT = "factor_mining/defense_panel.parquet"
START = "2021-01-01"     # OOS 窗 + 预热(20/60日窗口)
END = None


def build() -> pd.DataFrame:
    t0 = time.time()
    codes = load_universe()
    print(f"[defense] 池 {len(codes)} 只, 加载 close...", flush=True)
    close = load_panel("close", codes=codes, start=START, end=END)
    close = close.sort_index()
    ret = close.pct_change()

    # --- 价量波动族 ---
    vol_20 = ret.rolling(20, min_periods=10).std()
    # 特质波动: 市场模型残差滚动std (等权市场收益作基准)
    # 注意: cov 按股票索引, mkt 按日期索引, 必须用 numpy 广播, 不能用 pandas 对齐乘法
    mkt = ret.mean(axis=1)
    mkt_arr = mkt.values
    mkt_dm = mkt_arr - mkt_arr.mean()
    ret_arr = ret.values
    cov_xm = (ret_arr - ret_arr.mean(axis=0)) * mkt_dm[:, None]
    beta = cov_xm.sum(axis=0) / (mkt_dm ** 2).sum()
    resid = ret_arr - beta[None, :] * mkt_arr[:, None]
    resid = pd.DataFrame(resid, index=ret.index, columns=ret.columns)
    ivol_60 = resid.rolling(60, min_periods=30).std()
    downside = ret.clip(upper=0.0)
    downside_vol_60 = downside.rolling(60, min_periods=30).std()
    print(f"[defense] 波动族算完 {vol_20.shape} | {time.time()-t0:.0f}s", flush=True)

    # --- chip 系(从 lake 直接读, 快) ---
    chip = load_chip_panels_from_lake(codes=codes, start=START, end=END)
    chip_disp = chip.get("chip_disp")
    chip_conc90 = chip.get("chip_conc90")
    print(f"[defense] chip 读入 disp={chip_disp is not None} conc90={chip_conc90 is not None}", flush=True)

    # --- 堆叠为长表 ---
    frames = []
    comp = {
        "vol_20": vol_20, "ivol_60": ivol_60,
        "downside_vol_60": downside_vol_60,
        "chip_disp": chip_disp, "chip_conc90": chip_conc90,
    }
    for name, pan in comp.items():
        if pan is None:
            print(f"[defense] 警告: {name} 缺失, 跳过", flush=True)
            continue
        p = pan.reindex(index=close.index, columns=codes)
        lp = p.stack().rename(name)
        frames.append(lp)
    df = pd.concat(frames, axis=1).dropna(how="all")
    df = df.reset_index().rename(columns={"level_0": "date", "level_1": "code"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    df.to_parquet(OUT)
    print(f"[defense] 落盘 {OUT} | {df.shape} | {time.time()-t0:.1f}min", flush=True)
    return df


if __name__ == "__main__":
    try:
        if os.path.exists(OUT):
            print(f"[defense] 已存在 {OUT}, 直接复用 (删除后重算)", flush=True)
        else:
            build()
    except Exception:
        traceback.print_exc()
        raise
