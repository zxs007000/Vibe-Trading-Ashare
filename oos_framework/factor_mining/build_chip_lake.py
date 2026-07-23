#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 全市场筹码面板构建 (落盘)
=========================================
逐股计算 chip_structure 6 维特征, 落 $STOCKLAKE/chip_panels/<field>.parquet
(date×stock 面板, float32)。增量: 已在面板中的股票跳过(--force 全量重算)。

用法:
  python -m factor_mining.build_chip_lake [--limit N] [--force]

内存: 逐股循环仅持有单股数组 + 6 个结果列表, 全市场 5500 只输出面板
约 6 × (1900日 × 5500股 × 4B) ≈ 250MB, 32GB 下无压力。
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from factor_mining.base_data import LAKE, DAILY, TURNOVER_DIR, turnover_available
from factor_mining.chip_structure import chip_single, CHIP_FIELDS
from factor_mining.universe import load_universe

CHIP_DIR = os.path.join(LAKE, "chip_panels")


def build(limit: int | None = None, force: bool = False):
    if not turnover_available():
        raise RuntimeError("turnover 湖不存在")
    os.makedirs(CHIP_DIR, exist_ok=True)

    codes = load_universe()
    if codes is None:
        from factor_mining.base_data import list_stocks
        codes = list_stocks()
        print("[chip_lake] 警告: 无 universe 名单, 用全市场")
    if limit:
        codes = codes[:limit]

    # 增量: 已有面板的列跳过
    existing_cols: set[str] = set()
    first_panel = os.path.join(CHIP_DIR, f"{CHIP_FIELDS[0]}.parquet")
    old_panels: dict[str, pd.DataFrame] = {}
    if os.path.exists(first_panel) and not force:
        for k in CHIP_FIELDS:
            p = os.path.join(CHIP_DIR, f"{k}.parquet")
            if os.path.exists(p):
                old_panels[k] = pd.read_parquet(p)
        if old_panels:
            existing_cols = set(old_panels[CHIP_FIELDS[0]].columns)
    todo = [c for c in codes if c not in existing_cols]
    print(f"[chip_lake] 池 {len(codes)} | 已有 {len(existing_cols)} | 待算 {len(todo)}", flush=True)

    new_cols: dict[str, list[pd.Series]] = {k: [] for k in CHIP_FIELDS}
    t0 = time.time()
    done = 0
    for i, s in enumerate(todo, 1):
        fd = os.path.join(DAILY, f"{s}.parquet")
        ft = os.path.join(TURNOVER_DIR, f"{s}.parquet")
        if not (os.path.exists(fd) and os.path.exists(ft)):
            continue
        d = pd.read_parquet(fd)
        if "date" not in d.columns:
            d = d.reset_index()
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date").sort_index()

        t = pd.read_parquet(ft)
        if "date" not in t.columns:
            t = t.reset_index()
        if "日期" in t.columns:
            t = t.rename(columns={"日期": "date"})
        t["date"] = pd.to_datetime(t["date"])
        t = t.set_index("date").sort_index()

        df = d[["open", "high", "low", "close", "volume"]].join(t["turnover"], how="left")
        df = df.dropna(subset=["close", "open", "high", "low", "volume"])
        if len(df) < 60:
            continue
        res = chip_single(
            df["open"].to_numpy("float64"), df["high"].to_numpy("float64"),
            df["low"].to_numpy("float64"), df["close"].to_numpy("float64"),
            df["volume"].to_numpy("float64"), df["turnover"].to_numpy("float64"))
        res = res.set_index(df.index)
        for k in CHIP_FIELDS:
            new_cols[k].append(res[k].rename(s))
        done += 1
        if i % 200 == 0 or i == len(todo):
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print(f"[chip_lake] {i}/{len(todo)} ok={done} 耗时{el/60:.1f}min ETA {eta/60:.1f}min", flush=True)

    if not new_cols[CHIP_FIELDS[0]] and not old_panels:
        print("[chip_lake] 无新增且无旧面板, 退出")
        return
    for k in CHIP_FIELDS:
        parts = []
        if k in old_panels:
            parts.append(old_panels[k])
        if new_cols[k]:
            parts.append(pd.concat(new_cols[k], axis=1))
        panel = pd.concat(parts, axis=1).sort_index().astype("float32")
        panel = panel.loc[:, ~panel.columns.duplicated()]
        panel.to_parquet(os.path.join(CHIP_DIR, f"{k}.parquet"))
        print(f"[chip_lake] {k}: {panel.shape} 落盘", flush=True)
    print(f"[chip_lake] 完成 | 总耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(limit=a.limit, force=a.force)
