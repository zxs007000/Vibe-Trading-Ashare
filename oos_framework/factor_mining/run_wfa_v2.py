#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4 · 全池分块 WFA (三维度 v2 因子)
====================================
流程: factors_v2_3dim.json → P3 衰减筛选(40 入选) → 全池(universe 5175)
      build_feature_table_chunked → run_wfa_chunked → backtest → 结果落盘。
每阶段独立计时 + flush 日志 + 增量落盘(纪律)。
"""
from __future__ import annotations

import json
import os
import time
import traceback

import pandas as pd

from factor_mining.factor_wfa import (
    _l2t, build_feature_table_chunked, run_wfa_chunked, backtest, write_results)
from factor_mining.factor_screen import screen_factors
from factor_mining.universe import load_universe

OUT_DIR = "factor_mining/wfa_v2_store"
RESULT_MD = "factor_mining/RESULT_v2_3dim.md"
SELECTED_JSON = "factor_mining/factors_v2_selected.json"


def main():
    t0 = time.time()
    # ---- 1) 载入 118 候选 → P3 筛选 ----
    with open("factor_mining/factors_v2_3dim.json", encoding="utf-8") as f:
        d = json.load(f)
    seen = {k: {"expr_tuple": _l2t(v["expr_tuple_list"]), "dir": v["dir"],
                "ic20": v["ic20"], "icir20": v["icir20"]} for k, v in d.items()}
    if os.path.exists(SELECTED_JSON):
        sel_names = json.load(open(SELECTED_JSON, encoding="utf-8"))
        exprs = [(n, seen[n]["expr_tuple"]) for n in sel_names if n in seen]
        print(f"[wfa_v2] 复用已筛选 {len(exprs)} 因子", flush=True)
    else:
        exprs = screen_factors(seen, n_sub=400)
        with open(SELECTED_JSON, "w", encoding="utf-8") as f:
            json.dump([n for n, _ in exprs], f, ensure_ascii=False)
        print(f"[wfa_v2] P3 筛选完成 {len(exprs)} 因子, 已落盘", flush=True)

    # ---- 2) 全池特征表(分块) ----
    codes = load_universe()
    print(f"[wfa_v2] 全池 {len(codes)} 只 | 因子 {len(exprs)} | 开始分块特征表", flush=True)
    t1 = time.time()
    feat_dir, feat_cols = build_feature_table_chunked(
        codes, exprs, out_dir=OUT_DIR, chunk_months=12)
    print(f"[wfa_v2] 特征表完成 {len(feat_cols)} 列 | {(time.time()-t1)/60:.1f}min", flush=True)

    # ---- 3) WFA ----
    t2 = time.time()
    rows, imp_acc, folds, oos_detail = run_wfa_chunked(feat_dir, feat_cols)
    print(f"[wfa_v2] WFA 完成 {len(rows)} 折 | {(time.time()-t2)/60:.1f}min", flush=True)
    oos_detail.to_parquet("factor_mining/oos_detail_v2.parquet", index=False)

    # ---- 4) 回测 + 落盘 ----
    bt = backtest(oos_detail, top_frac=0.3)
    print(f"[wfa_v2] 回测: {bt}", flush=True)
    meta = {"mine": 300, "stocks": len(codes), "n_total": len(seen)}
    write_results(RESULT_MD, meta, rows, imp_acc, folds, n_used=len(exprs), bt=bt)
    print(f"[wfa_v2] 全部完成 → {RESULT_MD} | 总耗时 {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
