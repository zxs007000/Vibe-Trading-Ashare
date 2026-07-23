#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4-full · 全历史(20年)分块 WFA (三维度 v2 因子, 复用40入选因子)
==============================================================
与 run_wfa_v2 同流程, 但把特征表起点拉到 2005 (start='2005-01-01'),
使 wfa_folds(TRAIN=3y/STEP=1y) 自动扩到 ~17 折, 覆盖 2008/2015/2016 股灾年。
产出 oos_detail_full.parquet(全历史融合概率), 供股灾年双尾防御对比。
独立 out_dir (wfa_v2_store_full) + 独立结果落盘, 不覆盖 4年版。
"""
from __future__ import annotations
import json, os, time, traceback
from pathlib import Path

import pandas as pd

from factor_mining.factor_wfa import (
    _l2t, build_feature_table_chunked, run_wfa_chunked, backtest, write_results)
from factor_mining.universe import load_universe

HERE = Path(__file__).resolve().parent
OUT_DIR = "factor_mining/wfa_v2_store_full"
RESULT_MD = "factor_mining/RESULT_v2_3dim_full.md"
SELECTED_JSON = "factor_mining/factors_v2_selected.json"
START = "2005-01-01"   # 20年回溯起点(因子lookback需前序, 故早于此)


def main():
    t0 = time.time()
    with open("factor_mining/factors_v2_3dim.json", encoding="utf-8") as f:
        d = json.load(f)
    seen = {k: {"expr_tuple": _l2t(v["expr_tuple_list"]), "dir": v["dir"],
                "ic20": v["ic20"], "icir20": v["icir20"]} for k, v in d.items()}
    sel_names = json.load(open(SELECTED_JSON, encoding="utf-8"))
    exprs = [(n, seen[n]["expr_tuple"]) for n in sel_names if n in seen]
    print(f"[wfa_full] 复用已筛选 {len(exprs)} 因子 | 起点 {START}", flush=True)

    codes = load_universe()
    print(f"[wfa_full] 全池 {len(codes)} 只 | 因子 {len(exprs)} | 分块特征表(起点{START})", flush=True)
    t1 = time.time()
    feat_dir, feat_cols = build_feature_table_chunked(
        codes, exprs, out_dir=OUT_DIR, chunk_months=12, start=START)
    print(f"[wfa_full] 特征表完成 {len(feat_cols)} 列 | {(time.time()-t1)/60:.1f}min", flush=True)

    t2 = time.time()
    rows, imp_acc, folds, oos_detail = run_wfa_chunked(feat_dir, feat_cols)
    print(f"[wfa_full] WFA 完成 {len(rows)} 折: " +
          " | ".join(f"折{i+1} {s.date()}~{e.date()}" for i,(_,_,s,e) in enumerate(folds)),
          flush=True)
    print(f"[wfa_full] WFA 耗时 {(time.time()-t2)/60:.1f}min", flush=True)
    oos_detail.to_parquet("factor_mining/oos_detail_full.parquet", index=False)

    bt = backtest(oos_detail, top_frac=0.3)
    print(f"[wfa_full] 回测: {bt}", flush=True)
    meta = {"mine": 300, "stocks": len(codes), "n_total": len(seen), "start": START}
    write_results(RESULT_MD, meta, rows, imp_acc, folds, n_used=len(exprs), bt=bt)
    print(f"[wfa_full] 全部完成 → {RESULT_MD} | 总耗时 {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
