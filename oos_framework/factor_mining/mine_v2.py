#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四方向全量挖掘(三维度变量池) → factors_v2_3dim.json。每方向单独 try + 增量落盘。"""
from __future__ import annotations

import json
import time
import traceback
from collections import Counter

import numpy as np

from factor_mining.base_data import list_stocks, load_base_data, derive_variables, forward_returns
from factor_mining.factor_wfa import _t2l
from factor_mining.grid_search import grid_search
from factor_mining.genetic_programming import evolve
from factor_mining.llm_mcts import MCTSAgent
from factor_mining.xgb_interaction import mine_interactions
from factor_mining.universe import load_universe

OUT = "factor_mining/factors_v2_3dim.json"
N_MINE = 300
SEEDS = (42, 7, 123)


def save(seen):
    dump = {k: {"expr_tuple_list": _t2l(v["expr_tuple"]), "dir": v["dir"],
                "ic20": float(v["ic20"]), "icir20": float(v["icir20"])}
            for k, v in seen.items()}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False)


def main():
    t0 = time.time()
    uni = load_universe()
    if uni:
        rng = np.random.default_rng(42)
        codes = sorted(rng.choice(uni, min(N_MINE, len(uni)), replace=False).tolist())
        print(f"[mine_v2] 过滤池采样 {len(codes)}/{len(uni)}", flush=True)
    else:
        codes = list_stocks(N_MINE)
    data = derive_variables(load_base_data(codes))
    fwd = forward_returns(data["close"], horizons=(5, 10, 20, 60))
    n_chip = sum(1 for k in data if k.startswith("chip_"))
    print(f"[mine_v2] 变量池 {len(data)} (chip {n_chip}) | {time.time()-t0:.0f}s", flush=True)

    seen: dict = {}

    # ---- 方向1 grid ----
    t = time.time()
    try:
        gs = grid_search(data, fwd, top_k=10 ** 7, require_ic_pos=True)
        for r in gs:
            seen.setdefault(r["expr"], {"expr_tuple": r["expr_tuple"], "dir": 1,
                                        "ic20": r["ic20"], "icir20": r["icir20"]})
        print(f"[mine_v2] 方向1 grid: {len(gs)} | {time.time()-t:.0f}s", flush=True)
    except Exception:
        traceback.print_exc()
    save(seen)

    # ---- 方向2 GP ----
    t = time.time()
    try:
        gp_n = 0
        for sd in SEEDS:
            res = evolve(data, fwd, pop_size=30, generations=12, seed=sd, verbose=False)
            for p in res["pareto"]:
                if p["expr"] not in seen:
                    seen[p["expr"]] = {"expr_tuple": p["expr_tuple"], "dir": 2,
                                       "ic20": p["ic20"], "icir20": p["icir20"]}
                    gp_n += 1
            print(f"[mine_v2] 方向2 GP seed={sd}: 累计新增 {gp_n} | {time.time()-t:.0f}s", flush=True)
    except Exception:
        traceback.print_exc()
    save(seen)

    # ---- 方向3 MCTS ----
    t = time.time()
    try:
        agent = MCTSAgent(data, fwd, max_depth=2, llm=None, seed=42)
        cands = agent.search(iterations=150, verbose=False)
        mcts_n = 0
        for c in cands:
            if c["icir"] > 0 and c["expr"] not in seen:
                seen[c["expr"]] = {"expr_tuple": c["expr_tuple"], "dir": 3,
                                   "ic20": c["ic20"], "icir20": c["icir"]}
                mcts_n += 1
        print(f"[mine_v2] 方向3 MCTS: 新增 {mcts_n} | {time.time()-t:.0f}s", flush=True)
    except Exception:
        traceback.print_exc()
    save(seen)

    # ---- 方向4 XGB 交互 ----
    t = time.time()
    try:
        xgb_res = mine_interactions(data, fwd, horizons=(5, 10, 20))
        xgb_n = 0
        for r in xgb_res:
            if r["expr"] not in seen:
                seen[r["expr"]] = {"expr_tuple": r["expr_tuple"], "dir": 4,
                                   "ic20": r["ic20"], "icir20": r["icir20"]}
                xgb_n += 1
        print(f"[mine_v2] 方向4 XGB: 新增 {xgb_n} | {time.time()-t:.0f}s", flush=True)
    except Exception:
        traceback.print_exc()
    save(seen)

    c = Counter(v["dir"] for v in seen.values())
    print(f"[mine_v2] TOTAL {len(seen)} | dir {dict(sorted(c.items()))} | {(time.time()-t0)/60:.1f}min", flush=True)
    chip_hit = [k for k in seen if "chip" in k]
    print(f"[mine_v2] 含chip变量: {len(chip_hit)}", flush=True)
    for k in sorted(seen, key=lambda x: -seen[x]["icir20"])[:20]:
        v = seen[k]
        print(f"  dir{v['dir']} ic={v['ic20']:+.4f} icir={v['icir20']:+.3f}  {k[:90]}", flush=True)


if __name__ == "__main__":
    main()
