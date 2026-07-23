#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_mining · 因子 → XGBoost WFA 验证
========================================
把 factor_mining 挖出的因子表达式当作特征, 用 xgboost_wfa 的方法论做 Walk-Forward
验证(WFA), 检验这些「机器挖出来的因子」在**样本外(OOS)**是否真有区分度。

口径严格对齐 xgboost_wfa/xgb_wfa_proto_v4_chunked.py:
  - 标签: 未来收益排全市场前 30% → 1 (horizon 5/20/60 日, 从 T+1 起算防泄露)
  - WFA: 训练 3y / 测试 1y / 步长 1y, 共 4 折
  - 模型: XGBoost 二分类; 每折 IS 内 RandomizedSearchCV 调参 + 3 horizon 模型, 融合概率均值
  - 指标: OOS AUC(单周期5日 + 融合) + OOS IC(Spearman, 融合)

子命令
------
  python factor_wfa.py collect [--mine 300]            # 挖掘并统计因子总数(四方向去重)
  python factor_wfa.py wfa [--mine 300] [--stocks 1500] [--out results.md]
  python factor_wfa.py all  [--mine 300] [--stocks 1500] [--out results.md]

说明
----
  collect 在 --mine 只股票上挖掘, 统计「正 IC 且去重」的因子总数(四方向合并)。
  wfa 用这些因子在 --stocks 只代表性股票上构建特征, 跑 WFA。因子数过多时自动按
  ICIR 截取到 200 个(报告会注明「发现 N / 用于验证 M」)。
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import gc
import glob
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # oos_framework

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from factor_mining import (
    list_stocks, load_base_data, load_panel, derive_variables, forward_returns,
    evaluate_expr, expr_to_str, grid_search, evolve, MCTSAgent,
    turnover_available,
)

RANDOM_STATE = 42
HORIZONS = (5, 20, 60)
TRAIN_YEARS, TEST_YEARS, STEP_YEARS = 3, 1, 1
MAX_FACTORS = 200  # WFA 验证用的因子数上限(过多时按 ICIR 截取)


def _t2l(o):
    """嵌套元组 → 列表(可 JSON 序列化)。"""
    return [_t2l(x) for x in o] if isinstance(o, tuple) else o


def _l2t(o):
    """列表 → 嵌套元组(还原表达式 DSL)。"""
    return tuple(_l2t(x) for x in o) if isinstance(o, list) else o


# ---------------------------------------------------------------------------
# WFA 分折(对齐 v4: 训练3y/测试1y/步长1y)
# ---------------------------------------------------------------------------
def wfa_folds(dates):
    d0 = pd.Timestamp(dates.min())
    d1 = pd.Timestamp(dates.max())
    folds, start = [], d0
    while True:
        is_s, is_e = start, start + pd.DateOffset(years=TRAIN_YEARS)
        oos_s, oos_e = is_e, is_e + pd.DateOffset(years=TEST_YEARS)
        if oos_e > d1:
            break
        folds.append((is_s, is_e, oos_s, oos_e))
        start = start + pd.DateOffset(years=STEP_YEARS)
    return folds


# ---------------------------------------------------------------------------
# 方向 collect: 四方向挖掘 + 去重统计
# ---------------------------------------------------------------------------
def collect_factors(n_mine: int = 300, seeds=(42, 7, 123), use_universe: bool = True,
                    with_xgb: bool = True, mine_horizons=(5, 10, 20)):
    t0 = time.time()
    codes = None
    if use_universe:
        try:
            from factor_mining.universe import load_universe
            uni = load_universe()
            if uni:
                rng = np.random.default_rng(42)
                codes = sorted(rng.choice(uni, min(n_mine, len(uni)), replace=False).tolist())
                print(f"[collect] 用过滤池采样 {len(codes)}/{len(uni)} 只 (剔ST/次新/低流动性)", flush=True)
        except Exception:
            pass
    if codes is None:
        codes = list_stocks(n_mine)
    data = derive_variables(load_base_data(codes))
    fwd = forward_returns(data["close"], horizons=tuple(sorted(set(mine_horizons) | {20, 60})))
    n_chip = sum(1 for k in data if k.startswith("chip_"))
    print(f"[collect] 挖掘样本 {len(codes)} 只, 变量池 {len(data)} (含chip {n_chip}) | 耗时 {time.time()-t0:.1f}s", flush=True)

    seen = {}  # expr_str -> meta

    # 方向1 网格: 返回所有正 IC 候选(require_ic_pos + 大 top_k)
    t = time.time()
    gs = grid_search(data, fwd, top_k=10 ** 7, require_ic_pos=True)
    for r in gs:
        seen.setdefault(r["expr"], {"expr_tuple": r["expr_tuple"], "dir": 1,
                                    "ic20": r["ic20"], "icir20": r["icir20"],
                                    "turnover": r["turnover"]})
    print(f"[collect] 方向1 网格: 正IC候选 {len(gs)} 个 | {time.time()-t:.1f}s", flush=True)

    # 方向2 遗传规划: 多 seed 的 Pareto 前沿(去重)
    t = time.time()
    gp_n = 0
    for sd in seeds:
        res = evolve(data, fwd, pop_size=30, generations=12, seed=sd, verbose=False)
        for p in res["pareto"]:
            if p["expr"] not in seen:
                seen[p["expr"]] = {"expr_tuple": p["expr_tuple"], "dir": 2,
                                   "ic20": p["ic20"], "icir20": p["icir20"],
                                   "turnover": p["turnover"]}
                gp_n += 1
    print(f"[collect] 方向2 遗传规划: 新增 {gp_n} 个 | {time.time()-t:.1f}s", flush=True)

    # 方向3 LLM+MCTS: 本地启发式搜索候选(正 IC)
    t = time.time()
    mcts_n = 0
    agent = MCTSAgent(data, fwd, max_depth=2, llm=None, seed=42)
    cands = agent.search(iterations=150, verbose=False)
    for c in cands:
        if c["icir"] > 0 and c["expr"] not in seen:
            seen[c["expr"]] = {"expr_tuple": c["expr_tuple"], "dir": 3,
                               "ic20": c["ic20"], "icir20": c["icir"],
                               "turnover": c["turnover"]}
            mcts_n += 1
    print(f"[collect] 方向3 MCTS: 新增 {mcts_n} 个 | {time.time()-t:.1f}s", flush=True)

    # 方向4 XGBoost 交互挖掘: 浅层树发现条件依赖 → 翻译公式因子
    if with_xgb:
        t = time.time()
        try:
            from factor_mining.xgb_interaction import mine_interactions
            xgb_res = mine_interactions(data, fwd, horizons=mine_horizons)
            xgb_n = 0
            for r in xgb_res:
                if r["expr"] not in seen:
                    seen[r["expr"]] = {"expr_tuple": r["expr_tuple"], "dir": 4,
                                       "ic20": r["ic20"], "icir20": r["icir20"],
                                       "turnover": r["turnover"]}
                    xgb_n += 1
            print(f"[collect] 方向4 XGB交互: 新增 {xgb_n} 个 | {time.time()-t:.1f}s", flush=True)
        except Exception as e:
            print(f"[collect] 方向4 失败(跳过): {e}", flush=True)

    return seen


# ---------------------------------------------------------------------------
# 构建特征长表(日期×股票, 每行一个观测)
# ---------------------------------------------------------------------------
def build_feature_table(codes, exprs, return_close=False):
    """exprs: list of (expr_str, expr_tuple). 返回 long DataFrame(含因子+标签)。

    return_close=True 时额外返回 close 面板(date×stock), 供防御门控构建市场等权指数。
    """
    t0 = time.time()
    data = derive_variables(load_base_data(codes))
    close = data["close"]
    dates = close.index
    n = len(codes)

    # 逐个因子求值(全股票面板, date×n), 存字典
    panels = {}
    for s, tup in exprs:
        try:
            panels[s] = evaluate_expr(tup, data).astype("float32")
        except Exception:
            panels[s] = None
    print(f"[build] 因子面板计算完成 {len(panels)} 个 | {time.time()-t0:.1f}s", flush=True)

    # ---- 筹码结构维度 (B1: 准确换手率驱动) ----
    # 接进 XGBoost 的「筹码结构」选股维度: 获利盘比例/平均成本偏离/成本集中度/离散度/偏度。
    # 特征名 chip_* 前缀; 对 qfq 调整乘子不变。turnover 湖缺失或计算失败则优雅跳过。
    chip_feat_cols: list[str] = []
    if turnover_available():
        try:
            from factor_mining.chip_structure import build_chip_panels, CHIP_FIELDS
            t1 = time.time()
            chip_panels = build_chip_panels(codes)
            for k in CHIP_FIELDS:
                if k in chip_panels:
                    panels[k] = chip_panels[k].astype("float32")
                    chip_feat_cols.append(k)
            print(f"[build] 筹码结构特征载入 {len(chip_feat_cols)} 个 | {time.time()-t1:.1f}s", flush=True)
        except Exception as e:
            print(f"[build] 筹码结构特征载入失败(跳过): {e}", flush=True)

    # 逐股拼长表(避免一次性持有全市场大矩阵); 索引=日期, 不另存 date 列
    recs = []
    for i, s in enumerate(codes):
        c = close[s]
        d = {"code": s}
        for hz in HORIZONS:
            d[f"fwd_ret_{hz}"] = (c.shift(-hz) / c.shift(-1) - 1.0).astype("float32").values
        # fwd_ret_1: T 日收盘买入、T+1 日收盘卖出收益(仅用于回测持仓, 防泄露用 shift(-1))
        d["fwd_ret_1"] = (c.shift(-1) / c - 1.0).astype("float32").values
        for name, panel in panels.items():
            if panel is None:
                d[name] = np.full(len(dates), np.nan, "float32")
            elif s in panel.columns:
                d[name] = panel[s].astype("float32").values
            else:
                d[name] = np.full(len(dates), np.nan, "float32")
        recs.append(pd.DataFrame(d, index=dates))
        if (i + 1) % 300 == 0:
            print(f"[build]  已处理 {i+1}/{n} 只", flush=True)
    long = pd.concat(recs)
    long = long.reset_index().rename(columns={"index": "date"})  # date 列为后续分折/WFA 用
    del recs, panels
    gc.collect()
    print(f"[build] 长表 {long.shape} | {time.time()-t0:.1f}s", flush=True)

    # 标签: 横截面前30% -> 1
    for hz in HORIZONS:
        long[f"cls_{hz}"] = long.groupby("date")[f"fwd_ret_{hz}"].transform(
            lambda x: (x.rank(pct=True) >= 0.7).astype("int8"))

    # 仅保留标签齐备的行(XGBoost 原生处理特征 NaN)
    label_cols = [f"fwd_ret_{hz}" for hz in HORIZONS]
    long = long.dropna(subset=label_cols).reset_index(drop=True)
    print(f"[build] 标签齐备后 {long.shape} | {time.time()-t0:.1f}s", flush=True)

    # 横截面 z-score 标准化(对齐 v4 的 cross_section_zscore, 防个股量纲)
    feat_cols = [name for name, _ in exprs] + chip_feat_cols
    g = long.groupby("date")[feat_cols]
    mu = g.transform("mean")
    sd = g.transform("std")
    long[feat_cols] = (long[feat_cols] - mu) / (sd + 1e-8)
    del mu, sd, g
    gc.collect()
    if return_close:
        return long, feat_cols, close
    return long, feat_cols


# ---------------------------------------------------------------------------
# WFA 训练 + 评估(对齐 v4 _train_fold)
# ---------------------------------------------------------------------------
def _hp_search(X, y, n_iter=8):
    clf = XGBClassifier(n_estimators=150, nthread=4, eval_metric="auc",
                        random_state=RANDOM_STATE, use_label_encoder=False)
    param_dist = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0, 1, 5],
        "gamma": [0, 1],
    }
    rs = RandomizedSearchCV(clf, param_dist, n_iter=n_iter, scoring="roc_auc",
                            cv=2, n_jobs=1, random_state=RANDOM_STATE)
    rs.fit(X, y)
    return rs.best_params_


def run_wfa(long, feat_cols, train_cap=500000):
    folds = wfa_folds(long["date"])
    print(f"[wfa] 折数 {len(folds)}: " + " | ".join(
        f"折{i+1} OOS {s.date()}~{e.date()}" for i, (_, _, s, e) in enumerate(folds)), flush=True)
    rows, imp_acc, oos_detail = [], {}, []
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        is_mask = (long["date"] >= is_s) & (long["date"] < is_e)
        oos_mask = (long["date"] >= oos_s) & (long["date"] < oos_e)
        isd = long[is_mask]
        osd = long[oos_mask]
        if len(isd) < 500 or len(osd) < 100:
            print(f"    折{i}: 样本不足跳过 (IS={len(isd)}, OOS={len(osd)})", flush=True)
            continue
        # IS 子抽样控内存/提速(对齐 v4 WFA_TRAIN_CAP)
        if len(isd) > train_cap:
            idx = isd.sample(train_cap, random_state=RANDOM_STATE).index
            isd = isd.loc[idx]
        Xis = isd[feat_cols].values.astype("float32")
        Xos = osd[feat_cols].values.astype("float32")
        best = _hp_search(Xis, isd["cls_5"].values)
        print(f"    折{i}: IS={len(isd)} OOS={len(osd)} 最佳参数 {best}", flush=True)
        models, probas = {}, {}
        for hz in HORIZONS:
            m = XGBClassifier(n_estimators=300, nthread=4, eval_metric="auc",
                              random_state=RANDOM_STATE, use_label_encoder=False, **best)
            m.fit(Xis, isd[f"cls_{hz}"].values)
            probas[hz] = m.predict_proba(Xos)[:, 1]
            if hz == 5:
                _raw = m.get_booster().get_score(importance_type="gain")
                imp_acc = {feat_cols[int(k[1:])]: v for k, v in _raw.items()
                           if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(feat_cols)}
        fused = np.mean([probas[hz] for hz in HORIZONS], axis=0)
        auc5 = roc_auc_score(osd["cls_5"].values, probas[5]) if len(set(osd["cls_5"].values)) > 1 else np.nan
        row = {"fold": i, "auc_single5": round(auc5, 4),
               "ic_single5": round(spearmanr(probas[5], osd["fwd_ret_5"].values).correlation, 4)}
        for hz in HORIZONS:
            ys, yt = osd[f"cls_{hz}"].values, osd[f"fwd_ret_{hz}"].values
            row[f"auc_fuse_{hz}"] = round(roc_auc_score(ys, fused), 4) if len(set(ys)) > 1 else np.nan
            row[f"ic_fuse_{hz}"] = round(spearmanr(fused, yt).correlation, 4)
        print(f"    折{i}: 单周期5日 AUC={row['auc_single5']:.3f}/IC={row['ic_single5']:+.4f} | "
              f"融合 AUC=[{row['auc_fuse_5']:.3f},{row['auc_fuse_20']:.3f},{row['auc_fuse_60']:.3f}] "
              f"IC=[{row['ic_fuse_5']:+.4f},{row['ic_fuse_20']:+.4f},{row['ic_fuse_60']:+.4f}]", flush=True)
        rows.append(row)
        # 收集 OOS 预测明细(date, code, 融合概率, 次日收益)供样本外回测
        seg = pd.DataFrame({
            "date": osd["date"].values,
            "code": osd["code"].values,
            "fused": fused,
            "fwd_ret_1": osd["fwd_ret_1"].values,
        })
        oos_detail.append(seg)
    oos_detail = pd.concat(oos_detail, ignore_index=True) if oos_detail else pd.DataFrame()
    return rows, imp_acc, folds, oos_detail


# ---------------------------------------------------------------------------
# 分块版(对齐 v4 内存策略): 时间分片 + 面板落盘缓存 + 断点续传 + TRAIN_CAP
# 设计目标: 内存安全的分块构建。实测本机物理内存 ≈ 32GB, 此处仍按保守峰值 ≈ 1.3GB
#   设计(不随股票数线性增长), 故全市场 5596 只 × 200 因子也远在预算内; 若内存更小
#   亦不会崩(单片 long + 单个因子面板常驻)。
#   Pass-1: 每个因子面板在全域算一次, 落盘 <tmp>/panel_<j>.parquet(断点续传),
#           同时在线累积横截面 z-score 统计量(和/平方和/计数)。
#   Pass-2: 按 CHUNK_MONTHS 时间切片; 每片只读覆盖股票的面板切片, 拼 long 子集,
#           用 Pass-1 统计量做横截面 z-score + 标签(前30%), 原子写 feat_<ci>.parquet。
#   run_wfa_chunked: 每折只读取覆盖该折窗的 feat 分片, IS 子采样到 TRAIN_CAP。
# ---------------------------------------------------------------------------
def _month_chunks(dates, chunk_months=12):
    """把日期索引切成不重叠的月窗; 返回 [(d0,d1), ...] (含端点, Timestamp)。"""
    d0 = pd.Timestamp(dates.min())
    d1 = pd.Timestamp(dates.max())
    chunks, start = [], d0
    while start <= d1:
        end = start + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1)
        if end > d1:
            end = d1
        chunks.append((start, end))
        start = end + pd.Timedelta(days=1)
    return chunks


def build_feature_table_chunked(codes, exprs, out_dir, chunk_months=12, lookback=252,
                                tmp_dir=None, resume=True, keep_tmp=False, start=None):
    """分块构建特征 store(对齐 v4)。返回 (feat_dir, feat_cols)。

    out_dir/feat_<ci>.parquet : 每片 long 子集(date,code,fwd_ret_*,cls_*,feat_cols), float32。
    内存峰值 ≈ 单片 long + 单个因子面板, 不随股票数线性增长, 故全市场可跑。
    lookback : 每片前视预热天数(覆盖 rolling 因子窗, 默认 252≈1y)。
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    tmp = tmp_dir or os.path.join(out_dir, "_panels_tmp")
    os.makedirs(tmp, exist_ok=True)
    dates_all = load_panel("close", [codes[0]], start=start).index
    feat_cols = [s for s, _ in exprs]
    # ---- 筹码结构(路径依赖, 全域算一次, 常驻内存, 体量小) ----
    chip_panels = {}
    chip_feat_cols = []
    if turnover_available():
        try:
            from factor_mining.chip_structure import build_chip_panels, CHIP_FIELDS
            t1 = time.time()
            chip_panels = build_chip_panels(codes) or {}
            chip_feat_cols = [k for k in CHIP_FIELDS if k in chip_panels]
            print(f"[chunk] 筹码结构特征 {len(chip_feat_cols)} 个 | {time.time()-t1:.1f}s", flush=True)
        except Exception as e:
            print(f"[chunk] 筹码结构特征失败(跳过): {e}", flush=True)
    all_feat = feat_cols + chip_feat_cols
    nfeat = len(all_feat)
    n = len(dates_all)

    # ---- Pass-1: 因子面板全域算一次 + 落盘 + 累积 z-score 统计量 ----
    S = np.zeros((n, nfeat), "float64")
    SQ = np.zeros((n, nfeat), "float64")
    C = np.zeros((n, nfeat), "float64")
    data = derive_variables(load_base_data(codes, start=start))
    panel_files = []
    for j, (s, tup) in enumerate(exprs):
        pf = os.path.join(tmp, f"panel_{j:04d}.parquet")
        panel_files.append(pf)
        if resume and os.path.exists(pf):
            p = pd.read_parquet(pf).reindex(index=dates_all, columns=codes)
        else:
            try:
                p = evaluate_expr(tup, data).astype("float32")
            except Exception:
                p = pd.DataFrame(index=dates_all, columns=codes, dtype="float32")
            p = p.reindex(index=dates_all, columns=codes)
            p.to_parquet(pf)
        v = p.values.astype("float64")          # (n_dates, n_stocks)
        nv = np.isnan(v)
        S[:, j] += np.where(nv, 0.0, v).sum(axis=1)
        SQ[:, j] += np.where(nv, 0.0, v * v).sum(axis=1)
        C[:, j] += (~nv).sum(axis=1)
        if (j + 1) % 50 == 0:
            print(f"[chunk] Pass-1 面板 {j+1}/{len(exprs)} | {time.time()-t0:.1f}s", flush=True)
    del data
    gc.collect()
    # chip 统计量
    for k, kname in enumerate(chip_feat_cols):
        p = chip_panels[kname].reindex(index=dates_all, columns=codes)
        v = p.values.astype("float64")
        nv = np.isnan(v)
        S[:, len(feat_cols) + k] += np.where(nv, 0.0, v).sum(axis=1)
        SQ[:, len(feat_cols) + k] += np.where(nv, 0.0, v * v).sum(axis=1)
        C[:, len(feat_cols) + k] += (~nv).sum(axis=1)
    mean = S / np.maximum(C, 1)
    var = SQ / np.maximum(C, 1) - mean * mean
    sd = np.sqrt(np.maximum(var, 0))
    print(f"[chunk] Pass-1 完成 {nfeat} 特征 | {time.time()-t0:.1f}s", flush=True)

    # ---- Pass-2: 时间分片落盘 long 子集 ----
    chunks = _month_chunks(dates_all, chunk_months)
    close_cols = ["date", "code"] + [f"fwd_ret_{hz}" for hz in HORIZONS] + ["fwd_ret_1"]
    for ci, (d0, d1) in enumerate(chunks):
        fout = os.path.join(out_dir, f"feat_{ci:03d}.parquet")
        if resume and os.path.exists(fout):
            print(f"[chunk] {ci}/{len(chunks)-1} 已存在, 跳过", flush=True)
            continue
        w0 = d0 - pd.DateOffset(days=lookback)
        w1 = d1 + pd.DateOffset(days=max(HORIZONS))
        c_mask = (dates_all >= d0) & (dates_all <= d1)
        w_mask = (dates_all >= w0) & (dates_all <= w1)
        cidx = dates_all[c_mask]                      # 本片日期(DatetimeIndex)
        widx = dates_all[w_mask]                      # 预热+前向窗口
        c_pos = np.searchsorted(dates_all.values, cidx.values)   # 在 dates_all 中的位置
        c_within_w = np.searchsorted(widx.values, cidx.values)   # 在 widx 中的位置
        # 读因子面板切片(窗口) -> (n_chunk, n_stocks, n_feat)
        feat_mat = np.full((len(cidx), len(codes), nfeat), np.nan, "float32")
        for j in range(len(exprs)):
            p = pd.read_parquet(panel_files[j]).reindex(index=widx, columns=codes)
            feat_mat[:, :, j] = p.values[c_within_w]
        for k, kname in enumerate(chip_feat_cols):
            p = chip_panels[kname].reindex(index=widx, columns=codes)
            feat_mat[:, :, len(feat_cols) + k] = p.values[c_within_w]
        # 横截面 z-score(用全域统计量; mean/sd 为 numpy, 按 c_pos 按位置切片)
        mu = mean[c_pos]                              # (n_chunk, nfeat)
        sg = sd[c_pos]
        feat_z = (feat_mat - mu[:, None, :]) / (sg[:, None, :] + 1e-8)
        # 前向收益(窗口 close)
        close_w = load_panel("close", codes, start=str(w0.date()), end=str(w1.date()))
        fwd = {}
        for hz in HORIZONS:
            fwd[hz] = (close_w.shift(-hz) / close_w.shift(-1) - 1.0).reindex(index=cidx)
        fwd1 = (close_w.shift(-1) / close_w - 1.0).reindex(index=cidx)
        # 拼 long
        recs = []
        for si, code in enumerate(codes):
            d = {"code": code}
            for jj in range(nfeat):
                d[all_feat[jj]] = feat_z[:, si, jj]
            for hz in HORIZONS:
                d[f"fwd_ret_{hz}"] = fwd[hz][code].values if code in fwd[hz] else np.full(len(cidx), np.nan)
            d["fwd_ret_1"] = fwd1[code].values if code in fwd1 else np.full(len(cidx), np.nan)
            recs.append(pd.DataFrame(d, index=cidx))
        long_c = pd.concat(recs)
        del recs, feat_mat, feat_z, close_w, fwd, fwd1
        gc.collect()
        long_c = long_c.reset_index().rename(columns={"index": "date"})
        # 标签: 横截面前30% -> 1
        for hz in HORIZONS:
            long_c[f"cls_{hz}"] = long_c.groupby("date")[f"fwd_ret_{hz}"].transform(
                lambda x: (x.rank(pct=True) >= 0.7).astype("int8"))
        label_cols = [f"fwd_ret_{hz}" for hz in HORIZONS]
        long_c = long_c.dropna(subset=label_cols).reset_index(drop=True)
        # 类别列降精度, 其余 float32
        long_c = long_c.astype({c: "float32" for c in all_feat + label_cols + ["fwd_ret_1"]}, errors="ignore")
        # 原子写
        _tmp = fout + ".tmp"
        long_c.to_parquet(_tmp, index=False)
        os.replace(_tmp, fout)
        print(f"[chunk] {ci}/{len(chunks)-1} 落盘 {long_c.shape} "
              f"[{d0.date()}~{d1.date()}] | {time.time()-t0:.1f}s", flush=True)
        del long_c
        gc.collect()
    if not keep_tmp:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[chunk] 全部分片完成 out_dir={out_dir} | {time.time()-t0:.1f}s", flush=True)
    return out_dir, all_feat


def run_wfa_chunked(feat_dir, feat_cols, train_cap=2000000):
    """从分片 store 跑 WFA(对齐 v4 Phase C): 每折只读取覆盖该折窗的 feat 分片, IS 子采样 TRAIN_CAP。

    本机 ≈ 32GB, 默认 TRAIN_CAP=2_000_000(对齐 v4) 远在预算内; 样本外窗全量(不抽样)以保证评估无偏。
    返回 (rows, imp_acc, folds, oos_detail) 与 run_wfa 完全兼容。
    """
    files = sorted(glob.glob(os.path.join(feat_dir, "feat_*.parquet")))
    if not files:
        raise RuntimeError("没有 feat 分片, 先运行 build_feature_table_chunked")
    # 每个分片的日期范围(只读 date 列, 省 IO)
    file_ranges = []
    for f in files:
        d = pd.read_parquet(f, columns=["date"])["date"]
        if len(d):
            file_ranges.append((f, pd.Timestamp(d.min()), pd.Timestamp(d.max())))
    lo = min(fr[1] for fr in file_ranges)
    hi = max(fr[2] for fr in file_ranges)
    folds = wfa_folds(pd.date_range(lo, hi, freq="D"))
    print(f"[wfa-chunked] 折数 {len(folds)}: " + " | ".join(
        f"折{i+1} OOS {s.date()}~{e.date()}" for i, (_, _, s, e) in enumerate(folds)), flush=True)
    rows, imp_acc, oos_detail = [], {}, []
    read_cols = ["date", "code", "fwd_ret_1"] + [f"cls_{hz}" for hz in HORIZONS] \
        + [f"fwd_ret_{hz}" for hz in HORIZONS] + feat_cols
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        is_parts, oos_parts = [], []
        for f, flo, fhi in file_ranges:
            if fhi >= is_s and flo < is_e:
                df = pd.read_parquet(f, columns=read_cols)
                m = (df["date"] >= is_s) & (df["date"] < is_e)
                is_parts.append(df[m])
            if fhi >= oos_s and flo < oos_e:
                df = pd.read_parquet(f, columns=read_cols)
                m = (df["date"] >= oos_s) & (df["date"] < oos_e)
                oos_parts.append(df[m])
        if not is_parts or not oos_parts:
            print(f"    折{i}: 无覆盖分片, 跳过", flush=True)
            continue
        isd = pd.concat(is_parts, ignore_index=True)
        osd = pd.concat(oos_parts, ignore_index=True)
        if len(isd) < 500 or len(osd) < 100:
            print(f"    折{i}: 样本不足跳过 (IS={len(isd)}, OOS={len(osd)})", flush=True)
            continue
        if len(isd) > train_cap:
            idx = isd.sample(train_cap, random_state=RANDOM_STATE).index
            isd = isd.loc[idx]
        Xis = isd[feat_cols].values.astype("float32")
        Xos = osd[feat_cols].values.astype("float32")
        best = _hp_search(Xis, isd["cls_5"].values)
        print(f"    折{i}: IS={len(isd)} OOS={len(osd)} 最佳参数 {best}", flush=True)
        probas = {}
        for hz in HORIZONS:
            m = XGBClassifier(n_estimators=300, nthread=4, eval_metric="auc",
                              random_state=RANDOM_STATE, use_label_encoder=False, **best)
            m.fit(Xis, isd[f"cls_{hz}"].values)
            probas[hz] = m.predict_proba(Xos)[:, 1]
            if hz == 5:
                _raw = m.get_booster().get_score(importance_type="gain")
                imp_acc = {feat_cols[int(k[1:])]: v for k, v in _raw.items()
                           if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(feat_cols)}
        fused = np.mean([probas[hz] for hz in HORIZONS], axis=0)
        auc5 = roc_auc_score(osd["cls_5"].values, probas[5]) if len(set(osd["cls_5"].values)) > 1 else np.nan
        row = {"fold": i, "auc_single5": round(auc5, 4),
               "ic_single5": round(spearmanr(probas[5], osd["fwd_ret_5"].values).correlation, 4)}
        for hz in HORIZONS:
            ys, yt = osd[f"cls_{hz}"].values, osd[f"fwd_ret_{hz}"].values
            row[f"auc_fuse_{hz}"] = round(roc_auc_score(ys, fused), 4) if len(set(ys)) > 1 else np.nan
            row[f"ic_fuse_{hz}"] = round(spearmanr(fused, yt).correlation, 4)
        print(f"    折{i}: 单周期5日 AUC={row['auc_single5']:.3f}/IC={row['ic_single5']:+.4f} | "
              f"融合 AUC=[{row['auc_fuse_5']:.3f},{row['auc_fuse_20']:.3f},{row['auc_fuse_60']:.3f}] "
              f"IC=[{row['ic_fuse_5']:+.4f},{row['ic_fuse_20']:+.4f},{row['ic_fuse_60']:+.4f}]", flush=True)
        rows.append(row)
        seg = pd.DataFrame({
            "date": osd["date"].values, "code": osd["code"].values,
            "fused": fused, "fwd_ret_1": osd["fwd_ret_1"].values,
        })
        oos_detail.append(seg)
    oos_detail = pd.concat(oos_detail, ignore_index=True) if oos_detail else pd.DataFrame()
    return rows, imp_acc, folds, oos_detail


# ---------------------------------------------------------------------------
# 样本外滚动回测: 用各折 OOS 融合概率做 top-30% 每日再平衡组合
# ---------------------------------------------------------------------------
def backtest(oos_detail, top_frac=0.3, gate=False, crisis=None, stress=None,
             crisis_pos=0.60, def_ann=0.04, max_pos_reduce=0.20, cost_bps=0.0,
             rebalance_freq=1):
    """样本外滚动回测: 用融合概率(或冻结因子信号)做 top-30% 组合。

    rebalance_freq=1(默认): 逐日再平衡, 每日按信号重选 top-30% 持有一日(原行为, 不变)。
    rebalance_freq>1(如 5≈周频): 每 `rebalance_freq` 个交易日调仓一次, 持仓持有到下次
        调仓; 交易成本仅在调仓日按换手率计(日频路径的 1/freq), 大幅削减换手损耗。

    gate=True 时叠加**防御门控**(对齐用户 defensive_gating.py 双层门):
      · 右侧确认 crisis(市场等权指数跌破250日线-10% 或 波动z>2)→ 仓位降到 crisis_pos(0.60, 不归零)。
      · 左侧缓冲 stress∈[0,1](指数跌破250日线的深度, 无宏观时作左翼代理)→ 仓位最多降 max_pos_reduce(20%)。
      空仓部分吃 def_ann 年化(4%)防御资产收益; 基准(全市场等权)不受影响, 用于公平对照。
    """
    if oos_detail is None or len(oos_detail) == 0:
        return None
    df = oos_detail.dropna(subset=["fused", "fwd_ret_1"]).copy()
    if len(df) == 0:
        return None
    freq = max(1, int(rebalance_freq))

    if freq == 1:
        # —— 逐日再平衡(原逻辑, 行为完全不变) ——
        df["rk"] = df.groupby("date")["fused"].rank(pct=True, ascending=False)  # 1=预测最高
        top = df[df["rk"] <= top_frac]
        daily = top.groupby("date")["fwd_ret_1"].mean().sort_index()   # 组合日收益(等权)
        base = df.groupby("date")["fwd_ret_1"].mean().reindex(daily.index).fillna(0.0)
        cost_daily = 0.0
        if cost_bps and cost_bps > 0:
            held_by_date = top.groupby("date")["code"].apply(set)
            turnovers, prev_held = [], set()
            for d in daily.index:
                curr = held_by_date.get(d, set())
                tov = (len(curr.symmetric_difference(prev_held)) / max(len(curr), 1)) if prev_held else 1.0
                turnovers.append(tov)
                prev_held = curr
            cost_series = pd.Series(np.array(turnovers) * 2 * cost_bps / 10000.0, index=daily.index)
            daily = daily - cost_series
            cost_daily = float(cost_series.mean())
            base = base - cost_series.mean()
    else:
        # —— 周频(低频)再平衡: 每 freq 交易日调仓, 持仓持有到下次调仓 ——
        df = df.sort_values("date").reset_index(drop=True)
        dates = list(dict.fromkeys(df["date"].tolist()))
        rebal_dates = set(dates[::freq])
        groups = {d: g for d, g in df.groupby("date")}      # 日期→当日截面(避免重复过滤)
        # 每个调仓日的持仓集合(top-frac by fused) + 每只股票归属的"生效调仓日"
        active, cur, rebal_sel = {}, None, {}
        for d in dates:
            if d in rebal_dates:
                cur = d
            active[d] = cur
        for d in rebal_dates:
            g = groups[d]
            k = int(np.ceil(top_frac * len(g)))
            rebal_sel[d] = set(g.nlargest(k, "fused")["code"]) if k > 0 else set()
        daily_ret = {}
        for d in dates:
            h = rebal_sel.get(active[d], set())
            if h:
                daily_ret[d] = float(groups[d].loc[groups[d]["code"].isin(h), "fwd_ret_1"].mean())
        daily = pd.Series(daily_ret).sort_index()
        base = df.groupby("date")["fwd_ret_1"].mean().reindex(daily.index).fillna(0.0)
        # 成本仅在调仓日按换手率计
        cost_daily = 0.0
        if cost_bps and cost_bps > 0:
            tovs, prev_held = [], set()
            for d in daily.index:
                if d in rebal_dates and prev_held:
                    h = rebal_sel.get(active[d], set())
                    tov = len(h.symmetric_difference(prev_held)) / max(len(h), 1)
                else:
                    tov = 0.0
                tovs.append(tov)
                if d in rebal_dates:
                    prev_held = rebal_sel.get(active[d], set())
            cost_series = pd.Series(np.array(tovs) * 2 * cost_bps / 10000.0, index=daily.index)
            daily = daily - cost_series
            cost_daily = float(cost_series.mean())
            base = base - cost_series.mean()

    n_crisis_days = 0
    if gate and crisis is not None:
        cr = crisis.reindex(daily.index).fillna(False).astype(float).values
        n_crisis_days = int((cr > 0.5).sum())
        st = stress.reindex(daily.index).fillna(0.0).values if stress is not None else 0.0
        # 急性危机→crisis_pos; 否则按左翼压力缓冲降仓(最多 max_pos_reduce)
        pos = np.where(cr > 0.5, crisis_pos, 1.0 - st * max_pos_reduce)
        r_def = def_ann / 252.0
        gated = pos * daily.values + (1.0 - pos) * r_def     # 空仓吃防御资产收益
        daily = pd.Series(gated, index=daily.index)
    nav = (1.0 + daily).cumprod()
    bnav = (1.0 + base).cumprod()
    n = len(daily)
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / n) - 1.0 if n > 1 else np.nan
    ann_b = (bnav.iloc[-1] / bnav.iloc[0]) ** (252.0 / n) - 1.0 if n > 1 else np.nan
    vol = daily.std() * np.sqrt(252)
    sharpe = (daily.mean() * 252) / vol if vol and vol > 0 else np.nan
    # 最大回撤: 净值从峰值到谷值的最大跌幅
    max_dd = float((nav / nav.cummax() - 1.0).min())
    max_dd_b = float((bnav / bnav.cummax() - 1.0).min())
    calmar = (ann / abs(max_dd)) if max_dd < 0 else np.nan
    return {
        "n_days": n, "n_crisis_days": n_crisis_days,
        "years": round(n / 252.0, 2),
        "start": str(daily.index.min().date()), "end": str(daily.index.max().date()),
        "ann_ret": round(float(ann), 4), "ann_base": round(float(ann_b), 4),
        "tot_ret": round(float(nav.iloc[-1] / nav.iloc[0] - 1), 4),
        "tot_base": round(float(bnav.iloc[-1] / bnav.iloc[0] - 1), 4),
        "ann_vol": round(float(vol), 4), "sharpe": round(float(sharpe), 3),
        "max_dd": round(max_dd, 4), "max_dd_base": round(max_dd_b, 4),
        "calmar": round(float(calmar), 3),
        "cost_bps": cost_bps, "avg_daily_cost": round(cost_daily, 6),
        "rebalance_freq": freq,
    }


# ---------------------------------------------------------------------------
# 结果落盘
# ---------------------------------------------------------------------------
def write_results(out_path, meta, rows, imp_acc, folds, n_used, bt=None):
    rec = pd.DataFrame(rows)
    mean = rec.mean(numeric_only=True).round(4)
    lines = []
    lines.append("# 因子挖掘 → XGBoost WFA 验证结果\n")
    lines.append(f"- 挖掘样本: {meta['mine']} 只 | WFA 样本: {meta['stocks']} 只 | "
                 f"发现因子(正IC去重): **{meta['n_total']}** | 用于验证: **{n_used}**")
    lines.append(f"- 标签: 未来收益排前 30% × 3 周期(5/20/60日) | WFA 共 {len(folds)} 折(训练3y/测试1y/步长1y)\n")
    lines.append("## 各折 OOS 表现\n")
    cols = ["fold", "auc_single5", "ic_single5", "auc_fuse_5", "auc_fuse_20", "auc_fuse_60",
            "ic_fuse_5", "ic_fuse_20", "ic_fuse_60"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in rec.iterrows():
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("| **均值** | " + " | ".join(
        f"**{mean.get(c, '')}**" for c in cols if c != "fold") + " |")
    if imp_acc:
        lines.append("\n## 因子重要性 Top10(Gain)\n")
        top = sorted(imp_acc.items(), key=lambda kv: kv[1], reverse=True)[:10]
        lines.append("| 因子 | Gain |")
        lines.append("|---|---|")
        for k, v in top:
            lines.append(f"| `{k}` | {v:.1f} |")
    if bt:
        lines.append("\n## 样本外策略回测(top-30% 每日再平衡, 毛收益未扣费)\n")
        lines.append(f"- 区间: **{bt['start']} ~ {bt['end']}** ({bt['years']} 年, {bt['n_days']} 交易日, "
                     f"覆盖 WFA 全部 OOS 窗, 无前视泄露)")
        lines.append(f"- 策略: 每日按模型融合概率排序, 买入前 30% 等权, 持有一日(T+1 收益), 次日再平衡\n")
        lines.append("| 指标 | 因子组合 | 全市场等权基准 |")
        lines.append("|---|---|---|")
        lines.append(f"| 总收益 | {bt['tot_ret']:+.1%} | {bt['tot_base']:+.1%} |")
        lines.append(f"| **年化收益率** | **{bt['ann_ret']:+.1%}** | **{bt['ann_base']:+.1%}** |")
        lines.append(f"| 年化波动率 | {bt['ann_vol']:.1%} | — |")
        lines.append(f"| 年化夏普(无风险=0) | {bt['sharpe']:.2f} | — |")
        lines.append(f"| **最大回撤** | **{bt['max_dd']:+.1%}** | **{bt['max_dd_base']:+.1%}** |")
        lines.append(f"| Calmar(年化/|回撤|) | {bt['calmar']:.2f} | — |")
    lines.append(f"\n---\n*由 `factor_mining/factor_wfa.py` 生成*")
    full = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full)
        print(f"\n结果已写入: {out_path}")
    return full, rec, mean


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "wfa", "all"])
    ap.add_argument("--mine", type=int, default=300)
    ap.add_argument("--stocks", type=int, default=1500)
    ap.add_argument("--out", type=str, default="FACTOR_WFA_RESULTS.md")
    ap.add_argument("--factors_json", type=str, default=None,
                    help="collect 产物路径(复用, 避免重复挖掘)")
    ap.add_argument("--chunked", action="store_true",
                    help="分块模式(对齐 v4 内存策略, 全市场 8GB 可跑): 特征落盘 feat_<ci>.parquet")
    ap.add_argument("--out_dir", type=str, default="feat_store",
                    help="分块模式特征 store 目录(默认 ./feat_store)")
    ap.add_argument("--tmp_dir", type=str, default=None,
                    help="分块 Pass-1 面板缓存目录(默认 <out_dir>/_panels_tmp)")
    ap.add_argument("--keep_tmp", action="store_true", help="分块完成后保留面板缓存")
    ap.add_argument("--lookback", type=int, default=252, help="分块每片预热天数(默认 252)")
    ap.add_argument("--chunk_months", type=int, default=12, help="分块月窗(默认 12)")
    args = ap.parse_args()

    factors = None
    if args.mode in ("collect", "all"):
        seen = collect_factors(args.mine)
        # 按方向统计
        from collections import Counter
        dc = Counter(v["dir"] for v in seen.values())
        print(f"\n=== 因子总数(正IC去重): {len(seen)} ===")
        print(f"  方向1 网格: {dc.get(1,0)} | 方向2 遗传规划: {dc.get(2,0)} | "
              f"方向3 MCTS: {dc.get(3,0)}")
        # 存 JSON(元组递归转 list, 无损可还原)
        dump = {k: {"expr_tuple_list": _t2l(v["expr_tuple"]),
                    "dir": v["dir"], "ic20": v["ic20"],
                    "icir20": v["icir20"], "turnover": v["turnover"]} for k, v in seen.items()}
        fj = args.factors_json or "factors_discovered.json"
        with open(fj, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False)
        print(f"已保存因子清单: {fj}")
        if args.mode == "collect":
            return
        factors = seen

    if args.mode in ("wfa", "all"):
        if factors is None:
            fj = args.factors_json or "factors_discovered.json"
            if not os.path.exists(fj):
                print(f"未找到 {fj}, 先执行 collect")
                return
            with open(fj, encoding="utf-8") as f:
                dump = json.load(f)
            # 重建 expr_tuple(从 list 还原为嵌套元组)
            factors = {}
            for k, v in dump.items():
                tup = _l2t(v["expr_tuple_list"])
                factors[k] = {"expr_tuple": tup, "dir": v["dir"],
                              "ic20": v["ic20"], "icir20": v["icir20"],
                              "turnover": v["turnover"]}
        # 截取: 过多时按 ICIR 取 Top MAX_FACTORS
        items = sorted(factors.items(), key=lambda kv: kv[1]["icir20"], reverse=True)
        if len(items) > MAX_FACTORS:
            items = items[:MAX_FACTORS]
        exprs = [(k, v["expr_tuple"]) for k, v in items]
        print(f"[wfa] 用于验证的因子: {len(exprs)} (发现 {len(factors)})", flush=True)

        codes = list_stocks(args.stocks)
        if args.chunked:
            feat_dir, feat_cols = build_feature_table_chunked(
                codes, exprs, args.out_dir, chunk_months=args.chunk_months,
                lookback=args.lookback, tmp_dir=args.tmp_dir, resume=True,
                keep_tmp=args.keep_tmp)
            rows, imp_acc, folds, oos_detail = run_wfa_chunked(feat_dir, feat_cols)
        else:
            long, feat_cols = build_feature_table(codes, exprs)
            rows, imp_acc, folds, oos_detail = run_wfa(long, feat_cols)
        bt = backtest(oos_detail)
        if bt:
            print(f"\n[回测] {bt['start']}~{bt['end']} 年化 {bt['ann_ret']:+.1%} | "
                  f"总收益 {bt['tot_ret']:+.1%} | 夏普 {bt['sharpe']:.2f} | "
                  f"基准年化 {bt['ann_base']:+.1%}")
        full, rec, mean = write_results(
            args.out, {"mine": args.mine, "stocks": args.stocks,
                       "n_total": len(factors)}, rows, imp_acc, folds, len(exprs), bt=bt)
        print("\n均值:", mean.to_string())


if __name__ == "__main__":
    main()
