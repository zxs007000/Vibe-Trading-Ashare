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
    list_stocks, load_base_data, derive_variables, forward_returns,
    evaluate_expr, expr_to_str, grid_search, evolve, MCTSAgent,
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
def collect_factors(n_mine: int = 300, seeds=(42, 7, 123)):
    t0 = time.time()
    codes = list_stocks(n_mine)
    data = derive_variables(load_base_data(codes))
    fwd = forward_returns(data["close"])
    print(f"[collect] 挖掘样本 {len(codes)} 只, 变量池 {len(data)} | 耗时 {time.time()-t0:.1f}s", flush=True)

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

    return seen


# ---------------------------------------------------------------------------
# 构建特征长表(日期×股票, 每行一个观测)
# ---------------------------------------------------------------------------
def build_feature_table(codes, exprs):
    """exprs: list of (expr_str, expr_tuple). 返回 long DataFrame(含因子+标签)。"""
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
            d[name] = panel[s].astype("float32").values if panel is not None else np.full(n, np.nan, "float32")
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
    feat_cols = [name for name, _ in exprs]
    g = long.groupby("date")[feat_cols]
    mu = g.transform("mean")
    sd = g.transform("std")
    long[feat_cols] = (long[feat_cols] - mu) / (sd + 1e-8)
    del mu, sd, g
    gc.collect()
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
# 样本外滚动回测: 用各折 OOS 融合概率做 top-30% 每日再平衡组合
# ---------------------------------------------------------------------------
def backtest(oos_detail, top_frac=0.3):
    if oos_detail is None or len(oos_detail) == 0:
        return None
    df = oos_detail.dropna(subset=["fused", "fwd_ret_1"]).copy()
    if len(df) == 0:
        return None
    df["rk"] = df.groupby("date")["fused"].rank(pct=True, ascending=False)  # 1=预测最高
    top = df[df["rk"] <= top_frac]
    daily = top.groupby("date")["fwd_ret_1"].mean()          # 组合日收益(等权)
    base = df.groupby("date")["fwd_ret_1"].mean()            # 全市场等权基准
    daily = daily.sort_index()
    base = base.reindex(daily.index).fillna(0.0)
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
        "n_days": n, "years": round(n / 252.0, 2),
        "start": str(daily.index.min().date()), "end": str(daily.index.max().date()),
        "ann_ret": round(float(ann), 4), "ann_base": round(float(ann_b), 4),
        "tot_ret": round(float(nav.iloc[-1] / nav.iloc[0] - 1), 4),
        "tot_base": round(float(bnav.iloc[-1] / bnav.iloc[0] - 1), 4),
        "ann_vol": round(float(vol), 4), "sharpe": round(float(sharpe), 3),
        "max_dd": round(max_dd, 4), "max_dd_base": round(max_dd_b, 4),
        "calmar": round(float(calmar), 3),
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
