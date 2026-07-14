# -*- coding: utf-8 -*-
"""全数据湖 · anti 闸门 WFA (含退市股 / 仅存活股 双管线对照)

用户决策 2026-07-14:
  - 定稿 anti: 熊市底仓 BEAR_FLOOR=0.50, 上限 STOCK_CAP=0.75, 上证综指(sh000001)判牛熊
  - "全部数据湖测试, 包括退市股" → 显式纳入退市股, 并量化幸存者偏差

实现:
  - 数据湖已含 361 只退市股(build_delist_into_lake.py). 强制清缓存重建面板, 保证用的是
    最新含退市股湖. 退市股识别 = 面板列 − 活跃股清单(metadata/stock_list.csv).
  - 因子全面板只算一次; 另用"存活股子集面板"重建因子, 让中性化也在存活股上做
    (真正还原"仅看存活股"的历史回测场景).
  - 每个折叠 k: 训练窗 IS_IC>0 且 IS_ICIR>0 冻结 → 测试窗真实 OOS(无前视).
  - 双管线各跑 anti(择时) + 无择时100%(隔离闸门贡献).
  - 输出: 全量(含退市) vs 仅存活 的 anti/静态 四组结果 + 幸存者偏差(delta).

输出: wfa_fullmarket_alldelisted.json
用法:
    python wfa_fullmarket_alldelisted.py --start 2010-01-01 --end 2026-07-14 --train-years 3
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FRAMEWORK = Path(__file__).parent
sys.path.insert(0, str(FRAMEWORK))

from panel_builder import build_survivor_free_panel, load_industry_map
from factor_zoo_daily import build_factors_neutralized
from factor_zoo_ortho import build_ortho_factors
from factor_zoo_quality import build_quality_factors
from factor_zoo_info import build_info_factors
from factor_zoo_fundamental import build_fundamental_factors
from factor_zoo_fundamental_plus import build_fundamental_plus_factors
from factor_zoo_regulatory import build_regulatory_factors
from factor_zoo_delist import build_delist_risk_factors
from oos_engine import (
    compute_forward_returns, get_frozen_set, build_signal, run_backtest,
    rank_ic, load_market_index, TOP_K, HOLD, COST, FWD_HORIZON,
)


def build_all_factors(panel, industry, args):
    families = {}
    families["tech"] = build_factors_neutralized(panel, industry)
    families["ortho"] = build_ortho_factors(panel)
    if not args.no_quality:
        families["quality"] = build_quality_factors(panel, industry)
    if not args.no_info:
        families["info"] = build_info_factors(panel, industry, use_external=True)
    if not args.no_fundamental:
        fund = build_fundamental_factors(panel, industry)
        fund_plus = build_fundamental_plus_factors(panel, industry)
        if fund_plus:
            fund = {**(fund or {}), **fund_plus}
        if fund:
            families["fundamental"] = fund
    if not args.no_regulatory:
        reg = build_regulatory_factors(panel, industry)
        if reg:
            families["regulatory"] = reg
    all_factors = {}
    for fam in families.values():
        all_factors.update(fam)
    return all_factors, families


def eval_fold(ic_all, split_date, names, train_years=None):
    """复用预计算 rank_ic, 按训练窗 IS 切分冻结因子.

    train_years: 若给定, IC 历史截断到 [split_date - train_years, split_date] → 真滚动窗;
                 若为 None, 用 split_date 之前全部历史(扩张窗, 旧默认行为)。
    """
    cutoff = None
    if train_years:
        cutoff = (pd.Timestamp(split_date) - pd.DateOffset(years=train_years)).strftime("%Y-%m-%d")
    rows = []
    for name in names:
        ic = ic_all[name]
        is_ic = ic.loc[cutoff:split_date] if cutoff else ic.loc[:split_date]
        if is_ic.dropna().empty:
            continue
        is_mean = is_ic.mean(skipna=True)
        is_icir = is_mean / is_ic.std() if is_ic.std() > 0 else 0
        rows.append({
            "name": name,
            "IS_IC_mean": round(float(is_mean), 6),
            "IS_ICIR": round(float(is_icir), 4),
            "alive": bool(is_mean > 0 and is_icir > 0),
        })
    if not rows:
        return pd.DataFrame(columns=["IS_IC_mean", "IS_ICIR", "alive"]).set_index("name")
    df = pd.DataFrame(rows).set_index("name").sort_values("IS_ICIR", ascending=False)
    return df


def summarize(port_ret, bench_ret):
    port = port_ret.dropna()
    bench = bench_ret.reindex(port.index).dropna()
    common = port.index.intersection(bench.index)
    port = port.loc[common]
    bench = bench.loc[common]
    if len(port) < 2:
        return {}
    nav_p = (1 + port.fillna(0)).cumprod()
    nav_b = (1 + bench.fillna(0)).cumprod()
    n_years = len(port) * HOLD / 252.0
    ann_p = nav_p.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_b = nav_b.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    sp = port.mean() / port.std() * np.sqrt(252 / HOLD) if port.std() > 0 else 0
    sb = bench.mean() / bench.std() * np.sqrt(252 / HOLD) if bench.std() > 0 else 0
    mdd = lambda nav: float((nav / nav.cummax() - 1).min())
    return {
        "n_periods": int(len(port)),
        "years": round(n_years, 2),
        "ann_return": round(float(ann_p), 4),
        "ann_benchmark": round(float(ann_b), 4),
        "cum_return": round(float(nav_p.iloc[-1] - 1), 4),
        "cum_benchmark": round(float(nav_b.iloc[-1] - 1), 4),
        "sharpe": round(float(sp), 3),
        "sharpe_benchmark": round(float(sb), 3),
        "max_drawdown": round(mdd(nav_p), 4),
        "max_drawdown_benchmark": round(mdd(nav_b), 4),
        "win_rate": round(float((port > 0).mean()), 3),
    }


def detect_delisted(panel_cols, stock_list_csv):
    """退市股 = 面板列 − 活跃股清单(当前上市)."""
    try:
        active = set(pd.read_csv(stock_list_csv, dtype=str).iloc[:, 0].tolist())
        active = {str(c).strip().zfill(6) for c in active}
    except Exception as e:
        print(f"  [warn] 读活跃股清单失败: {repr(e)[:80]}, 退市股数记为0")
        return set()
    cols = {str(c).zfill(6) for c in panel_cols}
    return cols - active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-07-14")
    ap.add_argument("--min-bars", type=int, default=500)
    ap.add_argument("--market-index", default="sh000001")
    ap.add_argument("--stock-cap", type=float, default=0.75)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--out", default="wfa_fullmarket_alldelisted.json")
    ap.add_argument("--no-delist-filter", action="store_true",
                    help="关闭退市风险过滤(隔离因子贡献用)")
    ap.add_argument("--delist-thr", type=float, default=0.50,
                    help="退市风险分阈值, >=此值剔除(默认0.50)")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--no-info", action="store_true")
    ap.add_argument("--no-fundamental", action="store_true")
    ap.add_argument("--no-regulatory", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 72)
    print(f"全数据湖 anti WFA  {args.start}~{args.end}  底仓=BEAR_FLOOR 上限={args.stock_cap} 大盘={args.market_index}")
    print(f"含退市股 vs 仅存活股 双管线对照")
    print("=" * 72)

    # ---- 1. 全量面板(含退市股, 强制清缓存由调用方 rm -rf .panel_cache 保证) ----
    tp = time.time()
    panel, industry = build_survivor_free_panel(
        start_date=args.start, end_date=args.end, min_bars=args.min_bars)
    close = panel["close"]
    n_full = close.shape[1]
    print(f"全量面板: {n_full} 只 × {len(close)} 天  ⏱ {time.time()-tp:.1f}s")

    # ---- 2. 退市股识别 ----
    sl_csv = Path(__file__).parent.parent / "stockworm" / "metadata" / "stock_list.csv"
    delisted = detect_delisted(close.columns, sl_csv)
    surv_cols = [c for c in close.columns if str(c).zfill(6) not in delisted]
    n_delist = len(delisted)
    n_surv = len(surv_cols)
    print(f"退市股识别: {n_delist} 只 (面板列 − 活跃股清单); 存活股: {n_surv} 只")

    # ---- 3. 存活股子集面板(列切片, 不重算 exdiv) ----
    panel_surv = {k: v[surv_cols] for k, v in panel.items() if k in ("open", "close", "high", "low", "volume", "amount")}
    industry_surv = {c: industry[c] for c in surv_cols if c in industry}
    close_surv = panel_surv["close"]

    # ---- 4. 因子(双管线各算一次) ----
    tf = time.time()
    all_factors, families = build_all_factors(panel, industry, args)
    all_factors_surv, _ = build_all_factors(panel_surv, industry_surv, args)
    print(f"因子: 全量 {len(all_factors)} / 存活 {len(all_factors_surv)} "
          f"({'+'.join(f'{k}={len(v)}' for k,v in families.items())})  ⏱ {time.time()-tf:.1f}s")

    # ---- 5. 前向收益 + rank_ic(双管线) ----
    fwd = compute_forward_returns(close, horizon=FWD_HORIZON)
    fwd_surv = compute_forward_returns(close_surv, horizon=FWD_HORIZON)
    tc = time.time()
    ic_all = {name: rank_ic(fv, fwd) for name, fv in all_factors.items()}
    ic_all_surv = {name: rank_ic(fv, fwd_surv) for name, fv in all_factors_surv.items()}
    print(f"rank_ic 预计算: 全量 {len(ic_all)} / 存活 {len(ic_all_surv)}  ⏱ {time.time()-tc:.1f}s")

    # ---- 4.5 退市风险因子(双管线) → 选股过滤矩阵 ----
    td = time.time()
    if not args.no_delist_filter:
        dr_full = build_delist_risk_factors(panel, industry).get("f_delist_risk")
        dr_surv = build_delist_risk_factors(panel_surv, industry_surv).get("f_delist_risk")
        print(f"退市风险因子: 全量 {dr_full.shape if dr_full is not None else None} / "
              f"存活 {dr_surv.shape if dr_surv is not None else None} "
              f"(阈值={args.delist_thr})  ⏱ {time.time()-td:.1f}s")
    else:
        dr_full = dr_surv = None
        print(f"退市风险过滤: 已关闭(--no-delist-filter)")

    # ---- 6. 折叠(滚动) ----
    start_yr = pd.Timestamp(args.start).year
    end_yr = pd.Timestamp(args.end).year
    test_years = list(range(start_yr + args.train_years, end_yr + 1))
    print(f"折叠数: {len(test_years)} (测试年 {test_years[0]}~{test_years[-1]}, 训练窗 {args.train_years}年滚动)")

    trend_w, dd_thr = 120, -0.10
    mk = args.market_index
    ports = {"anti_full": [], "static_full": [], "anti_surv": [], "static_surv": []}
    benches = {"anti_full": [], "static_full": [], "anti_surv": [], "static_surv": []}
    fold_rows = []

    for Y in test_years:
        test_start = pd.Timestamp(f"{Y}-01-01")
        test_end = pd.Timestamp(f"{Y}-12-31")
        split_date = (test_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # --- 全量管线 ---
        ev_f = eval_fold(ic_all, split_date, list(all_factors.keys()), train_years=args.train_years)
        frozen_f = get_frozen_set(ev_f)
        row = {"fold": Y, "n_frozen_full": len(frozen_f)}
        if frozen_f:
            sig_f = build_signal(all_factors, ev_f, frozen_f, weight_src="is")
            sig_f = sig_f.loc[str(test_start):str(test_end)]
            bt_a = run_backtest(sig_f, close, TOP_K, HOLD, COST, gate_mode="anti",
                                regime_trend_w=trend_w, regime_dd_thr=dd_thr,
                                stock_cap=args.stock_cap, market_index=mk,
                                delist_risk=dr_full, delist_thr=args.delist_thr)
            bt_s = run_backtest(sig_f, close, TOP_K, HOLD, COST, gate_mode="anti",
                                regime_trend_w=trend_w, regime_dd_thr=dd_thr,
                                stock_cap=args.stock_cap, market_index=mk, use_regime=False,
                                delist_risk=dr_full, delist_thr=args.delist_thr)
            ports["anti_full"].append(bt_a["portfolio"]); benches["anti_full"].append(bt_a["benchmark"])
            ports["static_full"].append(bt_s["portfolio"]); benches["static_full"].append(bt_s["benchmark"])
            m = summarize(bt_a["portfolio"], bt_a["benchmark"])
            row.update({"sharpe_full": m.get("sharpe"), "ann_full": m.get("ann_return"), "mdd_full": m.get("max_drawdown")})

        # --- 仅存活管线 ---
        ev_s = eval_fold(ic_all_surv, split_date, list(all_factors_surv.keys()), train_years=args.train_years)
        frozen_s = get_frozen_set(ev_s)
        row["n_frozen_surv"] = len(frozen_s)
        if frozen_s:
            sig_s = build_signal(all_factors_surv, ev_s, frozen_s, weight_src="is")
            sig_s = sig_s.loc[str(test_start):str(test_end)]
            bt_a = run_backtest(sig_s, close_surv, TOP_K, HOLD, COST, gate_mode="anti",
                                regime_trend_w=trend_w, regime_dd_thr=dd_thr,
                                stock_cap=args.stock_cap, market_index=mk,
                                delist_risk=dr_surv, delist_thr=args.delist_thr)
            bt_s = run_backtest(sig_s, close_surv, TOP_K, HOLD, COST, gate_mode="anti",
                                regime_trend_w=trend_w, regime_dd_thr=dd_thr,
                                stock_cap=args.stock_cap, market_index=mk, use_regime=False,
                                delist_risk=dr_surv, delist_thr=args.delist_thr)
            ports["anti_surv"].append(bt_a["portfolio"]); benches["anti_surv"].append(bt_a["benchmark"])
            ports["static_surv"].append(bt_s["portfolio"]); benches["static_surv"].append(bt_s["benchmark"])
            m = summarize(bt_a["portfolio"], bt_a["benchmark"])
            row.update({"sharpe_surv": m.get("sharpe"), "ann_surv": m.get("ann_return"), "mdd_surv": m.get("max_drawdown")})

        fold_rows.append(row)
        print(f"  折{Y}: 冻结 全量{row.get('n_frozen_full',0)}/存活{row.get('n_frozen_surv',0)}"
              f" | anti Sharpe 全量={row.get('sharpe_full')} 存活={row.get('sharpe_surv')}")

    # ---- 7. 汇总 ----
    def concat(dfs):
        if not dfs:
            return pd.Series(dtype=float)
        s = pd.concat(dfs)
        return s[~s.index.duplicated(keep="first")].sort_index()
    res = {}
    for key in ("anti_full", "static_full", "anti_surv", "static_surv"):
        res[key] = summarize(concat(ports[key]), concat(benches[key]))

    print(f"\n[A] anti(含退市股) OOS: {res['anti_full']}")
    print(f"[B] anti(仅存活)   OOS: {res['anti_surv']}")
    print(f"[C] 无择时100%(含退市股) OOS: {res['static_full']}")
    print(f"[D] 无择时100%(仅存活)   OOS: {res['static_surv']}")

    def delta(a, b):
        return {k: round(float(res[a].get(k, 0) - res[b].get(k, 0)), 4)
                for k in ("ann_return", "sharpe", "max_drawdown", "cum_return")}
    out = {
        "config": {
            "start": args.start, "end": args.end,
            "market_index": mk, "stock_cap": args.stock_cap,
            "bear_floor": 0.50,
            "train_years": args.train_years, "n_folds": len(test_years),
            "window_type": "rolling" if args.train_years else "expanding(旧默认)",
            "top_k": TOP_K, "hold": HOLD, "cost": COST,
            "delist_filter": (not args.no_delist_filter), "delist_thr": args.delist_thr,
            "n_stocks_full": n_full, "n_delisted": n_delist, "n_stocks_surv": n_surv,
            "n_factors_full": len(all_factors), "n_factors_surv": len(all_factors_surv),
            "method": "WFA: 因子全面板算一次, 每折 split_date 重算 IS_IC/冻结(训练窗=train_years 真实截断), "
                      "测试窗=真实OOS; 双管线(全量含退市 / 仅存活子集重建因子)对照幸存者偏差; "
                      "可选退市风险过滤(f_delist_risk>=阈值 剔除)",
            "delist_source": "数据湖 daily/ 已含退市股; 风险因子=面值(股价~1元)+困境深跌+流动性枯竭+监管重罚衰减",
        },
        "results": {
            "A_anti_full_incl_delist": res["anti_full"],
            "B_anti_surv_only": res["anti_surv"],
            "C_static_full_incl_delist": res["static_full"],
            "D_static_surv_only": res["static_surv"],
        },
        "survivorship_bias": {
            "anti_ann_delta(full_minus_surv)": delta("anti_full", "anti_surv"),
            "anti_sharpe_delta(full_minus_surv)": round(float(res["anti_full"].get("sharpe",0)-res["anti_surv"].get("sharpe",0)),3),
            "static_ann_delta(full_minus_surv)": delta("static_full", "static_surv"),
            "static_sharpe_delta(full_minus_surv)": round(float(res["static_full"].get("sharpe",0)-res["static_surv"].get("sharpe",0)),3),
            "interpretation": "含退市股 − 仅存活股 = 幸存者偏差对策略的净影响(正数=退市股拖累, "
                              "负数/近零=原回测已被幸存者偏差虚高, 含退市后更真实)",
        },
        "folds": fold_rows,
        "gate_value": {
            "anti_vs_static_full_incl_delist": {
                "ann_drag": round(float(res["static_full"].get("ann_return",0)-res["anti_full"].get("ann_return",0)),4),
                "sharpe_drag": round(float(res["static_full"].get("sharpe",0)-res["anti_full"].get("sharpe",0)),3),
            },
            "anti_vs_static_surv_only": {
                "ann_drag": round(float(res["static_surv"].get("ann_return",0)-res["anti_surv"].get("ann_return",0)),4),
                "sharpe_drag": round(float(res["static_surv"].get("sharpe",0)-res["anti_surv"].get("sharpe",0)),3),
            },
        },
    }
    with open(FRAMEWORK / args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*72}")
    print(f"完成! 总耗时 {(time.time()-t0)/60:.1f} 分钟 → {args.out}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
