# -*- coding: utf-8 -*-
"""全市场 · anti 闸门 WFA (2010-2023, 熊市上限75%, 上证综指判牛熊)

用户决策 2026-07-14:
  - anti 熊市上限 0.50 → 0.75 (信任因子 + 上证综指判熊)
  - anti 牛熊判定由等权全A净值 → 真实上证综指(sh000001, stock-worm 拉取)
  - 跑 WFA 去除常规回测的因子选择前视乐观

WFA 方法 (滚动窗口):
  - 因子计算: 全面板(2010-2023)只算一次
  - 每个折叠 k: 训练窗 = [START, Y_k-01-01) (默认滚动3年), 测试窗 = [Y_k 全年]
  - 冻结因子仅在训练窗 IS_IC>0 且 IS_ICIR>0 → 测试窗为真实样本外(OOS, 与训练不重叠, 无前视)
  - 同时跑 "无择时 100%" 基线(同信号同成本, use_regime=False) 隔离闸门贡献

输出: wfa_fullmarket_anti.json
用法:
    python wfa_fullmarket_anti.py --start 2010-01-01 --end 2023-12-31 --train-years 3
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
from oos_engine import (
    compute_forward_returns, get_frozen_set, build_signal, run_backtest,
    rank_ic, load_market_index, TOP_K, HOLD, COST,
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


def eval_fold(ic_all, split_date, names):
    """与 evaluate_factors 的 IS 逻辑一致, 但复用预计算的 rank_ic, 避免重复计算。"""
    rows = []
    for name in names:
        ic = ic_all[name]
        is_ic = ic.loc[:split_date]
        if is_ic.dropna().empty:
            continue
        is_mean = is_ic.mean(skipna=True)
        is_icir = is_mean / is_ic.std() if is_ic.std() > 0 else 0
        rows.append({
            "name": name,
            "IS_IC_mean": round(float(is_mean), 6),
            "IS_ICIR": round(float(is_icir), 4),
            "OOS_IC_mean": np.nan, "OOS_ICIR": np.nan,
            "alive": bool(is_mean > 0 and is_icir > 0),
        })
    if not rows:
        return pd.DataFrame(columns=["IS_IC_mean", "IS_ICIR", "alive"]).set_index("name")
    df = pd.DataFrame(rows).set_index("name").sort_values("IS_ICIR", ascending=False)
    return df


def summarize(port_ret: pd.Series, bench_ret: pd.Series) -> dict:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--min-bars", type=int, default=500)
    ap.add_argument("--market-index", default="sh000001")
    ap.add_argument("--stock-cap", type=float, default=0.75)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--no-info", action="store_true")
    ap.add_argument("--no-fundamental", action="store_true")
    ap.add_argument("--no-regulatory", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print(f"全市场 anti WFA  {args.start}~{args.end}  熊市上限={args.stock_cap}  大盘={args.market_index}")
    print("=" * 70)

    # 1. 面板(一次)
    tp = time.time()
    panel, industry = build_survivor_free_panel(
        start_date=args.start, end_date=args.end, min_bars=args.min_bars)
    close = panel["close"]
    print(f"面板: {close.shape[1]} 只 × {len(close)} 天  ⏱ {time.time()-tp:.1f}s")

    # 2. 因子(一次)
    tf = time.time()
    all_factors, families = build_all_factors(panel, industry, args)
    print(f"因子族: " + ", ".join(f"{k}={len(v)}" for k, v in families.items())
          + f"  合计 {len(all_factors)}  ⏱ {time.time()-tf:.1f}s")

    # 3. 前向收益 + 预计算 rank_ic(一次)
    from oos_engine import FWD_HORIZON
    fwd = compute_forward_returns(close, horizon=FWD_HORIZON)
    tc = time.time()
    ic_all = {name: rank_ic(fv, fwd) for name, fv in all_factors.items()}
    print(f"rank_ic 预计算: {len(ic_all)} 因子  ⏱ {time.time()-tc:.1f}s")

    # 4. 折叠定义(滚动 train-years / 1年测试)
    start_yr = pd.Timestamp(args.start).year
    end_yr = pd.Timestamp(args.end).year
    test_years = list(range(start_yr + args.train_years, end_yr + 1))
    print(f"折叠数: {len(test_years)} (测试年 {test_years[0]}~{test_years[-1]}, 训练窗 {args.train_years}年滚动)")

    anti_port_dfs, anti_bench_dfs = [], []
    static_port_dfs, static_bench_dfs = [], []
    fold_rows = []

    for Y in test_years:
        test_start = pd.Timestamp(f"{Y}-01-01")
        test_end = pd.Timestamp(f"{Y}-12-31")
        train_start = pd.Timestamp(f"{Y - args.train_years}-01-01")
        train_start = max(train_start, pd.Timestamp(args.start))
        split_date = (test_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # 该折叠冻结因子(仅训练窗 IS)
        ev = eval_fold(ic_all, split_date, list(all_factors.keys()))
        frozen = get_frozen_set(ev)
        if not frozen:
            print(f"  折{Y}: 冻结0因子, 跳过")
            fold_rows.append({"fold": Y, "n_frozen": 0, "ann": None, "sharpe": None, "mdd": None})
            continue
        signal = build_signal(all_factors, ev, frozen, weight_src="is")
        # 切到测试窗
        sig_test = signal.loc[str(test_start):str(test_end)]
        if sig_test.dropna(how="all").empty:
            print(f"  折{Y}: 测试窗无信号, 跳过")
            continue

        trend_w, dd_thr = 120, -0.10
        # A) anti 75% (上证综指判牛熊)
        bt_anti = run_backtest(sig_test, close, TOP_K, HOLD, COST,
                               gate_mode="anti", regime_trend_w=trend_w, regime_dd_thr=dd_thr,
                               stock_cap=args.stock_cap, market_index=args.market_index)
        # B) 无择时 100% (同信号同成本, 仅隔离闸门贡献)
        bt_static = run_backtest(sig_test, close, TOP_K, HOLD, COST,
                                 gate_mode="anti", regime_trend_w=trend_w, regime_dd_thr=dd_thr,
                                 stock_cap=args.stock_cap, market_index=args.market_index,
                                 use_regime=False)

        anti_port_dfs.append(bt_anti["portfolio"])
        anti_bench_dfs.append(bt_anti["benchmark"])
        static_port_dfs.append(bt_static["portfolio"])
        static_bench_dfs.append(bt_static["benchmark"])

        m = summarize(bt_anti["portfolio"], bt_anti["benchmark"])
        fold_rows.append({
            "fold": Y, "n_frozen": len(frozen),
            "ann": m.get("ann_return"), "sharpe": m.get("sharpe"),
            "mdd": m.get("max_drawdown"), "win": m.get("win_rate"),
        })
        print(f"  折{Y}: 训练[{train_start.date()}~{split_date}] 冻结{len(frozen)} "
              f"| anti Sharpe={m.get('sharpe')} 年化={m.get('ann_return')}")

    # 5. 汇总 OOS
    def concat(dfs):
        s = pd.concat(dfs)
        return s[~s.index.duplicated(keep="first")].sort_index()
    anti_port = concat(anti_port_dfs)
    anti_bench = concat(anti_bench_dfs)
    static_port = concat(static_port_dfs)
    static_bench = concat(static_bench_dfs)

    res_anti = summarize(anti_port, anti_bench)
    res_static = summarize(static_port, static_bench)
    print(f"\n[A] anti(75%, 上证综指) OOS: {res_anti}")
    print(f"[B] 无择时100% OOS:         {res_static}")

    out = {
        "config": {
            "start": args.start, "end": args.end,
            "market_index": args.market_index, "stock_cap": args.stock_cap,
            "train_years": args.train_years, "n_folds": len(test_years),
            "n_stocks": close.shape[1], "n_factors": len(all_factors),
            "top_k": TOP_K, "hold": HOLD, "cost": COST,
            "method": "rolling WFA: 因子全面板算一次, 每折 split_date 重算 IS_IC/冻结, 测试窗=真实OOS",
        },
        "results": {
            "A_anti_75_shindex": res_anti,
            "B_no_timing_100": res_static,
        },
        "folds": fold_rows,
        "gate_value": {
            "ann_drag": round(float(res_static.get("ann_return", 0) - res_anti.get("ann_return", 0)), 4),
            "sharpe_drag": round(float(res_static.get("sharpe", 0) - res_anti.get("sharpe", 0)), 3),
        },
    }
    with open(FRAMEWORK / "wfa_fullmarket_anti.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"完成! 总耗时 {(time.time()-t0)/60:.1f} 分钟 → wfa_fullmarket_anti.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
