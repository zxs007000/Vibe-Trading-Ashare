"""xgb_selector.py — 用 XGBoost 选股(排雷结论的"用法"落地).

排雷结论(见 排雷报告 Section 十): founder 头部因子 IC 为正但因子->次日收益
呈"非单调(倒U)"结构, 因此线性多空组合必然亏钱. XGBoost 是树模型, 原生
支持非单调映射, 可把因子信息转化为可用的多头组合.

方法:
  - 7 个头部因子(已静态定向)作为特征, 每个交易日做横截面 z-score(可比).
  - 目标: 次日收益率 fwd (回归).
  - 训练/评估: walk-forward(扩张窗口训练 / 20 交易日验证, 无未来泄漏).
  - 评估: OOS rank-IC / ICIR; 多头组合(每日做多预测收益最高的前 K%) 对比
         等权基准, 以及线性组合的多头基线.

用法:
  python backtest/xgb_selector.py
"""
from __future__ import annotations
import sys, pickle, warnings, time
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import xgboost as xgb
from backtest.factors import founder as F
from src.factors.factor_analysis_core import compute_ic_series
from backtest.validation import _sharpe

CACHE = Path("/workspace/stock_worm/data/ashare_5m_cache.pkl")
START, END = "2023-01-01", "2026-06-30"
N_STOCKS = 100            # 参与建模的股票数(缓存里取前 N 只, 数据越多越好)
TEST_DAYS = 20            # walk-forward 验证窗
TRAIN_MIN = 120           # 训练窗最小长度(交易日)
TOP_K = 0.30              # 多头组合: 每日做多预测收益最高的前 30%

SPECS = [
    ("smart_money",         F.smart_money_batch,         "minute"),
    ("flower_hidden",       F.flower_hidden_batch,       "minute"),
    ("complete_tide",       F.complete_tide_batch,       "minute"),
    ("scaling_heights",     F.scaling_heights_batch,     "minute"),
    ("undercurrent",        F.undercurrent_batch,        "minute"),
    ("withered_tree_blooms",F.withered_tree_blooms,      "single_daily"),
    ("clouds_disperse",     F.clouds_disperse_batch,     "minute"),
]


def _zscore_row(row: pd.Series) -> pd.Series:
    return row.sub(row.mean()).div(row.std(ddof=0) + 1e-9)


def main():
    t0 = time.time()
    all_min = pickle.load(open(CACHE, "rb"))
    codes = sorted(all_min)[:N_STOCKS]
    stocks = {c: all_min[c] for c in codes}
    daily_close = pd.DataFrame({c: d["close"].resample("D").last() for c, d in stocks.items()}).sort_index()
    daily_ret = daily_close.pct_change()
    fwd = daily_ret.shift(-1)
    daily_bars = {c: pd.DataFrame({
        "open": d["open"].resample("D").first(), "high": d["high"].resample("D").max(),
        "low": d["low"].resample("D").min(), "close": d["close"].resample("D").last(),
        "volume": d["volume"].resample("D").sum()}) for c, d in stocks.items()}
    trading = daily_close.dropna(how="all").index
    daily_close = daily_close.reindex(trading)
    daily_ret = daily_close.pct_change(); fwd = daily_ret.shift(-1)
    daily_bars = {c: b.reindex(trading) for c, b in daily_bars.items()}

    # 1) 计算 7 因子矩阵(交易日对齐), 静态定向
    raw = {}
    for name, fn, kind in SPECS:
        res = fn(stocks) if kind == "minute" else {c: fn(b) for c, b in daily_bars.items()}
        fdf = pd.DataFrame({c: s for c, s in res.items() if len(s.dropna()) > 5}).reindex(trading)
        if fdf.shape[1] < 5 or fdf.dropna(how="all").empty:
            print(f"  ⚠ {name} 有效股票不足, 跳过"); continue
        ic = compute_ic_series(fdf, fwd)
        orient = 1.0 if ic.mean() >= 0 else -1.0
        raw[name] = fdf * orient
        print(f"  {name:<20} 有效股票={fdf.shape[1]:3d}  全窗口IC={ic.mean():+.4f} 定向={'高' if orient>0 else '低'}")

    factor_names = list(raw)
    print(f"\n  建模特征={factor_names}  股票={len(codes)}  交易日={len(trading)}")

    # 2) 构造面板: 每个因子按交易日横截面 z-score(可比), 目标=次日收益
    #    统一用 stack() 得到 (date, code) 二级索引, 避免 join 层级错位.
    #    只要求目标(次日收益)非空; 因子特征缺失交给 XGBoost 原生 NaN 处理,
    #    不要把 7 因子做硬交集(各因子覆盖天数不一, 硬交集会清空样本).
    zfac = {n: raw[n].apply(_zscore_row, axis=1) for n in factor_names}
    feat_df = pd.concat([zfac[n].stack().rename(n) for n in factor_names], axis=1)
    y_series = fwd.stack().rename("y")
    panel = pd.concat([feat_df, y_series], axis=1).dropna(subset=["y"])
    lin_series = feat_df.mean(axis=1).rename("lin")    # 线性组合(等权)基线
    print(f"  面板样本数={len(panel)} (date×code), 覆盖交易日={panel.index.get_level_values(0).nunique()}")

    # 3) walk-forward: 扩张窗口训练, 滚动验证
    dates = sorted(panel.index.get_level_values(0).unique())
    feats = factor_names
    splits, pred_records = [], []
    start = 0
    while start + TRAIN_MIN + TEST_DAYS <= len(dates):
        tr_end = start + TRAIN_MIN
        te_end = min(tr_end + TEST_DAYS, len(dates))
        train_dates = dates[:tr_end]
        test_dates = dates[tr_end:te_end]
        tr = panel.loc[panel.index.get_level_values(0).isin(train_dates)]
        te = panel.loc[panel.index.get_level_values(0).isin(test_dates)]
        if len(tr) < 200 or te.empty:
            start += TEST_DAYS; continue
        model = xgb.XGBRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=42, verbosity=0)
        model.fit(tr[feats].values, tr["y"].values)
        p = model.predict(te[feats].values)
        rec = te.copy(); rec["pred"] = p
        pred_records.append(rec)
        start += TEST_DAYS

    if not pred_records:
        print("  样本不足, 无法 walk-forward"); return
    oos = pd.concat(pred_records).sort_index()
    print(f"  OOS 预测样本={len(oos)}  覆盖交易日={oos.index.get_level_values(0).nunique()}")

    # 4) OOS rank-IC / ICIR(每日横截面: pred vs 真实 fwd)
    ic_rows = []
    for d, g in oos.groupby(level=0):
        if g["pred"].notna().sum() < 5 or g["y"].notna().sum() < 5: continue
        pr = g["pred"].rank(); yr = g["y"].rank()
        ic_rows.append(pr.corr(yr))
    ic_oos = pd.Series(ic_rows).dropna()
    icir = ic_oos.mean() / ic_oos.std() * np.sqrt(252) if ic_oos.std() > 0 else np.nan
    print(f"\n  ── XGBoost OOS 评估 ──")
    print(f"    OOS IC_mean = {ic_oos.mean():+.4f}  ICIR = {icir:.3f}  ic_pos = {(ic_oos>0).mean():.3f}")

    # 5) 多头组合(带持有期 + 换手成本敏感性)
    #    hold_days=1: 每日调仓(换手最高, 成本敏感); hold_days=5: 每周调仓.
    #    成本模型: 每次调仓满仓换 -> 换手≈sleeve, 成本=sleeve*2*单边, 仅在调仓日计.
    def long_only_portfolio(oos, top_k, hold_days, cost_oneway=0.001):
        dates = sorted(oos.index.get_level_values(0).unique())
        port_ret, bench_ret = [], []
        held = None
        for i, d in enumerate(dates):
            g = oos.xs(d, level=0).dropna(subset=["pred", "y"])
            if len(g) < 5:
                port_ret.append(np.nan); bench_ret.append(np.nan); continue
            if i % hold_days == 0:
                k = max(3, int(len(g) * top_k))
                held = set(g["pred"].nlargest(k).index)
            sub = g.loc[g.index.intersection(held), "y"] if held else g["y"]
            r = sub.mean() if len(sub) > 0 else np.nan
            if i % hold_days == 0 and held is not None:
                r = r - top_k * 2 * cost_oneway          # 调仓日扣双边成本
            port_ret.append(r)
            bench_ret.append(g["y"].mean())
        port = pd.Series(port_ret).dropna(); bench = pd.Series(bench_ret).dropna()
        return port, bench

    for hd in (1, 5):
        port, bench = long_only_portfolio(oos, TOP_K, hd)
        excess = port - bench
        tag = f"XGBoost 多头(前{TOP_K:.0%}, 持有{hd}日, 含成本)" if hd == 1 else f"XGBoost 多头(前{TOP_K:.0%}, 持有{hd}日, 含成本)"
        print(f"\n  ── {tag} ──")
        print(f"    多头日收益均值={port.mean():+.5f}  夏普={_sharpe(port.values):.3f}  累计={(1+port).cumprod().iloc[-1]:.4f}")
        print(f"    基准日收益均值={bench.mean():+.5f}  夏普={_sharpe(bench.values):.3f}  累计={(1+bench).cumprod().iloc[-1]:.4f}")
        print(f"    超额(减基准) 日收益均值={excess.mean():+.5f}  夏普={_sharpe(excess.values):.3f}  累计={(1+excess).cumprod().iloc[-1]:.4f}")

    # 6) 对照: 线性组合的多头基线(每日, 前 TOP_K, 含成本) —— 结构对比
    lin_oos = oos.join(lin_series.reindex(oos.index), rsuffix="_lin")
    lin_ret, lin_bench = [], []
    held = None
    dates = sorted(lin_oos.index.get_level_values(0).unique())
    for i, d in enumerate(dates):
        g = lin_oos.xs(d, level=0).dropna(subset=["lin", "y"])
        if len(g) < 5:
            lin_ret.append(np.nan); lin_bench.append(np.nan); continue
        if i % 1 == 0:
            k = max(3, int(len(g) * TOP_K))
            held = set(g["lin"].nlargest(k).index)
        sub = g.loc[g.index.intersection(held), "y"] if held else g["y"]
        r = sub.mean() if len(sub) > 0 else np.nan
        r = r - TOP_K * 2 * 0.001
        lin_ret.append(r); lin_bench.append(g["y"].mean())
    lin_port = pd.Series(lin_ret).dropna()
    print(f"\n  ── [对照]线性组合多头(前{TOP_K:.0%}, 持有1日, 含成本) ──")
    print(f"    多头日收益均值={lin_port.mean():+.5f}  夏普={_sharpe(lin_port.values):.3f}  累计={(1+lin_port).cumprod().iloc[-1]:.4f}")

    print(f"\n  耗时 {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
