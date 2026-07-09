"""combine_founder_heads.py — 头部因子方向自适应组合(排雷的"用"那一步).

把已排雷、方向自洽的方正头部因子, 按各自实测方向定向 → 每日截面 z-score
→ 等权 / ICIR 加权合成综合打分 → 评估组合层 IC / ICIR / 多空夏普.

用法:
  python backtest/combine_founder_heads.py
"""
from __future__ import annotations

import sys, logging, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))      # agent/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from backtest.factors import founder as F
from backtest.loaders.astockdata_loader import DataLoader
from src.factors.factor_analysis_core import compute_ic_series, compute_group_equity
from backtest.validation import _sharpe

logging.disable(logging.CRITICAL)
CACHE = Path("/workspace/stock_worm/data/ashare_5m_cache.pkl")  # 备份: 全CSI300 3.5y 5m(带amount)
START, END = "2025-01-01", "2026-06-30"

# 头部因子(来自 eval_custom_factors Part B 排雷: 方向自洽, 可用多空夏普>=0.3)
# 每个: (batch_fn, 输入类型, 单股函数备用)
SPECS = [
    ("smart_money",        F.smart_money_batch,        "minute"),
    ("flower_hidden",      F.flower_hidden_batch,      "minute"),
    ("complete_tide",      F.complete_tide_batch,      "minute"),
    ("scaling_heights",    F.scaling_heights_batch,    "minute"),
    ("undercurrent",       F.undercurrent_batch,       "minute"),
    ("withered_tree_blooms", F.withered_tree_blooms,    "single_daily"),
    ("clouds_disperse",    F.clouds_disperse_batch,    "minute"),
]


def _zscore_row(row: pd.Series) -> pd.Series:
    s = row.sub(row.mean()).div(row.std(ddof=0) + 1e-9)
    return s


def _load_or_pull() -> dict:
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    print("  缓存缺失, 重新拉取 5m (带 amount)...")
    codes = ["600519","000858","601318","600036","000333","601899","300750","601166",
             "600900","000651","600276","601398","000001","603259","600030","002415",
             "601288","600809","000725","601088","601012","002714","000002","600887",
             "601857","600028","688981","002475","300059","601688","000063","300124",
             "600585","600309","600436","002594","601225","603288","002304","000568",
             "601995","600570","002352","300015","601066","600104","601628","000776",
             "300498","002230"]
    stocks = {}
    for code in codes:
        r = DataLoader().fetch([code], START, END, interval="5m")
        for k, v in r.items():
            if v is not None and not v.empty:
                stocks[k] = v
    pickle.dump(stocks, open(CACHE, "wb"))
    print(f"  拉取 {len(stocks)} 只并落盘缓存")
    return stocks


def main():
    all_min = _load_or_pull()
    SUB = sorted(all_min)[:30]
    stocks_minute = {c: all_min[c] for c in SUB}

    daily_close = pd.DataFrame({c: d["close"].resample("D").last()
                                for c, d in stocks_minute.items()}).sort_index()
    daily_ret = daily_close.pct_change()
    fwd = daily_ret.shift(-1)
    daily_bars = {c: pd.DataFrame({
        "open": d["open"].resample("D").first(),
        "high": d["high"].resample("D").max(),
        "low": d["low"].resample("D").min(),
        "close": d["close"].resample("D").last(),
        "volume": d["volume"].resample("D").sum()}) for c, d in stocks_minute.items()}

    oriented = {}   # name -> 定向后(做多高值) 日因子 DataFrame
    ic_means = {}
    for name, fn, kind in SPECS:
        if kind == "minute":
            res = fn(stocks_minute)
        else:  # single_daily
            res = {c: fn(b) for c, b in daily_bars.items()}
        fdf = pd.DataFrame({c: s for c, s in res.items() if len(s.dropna()) > 5})
        if fdf.shape[1] < 5:
            print(f"  ⚠ {name} 有效股票不足, 跳过"); continue
        ic = compute_ic_series(fdf, fwd)
        if ic.empty:
            print(f"  ⚠ {name} IC 为空, 跳过"); continue
        orient = 1.0 if ic.mean() >= 0 else -1.0
        ic_means[name] = float(ic.mean())
        oriented[name] = fdf * orient
        print(f"  {name:<20} IC_mean={ic.mean():+.4f} 定向={'做多高值' if orient>0 else '做多低值(取反)'}")

    if not oriented:
        print("无可用因子"); return

    # 截面 z-score, dict(name -> DataFrame[date,stock])
    zscored = {n: df.apply(_zscore_row, axis=1) for n, df in oriented.items()}
    # 堆成 (date, factor) 多级索引, 便于按日期跨因子聚合
    zmat = pd.concat(zscored.values(), axis=0, keys=list(zscored)).swaplevel().sort_index()

    # 等权组合: 按日期对"可用因子"取均值(pandas groupby.mean 自动跳过 NaN)
    equal = zmat.groupby(level=0).mean()

    # ICIR 加权组合: 各因子先乘权重, 再按日期求和(先乘后聚合, 绕开 MultiIndex mul 坑)
    icir = {}
    for n, df in oriented.items():
        ic = compute_ic_series(df, fwd)
        icir[n] = abs(ic.mean() / ic.std() * np.sqrt(252)) if ic.std() > 0 else 0.0
    w = pd.Series(icir)
    w = w / w.sum()
    wmat = pd.concat([zscored[n] * w[n] for n in zscored],
                     axis=0, keys=list(zscored)).swaplevel().sort_index()
    weighted = wmat.groupby(level=0).sum()

    def report(tag, score: pd.DataFrame):
        ic = compute_ic_series(score, fwd)
        ic_mean, ic_std = float(ic.mean()), float(ic.std())
        icir_v = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else np.nan
        eq = compute_group_equity(score, fwd, n_groups=5)
        gr = eq.pct_change().dropna()
        ls = _sharpe((gr["Group_5"] - gr["Group_1"]).values) if len(gr) > 20 else np.nan
        top = _sharpe(gr["Group_5"].values) if len(gr) > 20 else np.nan
        print(f"\n  ── {tag} ──")
        print(f"    IC_mean = {ic_mean:+.4f}   ICIR = {icir_v:.3f}   "
              f"ic_pos = {float((ic>0).mean()):.3f}")
        print(f"    多空夏普(Group5-Group1) = {ls:.3f}   最高组夏普 = {top:.3f}")

    print("\n" + "=" * 64)
    print("头部因子组合(方向自适应 + 截面zscore)")
    print("=" * 64)
    report("等权组合", equal)
    report("ICIR加权组合", weighted)
    print(f"\n  权重(ICIR): " + ", ".join(f"{n}={w[n]:.2f}" for n in w.index))
    print("=" * 64)


if __name__ == "__main__":
    main()
