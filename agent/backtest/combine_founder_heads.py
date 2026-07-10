"""combine_founder_heads.py — 头部因子方向自适应组合(排雷的"用"那一步).

对照两种方向处理方式:
  A. 静态定向: 全窗口定一次方向+权重(跨行情会失效, 见报告 Section 八)
  B. walk-forward: 滚动窗口(训练 120 交易日 / 验证 20 交易日)逐窗口重定方向+ICIR 权重,
     仅用过去数据, 无未来泄漏. 验证段拼接成完整 OOS 序列再评估.

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
CACHE = Path("/workspace/stock_worm/data/ashare_5m_cache.pkl")
START, END = "2023-01-01", "2026-06-30"
TRAIN_DAYS = 120   # 训练窗口(交易日)
TEST_DAYS = 20     # 验证窗口(交易日)

# 头部因子(排雷: 方向自洽, 可用多空夏普>=0.3)
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
    return row.sub(row.mean()).div(row.std(ddof=0) + 1e-9)


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


def _orient_combine_static(raw: dict, fwd: pd.DataFrame, mode: str = "icir"):
    """静态: 全窗口定一次方向 + 权重(look-ahead 基线)."""
    oriented = {}
    for name, df in raw.items():
        ic = compute_ic_series(df, fwd)
        orient = 1.0 if ic.mean() >= 0 else -1.0
        oriented[name] = df * orient
    zscored = {n: df.apply(_zscore_row, axis=1) for n, df in oriented.items()}
    equal = pd.concat(zscored.values(), axis=0, keys=list(zscored)).swaplevel().sort_index().groupby(level=0).mean()
    icir = {n: (abs(compute_ic_series(df, fwd).mean() / compute_ic_series(df, fwd).std() * np.sqrt(252))
               if compute_ic_series(df, fwd).std() > 0 else 0.0) for n, df in raw.items()}
    w = pd.Series(icir); w = w / w.sum()
    wmat = pd.concat([zscored[n] * w[n] for n in zscored], axis=0, keys=list(zscored)).swaplevel().sort_index()
    weighted = wmat.groupby(level=0).sum() if mode == "icir" else equal
    return equal, (weighted if mode == "icir" else equal)


def walk_forward_combine(raw: dict, fwd: pd.DataFrame, train_days: int = TRAIN_DAYS,
                         test_days: int = TEST_DAYS, mode: str = "icir") -> pd.DataFrame | None:
    """walk-forward: 方向用扩张窗口(累计历史, 稳定慢适应, 避免短窗噪声导致方向乱翻),
    权重用滚动训练窗口(responsive). 仅用过去数据, 无未来泄漏."""
    all_dates = sorted(set().union(*[set(df.index) for df in raw.values()]))
    all_dates = [d for d in all_dates if d in fwd.index]
    n = len(all_dates)
    if n < train_days + test_days:
        return None
    chunks = []
    start = 0
    while start + train_days < n:
        tr_end = start + train_days
        te_end = min(tr_end + test_days, n)
        train_d = all_dates[start:tr_end]
        test_d = all_dates[tr_end:te_end]
        past_d = all_dates[:tr_end]   # 扩张窗口: 截至测试窗前的全部历史
        zparts, wsum = [], 0.0
        for name, df in raw.items():
            # 方向: 扩张窗口(稳定)
            ic_all = compute_ic_series(df.loc[past_d], fwd.loc[past_d])
            if ic_all.empty or ic_all.isna().all():
                continue
            orient = 1.0 if ic_all.mean() >= 0 else -1.0
            # 权重: 滚动训练窗口(responsive)
            ic_tr = compute_ic_series(df.loc[train_d], fwd.loc[train_d])
            icir_v = abs(ic_tr.mean() / ic_tr.std() * np.sqrt(252)) if (not ic_tr.empty and ic_tr.std() > 0) else 0.0
            z = (df.loc[test_d] * orient).apply(_zscore_row, axis=1)
            if mode == "icir":
                zparts.append(z * icir_v); wsum += icir_v
            else:
                zparts.append(z)
        if zparts:
            combo = (sum(zparts) / wsum) if (mode == "icir" and wsum > 0) else (sum(zparts) / len(zparts))
            chunks.append(combo)
        start += test_days
    return pd.concat(chunks).sort_index() if chunks else None


def report(tag: str, score: pd.DataFrame, fwd: pd.DataFrame):
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
    # 统一对齐到交易日(去掉周末/节假日), 避免 5m因子(交易日) 与 daily_bars/fwd(日历日) 索引错位
    trading = daily_close.dropna(how="all").index
    daily_close = daily_close.reindex(trading)
    daily_ret = daily_close.pct_change()
    fwd = daily_ret.shift(-1)
    daily_bars = {c: b.reindex(trading) for c, b in daily_bars.items()}

    # 原始因子(未定向), 用于 walk-forward 与静态基线
    raw = {}
    for name, fn, kind in SPECS:
        if kind == "minute":
            res = fn(stocks_minute)
        else:
            res = {c: fn(b) for c, b in daily_bars.items()}
        fdf = pd.DataFrame({c: s for c, s in res.items() if len(s.dropna()) > 5})
        fdf = fdf.reindex(trading)  # 对齐交易日(去掉周末/节假日)
        if fdf.shape[1] < 5 or fdf.dropna(how="all").empty:
            print(f"  ⚠ {name} 有效股票不足, 跳过"); continue
        raw[name] = fdf
        ic = compute_ic_series(fdf, fwd)
        print(f"  {name:<20} 全窗口 IC_mean={ic.mean():+.4f} 静态方向={'做多高值' if ic.mean()>=0 else '做多低值(取反)'}")

    if not raw:
        print("无可用因子"); return

    print("\n" + "=" * 64)
    print("头部因子组合: 静态定向 vs walk-forward")
    print("=" * 64)
    s_eq, s_w = _orient_combine_static(raw, fwd, "icir")
    report("静态-等权(全窗口定向)", s_eq, fwd)
    report("静态-ICIR加权(全窗口定向)", s_w, fwd)

    wf_eq = walk_forward_combine(raw, fwd, mode="equal")
    wf_w = walk_forward_combine(raw, fwd, mode="icir")
    if wf_eq is not None:
        report(f"walk-forward-等权(训{TRAIN_DAYS}/验{TEST_DAYS})", wf_eq, fwd)
    else:
        print("  walk-forward 等权: 数据不足")
    if wf_w is not None:
        report(f"walk-forward-ICIR加权(训{TRAIN_DAYS}/验{TEST_DAYS})", wf_w, fwd)
    else:
        print("  walk-forward ICIR加权: 数据不足")
    print("=" * 64)


if __name__ == "__main__":
    main()
