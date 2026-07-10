"""compute_founder_factors.py — 计算 7 个 founder 头部因子并落盘缓存(可断点续算).

缓存路径: /workspace/stock_worm/data/founder_factors.pkl
  -> dict: {因子名: DataFrame(date×code, 已交易日对齐, 静态定向前)}
数据: stock_worm 源 5m 本地缓存(288 只沪深300, 2024-05-27~2026-06-30).

支持续算: 已算出的因子存在缓存里就跳过, 进程被 kill 也不会丢进度.

用法:
  python backtest/compute_founder_factors.py
"""
from __future__ import annotations
import sys, pickle, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.factors import founder as F

CACHE = Path("/workspace/stock_worm/data/ashare_5m_cache.pkl")
OUT = Path("/workspace/stock_worm/data/founder_factors.pkl")

SPECS = [
    ("smart_money",          F.smart_money_batch,         "minute"),
    ("flower_hidden",        F.flower_hidden_batch,       "minute"),
    ("complete_tide",        F.complete_tide_batch,       "minute"),
    ("scaling_heights",      F.scaling_heights_batch,     "minute"),
    ("undercurrent",         F.undercurrent_batch,        "minute"),
    ("withered_tree_blooms", F.withered_tree_blooms,      "single_daily"),
    ("clouds_disperse",      F.clouds_disperse_batch,     "minute"),
]


def main():
    t0 = time.time()
    all_min = pickle.load(open(CACHE, "rb"))
    codes = sorted(all_min)
    stocks = {c: all_min[c] for c in codes}
    print(f"[{time.time()-t0:.1f}s] 加载 {len(stocks)} 只")

    trading = pd.DataFrame({c: d["close"].resample("D").last() for c, d in stocks.items()}).sort_index()
    trading = trading.dropna(how="all").index
    daily_bars = {c: pd.DataFrame({
        "open": d["open"].resample("D").first(), "high": d["high"].resample("D").max(),
        "low": d["low"].resample("D").min(), "close": d["close"].resample("D").last(),
        "volume": d["volume"].resample("D").sum()}).reindex(trading) for c, d in stocks.items()}

    # 续算: 读已有缓存
    done = {}
    if OUT.exists():
        done = pickle.load(open(OUT, "rb"))
        print(f"[{time.time()-t0:.1f}s] 已有缓存因子: {sorted(done)}")

    for name, fn, kind in SPECS:
        if name in done and isinstance(done[name], pd.DataFrame) and done[name].shape[1] >= 5:
            print(f"  ✓ {name} 已缓存({done[name].shape[1]} 只), 跳过")
            continue
        t = time.time()
        if kind == "minute":
            res = fn(stocks)
        else:
            res = {c: fn(b) for c, b in daily_bars.items()}
        fdf = pd.DataFrame({c: s for c, s in res.items() if len(s.dropna()) > 5}).reindex(trading)
        if fdf.shape[1] < 5 or fdf.dropna(how="all").empty:
            print(f"  ⚠ {name} 有效股票不足, 跳过"); continue
        done[name] = fdf
        pickle.dump(done, open(OUT, "wb"))   # 算完一个存一个(断点续算)
        print(f"  ✓ {name:<20} {fdf.shape[1]} 只  [+{time.time()-t:.1f}s, 累计 {time.time()-t0:.1f}s]")

    print(f"[{time.time()-t0:.1f}s] 完成, 共 {len(done)} 个因子 -> {OUT}")
    # 打印各因子全窗口 IC 方向(静态), 供回测参考
    dc = pd.DataFrame({c: all_min[c]["close"].resample("D").last() for c in codes}).sort_index()
    fwd = dc.pct_change().shift(-1).reindex(trading)
    from src.factors.factor_analysis_core import compute_ic_series
    for n, df in done.items():
        ic = compute_ic_series(df, fwd)
        print(f"    {n:<20} IC={ic.mean():+.4f} 定向={'高' if ic.mean()>=0 else '低'}")


if __name__ == "__main__":
    main()
