"""diag_ls.py — 诊断: 为何组合 IC 为正但多空夏普为负.

不改动主脚本, 单独跑一份诊断:
  - 复算静态等权组合 score
  - 打印 score 与 fwd 的逐日关系 (IC, 以及直接 Pearson)
  - 打印 Group_5 - Group_1 的逐日收益符号频率 / 累计净值
  - 检查是否因子对 fwd 存在"符号正确但被少数极端股拖垮"的非线性
"""
from __future__ import annotations
import sys, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.factors import founder as F
from backtest.loaders.astockdata_loader import DataLoader
from src.factors.factor_analysis_core import compute_ic_series, compute_group_equity
from backtest.validation import _sharpe

CACHE = Path("/workspace/stock_worm/data/ashare_5m_cache.pkl")
START, END = "2023-01-01", "2026-06-30"

SPECS = [
    ("smart_money", F.smart_money_batch, "minute"),
    ("flower_hidden", F.flower_hidden_batch, "minute"),
    ("complete_tide", F.complete_tide_batch, "minute"),
    ("scaling_heights", F.scaling_heights_batch, "minute"),
    ("undercurrent", F.undercurrent_batch, "minute"),
    ("withered_tree_blooms", F.withered_tree_blooms, "single_daily"),
    ("clouds_disperse", F.clouds_disperse_batch, "minute"),
]

def _zscore_row(row): return row.sub(row.mean()).div(row.std(ddof=0)+1e-9)

def main():
    all_min = pickle.load(open(CACHE, "rb"))
    SUB = sorted(all_min)[:30]
    stocks = {c: all_min[c] for c in SUB}
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

    raw = {}
    for name, fn, kind in SPECS:
        res = fn(stocks) if kind == "minute" else {c: fn(b) for c, b in daily_bars.items()}
        fdf = pd.DataFrame({c: s for c, s in res.items() if len(s.dropna()) > 5}).reindex(trading)
        if fdf.shape[1] < 5 or fdf.dropna(how="all").empty: continue
        raw[name] = fdf

    # 静态定向 + 等权组合
    oriented = {}
    for name, df in raw.items():
        ic = compute_ic_series(df, fwd); orient = 1.0 if ic.mean() >= 0 else -1.0
        oriented[name] = df * orient
    zscored = {n: df.apply(_zscore_row, axis=1) for n, df in oriented.items()}
    score = pd.concat(zscored.values(), axis=0, keys=list(zscored)).swaplevel().sort_index().groupby(level=0).mean()

    ic = compute_ic_series(score, fwd)
    print(f"组合 IC_mean={ic.mean():+.4f}  ICIR={ic.mean()/ic.std()*np.sqrt(252):.3f}  ic_pos={(ic>0).mean():.3f}")

    # 直接 Pearson 相关 (z-score 组合 vs fwd)
    common = score.index.intersection(fwd.index)
    sc = score.loc[common].rank(axis=1, method="average")
    fr = fwd.loc[common].rank(axis=1, method="average")
    # 每个子因子的 rank-IC
    print("\n逐因子 rank-IC (静态定向后):")
    for n, df in oriented.items():
        f = compute_ic_series(df, fwd)
        print(f"  {n:<20} IC={f.mean():+.4f}  ICIR={f.mean()/f.std()*np.sqrt(252):.2f}  pos={(f>0).mean():.2f}")

    # 多空逐日收益
    eq = compute_group_equity(score, fwd, 5)
    gr = eq.pct_change().dropna()
    ls = gr["Group_5"] - gr["Group_1"]
    print(f"\n多空逐日收益: 均值={ls.mean():+.5f}  胜率={(ls>0).mean():.3f}  sharpe={_sharpe(ls.values):.3f}")
    print(f"Group_5 逐日: 均值={gr['Group_5'].mean():+.5f}  sharpe={_sharpe(gr['Group_5'].values):.3f}")
    print(f"Group_1 逐日: 均值={gr['Group_1'].mean():+.5f}  sharpe={_sharpe(gr['Group_1'].values):.3f}")
    print(f"Group_5 累计净值期末 = {eq['Group_5'].iloc[-1]:.4f}  (期初1.0)")
    print(f"Group_1 累计净值期末 = {eq['Group_1'].iloc[-1]:.4f}")
    print(f"多空累计净值期末   = {(1+ls).cumprod().iloc[-1]:.4f}")

    # 关键检查: 组合分数最高组真的对应高 fwd 吗? 用十分位看单调性
    print("\n十分位单调检查 (组合分数 -> 次日收益均值):")
    cd = score.index.intersection(fwd.index)
    cc = score.columns.intersection(fwd.columns)
    s2 = score.loc[cd, cc]; f2 = fwd.loc[cd, cc]
    rows = []
    for d in cd:
        f = s2.loc[d].dropna(); r = f2.loc[d].dropna()
        sh = f.index.intersection(r.index)
        if len(sh) < 10: continue
        rk = f[sh].rank(method="first")
        dec = pd.qcut(rk, 10, labels=False, duplicates="drop")
        rows.append(r[sh].groupby(dec).mean())
    dec_df = pd.DataFrame(rows)
    print(dec_df.mean().round(5).to_string())

    # 分年度看 ls 与 IC, 验证"regime 依赖"
    print("\n分年度 IC 与 多空逐日收益:")
    yr = pd.Series(cd).apply(lambda x: str(x.year))
    for y, idx in pd.Series(cd).groupby(yr):
        sub = ic.loc[[i for i in idx if i in ic.index]]
        if len(sub) == 0: continue
        subls = ls.loc[[i for i in idx if i in ls.index]]
        print(f"  {y}: n={len(sub)} IC={sub.mean():+.4f} ls_mean={subls.mean():+.5f} ls_sharpe={_sharpe(subls.values):.3f}")

if __name__ == "__main__":
    main()
