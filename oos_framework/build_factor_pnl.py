"""
build_factor_pnl.py — 因子日频多空收益序列 (REGIME_FACTOR Plan Phase 1, 优化版)

优化: 用 numpy argsort 替代逐日 pandas nlargest, 将 O(n*dates*log(n)) 开销降低 10x+.

方法:
  对每个因子, 每日对截面因子值 argsort → top20% 做多 / bottom20% 做空.
  多空收益 = mean(forward_return[long]) - mean(forward_return[short]).

产出:
  oos_framework/screen_results/factor_pnl.parquet (dates × factors)

用法:
  cd /workspace/VibeTradingPush && python oos_framework/build_factor_pnl.py
"""

import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = Path("/workspace/stock_worm/data")
SF_PANEL = DATA / "ashare_daily_panel_survivorfree.parquet"
ALIVE_PANEL = DATA / "ashare_daily_panel.parquet"
CSRC_MAP = DATA / "csrc_industry_map.parquet"
FUND_PARQUET = DATA / "fundamentals/fund_factors_daily.parquet"
FUND_NAMES = ["ROE", "rev_yoy", "profit_yoy"]
OUT_DIR = HERE / "screen_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLD = 5
TOP_PCT = 0.20

sys.path.insert(0, str(HERE.parent / "agent/backtest"))
from factor_zoo_daily import build_factors, neutralize_factors, ALL_FACTOR_NAMES
from oos_validation_corrected import load_wide_sf, build_zarr


def build_factor_pnl_vectorized(zarr, fwd, factor_names, top_pct=TOP_PCT):
    """向量化构建因子日频多空收益.

    用 numpy argsort 替代逐行 pandas nlargest/nsmallest, 速度可快 10x+.
    """
    dates = fwd.index
    codes = fwd.columns
    n_dates, n_codes = len(dates), len(codes)
    n_sel = max(1, int(n_codes * top_pct))
    pnl = np.full((n_dates, len(factor_names)), np.nan, dtype=np.float32)
    fwd_vals = fwd.values.astype(np.float32)

    for i, f in enumerate(factor_names):
        scores = zarr[f]  # numpy array (n_dates, n_codes)
        # 每行降序 argsort (最大值的索引在最前)
        order = np.argsort(-scores, axis=1)  # (n_dates, n_codes)

        # 取多头: 前 top_pct% 的列索引
        long_idx = order[:, :n_sel]   # (n_dates, n_sel)
        short_idx = order[:, -n_sel:]  # (n_dates, n_sel)

        # 用 take_along_axis 取对应位置的远期收益
        ret_long = np.take_along_axis(fwd_vals, long_idx, axis=1)
        ret_short = np.take_along_axis(fwd_vals, short_idx, axis=1)

        # 忽略 NaN, 算有效值均值
        long_mean = np.nanmean(ret_long, axis=1)
        short_mean = np.nanmean(ret_short, axis=1)

        # 去掉 NaN 占太大的日期
        long_valid = (~np.isnan(ret_long)).sum(axis=1)
        short_valid = (~np.isnan(ret_short)).sum(axis=1)
        valid = (long_valid >= 3) & (short_valid >= 3)
        pnl[valid, i] = long_mean[valid] - short_mean[valid]

    return pd.DataFrame(pnl, index=dates, columns=factor_names)


def main():
    t0 = time.time()
    print("=" * 60)
    print("因子日频多空收益构建 (向量化版)")
    print("=" * 60)

    print("\n[1/2] 加载面板 + 构建因子 z-score...")
    w = load_wide_sf()
    n_codes = w["close"].shape[1]
    fwd = w["close"].pct_change(HOLD).shift(-HOLD).clip(-0.5, 0.5)
    dates, codes = fwd.index, fwd.columns
    mp = pd.read_parquet(CSRC_MAP)
    ind_map = dict(zip(mp["code"], mp["csrc_industry"]))
    fac = build_factors(w)
    fac = neutralize_factors(fac, ind_map)
    del w
    fund = pd.read_pickle(FUND_PARQUET)
    for f in FUND_NAMES:
        fac[f] = fund[f]
    ALL = ALL_FACTOR_NAMES + FUND_NAMES
    zarr = build_zarr(fac, ALL, dates, codes)
    del fac
    for f in FUND_NAMES:
        zarr[f] = np.nan_to_num(zarr[f], nan=0.0)
    print(f"  面板: {n_codes} 只 × {dates[0].date()}~{dates[-1].date()}")
    print(f"  因子: {len(ALL)}")

    print("\n[2/2] 向量化计算每因子日频多空收益...")
    pnl = build_factor_pnl_vectorized(zarr, fwd, ALL)
    out_path = OUT_DIR / "factor_pnl.parquet"
    pnl.to_parquet(out_path)
    print(f"\n  输出: {out_path}")
    print(f"  维度: {pnl.shape}")
    print(f"  有效日范围: {pnl.index[0].date()}~{pnl.index[-1].date()}")
    mv = pnl.mean()
    print(f"  各因子均值: [{mv.min():+.5f}, {mv.max():+.5f}]")
    print(f"  有数据因子: {(~pnl.isna().all()).sum()}/{len(ALL)}")
    print(f"耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
