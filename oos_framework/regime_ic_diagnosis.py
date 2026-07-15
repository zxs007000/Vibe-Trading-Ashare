"""
regime_ic_diagnosis.py — 市场级 Regime IC 诊断 (REGIME_TRAINING Plan Phase 1)

目的:
  确认因子是否真的有 regime 依赖: 在牛/熊市中同一因子的 IC 符号或量级是否显著不同。
  若 ≥30% 因子有 regime 符号反转 (牛市正 IC、熊市负 IC, 或反之),
  则值得做 Phase 2 (regime-conditioned WFA)。

方法:
  1. 用上证综指 MA120 把全量历史分割为 bull/bear 两段
  2. 对每个因子分段计算 rank_IC 均值、标准差、ICIR
  3. 标记: 符号反转 (IC 符号在牛熊间对调)、量级差异 (|ΔIC| > 2σ)
  4. 输出诊断表 CSV + 结论

用法:
  python oos_framework/regime_ic_diagnosis.py
"""

import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 路径 ──
HERE = Path(__file__).parent
DATA = Path("/workspace/stock_worm/data")
SF_PANEL = DATA / "ashare_daily_panel_survivorfree.parquet"
ALIVE_PANEL = DATA / "ashare_daily_panel.parquet"
CSRC_MAP = DATA / "csrc_industry_map.parquet"
FUND_PARQUET = DATA / "fundamentals/fund_factors_daily.parquet"
FUND_NAMES = ["ROE", "rev_yoy", "profit_yoy"]
INDEX_LAKE = Path("D:/work Buddy GZ/Claw/stockworm/index")  # 用户本地路径(回退即不使用)

OUT_DIR = HERE / "screen_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 参数 ──
TRAIL = 250         # IC 滚动窗口 (WFA 一致)
HOLD = 5            # 持有期 (IC 计算)
TOP_K = 0.30
COST = 0.001
BEAR_MA = 120       # MA120 判熊
BEAR_THR = -0.10    # close/MA120 - 1 < -10% 判熊

sys.path.insert(0, str(HERE.parent / "agent/backtest"))
from factor_zoo_daily import build_factors, neutralize_factors, ALL_FACTOR_NAMES, daily_rank_ic
from oos_validation_corrected import load_wide_sf, build_zarr


def _alive_mkt_level(dates):
    """活股等权指数 (日收益等权均值累乘), 作 regime 判定的市场代理."""
    alive = pd.read_parquet(ALIVE_PANEL)
    alive["_d"] = pd.to_datetime(alive["date"]).dt.normalize()
    cal = pd.to_datetime(dates)
    alive = alive[alive["_d"].isin(cal)]
    wide = alive.pivot(index="_d", columns="code", values="close").reindex(dates)
    ret = wide.pct_change()
    lvl = (1.0 + ret.mean(axis=1, skipna=True).fillna(0.0)).cumprod()
    return lvl


def main():
    t0 = time.time()
    print("=" * 60)
    print("市场级 Regime IC 诊断 (REGIME_TRAINING Plan Phase 1)")
    print("=" * 60)

    # ── 1. 加载数据 ──
    print("\n[1/4] 加载面板 + 构建因子...")
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
    print(f"  因子: {len(ALL)} (技术{len(ALL_FACTOR_NAMES)} + 基本面{len(FUND_NAMES)})")

    # ── 2. 逐日 rank IC ──
    print("\n[2/4] 计算逐日 rank IC...")
    fac_ic = {
        f: daily_rank_ic(pd.DataFrame(zarr[f], index=dates, columns=codes), fwd)
        for f in ALL
    }
    fac_ic = {f: s.dropna() for f, s in fac_ic.items()}
    ic_df = pd.DataFrame(fac_ic)
    print(f"  IC 矩阵: {ic_df.shape}")

    # ── 3. Regime 分割 ──
    print("\n[3/4] 用活股等权指数 MA120 分割牛/熊 regime...")
    mkt_idx = _alive_mkt_level(dates)
    ma = mkt_idx.rolling(BEAR_MA).mean()
    ratio = mkt_idx / ma - 1
    bear_regime = (ratio < BEAR_THR).fillna(False)
    regime_label = bear_regime.replace({True: "bear", False: "bull"}).reindex(ic_df.index).ffill()
    # 统计
    n_bull = (regime_label == "bull").sum()
    n_bear = (regime_label == "bear").sum()
    print(f"  牛市日: {n_bull}  | 熊市日: {n_bear}  | 比率: {n_bear/(n_bull+n_bear):.1%}")

    # ── 4. 分段 IC 统计 ──
    print("\n[4/4] 分段计算因子 IC 统计...")
    rows = []
    for f in ALL:
        ic = ic_df[f].dropna()
        ic_bull = ic[regime_label.reindex(ic.index) == "bull"]
        ic_bear = ic[regime_label.reindex(ic.index) == "bear"]
        if len(ic_bull) < 10 or len(ic_bear) < 10:
            rows.append({
                "factor": f, "n_bull": len(ic_bull), "n_bear": len(ic_bear),
                "bull_ic_mean": None, "bear_ic_mean": None,
                "bull_icir": None, "bear_icir": None,
                "sign_flip": None, "ic_diff": None, "ic_diff_sigma": None,
            })
            continue

        mu_bull = float(ic_bull.mean())
        mu_bear = float(ic_bear.mean())
        sd_bull = float(ic_bull.std())
        sd_bear = float(ic_bear.std())

        # 符号反转: 牛市 IC 为正、熊市为负 (或反之)
        sign_flip = (mu_bull > 0 and mu_bear < 0) or (mu_bull < 0 and mu_bear > 0)

        # IC 差异显著性: |ΔIC| / pooled_std
        pooled_sd = np.sqrt((sd_bull ** 2 + sd_bear ** 2) / 2 + 1e-9)
        ic_diff = mu_bull - mu_bear
        ic_diff_sigma = ic_diff / pooled_sd if pooled_sd > 1e-12 else 0.0

        rows.append({
            "factor": f,
            "n_bull": len(ic_bull), "n_bear": len(ic_bear),
            "bull_ic_mean": mu_bull,
            "bear_ic_mean": mu_bear,
            "bull_icir": mu_bull / (sd_bull + 1e-9) * np.sqrt(252),
            "bear_icir": mu_bear / (sd_bear + 1e-9) * np.sqrt(252),
            "sign_flip": sign_flip,
            "ic_diff": ic_diff,
            "ic_diff_sigma": ic_diff_sigma,
        })

    diag_df = pd.DataFrame(rows).sort_values("ic_diff_sigma", key=abs, ascending=False)
    out_csv = OUT_DIR / "regime_ic_diagnosis.csv"
    diag_df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\n  诊断表: {out_csv}")

    # ── 汇总 ──
    n_flip = diag_df["sign_flip"].sum()
    n_total = len(diag_df)
    flip_pct = n_flip / n_total
    n_strong = (diag_df["ic_diff_sigma"].abs() > 2.0).sum()
    print(f"\n{'=' * 60}")
    print(f"诊断结论")
    print(f"  {'符号反转因子数':<20} {n_flip}/{n_total} = {flip_pct:.0%}")
    print(f"  {'强差异因子数(|ΔIC|>2σ)':<20} {n_strong}/{n_total}")
    print(f"  建议:", end=" ")
    if flip_pct >= 0.30:
        print("Phase 2 (regime 条件 WFA) 值得推进 — ≥30% 因子符号反转")
    elif n_strong >= 0.50:
        print("Phase 2 可能有益 — ≥50% 因子有强量级差异 (虽无符号反转)")
    else:
        print("Regime 分训帮助有限 (符号翻转 <30% 且量级差异 <50%)")
    print(f"耗时: {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
