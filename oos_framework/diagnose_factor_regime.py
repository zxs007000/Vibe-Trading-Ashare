"""
diagnose_factor_regime.py — 因子级 Regime 诊断 (REGIME_FACTOR Plan Phase 2)

目的:
  基于因子绩效序列 (来自 build_factor_pnl.py 产出的 factor_pnl.parquet),
  为每个因子独立判定其历史上每日的牛/熊/震荡状态。

方法 (简化版 SJM, 每因子独立):
  对每个因子计算日频滚动特征:
    - pnl_cum_63   (63 日累计收益)
    - pnl_sharpe_63 (63 日滚动 Sharpe = mean/std)
  判定规则:
    - bear  = (cum_63 < 0) & (sharpe_63 < 0)
    - bull  = (cum_63 > 0) & (sharpe_63 > 0.2)
    - neutral = 其余

产出:
  oos_framework/screen_results/factor_regime.csv
    shape = (dates × n_factors), 值: 1=bull, 0=neutral, -1=bear

用法:
  python oos_framework/diagnose_factor_regime.py
"""

import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
OUT_DIR = HERE / "screen_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PNL_PATH = OUT_DIR / "factor_pnl.parquet"
REGIME_CSV = OUT_DIR / "factor_regime.csv"

ROLL_SHORT = 21
ROLL_LONG = 63
SHARPE_THR_BULL = 0.20


def diagnose_factor_regime(pnl: pd.DataFrame) -> pd.DataFrame:
    """从因子 PnL 矩阵计算每日各因子的 bull/bear/neutral 标签.

    Args:
        pnl: DataFrame(index=dates, columns=factor_names), 因子日频多空收益.

    Returns:
        DataFrame(index=dates, columns=factor_names), 值 ∈ {1=bull, 0=neutral, -1=bear}.
    """
    labels = pd.DataFrame(0, index=pnl.index, columns=pnl.columns, dtype=int)
    for f in pnl.columns:
        s = pnl[f].dropna()
        if len(s) < ROLL_LONG + 5:
            continue
        cum = s.rolling(ROLL_LONG).sum()
        sharpe = s.rolling(ROLL_LONG).mean() / (s.rolling(ROLL_LONG).std() + 1e-9)
        is_bull = (cum > 0) & (sharpe > SHARPE_THR_BULL)
        is_bear = (cum < 0) & (sharpe < 0)
        labels.loc[s.index, f] = np.where(is_bear, -1, np.where(is_bull, 1, 0))
    return labels


def main():
    t0 = time.time()
    print("=" * 60)
    print("因子级 Regime 诊断 (REGIME_FACTOR Plan Phase 2)")
    print("=" * 60)

    if not PNL_PATH.exists():
        print(f"\n[ERROR] 因子 PnL 文件不存在: {PNL_PATH}")
        print("请先运行: python oos_framework/build_factor_pnl.py")
        sys.exit(1)

    print(f"\n[1/2] 加载因子 PnL: {PNL_PATH}")
    pnl = pd.read_parquet(PNL_PATH)
    print(f"  维度: {pnl.shape}")
    print(f"  日期: {pnl.index[0].date()} ~ {pnl.index[-1].date()}")
    print(f"  因子数: {len(pnl.columns)}")

    print("\n[2/2] 诊断因子级 regime 标签...")
    labels = diagnose_factor_regime(pnl)
    labels.to_csv(REGIME_CSV)
    print(f"  输出: {REGIME_CSV}")

    # ── 汇总统计 ──
    stats = []
    for f in labels.columns:
        col = labels[f]
        n_bull = int((col == 1).sum())
        n_bear = int((col == -1).sum())
        n_neu = int((col == 0).sum())
        total = n_bull + n_bear + n_neu
        stats.append({
            "factor": f, "total_days": total,
            "bull_pct": n_bull / total, "bear_pct": n_bear / total,
            "neu_pct": n_neu / total,
        })
    stats_df = pd.DataFrame(stats).sort_values("bear_pct", ascending=False)
    print(f"\n  各因子 Bull/Bear/Neutral 占比:")
    print(stats_df.to_string(index=False))

    # ── 总体分布 ──
    n_all = len(labels.columns) * len(labels)
    n_bull_all = int((labels.values == 1).sum())
    n_bear_all = int((labels.values == -1).sum())
    n_neu_all = n_all - n_bull_all - n_bear_all
    print(f"\n  {'总体':>20}: bull={n_bull_all/n_all:.0%} bear={n_bear_all/n_all:.0%} neutral={n_neu_all/n_all:.0%}")

    print(f"\n耗时: {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
