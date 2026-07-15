"""diag_factor_count.py — 只诊断每 fold train 窗锁定的因子数(不跑完整回测).

回答: '你拿了多少因子去训练的?'
  - 标准 Frozen (A/B/D 用): 全量 IC>0 且 ICIR>0 的因子.
  - 市场级 regime 选股 (C/E 用): 仅 train 窗匹配 regime 下 IC>0 的因子(可能更少).
"""
import sys, time
import numpy as np
sys.path.insert(0, str(__file__ and __import__("pathlib").Path(__file__).parent / ".." / "agent" / "backtest"))
from regime_wfa import (_load_data, _market_regime_label, _build_regime_factor_set,
                        TRAIN_DAYS, TEST_DAYS, PURGE, BEAR_MA, BEAR_THR)

t0 = time.time()
inp = _load_data()
fac_ic, ALL = inp["fac_ic"], inp["ALL"]
dates, codes = inp["dates"], inp["codes"]
mkt_level = inp["mkt_level"]
print(f"[数据] {inp['n_codes']}只 × {dates[0].date()}~{dates[-1].date()} | 因子池={len(ALL)}")

regime_label = _market_regime_label(mkt_level, dates)

n = len(dates)
folds = []
i = TRAIN_DAYS
while i + PURGE + TEST_DAYS <= n:
    folds.append((i - TRAIN_DAYS, i, i + PURGE, i + PURGE + TEST_DAYS))
    i += TEST_DAYS
print(f"[fold] 共 {len(folds)} 个 WFA fold\n")

frozen_counts, regime_counts = [], []
print(f"{'fold':>4} {'train窗口':<24} {'train末regime':<8} {'Frozen锁定':>10} {'Regime锁定':>10}")
for k, (ts, te, ve, vb) in enumerate(folds):
    # 标准 Frozen
    frozen = []
    for f in ALL:
        ic = fac_ic[f].iloc[ts:te]
        m, s = ic.mean(), ic.std()
        if (m == m) and (s == s) and s > 1e-9:
            icir = m / s * 252 ** 0.5
            if m > 0 and icir > 0:
                frozen.append(f)
    # regime 选股 (用 train 末状态预测 test regime)
    last_regime = regime_label.iloc[te - 1]
    reg_sel = _build_regime_factor_set(fac_ic, (ts, te), regime_label, last_regime)
    frozen_counts.append(len(frozen)); regime_counts.append(len(reg_sel))
    print(f"{k:>4} {str(dates[ts].date())}~{str(dates[te].date()):<17} "
          f"{last_regime:<8} {len(frozen):>10} {len(reg_sel):>10}")

import numpy as np
def stat(x):
    a = np.array(x, float)
    return f"min={a.min():.0f} median={np.median(a):.0f} max={a.max():.0f} mean={a.mean():.1f}"
print("\n[Frozen 每 fold 锁定因子数] " + stat(frozen_counts))
print("[Regime 每 fold 锁定因子数] " + stat(regime_counts))
print(f"[Regime 相对 Frozen 缩减] 平均少 {np.mean(frozen_counts)-np.mean(regime_counts):.1f} 个 "
      f"({(1-np.mean(regime_counts)/np.mean(frozen_counts))*100:.0f}%)")
print(f"\n耗时 {time.time()-t0:.1f}s")
