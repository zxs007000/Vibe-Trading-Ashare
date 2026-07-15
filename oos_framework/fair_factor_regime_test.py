"""fair_factor_regime_test.py — 公平版因子级 regime 测试(全市场).

问题: regime_wfa.py 用的 factor_regime.csv 标签来自**小宇宙(1803只)因子的多空 pnl**,
套到全市场 5515 只时, 标签反映的是小宇宙因子行为 -> D 在全市场"失效"很可能是标签-宇宙不匹配,
而非因子级 regime 本身无效.

本脚本从**全市场 fac_ic**(逐因子滚动 IC, 因果, 宇宙一致)重新生成因子 regime 标签:
  bull=滚动IC均值>0, bear=滚动IC均值<0, neutral=0,
再重跑 5 变体, 与"小宇宙标签"版头对头, 看 D/E 的真实边缘.

用法: python oos_framework/fair_factor_regime_test.py
"""
import sys, time
sys.path.insert(0, "oos_framework")
sys.path.insert(0, "agent/backtest")
import numpy as np, pandas as pd
import regime_wfa as R

t0 = time.time()
inp = R.load_engine_inputs_cached()
zarr, fac_ic, ALL = inp["zarr"], inp["fac_ic"], inp["ALL"]
fwd, dates, codes, mkt_level = inp["fwd"], inp["dates"], inp["codes"], inp["mkt_level"]

# ── 公平标签: 从全市场 fac_ic 滚动 IC 生成(因果, 宇宙一致) ──
ROLL = 63
labels = pd.DataFrame(0, index=dates, columns=ALL, dtype=int)
for f in ALL:
    ic = fac_ic[f].fillna(0.0)
    rm = ic.rolling(ROLL).mean()
    labels[f] = np.where(rm > 0, 1, np.where(rm < 0, -1, 0))
print(f"公平因子regime标签: {labels.shape}  (bull={int((labels==1).values.sum())} "
      f"bear={int((labels==-1).values.sum())} neu={int((labels==0).values.sum())})", flush=True)

base = dict(zarr=zarr, fac_ic=fac_ic, factor_names=ALL, fwd=fwd,
            dates=dates, codes=codes, mkt_level=mkt_level,
            factor_regime_labels=labels)
variants = {
    "A: 无闸": dict(gate=False, use_market_regime=False, use_factor_regime=False),
    "B: 两档位闸门": dict(gate=True, bear_mode="ma120", use_market_regime=False, use_factor_regime=False),
    "C: 两档位+市场regime": dict(gate=True, bear_mode="ma120", use_market_regime=True, use_factor_regime=False),
    "D: 两档位+因子regime(公平标签)": dict(gate=True, bear_mode="ma120", use_market_regime=False, use_factor_regime=True),
    "E: 两档位+双regime(公平标签)": dict(gate=True, bear_mode="ma120", use_market_regime=True, use_factor_regime=True),
}
print(f"\n全市场 {inp['n_codes']}只 × {dates[0].date()}~{dates[-1].date()} 公平因子regime WFA:")
results = {}
for name, kw in variants.items():
    r = R.rolling_wfa_dual_regime(**base, **kw)
    results[name] = r
    a = r["agg"]
    print(f"  [{name}] Sharpe={a['sharpe']:+.3f} 回撤={a['maxdd']:+.2%} "
          f"通过率={r['pass_rate']:.0%} 否决={r['n_veto']} 决策={r['decision']}", flush=True)

print(f"\n{'='*70}")
print(f"{'变体':<32} {'Sharpe':>8} {'超额':>8} {'年化':>8} {'回撤':>8} {'通过率':>6}")
print(f"{'-'*70}")
for name, r in results.items():
    a = r["agg"]
    print(f"{name:<32} {a['sharpe']:>+8.3f} {a['ex_sharpe']:>+8.3f} "
          f"{a['ann']:>+8.2%} {a['maxdd']:>+8.2%} {r['pass_rate']:>6.0%}")
print(f"\n总耗时: {time.time()-t0:.1f}s")
print("="*70)
