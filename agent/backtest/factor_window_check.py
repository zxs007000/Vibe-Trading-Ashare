"""factor_window_check.py — 因子家族在 2020-2023(含新冠熔断+2022熊)的表现检验.

用户质疑: 因子在 2024-09 之后的纯牛市里回测, 没参考意义.
本脚本把窗口钉死在 2020-01-01~2023-12-31(跨新冠熔断 + 2022 全年熊), 看因子家族成色:
  (1) 每个因子在该窗口的 rank-IC / ICIR(家族健康度, 描述性);
  (2) 策略: 用 2020 之前数据冻结因子集(IS锁定), 在 2020-2023 严格 OOS 实测夏普/回撤/累计.

用法: python backtest/factor_window_check.py
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import oos_validation as OOS
import oos_validation_corrected as M
from factor_zoo_daily import build_factors, neutralize_factors, ALL_FACTOR_NAMES, daily_rank_ic
from factor_zoo_ortho import ORTHO_NAMES, ORTHO_FAMILY

SF_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel_survivorfree.parquet")
CSRC_MAP = Path("/workspace/stock_worm/data/csrc_industry_map.parquet")
FUND_PARQUET = Path("/workspace/stock_worm/data/fundamentals/fund_factors_daily.parquet")
FUND_NAMES = ["ROE", "rev_yoy", "profit_yoy"]

TOP_K, HOLD, COST, TRAIL = OOS.TOP_K, OOS.HOLD, OOS.COST, OOS.TRAIL
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2023-12-31")
FREEZE_CUT = pd.Timestamp("2020-01-01")   # 冻结用 2020 之前数据

FAMILY = {
    # 反转
    "rev_5": "反转", "rev_20": "反转", "rev_60": "反转", "rev_intraday": "反转",
    "mom_5": "反转", "mom_20": "反转",
    # 动量
    "mom_60": "动量", "mom_120": "动量", "mom_250": "动量", "mom_12_1": "动量",
    "ma_dev_20": "动量", "ma_dev_60": "动量",
    # 波动/风险
    "vol_20": "波动", "vol_60": "波动", "ret_skew_60": "波动", "ivol_60": "波动",
    "downside_vol_60": "波动", "boll_w": "波动", "drawup_60": "波动",
    # 流动性/微结构
    "amihud_20": "流动性", "dolvol_trend": "流动性", "macd_hist": "流动性", "rsi_14": "流动性",
    "adx_14": "流动性", "high_52w": "流动性", "overnight_gap": "流动性", "intraday_range": "流动性",
    "vol_ratio": "流动性", "vol_price_corr": "流动性", "amount_strength": "流动性",
    # 质量
    "ROE": "质量", "rev_yoy": "质量", "profit_yoy": "质量",
    # 正交
    "beta_60": "正交", "lowvol_60": "正交", "lowivol_60": "正交", "liq_stress_20": "正交", "distress_60": "正交",
}


def backtest_with_holdings(signal_w, fwd_w, top_k=TOP_K, hold=HOLD, cost=COST):
    dates = signal_w.index
    port, rdates, held_list = [], [], []
    for i in range(len(dates)):
        if i % hold != 0:
            continue
        d = dates[i]; s = signal_w.loc[d]; r = fwd_w.loc[d]
        shared = s.dropna().index.intersection(r.dropna().index)
        if len(shared) < 5:
            continue
        s, r = s[shared], r[shared]
        k = max(3, int(len(s) * top_k))
        held = set(s.nlargest(k).index)
        pr = r[list(held)].mean() - top_k * 2 * cost
        port.append(pr); rdates.append(d); held_list.append(held)
    return pd.Series(port, index=rdates), held_list


def _stat(series, label=""):
    s = series.dropna(); n = len(s); yrs = max(n / 252, 1e-9)
    eq = (1 + s).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = s.mean() / (s.std() + 1e-12) * np.sqrt(252)
    return dict(name=label, sharpe=sharpe, cagr=cagr, maxdd=dd, cum=eq.iloc[-1] - 1)


def main():
    t0 = time.time()
    w = M.load_wide_sf()
    n_codes = w["close"].shape[1]
    fwd = w["close"].pct_change(HOLD).shift(-HOLD).clip(-0.5, 0.5)
    dates, codes = fwd.index, fwd.columns
    mp = pd.read_parquet(CSRC_MAP); ind_map = dict(zip(mp["code"], mp["csrc_industry"]))

    fac = build_factors(w); ortho = build_ortho_factors_local(w); del w
    fac = neutralize_factors(fac, ind_map)
    fund = pd.read_pickle(FUND_PARQUET)
    for f in FUND_NAMES:
        fac[f] = fund[f]
    for f in ORTHO_NAMES:
        fac[f] = ortho[f]
    del ortho
    ALL = ALL_FACTOR_NAMES + FUND_NAMES + ORTHO_NAMES
    zarr = M.build_zarr(fac, ALL, dates, codes); del fac
    for f in FUND_NAMES:
        zarr[f] = np.nan_to_num(zarr[f], nan=0.0)
    print(f"面板 {n_codes}只 × {dates[0].date()}~{dates[-1].date()} | 测试窗 {TEST_START.date()}~{TEST_END.date()}")

    fac_ic = {f: daily_rank_ic(pd.DataFrame(zarr[f], index=dates, columns=codes), fwd) for f in ALL}
    fac_ic = {f: s.reindex(dates) for f, s in fac_ic.items()}
    ic_mean = {f: fac_ic[f].rolling(TRAIL).mean() for f in ALL}
    ic_std = {f: fac_ic[f].rolling(TRAIL).std() for f in ALL}

    # 冻结集: 2020 之前(IS) IC>0 且 ICIR>0
    is_mask = (dates >= dates[0]) & (dates < FREEZE_CUT)
    is_ic = {f: fac_ic[f][is_mask].mean() for f in ALL}
    is_icir = {f: (is_ic[f] / (fac_ic[f][is_mask].std() + 1e-9)) * np.sqrt(252) for f in ALL}
    frozen_set = [f for f in ALL if is_ic[f] > 0 and is_icir[f] > 0]

    # 测试窗 2020-2023
    tst = (dates >= TEST_START) & (dates <= TEST_END)
    print(f"\n=== (1) 因子家族在 2020-2023 的 IC/ICIR (窗口内, 含新冠熔断+2022熊) ===")
    rows = []
    for f in ALL:
        ic = fac_ic[f][tst].mean()
        icir = (ic / (fac_ic[f][tst].std() + 1e-9)) * np.sqrt(252)
        rows.append((f, FAMILY.get(f, "技术"), ic, icir, f in frozen_set))
    # 家族汇总
    fams = {}
    for f, fa, ic, icir, inf in rows:
        d = fams.setdefault(fa, [])
        d.append((f, ic, icir, inf))
    print(f"{'家族':<6} {'n':>2} {'均IC':>8} {'均ICIR':>8} {'活':>2} {'死':>2}")
    for fa in ["反转", "动量", "波动", "流动性", "质量", "正交"]:
        ds = fams.get(fa, [])
        if not ds:
            continue
        nic = np.mean([x[1] for x in ds]); nicr = np.mean([x[2] for x in ds])
        nalive = sum(1 for x in ds if x[3]); ndead = len(ds) - nalive
        print(f"{fa:<6} {len(ds):>2} {nic:>+.4f} {nicr:>+.2f} {nalive:>2} {ndead:>2}")
    # 逐因子(按家族+ICIR降序)
    print(f"\n{'因子':<16}{'家族':<6}{'IC':>8}{'ICIR':>8}  冻结")
    rows.sort(key=lambda r: (["反转", "动量", "波动", "流动性", "质量", "正交"].index(r[1]), -r[3]))
    for f, fa, ic, icir, inf in rows:
        print(f"{f:<16}{fa:<6}{ic:>+.4f} {icir:>+.2f}  {'✓' if inf else ''}")

    # (2) 策略: 冻结(pre-2020) → 2020-2023 实测
    sigF = OOS.build_signal(zarr, ic_mean, ic_std, ALL, dates, codes,
                            allowed=frozen_set, gate=False, weight_src="is", is_icir=is_icir)
    del zarr
    portS, _ = backtest_with_holdings(sigF, fwd)
    portS_t = portS[(portS.index >= TEST_START) & (portS.index <= TEST_END)]
    sS = _stat(portS_t, "Frozen策略(2020前冻结)")
    # 等权基准
    rebal_pos = list(range(0, len(dates), HOLD))
    bench = fwd.iloc[rebal_pos].mean(axis=1).dropna()
    bench_t = bench[(bench.index >= TEST_START) & (bench.index <= TEST_END)]
    sB = _stat(bench_t, "等权基准")
    print(f"\n=== (2) 策略在 2020-2023 严格 OOS (因子冻结于2020前) ===")
    print(f"  冻结集 {len(frozen_set)}/{len(ALL)}")
    print(f"  Frozen策略: Sharpe={sS['sharpe']:+.3f} 年化={sS['cagr']:+.2%} 最大回撤={sS['maxdd']:+.2%} 累计={sS['cum']:+.1%}")
    print(f"  等权基准  : Sharpe={sB['sharpe']:+.3f} 年化={sB['cagr']:+.2%} 最大回撤={sB['maxdd']:+.2%} 累计={sB['cum']:+.1%}")
    print(f"  超额Sharpe={sS['sharpe']-sB['sharpe']:+.3f}  超额累计={(1+sS['cum'])/(1+sB['cum'])-1:+.1%}")
    print(f"\n*耗时 {time.time()-t0:.1f}s*")


def build_ortho_factors_local(w):
    from factor_zoo_ortho import build_ortho_factors
    return build_ortho_factors(w)


if __name__ == "__main__":
    main()
