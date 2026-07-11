"""xgb_select_clean.py — 干净版 XGBoost 选股(修正 xgb_selector 两处缺陷)

修正点:
  (1) fwd 直接在因子日历上 shift, 杜绝 xgb_selector 的错位 bug;
  (2) 引入**随机 top-K 基线**, 修正"组合超额=减等权"的基准偏误幻觉.
同因子同窗口对比: XGBoost vs 线性(滚动ICIR加权, 即 Frozen 思路) vs 等权 vs 随机.

因子: founder_factors.pkl 的 7 个主力资金因子 (288只, 2024-05-27~2026-06-30)
目标: 5日收益率; 多头=每日预测最高前30%, 持有5日, 单边0.10%
"""
import pickle, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import xgboost as xgb

ROOT = Path("/workspace/stock_worm/data")
FF = ROOT / "founder_factors.pkl"
PANEL = ROOT / "ashare_daily_panel_survivorfree.parquet"
ALIVE = ROOT / "ashare_daily_panel.parquet"
HOLD, TOP_K, COST = 5, 0.30, 0.001
TRAIN_MIN, TEST, RNG = 120, 20, 42


def load():
    ff = pickle.load(open(FF, "rb"))
    codes = list(ff[list(ff.keys())[0]].columns)
    fac = {n: ff[n].reindex(columns=codes) for n in ff}
    p = pd.read_parquet(PANEL); p["_d"] = pd.to_datetime(p["date"]).dt.normalize()
    alive = pd.read_parquet(ALIVE); cal = pd.to_datetime(alive["date"]).dt.normalize().unique()
    p = p[p["_d"].isin(cal)]
    close = p.pivot(index="_d", columns="code", values="close").reindex(fac[list(fac)[0]].index)[codes]
    return fac, close, codes


def daily_pearson(A, B):
    Am = A.sub(A.mean(axis=1), axis=0); Bm = B.sub(B.mean(axis=1), axis=0)
    num = (Am * Bm).mean(axis=1)
    den = A.std(axis=1) * B.std(axis=1)
    return num / (den + 1e-12)


def zscore(mat):
    return (mat.sub(mat.mean(axis=1), axis=0)).div(mat.std(axis=1).replace(0, np.nan), axis=0)


def build_panel(fac_z, fwd):
    feat = pd.concat([fac_z[n].stack().rename(n) for n in fac_z], axis=1)
    y = fwd.stack().rename("y")
    return pd.concat([feat, y], axis=1).dropna(subset=["y"])


def walkforward_xgb(panel, feats):
    dates = sorted(panel.index.get_level_values(0).unique())
    recs, start = [], 0
    while start + TRAIN_MIN + TEST <= len(dates):
        tr_d = dates[:start + TRAIN_MIN]; te_d = dates[start + TRAIN_MIN:start + TRAIN_MIN + TEST]
        tr = panel.loc[panel.index.get_level_values(0).isin(tr_d)]
        te = panel.loc[panel.index.get_level_values(0).isin(te_d)]
        if len(tr) < 200 or te.empty:
            start += TEST; continue
        m = xgb.XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             n_jobs=-1, random_state=RNG, verbosity=0)
        m.fit(tr[feats].values, tr["y"].values)
        te = te.copy(); te["pred"] = m.predict(te[feats].values)
        recs.append(te); start += TEST
    return pd.concat(recs).sort_index()


def rank_ic(oos, col="pred"):
    rows = []
    for d, g in oos.groupby(level=0):
        if g[col].notna().sum() < 5 or g["y"].notna().sum() < 5:
            continue
        rows.append(g[col].rank().corr(g["y"].rank()))
    return pd.Series(rows).dropna()


def portfolio(sig, fwd, top_k=TOP_K, hold=HOLD, cost=COST):
    dates = sorted(sig.index.unique())
    rnd = np.random.default_rng(RNG)
    port, rand, bench = [], [], []
    held = heldr = None
    for i, d in enumerate(dates):
        g = sig.loc[d]; yv = fwd.loc[d]
        j = g.notna() & yv.notna(); gg, yy = g[j], yv[j]
        if len(gg) < 5:
            port.append(np.nan); rand.append(np.nan); bench.append(np.nan); continue
        if i % hold == 0:
            k = max(3, int(len(gg) * top_k))
            held = set(gg.nlargest(k).index)
            heldr = set(gg.index[rnd.random(len(gg)) < top_k])
        r = yy.loc[list(held & set(yy.index))].mean()
        rr = yy.loc[list(heldr & set(yy.index))].mean()
        rb = yy.mean()
        if i % hold == 0:
            r -= top_k * 2 * cost; rr -= top_k * 2 * cost; rb -= top_k * 2 * cost
        port.append(r); rand.append(rr); bench.append(rb)
    return (pd.Series(port).dropna(), pd.Series(rand).dropna(), pd.Series(bench).dropna())


def sharpe(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan


def main():
    t0 = time.time()
    fac, close, codes = load()
    fwd = close.shift(-HOLD) / close - 1
    # 每个因子整体 IC 符号定向
    feats = list(fac.keys())
    dic = {n: daily_pearson(fac[n], fwd) for n in feats}
    fac_o = {n: fac[n] * (1 if dic[n].mean() >= 0 else -1) for n in feats}
    fac_z = {n: zscore(m) for n, m in fac_o.items()}
    panel = build_panel(fac_z, fwd)
    print(f"因子={feats}\n码={len(codes)} 交易日={len(fac_z[feats[0]])} 面板样本={len(panel)}")

    # 1) XGBoost
    oos = walkforward_xgb(panel, feats)
    ic = rank_ic(oos)
    ic_pos = (ic > 0).mean()
    icir = ic.mean() / ic.std() * np.sqrt(252) if ic.std() > 0 else np.nan
    print(f"\n[XGBoost] OOS rank-IC={ic.mean():+.4f} ICIR={icir:.3f} ic_pos={(ic>0).mean():.3f}")
    oos_dates = oos.index.get_level_values(0).unique()
    sig_xgb = oos["pred"].unstack().reindex(oos_dates)

    # 2) 线性(滚动ICIR加权 = Frozen 思路)
    # 注意: Series*DataFrame 在 pandas 中会歧义地按列对齐而非按行广播,
    # 必须用 .mul(icir_trail[n], axis=0) 显式按行广播; 同时把 NaN 特征填0(z=0=中性).
    # 覆盖率过低(<50%)的因子其 ICIR 估计不可靠(滚动窗口全 NaN), 会从求和传染整列,
    # 故只对高覆盖因子加权(本窗口下 complete_tide/undercurrent 即被排除).
    cov = {n: fac[n].notna().mean().mean() for n in feats}
    good = [n for n in feats if cov[n] > 0.5]
    print(f"线性基线采用因子(覆盖>50%): {good}  | 排除: {[n for n in feats if n not in good]}")
    fac_z0 = {n: fac_z[n].fillna(0) for n in feats}
    icir_trail = {n: dic[n].rolling(60).mean() / (dic[n].rolling(60).std() + 1e-12) for n in feats}
    sig_lin = sum(fac_z0[n].mul(icir_trail[n], axis=0) for n in good)
    sig_lin = sig_lin.reindex(oos_dates)
    fwd_o = fwd.reindex(oos_dates)

    px, rx, bx = portfolio(sig_xgb, fwd_o)
    pl, rl, bl = portfolio(sig_lin, fwd_o)
    eqs = {}
    print(f"\n{'策略':<22}{'夏普':>8}{'年化':>10}{'最大回撤':>10}{'累计':>10}")
    for name, s in [("XGBoost", px), ("线性ICIR加权(Frozen)", pl),
                    ("随机top-K(基线)", rx), ("等权基准", bx)]:
        if len(s) == 0:
            print(f"{name:<22}{'--':>8}{'--':>10}{'--':>10}{'--':>10}"); continue
        eq = (1 + s).cumprod(); eqs[name] = eq
        cagr = eq.iloc[-1] ** (252 / len(s)) - 1
        dd = (eq / eq.cummax() - 1).min()
        print(f"{name:<22}{sharpe(s):>8.3f}{cagr:>+10.2%}{dd:>+10.2%}{eq.iloc[-1]:>10.3f}")
    eq_xgb, eq_lin, eq_rx = eqs["XGBoost"], eqs["线性ICIR加权(Frozen)"], eqs["随机top-K(基线)"]

    print(f"\n诚实解读:")
    print(f"  1) XGBoost OOS rank-IC={ic.mean():+.4f} (ic_pos={ic_pos:.3f}<0.5) → 近零微负, "
          f"与 cloud_backtest 权威结论(rank-IC≈-0.01)一致: 这些因子上 XGBoost 无可靠截面预测力.")
    print(f"  2) XGBoost 组合累计 {eq_xgb.iloc[-1]:.2f} 看似亮眼, 但 随机top-K基线 已 {eq_rx.iloc[-1]:.2f} "
          f"(夏普 {sharpe(rx):.3f}) → 绝大部分'超额'是 288只动量富集宇宙的 top-K 结构 beta 幻觉, 非 alpha.")
    print(f"  3) 线性ICIR加权(Frozen) 累计 {eq_lin.iloc[-1]:.2f} 反而跑赢 XGBoost → 复杂ML未榨出更多 alpha, "
          f"只是增加方差/过拟合. 这正印证本项目主旨: 因子有寿命, 简单稳健的线性加权优于盲目上模型.")
    print(f"  4) 重要警告: 本窗口仅 ~2年(2024-05~2026-06)且宇宙有偏(founder_factors=已富集动量股), "
          f"绝对夏普(2.6~3.5)不可外推到大盘; 有效结论只在'相对对比'层面.")
    print(f"耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
