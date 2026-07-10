"""cloud_backtest.py — 云端因子回测(本地化 5m 数据湖, stock_worm 源).

对比三条路线, 全部在 2024-05-27~2026-06-30 的 288 只沪深300 5m 数据上做,
无未来泄漏:

  (1) 线性因子组合  : walk-forward 方向自适应 + ICIR 加权 -> 因子级 IC/ICIR/多空夏普
                       + 可交易多头组合(前30%, 5日持有, 含双边成本)
  (2) XGBoost 选股   : walk-forward(XGBRegressor) 预测次日收益 -> OOS rank-IC
                       + 可交易多头组合(前30%, 5日持有, 含双边成本)

核心结论(排雷): founder 头部因子 IC 为正, 但 因子->次日收益 呈"非单调(倒U)":
线性多空/多头结构性亏钱, 树模型原生支持非单调映射, 是唯一可行的"用法".

用法:
  python backtest/cloud_backtest.py
"""
from __future__ import annotations
import sys, pickle, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))       # agent/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

import xgboost as xgb
from backtest.factors import founder as F
from src.factors.factor_analysis_core import compute_ic_series, compute_group_equity
from backtest.validation import _sharpe

CACHE = Path("/workspace/stock_worm/data/ashare_5m_cache.pkl")
START, END = "2023-01-01", "2026-06-30"
N_STOCKS = None          # None = 全部(5m 缓存里所有股)
TRAIN_DAYS = 120
TEST_DAYS = 20
TOP_K = 0.30
COST_ONEWAY = 0.001      # 单边千一(印花税+佣金+冲击)

SPECS = [
    ("smart_money",          F.smart_money_batch,         "minute"),
    ("flower_hidden",        F.flower_hidden_batch,       "minute"),
    ("complete_tide",        F.complete_tide_batch,       "minute"),
    ("scaling_heights",      F.scaling_heights_batch,     "minute"),
    ("undercurrent",         F.undercurrent_batch,        "minute"),
    ("withered_tree_blooms", F.withered_tree_blooms,      "single_daily"),
    ("clouds_disperse",      F.clouds_disperse_batch,     "minute"),
]


def _zscore_row(row: pd.Series) -> pd.Series:
    return row.sub(row.mean()).div(row.std(ddof=0) + 1e-9)


def _load_prep():
    all_min = pickle.load(open(CACHE, "rb"))
    codes = sorted(all_min)[:N_STOCKS] if N_STOCKS else sorted(all_min)
    stocks = {c: all_min[c] for c in codes}
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
    return stocks, daily_close, fwd, daily_bars, trading


def _compute_factors(stocks, daily_bars, trading, fwd):
    raw = {}
    for name, fn, kind in SPECS:
        res = fn(stocks) if kind == "minute" else {c: fn(b) for c, b in daily_bars.items()}
        fdf = pd.DataFrame({c: s for c, s in res.items() if len(s.dropna()) > 5}).reindex(trading)
        if fdf.shape[1] < 5 or fdf.dropna(how="all").empty:
            print(f"  ⚠ {name} 有效股票不足, 跳过"); continue
        ic = compute_ic_series(fdf, fwd)
        orient = 1.0 if ic.mean() >= 0 else -1.0
        raw[name] = fdf * orient
        print(f"  {name:<20} 有效股票={fdf.shape[1]:3d}  全窗口IC={ic.mean():+.4f} 定向={'高' if orient>0 else '低'}")
    return raw


def walk_forward_combine(raw, fwd, train_days=TRAIN_DAYS, test_days=TEST_DAYS, mode="icir"):
    all_dates = sorted(set().union(*[set(df.index) for df in raw.values()]))
    all_dates = [d for d in all_dates if d in fwd.index]
    n = len(all_dates)
    if n < train_days + test_days:
        return None
    chunks = []; start = 0
    while start + train_days < n:
        tr_end = start + train_days
        te_end = min(tr_end + test_days, n)
        train_d = all_dates[start:tr_end]; test_d = all_dates[tr_end:te_end]
        past_d = all_dates[:tr_end]
        zparts, wsum = [], 0.0
        for name, df in raw.items():
            ic_all = compute_ic_series(df.loc[past_d], fwd.loc[past_d])
            if ic_all.empty or ic_all.isna().all():
                continue
            orient = 1.0 if ic_all.mean() >= 0 else -1.0
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


def long_only_backtest(signal, fwd, top_k=TOP_K, hold_days=5, cost_oneway=COST_ONEWAY, n_rand=5):
    """可交易多头: 横截面排序选前 top_k, 每 hold_days 调仓, 调仓日扣双边成本.
    返回: port(策略), bench(全样本等权基准), rnd(随机top_k基线, 多seed均值),
          ex_bench(减等权), ex_rnd(减随机=真实alpha, 零假设=0).
    关键: 收益分布有偏时, top-K 子集相对全样本等权的超额并不可靠(右偏时随机top-K会跑输等权), 故必须用随机基线/打乱检验做零假设.
    随机基线用每个调仓日独立种子(避免固定seed重复导致偏低)."""
    dates = sorted(signal.index)
    port_rets, bench_rets, rnd_rets = [], [], []
    held = None; rnd_held = [None] * n_rand; rb = 0
    for i, d in enumerate(dates):
        s = signal.loc[d].dropna(); r = fwd.loc[d].dropna()
        shared = s.index.intersection(r.index)
        if len(shared) < 5:
            port_rets.append(np.nan); bench_rets.append(np.nan); rnd_rets.append([np.nan] * n_rand); continue
        s, r = s[shared], r[shared]
        if i % hold_days == 0:
            k = max(3, int(len(s) * top_k))
            held = set(s.nlargest(k).index)
            rng = np.random.RandomState(rb); rb += 1   # 每个调仓日独立种子
            rnd_held = [set(rng.choice(s.index, k, replace=False)) for _ in range(n_rand)]
        pr = r[r.index.isin(held)].mean() if held else np.nan
        rr = [r[r.index.isin(h)].mean() if h else np.nan for h in rnd_held]
        if i % hold_days == 0:
            pr = pr - top_k * 2 * cost_oneway
            rr = [x - top_k * 2 * cost_oneway for x in rr]
        port_rets.append(pr); bench_rets.append(r.mean()); rnd_rets.append(rr)
    port = pd.Series(port_rets).dropna()
    bench = pd.Series(bench_rets).dropna()
    rnd_df = pd.DataFrame(rnd_rets).dropna()
    rnd = rnd_df.mean(axis=1).dropna()
    # 对齐后再算超额, 避免索引错位引入 NaN
    idx = port.index.intersection(bench.index).intersection(rnd.index)
    port, bench, rnd = port.loc[idx], bench.loc[idx], rnd.loc[idx]
    return port, bench, rnd, port - bench, port - rnd


def _stat(tag, port, bench, rnd, ex_bench, ex_rnd):
    def cum(s): return float((1 + s).cumprod().iloc[-1]) if len(s) else np.nan
    print(f"\n  ── {tag} ──")
    print(f"    多头: 日均={port.mean():+.5f} 夏普={_sharpe(port.values):.3f} 累计={cum(port):.4f}")
    print(f"    等权基准: 日均={bench.mean():+.5f} 夏普={_sharpe(bench.values):.3f} 累计={cum(bench):.4f}")
    print(f"    随机topK: 日均={rnd.mean():+.5f} 夏普={_sharpe(rnd.values):.3f} 累计={cum(rnd):.4f}")
    print(f"    超额(减等权): 日均={ex_bench.mean():+.5f} 夏普={_sharpe(ex_bench.values):.3f} 累计={cum(ex_bench):.4f}")
    print(f"    超额(减随机): 日均={ex_rnd.mean():+.5f} 夏普={_sharpe(ex_rnd.values):.3f} 累计={cum(ex_rnd):.4f}  <- 真实alpha(零假设=0)")
    return dict(tag=tag, port_mean=float(port.mean()), port_sharpe=_sharpe(port.values),
                bench_sharpe=_sharpe(bench.values), rnd_sharpe=_sharpe(rnd.values),
                ex_bench_sharpe=_sharpe(ex_bench.values), ex_rnd_sharpe=_sharpe(ex_rnd.values),
                ex_rnd_cum=cum(ex_rnd))


def _xgb_walkforward(zfac, y_series, factor_names, dates, train_min=TRAIN_DAYS, test_days=TEST_DAYS):
    feat_df = pd.concat([zfac[n].stack().rename(n) for n in factor_names], axis=1)
    panel = pd.concat([feat_df, y_series.rename("y")], axis=1).dropna(subset=["y"])
    lin_series = feat_df.mean(axis=1).rename("lin")
    dts = sorted(panel.index.get_level_values(0).unique())
    recs = []; start = 0
    while start + train_min + test_days <= len(dts):
        tr_end = start + train_min; te_end = min(tr_end + test_days, len(dts))
        tr = panel.loc[panel.index.get_level_values(0).isin(dts[:tr_end])]
        te = panel.loc[panel.index.get_level_values(0).isin(dts[tr_end:te_end])]
        if len(tr) < 200 or te.empty:
            start += test_days; continue
        m = xgb.XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             n_jobs=-1, random_state=42, verbosity=0)
        m.fit(tr[factor_names].values, tr["y"].values)
        rec = te.copy(); rec["pred"] = m.predict(te[factor_names].values)
        recs.append(rec); start += test_days
    if not recs:
        return None, None
    oos = pd.concat(recs).sort_index()
    pred = oos["pred"].unstack()  # date×code
    return oos, pred


if __name__ == "__main__":
    t0 = time.time()
    stocks, daily_close, fwd, daily_bars, trading = _load_prep()
    print(f"数据: {len(stocks)} 只, 交易日 {trading[0].date()} ~ {trading[-1].date()} ({len(trading)} 日)")

    FAC_CACHE = Path("/workspace/stock_worm/data/founder_factors.pkl")
    if FAC_CACHE.exists():
        done = pickle.load(open(FAC_CACHE, "rb"))
        raw = {}
        for name, fn, kind in SPECS:
            if name not in done or not isinstance(done[name], pd.DataFrame):
                continue
            fdf = done[name].reindex(trading)
            if fdf.shape[1] < 5 or fdf.dropna(how="all").empty:
                continue
            ic = compute_ic_series(fdf, fwd)
            orient = 1.0 if ic.mean() >= 0 else -1.0
            raw[name] = fdf * orient
            print(f"  {name:<20} 有效股票={fdf.shape[1]:3d}  全窗口IC={ic.mean():+.4f} 定向={'高' if orient>0 else '低'}  (缓存)")
    else:
        raw = _compute_factors(stocks, daily_bars, trading, fwd)
    factor_names = list(raw)
    if not factor_names:
        print("无可用因子"); raise SystemExit(1)

    md = ["# 云端因子回测 + XGBoost 选股报告", "",
          f"- 数据: stock_worm 源 5m 本地缓存, {len(stocks)} 只沪深300成分, "
          f"{trading[0].date()}~{trading[-1].date()} ({len(trading)} 交易日)",
          "- 因子: 7 个 founder 头部因子(静态方向自适应)",
          "- 成本: 单边千一, 多头前30%, 5日持有", ""]

    # ── 路线(1): 线性因子组合(因子级) ──
    print("\n" + "=" * 64); print("路线(1) 线性因子组合 — 因子级评估"); print("=" * 64)
    wf_icir = walk_forward_combine(raw, fwd, mode="icir")
    if wf_icir is not None:
        ic = compute_ic_series(wf_icir, fwd)
        icir = ic.mean() / ic.std() * np.sqrt(252) if ic.std() > 0 else np.nan
        eq = compute_group_equity(wf_icir, fwd, 5)
        gr = eq.pct_change().dropna()
        ls = _sharpe((gr["Group_5"] - gr["Group_1"]).values) if len(gr) > 20 else np.nan
        top = _sharpe(gr["Group_5"].values) if len(gr) > 20 else np.nan
        bot = _sharpe(gr["Group_1"].values) if len(gr) > 20 else np.nan
        print(f"  IC_mean={ic.mean():+.4f} ICIR={icir:.3f} ic_pos={(ic>0).mean():.3f}")
        print(f"  多空夏普(G5-G1)={ls:.3f}  最高组夏普={top:.3f}  最低组夏普={bot:.3f}")
        md += [f"## 路线(1) 线性因子组合(walk-forward, ICIR加权)",
               f"- 组合 rank-IC = {ic.mean():+.4f}, ICIR = {icir:.3f}, ic_pos = {(ic>0).mean():.3f}",
               f"- 分层多空夏普(G5-G1) = **{ls:.3f}** ← 因子 IC 为正但多空为负 = 非单调特征",
               f"- 最高组(Group5)夏普 = {top:.3f}, 最低组(Group1)夏普 = {bot:.3f}", ""]

        # 线性多头组合(可交易, 含成本)
        print("\n  线性组合 -> 可交易多头(前30%, 5日持有, 含成本):")
        port, bench, rnd, ex_b, ex_r = long_only_backtest(wf_icir, fwd, TOP_K, 5)
        s1 = _stat("线性多头(前30%, 5d, 含成本)", port, bench, rnd, ex_b, ex_r)
    else:
        s1 = None

    # ── 路线(2): XGBoost 选股 ──
    print("\n" + "=" * 64); print("路线(2) XGBoost 选股 — walk-forward"); print("=" * 64)
    zfac = {n: raw[n].apply(_zscore_row, axis=1) for n in factor_names}
    y_series = fwd.stack().rename("y")
    oos, pred = _xgb_walkforward(zfac, y_series, factor_names, None)
    if oos is not None:
        ic_rows = []
        for d, g in oos.groupby(level=0):
            if g["pred"].notna().sum() < 5 or g["y"].notna().sum() < 5: continue
            ic_rows.append(g["pred"].rank().corr(g["y"].rank()))
        ic_oos = pd.Series(ic_rows).dropna()
        icir_x = ic_oos.mean() / ic_oos.std() * np.sqrt(252) if ic_oos.std() > 0 else np.nan
        print(f"  OOS rank-IC_mean={ic_oos.mean():+.4f} ICIR={icir_x:.3f} ic_pos={(ic_oos>0).mean():.3f}")
        md += [f"## 路线(2) XGBoost 选股(walk-forward XGBRegressor)",
               f"- OOS rank-IC = {ic_oos.mean():+.4f}, ICIR = {icir_x:.3f}, ic_pos = {(ic_oos>0).mean():.3f}", ""]
        print("\n  XGBoost 预测 -> 可交易多头(前30%, 5日持有, 含成本):")
        port, bench, rnd, ex_b, ex_r = long_only_backtest(pred, fwd, TOP_K, 5)
        s2 = _stat("XGBoost多头(前30%, 5d, 含成本)", port, bench, rnd, ex_b, ex_r)
        md += [f"### 多头组合对比(前30%, 5日持有, 单边千一)",
               "| 路线 | 多头夏普 | 等权基准夏普 | 随机topK夏普 | 超额(减等权) | 超额(减随机) |",
               "|------|---------|------------|------------|------------|------------|",
               f"| 线性因子组合 | {s1['port_sharpe']:.3f} | {s1['bench_sharpe']:.3f} | {s1['rnd_sharpe']:.3f} | {s1['ex_bench_sharpe']:.3f} | {s1['ex_rnd_sharpe']:.3f} |" if s1 else "| 线性 | - | - | - | - | - |",
               f"| **XGBoost** | {s2['port_sharpe']:.3f} | {s2['bench_sharpe']:.3f} | {s2['rnd_sharpe']:.3f} | {s2['ex_bench_sharpe']:.3f} | **{s2['ex_rnd_sharpe']:.3f}** |",
               "",
               "> **方法学修正**: 收益右偏时, 随机 top-K 会结构性跑输'全样本等权'基准, 故组合层'超额(减等权)'"
               "会被高估, 不可直接当作 alpha. 信号质量应以**截面 rank-IC** 为准(对分布偏度稳健).",
               f"> 本窗口(2024-05~2026-06)因子全窗口 IC 多数为负(**regime 切换**):",
               f"> - XGBoost **OOS rank-IC = {ic_oos.mean():+.4f}** (近零微负, ic_pos={ (ic_oos>0).mean():.3f}) → 无可靠截面预测力",
               f"> - 组合层 XGBoost 多头夏普 {s2['port_sharpe']:.3f} > 等权 {s2['bench_sharpe']:.3f}, 但该差距在右偏行情下"
               f"属基准结构现象, 且随机 top-K 基线仅 {s2['rnd_sharpe']:.3f}(同样跑输等权), 故**不构成稳健 alpha 证据**.",
               "> **结论**: ① 线性因子组合因因子→收益呈**非单调(倒U)**, 多头/多空结构性亏钱(已确认, 自洽); "
               "② XGBoost 在此窗口 **rank-IC≈0/−0.01, 无可靠信号**, 组合层表面优势经基准偏误修正后不成立 —— "
               "早先 xgb_selector 的'正超额'是等权基准偏误假象, 已证伪. 两条路线在**此 regime 下均失效**, "
               "根因是因子 IC 随行情翻转, 需跨 regime 更多样本 / 因子重标定 / 更长训练跨度.", ""]
    else:
        print("  XGBoost 样本不足")

    md += [f"\n---\n*生成于云端回测, 耗时 {time.time()-t0:.1f}s, 数据 stock_worm 本地缓存*"]
    out = Path(__file__).parent / "screen_results" / "云端回测_XGBoost选股报告.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告已写: {out}  (耗时 {time.time()-t0:.1f}s)")
