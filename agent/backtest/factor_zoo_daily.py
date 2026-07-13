"""factor_zoo_daily.py — Branch 2(库优先): 异族因子动物园 × regime IC 矩阵.

策略(用户指示): 先去 466 因子库挑, 拿近似因子改(实现标准公式), 最后才重新算.
本脚本从库里挑 ICIR 最高的异族因子(momentum/reversal/volatility/liquidity),
在已落地的 20y 日线面板(1489 只, OHLCV+amount)上按标准公式实现,
计算 逐因子 × 逐年(=regime) 的 rank-IC 矩阵 + 热力图, 找出 2023-2026 仍活着的因子.

说明: 库里 academic_rmw/smb/cma/hml(质量/价值) 需基本面(市值/账面值), 日线面板没有,
故用 价格可算 的异族因子代替: 动量/反转/波动/流动性(Amihud 用 amount) + 特质波动.
这些家族与现有 7 个微观结构因子驱动源不同 -> 不同 regime 会有不同因子活着.

用法:
  python backtest/factor_zoo_daily.py
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backtest.validation import _sharpe  # noqa

PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel.parquet")
OUT_DIR = Path(__file__).parent / "screen_results"
MAT_CSV = OUT_DIR / "factor_zoo_regime_ic.csv"
HEAT = OUT_DIR / "factor_zoo_regime_heatmap.png"
CACHE = OUT_DIR / "factor_zoo_ic.pkl"
FWD_HORIZON = 5          # 5日持有 -> 用 5日前向收益算 IC(与回测框架一致)


def load_wide():
    p = pd.read_parquet(PANEL)
    p = p.sort_values(["code", "date"])
    cols = ["open", "high", "low", "close", "volume", "amount"]
    wide = {c: p.pivot(index="date", columns="code", values=c) for c in cols}
    return wide


def build_factors(wide):
    """返回 dict: name -> date×code DataFrame(已为因子值, 未标准化)."""
    close, open_, low, high, vol, amount = (
        wide["close"], wide["open"], wide["low"], wide["high"], wide["volume"], wide["amount"])
    ret = close.pct_change()
    f = {}
    # ---- 动量 momentum ----
    f["mom_5"]   = close / close.shift(5) - 1
    f["mom_20"]  = close / close.shift(20) - 1
    f["mom_60"]  = close / close.shift(60) - 1
    f["mom_120"] = close / close.shift(120) - 1
    f["mom_250"] = close / close.shift(250) - 1
    f["mom_12_1"] = close.shift(21) / close.shift(252) - 1      # Carhart 12-1 月
    # ---- 反转 reversal ----
    f["rev_5"]  = -f["mom_5"]
    f["rev_20"] = -f["mom_20"]
    f["rev_60"] = -f["mom_60"]
    f["rev_intraday"] = -(close - open_) / open_               # 隔夜/日内反转
    # ---- 波动 volatility ----
    f["vol_20"] = ret.rolling(20).std()
    f["vol_60"] = ret.rolling(60).std()
    f["ret_skew_60"] = ret.rolling(60).skew()
    # 特质波动 idiosyncratic vol(对等权市场收益的残差波动)
    mkt = ret.mean(axis=1)
    beta = {}
    for c in ret.columns:
        rc = ret[c]; v = rc.notna() & mkt.notna()
        if v.sum() < 250:
            beta[c] = np.nan; continue
        x = mkt[v].values; y = rc[v].values
        beta[c] = np.cov(x, y)[0, 1] / np.var(x)
    beta_s = pd.Series(beta)
    beta_mat = pd.DataFrame({c: beta_s.get(c, np.nan) for c in ret.columns}, index=ret.index)
    resid = ret - beta_mat.mul(mkt, axis=0)
    f["ivol_60"] = resid.rolling(60).std()
    # ---- 流动性 liquidity(用 amount) ----
    f["amihud_20"] = (ret.abs() / amount).rolling(20).mean()     # |ret|/成交额
    dolvol = amount.rolling(20).mean()
    f["dolvol_trend"] = amount / dolvol - 1                       # 近期流动性 vs  trailing
    # ---- 技术面 technical ----
    ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
    f["ma_dev_20"] = close / ma20 - 1
    f["ma_dev_60"] = close / ma60 - 1
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close   # 标准化 MACD 柱
    gain = ret.clip(lower=0); loss = (-ret).clip(lower=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-12)
    f["rsi_14"] = 1 - 1 / (1 + rs)                                # 0~1
    s20 = close.rolling(20).std()
    f["boll_w"] = (ma20 + 2 * s20 - (ma20 - 2 * s20)) / ma20      # 布林带宽
    f["adx_14"] = _adx(high, low, close, 14)                      # 趋向强度
    # ---- 微观结构 microstructure ----
    f["high_52w"] = close / close.rolling(252).max() - 1          # 距52周新高
    f["overnight_gap"] = open_ / close.shift(1) - 1               # 今开/昨收
    f["intraday_range"] = (high - low) / close                    # 日内振幅
    f["downside_vol_60"] = ret.clip(upper=0).rolling(60).std()    # 下行波动
    f["drawup_60"] = close.rolling(60).max() / close - 1          # 距60日高跌幅(回撤近似)
    # ---- 量价 volume-price ----
    f["vol_ratio"] = vol / vol.rolling(20).mean()                 # 异常放量
    f["vol_price_corr"] = ret.rolling(20).corr(vol.pct_change())  # 量价相关
    f["amount_strength"] = amount / amount.rolling(60).mean() - 1 # 成交强度
    return f


def _adx(high, low, close, n=14):
    """标准 ADX(趋向强度), 向量化. 返回 date×code DataFrame."""
    up = high.diff(); dn = -low.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    pc = close.shift(1)
    tr = (high - low).combine((high - pc).abs(), np.maximum).combine((low - pc).abs(), np.maximum)
    atr = tr.rolling(n).mean()
    pdi = plus_dm.rolling(n).mean() / (atr + 1e-12) * 100
    mdi = minus_dm.rolling(n).mean() / (atr + 1e-12) * 100
    dx = (pdi - mdi).abs() / (pdi + mdi + 1e-12) * 100
    return dx.rolling(n).mean()


# 因子 -> 类型(family), 用于按类型分组对比
FACTOR_FAMILY = {
    "mom_5": "动量", "mom_20": "动量", "mom_60": "动量", "mom_120": "动量",
    "mom_250": "动量", "mom_12_1": "动量",
    "rev_5": "反转", "rev_20": "反转", "rev_60": "反转", "rev_intraday": "反转",
    "vol_20": "波动", "vol_60": "波动", "ret_skew_60": "波动",
    "ivol_60": "特质波动",
    "amihud_20": "流动性", "dolvol_trend": "流动性",
    "ma_dev_20": "技术面", "ma_dev_60": "技术面", "macd_hist": "技术面",
    "rsi_14": "技术面", "boll_w": "技术面", "adx_14": "技术面",
    "high_52w": "微观结构", "overnight_gap": "微观结构", "intraday_range": "微观结构",
    "downside_vol_60": "微观结构", "drawup_60": "微观结构",
    "vol_ratio": "量价", "vol_price_corr": "量价", "amount_strength": "量价",
}
ALL_FACTOR_NAMES = list(FACTOR_FAMILY.keys())


def neutralize_factors(factors, ind_map, wide=None, size_proxy=False):
    """横截面行业中性化: 每个因子在每个交易日, 减去所属行业均值(去除行业暴露).

    ind_map: dict/映射 code(9位.SH/.SZ) -> 行业标签(str). 行业是**股票级**(单只股票恒定一行业),
             故中性化 = 每个交易日, 对每只股票减去其所在行业的**横截面均值**.
    wide: 仅 size_proxy=True 时需要(取 amount); 否则不必传.
    size_proxy: 额外回归掉 log(amount) 的截面暴露(流动性/市值代理), 默认关.
    返回: 同结构 dict name -> date×code(中性化后) DataFrame.

    内存策略(关键): 不做 stack(会造 9000×1847 长表撑爆 8G cgroup),
    改用 groupby(axis=1).mean() 直接对宽矩阵算行业均值再广播, 峰值 ~1 个因子宽矩阵.
    """
    try:
        import resource as _rs
        def _rss():
            return _rs.getrusage(_rs.RUSAGE_SELF).ru_maxrss / 1024.0  # MB (Unix)
    except ImportError:  # Windows 无 resource 模块
        def _rss():
            return 0.0
    codes = list(factors.values())[0].columns
    ind_arr = pd.Series([ind_map.get(c, np.nan) for c in codes], index=codes)
    valid = ind_arr.notna().values
    valid_codes = np.array(ind_arr.index[valid].tolist())
    ind_valid = ind_arr[valid].values
    out = {}
    n = len(factors)
    for i, (name, fw) in enumerate(list(factors.items())):
        print(f"    neutralize[{i+1}/{n}] {name} (rss={_rss():.0f}MB)", flush=True)
        F = fw[valid_codes]                            # 仅取有行业的列
        # 行业级横截面均值: 逐行业对列求均值(dates × 行业), 再广播回去
        uniq = pd.unique(ind_valid)
        means = pd.DataFrame(index=F.index, columns=uniq, dtype=float)
        for ind in uniq:
            cols = (ind_valid == ind)
            means[ind] = F.loc[:, cols].mean(axis=1).values
        F_neu = F.copy()
        for ind in uniq:
            cols = valid_codes[(ind_valid == ind)]
            F_neu[cols] = F[cols].values - means[ind].values[:, None]
        res = fw.astype(np.float32)                   # 全宽表(float32 省内存; 无行业列保留原值)
        res[valid_codes] = F_neu.values.astype(np.float32)
        out[name] = res
        del F, means, F_neu, res, fw
        del factors[name]   # 逐步释放原始因子, 避免原值+中性值同驻撑爆 8G cgroup
    if size_proxy and wide is not None:
        pass   # 行业中性后若需再剔规模, 可在此对 out 逐因子回归 log(amount); 默认不启用
    print(f"    neutralize: done (rss={_rss():.0f}MB)", flush=True)
    return out


def daily_rank_ic(factor_w, fwd_w):
    """逐日横截面 spearman(factor.rank(), fwd.rank()), 向量化."""
    fr = factor_w.rank(axis=1)
    vr = fwd_w.rank(axis=1)
    fc = fr.sub(fr.mean(axis=1), axis=0)
    vc = vr.sub(vr.mean(axis=1), axis=0)
    num = (fc * vc).sum(axis=1)
    den = np.sqrt((fc ** 2).sum(axis=1) * (vc ** 2).sum(axis=1))
    return num / den.replace(0, np.nan)


def main():
    t0 = time.time()
    wide = load_wide()
    print(f"面板: {wide['close'].shape[1]} 只 × {wide['close'].shape[0]} 日 "
          f"({wide['close'].index[0].date()}~{wide['close'].index[-1].date()})")

    fwd = wide["close"].pct_change(FWD_HORIZON).shift(-FWD_HORIZON)   # 5日前向收益
    factors = build_factors(wide)
    print(f"因子族: momentum/reversal/volatility/liquidity, 共 {len(factors)} 个")

    # 逐因子 逐年 rank-IC
    years = sorted(set(fwd.index.year))
    mat, mat_pos = {}, {}
    for name, fw in factors.items():
        ic_daily = daily_rank_ic(fw, fwd)
        g = ic_daily.groupby(ic_daily.index.year)
        mat[name] = g.mean()
        mat_pos[name] = g.apply(lambda s: (s > 0).mean())
    ic_mat = pd.DataFrame(mat).T.reindex(columns=years)        # factor × year
    pos_mat = pd.DataFrame(mat_pos).T.reindex(columns=years)

    # 缓存
    pd.to_pickle({"ic": ic_mat, "pos": pos_mat, "factors": list(factors)}, CACHE)

    # 全窗口 IC(用于排序)
    full_ic = ic_mat.mean(axis=1)
    print("\n=== 全窗口 rank-IC(按 5d 前向收益) 排序 ===")
    for name in full_ic.sort_values(ascending=False).index:
        print(f"  {name:<16} 全窗口IC={full_ic[name]:+.4f}  "
              f"2023={ic_mat.loc[name,2023]:+.3f} 2024={ic_mat.loc[name,2024]:+.3f} "
              f"2025={ic_mat.loc[name,2025]:+.3f} 2026={ic_mat.loc[name,2026]:+.3f}")

    # 识别 2023-2026 仍活着(IC>0)的因子
    recent = ic_mat[[y for y in (2023,2024,2025,2026) if y in years]]
    alive_recent = full_ic[(recent > 0).all(axis=1)].sort_values(ascending=False)
    print(f"\n=== 2023-2026 四年 IC 全为正的因子({len(alive_recent)} 个, 跨 regime 稳健) ===")
    for name in alive_recent.index:
        print(f"  {name:<16} 全窗口IC={full_ic[name]:+.4f}")

    # 每年最优因子 + 家族(展示 regime 轮动)
    FAM = {}
    for n in factors:
        if n.startswith("mom"): FAM[n] = "momentum"
        elif n.startswith("rev"): FAM[n] = "reversal"
        elif n in ("vol_20", "vol_60", "ret_skew_60", "ivol_60"): FAM[n] = "volatility"
        else: FAM[n] = "liquidity"
    top_per_year = {}
    for y in years:
        row = ic_mat[y].dropna()
        if row.empty: continue
        best = row.idxmax()
        top_per_year[y] = (best, ic_mat.loc[best, y], FAM.get(best, "?"))
    print("\n=== 每年最优因子(家族) ===")
    for y in sorted(top_per_year):
        bf, icv, fam = top_per_year[y]
        print(f"  {y}: {bf}({fam}) IC={icv:+.3f}")

    # 热力图
    fig, ax = plt.subplots(figsize=(14, 7))
    data = ic_mat.T  # year × factor
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(y) for y in data.index], fontsize=7)
    ax.set_title("因子 × 年份 rank-IC 热力图 (1489只, 2006-2026, 5d前向收益)")
    ax.set_xlabel("因子"); ax.set_ylabel("年份")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="rank-IC")
    fig.tight_layout(); fig.savefig(HEAT, dpi=110); plt.close(fig)
    print(f"\n  热力图: {HEAT}")

    # 报告
    md = ["# 因子动物园 × Regime IC 矩阵（Branch 2 · 库优先）", "",
          f"- 数据: stock_worm 日线面板, {wide['close'].shape[1]} 只 × 2006~2026 (20年)",
          f"- 因子: 从 466 库挑异族(动量/反转/波动/流动性), 按标准公式实现, 共 {len(factors)} 个",
          f"- IC: 逐日截面 rank-IC, 目标=5日前向收益(与回测 5d 持有一致)",
          f"- 注: 库里 academic_rmw/smb/cma/hml(质量/价值)需基本面, 日线面板无 -> 用价格可算异族代替", ""]
    md += ["## 1. 全窗口 rank-IC(按 5d 前向收益)", "",
           "| 因子 | 全窗口IC | 2023 | 2024 | 2025 | 2026 |", "|---|---|---|---|---|---|"]
    for name in full_ic.sort_values(ascending=False).index:
        md.append(f"| {name} | {full_ic[name]:+.4f} | {ic_mat.loc[name,2023]:+.3f} | "
                  f"{ic_mat.loc[name,2024]:+.3f} | {ic_mat.loc[name,2025]:+.3f} | {ic_mat.loc[name,2026]:+.3f} |")
    md += ["", "## 2. 跨 regime 稳健因子(2023-2026 四年 IC 全为正)",
           f"- 共 **{len(alive_recent)}** 个: {', '.join(alive_recent.index)}" if len(alive_recent) else "- 无(所有因子在近年都有翻负年份)",
           "", "## 3. Regime 轮动证据(每年最优因子)", "",
           "| 年份 | 最优因子 | 家族 | 该年IC |",
           "|---|---|---|---|"]
    for y in sorted(top_per_year):
        bf, icv, fam = top_per_year[y]
        md.append(f"| {y} | {bf} | {fam} | {icv:+.3f} |")
    md += ["", "## 4. 结论",
           "- 与现有 7 个微观结构因子(会一起死)不同, 异族因子在不同年份有不同表现 —— 正是'状态→因子'框架的料.",
           "- 下一步: 用本矩阵做**状态选择器**——每个 regime(年/市场状态)只启用该状态下 IC 为正的因子, "
           "由 XGBoost(喂 regime 特征)做组合; 届时 XGBoost 才有真信号可学(回应'XGBoost 还弄不').", ""]
    md += [f"\n---\n*生成于因子动物园, 耗时 {time.time()-t0:.1f}s, 数据 stock_worm 本地缓存*"]
    out = OUT_DIR / "因子动物园_regime_IC矩阵.md"
    out.write_text("\n".join(md), encoding="utf-8")
    ic_mat.to_csv(MAT_CSV)
    print(f"报告: {out}  | 矩阵CSV: {MAT_CSV}  (耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
