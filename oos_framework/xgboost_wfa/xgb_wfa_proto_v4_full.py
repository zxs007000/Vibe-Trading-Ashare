#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 因子挖掘 · WFA 原型 v4（v3 + 筹码结构维度）
====================================================
在 v3（v2 交互+多周期融合+IS调参 + 因子拥挤度）基础上，新增用户要求的「筹码结构」维度，
严格落地《筹码结构量化建模与XGBoost实战指南》：

  §2  VWAP 中心三角分布递推动态筹码分布 —— 每日筹码按换手率衰减、新筹码以 VWAP 为峰注入三角分布。
  §4  筹码指标 —— 获利比例 PR / 成本集中度 CC(HHI归一化) / 筹码乖离 CB=(价-加权成本)/标准差 / 短期CB。
  §5.1 PR×CB 交互；§5.2 短期乖离>95%分位惩罚 + 成本集中度 CC 作训练样本权重。

数据缺口 / 建模调整(本数据湖无对应字段，用代理并标注)：
  §1 真实换手率(需流通股本) —— 用 量/60日均量 截断[0.2,3]×decay_base 代理。
  §3 逐笔 tick / 资金类型分层(需 Level-2) —— 数据湖无逐笔数据，无法做资金属性分层。
  §4 CC 指南用「峰度」，但三角分布注入使峰度恒≈-0.6(退化、且为负→样本权重失效)，
       改用语义等价的 HHI 归一化集中度(∈[0,1]，越集中越接近1)，更贴合§4意图且可用于样本权重。

前置维度(来自 v3，本文件一并保留)：
  §1.1 交易行为拥挤度(成交额代理) + §1.4 PCA吸收比率 + §2 特征展开；
  §1.2/§1.3 因缺 PE/基金持仓数据跳过。
修复 v2 瑕疵：市场环境/拥挤度/筹码均用「原始(未z-score)特征」计算，
再对 per-stock 的 Alpha+筹码 做截面标准化，避免市场环境特征退化成常数。
"""
import os, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180)

LAKE = os.environ.get("STOCKLAKE", "/workspace/stocklake")
DAILY = f"{LAKE}/daily"
FUND = f"{LAKE}/fundamentals"
INCOME = f"{FUND}/income_statement"
CASH = f"{FUND}/cash_flow_statement"
BAL = f"{FUND}/balance_sheet"
# 落盘目录: 所有 booster / shap 子样本 / 特征表 / 结果 markdown 都写到这里。
# 可用环境变量 WFA_OUT 覆盖(本地复现时指向你自己的结果目录); 默认与脚本同级的 v4proto_out/。
OUT = os.environ.get("WFA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4proto_out"))
os.makedirs(OUT, exist_ok=True)
# 注: cnstock(data.cnstock.com) 在构建期被我们的高并发爬取触发整域 403 封禁, 暂不可用。
# 基本面改回东财三大表(已建 95%+)派生, 见 load_fundamentals()。
MAX_STOCKS = int(os.environ.get("WFA_MAX_STOCKS", 5500))   # 全量≈5448; 内存受限时用 WFA_MAX_STOCKS 调低(代表性随机子样本)
TRAIN_YEARS = 3
TEST_YEARS = 1
STEP_YEARS = 1
RANDOM_STATE = 42
HORIZONS = [5, 20, 60]
ZYEARS = 5 * 252          # 5年滚动窗口(交易日)
PCT_WIN = 1250           # 历史分位数窗口(文档§2.1)

PRICE_FEATS = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "rsi_14",
               "amt_chg_20", "ma_dev_20", "amp_20"]
# 东财三大表派生的基本面因子(可靠且跨行业均匀):
#   eps            <- income.BASIC_EPS
#   netprofit_yoy  <- income.PARENT_NETPROFIT_YOY
#   debt_ratio     <- balance.TOTAL_LIABILITIES / TOTAL_ASSETS
#   roe            <- income.PARENT_NETPROFIT / balance.TOTAL_PARENT_EQUITY
#   bvps           <- balance.TOTAL_PARENT_EQUITY / SHARE_CAPITAL(股)
#   ocf_ps         <- cash.NETCASH_OPERATE / SHARE_CAPITAL(股)
#   ocf_netprofit  <- cash.NETCASH_OPERATE / income.PARENT_NETPROFIT
#   current_ratio  <- balance.CURRENT_ASSET_BALANCE / CURRENT_LIAB_BALANCE(银行股缺→NaN)
# 注: gross_margin 东财利润表无成本列, 无法计算, 已弃用。
FUND_FEATS = ["roe", "debt_ratio", "current_ratio", "ocf_netprofit",
              "netprofit_yoy", "eps", "bvps", "ocf_ps"]
ALPHA_FEATS = PRICE_FEATS + FUND_FEATS
CHIP_FEATS = ["chip_pr", "chip_cc", "chip_cb", "chip_cb_short"]  # 筹码结构(§4)


# ───────────────────────── 数据加载（东财三大表派生基本面） ─────────────────────────
def eligible_codes():
    # 日线 ∩ 利润表 ∩ 现金流 ∩ 资产负债表 三者齐全
    daily = {f[:-8] for f in os.listdir(DAILY) if f.endswith(".parquet")}
    inc = {f[:-8] for f in os.listdir(INCOME) if f.endswith(".parquet")}
    cash = {f[:-8] for f in os.listdir(CASH) if f.endswith(".parquet")}
    bal = {f[:-8] for f in os.listdir(BAL) if f.endswith(".parquet")}
    codes = sorted(daily & inc & cash & bal)
    if MAX_STOCKS < len(codes):
        # 子样本: 固定随机种子抽样, 使任意 MAX_STOCKS 都是全市场的**代表性随机样本**,
        # 避免 sorted()[:N] 偏向低代码号(沪市 600xxx 前置)带来的结构性偏差。
        # 注: 全市场 5448 只的面板(~11M行×208维)在 8GB cgroup 内存上限下会 OOM,
        # 故默认用子样本; 更大内存或分块处理时可调高 WFA_MAX_STOCKS。
        import random
        codes = random.Random(42).sample(codes, MAX_STOCKS)
    return codes


def _read_parquet(path):
    try:
        d = pd.read_parquet(path)
        if "REPORT_DATE" in d.columns:
            d["REPORT_DATE"] = pd.to_datetime(d["REPORT_DATE"], errors="coerce")
            d = d.dropna(subset=["REPORT_DATE"]).set_index("REPORT_DATE")
        else:
            d.index = pd.to_datetime(d.index, errors="coerce")
            d = d[~d.index.isna()]
        return d
    except Exception:
        return None


def load_fundamentals(code):
    """东财三大表派生 8 个基本面因子。

    利润表: eps(BASIC_EPS), netprofit_yoy(PARENT_NETPROFIT_YOY), _pn(PARENT_NETPROFIT)
    资产负债表: debt_ratio, roe(_pn/TOTAL_PARENT_EQUITY), bvps(TOTAL_PARENT_EQUITY/SHARE_CAPITAL),
                current_ratio(CURRENT_ASSET_BALANCE/CURRENT_LIAB_BALANCE, 银行股缺→NaN)
    现金流: ocf_ps(NETCASH_OPERATE/SHARE_CAPITAL), ocf_netprofit(NETCASH_OPERATE/_pn)
    索引对齐到三表报告期并集, 裁到 2017+ 后前向填充。
    """
    inc = _read_parquet(f"{INCOME}/{code}.parquet")
    cash = _read_parquet(f"{CASH}/{code}.parquet")
    bal = _read_parquet(f"{BAL}/{code}.parquet")
    if inc is None and cash is None and bal is None:
        return None
    # 关键: 提前初始化 pn/sc, 否则当某股票缺资产负债表(bal=None)但现金流存在时,
    # 下方 `if sc is not None` 会触发 UnboundLocalError(全市场跑必现)
    pn = None
    sc = None
    # 报告期并集作为基底
    idxs = [d.index for d in (inc, cash, bal) if d is not None]
    out = pd.DataFrame(index=sorted(set().union(*idxs)))
    # 利润表
    if inc is not None:
        if "BASIC_EPS" in inc.columns:
            out["eps"] = inc["BASIC_EPS"]
        if "PARENT_NETPROFIT_YOY" in inc.columns:
            out["netprofit_yoy"] = inc["PARENT_NETPROFIT_YOY"]
        pn = inc["PARENT_NETPROFIT"] if "PARENT_NETPROFIT" in inc.columns else None
    else:
        pn = None
    # 资产负债表
    if bal is not None:
        ta = bal["TOTAL_ASSETS"] if "TOTAL_ASSETS" in bal.columns else None
        tl = bal["TOTAL_LIABILITIES"] if "TOTAL_LIABILITIES" in bal.columns else None
        if ta is not None and tl is not None:
            out["debt_ratio"] = tl / ta.replace(0, float("nan"))
        tpe = bal["TOTAL_PARENT_EQUITY"] if "TOTAL_PARENT_EQUITY" in bal.columns else None
        sc = bal["SHARE_CAPITAL"] if "SHARE_CAPITAL" in bal.columns else None
        if tpe is not None:
            if pn is not None:
                out["roe"] = pn / tpe.replace(0, float("nan"))
            if sc is not None:
                out["bvps"] = tpe / sc.replace(0, float("nan"))
        ca = bal["CURRENT_ASSET_BALANCE"] if "CURRENT_ASSET_BALANCE" in bal.columns else None
        cl = bal["CURRENT_LIAB_BALANCE"] if "CURRENT_LIAB_BALANCE" in bal.columns else None
        if ca is not None and cl is not None:
            out["current_ratio"] = ca / cl.replace(0, float("nan"))
    # 现金流
    if cash is not None:
        nco = cash["NETCASH_OPERATE"] if "NETCASH_OPERATE" in cash.columns else None
        if nco is not None:
            if sc is not None:
                out["ocf_ps"] = nco / sc.replace(0, float("nan"))
            if pn is not None:
                out["ocf_netprofit"] = nco / pn.replace(0, float("nan"))
    # 保证 8 个基本面列齐全(银行股等缺 current_ratio 时填 NaN),
    # 否则下游 panel[FUND_FEATS] 会因某只票缺列而 KeyError
    for c in ["roe", "debt_ratio", "current_ratio", "ocf_netprofit",
              "netprofit_yoy", "eps", "bvps", "ocf_ps"]:
        if c not in out.columns:
            out[c] = float("nan")
    # 裁到近窗口(避免 2001 等远古值前滚污染 2018+), 再前向填充
    out = out[out.index >= pd.Timestamp("2017-01-01")]
    out = out.sort_index().ffill().replace([float("inf"), float("-inf")], float("nan"))
    out.index.name = "REPORT_DATE"
    return out


def chip_features(df, price_bins=100, decay_base=0.02):
    """§2 VWAP中心三角分布递推 + §4 PR/CC/CB(筹码结构)。
    数据缺口: 真实换手率(需流通股本)、逐笔tick(需Level2) 本数据湖无 → 用代理。
      VWAP   ≈ 成交额/成交量 (已验证≈close, 单位一致)
      turnover ≈ 量/60日均量 截断[0.2,3]×decay_base (无流通股本, 用相对活跃度代理)
    返回与 df 对齐的 DataFrame[chip_pr, chip_cc, chip_cb, chip_cb_short]。"""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    vol = df["volume"].values.astype(float)
    amt = df["amount"].values.astype(float)
    T = len(close)
    vwap = np.where(vol > 0, amt / np.where(vol > 0, vol, 1.0), close)
    roll = pd.Series(vol).rolling(60, min_periods=10).mean().values
    turn = np.clip(np.where(roll > 0, vol / roll, 1.0), 0.2, 3.0) * decay_base
    min_p = np.minimum(np.nanmin(low), np.nanmin(close))
    max_p = np.maximum(np.nanmax(high), np.nanmax(close))
    if not (np.isfinite(min_p) and np.isfinite(max_p) and max_p > min_p):
        return pd.DataFrame(index=df.index,
                            data={k: np.nan for k in CHIP_FEATS})
    bins = np.linspace(min_p, max_p, price_bins + 1)
    bc = (bins[:-1] + bins[1:]) / 2.0
    chip = np.zeros((T, price_bins))
    newmat = np.zeros((T, price_bins))
    for i in range(T):
        if i > 0:
            chip[i] = chip[i - 1] * (1.0 - turn[i])
        # 当日新筹码聚集在 VWAP 附近，宽度取真实振幅(下限2%防一字板)；
        # 注：指南「三角分布」指单日成交量围绕当日VWAP聚集于当日高低区间，
        # 而非跨越全历史价格区间——后者会让筹码被抹平为近似均匀，使CC退化。
        hw = np.maximum(high[i] - low[i], 0.02 * close[i])
        d = np.abs(bc - vwap[i])
        if hw <= 0:
            w = np.zeros(price_bins); w[int(np.argmin(np.abs(bc - vwap[i])))] = 1.0
        else:
            w = np.maximum(0.0, 1.0 - d / hw)
            w = w / w.sum() if w.sum() > 0 else np.zeros(price_bins)
        nv = vol[i] * turn[i]
        chip[i] += w * nv
        newmat[i] = w * nv
    tot = chip.sum(axis=1)
    pr, wmean, wstd, cc, cb = (np.full(T, np.nan) for _ in range(5))
    denom = 1.0 - 1.0 / price_bins        # HHI 归一化分母(均匀分布时→0)
    for i in range(T):
        if tot[i] <= 0:
            continue
        pr[i] = chip[i][bc < close[i]].sum() / tot[i]
        wmean[i] = (chip[i] * bc).sum() / tot[i]
        wstd[i] = np.sqrt(((bc - wmean[i]) ** 2 * chip[i]).sum() / tot[i])
        # 成本集中度 CC：指南用峰度(kurtosis)，但三角分布注入使峰度恒≈-0.6(退化)。
        # 改用语义等价的「HHI 归一化集中度」：筹码越集中(单峰越尖)→越接近1，更贴合§4意图。
        hhi = (chip[i] ** 2).sum() / (tot[i] ** 2)
        cc[i] = (hhi - 1.0 / price_bins) / denom
        cb[i] = np.clip((close[i] - wmean[i]) / wstd[i], -20.0, 20.0) if wstd[i] > 1e-8 else 0.0
    cb_short = np.full(T, np.nan)
    for i in range(T):
        lo = max(0, i - 4)
        sc = newmat[lo:i + 1].sum(axis=0)
        s = sc.sum()
        if s > 0:
            m = (sc * bc).sum() / s
            sd = np.sqrt(((bc - m) ** 2 * sc).sum() / s)
            # 近期筹码极集中时 sd→0 会放大乖离，加下限并截断，避免数值爆炸(后续仍经 winsorize)
            cb_short[i] = np.clip((close[i] - m) / sd, -20.0, 20.0) if sd > 1e-8 else 0.0
    return pd.DataFrame({"chip_pr": pr, "chip_cc": cc, "chip_cb": cb,
                         "chip_cb_short": cb_short}, index=df.index)


def build_stock_panel(code):
    df = pd.read_parquet(f"{DAILY}/{code}.parquet")
    if df is None or len(df) < 80:
        return None
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    c, h, l, v, a = df["close"], df["high"], df["low"], df["volume"], df["amount"]
    ret = c.pct_change()
    df["ret_1"] = ret
    df["ret_5"] = c / c.shift(5) - 1
    df["ret_20"] = c / c.shift(20) - 1
    df["ret_60"] = c / c.shift(60) - 1
    df["vol_20"] = ret.rolling(20).std()
    up, dn = ret.clip(lower=0), (-ret).clip(lower=0)
    rs = up.rolling(14, min_periods=1).mean() / (dn.rolling(14, min_periods=1).mean() + 1e-12)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    df["amt_chg_20"] = a / a.rolling(20).mean() - 1
    df["ma_dev_20"] = c / c.rolling(20).mean() - 1
    df["amp_20"] = (h.rolling(20).max() - l.rolling(20).min()) / c
    # 拥挤度计算需要的原始(未z-score)字段，临时保留，最后丢弃
    df["_amount"] = a
    df["_ret"] = ret
    for hz in HORIZONS:
        df[f"fwd_ret_{hz}"] = c.shift(-hz) / c.shift(-1) - 1   # T+1 起算，防泄露
    fin = load_fundamentals(code)
    if fin is not None and len(fin):
        df["date"] = df["date"].astype("datetime64[ns]")
        fdf = fin.reset_index()
        fdf["REPORT_DATE"] = fdf["REPORT_DATE"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("date"), fdf.sort_values("REPORT_DATE"),
                           left_on="date", right_on="REPORT_DATE", direction="backward")
        df.drop(columns=["REPORT_DATE"], inplace=True, errors="ignore")
    df["code"] = code
    # 筹码结构特征(§2/§4)：基于OHLCV+成交额递推，无需额外数据
    chip = chip_features(df)
    for col in CHIP_FEATS:
        df[col] = chip[col].values
    keep = ["date", "code", "_amount", "_ret"] + ALPHA_FEATS + CHIP_FEATS + [f"fwd_ret_{hz}" for hz in HORIZONS]
    df.dropna(subset=PRICE_FEATS + [f"fwd_ret_{hz}" for hz in HORIZONS], inplace=True)
    return df[keep]


# ───────────────────────── 市场环境(原始特征) + 拥挤度 ─────────────────────────
def market_env(panel):
    """市场环境：每日横截面统计，必须用原始(未z-score)特征。"""
    g = panel.groupby("date")
    env = pd.DataFrame({
        "mkt_ret20_mean": g["ret_20"].mean(),
        "mkt_ret20_std": g["ret_20"].std(),
        "mkt_adv": g["ret_20"].apply(lambda s: (s > 0).mean()),
        "mkt_amt_chg_mean": g["amt_chg_20"].mean(),
        "mkt_ma_dev_mean": g["ma_dev_20"].mean(),
        "mkt_amp_mean": g["amp_20"].mean(),
        "mkt_vol_mean": g["vol_20"].mean(),
        # gross_margin 已弃用(东财利润表无成本列); 改用均匀可得的 roe 截面均值作基本面市场环境代理
        "mkt_roe_mean": g["roe"].mean(),
    }).sort_index()
    # 市场过热标志(巴菲特指标代理)：截面日均收益 5年 z-score > 1.5 → 极端乐观/拥挤 regime
    mr = env["mkt_ret20_mean"]
    mz = (mr - mr.rolling(ZYEARS, min_periods=250).mean()) / (mr.rolling(ZYEARS, min_periods=250).std() + 1e-8)
    env["is_extreme_macro"] = (mz > 1.5).astype(float)
    return env


def trading_crowding(panel, leg_factor, label):
    """§1.1 交易行为拥挤度：多空组合 活跃度比+波动比 → 5年 z-score。
    换手率数据缺失，用成交额(_amount)代理活跃度。"""
    r = panel.groupby("date")[leg_factor].rank(pct=True)
    long_mask = (r >= 0.9).values
    short_mask = (r <= 0.1).values
    gd = panel.groupby("date")
    long_amt = panel.assign(_m=long_mask).groupby("date").apply(
        lambda d: d.loc[d["_m"], "_amount"].mean())
    short_amt = panel.assign(_m=short_mask).groupby("date").apply(
        lambda d: d.loc[d["_m"], "_amount"].mean())
    long_ret = panel.assign(_m=long_mask).groupby("date").apply(
        lambda d: d.loc[d["_m"], "_ret"].mean())
    short_ret = panel.assign(_m=short_mask).groupby("date").apply(
        lambda d: d.loc[d["_m"], "_ret"].mean())
    la = long_amt.rolling(120, min_periods=60).mean()
    sa = short_amt.rolling(120, min_periods=60).mean()
    lv = long_ret.rolling(120, min_periods=60).std()
    sv = short_ret.rolling(120, min_periods=60).std()
    to_ratio = (la / (sa + 1e-8))
    vol_ratio = (lv / (sv + 1e-8))
    raw = (to_ratio + vol_ratio) / 2
    z = (raw - raw.rolling(ZYEARS, min_periods=250).mean()) / (
        raw.rolling(ZYEARS, min_periods=250).std() + 1e-8)
    return z.rename(label).sort_index()


def pca_absorption(panel, w=60):
    """§1.4 资产集中度/PCA吸收比率：每日截面收益滚动w天协方差，第一主成分解释力。"""
    piv = panel.pivot(index="date", columns="code", values="_ret").sort_index()
    M = piv.values
    T, N = M.shape
    out = np.full(T, np.nan)
    for t in range(w - 1, T):
        X = M[t - w + 1:t + 1, :]
        mask = ~np.isnan(X).any(axis=0)
        if mask.sum() < 10:
            continue
        Xc = X[:, mask]
        Xm = Xc - Xc.mean(axis=0)
        try:
            s = np.linalg.svd(Xm, full_matrices=False, compute_uv=False)
            out[t] = (s[0] ** 2) / ((s ** 2).sum() + 1e-12)
        except Exception:
            out[t] = np.nan
    return pd.Series(out, index=piv.index, name="pca_absorp")


def build_crowding(panel, env):
    """§1 三个可行基础指标 + §2 特征展开(原始/分位/变化率/宏观交互/惩罚)。"""
    base = pd.DataFrame(index=panel["date"].unique())
    base.index = pd.to_datetime(base.index)
    base["crowd_mom"] = trading_crowding(panel, "ret_60", "crowd_mom")      # 动量因子拥挤
    base["crowd_liq"] = trading_crowding(panel, "amt_chg_20", "crowd_liq")  # 流动性因子拥挤
    base["pca_absorp"] = pca_absorption(panel)                             # 市场集中度

    macro = env["is_extreme_macro"].reindex(base.index).fillna(0.0)
    feat = pd.DataFrame(index=base.index)
    for col in base.columns:
        s = base[col].astype(float)
        feat[col] = s                                                     # 原始值
        pct = s.rolling(PCT_WIN, min_periods=250).apply(
            lambda w: float((w < w[-1]).mean()), raw=True)               # 历史分位数
        feat[f"{col}_pct"] = pct
        roc = s.diff(5) / (s.abs() + 1e-8)                              # 变化率(加速/缓解)
        feat[f"{col}_roc"] = roc
        feat[f"{col}_x_macro"] = s * macro                              # 宏观过热 × 拥挤
        feat[f"{col}_delta_x_macro"] = roc * macro                      # 拥挤加速 × 宏观
        feat[f"{col}_pen"] = np.where(pct > 0.95, (pct - 0.95) * -10.0, 0.0)  # 惩罚项
    return base, feat


def add_interactions(panel, env):
    """Alpha(已z-score) × 市场环境变量 笛卡尔积；筹码结构特征同样参与交互。"""
    panel = panel.merge(env, left_on="date", right_index=True, how="left")
    env_feats = [c for c in env.columns if c != "is_extreme_macro"]
    inter = []
    for e in env_feats:
        for a in ALPHA_FEATS + CHIP_FEATS:
            nm = f"ix_{e}__{a}"
            panel[nm] = panel[e] * panel[a]
            inter.append(nm)
    return panel, ALPHA_FEATS + CHIP_FEATS + env_feats + inter


def winsorize(panel, cols, lo=0.01, hi=0.99):
    for col in cols:
        if col in panel:
            ql, qh = panel[col].quantile(lo), panel[col].quantile(hi)
            panel[col] = panel[col].clip(ql, qh)
    return panel


def cross_section_zscore(panel, cols):
    grp = panel.groupby("date")[cols]
    panel[cols] = (panel[cols] - grp.transform("mean")) / (grp.transform("std") + 1e-12)
    return panel


def wfa_folds(dates, train=TRAIN_YEARS, test=TEST_YEARS, step=STEP_YEARS):
    d0, d1 = dates.min(), dates.max()
    folds, start = [], pd.Timestamp(d0)
    while True:
        is_s, is_e = start, start + pd.DateOffset(years=train)
        oos_s, oos_e = is_e, is_e + pd.DateOffset(years=test)
        if oos_e > pd.Timestamp(d1):
            break
        folds.append((is_s, is_e, oos_s, oos_e))
        start = start + pd.DateOffset(years=step)
    return folds


def hp_search(X, y, sample_weight=None, n_iter=12):
    """IS 内参数寻优（cv=3，绝不看 OOS）。"""
    clf = XGBClassifier(n_estimators=150, nthread=4, eval_metric="auc",
                        random_state=RANDOM_STATE, use_label_encoder=False)
    param_dist = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0, 1, 5],
        "gamma": [0, 1],
    }
    rs = RandomizedSearchCV(clf, param_dist, n_iter=n_iter, scoring="roc_auc",
                            cv=3, n_jobs=2, random_state=RANDOM_STATE)
    if sample_weight is not None:
        rs.fit(X, y, sample_weight=sample_weight)
    else:
        rs.fit(X, y)
    return rs.best_params_


def main():
    t0 = time.time()
    codes = eligible_codes()
    print(f"[1] 股票数: {len(codes)}")

    print("[2] 构建特征面板(含原始amount/ret用于拥挤度)...")
    panels = [build_stock_panel(c) for c in codes]
    panel = pd.concat([p for p in panels if p is not None], ignore_index=True)
    print(f"    原始面板: {len(panel):,} 行 | "
          f"{panel['date'].min().date()}~{panel['date'].max().date()}")

    # 排序打分标签（每 horizon 截面前30% = 1）
    for hz in HORIZONS:
        panel[f"cls_{hz}"] = panel.groupby("date")[f"fwd_ret_{hz}"].transform(
            lambda s: (s.rank(pct=True) >= 0.7).astype(int))

    print("[3] 预处理：市场环境(原始) → 拥挤度/筹码(原始) → Winsorize → 截面Z-score → 交互")
    env = market_env(panel)                                   # 原始特征算市场环境
    base_crowd, crowd_feat = build_crowding(panel, env)       # 原始特征算拥挤度
    panel = panel.drop(columns=["_amount", "_ret"], errors="ignore")

    # 筹码结构衍生(§5.1 PR×CB 交互, §5.2 短期乖离超95%分位惩罚)，用原始 chip 值算
    panel["chip_pr_x_cb"] = panel["chip_pr"] * panel["chip_cb"]
    _pct = panel.groupby("code")["chip_cb_short"].transform(
        lambda s: s.rolling(PCT_WIN, min_periods=250).apply(
            lambda w: float((w < w[-1]).mean()), raw=True))
    panel["chip_short_pen"] = np.where(_pct > 0.95, -1.0, 0.0)
    CHIP_ALL = CHIP_FEATS + ["chip_pr_x_cb", "chip_short_pen"]
    panel["_chip_cc_raw"] = panel["chip_cc"]                  # 留作训练样本权重(§5.2)

    winsorize(panel, ALPHA_FEATS + CHIP_ALL)
    cross_section_zscore(panel, ALPHA_FEATS + CHIP_ALL)       # per-stock Alpha+筹码 截面标准化
    panel, FEATS = add_interactions(panel, env)               # 交互 = z-score Alpha/筹码 × 市场环境
    winsorize(panel, [c for c in FEATS if c.startswith("ix_")])

    # 拥挤度特征并入(市场级日频)。用 int64 纳秒历元 key + .map() 逐列映射，
    # 规避 pandas 的 merge 对齐怪癖与 reset_index 列名依赖（已验证 key 命中率 100%）。
    _kdate = panel["date"].astype("int64").values
    _cf = crowd_feat.copy()
    _cf.index = _cf.index.astype("int64")
    _cf_dates = _cf.index
    CROWD_FEATS = list(crowd_feat.columns)
    for col in CROWD_FEATS:
        panel[col] = pd.Series(_kdate).map(dict(zip(_cf_dates, _cf[col]))).values
    FEATS = FEATS + CROWD_FEATS
    CHIP_DERIVED = ["chip_pr_x_cb", "chip_short_pen"]
    FEATS = FEATS + CHIP_DERIVED
    # 去掉全程全 NaN 的列(如部分样本 roe/current_ratio 全缺)；其余 NaN 交由 XGBoost missing 机制处理(v2 同)。
    FEATS = [c for c in FEATS if not panel[c].isna().all()]
    CORE_CROWD = [c for c in ["crowd_mom", "crowd_liq", "pca_absorp"] if c in FEATS]
    CHIP_BASE = [c for c in CHIP_FEATS if c in FEATS]
    DROP = CORE_CROWD + CHIP_BASE   # 丢弃预热期/无效行(其余 NaN 交 XGBoost)
    print(f"    总特征数: {len(FEATS)} (Alpha {len(ALPHA_FEATS)} + 市场环境 {len(env.columns)-1} "
          f"+ 交互 {len(FEATS)-len(ALPHA_FEATS)-(len(env.columns)-1)-len(CROWD_FEATS)} "
          f"+ 拥挤度 {len(CROWD_FEATS)})")
    print(f"    拥挤度基础指标相关性(§3.4 多重共线性检查):")
    print(base_crowd.corr().round(3).to_string())

    folds = wfa_folds(panel["date"])
    print(f"[4] WFA 折数: {len(folds)}")
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        print(f"    折{i}: IS {is_s.date()}~{is_e.date()} | OOS {oos_s.date()}~{oos_e.date()}")

    rec = []
    last_imp = None
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        is_df = panel[(panel["date"] >= is_s) & (panel["date"] < is_e)]
        oos_df = panel[(panel["date"] >= oos_s) & (panel["date"] < oos_e)]
        is_df = is_df.dropna(subset=DROP)   # 丢弃拥挤度/筹码预热期行；其余 NaN 交 XGBoost
        oos_df = oos_df.dropna(subset=DROP)
        if len(is_df) < 500 or len(oos_df) < 100:
            print(f"    折{i}: 样本不足跳过"); continue
        Xis = is_df[FEATS]
        yis = is_df["cls_5"]
        Xte = oos_df[FEATS]
        # 筹码成本集中度↑ → 样本权重↑(§5.2)：成本越集中信号越可信，最高 ~11 倍
        cc = is_df["_chip_cc_raw"].fillna(0.0)
        sw_is = 1.0 + np.clip(cc, 0.0, 10.0)
        best = hp_search(Xis, yis, sample_weight=sw_is)      # [4] IS内调参(含样本权重)
        print(f"    折{i}: 最佳参数 {best}")
        probas = {}
        for hz in HORIZONS:
            m = XGBClassifier(n_estimators=300, nthread=8, eval_metric="auc",
                              random_state=RANDOM_STATE, use_label_encoder=False, **best)
            m.fit(Xis, is_df[f"cls_{hz}"], sample_weight=sw_is)
            probas[hz] = m.predict_proba(Xte)[:, 1]
        fused = np.mean([probas[hz] for hz in HORIZONS], axis=0)
        ytrue = {hz: oos_df[f"fwd_ret_{hz}"].values for hz in HORIZONS}
        ycls = {hz: oos_df[f"cls_{hz}"].values for hz in HORIZONS}
        auc5 = roc_auc_score(ycls[5], probas[5]) if len(set(ycls[5])) > 1 else np.nan
        ic5 = pd.Series(probas[5]).corr(pd.Series(ytrue[5]), method="spearman")
        row = {"fold": i, "auc_single5": auc5, "ic_single5": ic5}
        for hz in HORIZONS:
            auc_f = roc_auc_score(ycls[hz], fused) if len(set(ycls[hz])) > 1 else np.nan
            ic_f = pd.Series(fused).corr(pd.Series(ytrue[hz]), method="spearman")
            row[f"auc_fuse_{hz}"] = auc_f
            row[f"ic_fuse_{hz}"] = ic_f
        rec.append(row)
        m5 = XGBClassifier(n_estimators=300, nthread=8, eval_metric="auc",
                           random_state=RANDOM_STATE, use_label_encoder=False, **best)
        m5.fit(Xis, yis, sample_weight=sw_is)
        # —— SHAP 持久化(供后续解释性分析, 规避"无模型可解释"漏洞) ——
        # 存 booster(精确 TreeExplainer 需它)；存该折 OOS 子样本(<=3000行, 规避全量 TreeExplainer 不可行)
        import json
        m5.get_booster().save_model(f"{OUT}/booster_fold{i}.json")
        oos_sub = oos_df.sample(min(3000, len(oos_df)), random_state=RANDOM_STATE)
        oos_sub[FEATS + ["cls_5", "fwd_ret_5", "code", "date"]].to_parquet(
            f"{OUT}/shap_data_fold{i}.parquet")
        json.dump(FEATS, open(f"{OUT}/feats_v4full.json", "w"))
        # xgboost 3.x 默认 feature_names=None → get_score 返回 f0..fN；
        # 映射回 FEATS 列名，使拥挤度特征可读。
        _raw = m5.get_booster().get_score(importance_type="gain")
        last_imp = {}
        for _k, _v in _raw.items():
            if _k.startswith("f") and _k[1:].isdigit() and int(_k[1:]) < len(FEATS):
                last_imp[FEATS[int(_k[1:])]] = _v
            else:
                last_imp[_k] = _v
        print(f"    折{i}: 单周期5日 AUC={auc5:.3f}/IC={ic5:+.4f} | "
              f"融合 AUC=[{row['auc_fuse_5']:.3f},{row['auc_fuse_20']:.3f},{row['auc_fuse_60']:.3f}] "
              f"IC=[{row['ic_fuse_5']:+.4f},{row['ic_fuse_20']:+.4f},{row['ic_fuse_60']:+.4f}]")

    rec_df = pd.DataFrame(rec)
    print(f"\n[5] 汇总（均值）:")
    print(rec_df[[c for c in rec_df.columns if c != 'fold']].mean().round(4).to_string())

    imp_df = pd.DataFrame({"gain": last_imp}).sort_values("gain", ascending=False)
    print("\n[6] 因子重要性 Top20 (by Gain):")
    print(imp_df.head(20).to_string())
    # 拥挤度/筹码 相关特征的重要性汇总
    crowd_mask = [x for x in imp_df.index if any(x.startswith(c) for c in CROWD_FEATS[:3])]
    chip_mask = [x for x in imp_df.index if x.startswith("chip_") or x.startswith("ix_") and "chip" in x]
    print(f"\n[7] 拥挤度特征总Gain占比: "
          f"{imp_df.loc[crowd_mask,'gain'].sum() / imp_df['gain'].sum():.3f}")
    print(f"[7] 筹码结构特征总Gain占比: "
          f"{imp_df.loc[[x for x in imp_df.index if x.startswith('chip_')],'gain'].sum() / imp_df['gain'].sum():.3f}")

    # ── 落盘 ──
    lines = ["# XGBoost 因子挖掘 · WFA 原型 v4 全量版（v3 + 筹码结构维度，全市场）", "",
             f"- 样本: {len(codes)} 只 | 面板: {len(panel):,} 行 | 特征: {len(FEATS)} 个",
             f"- Alpha {len(ALPHA_FEATS)} + 市场环境 {len(env.columns)-1} + 交互 "
             f"{len(FEATS)-len(ALPHA_FEATS)-len(CHIP_FEATS)-(len(env.columns)-1)-len(CROWD_FEATS)-len(CHIP_DERIVED)} "
             f"+ 拥挤度 {len(CROWD_FEATS)} + 筹码 {len(CHIP_FEATS)+len(CHIP_DERIVED)}",
             f"- 拥挤度实现: §1.1 交易行为(成交额代理, mom/liq) + §1.4 PCA吸收比率；",
             f"  §1.2 估值价差(需PE)、§1.3 机构持仓(需基金持仓) 数据湖缺失→跳过",
             f"- 筹码结构: §2 VWAP中心三角分布递推 + §4 PR/CC/CB/短期CB；",
             f"  §5.1 PR×CB 交互、§5.2 短期乖离>95%惩罚 + CC 作训练样本权重(无tick/换手→代理)",
             f"- 特征展开 §2: 原始值/历史分位数/变化率/宏观交互/惩罚项",
             f"- 标签: 排序打分(前30%) × 3周期(5/20/60日) | WFA 共 {len(folds)} 折",
             f"- 调参: 每折 IS 内 RandomizedSearchCV(cv=3), OOS 全程隔离", "",
             "## 各折结果", ""]
    for _, r in rec_df.iterrows():
        lines.append(f"- 折{r['fold']}: 单周期5日 AUC={r['auc_single5']:.3f} | "
                     f"融合 AUC=[{r['auc_fuse_5']:.3f},{r['auc_fuse_20']:.3f},{r['auc_fuse_60']:.3f}] "
                     f"IC=[{r['ic_fuse_5']:+.4f},{r['ic_fuse_20']:+.4f},{r['ic_fuse_60']:+.4f}]")
    lines += ["", "## 汇总均值", ""]
    for c in rec_df.columns:
        if c == "fold":
            continue
        lines.append(f"- {c}: {rec_df[c].mean():.4f}")
    lines += ["", "## 因子重要性 Top20 (by Gain)", ""]
    for name, row in imp_df.head(20).iterrows():
        tag = "【拥挤度】" if any(name.startswith(c) for c in CROWD_FEATS[:3]) else ""
        lines.append(f"- {name}: gain={row['gain']:.3f} {tag}")
    lines += ["", "## 拥挤度基础指标相关性(§3.4)", ""]
    for a in base_crowd.columns:
        for b in base_crowd.columns:
            if a < b:
                lines.append(f"- corr({a},{b}) = {base_crowd[a].corr(base_crowd[b]):+.3f}")
    lines += ["", "## 特征Gain占比(末折重要性)", "",
              f"- 拥挤度: {imp_df.loc[crowd_mask,'gain'].sum() / imp_df['gain'].sum():.3f}",
              f"- 筹码结构: {imp_df.loc[[x for x in imp_df.index if x.startswith('chip_')],'gain'].sum() / imp_df['gain'].sum():.3f}"]
    lines += ["", "## 结论", "",
              "在 v2(交互+多周期融合+IS调参) 基础上先加因子拥挤度(v3)，再加筹码结构维度(v4)；",
              "对比 v1(AUC≈0.528)、v2 与 v3，观察两个新维度的边际贡献。",
              "IC 略负属正常(原始因子含噪声)；重点看 AUC 是否提升及拥挤度/筹码特征重要性。"]
    out = f"{OUT}/proto_v4_full_results.md"
    open(out, "w").write("\n".join(lines))
    print(f"\n[8] 结果已写入: {out}  总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
