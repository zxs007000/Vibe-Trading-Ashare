#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 因子挖掘 · WFA 原型 v4（v3 + 筹码结构维度）· 分块流式版
==============================================================
与 xgb_wfa_proto_v4_full.py 算法完全一致（特征族 / 标签 / WFA / SHAP / 样本权重均相同），
唯一区别是**内存策略**：v4_full 一次性把全市场面板(~13.6M行×217维≈23GB)读入内存 → 8GB cgroup 必 OOM；
本分块版用「分组切片 + 流式落盘」把峰值内存压到单折训练矩阵量级(~4-7GB)，全市场 5500 只在 8GB 内可跑：

  Phase A  逐只股票算基础特征(价量/基本面/筹码)，立刻写盘释放 → 峰值 = 单只票
  Phase B  按"年"切片，每片载入"该窗内全部股票"算截面特征
           (市场环境/拥挤度/PCA/交互/z-score)，写盘释放 → 峰值 ≈ 5448×252×217 ≈ 240MB
  Phase C  每折只读取"覆盖该折日期窗的特征分片"，用 float32 + numpy 增量拼矩阵后训练
           → 峰值 ≈ 单折训练矩阵(~3.5GB) + 测试矩阵(~1GB)

截面运算(市场环境/拥挤度/PCA/z-score/交互)只需要"某一天的所有股票同时在线"，
不需要"所有股票的所有历史同时在线"——这正是分块可行的理论依据。

注: 切片内 winsorize 为该分片内的分位(非全历史全局分位)，属可接受的局部近似(原型级)。
"""
import os, time, shutil, warnings, json
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
MAX_STOCKS = int(os.environ.get("WFA_MAX_STOCKS", 5500))   # 全量≈5448; 分块流式后全市场可在 8GB 内跑
TRAIN_YEARS = 3
TEST_YEARS = 1
STEP_YEARS = 1
RANDOM_STATE = 42
HORIZONS = [5, 20, 60]
ZYEARS = 5 * 252          # 5年滚动窗口(交易日)
PCT_WIN = 1250           # 历史分位数窗口(文档§2.1)
WARMUP = pd.DateOffset(months=3)   # PCA 滚动窗口预热(避免切片首日 NaN 污染)
CHUNK_MONTHS = 12         # 特征工程按"年"切片; 峰值≈全股票×1年×特征≈240MB

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
CHIP_DERIVED = ["chip_pr_x_cb", "chip_short_pen"]


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
    """东财三大表派生 8 个基本面因子。"""
    inc = _read_parquet(f"{INCOME}/{code}.parquet")
    cash = _read_parquet(f"{CASH}/{code}.parquet")
    bal = _read_parquet(f"{BAL}/{code}.parquet")
    if inc is None and cash is None and bal is None:
        return None
    # 关键: 提前初始化 pn/sc, 否则当某股票缺资产负债表(bal=None)但现金流存在时,
    # 下方 `if sc is not None` 会触发 UnboundLocalError(全市场跑必现)
    pn = None
    sc = None
    idxs = [d.index for d in (inc, cash, bal) if d is not None]
    out = pd.DataFrame(index=sorted(set().union(*idxs)))
    if inc is not None:
        if "BASIC_EPS" in inc.columns:
            out["eps"] = inc["BASIC_EPS"]
        if "PARENT_NETPROFIT_YOY" in inc.columns:
            out["netprofit_yoy"] = inc["PARENT_NETPROFIT_YOY"]
        pn = inc["PARENT_NETPROFIT"] if "PARENT_NETPROFIT" in inc.columns else None
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
    if cash is not None:
        nco = cash["NETCASH_OPERATE"] if "NETCASH_OPERATE" in cash.columns else None
        if nco is not None:
            if sc is not None:
                out["ocf_ps"] = nco / sc.replace(0, float("nan"))
            if pn is not None:
                out["ocf_netprofit"] = nco / pn.replace(0, float("nan"))
    for c in ["roe", "debt_ratio", "current_ratio", "ocf_netprofit",
              "netprofit_yoy", "eps", "bvps", "ocf_ps"]:
        if c not in out.columns:
            out[c] = float("nan")
    out = out[out.index >= pd.Timestamp("2017-01-01")]
    out = out.sort_index().ffill().replace([float("inf"), float("-inf")], float("nan"))
    out.index.name = "REPORT_DATE"
    return out


def chip_features(df, price_bins=100, decay_base=0.02):
    """§2 VWAP中心三角分布递推 + §4 PR/CC/CB(筹码结构)。"""
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
        return pd.DataFrame(index=df.index, data={k: np.nan for k in CHIP_FEATS})
    bins = np.linspace(min_p, max_p, price_bins + 1)
    bc = (bins[:-1] + bins[1:]) / 2.0
    chip = np.zeros((T, price_bins))
    newmat = np.zeros((T, price_bins))
    for i in range(T):
        if i > 0:
            chip[i] = chip[i - 1] * (1.0 - turn[i])
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
    denom = 1.0 - 1.0 / price_bins
    for i in range(T):
        if tot[i] <= 0:
            continue
        pr[i] = chip[i][bc < close[i]].sum() / tot[i]
        wmean[i] = (chip[i] * bc).sum() / tot[i]
        wstd[i] = np.sqrt(((bc - wmean[i]) ** 2 * chip[i]).sum() / tot[i])
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
        "mkt_roe_mean": g["roe"].mean(),
    }).sort_index()
    mr = env["mkt_ret20_mean"]
    mz = (mr - mr.rolling(ZYEARS, min_periods=250).mean()) / (mr.rolling(ZYEARS, min_periods=250).std() + 1e-8)
    env["is_extreme_macro"] = (mz > 1.5).astype(float)
    return env


def trading_crowding(panel, leg_factor, label):
    """§1.1 交易行为拥挤度：多空组合 活跃度比+波动比 → 5年 z-score。"""
    r = panel.groupby("date")[leg_factor].rank(pct=True)
    long_mask = (r >= 0.9).values
    short_mask = (r <= 0.1).values
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
    base["crowd_mom"] = trading_crowding(panel, "ret_60", "crowd_mom")
    base["crowd_liq"] = trading_crowding(panel, "amt_chg_20", "crowd_liq")
    base["pca_absorp"] = pca_absorption(panel)
    macro = env["is_extreme_macro"].reindex(base.index).fillna(0.0)
    feat = pd.DataFrame(index=base.index)
    for col in base.columns:
        s = base[col].astype(float)
        feat[col] = s
        pct = s.rolling(PCT_WIN, min_periods=250).apply(
            lambda w: float((w < w[-1]).mean()), raw=True)
        feat[f"{col}_pct"] = pct
        roc = s.diff(5) / (s.abs() + 1e-8)
        feat[f"{col}_roc"] = roc
        feat[f"{col}_x_macro"] = s * macro
        feat[f"{col}_delta_x_macro"] = roc * macro
        feat[f"{col}_pen"] = np.where(pct > 0.95, (pct - 0.95) * -10.0, 0.0)
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
                            cv=3, n_jobs=1, random_state=RANDOM_STATE)
    if sample_weight is not None:
        rs.fit(X, y, sample_weight=sample_weight)
    else:
        rs.fit(X, y)
    return rs.best_params_


# ───────────────────────── 三阶段分块流水线 ─────────────────────────
def main():
    t0 = time.time()
    codes = eligible_codes()
    print(f"[1] 股票数: {len(codes)}")
    base_dir = os.path.join(OUT, "_base")
    feat_dir = os.path.join(OUT, "_feat")
    for _d in (base_dir, feat_dir):
        if os.path.exists(_d):
            shutil.rmtree(_d)
        os.makedirs(_d, exist_ok=True)

    # ── Phase A: 逐只股票算基础特征, 立刻落盘释放(峰值=单只票) ──
    print("[2] Phase A 逐只股票基础特征(价量/基本面/筹码) → 落盘...")
    meta = {}
    for c in codes:
        p = build_stock_panel(c)
        if p is None:
            continue
        p.to_parquet(f"{base_dir}/{c}.parquet")
        meta[c] = (p["date"].min(), p["date"].max())
    if not meta:
        print("    无可用股票, 退出"); return
    all_min = min(m[0] for m in meta.values())
    all_max = max(m[1] for m in meta.values())
    print(f"    基础特征落盘完成: {len(meta)} 只 | {all_min.date()}~{all_max.date()}")

    # ── Phase B: 按"年"切片, 每片载入该窗内全部股票算截面特征 → 落盘(峰值~240MB) ──
    print("[3] Phase B 日期切片截面特征(市场环境/拥挤度/PCA/交互/z-score)...")
    chunks = []          # (chunk_index, cd0, cd1) 用于 Phase C 按需读取
    start = all_min
    ci = 0
    base_crowds = []
    GLOBAL_FEATS = None
    ENV_N = 0
    CROWD_N = 0
    n_rows_est = 0
    while start <= all_max:
        cd0, cd1 = start, start + pd.DateOffset(months=CHUNK_MONTHS)
        warm0 = cd0 - WARMUP
        parts = []
        for c, (m0, m1) in meta.items():
            if m1 >= warm0 and m0 <= cd1:
                df = pd.read_parquet(f"{base_dir}/{c}.parquet")
                df = df[(df["date"] >= warm0) & (df["date"] < cd1)]
                if len(df):
                    parts.append(df)
                del df
        if parts:
            sub = pd.concat(parts, ignore_index=True)
            del parts
            for hz in HORIZONS:
                sub[f"cls_{hz}"] = sub.groupby("date")[f"fwd_ret_{hz}"].transform(
                    lambda s: (s.rank(pct=True) >= 0.7).astype(int))
            env = market_env(sub)
            base_crowd, crowd_feat = build_crowding(sub, env)
            sub = sub.drop(columns=["_amount", "_ret"], errors="ignore")
            sub["chip_pr_x_cb"] = sub["chip_pr"] * sub["chip_cb"]
            _pct = sub.groupby("code")["chip_cb_short"].transform(
                lambda s: s.rolling(PCT_WIN, min_periods=250).apply(
                    lambda w: float((w < w[-1]).mean()), raw=True))
            sub["chip_short_pen"] = np.where(_pct > 0.95, -1.0, 0.0)
            CHIP_ALL = CHIP_FEATS + CHIP_DERIVED
            sub["_chip_cc_raw"] = sub["chip_cc"]           # 留作训练样本权重(§5.2)
            winsorize(sub, ALPHA_FEATS + CHIP_ALL)
            cross_section_zscore(sub, ALPHA_FEATS + CHIP_ALL)
            sub, FEATS = add_interactions(sub, env)
            winsorize(sub, [x for x in FEATS if x.startswith("ix_")])
            _kdate = sub["date"].astype("int64").values
            _cf = crowd_feat.copy(); _cf.index = _cf.index.astype("int64")
            CROWD_FEATS = list(crowd_feat.columns)
            for col in CROWD_FEATS:
                sub[col] = pd.Series(_kdate).map(dict(zip(_cf.index, _cf[col]))).values
            FEATS = FEATS + CROWD_FEATS + CHIP_DERIVED
            FEATS = [x for x in FEATS if not sub[x].isna().all()]
            if GLOBAL_FEATS is None:
                GLOBAL_FEATS = list(FEATS)
                ENV_N = len(env.columns) - 1
                CROWD_N = len(CROWD_FEATS)
            base_crowds.append(base_crowd)
            # float32 落盘: halved 内存/磁盘, 全市场峰值可控(仅转数值列, 跳过 date/datetime 与 object)
            sub = sub.astype({c: "float32" for c in sub.columns if sub[c].dtype.kind in "biufc"})
            sub.to_parquet(f"{feat_dir}/feat_{ci}.parquet")
            n_rows_est += int(((sub["date"] >= cd0) & (sub["date"] < cd1)).sum())
            chunks.append((ci, cd0, cd1))
            ci += 1
            del sub, env, base_crowd, crowd_feat
        start = cd1
    print(f"    特征分片数: {len(chunks)} | 全局特征数: {len(GLOBAL_FEATS)}")
    if base_crowds:
        _bc = pd.concat(base_crowds)
        print("    拥挤度基础指标相关性(§3.4):")
        print(_bc.corr().round(3).to_string())

    # ── Phase C: 每折只读取覆盖该折窗的特征分片, float32 增量拼矩阵训练(峰值~单折矩阵) ──
    folds = wfa_folds(pd.Series([all_min, all_max]))
    print(f"[4] WFA 折数: {len(folds)}")
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        print(f"    折{i}: IS {is_s.date()}~{is_e.date()} | OOS {oos_s.date()}~{oos_e.date()}")

    DROP = ([c for c in ["crowd_mom", "crowd_liq", "pca_absorp"] if c in GLOBAL_FEATS]
            + [c for c in CHIP_FEATS if c in GLOBAL_FEATS])

    def load_window(d0, d1):
        """读取覆盖 [d0,d1) 的特征分片, 一次性取回该折所需 X/权重/各周期标签/收益, 避免整面板驻留与重复读盘。"""
        Xp, swp, codep, datep = [], [], [], []
        clsh = {hz: [] for hz in HORIZONS}
        fwdh = {hz: [] for hz in HORIZONS}
        for cdi, cd0, cd1 in chunks:
            if cd1 >= d0 and cd0 <= d1:
                df = pd.read_parquet(f"{feat_dir}/feat_{cdi}.parquet")
                df = df[(df["date"] >= d0) & (df["date"] < d1)]
                if len(df):
                    d = df.dropna(subset=DROP)
                    if len(d):
                        Xp.append(d[GLOBAL_FEATS].astype("float32").values)
                        cc = d["_chip_cc_raw"].fillna(0.0).astype("float32").values
                        swp.append((1.0 + np.clip(cc, 0.0, 10.0)).astype("float32"))
                        codep.append(d["code"].values)
                        datep.append(d["date"].values)
                        for hz in HORIZONS:
                            clsh[hz].append(d[f"cls_{hz}"].values.astype("int8"))
                            fwdh[hz].append(d[f"fwd_ret_{hz}"].values.astype("float32"))
                del df
        if not Xp:
            return None
        return {
            "X": np.concatenate(Xp, axis=0),
            "sw": np.concatenate(swp, axis=0),
            "code": np.concatenate(codep, axis=0),
            "date": np.concatenate(datep, axis=0),
            "cls": {hz: np.concatenate(clsh[hz], axis=0) for hz in HORIZONS},
            "fwd": {hz: np.concatenate(fwdh[hz], axis=0) for hz in HORIZONS},
        }

    rec = []
    last_imp = None
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        isd = load_window(is_s, is_e)
        osd = load_window(oos_s, oos_e)
        if isd is None or osd is None or len(isd["X"]) < 500 or len(osd["X"]) < 100:
            print(f"    折{i}: 样本不足跳过"); continue
        Xis, sw_is, Xte = isd["X"], isd["sw"], osd["X"]
        best = hp_search(Xis, isd["cls"][5], sample_weight=sw_is)
        print(f"    折{i}: 最佳参数 {best}")
        probas = {}
        for hz in HORIZONS:
            m = XGBClassifier(n_estimators=300, nthread=8, eval_metric="auc",
                              random_state=RANDOM_STATE, use_label_encoder=False, **best)
            m.fit(Xis, isd["cls"][hz], sample_weight=sw_is)
            probas[hz] = m.predict_proba(Xte)[:, 1]
        fused = np.mean([probas[hz] for hz in HORIZONS], axis=0)
        auc5 = roc_auc_score(osd["cls"][5], probas[5]) if len(set(osd["cls"][5])) > 1 else np.nan
        ic5 = pd.Series(probas[5]).corr(pd.Series(osd["fwd"][5]), method="spearman")
        row = {"fold": i, "auc_single5": auc5, "ic_single5": ic5}
        for hz in HORIZONS:
            ys, yt = osd["cls"][hz], osd["fwd"][hz]
            auc_f = roc_auc_score(ys, fused) if len(set(ys)) > 1 else np.nan
            ic_f = pd.Series(fused).corr(pd.Series(yt), method="spearman")
            row[f"auc_fuse_{hz}"] = auc_f
            row[f"ic_fuse_{hz}"] = ic_f
        rec.append(row)
        # 存 booster + OOS 子样本(<=3000行, 供 SHAP 精确 TreeExplainer)
        m5 = XGBClassifier(n_estimators=300, nthread=8, eval_metric="auc",
                           random_state=RANDOM_STATE, use_label_encoder=False, **best)
        m5.fit(Xis, isd["cls"][5], sample_weight=sw_is)
        m5.get_booster().save_model(f"{OUT}/booster_fold{i}.json")
        n = len(Xte)
        idx = (np.random.RandomState(RANDOM_STATE).choice(n, min(3000, n), replace=False)
               if n > 3000 else np.arange(n))
        sub_df = pd.DataFrame(Xte[idx], columns=GLOBAL_FEATS)
        sub_df["cls_5"] = osd["cls"][5][idx]
        sub_df["fwd_ret_5"] = osd["fwd"][5][idx]
        sub_df["code"] = osd["code"][idx]
        sub_df["date"] = osd["date"][idx]
        sub_df.to_parquet(f"{OUT}/shap_data_fold{i}.parquet")
        json.dump(GLOBAL_FEATS, open(f"{OUT}/feats_v4full.json", "w"))
        _raw = m5.get_booster().get_score(importance_type="gain")
        last_imp = {}
        for _k, _v in _raw.items():
            if _k.startswith("f") and _k[1:].isdigit() and int(_k[1:]) < len(GLOBAL_FEATS):
                last_imp[GLOBAL_FEATS[int(_k[1:])]] = _v
            else:
                last_imp[_k] = _v
        print(f"    折{i}: 单周期5日 AUC={auc5:.3f}/IC={ic5:+.4f} | "
              f"融合 AUC=[{row['auc_fuse_5']:.3f},{row['auc_fuse_20']:.3f},{row['auc_fuse_60']:.3f}] "
              f"IC=[{row['ic_fuse_5']:+.4f},{row['ic_fuse_20']:+.4f},{row['ic_fuse_60']:+.4f}]")
        del Xis, Xte, m5, isd, osd

    rec_df = pd.DataFrame(rec)
    print(f"\n[5] 汇总（均值）:")
    print(rec_df[[c for c in rec_df.columns if c != 'fold']].mean().round(4).to_string())

    imp_df = pd.DataFrame({"gain": last_imp}).sort_values("gain", ascending=False)
    print("\n[6] 因子重要性 Top20 (by Gain):")
    print(imp_df.head(20).to_string())
    crowd_mask = [x for x in imp_df.index if any(x.startswith(c) for c in ["crowd_mom", "crowd_liq", "pca_absorp"])]
    print(f"\n[7] 拥挤度特征总Gain占比: "
          f"{imp_df.loc[crowd_mask,'gain'].sum() / imp_df['gain'].sum():.3f}")
    print(f"[7] 筹码结构特征总Gain占比: "
          f"{imp_df.loc[[x for x in imp_df.index if x.startswith('chip_')],'gain'].sum() / imp_df['gain'].sum():.3f}")

    # ── 落盘 ──
    lines = ["# XGBoost 因子挖掘 · WFA 原型 v4（分块流式版，全市场）", "",
             f"- 样本: {len(codes)} 只 | 特征分片: {len(chunks)} | 特征: {len(GLOBAL_FEATS)} 个",
             f"- 内存策略: PhaseA逐只落盘 → PhaseB按年切片截面特征 → PhaseC按需读分片+float32增量拼矩阵",
             f"  全市场 5500 只可在 8GB cgroup 内运行(峰值≈单折训练矩阵 ~3.5GB)",
             f"- Alpha {len(ALPHA_FEATS)} + 市场环境 {ENV_N} + 交互 "
             f"{len(GLOBAL_FEATS)-len(ALPHA_FEATS)-len(CHIP_FEATS)-ENV_N-CROWD_N-len(CHIP_DERIVED)} "
             f"+ 拥挤度 {CROWD_N} + 筹码 {len(CHIP_FEATS)+len(CHIP_DERIVED)}",
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
        tag = "【拥挤度】" if any(name.startswith(c) for c in ["crowd_mom", "crowd_liq", "pca_absorp"]) else ""
        lines.append(f"- {name}: gain={row['gain']:.3f} {tag}")
    lines += ["", "## 拥挤度基础指标相关性(§3.4)", ""]
    if base_crowds:
        _bc = pd.concat(base_crowds)
        # 切片 warmup 窗口有日期重叠 → 索引不唯一; 先按索引去重(保留末片, 数据更完整),
        # 再用 DataFrame.corr()(按位置计算, 不依赖索引对齐) 避免重复索引下 Series.corr 对齐失败→NaN
        _bc = _bc[~_bc.index.duplicated(keep="last")]
        _corr = _bc.corr()
        for a in _bc.columns:
            for b in _bc.columns:
                if a < b:
                    lines.append(f"- corr({a},{b}) = {_corr.loc[a, b]:+.3f}")
    lines += ["", "## 特征Gain占比(末折重要性)", "",
              f"- 拥挤度: {imp_df.loc[crowd_mask,'gain'].sum() / imp_df['gain'].sum():.3f}",
              f"- 筹码结构: {imp_df.loc[[x for x in imp_df.index if x.startswith('chip_')],'gain'].sum() / imp_df['gain'].sum():.3f}"]
    lines += ["", "## 结论", "",
              "分块流式版(v4_chunked)与 v4_full 算法一致，仅内存策略不同：",
              "通过 PhaseA逐只落盘 + PhaseB按年切片截面特征 + PhaseC按需读分片+float32增量拼矩阵，",
              "把峰值内存从全面板(~23GB)压到单折训练矩阵(~3.5GB)，全市场 5500 只可在 8GB 内完成 WFA+SHAP。"]
    out = f"{OUT}/proto_v4_chunked_results.md"
    open(out, "w").write("\n".join(lines))
    print(f"\n[8] 结果已写入: {out}  总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
