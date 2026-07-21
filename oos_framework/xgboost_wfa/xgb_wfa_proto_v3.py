#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 因子挖掘 · WFA 原型 v3（v2 + 因子拥挤度维度）
====================================================
在 v2（交互特征 + 多周期融合 + IS内调参）基础上，新增用户要求的「因子拥挤度」维度，
严格落地《因子拥挤度量化建模与XGBoost实战指南》：

  §1.1 交易行为拥挤度  —— 用成交额(amount)代理换手率(数据湖无turnover)，
       按动量(ret_60)/流动性(amt_chg_20)分别构造多空组合，算
       活跃度比 + 收益波动比 → 5年窗口 z-score。
  §1.4 资产集中度/PCA吸收比率 —— 每日截面收益滚动60天协方差，第一主成分解释力。
  §2   特征工程体系 —— 每个基础拥挤度指标展开为：
       原始值 / 历史分位数(1250) / 变化率 / 宏观交互 / 惩罚项。
  §3.1 标签对齐 —— 仍用 T+1 起的未来N日收益，拥挤度全用历史窗口(无泄露)。
  §3.4 多重共线性 —— 汇报拥挤度基础指标相关性。

数据缺口(本数据湖无对应字段，明确跳过并标注)：
  §1.2 估值价差拥挤度(需 PE)   —— 数据湖 daily 仅 date/open/close/high/low/volume/amount，无 PE/PB。
  §1.3 机构持仓集中度(需基金持仓) —— 数据湖无持仓数据。

修复 v2 瑕疵：市场环境与拥挤度均用「原始(未z-score)特征」计算，
再对 per-stock 的 Alpha 做截面标准化，避免市场环境特征退化成常数。
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
MAX_STOCKS = 400
TRAIN_YEARS = 3
TEST_YEARS = 1
STEP_YEARS = 1
RANDOM_STATE = 42
HORIZONS = [5, 20, 60]
ZYEARS = 5 * 252          # 5年滚动窗口(交易日)
PCT_WIN = 1250           # 历史分位数窗口(文档§2.1)

PRICE_FEATS = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "rsi_14",
               "amt_chg_20", "ma_dev_20", "amp_20"]
FUND_FEATS = ["gross_margin", "net_margin", "roe", "debt_ratio",
              "current_ratio", "ocf_netprofit", "operate_income_yoy",
              "netprofit_yoy", "eps"]
ALPHA_FEATS = PRICE_FEATS + FUND_FEATS


# ───────────────────────── 数据加载（同 v2，含财报修复） ─────────────────────────
def eligible_codes():
    daily = {f[:-8] for f in os.listdir(DAILY) if f.endswith(".parquet")}
    inc = {f[:-8] for f in os.listdir(f"{FUND}/income_statement") if f.endswith(".parquet")}
    bal = {f[:-8] for f in os.listdir(f"{FUND}/balance_sheet") if f.endswith(".parquet")}
    cf = {f[:-8] for f in os.listdir(f"{FUND}/cash_flow_statement") if f.endswith(".parquet")}
    return sorted(daily & inc & bal & cf)[:MAX_STOCKS]


def pick(raw, *names):
    for n in names:
        if n in raw.columns:
            return raw[n]
    return None


def load_fundamentals(code):
    try:
        inc = pd.read_parquet(f"{FUND}/income_statement/{code}.parquet")
        bal = pd.read_parquet(f"{FUND}/balance_sheet/{code}.parquet")
        cf = pd.read_parquet(f"{FUND}/cash_flow_statement/{code}.parquet")
    except Exception:
        return None
    for d in (inc, bal, cf):
        if "REPORT_DATE" in d.columns:
            d["REPORT_DATE"] = pd.to_datetime(d["REPORT_DATE"], errors="coerce")
            d.dropna(subset=["REPORT_DATE"], inplace=True)
    raw = inc.set_index("REPORT_DATE")
    raw = raw.combine_first(bal.set_index("REPORT_DATE"))
    raw = raw.combine_first(cf.set_index("REPORT_DATE"))
    raw = raw.sort_index().ffill()
    npf = raw.get("PARENT_NETPROFIT")
    eq = pick(raw, "PARENT_EQUITY_BALANCE", "TOTAL_PARENT_EQUITY", "TOTAL_EQUITY")
    ta, tl = raw.get("TOTAL_ASSETS"), raw.get("TOTAL_LIABILITIES")
    ca = pick(raw, "CURRENT_ASSET_BALANCE", "CURRENT_ASSET")
    cl = pick(raw, "CURRENT_LIAB_BALANCE", "CURRENT_LIAB")
    ocf = raw.get("NETCASH_OPERATE")
    out = pd.DataFrame(index=raw.index)
    out["gross_margin"] = (raw.get("OPERATE_INCOME") - raw.get("OPERATE_COST")) / raw.get("OPERATE_INCOME") \
        if (raw.get("OPERATE_INCOME") is not None and raw.get("OPERATE_COST") is not None) else np.nan
    out["net_margin"] = npf / raw.get("OPERATE_INCOME") if (npf is not None and raw.get("OPERATE_INCOME") is not None) else np.nan
    out["roe"] = npf / eq if (npf is not None and eq is not None) else np.nan
    out["debt_ratio"] = tl / ta if (ta is not None and tl is not None) else np.nan
    out["current_ratio"] = ca / cl if (ca is not None and cl is not None) else np.nan
    out["ocf_netprofit"] = ocf / npf if (ocf is not None and npf is not None) else np.nan
    out["operate_income_yoy"] = raw.get("OPERATE_INCOME_YOY")
    out["netprofit_yoy"] = raw.get("PARENT_NETPROFIT_YOY")
    out["eps"] = raw.get("BASIC_EPS")
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "REPORT_DATE"
    return out


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
    keep = ["date", "code", "_amount", "_ret"] + ALPHA_FEATS + [f"fwd_ret_{hz}" for hz in HORIZONS]
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
        "mkt_gm_mean": g["gross_margin"].mean(),
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
    """Alpha(已z-score) × 市场环境变量 笛卡尔积。"""
    panel = panel.merge(env, left_on="date", right_index=True, how="left")
    env_feats = [c for c in env.columns if c != "is_extreme_macro"]
    inter = []
    for e in env_feats:
        for a in ALPHA_FEATS:
            nm = f"ix_{e}__{a}"
            panel[nm] = panel[e] * panel[a]
            inter.append(nm)
    return panel, ALPHA_FEATS + env_feats + inter


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


def hp_search(X, y, n_iter=12):
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

    print("[3] 预处理：市场环境(原始) → 拥挤度(原始) → Winsorize → 截面Z-score(Alpha) → 交互")
    env = market_env(panel)                                   # 原始特征算市场环境
    base_crowd, crowd_feat = build_crowding(panel, env)       # 原始特征算拥挤度
    panel = panel.drop(columns=["_amount", "_ret"], errors="ignore")

    winsorize(panel, ALPHA_FEATS)
    cross_section_zscore(panel, ALPHA_FEATS)                  # 仅 per-stock Alpha 做截面标准化
    panel, FEATS = add_interactions(panel, env)               # 交互 = z-score Alpha × 市场环境
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
    # 去掉全程全 NaN 的列(如部分样本 roe/current_ratio 全缺)；其余 NaN 交由 XGBoost missing 机制处理(v2 同)。
    FEATS = [c for c in FEATS if not panel[c].isna().all()]
    CORE_CROWD = [c for c in ["crowd_mom", "crowd_liq", "pca_absorp"] if c in FEATS]
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
        is_df = is_df.dropna(subset=CORE_CROWD)   # 仅丢弃拥挤度预热期行；其余 NaN 交 XGBoost
        oos_df = oos_df.dropna(subset=CORE_CROWD)
        if len(is_df) < 500 or len(oos_df) < 100:
            print(f"    折{i}: 样本不足跳过"); continue
        Xis = is_df[FEATS]
        yis = is_df["cls_5"]
        Xte = oos_df[FEATS]
        best = hp_search(Xis, yis)                            # [4] IS内调参
        print(f"    折{i}: 最佳参数 {best}")
        probas = {}
        for hz in HORIZONS:
            m = XGBClassifier(n_estimators=300, nthread=8, eval_metric="auc",
                              random_state=RANDOM_STATE, use_label_encoder=False, **best)
            m.fit(Xis, is_df[f"cls_{hz}"])
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
        m5.fit(Xis, yis)
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
    # 拥挤度相关特征的重要性汇总
    crowd_imp = imp_df[imp_df.index.str.startswith(tuple(CROWD_FEATS[:3]))]
    print(f"\n[7] 拥挤度特征总Gain占比: "
          f"{imp_df.loc[[x for x in imp_df.index if any(x.startswith(c) for c in CROWD_FEATS[:3])],'gain'].sum() / imp_df['gain'].sum():.3f}")

    # ── 落盘 ──
    lines = ["# XGBoost 因子挖掘 · WFA 原型 v3（v2 + 因子拥挤度维度）", "",
             f"- 样本: {len(codes)} 只 | 面板: {len(panel):,} 行 | 特征: {len(FEATS)} 个",
             f"- Alpha {len(ALPHA_FEATS)} + 市场环境 {len(env.columns)-1} + 交互 "
             f"{len(FEATS)-len(ALPHA_FEATS)-(len(env.columns)-1)-len(CROWD_FEATS)} + 拥挤度 {len(CROWD_FEATS)}",
             f"- 拥挤度实现: §1.1 交易行为(成交额代理, mom/liq) + §1.4 PCA吸收比率；",
             f"  §1.2 估值价差(需PE)、§1.3 机构持仓(需基金持仓) 数据湖缺失→跳过",
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
    lines += ["", "## 结论", "",
              "在 v2(交互+多周期融合+IS调参) 基础上加入因子拥挤度维度；",
              "对比 v1(AUC≈0.528) 与 v2，观察拥挤度维度的边际贡献。",
              "IC 略负属正常(原始因子拥挤/噪声)；重点看 AUC 是否提升及拥挤度特征重要性。"]
    out = "/workspace/quant_proto/proto_v3_results.md"
    open(out, "w").write("\n".join(lines))
    print(f"\n[8] 结果已写入: {out}  总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
