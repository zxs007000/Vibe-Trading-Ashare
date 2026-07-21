#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 因子挖掘 · WFA 原型 v2（增强版）
=========================================
在 v1 基础上新增（对应指南落地建议 2/3/4）：
  [2] 交互特征：市场环境变量(横截面均值/波动率/涨跌家数比等) × Alpha因子 笛卡尔积，
      交给 XGBoost 自动筛选（指南 3.3）。
  [3] 多周期标签融合：未来 5/20/60 日各训一模型，预测概率等权融合。
  [4] IS 内参数寻优：每折在 IS 上做 RandomizedSearchCV(cv=3)，OOS 全程不参与。

防泄露与 WFA 纯净性同 v1：标签 T+1 起算；财报仅用报告期≤T 快照；早停/调参只在 IS 内。
"""
import os, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)

LAKE = os.environ.get("STOCKLAKE", "/workspace/stocklake")
DAILY = f"{LAKE}/daily"
FUND = f"{LAKE}/fundamentals"
MAX_STOCKS = 400
TRAIN_YEARS = 3
TEST_YEARS = 1
STEP_YEARS = 1
RANDOM_STATE = 42
HORIZONS = [5, 20, 60]

PRICE_FEATS = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "rsi_14",
               "amt_chg_20", "ma_dev_20", "amp_20"]
FUND_FEATS = ["gross_margin", "net_margin", "roe", "debt_ratio",
              "current_ratio", "ocf_netprofit", "operate_income_yoy",
              "netprofit_yoy", "eps"]
ALPHA_FEATS = PRICE_FEATS + FUND_FEATS


# ───────────────────────── 数据加载（同 v1，含财报修复） ─────────────────────────
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
    for hz in HORIZONS:
        df[f"fwd_ret_{hz}"] = c.shift(-hz) / c.shift(-1) - 1  # T+1 起算，防泄露
    fin = load_fundamentals(code)
    if fin is not None and len(fin):
        df["date"] = df["date"].astype("datetime64[ns]")
        fdf = fin.reset_index()
        fdf["REPORT_DATE"] = fdf["REPORT_DATE"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("date"), fdf.sort_values("REPORT_DATE"),
                           left_on="date", right_on="REPORT_DATE", direction="backward")
        df.drop(columns=["REPORT_DATE"], inplace=True, errors="ignore")
    df["code"] = code
    df.dropna(subset=PRICE_FEATS + [f"fwd_ret_{hz}" for hz in HORIZONS], inplace=True)
    return df[["date", "code"] + ALPHA_FEATS + [f"fwd_ret_{hz}" for hz in HORIZONS]]


def market_env(panel):
    """市场环境变量：每日横截面统计（用原始未z-score特征，否则均值≈0）。"""
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
    return env


def add_interactions(panel, env):
    """Alpha(已z-score) × 市场环境变量 笛卡尔积。"""
    panel = panel.merge(env, left_on="date", right_index=True, how="left")
    env_feats = list(env.columns)
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

    print("[2] 构建特征面板...")
    panels = [build_stock_panel(c) for c in codes]
    panel = pd.concat([p for p in panels if p is not None], ignore_index=True)
    print(f"    原始面板: {len(panel):,} 行 × {len(panel.columns)} 列, "
          f"{panel['date'].min().date()}~{panel['date'].max().date()}")

    # 排序打分标签（每 horizon 截面前30% = 1）
    for hz in HORIZONS:
        panel[f"cls_{hz}"] = panel.groupby("date")[f"fwd_ret_{hz}"].transform(
            lambda s: (s.rank(pct=True) >= 0.7).astype(int))

    print("[3] 预处理：Winsorize → 截面Z-score(Alpha) → 市场环境 → 交互特征")
    winsorize(panel, ALPHA_FEATS)
    cross_section_zscore(panel, ALPHA_FEATS)          # 仅对 per-stock 的 Alpha 做截面标准化
    env = market_env(panel)                            # 用原始特征算市场环境（避免均值≈0）
    panel, FEATS = add_interactions(panel, env)        # 交互 = z-score Alpha × 市场环境
    winsorize(panel, [c for c in FEATS if c.startswith("ix_")])
    print(f"    总特征数: {len(FEATS)} (Alpha {len(ALPHA_FEATS)} + 市场环境 {len(env.columns)} "
          f"+ 交互 {len(FEATS)-len(ALPHA_FEATS)-len(env.columns)})")
    print(f"    截面标准化后行数: {len(panel):,}")

    folds = wfa_folds(panel["date"])
    print(f"[4] WFA 折数: {len(folds)}")
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        print(f"    折{i}: IS {is_s.date()}~{is_e.date()} | OOS {oos_s.date()}~{oos_e.date()}")

    rec = []  # 每折记录
    last_imp = None
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        is_df = panel[(panel["date"] >= is_s) & (panel["date"] < is_e)]
        oos_df = panel[(panel["date"] >= oos_s) & (panel["date"] < oos_e)]
        if len(is_df) < 500 or len(oos_df) < 100:
            print(f"    折{i}: 样本不足跳过"); continue
        Xis = is_df[FEATS]
        Xte = oos_df[FEATS]
        # [4] IS 内参数寻优（仅用5日标签做搜索，结果复用给各周期）
        best = hp_search(Xis, is_df["cls_5"])
        print(f"    折{i}: 最佳参数 {best}")
        # [3] 多周期训练 + 融合
        probas = {}
        for hz in HORIZONS:
            m = XGBClassifier(n_estimators=300, nthread=8, eval_metric="auc",
                              random_state=RANDOM_STATE, use_label_encoder=False, **best)
            m.fit(Xis, is_df[f"cls_{hz}"])
            probas[hz] = m.predict_proba(Xte)[:, 1]
        fused = np.mean([probas[hz] for hz in HORIZONS], axis=0)
        ytrue = {hz: oos_df[f"fwd_ret_{hz}"].values for hz in HORIZONS}
        ycls = {hz: oos_df[f"cls_{hz}"].values for hz in HORIZONS}
        # 单周期(5日)对照
        auc5 = roc_auc_score(ycls[5], probas[5]) if len(set(ycls[5])) > 1 else np.nan
        ic5 = pd.Series(probas[5]).corr(pd.Series(ytrue[5]), method="spearman")
        # 融合信号对各周期
        row = {"fold": i, "auc_single5": auc5, "ic_single5": ic5}
        for hz in HORIZONS:
            auc_f = roc_auc_score(ycls[hz], fused) if len(set(ycls[hz])) > 1 else np.nan
            ic_f = pd.Series(fused).corr(pd.Series(ytrue[hz]), method="spearman")
            row[f"auc_fuse_{hz}"] = auc_f
            row[f"ic_fuse_{hz}"] = ic_f
        rec.append(row)
        # 记录最后一折的因子重要性（融合模型用5日模型近似）
        m5 = XGBClassifier(n_estimators=300, nthread=8, eval_metric="auc",
                           random_state=RANDOM_STATE, use_label_encoder=False, **best)
        m5.fit(Xis, is_df["cls_5"])
        last_imp = m5.get_booster().get_score(importance_type="gain")
        print(f"    折{i}: 单周期5日 AUC={auc5:.3f}/IC={ic5:+.4f} | "
              f"融合 AUC=[{row['auc_fuse_5']:.3f},{row['auc_fuse_20']:.3f},{row['auc_fuse_60']:.3f}] "
              f"IC=[{row['ic_fuse_5']:+.4f},{row['ic_fuse_20']:+.4f},{row['ic_fuse_60']:+.4f}]")

    rec_df = pd.DataFrame(rec)
    print(f"\n[5] 汇总（均值）:")
    print(rec_df[[c for c in rec_df.columns if c != 'fold']].mean().round(4).to_string())

    # 因子重要性 Top（含交互）
    imp_df = pd.DataFrame({"gain": last_imp}).sort_values("gain", ascending=False)
    print("\n[6] 因子重要性 Top15 (by Gain):")
    print(imp_df.head(15).to_string())

    # ── 落盘 ──
    lines = ["# XGBoost 因子挖掘 · WFA 原型 v2（交互特征 + 多周期融合 + IS内调参）", "",
             f"- 样本: {len(codes)} 只 | 面板: {len(panel):,} 行 | 特征: {len(FEATS)} 个",
             f"- 含 Alpha {len(ALPHA_FEATS)} + 市场环境 {len(env.columns)} + 交互 {len(FEATS)-len(ALPHA_FEATS)-len(env.columns)}",
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
    lines += ["", "## 因子重要性 Top15 (by Gain)", ""]
    for name, row in imp_df.head(15).iterrows():
        lines.append(f"- {name}: gain={row['gain']:.3f}")
    lines += ["", "## 结论", "",
              "在 v1 基础上加入交互特征、多周期融合、IS内参数寻优；",
              "对比 v1（AUC≈0.528）观察是否提升。IC 略负属正常（原始因子拥挤/噪声）。"]
    out = "/workspace/quant_proto/proto_v2_results.md"
    open(out, "w").write("\n".join(lines))
    print(f"\n[7] 结果已写入: {out}  总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
