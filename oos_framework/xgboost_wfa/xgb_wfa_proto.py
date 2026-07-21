#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 量化因子挖掘 · WFA 最小原型
=====================================
数据来源：本地数据湖 /workspace/stocklake （全A含退市 · 近8年日线 + 三大表）
流程：数据湖 → 量价+基本面特征 → 防泄露标签 → 滚动 WFA → XGBoost → IC/ICIR + 因子重要性

设计要点（对照《XGBoost量化因子挖掘与WFA测试落地指南》）：
- 标签：未来5日收益率，起点用 T+1 收盘价（避开当日已发生价格，防未来信息泄露）。
- 特征仅在 T 及之前的数据上计算（ret/vol/rsi 等均用 shift/rolling，无前视）。
- 基本面：用 merge_asof(backward) 取“报告期 ≤ T”的最新财报快照，绝不使用未来财报。
- 截面标准化：每个交易日对特征做 Z-score（跨股票）。
- WFA：训练窗3年 / 测试窗1年 / 步长1年，按数据湖实际日期滚动；OOS 结果严格隔离。
- 防作弊：超参与特征在 WFA 前固定，过程中不按 OOS 调参。

说明：原型用“日线全量 + 已落盘三大表”的子集（默认最多 400 只）验证整条链路；
全量三大表到位后可把 MAX_STOCKS 调大或改为全市场。
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)

LAKE = os.environ.get("STOCKLAKE", "/workspace/stocklake")
DAILY = f"{LAKE}/daily"
FUND = f"{LAKE}/fundamentals"
MAX_STOCKS = 400          # 原型样本上限（三表齐全股票中截取）
TRAIN_YEARS = 3
TEST_YEARS = 1
STEP_YEARS = 1
RANDOM_STATE = 42

PRICE_FEATS = ["ret_5", "ret_20", "ret_60", "vol_20", "rsi_14",
               "amt_chg_20", "ma_dev_20", "amp_20"]
FUND_FEATS = ["gross_margin", "net_margin", "roe", "debt_ratio",
              "current_ratio", "ocf_netprofit", "operate_income_yoy",
              "netprofit_yoy", "eps"]
ALL_FEATS = PRICE_FEATS + FUND_FEATS
LABEL = "fwd_ret_5"        # 连续标签：未来5日收益（T+1起算）
LABEL_CLS = "label_cls"    # 排序打分标签：截面前30%=1（指南最推荐）


def pick(raw, *names):
    """取 raw 中第一个存在的列（不同股票报表列名不一）。"""
    for n in names:
        if n in raw.columns:
            return raw[n]
    return None


# ───────────────────────── 数据加载 ─────────────────────────
def eligible_codes():
    """取 日线 + 三表 都齐全的股票代码（交集）。"""
    daily = {f[:-8] for f in os.listdir(DAILY) if f.endswith(".parquet")}
    inc = {f[:-8] for f in os.listdir(f"{FUND}/income_statement") if f.endswith(".parquet")}
    bal = {f[:-8] for f in os.listdir(f"{FUND}/balance_sheet") if f.endswith(".parquet")}
    cf = {f[:-8] for f in os.listdir(f"{FUND}/cash_flow_statement") if f.endswith(".parquet")}
    ok = sorted(daily & inc & bal & cf)
    return ok[:MAX_STOCKS]


def load_fundamentals(code):
    """合并三大表 → 计算财务比率 → 返回按 REPORT_DATE 索引的快照表。

    注意：先把三表按 REPORT_DATE 合并原始列（同索引），再统一算比率，
    避免“比率 Series 用整数索引、out 用日期索引”导致的对齐错位→全 NaN。
    """
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
    # 合并原始列（按 REPORT_DATE 取并集，向前填充），统一索引后算比率
    raw = inc.set_index("REPORT_DATE")
    raw = raw.combine_first(bal.set_index("REPORT_DATE"))
    raw = raw.combine_first(cf.set_index("REPORT_DATE"))
    raw = raw.sort_index().ffill()

    oi, oc = raw.get("OPERATE_INCOME"), raw.get("OPERATE_COST")
    npf = raw.get("PARENT_NETPROFIT")
    eq = pick(raw, "PARENT_EQUITY_BALANCE", "TOTAL_PARENT_EQUITY", "TOTAL_EQUITY")
    ta, tl = raw.get("TOTAL_ASSETS"), raw.get("TOTAL_LIABILITIES")
    ca = pick(raw, "CURRENT_ASSET_BALANCE", "CURRENT_ASSET")
    cl = pick(raw, "CURRENT_LIAB_BALANCE", "CURRENT_LIAB")
    ocf = raw.get("NETCASH_OPERATE")

    out = pd.DataFrame(index=raw.index)
    out["gross_margin"] = (oi - oc) / oi if (oi is not None and oc is not None) else np.nan
    out["net_margin"] = npf / oi if (npf is not None and oi is not None) else np.nan
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
    """单只股票：日线特征 + 财报快照(backward) + 防泄露标签，返回面板行。"""
    df = pd.read_parquet(f"{DAILY}/{code}.parquet")
    if df is None or len(df) < 80:
        return None
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]; a = df["amount"]
    ret = c.pct_change()

    df["ret_5"] = c / c.shift(5) - 1
    df["ret_20"] = c / c.shift(20) - 1
    df["ret_60"] = c / c.shift(60) - 1
    df["vol_20"] = ret.rolling(20).std()
    # RSI(14) Wilder
    up = ret.clip(lower=0); dn = -ret.clip(upper=0)
    rs = up.rolling(14, min_periods=1).mean() / (dn.rolling(14, min_periods=1).mean() + 1e-12)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    df["amt_chg_20"] = a / a.rolling(20).mean() - 1
    df["ma_dev_20"] = c / c.rolling(20).mean() - 1
    df["amp_20"] = (h.rolling(20).max() - l.rolling(20).min()) / c

    # 标签：未来5日收益率，起点 T+1（防当日泄露）
    df[LABEL] = (c.shift(-5) / c.shift(-1)) - 1

    # 财报快照（仅用报告期 ≤ T 的最新一份）
    fin = load_fundamentals(code)
    if fin is not None and len(fin):
        df["date"] = df["date"].astype("datetime64[ns]")
        fdf = fin.reset_index()
        fdf["REPORT_DATE"] = fdf["REPORT_DATE"].astype("datetime64[ns]")
        df = pd.merge_asof(
            df.sort_values("date"), fdf.sort_values("REPORT_DATE"),
            left_on="date", right_on="REPORT_DATE", direction="backward")
        df.drop(columns=["REPORT_DATE"], inplace=True, errors="ignore")
    df["code"] = code
    df.dropna(subset=["ret_60", "vol_20", "rsi_14", "ma_dev_20", "amt_chg_20", "amp_20", LABEL],
              inplace=True)
    return df[["date", "code"] + ALL_FEATS + [LABEL]]


# ───────────────────────── 预处理 ─────────────────────────
def winsorize(panel, cols, lo=0.01, hi=0.99):
    for col in cols:
        if col in panel:
            ql, qh = panel[col].quantile(lo), panel[col].quantile(hi)
            panel[col] = panel[col].clip(ql, qh)
    return panel


def cross_section_zscore(panel, cols):
    """每个交易日对特征做跨股票 Z-score（指南 2.3.2）。"""
    grp = panel.groupby("date")[cols]
    panel[cols] = (panel[cols] - grp.transform("mean")) / (grp.transform("std") + 1e-12)
    return panel


# ───────────────────────── WFA ─────────────────────────
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


def main():
    t0 = time.time()
    codes = eligible_codes()
    print(f"[1] 三表齐全且取样的股票数: {len(codes)}")

    print("[2] 构建特征面板（日线+财报）...")
    panels = [build_stock_panel(c) for c in codes]
    panel = pd.concat([p for p in panels if p is not None], ignore_index=True)
    print(f"    面板规模: {len(panel):,} 行 × {len(panel.columns)} 列, "
          f"日期 {panel['date'].min().date()} ~ {panel['date'].max().date()}")

    print("[3] Winsorize + 截面 Z-score...")
    winsorize(panel, ALL_FEATS)
    cross_section_zscore(panel, ALL_FEATS)
    # 仅要求量价特征 + 标签非空（量价必算得出）；基本面 NaN 保留，
    # XGBoost 原生支持 NaN 分支，留空反而更稳。
    panel.dropna(subset=PRICE_FEATS + [LABEL], inplace=True)
    print(f"    清洗后: {len(panel):,} 行")
    fill = (panel[FUND_FEATS].notna().mean() * 100).round(1)
    print(f"    基本面特征填充率(%): {fill.to_dict()}")
    # 排序打分标签（指南最推荐）：每个交易日，未来5日收益处于前30%的股票=1
    panel[LABEL_CLS] = panel.groupby("date")[LABEL].transform(
        lambda s: (s.rank(pct=True) >= 0.7).astype(int))
    print(f"    正样本(前30%)占比: {panel[LABEL_CLS].mean():.2%}")

    folds = wfa_folds(panel["date"])
    print(f"[4] WFA 折数: {len(folds)} （训练{TRAIN_YEARS}年/测试{TEST_YEARS}年/步长{STEP_YEARS}年）")
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        print(f"    折{i}: IS {is_s.date()}~{is_e.date()} | OOS {oos_s.date()}~{oos_e.date()}")

    from sklearn.metrics import roc_auc_score
    fold_auc, fold_ic, fold_dates = [], [], []
    last_model = None
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        is_df = panel[(panel["date"] >= is_s) & (panel["date"] < is_e)]
        oos_df = panel[(panel["date"] >= oos_s) & (panel["date"] < oos_e)]
        if len(is_df) < 500 or len(oos_df) < 100:
            print(f"    折{i}: 样本不足，跳过 (IS={len(is_df)}, OOS={len(oos_df)})")
            continue
        # IS 内部再切出验证集用于早停（OOS 全程隔离，绝不参与模型选择）
        is_sorted = is_df.sort_values("date")
        cut = int(len(is_sorted) * 0.8)
        tr_df, val_df = is_sorted.iloc[:cut], is_sorted.iloc[cut:]
        Xtr, ytr = tr_df[ALL_FEATS], tr_df[LABEL_CLS]
        Xval, yval = val_df[ALL_FEATS], val_df[LABEL_CLS]
        Xte, yte = oos_df[ALL_FEATS], oos_df[LABEL_CLS]
        pos = max(1, int(ytr.sum())); neg = len(ytr) - pos
        params = dict(objective="binary:logistic", max_depth=6, eta=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                      reg_lambda=1, gamma=1, eval_metric="auc", nthread=8,
                      scale_pos_weight=neg / pos)
        dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=ALL_FEATS)
        dval = xgb.DMatrix(Xval, label=yval, feature_names=ALL_FEATS)
        dte = xgb.DMatrix(Xte, label=yte, feature_names=ALL_FEATS)
        bst = xgb.train(params, dtr, num_boost_round=400,
                        evals=[(dtr, "train"), (dval, "val")],
                        early_stopping_rounds=40, verbose_eval=False)
        prob = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
        ytrue = oos_df[LABEL].values
        auc = roc_auc_score(yte.values, prob) if yte.nunique() > 1 else float("nan")
        ic_pooled = pd.Series(prob).corr(pd.Series(ytrue), method="spearman")  # 合并IC
        tmp = pd.DataFrame({"date": oos_df["date"].values, "pred": prob, "y": ytrue})
        day_ic = tmp.groupby("date").apply(lambda g: g["pred"].corr(g["y"], method="spearman"))
        ic_day = day_ic.dropna().mean()
        fold_auc.append(auc); fold_ic.append(ic_pooled); fold_dates.append(f"折{i}")
        last_model = bst
        print(f"    折{i}: OOS {len(oos_df):,} | AUC={auc:.3f} | 合并IC={ic_pooled:+.4f} | 日均IC={ic_day:+.4f}")

    fold_auc = np.array(fold_auc); fold_ic = np.array(fold_ic)
    mean_auc, mean_ic = np.nanmean(fold_auc), np.nanmean(fold_ic)
    print(f"\n[5] WFA 汇总:")
    print(f"    各折 AUC   = {np.round(fold_auc,3).tolist()}")
    print(f"    各折 合并IC = {np.round(fold_ic,4).tolist()}")
    print(f"    均值 AUC={mean_auc:.3f} | 均值IC={mean_ic:+.4f}")

    print("[6] 因子重要性（最后一折模型, Gain/Weight）:")
    imp = last_model.get_score(importance_type="gain")
    imp_w = last_model.get_score(importance_type="weight")
    imp_df = pd.DataFrame({"gain": imp, "weight": imp_w}).fillna(0).sort_values("gain", ascending=False)
    print(imp_df.head(12).to_string())

    # ── 落盘结果 ──
    lines = ["# XGBoost 因子挖掘 · WFA 原型结果", "",
             f"- 样本股票数: {len(codes)}（三表齐全子集，MAX_STOCKS={MAX_STOCKS}）",
             f"- 面板规模: {len(panel):,} 行",
             f"- 特征: {len(ALL_FEATS)} 个（量价{len(PRICE_FEATS)} + 基本面{len(FUND_FEATS)}）",
             f"- 标签: 排序打分（截面前30%未来5日收益=1，T+1起算）",
             f"- WFA: 训练{TRAIN_YEARS}年/测试{TEST_YEARS}年/步长{STEP_YEARS}年, 共{len(folds)}折",
             "", "## WFA 各折结果", ""]
    for i, (a, c) in enumerate(zip(fold_auc, fold_ic), 1):
        lines.append(f"- 折{i}: AUC={a:.3f}, 合并IC={c:+.4f}")
    lines += [f"- **均值 AUC={mean_auc:.3f}**, **均值IC={mean_ic:+.4f}**", "",
              "## 因子重要性 (Top, by Gain)", ""]
    for name, row in imp_df.head(12).iterrows():
        lines.append(f"- {name}: gain={row['gain']:.3f}, weight={row['weight']:.0f}")
    lines += ["", "## 结论", "",
              "链路验证：数据湖 → 特征 → WFA → XGBoost → AUC/IC/因子重要性 已跑通。",
              f"均值 AUC={mean_auc:.3f}（>0.5 有效，>0.6 合格，>0.7 优秀），均值IC={mean_ic:+.4f}（>0 方向有效）。",
              "注：原型仅用部分数据(三表齐全子集)+固定超参，未做参数寻优；全量三大表到位后可扩展至全市场并扩充交互特征。"]
    out = "/workspace/quant_proto/proto_results.md"
    open(out, "w").write("\n".join(lines))
    print(f"\n[7] 结果已写入: {out}")
    print(f"    总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
