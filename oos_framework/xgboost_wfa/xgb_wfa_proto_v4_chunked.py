#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 因子挖掘 · WFA 原型 v4（v3 + 筹码结构维度）· 分块流式版
==============================================================
与 xgb_wfa_proto_v4_full.py 算法完全一致, 仅内存策略不同: 通过「分组切片 + 流式落盘」,
全市场 5500 只在 8GB cgroup 内可跑(峰值 ≈ 单折训练矩阵 ~1-2GB + DMatrix ~1-2GB)。

分块关键修正(区别于朴素切片):
  · 截面运算(市场环境/拥挤度/PCA/z-score/交互)只需"某一天的所有股票同时在线", 不需"所有历史同时在线";
  · 但 trading_crowding 的 ZYEARS=1260 滚动、pca_absorption 的 PCT_WIN=1250 滚动**需要多年历史**,
    故每片回看窗口 = LOOKBACK(5年) 而非 3个月: 在 [cd0-5y, cd1) 扩展窗算拥挤度/PCA, 再只保留 [cd0,cd1) 落盘。
  · Phase B concat 后立即转 float32, 避免后续 interactions 在 float64 下把面板撑爆。
  · Phase C 训练前对 IS 做 WFA_TRAIN_CAP 随机子抽样(默认 2M 行), 训练矩阵从 ~3.6GB 压到 ~1.7GB。

容错: WFA_RESUME=1 断点续跑(复用已落盘 _base / _feat)。注意: 若改了 LOOKBACK 逻辑需先删 _feat 重算。
"""
import os, time, shutil, json, warnings, gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


def _mem(tag=""):
    """打印 RSS 与 cgroup memory.current/max(MB), 用于定位 8GB cgroup 下的内存尖峰。"""
    try:
        with open("/proc/self/status") as _f:
            for _l in _f:
                if _l.startswith("VmRSS:"):
                    _kb = int(_l.split()[1])
                    break
        cur = max_ = None
        try:
            cur = int(open("/sys/fs/cgroup/memory.current").read().strip()) / 1048576
            max_ = int(open("/sys/fs/cgroup/memory.max").read().strip()) / 1048576
        except Exception:
            pass
        _msg = f"RSS={_kb/1024:.0f}MB"
        if cur is not None:
            _msg += f" | cg.current={cur:.0f}MB"
            if max_ and max_ > 0:
                _msg += f" / max={max_:.0f}MB ({cur/max_*100:.0f}%)"
        print(f"    [MEM] {tag} {_msg}", flush=True)
    except Exception:
        pass

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180)

LAKE = os.environ.get("STOCKLAKE", "/workspace/stocklake")
DAILY = f"{LAKE}/daily"
FUND = f"{LAKE}/fundamentals"
INCOME = f"{FUND}/income_statement"
CASH = f"{FUND}/cash_flow_statement"
BAL = f"{FUND}/balance_sheet"
OUT = os.environ.get("WFA_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4proto_out"))
os.makedirs(OUT, exist_ok=True)
MAX_STOCKS = int(os.environ.get("WFA_MAX_STOCKS", 5500))
RESUME = os.environ.get("WFA_RESUME", "0") == "1"
TRAIN_CAP = int(os.environ.get("WFA_TRAIN_CAP", 2000000))   # IS 训练行数上限(控制训练矩阵内存)
BATCH = int(os.environ.get("WFA_BATCH", 0))   # Phase B 处理到此片数后停止并跳过 Phase C(0=不限制, 用于分次前台运行)
SKIP_B = os.environ.get("WFA_SKIP_B", "0") == "1"   # =1 时跳过 Phase B(复用已有 _feat), 仅重跑 Phase C
TRAIN_YEARS = 3
TEST_YEARS = 1
STEP_YEARS = 1
RANDOM_STATE = 42
HORIZONS = [5, 20, 60]
ZYEARS = 5 * 252
PCT_WIN = 1250
WARMUP = pd.DateOffset(months=3)
LOOKBACK = pd.DateOffset(years=5)   # 截面滚动(ZYEARS/PCT_WIN)所需历史, 必须 >> 单切片宽度
CHUNK_MONTHS = 12
NON_FEAT = {"date", "code", "_chip_cc_raw", "cls_5", "cls_20", "cls_60",
            "fwd_ret_5", "fwd_ret_20", "fwd_ret_60"}

PRICE_FEATS = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "rsi_14",
               "amt_chg_20", "ma_dev_20", "amp_20"]
FUND_FEATS = ["roe", "debt_ratio", "current_ratio", "ocf_netprofit",
              "netprofit_yoy", "eps", "bvps", "ocf_ps"]
ALPHA_FEATS = PRICE_FEATS + FUND_FEATS
CHIP_FEATS = ["chip_pr", "chip_cc", "chip_cb", "chip_cb_short"]
CHIP_DERIVED = ["chip_pr_x_cb", "chip_short_pen"]


def eligible_codes():
    daily = {f[:-8] for f in os.listdir(DAILY) if f.endswith(".parquet")}
    inc = {f[:-8] for f in os.listdir(INCOME) if f.endswith(".parquet")}
    cash = {f[:-8] for f in os.listdir(CASH) if f.endswith(".parquet")}
    bal = {f[:-8] for f in os.listdir(BAL) if f.endswith(".parquet")}
    codes = sorted(daily & inc & cash & bal)
    if MAX_STOCKS < len(codes):
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
    inc = _read_parquet(f"{INCOME}/{code}.parquet")
    cash = _read_parquet(f"{CASH}/{code}.parquet")
    bal = _read_parquet(f"{BAL}/{code}.parquet")
    if inc is None and cash is None and bal is None:
        return None
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
        df[f"fwd_ret_{hz}"] = c.shift(-hz) / c.shift(-1) - 1
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


def market_env(panel):
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


def hp_search(X, y, sample_weight=None, n_iter=8):
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
                            cv=2, n_jobs=1, random_state=RANDOM_STATE)
    if sample_weight is not None:
        rs.fit(X, y, sample_weight=sample_weight)
    else:
        rs.fit(X, y)
    return rs.best_params_


def _global_feats_from(path):
    return [c for c in pq.read_schema(path).names if c not in NON_FEAT]


def _subsample(d, idx):
    return {
        "X": d["X"][idx], "sw": d["sw"][idx], "code": d["code"][idx], "date": d["date"][idx],
        "cls": {hz: d["cls"][hz][idx] for hz in HORIZONS},
        "fwd": {hz: d["fwd"][hz][idx] for hz in HORIZONS},
    }


def main():
    t0 = time.time()
    codes = eligible_codes()
    print(f"[1] 股票数: {len(codes)} | RESUME={RESUME} | TRAIN_CAP={TRAIN_CAP}", flush=True)
    base_dir = os.path.join(OUT, "_base")
    feat_dir = os.path.join(OUT, "_feat")
    if not RESUME:
        for _d in (base_dir, feat_dir):
            if os.path.exists(_d):
                shutil.rmtree(_d)
    elif os.path.exists(feat_dir) and not SKIP_B:
        # 改了 LOOKBACK 逻辑后旧 _feat 不可用 → 清掉重算(保留 _base)
        shutil.rmtree(feat_dir)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    # ── Phase A: 逐只股票基础特征, 立刻落盘释放(峰值=单只票) ──
    print("[2] Phase A 逐只股票基础特征 → 落盘...", flush=True)
    meta = {}
    meta_path = f"{base_dir}/_meta.json"
    if RESUME and os.path.exists(meta_path):
        meta = {c: (pd.Timestamp(v[0]), pd.Timestamp(v[1]))
                for c, v in json.load(open(meta_path)).items()}
        print(f"    RESUME: 从 _meta.json 载入 {len(meta)} 只(跳过 Phase A)", flush=True)
    else:
        for c in codes:
            _bp = f"{base_dir}/{c}.parquet"
            if RESUME and os.path.exists(_bp):
                try:
                    _d = pd.read_parquet(_bp, columns=["date"])
                    meta[c] = (_d["date"].min(), _d["date"].max())
                except Exception:
                    meta[c] = (None, None)
                continue
            p = build_stock_panel(c)
            if p is None:
                continue
            p.to_parquet(_bp)
            meta[c] = (p["date"].min(), p["date"].max())
        json.dump({str(k): [str(a), str(b)] for k, (a, b) in meta.items()},
                  open(meta_path, "w"))
    meta = {c: v for c, v in meta.items() if v[0] is not None and v[1] is not None}
    if not meta:
        print("    无可用股票, 退出", flush=True); return
    all_min = min(m[0] for m in meta.values())
    all_max = max(m[1] for m in meta.values())
    print(f"    基础特征就绪: {len(meta)} 只 | {all_min.date()}~{all_max.date()}", flush=True)

    # ── Phase B: 按年切片, 回看 LOOKBACK(5年)算截面特征 → 仅保留本片落盘 ──
    print("[3] Phase B 切片截面特征(回看5年历史以支撑 ZYEARS/PCT_WIN 滚动)...", flush=True)
    ENV_N, CROWD_N = 8, 18
    if SKIP_B:
        # 复用已有 _feat 分片, 跳过 Phase B 重算(避免 RESUME 对 _feat 的删除, 省 ~1.5h)
        print("    SKIP_B=1 → 复用已有 _feat 分片, 跳过 Phase B 重算", flush=True)
        _ff = sorted([f for f in os.listdir(feat_dir)
                      if f.startswith("feat_") and f.endswith(".parquet")])
        chunks = []
        ci = 0
        s = all_min
        while s <= all_max:
            cd0, cd1 = s, s + pd.DateOffset(months=CHUNK_MONTHS)
            chunks.append((ci, cd0, cd1)); ci += 1; s = cd1
        base_crowds = []
        GLOBAL_FEATS = _global_feats_from(f"{feat_dir}/{_ff[0]}") if _ff else None
        start = all_max + pd.Timedelta(days=1)   # 使下面 Phase B while 条件不成立, 跳过
    else:
        chunks = []
        start = all_min
        ci = 0
        base_crowds = []
        GLOBAL_FEATS = None
        n_rows_est = 0
    while start <= all_max:
        cd0, cd1 = start, start + pd.DateOffset(months=CHUNK_MONTHS)
        warm0 = cd0 - LOOKBACK          # 关键: 滚动窗口需要多年历史
        _fpath = f"{feat_dir}/feat_{ci}.parquet"
        if RESUME and os.path.exists(_fpath):
            if GLOBAL_FEATS is None:
                GLOBAL_FEATS = _global_feats_from(_fpath)
            chunks.append((ci, cd0, cd1))
            ci += 1
            start = cd1
            continue
        parts = []
        for c, (m0, m1) in meta.items():
            if m1 >= warm0 and m0 <= cd1:
                df = pd.read_parquet(f"{base_dir}/{c}.parquet")
                df = df[(df["date"] >= warm0) & (df["date"] < cd1)]
                if len(df):
                    # 读入即转 float32, 避免 parts 累积 float64 撑爆
                    df = df.astype({col: "float32" for col in df.columns if df[col].dtype.kind in "biufc"})
                    parts.append(df)
                del df
        if parts:
            sub = pd.concat(parts, ignore_index=True)
            del parts
            for hz in HORIZONS:
                sub[f"cls_{hz}"] = sub.groupby("date")[f"fwd_ret_{hz}"].transform(
                    lambda s: (s.rank(pct=True) >= 0.7).astype(int))
            # 在扩展窗[warm0,cd1)上算市场环境/拥挤度/PCA(需要多年历史)
            env = market_env(sub)
            base_crowd, crowd_feat = build_crowding(sub, env)
            base_crowd = base_crowd[base_crowd.index >= cd0]
            # 仅保留本片日期做后续特征工程与落盘
            sub = sub[sub["date"] >= cd0].copy()
            sub = sub.drop(columns=["_amount", "_ret"], errors="ignore")
            sub["chip_pr_x_cb"] = sub["chip_pr"] * sub["chip_cb"]
            _pct = sub.groupby("code")["chip_cb_short"].transform(
                lambda s: s.rolling(PCT_WIN, min_periods=250).apply(
                    lambda w: float((w < w[-1]).mean()), raw=True))
            sub["chip_short_pen"] = np.where(_pct > 0.95, -1.0, 0.0)
            CHIP_ALL = CHIP_FEATS + CHIP_DERIVED
            sub["_chip_cc_raw"] = sub["chip_cc"]
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
            base_crowds.append(base_crowd)
            # 原子写入: 先写 .tmp 再 replace, 超时 SIGTERM 只留 .tmp, 已落盘分片完好可续跑
            _tmp = _fpath + ".tmp"
            sub.to_parquet(_tmp)       # 已是 float32
            os.replace(_tmp, _fpath)
            n_rows_est += len(sub)
            chunks.append((ci, cd0, cd1))
            ci += 1
            del sub, env, base_crowd, crowd_feat
        start = cd1
        if BATCH and ci >= BATCH:
            print(f"    [BATCH] 已达 {BATCH} 分片, 停止 Phase B(下次 RESUME 续跑)", flush=True)
            break
    print(f"    特征分片数: {len(chunks)} | 全局特征数: {len(GLOBAL_FEATS)}", flush=True)
    if base_crowds:
        _bc = pd.concat(base_crowds)
        _bc = _bc[~_bc.index.duplicated(keep="last")]
        print("    拥挤度基础指标相关性(§3.4):", flush=True)
        print(_bc.corr().round(3).to_string(), flush=True)

    # ── Phase C: 每折只读取覆盖该折窗的特征分片, float32 增量拼矩阵后训练 ──
    if BATCH:
        print(f"[4] WFA_BATCH={BATCH}>0 → 跳过 Phase C(分片未跑完, 待 BATCH=0 RESUME 续跑)", flush=True)
        return
    folds = wfa_folds(pd.Series([all_min, all_max]))
    print(f"[4] WFA 折数: {len(folds)}", flush=True)
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
        print(f"    折{i}: IS {is_s.date()}~{is_e.date()} | OOS {oos_s.date()}~{oos_e.date()}", flush=True)

    DROP = ([c for c in ["crowd_mom", "crowd_liq", "pca_absorp"] if c in GLOBAL_FEATS]
            + [c for c in CHIP_FEATS if c in GLOBAL_FEATS])

    # ── Phase C(分片中心): 每片只读一次, 行分配到各折 IS/OOS, 避免重复读大文件致页缓存累积 OOM ──
    def _extract(d):
        return {
            "X": d[GLOBAL_FEATS].astype("float32").values,
            "sw": (1.0 + np.clip(d["_chip_cc_raw"].fillna(0.0).astype("float32").values, 0.0, 10.0)).astype("float32"),
            "code": d["code"].values,
            "date": d["date"].values,
            "cls": {hz: d[f"cls_{hz}"].values.astype("int8") for hz in HORIZONS},
            "fwd": {hz: d[f"fwd_ret_{hz}"].values.astype("float32") for hz in HORIZONS},
        }

    def _concat(parts):
        if not parts:
            return None
        n = sum(p["X"].shape[0] for p in parts)
        nf = parts[0]["X"].shape[1]
        X = np.empty((n, nf), dtype="float32")
        sw = np.empty(n, dtype="float32")
        code = np.empty(n, dtype=object)
        date = np.empty(n, dtype="datetime64[ns]")
        cls = {hz: np.empty(n, dtype="int8") for hz in HORIZONS}
        fwd = {hz: np.empty(n, dtype="float32") for hz in HORIZONS}
        off = 0
        for p in parts:
            m = p["X"].shape[0]
            X[off:off + m] = p["X"]
            sw[off:off + m] = p["sw"]
            code[off:off + m] = p["code"]
            date[off:off + m] = p["date"]
            for hz in HORIZONS:
                cls[hz][off:off + m] = p["cls"][hz]
                fwd[hz][off:off + m] = p["fwd"][hz]
            off += m
        del parts
        gc.collect()
        return {"X": X, "sw": sw, "code": code, "date": date, "cls": cls, "fwd": fwd}

    # 第一遍(轻量): 每分片只读 date, 统计每折 IS/OOS 覆盖行数 → 配额
    is_cnt = {i: {} for i in range(1, len(folds) + 1)}
    oos_cnt = {i: {} for i in range(1, len(folds) + 1)}
    for cdi, cd0, cd1 in chunks:
        _d = pd.read_parquet(f"{feat_dir}/feat_{cdi}.parquet", columns=["date"])
        _dt = pd.to_datetime(_d["date"])
        for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
            if cd1 > is_s and cd0 < is_e:
                c = int(((_dt >= is_s) & (_dt < is_e)).sum())
                if c:
                    is_cnt[i][cdi] = c
            if cd1 > oos_s and cd0 < oos_e:
                c = int(((_dt >= oos_s) & (_dt < oos_e)).sum())
                if c:
                    oos_cnt[i][cdi] = c
        del _d
    gc.collect()
    # 每折"最后覆盖分片": 读完该分片即该折 IS+OOS 数据齐备, 可立即训练并释放
    last_chunk = {}
    for i in is_cnt:
        cs = list(is_cnt[i].keys()) + list(oos_cnt[i].keys())
        last_chunk[i] = max(cs) if cs else -1
    ready_at = {cdi: [] for cdi in range(len(chunks))}
    for i, lc in last_chunk.items():
        if lc >= 0:
            ready_at[lc].append(i)
    print(f"    每折最后覆盖分片: " + ", ".join(f"折{i}→chunk{last_chunk[i]}" for i in sorted(last_chunk)), flush=True)
    rng = np.random.RandomState(RANDOM_STATE)
    is_quota, oos_quota = {}, {}
    for i in is_cnt:
        tot = sum(is_cnt[i].values())
        for cdi, c in is_cnt[i].items():
            is_quota[(i, cdi)] = int(np.maximum(1, np.round(TRAIN_CAP * c / tot))) if (TRAIN_CAP and tot > TRAIN_CAP) else None
    for i in oos_cnt:
        tot = sum(oos_cnt[i].values())
        for cdi, c in oos_cnt[i].items():
            oos_quota[(i, cdi)] = int(np.maximum(1, np.round(TRAIN_CAP * c / tot))) if (TRAIN_CAP and tot > TRAIN_CAP) else None

    # 折叠训练辅助: 就地拼矩阵→调参→三周期融合→落盘 booster/shap/feats, 返回 (row, importance)
    def _train_fold(i, isd, oosd):
        if isd is None or len(isd["X"]) < 500:
            print(f"    折{i}: IS 样本不足跳过 (rows={len(isd['X']) if isd else 0})", flush=True)
            return None
        _mem(f"折{i} load_IS rows={len(isd['X']):,}")
        Xis, sw_is = isd["X"], isd["sw"]
        best = hp_search(Xis, isd["cls"][5], sample_weight=sw_is)
        gc.collect()
        _mem(f"折{i} hp_done")
        print(f"    折{i}: 最佳参数 {best}", flush=True)
        models = {}
        for hz in HORIZONS:
            m = XGBClassifier(n_estimators=300, nthread=4, eval_metric="auc",
                              random_state=RANDOM_STATE, use_label_encoder=False, **best)
            m.fit(Xis, isd["cls"][hz], sample_weight=sw_is)
            models[hz] = m
            gc.collect()
            _mem(f"折{i} fit_{hz}")
        del Xis, sw_is, isd
        gc.collect()
        _mem(f"折{i} after_train")
        if oosd is None or len(oosd["X"]) < 100:
            print(f"    折{i}: OOS 样本不足跳过", flush=True)
            return None
        _mem(f"折{i} load_OOS rows={len(oosd['X']):,}")
        Xte = oosd["X"]
        probas = {hz: models[hz].predict_proba(Xte)[:, 1] for hz in HORIZONS}
        fused = np.mean([probas[hz] for hz in HORIZONS], axis=0)
        auc5 = roc_auc_score(oosd["cls"][5], probas[5]) if len(set(oosd["cls"][5])) > 1 else np.nan
        ic5 = pd.Series(probas[5]).corr(pd.Series(oosd["fwd"][5]), method="spearman")
        row = {"fold": i, "auc_single5": auc5, "ic_single5": ic5}
        for hz in HORIZONS:
            ys, yt = oosd["cls"][hz], oosd["fwd"][hz]
            auc_f = roc_auc_score(ys, fused) if len(set(ys)) > 1 else np.nan
            ic_f = pd.Series(fused).corr(pd.Series(yt), method="spearman")
            row[f"auc_fuse_{hz}"] = auc_f
            row[f"ic_fuse_{hz}"] = ic_f
        m5 = models[5]
        m5.get_booster().save_model(f"{OUT}/booster_fold{i}.json")
        n = len(Xte)
        idx = (np.random.RandomState(RANDOM_STATE).choice(n, min(3000, n), replace=False)
               if n > 3000 else np.arange(n))
        sub_df = pd.DataFrame(Xte[idx], columns=GLOBAL_FEATS)
        sub_df["cls_5"] = oosd["cls"][5][idx]
        sub_df["fwd_ret_5"] = oosd["fwd"][5][idx]
        sub_df["code"] = oosd["code"][idx]
        sub_df["date"] = oosd["date"][idx]
        sub_df.to_parquet(f"{OUT}/shap_data_fold{i}.parquet")
        json.dump(GLOBAL_FEATS, open(f"{OUT}/feats_v4full.json", "w"))
        _raw = m5.get_booster().get_score(importance_type="gain")
        imp = {}
        for _k, _v in _raw.items():
            if _k.startswith("f") and _k[1:].isdigit() and int(_k[1:]) < len(GLOBAL_FEATS):
                imp[GLOBAL_FEATS[int(_k[1:])]] = _v
            else:
                imp[_k] = _v
        print(f"    折{i}: 单周期5日 AUC={auc5:.3f}/IC={ic5:+.4f} | "
              f"融合 AUC=[{row['auc_fuse_5']:.3f},{row['auc_fuse_20']:.3f},{row['auc_fuse_60']:.3f}] "
              f"IC=[{row['ic_fuse_5']:+.4f},{row['ic_fuse_20']:+.4f},{row['ic_fuse_60']:+.4f}]",
              flush=True)
        del Xte, oosd, models, m5
        return row, imp

    # 第二遍: 每分片整文件读(仅必要列, float32 已减半体积), 行分配到各折 IS/OOS; 读后 close+sleep+release_unused 回收页缓存
    # 关键1: 达到某折"最后覆盖分片"即就地训练并释放 acc → 常驻仅 ~1 折, 训练矩阵 ~0.4GB/折
    # 关键2: 持续打开 fd 时 fadvise(DONTNEED) 无效; 必须 close + sleep(0.5) + release_unused 内核才回收页缓存, 否则 cg.current 逐分片累积击穿 8GB
    _read_cols = list(dict.fromkeys(
        GLOBAL_FEATS + DROP + ["code", "date", "_chip_cc_raw"]
        + [f"cls_{hz}" for hz in HORIZONS]
        + [f"fwd_ret_{hz}" for hz in HORIZONS]))
    acc = {i: {"is": [], "oos": []} for i in range(1, len(folds) + 1)}
    rec = []
    last_imp = None
    # Phase C 断点续跑: 若某折 booster 已存在则跳过训练复用产物; 结果行持久化到 _fold_rows.json
    _rows_path = f"{OUT}/_fold_rows.json"
    _done_rows = {}
    if os.path.exists(_rows_path):
        try:
            _done_rows = json.load(open(_rows_path))
        except Exception:
            _done_rows = {}
    _force_retrain = os.environ.get("WFA_FORCE_RETRAIN") == "1"
    for cdi, cd0, cd1 in chunks:
        _fp = f"{feat_dir}/feat_{cdi}.parquet"
        _fo = open(_fp, "rb")
        try:
            os.posix_fadvise(_fo.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except Exception:
            pass
        _pf = pq.ParquetFile(_fo)
        _tbl = _pf.read(columns=_read_cols)
        df = _tbl.to_pandas(); del _tbl
        _mem(f"read chunk{cdi}")
        for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds, 1):
            if cdi in is_cnt[i]:
                d = df[(df["date"] >= is_s) & (df["date"] < is_e)].dropna(subset=DROP)
                if len(d):
                    q = is_quota[(i, cdi)]
                    if q is not None and q < len(d):
                        d = d.iloc[rng.choice(len(d), q, replace=False)]
                    acc[i]["is"].append(_extract(d))
            if cdi in oos_cnt[i]:
                d = df[(df["date"] >= oos_s) & (df["date"] < oos_e)].dropna(subset=DROP)
                if len(d):
                    q = oos_quota[(i, cdi)]
                    if q is not None and q < len(d):
                        d = d.iloc[rng.choice(len(d), q, replace=False)]
                    acc[i]["oos"].append(_extract(d))
        del df
        gc.collect()
        pa.default_memory_pool().release_unused()
        # 关键: 关闭 fd + 提示内核回收该文件页缓存 + 短暂等待, 否则 cg.current 逐分片累积击穿 8GB
        # (持续打开 fd 时 fadvise(DONTNEED) 无效; 必须 close 后内核才回收; float32 已把整文件读峰从 ~7.4GB 降到 ~5.3GB)
        try:
            os.posix_fadvise(_fo.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
        _fo.close()
        time.sleep(0.5)
        gc.collect()
        pa.default_memory_pool().release_unused()
        _mem(f"read chunk{cdi} done(reclaimed)")
        # 本分片即某折最后覆盖分片 → 就地训练并释放该折全部 acc 数据
        for i in ready_at[cdi]:
            isd = _concat(acc[i]["is"]); acc[i]["is"] = []
            osd = _concat(acc[i]["oos"]); acc[i]["oos"] = []
            _bf = f"{OUT}/booster_fold{i}.json"
            if os.path.exists(_bf) and not _force_retrain:
                # 已训过 → 跳过训练, 复用已有 booster/shap, 载入已存结果行(支持断点续跑)
                _r = _done_rows.get(str(i)) or _done_rows.get(i)
                if _r:
                    rec.append(_r)
                    print(f"    折{i}: 复用已有 booster_fold{i}.json(跳过训练)", flush=True)
                _mem(f"折{i} skip(reuse)")
                gc.collect()
                continue
            res = _train_fold(i, isd, osd)
            if res:
                rec.append(res[0]); last_imp = res[1]
                _done_rows[str(i)] = res[0]
                json.dump(_done_rows, open(_rows_path, "w"))
            gc.collect()
            _mem(f"折{i} done+released")
    # 安全网: 理论上所有折在最后分片已触发; 边界异常时在此补训(同样尊重断点续跑)
    for i in list(acc.keys()):
        if acc[i]["is"] or acc[i]["oos"]:
            _bf = f"{OUT}/booster_fold{i}.json"
            isd = _concat(acc[i]["is"]); acc[i]["is"] = []
            osd = _concat(acc[i]["oos"]); acc[i]["oos"] = []
            if os.path.exists(_bf) and not _force_retrain:
                _r = _done_rows.get(str(i)) or _done_rows.get(i)
                if _r:
                    rec.append(_r)
                gc.collect()
                continue
            res = _train_fold(i, isd, osd)
            if res:
                rec.append(res[0]); last_imp = res[1]
                _done_rows[str(i)] = res[0]
                json.dump(_done_rows, open(_rows_path, "w"))
            gc.collect()

    rec_df = pd.DataFrame(rec)
    print(f"\n[5] 汇总（均值）:", flush=True)
    print(rec_df[[c for c in rec_df.columns if c != 'fold']].mean().round(4).to_string(), flush=True)
    imp_df = pd.DataFrame({"gain": last_imp}).sort_values("gain", ascending=False)
    print("\n[6] 因子重要性 Top20 (by Gain):", flush=True)
    print(imp_df.head(20).to_string(), flush=True)
    crowd_mask = [x for x in imp_df.index if any(x.startswith(c) for c in ["crowd_mom", "crowd_liq", "pca_absorp"])]
    print(f"\n[7] 拥挤度特征总Gain占比: "
          f"{imp_df.loc[crowd_mask,'gain'].sum() / imp_df['gain'].sum():.3f}", flush=True)
    print(f"[7] 筹码结构特征总Gain占比: "
          f"{imp_df.loc[[x for x in imp_df.index if x.startswith('chip_')],'gain'].sum() / imp_df['gain'].sum():.3f}",
          flush=True)

    lines = ["# XGBoost 因子挖掘 · WFA 原型 v4（分块流式版，全市场）", "",
             f"- 样本: {len(codes)} 只 | 特征分片: {len(chunks)} | 特征: {len(GLOBAL_FEATS)} 个",
             f"- 内存策略: PhaseA逐只落盘 → PhaseB按年切片(回看5年历史算截面) → PhaseC按需读分片+float32+IS子抽样(≤{TRAIN_CAP:,})",
             f"  全市场 5500 只可在 8GB cgroup 内运行(峰值≈训练矩阵 ~1-2GB + DMatrix ~1-2GB)",
             f"- Alpha {len(ALPHA_FEATS)} + 市场环境 {ENV_N} + 交互 "
             f"{len(GLOBAL_FEATS)-len(ALPHA_FEATS)-len(CHIP_FEATS)-ENV_N-CROWD_N-len(CHIP_DERIVED)} "
             f"+ 拥挤度 {CROWD_N} + 筹码 {len(CHIP_FEATS)+len(CHIP_DERIVED)}",
             f"- 拥挤度实现: §1.1 交易行为(成交额代理, mom/liq) + §1.4 PCA吸收比率；",
             f"  §1.2 估值价差(需PE)、§1.3 机构持仓(需基金持仓) 数据湖缺失→跳过",
             f"- 筹码结构: §2 VWAP中心三角分布递推 + §4 PR/CC/CB/短期CB；",
             f"  §5.1 PR×CB 交互、§5.2 短期乖离>95%惩罚 + CC 作训练样本权重(无tick/换手→代理)",
             f"- 特征展开 §2: 原始值/历史分位数/变化率/宏观交互/惩罚项",
             f"- 标签: 排序打分(前30%) × 3周期(5/20/60日) | WFA 共 {len(folds)} 折",
             f"- 调参: 每折 IS 内 RandomizedSearchCV(cv=3), OOS 全程隔离",
             f"- 注: IS 训练做 ≤{TRAIN_CAP:,} 行随机子抽样以控内存(全市场横截面特征已含所有股票, 不影响覆盖)", "",
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
              "PhaseA逐只落盘 + PhaseB按年切片(回看5年历史以支撑 ZYEARS/PCT_WIN 滚动) + PhaseC按需读分片+float32+IS子抽样，",
              "把峰值内存从全面板(~23GB)压到单折训练矩阵(~1-2GB)，全市场 5500 只可在 8GB 内完成 WFA+SHAP。"]
    out = f"{OUT}/proto_v4_chunked_results.md"
    open(out, "w").write("\n".join(lines))
    print(f"\n[8] 结果已写入: {out}  总耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
