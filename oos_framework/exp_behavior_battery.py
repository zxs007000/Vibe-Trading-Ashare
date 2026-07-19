"""exp_behavior_battery.py — 行为金融因子批量实测 (手册公式落地, HS300)

价格+成交量源: Sina stock_zh_a_daily (close/open/high/low/volume/outstanding_share)
批量因子(全部 PIT, 因子值已知于 t 收盘):
  价格类:
    MAX1/3/5  = 月内(21d) 最大 1/3/5 日累计收益        (Bali 2011 彩票偏好)
    SKEW/IVOL = 月内收益偏度 / 特质波动率(市场中性)      (彩票偏好组成)
    LOTTERY    = z(MAX5)+z(SKEW)+z(IVOL)                (彩票偏好综合)
    ANCHOR_52W = close / 252d高                         (George&Hwang 锚定)
    ANCHOR_INT = (close%10)/10 距整数10元关口比例        (整数关口锚定)
    ZT_DIST    = 距上次涨停(>=9.5%)交易日数              (涨停注意力)
  量价类:
    CGO        = (P-RP)/P, RP=换手率加权参考成本(Grinblatt&Han 2005 处置效应)
    AVOL       = volume / 20d均量                       (异常成交量=散户涌入)
IC: 截面 rank-IC vs 前向20/60d, 全样本 + 2023-25; 因子正交性.
注: 处置/注意力是散户行为, HS300(机构化)应偏弱; 真测需小盘价. 本脚本给第一手读数.
"""
from __future__ import annotations
import os
import time
import argparse
import logging
import threading

import numpy as np
import pandas as pd
import akshare as ak

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("behavior_bat")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "behavior_cache")
os.makedirs(CACHE, exist_ok=True)
PRICE_VOL = os.path.join(CACHE, "hs300_prices_vol.parquet")
OUT_JSON = os.path.join(CACHE, "behavior_ic.json")
_NET_TIMEOUT = 25


def _safe(fn, timeout=_NET_TIMEOUT):
    holder = {}
    def _t():
        try:
            holder["r"] = fn()
        except Exception as e:
            holder["e"] = repr(e)[:160]
    th = threading.Thread(target=_t, daemon=True)
    th.start(); th.join(timeout)
    if th.is_alive():
        return None, "timeout"
    if "e" in holder:
        return None, holder["e"]
    return holder.get("r"), None


def _sina_code(code: str) -> str:
    code = str(code).strip().split(".")[0]
    return ("sh" if code[:1] in "69" else "sz") + code


def _fetch_vol(code: str):
    d = ak.stock_zh_a_daily(symbol=_sina_code(code), adjust="")
    if d is None or d.empty:
        return None
    d = d.copy()
    d["code"] = code
    dc, cc = ("date" if "date" in d.columns else "日期", "close" if "close" in d.columns else "收盘")
    vc = "volume" if "volume" in d.columns else "成交量"
    oc = "outstanding_share" if "outstanding_share" in d.columns else "流通股本"
    out = pd.DataFrame({
        "date": pd.to_datetime(d[dc], errors="coerce"),
        "open": pd.to_numeric(d["open" if "open" in d.columns else "开盘"], errors="coerce"),
        "high": pd.to_numeric(d["high" if "high" in d.columns else "最高"], errors="coerce"),
        "low": pd.to_numeric(d["low" if "low" in d.columns else "最低"], errors="coerce"),
        "close": pd.to_numeric(d[cc], errors="coerce"),
        "volume": pd.to_numeric(d[vc], errors="coerce"),
        "out_share": pd.to_numeric(d[oc], errors="coerce"),
    })
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out[~out["date"].duplicated(keep="last")]
    out["code"] = code
    return out


def load_prices_vol(use_cache=True):
    if use_cache and os.path.exists(PRICE_VOL):
        log.info("复用量价缓存: %s", PRICE_VOL)
        return pd.read_parquet(PRICE_VOL)
    codes = pd.read_parquet("D:/stcok-worm/yjyg_hs300_prices.parquet")["code"].unique().tolist()
    frames = []
    ok = 0
    for i, code in enumerate(codes):
        d, err = _safe(lambda c=code: _fetch_vol(c))
        if d is not None and not d.empty:
            frames.append(d); ok += 1
        elif err and "timeout" in str(err):
            log.warning("量价 %s 超时", code)
        if (i + 1) % 50 == 0:
            log.info("  量价 %d/%d (成功 %d)", i + 1, len(codes), ok)
    if not frames:
        log.error("量价全部失败"); return pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    allp.to_parquet(PRICE_VOL)
    log.info("量价落盘: %d 行, %d 只", len(allp), allp["code"].nunique())
    return allp


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    n = len(a)
    if n < 5:
        return np.nan
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    da = ra - ra.mean(); db = rb - rb.mean()
    den = np.sqrt(float((da ** 2).sum()) * float((db ** 2).sum()))
    return float((da * db).sum() / den) if den > 0 else np.nan


def build_factors(prices):
    recs = []
    # 市场收益(等权, 作 IVOL 中性化基准)
    wide_close = prices.pivot_table(index="date", columns="code", values="close")
    mkt_ret = wide_close.pct_change().mean(axis=1)
    for code, g in prices.groupby("code"):
        g = g.sort_values("date").set_index("date")
        if len(g) < 252:
            continue
        r = g["close"].pct_change()
        # MAX 系列
        cum3 = r + r.shift(1) + r.shift(2)
        cum5 = sum(r.shift(k) for k in range(5))
        max1 = r.rolling(21, min_periods=10).max()
        max3 = cum3.rolling(21, min_periods=10).max()
        max5 = cum5.rolling(21, min_periods=10).max()
        skew = r.rolling(21, min_periods=10).skew()
        mkt = mkt_ret.reindex(g.index).fillna(0)
        idioret = r - mkt
        ivol = idioret.rolling(21, min_periods=10).std()
        # 彩票综合(截面z在 assemble 阶段做, 这里存原始)
        # 锚定
        anchor52 = g["close"] / g["close"].rolling(252, min_periods=120).max() - 1.0
        anchor_int = (g["close"] % 10) / 10.0
        # 涨停距离: 距上次涨停(>=9.5%)的交易日数 (0=当天涨停)
        is_zt = (r >= 0.095).astype(int)
        grp = (is_zt == 0).cumsum()
        zt_dist = grp.groupby(grp).cumcount()
        # CGO (Grinblatt-Han 2005) — 换手率加权参考成本
        # Sina volume 单位为"手"(=100股), 流通股本为"股"; 自动校正量级
        raw_to = (g["volume"] / g["out_share"]).replace([np.inf, -np.inf], np.nan)
        med_to = raw_to.median()
        scale = 100.0 if (med_to > 0 and med_to < 0.001) else 1.0
        to = (raw_to * scale).clip(upper=1.0, lower=0.0).fillna(0.0)
        vals = g["close"].values; tov = to.values
        rp_arr = np.empty(len(vals)); prev = vals[0]
        for i in range(len(vals)):
            prev = (1 - tov[i]) * prev + tov[i] * vals[i]
            rp_arr[i] = prev
        rp = pd.Series(rp_arr, index=g.index)
        cgo = (g["close"] - rp) / g["close"]
        # AVOL
        avol = g["volume"] / g["volume"].rolling(20, min_periods=10).mean()
        # 前向收益
        fwd20 = g["close"].shift(-20) / g["close"] - 1.0
        fwd60 = g["close"].shift(-60) / g["close"] - 1.0
        df = pd.DataFrame({
            "max1": max1, "max3": max3, "max5": max5, "skew": skew, "ivol": ivol,
            "anchor52": anchor52, "anchor_int": anchor_int, "zt_dist": zt_dist,
            "cgo": cgo, "avol": avol, "fwd20": fwd20, "fwd60": fwd60,
        })
        df.index.name = "date"
        df["code"] = code
        df = df.reset_index()
        recs.append(df)
    panel = pd.concat(recs, ignore_index=True)
    # 截面 z 合成 LOTTERY
    lottery = panel.groupby("date")[["max5", "skew", "ivol"]].transform(
        lambda x: (x - x.mean()) / x.std())
    panel["lottery"] = lottery["max5"] + lottery["skew"] + lottery["ivol"]
    # 诊断: CGO 数值分布 + 换手率量级是否合理
    log.info("CGO 中位数=%.4f 分位[1,99]=%.4f, %.4f (应落在[-1,1]且非恒定)",
             panel["cgo"].median(), panel["cgo"].quantile(0.01), panel["cgo"].quantile(0.99))
    log.info("AVOL 中位数=%.3f 分位[1,99]=%.3f, %.3f", panel["avol"].median(),
             panel["avol"].quantile(0.01), panel["avol"].quantile(0.99))
    return panel


FACTORS = ["max1", "max3", "max5", "skew", "ivol", "lottery",
           "anchor52", "anchor_int", "zt_dist", "cgo", "avol"]


def compute_ic(panel, fwd):
    out = {}
    for f in FACTORS:
        ic = panel.groupby("date").apply(lambda x: _spearman(x[f].values, x[fwd].values))
        out[f] = ic.mean()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    log.info("加载量价(HS300)...")
    prices = load_prices_vol(use_cache=not args.no_fetch)
    log.info("构建行为因子面板...")
    panel = build_factors(prices)
    panel.to_parquet(os.path.join(CACHE, "behavior_panel.parquet"))
    ic20 = compute_ic(panel, "fwd20")
    ic60 = compute_ic(panel, "fwd60")
    sub = panel[panel["date"] >= pd.Timestamp("2023-01-01")]
    ic20_25 = compute_ic(sub, "fwd20")
    ic60_25 = compute_ic(sub, "fwd60")
    corr = panel[FACTORS].corr()
    import json
    res = {
        "ic20_full": {k: round(float(v), 4) for k, v in ic20.items()},
        "ic60_full": {k: round(float(v), 4) for k, v in ic60.items()},
        "ic20_2325": {k: round(float(v), 4) for k, v in ic20_25.items()},
        "ic60_2325": {k: round(float(v), 4) for k, v in ic60_25.items()},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 78)
    print("行为金融因子批量实测 — HS300 (PIT, 截面rank-IC)")
    print("=" * 78)
    print(f"样本: {panel.shape[0]} (date,code) | 代码 {panel['code'].nunique()}")
    print(f"{'因子':12s} {'IC20全':>9s} {'IC60全':>9s} {'IC20 23-25':>11s} {'IC60 23-25':>11s}")
    for f in FACTORS:
        print(f"{f:12s} {ic20[f]:+9.3f} {ic60[f]:+9.3f} {ic20_25[f]:+11.3f} {ic60_25[f]:+11.3f}")
    print("-" * 78)
    print("方向参考(做空高值): CGO/MAX/AVOL/anchor52≈1/zt_dist小; 做多高: anchor_int")
    print("=" * 78)


if __name__ == "__main__":
    main()
