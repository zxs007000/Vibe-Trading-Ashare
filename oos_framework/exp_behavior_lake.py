"""exp_behavior_lake.py — 行为因子批量实测(直接读本地数据湖 daily/, 全A 5596只)

相比 exp_behavior_battery.py: 不再从 Sina 拉 HS300, 直接读 stockworm/daily/{code}.parquet
(全A OHLCV, 2002-2026). 这样小盘天然在内, 可测"行为edge需小盘"假设.

因子(OHLCV即可, 无需流通股本):
  MAX1/3/5 彩票偏好(Bali 2011) | SKEW/IVOL 特质波动 | LOTTERY=z(MAX5)+z(SKEW)+z(IVOL)
  ANCHOR_52W 52周高锚定 | ANCHOR_INT 整数关口 | ZT_DIST 涨停距离 | AVOL 异常量
  (CGO 需流通股本, 湖里没有 → 本脚本跳过, 由 anchor52 代理处置效应维度)
IVOL 市场中性化: 用 index/sh000300.parquet 作市场收益基准(本地).
HS300 拆分: 用 D:/stcok-worm/yjyg_hs300_prices.parquet 的 288 只作本地基准(非HS300=小盘+中盘代理).
IC: 截面 rank-IC vs 前向20/60d, 全样本 + HS300 + 非HS300.
"""
from __future__ import annotations
import os, argparse, json, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("behavior_lake")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "behavior_cache")
os.makedirs(CACHE, exist_ok=True)
LAKE_DAILY = "D:/work Buddy GZ/Claw/stockworm/daily"
MKT_IDX = "D:/work Buddy GZ/Claw/stockworm/index/sh000300.parquet"
HS300_PRICE = "D:/stcok-worm/yjyg_hs300_prices.parquet"
OUT_JSON = os.path.join(CACHE, "behavior_lake_ic.json")

FACTORS = ["max1", "max3", "max5", "skew", "ivol", "lottery",
           "anchor52", "anchor_int", "zt_dist", "avol"]


def load_mkt_ret():
    idx = pd.read_parquet(MKT_IDX).sort_index()
    return idx["close"].pct_change()


def load_codes(cap=None):
    fs = [f[:-8] for f in os.listdir(LAKE_DAILY) if f.endswith(".parquet")]
    fs = sorted(fs)
    if cap:
        fs = fs[:cap]
    return fs


def build_factors(codes, mkt_ret):
    recs = []
    for ci, code in enumerate(codes):
        try:
            d = pd.read_parquet(os.path.join(LAKE_DAILY, code + ".parquet"))
        except Exception:
            continue
        d = d.sort_index()
        if len(d) < 252:
            continue
        close = d["close"]
        r = close.pct_change()
        cum3 = r + r.shift(1) + r.shift(2)
        cum5 = sum(r.shift(k) for k in range(5))
        max1 = r.rolling(21, min_periods=10).max()
        max3 = cum3.rolling(21, min_periods=10).max()
        max5 = cum5.rolling(21, min_periods=10).max()
        skew = r.rolling(21, min_periods=10).skew()
        mkt = mkt_ret.reindex(d.index).fillna(0)
        idioret = r - mkt
        ivol = idioret.rolling(21, min_periods=10).std()
        anchor52 = close / close.rolling(252, min_periods=120).max() - 1.0
        anchor_int = (close % 10) / 10.0
        is_zt = (r >= 0.095).astype(int)
        grp = (is_zt == 0).cumsum()
        zt_dist = grp.groupby(grp).cumcount()
        avol = d["volume"] / d["volume"].rolling(20, min_periods=10).mean()
        fwd20 = close.shift(-20) / close - 1.0
        fwd60 = close.shift(-60) / close - 1.0
        df = pd.DataFrame({
            "max1": max1, "max3": max3, "max5": max5, "skew": skew, "ivol": ivol,
            "anchor52": anchor52, "anchor_int": anchor_int, "zt_dist": zt_dist,
            "avol": avol, "fwd20": fwd20, "fwd60": fwd60,
        })
        df.index.name = "date"
        df["code"] = code
        recs.append(df.reset_index())
        if (ci + 1) % 500 == 0:
            log.info("  因子构建 %d/%d", ci + 1, len(codes))
    if not recs:
        return pd.DataFrame()
    panel = pd.concat(recs, ignore_index=True)
    lot = panel.groupby("date")[["max5", "skew", "ivol"]].transform(
        lambda x: (x - x.mean()) / x.std())
    panel["lottery"] = lot["max5"] + lot["skew"] + lot["ivol"]
    return panel


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


def compute_ic(panel, fwd):
    out = {}
    for f in FACTORS:
        ic = panel.groupby("date").apply(lambda x: _spearman(x[f].values, x[fwd].values))
        out[f] = ic.mean()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=None)
    args = ap.parse_args()
    log.info("读取数据湖代码列表(cap=%s)...", args.cap)
    codes = load_codes(args.cap)
    log.info("加载沪深300基准收益(IVOL中性化)...")
    mkt = load_mkt_ret()
    log.info("构建行为因子面板(%d 只)...", len(codes))
    panel = build_factors(codes, mkt)
    panel.to_parquet(os.path.join(CACHE, "behavior_lake_panel.parquet"))
    hs300 = set(pd.read_parquet(HS300_PRICE)["code"].unique().tolist())
    panel["is_hs300"] = panel["code"].isin(hs300)
    n_hs = panel[panel["is_hs300"]]["code"].nunique()
    n_non = panel[~panel["is_hs300"]]["code"].nunique()
    log.info("HS300=%d只, 非HS300=%d只", n_hs, n_non)
    ic_full = compute_ic(panel, "fwd60")
    ic_hs = compute_ic(panel[panel["is_hs300"]], "fwd60")
    ic_non = compute_ic(panel[~panel["is_hs300"]], "fwd60")
    res = {
        "n_codes": len(codes), "n_hs300": n_hs, "n_non_hs300": n_non,
        "ic60_full": {k: round(float(v), 4) for k, v in ic_full.items()},
        "ic60_hs300": {k: round(float(v), 4) for k, v in ic_hs.items()},
        "ic60_non_hs300": {k: round(float(v), 4) for k, v in ic_non.items()},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 82)
    print("行为因子 · 数据湖全A (IC60, PIT, 做空高值=负IC盈利)")
    print("=" * 82)
    print(f"样本: {len(codes)}只 | HS300 {n_hs} / 非HS300 {n_non}")
    print(f"{'因子':12s} {'IC60全':>9s} {'IC60_HS300':>12s} {'IC60_非HS300':>14s}")
    for f in FACTORS:
        print(f"{f:12s} {ic_full[f]:+9.3f} {ic_hs[f]:+12.3f} {ic_non[f]:+14.3f}")
    print("=" * 82)


if __name__ == "__main__":
    main()
