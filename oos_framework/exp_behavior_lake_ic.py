"""exp_behavior_lake_ic.py — 从已落盘 panel 重算 IC(宽表逐日Spearman, 省内存)

避免原 groupby.apply 在 5596×3800 面板上 OOM: 按因子读列, pivot 成 date×code 宽表,
逐日 rank→Pearson(=Spearman), 掩码处理 NaN. 分 全 / HS300 / 非HS300.
"""
import os, json, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lake_ic")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "behavior_cache")
PANEL = os.path.join(CACHE, "behavior_lake_panel.parquet")
HS300_PRICE = "D:/stcok-worm/yjyg_hs300_prices.parquet"
OUT_JSON = os.path.join(CACHE, "behavior_lake_ic.json")
FACTORS = ["max1", "max3", "max5", "skew", "ivol", "lottery",
           "anchor52", "anchor_int", "zt_dist", "avol"]


def row_spearman(A: pd.DataFrame, B: pd.DataFrame) -> float:
    Ar = A.rank(axis=1, method="average").values
    Br = B.rank(axis=1, method="average").values
    ma = A.notna().values & B.notna().values
    Ar_c = Ar - np.nanmean(Ar, axis=1, keepdims=True)
    Br_c = Br - np.nanmean(Br, axis=1, keepdims=True)
    num = np.nansum(Ar_c * Br_c * ma, axis=1)
    den = np.sqrt(np.nansum(Ar_c ** 2 * ma, axis=1) * np.nansum(Br_c ** 2 * ma, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return float(np.nanmean(r))


def ic_for(df: pd.DataFrame, f: str) -> float:
    wf = df.pivot(index="date", columns="code", values=f)
    wfwd = df.pivot(index="date", columns="code", values="fwd60")
    return row_spearman(wf, wfwd)


def main():
    hs300 = set(pd.read_parquet(HS300_PRICE)["code"].unique().tolist())
    log.info("HS300 基准 %d 只", len(hs300))
    res = {"ic60_full": {}, "ic60_hs300": {}, "ic60_non_hs300": {}}
    for f in FACTORS:
        log.info("IC: %s", f)
        df = pd.read_parquet(PANEL, columns=["date", "code", f, "fwd60"])
        df["is_hs300"] = df["code"].isin(hs300)
        res["ic60_full"][f] = round(ic_for(df, f), 4)
        res["ic60_hs300"][f] = round(ic_for(df[df["is_hs300"]], f), 4)
        res["ic60_non_hs300"][f] = round(ic_for(df[~df["is_hs300"]], f), 4)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("\n" + "=" * 82)
    print("行为因子 · 数据湖全A 5596只 (IC60, PIT, 做空高值=负IC盈利)")
    print("=" * 82)
    print(f"{'因子':12s} {'IC60全':>9s} {'IC60_HS300':>12s} {'IC60_非HS300':>14s}")
    for f in FACTORS:
        print(f"{f:12s} {res['ic60_full'][f]:+9.3f} {res['ic60_hs300'][f]:+12.3f} {res['ic60_non_hs300'][f]:+14.3f}")
    print("=" * 82)


if __name__ == "__main__":
    main()
