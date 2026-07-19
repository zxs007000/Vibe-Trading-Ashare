"""exp_doc_factors.py — 复现文档框架可湖化的 3 个因子 + IC 测试

文档六因子中, 数据湖能直接复现的 3 个:
  ③ 景气动量(prosperity): 营收同比加速度 = ΔYSTZ_t - ΔYSTZ_{t-1}
      源 stockworm/fundamentals/{code}.parquet (YSTZ=营收同比%); PIT=NOTICE_DATE+45d 前向填充.
  ④ 低关注(neglected): 成交额代理 -rolling20(amount)
      源 stockworm/daily/{code}.parquet (amount=成交额); 真 turnover 需流通股本, 用成交额低做流动性/低关注代理.
  ② 规模(size): -ln(流通市值)
      源 akshare stock_zh_a_spot_em (一次 bulk, 含流通市值); 剔 ST. 当前快照→仅近期窗口 IC(排名稳定).

IC: 截面 rank-IC(PIT, 宽表逐日Spearman, 省内存). 分 全 / HS300 / 非HS300.
规模因快照, IC 仅算 2018+ 近期窗口并标注. 其余全历史.
"""
from __future__ import annotations
import os, json, logging
import numpy as np
import pandas as pd
import akshare as ak

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("doc_factors")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "behavior_cache")
os.makedirs(CACHE, exist_ok=True)
PANEL = os.path.join(CACHE, "behavior_lake_panel.parquet")
DAILY = "D:/work Buddy GZ/Claw/stockworm/daily"
FUND = "D:/work Buddy GZ/Claw/stockworm/fundamentals"
HS300_PRICE = "D:/stcok-worm/yjyg_hs300_prices.parquet"
OUT_JSON = os.path.join(CACHE, "doc_factors_ic.json")
SIZE_WINDOW = "2018-01-01"


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


def build_low_att():
    codes = [f[:-8] for f in os.listdir(DAILY) if f.endswith(".parquet")]
    recs = []
    for code in codes:
        try:
            d = pd.read_parquet(os.path.join(DAILY, code + ".parquet"), columns=["date", "amount"])
        except Exception:
            continue
        d = d.sort_index()
        if len(d) < 20:
            continue
        low = -d["amount"].rolling(20, min_periods=10).mean()
        recs.append(pd.DataFrame({"date": d.index, "code": code, "low_att": low.values}))
    out = pd.concat(recs, ignore_index=True)
    log.info("低关注(成交额代理): %d 行", len(out))
    return out


def build_prosperity():
    codes = [f[:-8] for f in os.listdir(FUND) if f.endswith(".parquet")]
    recs = []
    for code in codes:
        try:
            d = pd.read_parquet(os.path.join(FUND, code + ".parquet"),
                                columns=["REPORTDATE", "NOTICE_DATE", "YSTZ"])
        except Exception:
            continue
        d = d.copy()
        d["rd"] = pd.to_datetime(d["REPORTDATE"], errors="coerce")
        d = d.dropna(subset=["rd"]).sort_values("rd")
        yoy = pd.to_numeric(d["YSTZ"], errors="coerce")
        if yoy.notna().sum() < 3:
            continue
        accel = yoy.diff() - yoy.diff().shift(1)   # ΔYoY_t - ΔYoY_{t-1}
        pub = pd.to_datetime(d["NOTICE_DATE"], errors="coerce").fillna(d["rd"])
        eff = (pub + pd.Timedelta(days=45)).astype("datetime64[ns]")  # PIT 滞后, 统一 ns 避免 merge_asof dtype 冲突
        step = pd.DataFrame({"date": eff.values, "code": code, "prosperity": accel.values})
        step = step.dropna(subset=["prosperity"]).sort_values("date")
        if not step.empty:
            recs.append(step)
    out = pd.concat(recs, ignore_index=True)
    log.info("景气动量(营收加速度): %d 步", len(out))
    return out


def build_size():
    try:
        log.info("akshare stock_zh_a_spot_em (流通市值)...")
        spot = ak.stock_zh_a_spot_em()
        spot = spot[["代码", "名称", "流通市值"]].copy()
        spot = spot.dropna(subset=["流通市值"])
        spot["流通市值"] = pd.to_numeric(spot["流通市值"], errors="coerce")
        spot = spot[spot["流通市值"] > 0]
        spot = spot[~spot["名称"].astype(str).str.contains("ST")]
        spot["size"] = -np.log(spot["流通市值"])
        return dict(zip(spot["代码"].astype(str).str.zfill(6), spot["size"]))
    except Exception as e:
        log.warning("规模因子跳过(akshare 东财 spot 经代理被墙: %s)", repr(e)[:120])
        return {}


def main():
    log.info("载入基础面板(behavior_lake_panel)...")
    base = pd.read_parquet(PANEL)
    hs300 = set(pd.read_parquet(HS300_PRICE)["code"].astype(str).str.zfill(6).tolist())
    base["code"] = base["code"].astype(str).str.zfill(6)
    base["is_hs300"] = base["code"].isin(hs300)
    base = base.sort_values(["code", "date"])

    log.info("复现 ④ 低关注...")
    low = build_low_att()
    low["code"] = low["code"].astype(str).str.zfill(6)
    base = base.merge(low, on=["date", "code"], how="left")

    log.info("复现 ③ 景气动量(PIT 前向填充)...")
    prop = build_prosperity()
    prop["code"] = prop["code"].astype(str).str.zfill(6)
    base = pd.merge_asof(
        base.sort_values("date"), prop.sort_values("date"),
        on="date", by="code", direction="backward", allow_exact_matches=True)

    log.info("复现 ② 规模(akshare)...")
    size_map = build_size()
    base["size"] = base["code"].map(size_map)

    new_factors = ["prosperity", "low_att"] + (["size"] if size_map else [])
    res = {"ic60_full": {}, "ic60_hs300": {}, "ic60_non_hs300": {},
           "ic60_size_recent": {}}
    for f in new_factors:
        log.info("IC: %s", f)
        res["ic60_full"][f] = round(ic_for(base, f), 4)
        res["ic60_hs300"][f] = round(ic_for(base[base["is_hs300"]], f), 4)
        res["ic60_non_hs300"][f] = round(ic_for(base[~base["is_hs300"]], f), 4)
    # size 仅近期窗口(快照排名稳定)
    base_r = base[base["date"] >= pd.Timestamp(SIZE_WINDOW)]
    res["ic60_size_recent"]["size"] = round(ic_for(base_r, "size"), 4)
    res["size_window"] = SIZE_WINDOW

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("\n" + "=" * 86)
    print("文档因子复现 · IC60 (PIT, 截面rank-IC, 做多高值=正IC盈利)")
    print("=" * 86)
    print(f"{'因子':14s} {'IC60全':>9s} {'IC60_HS300':>13s} {'IC60_非HS300':>15s} {'备注':>10s}")
    label = {"prosperity": "景气动量", "low_att": "低关注", "size": "规模"}
    for f in new_factors:
        print(f"{f:14s} {res['ic60_full'][f]:+9.3f} "
              f"{res['ic60_hs300'][f]:+13.3f} {res['ic60_non_hs300'][f]:+15.3f} {label.get(f, ''):>8s}")
    if size_map:
        print(f"{'size(近期'+SIZE_WINDOW[:4]+')':14s} {res['ic60_size_recent']['size']:+9.3f} "
              f"{'-':>13s} {'-':>15s} {'快照':>8s}")
    print("=" * 86)
    if not size_map:
        print("⚠️ size 经代理被墙, 本次跳过; 可开 v2rayN 或换腾讯源后补.")
    print("注: size 用 akshare 当前流通市值快照, 仅近期(2018+)排名稳定, 全历史 IC 无意义.")


if __name__ == "__main__":
    main()
