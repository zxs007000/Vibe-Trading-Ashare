"""build_survivorship_free_panel.py — 构建去生存者偏差面板(含退市股).

用户目标: '从前往后收集股票, 把退市的也能收集到' —— 去掉当前快照的生存者偏差.

数据源(均在本沙箱验证可用):
  - akshare: stock_info_sh_delist / stock_info_sz_delist -> 沪深退市清单(共364只, 含上市/退市日期)
  - 腾讯日线接口 web.ifzq.gtimg.cn -> 直接返回 JSON, 不封IP, 可拿退市股历史K线
    (stock_worm.tencent 即此接口; 本脚本自行分段调用以突破腾讯单次~640条上限)
  - 现有面板 ashare_daily_panel.parquet (1489只当前股, 含精确 amount) 作为基础

方法:
  - 对每只退市股, 按自然年分段拉腾讯日线, 合并去重 -> 拿到 2001 起全历史(腾讯历史始于~2001)
  - 腾讯日线无成交额(amount), 用 volume(手)*100*均价 估算(与现有精确 amount 同量级, 做流动性因子足够)
  - 现有1489只 + 退市N只 -> 按(date,code)去重合并 -> 新面板
  - 断点续拉: 每只退市股拉完存 cache pkl, 重跑跳过

用法:
  LIMIT=3 python build_survivorship_free_panel.py      # 小批量验证(前3只退市)
  python build_survivorship_free_panel.py              # 全量(364只退市, 约20-30分钟)
  SLEEP=0.3 LIMIT=50 python ...                         # 自定义间隔/批量
"""
from __future__ import annotations
import os, sys, time, warnings, concurrent.futures
from pathlib import Path
import numpy as np, pandas as pd, requests
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/stock_worm")
import akshare as ak

TENCENT_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
EXIST_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel.parquet")
OUT_PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel_survivorfree.parquet")
CACHE_DIR = Path("/workspace/stock_worm/data/delist_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LIMIT = int(os.environ.get("LIMIT", "0")) or None
SLEEP = float(os.environ.get("SLEEP", "0.2"))
WORKERS = int(os.environ.get("WORKERS", "6"))
UA = {"User-Agent": "Mozilla/5.0"}
COLS = ["date", "code", "open", "high", "low", "close", "volume", "amount"]


def pull_segment(prefix_code: str, start: str, end: str):
    """拉腾讯一段日线, 返回 records(含估算 amount). 空/异常返回 []."""
    url = f"{TENCENT_KLINE}?param={prefix_code},day,{start},{end},2000,qfq"
    try:
        d = requests.get(url, headers=UA, timeout=15).json()
    except Exception:
        return []
    inner = (d.get("data") or {}).get(prefix_code) or {}
    rows = inner.get("qfqday") or inner.get("day") or []
    recs = []
    for r in rows:
        try:
            o, h, l, c, v = float(r[1]), float(r[3]), float(r[4]), float(r[2]), float(r[5])
        except Exception:
            continue
        avg = (o + c) / 2.0
        amt = v * 100.0 * avg          # 手->股->元 估算成交额
        recs.append({"date": str(r[0]), "open": o, "high": h, "low": l,
                     "close": c, "volume": v, "amount": amt})
    return recs


def get_delist_universe():
    """akshare 拿沪深退市清单, 统一为 9位带后缀 code."""
    out = []
    for fn, suff in [("stock_info_sh_delist", "SH"), ("stock_info_sz_delist", "SZ")]:
        df = getattr(ak, fn)()
        code_col = "公司代码" if "公司代码" in df.columns else "证券代码"
        name_col = "公司简称" if "公司简称" in df.columns else "证券简称"
        delist_col = "暂停上市日期" if suff == "SH" else "终止上市日期"
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            if not code.isdigit():
                continue
            dd = row[delist_col]
            out.append({
                "code": f"{code}.{suff}",
                "raw": code,
                "name": str(row[name_col]),
                "list_date": str(row["上市日期"])[:10],
                "delist_date": (str(dd)[:10] if pd.notna(dd) else None),
            })
    return out


def pull_one(u: dict):
    """分段(按年)拉一只退市股全历史, 返回带 code 列的 DataFrame 或 None."""
    raw = u["raw"]
    prefix = "sh" if u["code"].endswith("SH") else ("bj" if u["code"].endswith("BJ") else "sz")
    pc = prefix + raw
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = u["list_date"] or "2000-01-01"
    end = u["delist_date"] or today
    recs = []
    y0, y1 = int(start[:4]), int(end[:4])
    for y in range(y0, y1 + 1):
        seg_start, seg_end = f"{y}-01-01", f"{y}-12-31"
        if seg_end > end:
            seg_end = end
        if seg_start > end:
            break
        recs += pull_segment(pc, seg_start, seg_end)
        time.sleep(SLEEP)
    if not recs:
        return None
    df = pd.DataFrame(recs).drop_duplicates("date").sort_values("date")
    df["code"] = u["code"]
    return df


def main():
    t0 = time.time()
    univ = get_delist_universe()
    n_sh = sum(1 for u in univ if u["code"].endswith("SH"))
    n_sz = sum(1 for u in univ if u["code"].endswith("SZ"))
    print(f"退市清单: {len(univ)} 只 (沪{n_sh} 深{n_sz})")
    if LIMIT:
        univ = univ[:LIMIT]
        print(f"[LIMIT] 仅拉前 {LIMIT} 只做验证")
    # 缓存命中的直接读, 未拉的并发拉取(线程池)
    cached = [u for u in univ if (CACHE_DIR / f"{u['code']}.pkl").exists()]
    to_pull = [u for u in univ if not (CACHE_DIR / f"{u['code']}.pkl").exists()]
    frames = [pd.read_pickle(CACHE_DIR / f"{u['code']}.pkl") for u in cached]
    if cached:
        print(f"  缓存命中 {len(cached)} 只")
    if to_pull:
        print(f"  并发拉取 {len(to_pull)} 只 (workers={WORKERS}) ...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for u, df in zip(to_pull, ex.map(pull_one, to_pull)):
                if df is not None and len(df):
                    df.to_pickle(CACHE_DIR / f"{u['code']}.pkl")
                    frames.append(df)
                    print(f"  {u['code']} {u['name']} {len(df)}条 "
                          f"{df['date'].iloc[0]}~{df['date'].iloc[-1]}")
                else:
                    print(f"  {u['code']} 无数据, 跳过")
    if not frames:
        print("无退市数据, 退出"); return
    delist = pd.concat(frames, ignore_index=True)[COLS]
    print(f"\n退市股合并: {delist['code'].nunique()} 只, {len(delist)} 行")

    exist = pd.read_parquet(EXIST_PANEL)[COLS].copy()
    exist["date"] = pd.to_datetime(exist["date"])
    delist["date"] = pd.to_datetime(delist["date"])
    merged = (pd.concat([exist, delist], ignore_index=True)
              .drop_duplicates(["date", "code"]).sort_values(["code", "date"]))
    merged.to_parquet(OUT_PANEL, index=False)
    print(f"合并面板: {merged['code'].nunique()} 只 (原1489 + 退市{delist['code'].nunique()}), "
          f"{len(merged)} 行 -> {OUT_PANEL}")
    print(f"耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
