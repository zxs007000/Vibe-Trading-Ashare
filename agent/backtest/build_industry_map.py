"""build_industry_map.py — 构建 code -> 申万一级行业 映射(去生存者偏差中性化用).

背景:
  用户要求因子做"行业中性", 但 stock_worm 自带的 industry 字段
  (company_info.f127 / sector_membership) 走 eastmoney, 在本沙箱被墙.
  按用户授权("实在不行, 咱们就用爬"), 改用 akshare 申万(数据源 legulegu.com, eastmoney 无关).

方法(最小请求, 鲁棒):
  - ak.sw_index_first_info() 取 31 个申万一级行业 (行业代码 801xxx.SI + 名称)
  - 每个一级行业访问 legulegu 成分页
      https://legulegu.com/stockdata/index-composition?industryCode=801xxx.SI
    用 requests(UA) 取 HTML, pd.read_html(io.StringIO) 解析(直连会 403, 加 UA 即可)
  - 每页直接含 [股票代码, 申万1级] 列 -> 拼成 code->申万1级 全量映射
  - pd.read_html 会把页内 JSON-LD 元数据误解析成以 '{' 开头的列, 过滤掉即可

输出:
  /workspace/stock_worm/data/sw_industry_map.parquet  列: [code, sw1_code, sw1_name]
  /workspace/stock_worm/data/sw_industry_map.csv      同上(便于人读/核对)

用法:
  python build_industry_map.py                 # 全量 31 个一级行业
  LIMIT=5 python build_industry_map.py         # 小批量验证
断点续拉: 每页缓存 pkl, 重跑跳过已抓取行业
"""
from __future__ import annotations
import os, sys, io, time, warnings, concurrent.futures
from pathlib import Path
import numpy as np, pandas as pd, requests
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/stock_worm")
import akshare as ak

LEGU = "https://legulegu.com/stockdata/index-composition?industryCode={code}"
OUT_PKL = Path("/workspace/stock_worm/data/sw_industry_map.parquet")
OUT_CSV = Path("/workspace/stock_worm/data/sw_industry_map.csv")
CACHE_DIR = Path("/workspace/stock_worm/data/industry_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LIMIT = int(os.environ.get("LIMIT", "0")) or None
SLEEP = float(os.environ.get("SLEEP", "0.3"))
WORKERS = int(os.environ.get("WORKERS", "6"))
UA = {"User-Agent": "Mozilla/5.0"}


def fetch_one(code: str, name: str, retries: int = 4):
    """抓一个一级行业成分页, 返回 (code, df) 或 (code, None). 失败重试(应对偶发封页)."""
    url = LEGU.format(code=code)
    last_err = None
    for attempt in range(retries):
        if attempt:
            time.sleep(2.0 * attempt)
        try:
            r = requests.get(url, headers=UA, timeout=20)
            html = r.text
            # 粗略识别封页/挑战页(无真实内容)
            if "股票代码" not in html and "index-composition" not in html and len(html) < 2000:
                last_err = f"疑似封页(len={len(html)})"
                continue
            tables = pd.read_html(io.StringIO(html))
        except Exception as e:
            last_err = f"{e!r}"
            continue
        if not tables:
            last_err = "No tables found"
            continue
        t = tables[0]
        real = [c for c in t.columns if not str(c).startswith("{")]
        t = t[real]
        if "股票代码" not in t.columns or "申万1级" not in t.columns:
            last_err = f"列缺失:{real}"
            continue
        df = t[["股票代码", "申万1级"]].copy()
        df["股票代码"] = df["股票代码"].astype(str).str.strip()
        df = df[df["股票代码"].str.match(r"^\d{6}\.(SH|SZ|BJ)$")]
        df["sw1_code"] = code
        df = df.rename(columns={"股票代码": "code", "申万1级": "sw1_name"})
        if not len(df):
            last_err = "空表"
            continue
        return code, df
    print(f"  {code} 抓取失败(重试{retries}次): {last_err}")
    return code, None


def main():
    t0 = time.time()
    first = ak.sw_index_first_info()[["行业代码", "行业名称"]]
    first = first.rename(columns={"行业代码": "code", "行业名称": "name"})
    n = len(first)
    print(f"申万一级行业: {n} 个")
    if LIMIT:
        first = first.head(LIMIT)
        print(f"[LIMIT] 仅前 {LIMIT} 个")

    cached = [r for _, r in first.iterrows() if (CACHE_DIR / f"{r['code']}.pkl").exists()]
    to_pull = [r for _, r in first.iterrows() if not (CACHE_DIR / f"{r['code']}.pkl").exists()]
    frames = [pd.read_pickle(CACHE_DIR / f"{r['code']}.pkl") for r in cached]
    if cached:
        print(f"  缓存命中 {len(cached)} 个")
    if to_pull:
        print(f"  并发抓取 {len(to_pull)} 个 (workers={WORKERS}) ...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for code, df in ex.map(lambda r: fetch_one(r["code"], r["name"]), to_pull):
                if df is not None and len(df):
                    df.to_pickle(CACHE_DIR / f"{code}.pkl")
                    frames.append(df)
                    print(f"  {code} {len(df)} 只")
                else:
                    print(f"  {code} 无数据")
                time.sleep(SLEEP)

    if not frames:
        print("无数据, 退出"); return
    mp = pd.concat(frames, ignore_index=True)
    # 同一 code 可能跨行业(应极少), 去重保留首次出现
    before = len(mp)
    mp = mp.drop_duplicates("code", keep="first")
    print(f"\n映射行数: {before} -> 去重 {len(mp)} (唯一 code {mp['code'].nunique()})")
    print(f"覆盖一级行业数: {mp['sw1_name'].nunique()} / {n}")
    mp = mp.sort_values("code").reset_index(drop=True)
    mp.to_parquet(OUT_PKL, index=False)
    mp.to_csv(OUT_CSV, index=False)
    print(f"-> {OUT_PKL}")
    print(f"耗时 {time.time()-t0:.1f}s")
    # 覆盖报告
    print("\n一级行业分布(前10):")
    print(mp["sw1_name"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
