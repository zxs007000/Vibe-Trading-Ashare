"""build_industry_map_cninfo.py — 用 cninfo(巨潮) 证监会行业 给面板全量股票打行业标签.

背景与取舍(透明记录):
  用户要"行业中性"源. 首选申万(legulegu.com):
    - stock_worm 自带 industry 字段走 eastmoney -> 本沙箱被墙
    - legulegu 申万成分页可爬, 但爬到第5个行业后被阿里云WAF 504限流(持续10min+),
      仅拿到 5 个申万一级(覆盖面板24%), 不足以做跨截面中性化
  退路(cninfo, 证监会行业分类):
    - ak.stock_profile_cninfo(symbol) 返回 所属行业(证监会大类, ~90类), 数据源 cninfo 在本沙箱可达
    - 逐只(1847只)取, 全量覆盖; 退市股 cninfo 无资料 -> 行业=NaN(中性化时按缺失处理)
  说明: 证监会行业 与 申万 口径不同, 但中性化目的(去除行业暴露)相同, 故作为可达的替代源.

输出:
  /workspace/stock_worm/data/csrc_industry_map.parquet  列: [code, csrc_industry]
  /workspace/stock_worm/data/csrc_industry_map.csv      同上

用法:
  python build_industry_map_cninfo.py            # 全量(面板1847只, 约3-5min)
  LIMIT=50 python build_industry_map_cninfo.py   # 小批量验证
断点续拉: 已完成 code 从输出 parquet 读取, 跳过
"""
from __future__ import annotations
import os, sys, time, warnings, concurrent.futures
from pathlib import Path
import numpy as np, pandas as pd, akshare as ak
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/stock_worm")

PANEL = Path("/workspace/stock_worm/data/ashare_daily_panel_survivorfree.parquet")
OUT_PKL = Path("/workspace/stock_worm/data/csrc_industry_map.parquet")
OUT_CSV = Path("/workspace/stock_worm/data/csrc_industry_map.csv")
LIMIT = int(os.environ.get("LIMIT", "0")) or None
WORKERS = int(os.environ.get("WORKERS", "10"))
SLEEP = float(os.environ.get("SLEEP", "0.05"))


def get_industry(code: str):
    """返回 (code, 所属行业 or None)."""
    try:
        df = ak.stock_profile_cninfo(symbol=code[:6])
        if df is not None and len(df) and "所属行业" in df.columns:
            v = df["所属行业"].iloc[0]
            return code, (str(v).strip() if pd.notna(v) else None)
        return code, None
    except Exception:
        return code, None


def main():
    t0 = time.time()
    panel = pd.read_parquet(PANEL)
    codes = list(panel["code"].drop_duplicates())
    print(f"面板 code 数: {len(codes)}")

    done = {}
    if OUT_PKL.exists():
        ex = pd.read_parquet(OUT_PKL)
        for _, r in ex.iterrows():
            done[r["code"]] = None if pd.isna(r["csrc_industry"]) else r["csrc_industry"]
        print(f"  已有映射 {len(done)} 只, 续拉剩余")

    todo = [c for c in codes if c not in done]
    if LIMIT:
        todo = todo[:LIMIT]
        print(f"[LIMIT] 仅 {LIMIT} 只")
    print(f"待拉 {len(todo)} 只 (workers={WORKERS}) ...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for c, ind in ex.map(get_industry, todo):
            done[c] = ind
            time.sleep(SLEEP)

    rows = [{"code": c, "csrc_industry": done[c]} for c in codes]
    mp = pd.DataFrame(rows)
    mp.to_parquet(OUT_PKL, index=False)
    mp.to_csv(OUT_CSV, index=False)
    cov = mp["csrc_industry"].notna().sum()
    print(f"\n映射完成: {len(mp)} 只, 有行业 {cov} ({cov/len(mp)*100:.1f}%), 缺失 {len(mp)-cov}")
    print(f"行业类别数: {mp['csrc_industry'].nunique()}")
    print(f"-> {OUT_PKL}")
    print(f"耗时 {time.time()-t0:.1f}s")
    print("\nTop 行业:")
    print(mp["csrc_industry"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
