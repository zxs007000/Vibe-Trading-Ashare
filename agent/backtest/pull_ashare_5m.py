"""pull_ashare_5m.py — 拉取全 CSI300 的 5 分钟 K 线(带 amount)并备份到 stock_worm/data。

设计:
  - 成分股用 akshare 取全 CSI300(300 只), 转 .SH/.SZ/.BJ 后缀。
  - 经 astockdata_loader(通达信 TCP, 现已保留 amount 列) 拉 5m。
  - 时间轴默认 2023-01-01~2026-06-30, 覆盖上涨/下跌/反弹/回调/盘整各行情。
  - 断点续拉: 已存在的缓存会被加载, 跳过已拉取的代码; 每 20 只 checkpoint 落盘。
  - 备份位置: /workspace/stock_worm/data/ashare_5m_cache.pkl(本地常驻, 不入库)。

用法:
  python backtest/pull_ashare_5m.py [--start 2023-01-01] [--end 2026-06-30]
"""
from __future__ import annotations

import argparse
import sys
import time
import pickle
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))      # agent/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from backtest.loaders.astockdata_loader import DataLoader

BACKUP = Path("/workspace/stock_worm/data/ashare_5m_cache.pkl")
START, END = "2023-01-01", "2026-06-30"
CHECKPOINT_EVERY = 20


def _suffix(code: str) -> str:
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SH"


def get_csi300() -> list[str]:
    import akshare as ak
    df = ak.index_stock_cons(symbol="000300")
    codes = [str(c) for c in df["品种代码"].tolist()]
    return [_suffix(c) for c in codes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    # 断点续拉: 加载已有缓存
    cache = {}
    if BACKUP.exists():
        try:
            cache = pickle.load(open(BACKUP, "rb"))
            print(f"  载入已有缓存 {len(cache)} 只, 续拉剩余")
        except Exception:
            cache = {}

    codes = get_csi300()
    print(f"  CSI300 共 {len(codes)} 只, 区间 {args.start}~{args.end}")

    dl = DataLoader()
    t0 = time.time()
    done = 0
    for i, code in enumerate(codes):
        if code in cache and len(cache[code]) > 100:
            continue
        ok = False
        for attempt in range(3):
            try:
                r = dl.fetch([code], args.start, args.end, interval="5m")
                for k, v in r.items():
                    if v is not None and not v.empty:
                        cache[k] = v
                        ok = True
                if ok:
                    break
            except Exception:
                time.sleep(0.5)
        done += 1
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(codes):
            pickle.dump(cache, open(BACKUP, "wb"))
            print(f"    {i+1}/{len(codes)} 已缓存 {len(cache)} 只, 耗时 {time.time()-t0:.0f}s")
    pickle.dump(cache, open(BACKUP, "wb"))
    print(f"  完成: {len(cache)} 只 -> {BACKUP}")


if __name__ == "__main__":
    main()
