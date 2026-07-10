"""pull_ashare_5m.py — 拉取全 CSI300 的 5 分钟 K 线(带 amount)并备份到 stock_worm/data。

设计:
  - 成分股用 akshare 取全 CSI300(300 只), 转 .SH/.SZ 后缀。
  - 数据源: 首选 stock_worm (stcok_worm.mootdx_source.get_kline, 通达信 TCP, 含 amount),
    从最新页往前翻页拼全历史, 不走 astockdata_loader(mootdx 另一封装)。
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
import signal
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))      # agent/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

# 首选数据源 = stock_worm (用户指定 A股第一优先)
import stcok_worm
from stcok_worm.mootdx_source import get_kline

BACKUP = Path("/workspace/stock_worm/data/ashare_5m_cache.pkl")
START, END = "2023-01-01", "2026-06-30"
CHECKPOINT_EVERY = 20
_FREQ = 0          # 5 分钟
_PAGE = 800
_MAX_OFFSET = 26000  # 3.5 年 5m ≈ 21600 根, 留余量


def _fetch_stockworm_5m(code: str, start: str, end: str) -> pd.DataFrame | None:
    """经首选数据源 stock_worm 拉 5m(从最新页往前翻, 遇空即止, 含 amount)."""
    recs: list[dict] = []
    seen: set[str] = set()
    try:
        for off in range(0, _MAX_OFFSET, _PAGE):
            page = get_kline(code, _FREQ, count=_PAGE, offset=off)
            if not page:
                break
            for r in page:
                if r["date"] in seen:
                    continue
                seen.add(r["date"])
                recs.append(r)
    except Exception as e:
        print(f"  ✗ {code} stock_worm 5m 异常: {repr(e)[:80]}", flush=True)
        return None
    if not recs:
        return None
    df = pd.DataFrame(recs)
    df["trade_date"] = pd.to_datetime(df["date"])
    df = df.set_index("trade_date").sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume", "amount") if c in df.columns]
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[keep].loc[pd.Timestamp(start):end_ts]
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df if not df.empty else None


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
    cache = {}
    if BACKUP.exists():
        try:
            cache = pickle.load(open(BACKUP, "rb"))
            print(f"  载入已有缓存 {len(cache)} 只, 续拉剩余")
        except Exception:
            cache = {}

    codes = get_csi300()
    print(f"  CSI300 共 {len(codes)} 只, 区间 {args.start}~{args.end} (数据源: stock_worm)")

    t0 = time.time()
    done = 0
    for i, code in enumerate(codes):
        if code in cache and len(cache[code]) > 100:
            continue
        # 单只超时保护(避免 TDX 调用挂起拖垮长任务)
        def _h(signum, frame):
            raise TimeoutError()
        old = signal.signal(signal.SIGALRM, _h)
        signal.alarm(120)
        try:
            df = _fetch_stockworm_5m(code, args.start, args.end)
        except TimeoutError:
            df = None
            print(f"  ✗ {code} 5m 超时, 跳过", flush=True)
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old)
        if df is not None and not df.empty:
            cache[code] = df
        done += 1
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(codes):
            pickle.dump(cache, open(BACKUP, "wb"))
            print(f"    {i+1}/{len(codes)} 已缓存 {len(cache)} 只, 耗时 {time.time()-t0:.0f}s", flush=True)
    pickle.dump(cache, open(BACKUP, "wb"))
    print(f"  完成: {len(cache)} 只 -> {BACKUP}")


if __name__ == "__main__":
    main()
