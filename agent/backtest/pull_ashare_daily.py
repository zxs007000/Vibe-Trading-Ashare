"""pull_ashare_daily.py — 本地化日线数据湖(中证800/1000, 2006~2026-06-30).

为深度学习 / 回测统一建本地数据, 省得每次重拉:
  - 成分: 沪深300 + 中证500 + 中证1000 (akshare, ~1557 只, 覆盖大/中/小盘)
  - 数据源: **stock_worm 第一优先**(stcok_worm.mootdx_source.get_kline_history, 通达信 TCP)
  - 数据: 翻页拉全历史日线 OHLCV + amount, 跨度 20 年
  - 落盘: /workspace/stock_worm/data/ashare_daily_cache.pkl (dict: code -> DataFrame)
  - 断点续拉: 已落盘的自动跳过; 每 CHECKPOINT 只增量写盘; 可反复跑补缺失

设计要点:
  - 经首选数据源 stock_worm 拉取(非 astockdata_loader/mootdx 另一封装), 保证来源一致.
  - 单只失败不影响整体; 失败股记录在失败集, 下次重跑会重试.
  - 长期任务: 网络抖动/超时后重跑即可从断点继续.

用法:
  python backtest/pull_ashare_daily.py
"""
from __future__ import annotations
import os, sys, pickle, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))   # agent/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

# 首选数据源 = stock_worm (用户指定 A股第一优先). 用其 mootdx_source.get_kline
# (通达信 TCP, 含 amount) 手动从最新页往前翻页拼全历史. 不走 astockdata_loader(mootdx 另一封装).
# 注: stock_worm 自带的 get_kline_history 从最老 offset 往前翻、遇空页即 break,
# 对历史不足 total 根(2015 后上市)的股票会漏数据, 故这里手动从 offset=0(最新)往前翻、遇空即止.
import stcok_worm
from stcok_worm.mootdx_source import get_kline

CACHE = Path("/workspace/stock_worm/data/ashare_daily_cache.pkl")
START, END = "2006-01-01", "2026-06-30"
CHECKPOINT = 50          # 每拉满 50 只写一次盘
# 成分库: 沪深300 + 中证500 + 中证1000 (~1800 只, 覆盖大/中/小盘), 稳健超过 800
INDEX_CODES = ["000300", "000905", "000852"]


def _suffix(code: str) -> str | None:
    """6 位代码 -> .SH/.SZ 后缀(北交所 4/8/92 开头通达信 std 不支持, 跳过)."""
    c = code.strip()
    if len(c) != 6 or not c.isdigit():
        return None
    if c.startswith("92"):          # 北交所新代码(920xxx/921xxx...)
        return None
    if c[0] in ("6", "9"):
        return f"{c}.SH"
    if c[0] in ("0", "3"):
        return f"{c}.SZ"
    return None


import signal

def _fetch_with_timeout(code, start, end, interval, secs: int = 90):
    """带超时保护地经 stock_worm 拉取单只(避免个别股票 TDX 调用挂起拖垮长任务)."""
    def _handler(signum, frame):
        raise TimeoutError("stock_worm fetch timeout")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(secs)
    try:
        return _fetch_stockworm(code, start, end, interval)
    except TimeoutError:
        print(f"  ✗ {code} 超时({secs}s), 跳过", flush=True)
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _get_universe() -> list[str]:
    """全市场上市A股清单(经 akshare 取全量代码, 去重; 经 _suffix 排除北交所4/8).

    退市股不在此拉(避免 mootdx 对退市股逐个超时挂死), 由 build_survivorship_free_panel.py 走腾讯补齐.
    """
    import akshare as ak
    codes: list[str] = []
    try:
        df = ak.stock_info_a_code_name()
        for raw in df["code"].astype(str).tolist():
            s = _suffix(raw)
            if s and s not in codes:
                codes.append(s)
    except Exception as e:
        print(f"  ⚠ 全市场清单获取失败: {repr(e)[:80]}")
    return codes


# 通达信频率编码: 9=日K, 0=5m, 1=15m, 2=30m, 3=1h
_FREQ = {"1D": 9}
_PAGE = 800
_MAX_OFFSET = 6000   # 20 年日线 ≈ 4900 根, 留余量


def _fetch_stockworm(code: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
    """经首选数据源 stock_worm 拉日线(从最新页往前翻, 遇空即止, 含 amount)."""
    freq = _FREQ[interval]
    recs: list[dict] = []
    seen: set[str] = set()
    try:
        for off in range(0, _MAX_OFFSET, _PAGE):
            page = get_kline(code, freq, count=_PAGE, offset=off)
            if not page:
                break  # 已到最老可用根, 停止
            for r in page:
                if r["date"] in seen:
                    continue
                seen.add(r["date"])
                recs.append(r)
    except Exception as e:
        print(f"  ✗ {code} stock_worm 拉取异常: {repr(e)[:80]}", flush=True)
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


def main():
    t0 = time.time()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    print(f"  已落盘 {len(cache)} 只, 目标=全市场上市A股(akshare stock_info_a_code_name)")

    universe = _get_universe()
    print(f"  全市场上市A股 {len(universe)} 只")

    pending = [c for c in universe if c not in cache]
    # 分批拉取: PULL_LIMIT 控制单次拉取上限(干净退出+断点续拉), 不设置则一次拉完
    pull_limit = int(os.environ.get("PULL_LIMIT", "0")) or None
    if pull_limit:
        pending = pending[:pull_limit]
        print(f"  [PULL_LIMIT={pull_limit}] 本批仅拉前 {len(pending)} 只")
    print(f"  待拉取 {len(pending)} 只 (已完成 {len(cache)} 只)")
    if not pending:
        print("  全部已完成, 无需拉取。如需刷新请删除缓存后重跑。")
        return

    done = 0
    failed = []
    for i, code in enumerate(pending, 1):
        try:
            df = _fetch_with_timeout(code, START, END, "1D")
            if df is not None and not df.empty:
                cache[code] = df
                done += 1
            else:
                failed.append(code)
        except Exception as e:
            failed.append(code)
            print(f"  ✗ {code} 失败: {repr(e)[:80]}", flush=True)

        if i % CHECKPOINT == 0 or i == len(pending):
            pickle.dump(cache, open(CACHE, "wb"))
            el = time.time() - t0
            rate = i / el if el > 0 else 0
            eta = (len(pending) - i) / rate if rate > 0 else 0
            print(f"  进度 {i}/{len(pending)}  已存盘 {len(cache)} 只  "
                  f"耗时{el/60:.1f}m  速率{rate:.2f}只/s  ETA{eta/60:.1f}m", flush=True)

    pickle.dump(cache, open(CACHE, "wb"))
    # 统计
    rows = [len(v) for v in cache.values()]
    print(f"\n  完成: 成功 {done} 只, 失败 {len(failed)} 只")
    if failed:
        print(f"  失败样本(前20): {failed[:20]}")
    if rows:
        total_rows = sum(rows)
        print(f"  数据湖: {len(cache)} 只, 合计 {total_rows:,} 行日线, "
              f"单只均值 {np.mean(rows):.0f} 行, 最早覆盖 "
              f"{min(v.index.min() for v in cache.values()).date()} ~ "
              f"{max(v.index.max() for v in cache.values()).date()}")
        print(f"  落盘: {CACHE}  ({CACHE.stat().st_size/1e6:.0f} MB)")
    # 同步导出规整 parquet 面板(供回测/深度学习直读)
    try:
        _export_parquet(cache, CACHE.with_suffix(""))  # 仅用于尺寸提示, 实际路径见下
    except Exception as e:
        print(f"  ⚠ parquet 导出跳过: {repr(e)[:80]}")
    print(f"  总耗时 {(time.time()-t0)/60:.1f} 分钟")


def _export_parquet(cache: dict, _unused) -> None:
    """把缓存导出为 date×code 规整面板 parquet(与 pickle 同目录)."""
    out = CACHE.parent / "ashare_daily_panel.parquet"
    frames = []
    for code, v in cache.items():
        if not isinstance(v, pd.DataFrame) or v.empty or "close" not in v.columns:
            continue
        d = v.copy()
        d.index = d.index.rename("date")
        d = d.reset_index()
        d["code"] = code
        frames.append(d)
    if not frames:
        return
    panel = pd.concat(frames, ignore_index=True)
    keep = [x for x in ("date", "code", "open", "high", "low", "close", "volume", "amount") if x in panel.columns]
    panel = panel[keep].sort_values(["date", "code"]).reset_index(drop=True)
    panel.to_parquet(out, index=False)
    print(f"  parquet: {out}  {out.stat().st_size/1e6:.0f}MB  {len(panel):,}行 股票={panel['code'].nunique()}")


if __name__ == "__main__":
    main()
