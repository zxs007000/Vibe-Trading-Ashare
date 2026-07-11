"""crawl_fundamentals_all.py — 把基本面因子爬取扩展到全 1803 只(并行 + 断点续爬)

复用 build_fundamental_factors 的 _fetch_one / build_daily, 仅把顺序爬改为并行:
  - 目标码 = 去生存偏差面板全部 1803 只
  - ThreadPoolExecutor 并行 fetch, 失败码打印后跳过(下轮续爬补齐)
  - FUND_RAW 增量缓存; 跑完触发 build_daily 重建全覆盖日线因子

用法:
  python backtest/crawl_fundamentals_all.py test   # 冒烟: 仅前20码
  python backtest/crawl_fundamentals_all.py all    # 全量 1803 码 + 重建
"""
import sys, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import build_fundamental_factors as BF

SF_PANEL = BF.SF_PANEL
FUND_RAW = BF.FUND_RAW


def _all_codes():
    p = pd.read_parquet(SF_PANEL)
    return sorted(p["code"].unique().tolist())


def _work(code):
    try:
        res = BF._fetch_one(code)
        recs = []
        for key, ser in res.items():
            if ser is None:
                continue
            for dt, v in ser.items():
                recs.append({"code": code, "report_date": dt, key: float(v)})
        return recs
    except Exception as e:
        return ("ERR", code, repr(e)[:90])


def crawl_parallel(workers=8, save_every=200, limit=None):
    codes = _all_codes()
    if limit:
        codes = codes[:limit]
    done = set()
    records = []
    if FUND_RAW.exists():
        cache = pd.read_parquet(FUND_RAW)
        done = set(cache["code"].unique().tolist())
        records = cache.to_dict("records")
        print(f"[crawl] 续爬: 已有 {len(done)} 码")
    todo = [c for c in codes if c not in done]
    print(f"[crawl] 目标 {len(codes)} 码, 待爬 {len(todo)}, workers={workers}")
    n = 0; errs = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_work, c): c for c in todo}
        for fut in as_completed(futs):
            r = fut.result()
            if isinstance(r, tuple) and r[0] == "ERR":
                errs += 1
                if errs <= 20:
                    print(f"[crawl] ERR {r[1]}: {r[2]}")
                continue
            records.extend(r); n += 1
            if n % save_every == 0:
                pd.DataFrame(records).to_parquet(FUND_RAW)
                print(f"[crawl] 进度 {n}/{len(todo)} 成功, 记录 {len(records)}")
    pd.DataFrame(records).to_parquet(FUND_RAW)
    print(f"[crawl] 完成: 成功 {n}, 失败 {errs}, 总记录 {len(records)} -> {FUND_RAW}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if mode == "test":
        crawl_parallel(workers=6, limit=20)
    else:
        crawl_parallel(workers=8)
        print("[build] 重建全覆盖日线因子 ...")
        BF.build_daily()
    print(f"耗时 {time.time()-t0:.1f}s")
