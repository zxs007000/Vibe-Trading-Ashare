#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnstock 基本面仓库构建器 (混合架构·快字段源)
=============================================
中国证券网 data.cnstock.com 财报摘要: 全市场 4000+ 只, 每只 stock_detail 给 2001-2024 季报历史。
映射出原型 load_fundamentals 需要的"快字段"(无需东财三大表):
    gross_margin   <- 毛利率(本期)
    roe            <- 净资产收益率(本期)
    eps            <- 每股收益(本期)
    netprofit_yoy  <- 净利润同比
    bvps           <- 每股净资产(本期)
    ocf_ps         <- 每股现金流(本期)
    ocf_netprofit  <- 每股现金流(本期) / 每股收益(本期)  (份额相消 = OCF/净利润, 无需绝对额)
    *_yoy 变体      <- 各指标同比(供后续扩展)
缺口(由东财资产负债表补缺): net_margin(需绝对营收), operate_income_yoy(需绝对营收),
                              debt_ratio, current_ratio。
产物: /workspace/stocklake/fundamentals_cnstock/{code}.parquet  (index=REPORT_DATE)
幂等: 已存在则跳过。
"""
import sys, os, time, json, logging
from pathlib import Path
import pandas as pd

sys.path.insert(0, "/workspace/stock-worm")
from stcok_worm import cnstock

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cnstock_fund")

LAKE = Path(os.environ.get("STOCKWORM_LAKE", "/workspace/stocklake"))
OUT = LAKE / "fundamentals_cnstock"
OUT.mkdir(parents=True, exist_ok=True)
META = LAKE / "metadata" / "stock_list.csv"

# ⚠️ 重要: data.cnstock.com 对高并发极度敏感, 曾因 workers=8 触发整域 403 封禁(IP级)。
# 本模块的 cnstock._get() 已用全局信号量强制并发<=SAFE_CONCURRENCY(=2) 并加间隔+退避,
# 故即便 workers 设大也会被限流排队; 这里显式降到 2, 与上游保护一致, 切勿再开高并发。
WORKERS = 2
BATCH = 200


def map_one(code):
    try:
        d = cnstock.stock_detail(code)
    except Exception as e:
        return code, None, f"fetch:{e}"
    if d is None or len(d) == 0:
        return code, None, "empty"
    # 列映射
    colmap = {
        "毛利率(本期)": "gross_margin",
        "净资产收益率(本期)": "roe",
        "每股收益(本期)": "eps",
        "净利润同比": "netprofit_yoy",
        "每股净资产(本期)": "bvps",
        "每股现金流(本期)": "ocf_ps",
        "毛利率(同比)": "gross_margin_yoy",
        "净资产收益率(同比)": "roe_yoy",
        "每股收益(同比)": "eps_yoy",
        "每股净资产(同比)": "bvps_yoy",
        "每股现金流(同比)": "ocf_yoy",
    }
    out = pd.DataFrame(index=d.index)  # index = 报告期(datetime)
    for src, dst in colmap.items():
        if src in d.columns:
            out[dst] = pd.to_numeric(d[src], errors="coerce")
    # ocf_netprofit = 每股现金流 / 每股收益 (份额相消)
    if "ocf_ps" in out.columns and "eps" in out.columns:
        out["ocf_netprofit"] = out["ocf_ps"] / out["eps"].replace(0, pd.NA)
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    out = out.dropna(how="all")
    if len(out) == 0:
        return code, None, "no_cols"
    return code, out, "ok"


def main():
    sl = pd.read_csv(META, dtype=str)
    codes = sl["code"].tolist()
    log.info("股票池: %d 只", len(codes))
    done = {f.stem for f in OUT.glob("*.parquet")}
    todo = [c for c in codes if c not in done]
    log.info("待处理 %d / 总 %d (已完成 %d)", len(todo), len(codes), len(done))
    if not todo:
        log.info("全部已完成 ✅")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(map_one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            code, out, status = fut.result()
            if status == "ok" and out is not None:
                out.to_parquet(OUT / f"{code}.parquet")
                ok += 1
            else:
                fail += 1
            if i % BATCH == 0:
                el = time.time() - t0
                log.info("[%d/%d] ok=%d fail=%d 累计%.1fmin", i, len(todo), ok, fail, el / 60)
    log.info("完成: ok=%d fail=%d  产物目录 %s", ok, fail, OUT)
    # 写快照
    json.dump({"done": len(done) + ok, "total": len(codes), "fail": fail},
              open(LAKE / "metadata" / "cnstock_fund_status.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
