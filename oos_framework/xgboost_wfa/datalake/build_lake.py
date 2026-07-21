#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股数据湖构建器 — 全A(沪深)含退市 · 近8年日线 + 近8年三大表。

数据源（均经实测验证）:
  上市日线     : 新浪 ak.stock_zh_a_daily            (8年完整, ~0.4s/只)
  退市日线     : 腾讯 tencent.get_kline             (仅最近~640条≈2.5年, 缺口标注)
  上市三大表   : 东财 ak.stock_*_sheet_by_report_em
  退市三大表   : 东财 ak.stock_*_sheet_by_report_delisted_em
  上市清单     : 东财 ak.stock_info_a_code_name
  退市清单     : 沪深 ak.stock_info_sh/sz_delist

用法（每批受 10min 命令上限约束, 断点续传, 反复调用直到完成）:
  python build_lake.py list
  python build_lake.py daily_listed
  python build_lake.py daily_delist
  python build_lake.py stmt_listed
  python build_lake.py stmt_delist
  python build_lake.py manifest
"""
import os, sys, time, json, logging, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/workspace/stock-worm")
import pandas as pd
import akshare as ak
from stcok_worm import tencent
import stcok_worm.fundamentals_ext as fe  # 注入20s超时补丁 + 0.5s限流

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_lake")

LAKE = Path(os.environ.get("STOCKWORM_LAKE", "/workspace/stocklake"))
WS, WE = "2018-07-21", "2026-07-21"
WIN_START_YEAR = 2018
TIME_BUDGET = 8.5 * 60  # 单批运行时间上限(秒)
PROG = LAKE / "metadata" / "progress"


def ensure_dirs():
    for p in [LAKE / "metadata", LAKE / "daily",
              LAKE / "fundamentals" / "income_statement",
              LAKE / "fundamentals" / "balance_sheet",
              LAKE / "fundamentals" / "cash_flow_statement"]:
        p.mkdir(parents=True, exist_ok=True)
    PROG.mkdir(parents=True, exist_ok=True)


def market_of(code):
    code = str(code)
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith("8"):
        return "bj"
    return "sz"


# ───────────────────────── 清单 ─────────────────────────
def build_stock_list():
    ensure_dirs()
    # 上市（多源回退：新浪spot优先，东财code_name兜底；东财间歇抖动）
    listed = pd.DataFrame(columns=["code", "name", "status", "list_date", "delist_date"])
    sources = [
        ("新浪spot", lambda: ak.stock_zh_a_spot().rename(columns={"代码": "code", "名称": "name"})[["code", "name"]]),
        ("东财code_name", lambda: ak.stock_info_a_code_name().rename(columns={"代码": "code", "名称": "name"})[["code", "name"]]),
    ]
    for sname, sfn in sources:
        for attempt in range(3):
            try:
                d = sfn()
                d["code"] = d["code"].astype(str)
                d = d[d["code"].str.startswith(("6", "0", "3"))].reset_index(drop=True)
                if len(d) >= 4000:
                    listed = d.copy()
                    listed["status"] = "上市"
                    listed["list_date"] = ""
                    listed["delist_date"] = ""
                    log.info("上市清单(%s)成功: %d 只", sname, len(listed))
                    break
            except Exception as e:
                log.warning("上市清单(%s)尝试%d失败: %s", sname, attempt + 1, str(e)[:50])
                time.sleep(2)
        if len(listed):
            break

    # 退市（沪深官方源，网络偶发抖动，重试合并取并集以保证拿全）
    delist_seen = {}
    for attempt in range(8):
        rows = []
        for fn in (ak.stock_info_sh_delist, ak.stock_info_sz_delist):
            try:
                d = fn()
                for _, r in d.iterrows():
                    rows.append((str(r.iloc[0]), str(r.iloc[1]), "退市",
                                 str(r.iloc[2]) if len(r) > 2 else "",
                                 str(r.iloc[3]) if len(r) > 3 else ""))
            except Exception as e:
                log.warning("退市清单 %s 失败(尝试%d): %s", fn.__name__, attempt + 1, str(e)[:50])
        for row in rows:
            delist_seen[row[0]] = row
        log.info("退市清单尝试%d: 累计 %d 只", attempt + 1, len(delist_seen))
        if len(delist_seen) >= 350:
            break
        time.sleep(3)
    delist = pd.DataFrame(list(delist_seen.values()), columns=["code", "name", "status", "list_date", "delist_date"])

    all_df = pd.concat([listed, delist], ignore_index=True)
    all_df["market"] = all_df["code"].apply(market_of)
    all_df = all_df.drop_duplicates(subset=["code"], keep="first")
    all_df.to_csv(LAKE / "metadata" / "stock_list.csv", index=False, encoding="utf-8-sig")
    log.info("清单完成: 上市%d 退市%d 合计%d", len(listed), len(delist), len(all_df))
    return all_df


def load_codes(status_filter=None):
    df = pd.read_csv(LAKE / "metadata" / "stock_list.csv", dtype=str)
    if status_filter:
        df = df[df["status"].isin(status_filter)]
    return df["code"].tolist()


# ───────────────────────── 进度/并发 ─────────────────────────
def load_done(path):
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except Exception:
            return set()
    return set()


def save_done(path, s):
    path.write_text(json.dumps(sorted(s), ensure_ascii=False))


def run_batch(name, codes, func, out_dir, done_file, max_workers, time_budget=TIME_BUDGET, empty_is_done=False):
    ensure_dirs()
    done = load_done(PROG / done_file)
    # 幂等：磁盘已有 parquet 的文件视为已完成，跳过（崩溃重跑零损失）
    existing = {f.stem for f in out_dir.glob("*.parquet")} if out_dir.exists() else set()
    pending = [c for c in codes if c not in done and c not in existing]
    log.info("[%s] 待处理 %d / 总 %d（已完成 %d）", name, len(pending), len(codes), len(done))
    if not pending:
        log.info("[%s] 已全部完成", name)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import wait, FIRST_COMPLETED
    todo = list(pending)
    deadline = time.time() + time_budget
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        submitted = {}
        idx = [0]

        def fill():
            while idx[0] < len(todo) and len(submitted) < max_workers and time.time() < deadline:
                c = todo[idx[0]]
                idx[0] += 1
                submitted[ex.submit(func, c)] = c

        def settle(f):
            nonlocal ok, fail
            c = submitted.pop(f)
            try:
                df = f.result()
                if df is None:
                    fail += 1
                elif empty_is_done and len(df) == 0:
                    # 早退市股窗口内无交易，视为已处理：不写文件、不重试
                    done.add(c)
                elif len(df):
                    df.to_parquet(out_dir / f"{c}.parquet", index=False)
                    ok += 1
                    done.add(c)
                else:
                    fail += 1
            except Exception as e:
                log.warning("[%s] %s 失败: %s", name, c, str(e)[:60])
                fail += 1

        fill()
        while submitted and time.time() < deadline:
            done_futs, _ = wait(list(submitted), return_when=FIRST_COMPLETED, timeout=30)
            for f in done_futs:
                settle(f)
                fill()
        # 预算到点：停止提交，仅处理仍在跑的活跃任务(≤workers)，不空等剩余队列
        for f in as_completed(list(submitted)):
            settle(f)
    save_done(PROG / done_file, done)
    left = len(codes) - len(done)
    log.info("[%s] 本批 ok=%d fail=%d 累计 %d/%d 剩余 %d", name, ok, fail, len(done), len(codes), left)
    return left


# ───────────────────────── 日线 ─────────────────────────
def fetch_daily_listed(code):
    prefix = market_of(code)
    df = ak.stock_zh_a_daily(symbol=f"{prefix}{code}", start_date=WS, end_date=WE, adjust="qfq")
    if df is None or len(df) == 0:
        return None
    df = df.reset_index()
    if "date" not in df.columns and "日期" in df.columns:
        df = df.rename(columns={"日期": "date"})
    keep = [c for c in ["date", "open", "close", "high", "low", "volume", "amount"] if c in df.columns]
    df = df[keep].copy()
    if "amount" not in df.columns:
        df["amount"] = 0.0
    return df


def fetch_daily_delist(code):
    for _ in range(5):
        time.sleep(0.6)  # 腾讯建议0.5s间隔，密集请求会被限流返回空
        try:
            rec = tencent.get_kline(code, "day")
            if rec:
                break
        except Exception:
            pass
    else:
        rec = None
    if not rec:
        return None
    df = pd.DataFrame(rec)
    df = df[df["date"] >= WS]
    keep = [c for c in ["date", "open", "close", "high", "low", "volume"] if c in df.columns]
    df = df[keep].copy() if keep else df
    if "amount" not in df.columns:
        df["amount"] = 0.0
    return df  # 可能为0行(早退市股窗口内无交易)，视为已处理


# ───────────────────────── 三大表 ─────────────────────────
def _report_date_col(df):
    for c in df.columns:
        cu = str(c).upper()
        if "REPORT_DATE" in cu or "报告期" in str(c) or cu == "DATE":
            return c
    return None


def filter_8y(df):
    col = _report_date_col(df)
    if not col:
        return df
    d = pd.to_datetime(df[col], errors="coerce")
    if d.notna().any():
        return df[d.dt.year >= WIN_START_YEAR]
    return df


def _retry_fetch(fn, code, tries=3):
    for i in range(tries):
        try:
            df = fn(code)
            if df is not None and not df.empty:
                return filter_8y(df)
            return None
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def fetch_stmt_listed(code, kind):
    fn = {"income": fe.income_statement, "balance": fe.balance_sheet, "cash": fe.cash_flow_statement}[kind]
    return _retry_fetch(fn, code)


def fetch_stmt_delist(code, kind):
    prefix = "SH" if code.startswith(("6", "9")) else "SZ"
    fn = {"income": ak.stock_profit_sheet_by_report_delisted_em,
          "balance": ak.stock_balance_sheet_by_report_delisted_em,
          "cash": ak.stock_cash_flow_sheet_by_report_delisted_em}[kind]
    return _retry_fetch(lambda c: fn(symbol=f"{prefix}{c}"), code)


def run_statements(status, status_label, done_file):
    codes = load_codes([status])
    kinds = [("income_statement", "income"), ("balance_sheet", "balance"), ("cash_flow_statement", "cash")]
    total_left = 0
    for sub, kind in kinds:
        out = LAKE / "fundamentals" / sub
        left = run_batch(f"stmt_{status_label}_{kind}", codes,
                         lambda c, k=kind: fetch_stmt_delist(c, k) if status == "退市" else fetch_stmt_listed(c, k),
                         out, f"{done_file}_{kind}.json",
                         max_workers=8)
        total_left += left
    return total_left


# ───────────────────────── manifest ─────────────────────────
def build_manifest():
    ensure_dirs()
    daily_dir = LAKE / "daily"
    inc = LAKE / "fundamentals" / "income_statement"
    bal = LAKE / "fundamentals" / "balance_sheet"
    cf = LAKE / "fundamentals" / "cash_flow_statement"
    daily_n = len(list(daily_dir.glob("*.parquet"))) if daily_dir.exists() else 0
    inc_n = len(list(inc.glob("*.parquet"))) if inc.exists() else 0
    bal_n = len(list(bal.glob("*.parquet"))) if bal.exists() else 0
    cf_n = len(list(cf.glob("*.parquet"))) if cf.exists() else 0

    # 日线条数合计 + 缺口（<1800条的上市股）
    total_bars = 0
    short_listed = 0
    if daily_dir.exists():
        sl = pd.read_csv(LAKE / "metadata" / "stock_list.csv", dtype=str)
        listed_set = set(sl[sl["status"] == "上市"]["code"])
        for f in daily_dir.glob("*.parquet"):
            try:
                n = len(pd.read_parquet(f, columns=["date"]))
            except Exception:
                n = 0
            total_bars += n
            if f.stem in listed_set and n > 0 and n < 1800:
                short_listed += 1

    lines = [
        "# A股数据湖 (全A含退市 · 近8年)",
        "",
        f"- 数据窗口: {WS} ~ {WE}",
        f"- 股票池: 见 metadata/stock_list.csv",
        "",
        "## 产物统计",
        f"- 日线文件数: {daily_n}  | 日线总条数: {total_bars:,}",
        f"- 上市股日线不足8年(<1800条): {short_listed} 只 (多为主板早期/新股)",
        f"- 利润表文件数: {inc_n}",
        f"- 资产负债表文件数: {bal_n}",
        f"- 现金流量表文件数: {cf_n}",
        "",
        "## 数据源",
        "- 上市日线: 新浪 stock_zh_a_daily （8年完整）",
        "- 退市日线: 腾讯 get_kline （仅最近约640条≈2.5年，8年窗口内早期缺失，已标注）",
        "- 上市三大表: 东财 stock_*_sheet_by_report_em",
        "- 退市三大表: 东财 stock_*_sheet_by_report_delisted_em",
        "- 清单: 上市=东财 stock_info_a_code_name；退市=沪深 stock_info_sh/sz_delist",
        "",
        "## 已知缺口",
        "- 退市股日线仅腾讯口径(≈2.5年)，非完整8年；更早历史缺失为数据源限制。",
        "- 8年窗口外已退市的股票(如2009年退)窗口内无交易，日线为空属正常。",
        "- 北交所未纳入（按需求仅沪深）。",
    ]
    (LAKE / "manifest.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("manifest 生成: 日线%d文件/%d条, 利润表%d, 资产表%d, 现金表%d",
             daily_n, total_bars, inc_n, bal_n, cf_n)
    print("\n".join(lines))


# ───────────────────────── 入口 ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["list", "daily_listed", "daily_delist",
                                     "stmt_listed", "stmt_delist", "manifest"])
    args = ap.parse_args()
    ensure_dirs()
    if args.step == "list":
        build_stock_list()
    elif args.step == "daily_listed":
        codes = load_codes(["上市"])
        run_batch("daily_listed", codes, fetch_daily_listed,
                  LAKE / "daily", "daily_listed.json", max_workers=16)
    elif args.step == "daily_delist":
        codes = load_codes(["退市"])
        run_batch("daily_delist", codes, fetch_daily_delist,
                  LAKE / "daily", "daily_delist.json", max_workers=1, empty_is_done=True)
    elif args.step == "stmt_listed":
        run_statements("上市", "listed", "stmt_listed")
    elif args.step == "stmt_delist":
        run_statements("退市", "delist", "stmt_delist")
    elif args.step == "manifest":
        build_manifest()


if __name__ == "__main__":
    main()
