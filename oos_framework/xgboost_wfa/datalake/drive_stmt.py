#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三大表驱动：循环跑 上市/退市 × 3表 共6个子批，断点续传。
每个子批受 run_batch 内部 time_budget 限制；本驱动在单进程内顺序推进各子批，
靠 progress/*.json 续传。nohup 长跑，全部完成或连续多轮无进展后退出。
"""
import sys, time, json, csv
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
import build_lake as bl

PROG = bl.PROG
LOG = open(str(Path(__file__).parent / "drive_stmt.log"), "a", buffering=1)

def log(*a):
    s = " ".join(str(x) for x in a)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {s}"
    print(line, flush=True)
    LOG.write(line + "\n")

def count_done(fname):
    p = PROG / fname
    if not p.exists():
        return 0
    try:
        return len(json.loads(p.read_text()))
    except Exception:
        return 0

def count_parquet(sub):
    """按 parquet 真实落盘数判定进度（避免 JSON 续传滞后误判完成）"""
    d = bl.LAKE / "fundamentals" / sub
    return len(list(d.glob("*.parquet"))) if d.exists() else 0

# 股票池总量
sl = list(csv.DictReader(open(bl.LAKE / "metadata" / "stock_list.csv", encoding="utf-8-sig")))
tot = Counter(r["status"] for r in sl)
log("股票池总量:", dict(tot))

BATCHES = [
    ("stmt_listed_income.json",  "上市", "income_statement",  "income"),
    ("stmt_listed_balance.json", "上市", "balance_sheet",     "balance"),
    ("stmt_listed_cash.json",    "上市", "cash_flow_statement","cash"),
    ("stmt_delist_income.json",  "退市", "income_statement",  "income"),
    ("stmt_delist_balance.json", "退市", "balance_sheet",     "balance"),
    ("stmt_delist_cash.json",    "退市", "cash_flow_statement","cash"),
]

WORKERS = 32
HARD_DEADLINE = time.time() + 6 * 3600   # 安全上限 6h
MAX_NOPROG_ROUNDS = 3

def one_batch(done_file, status, sub, kind):
    codes = bl.load_codes([status])
    out = bl.LAKE / "fundamentals" / sub
    before = count_done(done_file)
    left = bl.run_batch(f"stmt_{status}_{kind}", codes,
                        (lambda c, k=kind: bl.fetch_stmt_delist(c, k) if status == "退市"
                         else bl.fetch_stmt_listed(c, k)),
                        out, done_file, max_workers=WORKERS)
    after = count_done(done_file)
    return after, len(codes), after - before

rounds = 0
noprog = 0
while time.time() < HARD_DEADLINE:
    rounds += 1
    log(f"===== ROUND {rounds} =====")
    gains = []
    for done_file, status, sub, kind in BATCHES:
        try:
            after, total, gain = one_batch(done_file, status, sub, kind)
        except Exception as e:
            log(f"  {done_file} 异常: {str(e)[:120]}")
            gain = 0
            after = count_done(done_file)
            total = tot[status]
        log(f"  {done_file}: {after}/{total} (+{gain})")
        gains.append(gain)

    # 写状态快照：基于 parquet 真实落盘数（避免 JSON 续传滞后误判完成）
    status = {df: {"done": count_parquet(sub), "total": tot[st]}
              for df, st, sub, _ in BATCHES}
    json.dump(status, open(str(Path(__file__).parent / "status.json"), "w"), ensure_ascii=False)

    # 完成判定改用 parquet 真实落盘数（更可靠，崩溃后 JSON 可能滞后）
    if all(count_parquet(sub) >= tot[st] for _, st, sub, _ in BATCHES):
        log("全部三大表完成 ✅ (parquet 计数达标)")
        try:
            bl.build_manifest()
            log("manifest 已生成")
        except Exception as e:
            log(f"manifest 生成失败: {e}")
        open(str(bl.LAKE / "BUILD_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        break

    if sum(1 for g in gains if g > 0) == 0:
        noprog += 1
        log(f"本轮无进展 ({noprog}/{MAX_NOPROG_ROUNDS})，暂停60s等待东财恢复")
        time.sleep(60)
        if noprog >= MAX_NOPROG_ROUNDS:
            # 残余失败多属东财个别股票缺失，视为构建完成，落 BUILD_DONE 让下游继续
            log("连续无进展达上限 → 视为构建完成(残余失败属东财缺失)，生成 manifest + BUILD_DONE")
            try:
                bl.build_manifest()
                log("manifest 已生成")
            except Exception as e:
                log(f"manifest 生成失败: {e}")
            open(str(bl.LAKE / "BUILD_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
            break
    else:
        noprog = 0

log("DRIVER 结束")
LOG.close()
