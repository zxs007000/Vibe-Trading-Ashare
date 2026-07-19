"""exp_pillar2_factors.py — Pillar2 股权/股东结构因子试点 (PIT 严格)

因子(慢熵基本面 / 知情交易 / 供给冲击, 正交价量池):
  F1 股东户数变化率  (chip concentration): 户数环比↓ = 筹码集中 = 看好
                                    signal = -股东户数-增减比例
  F2 解禁压力        (unlock overhang) : unlocks.parquet 占解禁前流通市值比例,
                                    未来 60 日滚动求和(供给冲击)
  F3 股权质押率      (pledge risk)     : stock_gpzy_pledge_ratio_em(date) 质押比例, 逐季
  (F4 内部人增减持: 本机 akshare 源受限, 暂缺)

收益源: D:/stcok-worm/yjyg_hs300_prices.parquet (288 只 HS300, 2011-2025)
        -> 前向 20/60 交易日收益 (PIT: 因子值已知于 t 收盘, 持有自 t)

IC: 截面 rank-IC(每因子 vs 前向收益), 全样本 + 2023-25 子区间;
    因子间相关性(正交性检验).

PIT 纪律: 因子值在"公告日/解禁日/统计截止日"当日即已知, 面板按交易日 ffill
          (值 = 截至该日最新事件), 不使用未来信息.

用法:
    python exp_pillar2_factors.py            # 拉数 + 算 IC(首跑慢, 已缓存则快)
    python exp_pillar2_factors.py --no-fetch  # 复用已缓存因子原始数据
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import logging
import threading

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pillar2")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "pillar2_cache")
os.makedirs(CACHE, exist_ok=True)

PRICE_CACHE = "D:/stcok-worm/yjyg_hs300_prices.parquet"
UNLOCK_PARQUET = "D:/work Buddy GZ/Claw/stockworm/fundamentals/unlocks.parquet"
SH_RAW = os.path.join(CACHE, "shareholder_raw.parquet")
PLEDGE_RAW = os.path.join(CACHE, "pledge_raw.parquet")
FACTOR_PANEL = os.path.join(CACHE, "factor_panel.parquet")
IC_OUT = os.path.join(CACHE, "pillar2_ic.json")

# 单只/单次网络抓取硬超时(秒): 代理抖动时 akshare 内部请求可能无超时挂死,
# 用守护线程 + join(timeout) 兜底, 超时跳过该标的/日期.
_NET_TIMEOUT = 25
_FFETCH_LIMIT = 120  # ffill 上限(交易日): 避免因子值过旧


# ── 网络抓取安全包装 ──
def _safe(fn, timeout=_NET_TIMEOUT):
    holder = {}
    def _t():
        try:
            holder["r"] = fn()
        except Exception as e:
            holder["e"] = repr(e)[:160]
    th = threading.Thread(target=_t, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return None, "timeout"
    if "e" in holder:
        return None, holder["e"]
    return holder.get("r"), None


# ── 收益源 ──
def load_returns():
    """从缓存价建 (code, date) 面板 + 前向 20/60d 收益."""
    p = pd.read_parquet(PRICE_CACHE)
    codes = sorted(p["code"].unique().tolist())
    out = {}
    for code, g in p.groupby("code"):
        g = g.sort_values("date").set_index("date")
        g = g[~g.index.duplicated(keep="last")]
        g["fwd20"] = g["close"].shift(-20) / g["close"] - 1.0
        g["fwd60"] = g["close"].shift(-60) / g["close"] - 1.0
        out[code] = g[["close", "fwd20", "fwd60"]]
    return codes, out


# ── F1 股东户数变化率 ──
def _fetch_shareholder(code: str):
    import akshare as ak
    d = ak.stock_zh_a_gdhs_detail_em(symbol=str(code))
    if d is None or d.empty:
        return None
    need = ["股东户数公告日期", "股东户数-增减比例", "股东户数-本次", "股东户数-上次"]
    if not all(c in d.columns for c in need):
        return None
    d = d.copy()
    d["code"] = code
    d["ann"] = pd.to_datetime(d["股东户数公告日期"], errors="coerce")
    d["chg_ratio"] = pd.to_numeric(d["股东户数-增减比例"], errors="coerce")
    d = d.dropna(subset=["ann", "chg_ratio"])
    # 去重(同一公告日可能多行)
    d = d.sort_values("ann").drop_duplicates("ann", keep="last")
    return d[["code", "ann", "chg_ratio"]]


def build_shareholder(codes, use_cache=True):
    if use_cache and os.path.exists(SH_RAW):
        log.info("复用股东户数原始缓存: %s", SH_RAW)
        return pd.read_parquet(SH_RAW)
    rows = []
    ok = 0
    for i, code in enumerate(codes):
        d, err = _safe(lambda c=code: _fetch_shareholder(c))
        if d is not None and not d.empty:
            rows.append(d)
            ok += 1
        elif err:
            if "timeout" in str(err):
                log.warning("股东户数 %s 超时, 跳过", code)
            # 其他错误静默跳过
        if (i + 1) % 50 == 0:
            log.info("  股东户数 %d/%d (成功 %d)", i + 1, len(codes), ok)
    if not rows:
        log.error("股东户数全部失败")
        return pd.DataFrame(columns=["code", "ann", "chg_ratio"])
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(SH_RAW)
    log.info("股东户数原始落盘: %d 行, %d 只", len(df), df["code"].nunique())
    return df


# ── F3 股权质押率 ──
def _quarter_ends(start=2011, end=2025):
    ends = []
    for y in range(start, end + 1):
        for m in (3, 6, 9, 12):
            day = 31 if m in (3, 12) else 30
            ends.append(f"{y}{m:02d}{day:02d}")
    return ends


def _fetch_pledge(date: str):
    import akshare as ak
    d = ak.stock_gpzy_pledge_ratio_em(date=date)
    if d is None or d.empty:
        return None
    need = ["股票代码", "交易日期", "质押比例"]
    if not all(c in d.columns for c in need):
        return None
    d = d.copy()
    d["code"] = d["股票代码"].astype(str).str.strip()
    d["date"] = pd.to_datetime(d["交易日期"], errors="coerce")
    d["pledge"] = pd.to_numeric(d["质押比例"], errors="coerce")
    d = d.dropna(subset=["date", "pledge"])
    d = d[["code", "date", "pledge"]]
    return d


def build_pledge(codes, use_cache=True):
    if use_cache and os.path.exists(PLEDGE_RAW):
        log.info("复用质押率原始缓存: %s", PLEDGE_RAW)
        return pd.read_parquet(PLEDGE_RAW)
    rows = []
    dates = _quarter_ends()
    ok = 0
    for i, dt in enumerate(dates):
        d, err = _safe(lambda q=dt: _fetch_pledge(q))
        if d is not None and not d.empty:
            rows.append(d)
            ok += 1
        elif err and "timeout" in str(err):
            log.warning("质押 %s 超时, 跳过", dt)
        if (i + 1) % 10 == 0:
            log.info("  质押 %d/%d (成功 %d)", i + 1, len(dates), ok)
    if not rows:
        log.error("质押率全部失败")
        return pd.DataFrame(columns=["code", "date", "pledge"])
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(PLEDGE_RAW)
    log.info("质押率原始落盘: %d 行, %d 只, %d 日期", len(df), df["code"].nunique(), df["date"].nunique())
    return df


# ── F2 解禁压力 (来自本地 parquet, 无需网络) ──
def build_unlock_pressure(codes):
    u = pd.read_parquet(UNLOCK_PARQUET)
    u["code"] = u["股票代码"].astype(str).str.strip()
    u["dt"] = pd.to_datetime(u["解禁时间"], errors="coerce")
    u["ratio"] = pd.to_numeric(u["占解禁前流通市值比例"], errors="coerce")
    u = u.dropna(subset=["code", "dt", "ratio"])
    # 按 (code, dt) 聚合(同日多笔解禁求和)
    agg = (u.groupby(["code", "dt"])["ratio"].sum().reset_index())
    out = {}
    for code in codes:
        g = agg[agg["code"] == code].sort_values("dt")
        if g.empty:
            continue
        # 日频事件序列: 在解禁日放置 ratio, 其余 0
        ev = pd.Series(0.0, index=pd.DatetimeIndex([], name="date"))
        s = pd.Series(g["ratio"].values, index=pd.DatetimeIndex(g["dt"].values, name="date"))
        ev = s.groupby(level=0).sum()
        # 未来 60 日滚动求和(含当日): 反向 rolling 再反向
        daily = ev.sort_index()
        fwd = daily[::-1].rolling("60D").sum()[::-1]
        out[code] = fwd.rename("unlock_pressure")
    return out


# ── 因子面板装配 (PIT ffill) ──
def assemble_panel(codes, rets, sh_raw, pledge_raw, unlock_press):
    # 统一交易日轴 = 各 code 价格日期并集(按 code 各自对齐)
    sh_raw = sh_raw.copy()
    sh_raw["ann"] = pd.to_datetime(sh_raw["ann"])
    # 质押 -> (code, date) 宽
    pledge_wide = pledge_raw.pivot_table(index="date", columns="code", values="pledge")

    recs = []
    for code in codes:
        r = rets.get(code)
        if r is None or len(r) < 62:
            continue
        dates = r.index
        # F1: 股东户数变化率 -> signal = -增减比例, 按 ann 对齐 ffill
        f1 = pd.Series(np.nan, index=dates, name="f1_shareholder")
        sg = sh_raw[sh_raw["code"] == code]
        if not sg.empty:
            s = pd.Series(-sg["chg_ratio"].values, index=pd.DatetimeIndex(sg["ann"].values))
            s = s[~s.index.duplicated(keep="last")].sort_index()
            f1 = s.reindex(dates, method="ffill").reindex(dates)
        # F3: 质押率 ffill
        f3 = pd.Series(np.nan, index=dates, name="f3_pledge")
        if code in pledge_wide.columns:
            pv = pledge_wide[code].dropna()
            if not pv.empty:
                f3 = pv.reindex(dates, method="ffill").reindex(dates)
        # F2: 解禁压力
        f2 = pd.Series(np.nan, index=dates, name="f2_unlock")
        if code in unlock_press:
            up = unlock_press[code]
            if not up.empty:
                f2 = up.reindex(dates, method="ffill").reindex(dates)

        df = pd.DataFrame({"f1_shareholder": f1, "f2_unlock": f2, "f3_pledge": f3})
        df.index.name = "date"
        df["code"] = code
        df["fwd20"] = r["fwd20"]
        df["fwd60"] = r["fwd60"]
        recs.append(df.reset_index())

    panel = pd.concat(recs, ignore_index=True)
    # ffill 上限: 截断过旧因子值(避免事件后无限沿用)
    for c in ["f1_shareholder", "f2_unlock", "f3_pledge"]:
        panel[c] = panel.groupby("code")[c].ffill(limit=_FFETCH_LIMIT)
    return panel


# ── IC 计算 ──
def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = len(a)
    if n < 5:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    da = ra - ra.mean(); db = rb - rb.mean()
    den = np.sqrt(float((da ** 2).sum()) * float((db ** 2).sum()))
    return float((da * db).sum() / den) if den > 0 else np.nan


def compute_ic(panel: pd.DataFrame, fwd_col: str, sub=None):
    if sub:
        p = panel[panel["date"] >= pd.Timestamp(sub)]
    else:
        p = panel
    facs = ["f1_shareholder", "f2_unlock", "f3_pledge"]
    out = {}
    for f in facs:
        daily = p.groupby("date").apply(lambda g: _spearman(g[f].values, g[fwd_col].values))
        out[f] = daily.mean()
    return out


def factor_corr(panel: pd.DataFrame):
    facs = ["f1_shareholder", "f2_unlock", "f3_pledge"]
    sub = panel[facs].dropna(how="all")
    return sub.corr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()

    log.info("加载收益源(HS300 缓存价)...")
    codes, rets = load_returns()
    log.info("代码数: %d", len(codes))

    # 拉原始因子数据
    sh_raw = build_shareholder(codes, use_cache=not args.no_fetch)
    pledge_raw = build_pledge(codes, use_cache=not args.no_fetch)
    log.info("构建解禁压力(本地 parquet)...")
    unlock_press = build_unlock_pressure(codes)

    log.info("装配 PIT 因子面板...")
    panel = assemble_panel(codes, rets, sh_raw, pledge_raw, unlock_press)
    panel.to_parquet(FACTOR_PANEL)
    log.info("面板: %d 行 × %d 列", panel.shape[0], panel.shape[1])

    # IC
    ic20 = compute_ic(panel, "fwd20")
    ic60 = compute_ic(panel, "fwd60")
    ic20_2325 = compute_ic(panel, "fwd20", sub="2023-01-01")
    ic60_2325 = compute_ic(panel, "fwd60", sub="2023-01-01")
    corr = factor_corr(panel)

    result = {
        "ic20_full": {k: round(float(v), 4) for k, v in ic20.items()},
        "ic60_full": {k: round(float(v), 4) for k, v in ic60.items()},
        "ic20_2023_2025": {k: round(float(v), 4) for k, v in ic20_2325.items()},
        "ic60_2023_2025": {k: round(float(v), 4) for k, v in ic60_2325.items()},
        "factor_corr": corr.round(3).to_dict(),
    }
    import json
    with open(IC_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 覆盖率
    cov = panel[["f1_shareholder", "f2_unlock", "f3_pledge"]].notna().mean()

    print("\n" + "=" * 70)
    print("Pillar2 因子试点  —  股东户数 / 解禁压力 / 股权质押率 (HS300, PIT)")
    print("=" * 70)
    print(f"样本: {panel.shape[0]} (date,code) | 代码 {panel['code'].nunique()} | "
          f"区间 {panel['date'].min():%Y-%m-%d} ~ {panel['date'].max():%Y-%m-%d}")
    print("-" * 70)
    print("因子覆盖率:")
    for k, v in cov.items():
        print(f"  {k:16s} {v:6.1%}")
    print("-" * 70)
    print(f"{'因子':18s} {'IC20全':>9s} {'IC60全':>9s} {'IC20 23-25':>11s} {'IC60 23-25':>11s}")
    for f in ["f1_shareholder", "f2_unlock", "f3_pledge"]:
        print(f"{f:18s} {ic20[f]:+9.3f} {ic60[f]:+9.3f} "
              f"{ic20_2325[f]:+11.3f} {ic60_2325[f]:+11.3f}")
    print("-" * 70)
    print("因子间相关性(正交性, 全样本):")
    print(corr.round(3).to_string())
    print("=" * 70)
    print(f"\n明细: {FACTOR_PANEL}\nIC JSON: {IC_OUT}")


if __name__ == "__main__":
    main()
