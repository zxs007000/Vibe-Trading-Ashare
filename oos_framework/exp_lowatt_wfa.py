"""exp_lowatt_wfa.py — 低关注(low_att)因子进池 WFA 回测

low_att = -rolling20(amount)  (成交额代理, 文档因子④). 是全场最强因子(IC60 全+0.122/非HS300+0.132).

两阶段:
  build : 从 stockworm/daily 湖重建 lean panel (date,code,low_att,ret,fwd20,amount), 缓存.
  run   : 月度再平衡多头回测. 四变体对比:
            NoGate      全市场, 轻流动性地板(5%), 无闸门  -> 展示微盘尾部风险(2024-02)
            Gate        全市场, 轻流动性地板, 确定性回撤闸(15%/回补5%) -> 闸门削尾
            SmallCapGate 非HS300, 轻地板, 闸门 -> 进池现实版(边缘集中在小盘)
            HS300Gate   HS300, 轻地板, 闸门 -> 健全性检查(应偏弱)

闸门=确定性规则(基于自身DD, 不衰减后盾), 阈值固定不拟合, 防过拟合.
输出: behavior_cache/lowatt_wfa_metrics.json + lowatt_wfa_equity.csv
"""
from __future__ import annotations
import os, json, time, logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "behavior_cache")
os.makedirs(CACHE, exist_ok=True)
DAILY = "D:/work Buddy GZ/Claw/stockworm/daily"
PANEL_OUT = os.path.join(CACHE, "lowatt_panel.parquet")
HS300_PRICE = "D:/stcok-worm/yjyg_hs300_prices.parquet"
METRICS_OUT = os.path.join(CACHE, "lowatt_wfa_metrics.json")
EQ_OUT = os.path.join(CACHE, "lowatt_wfa_equity.csv")
START_DATE = "2019-01-01"   # 近7年窗口

log = logging.getLogger("lowatt_wfa")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ───────────────────────── build ─────────────────────────
def build_panel():
    if os.path.exists(PANEL_OUT):
        log.info("缓存存在, 跳过 build: %s", PANEL_OUT)
        return
    codes = [f[:-8] for f in os.listdir(DAILY) if f.endswith(".parquet")]
    log.info("build lowatt_panel: %d 只", len(codes))
    schema = pa.schema([
        ("date", pa.timestamp("ns")),
        ("code", pa.string()),
        ("low_att", pa.float64()),
        ("ret", pa.float64()),
        ("fwd20", pa.float64()),
        ("amount", pa.float64()),
    ])
    writer = pq.ParquetWriter(PANEL_OUT, schema)
    t0 = time.time()
    SPLIT_THR = 0.30   # |单日回报|>30% 视为拆/分红跳(非涨跌停真实波动), 后复权归零
    for i, code in enumerate(codes):
        try:
            d = pd.read_parquet(os.path.join(DAILY, code + ".parquet"),
                                columns=["close", "amount"])
        except Exception:
            continue
        d = d.sort_index()
        if len(d) < 20:
            continue
        close = d["close"].astype(float).to_numpy()
        amount = d["amount"].astype(float).to_numpy()
        # —— HFQ 后复权: 拆/分红跳的单日回报归零, 真实收益不被假崩吞噬 ——
        raw_ret = pd.Series(close).pct_change().to_numpy()
        mask = np.abs(raw_ret) > SPLIT_THR
        f = np.where(mask, 1.0 + raw_ret, 1.0)
        suffix = np.concatenate([[1.0], np.cumprod(f[::-1])[::-1][:-1]])  # Π_{j>t}(1+r_j)
        adj_close = close * suffix
        adj_ret = pd.Series(adj_close).pct_change().fillna(0.0).to_numpy()
        low_att = -pd.Series(amount).rolling(20, min_periods=10).mean().to_numpy()
        fwd20 = pd.Series(adj_close).shift(-20).to_numpy() / adj_close - 1.0
        proc = pd.DataFrame({
            "date": d.index,
            "code": code,
            "low_att": low_att,
            "ret": adj_ret,
            "fwd20": fwd20,
            "amount": amount,
        }).dropna(subset=["low_att"])
        if proc.empty:
            continue
        tbl = pa.Table.from_pandas(proc, schema=schema, preserve_index=False)
        writer.write_table(tbl)
        if (i + 1) % 1000 == 0:
            log.info("  %d/%d 耗时 %.0fs", i + 1, len(codes), time.time() - t0)
    writer.close()
    log.info("完成 build: %s", PANEL_OUT)


# ───────────────────────── helpers ─────────────────────────
def month_first_dates(dates):
    """每月首个交易日 = 再平衡日 (dates 为 numpy datetime64 数组)"""
    ym = dates.astype("datetime64[M]")
    changes = np.empty(len(ym), dtype=bool)
    changes[0] = True
    changes[1:] = ym[1:] != ym[:-1]
    # 用 int64 作为稳健的 membership key (规避 numpy datetime64 哈希不稳定)
    return set(int(d.astype("int64")) for d in dates[changes])


def turnover(h_new: dict, h_old: dict) -> float:
    if not h_new and not h_old:
        return 0.0
    codes = set(h_new) | set(h_old)
    s = 0.0
    for c in codes:
        s += abs(h_new.get(c, 0.0) - h_old.get(c, 0.0))
    return 0.5 * s


def select(t, by_date, first_date, hs300, universe, liq_pct, top_frac, top_min=20):
    g = by_date.get(t)
    if g is None or len(g) == 0:
        return {}
    if universe == "nonhs300":
        codes = [c for c in g.index if c not in hs300]
    elif universe == "hs300":
        codes = [c for c in g.index if c in hs300]
    else:
        codes = list(g.index)
    sub = g.loc[codes]
    # 新上市过滤: 至少 60 日历日历史
    sub = sub[sub.index.map(lambda c: (t - first_date.get(c, t)).days >= 60)]
    if sub.empty:
        return {}
    # 流动性地板: 剔除成交额最低 liq_pct 分位(仅去最不流动尾, 不伤因子主体)
    if liq_pct > 0:
        floor = sub["amount"].quantile(liq_pct)
        sub = sub[sub["amount"] >= floor]
    if sub.empty:
        return {}
    # 做多低关注(高 low_att = 低成交额 = 被忽视), 取前 top_frac
    ranked = sub["low_att"].sort_values(ascending=False)
    k = max(int(round(len(ranked) * top_frac)), top_min)
    chosen = ranked.index[:k]
    w = 1.0 / len(chosen)
    return {c: w for c in chosen}


def run_variant(by_date, dates, first_date, hs300, universe,
                liq_pct, gate, reenter, top_frac):
    eq = 1.0
    peak = 1.0
    cash = False
    holdings = {}
    prev = {}
    eq_series = []
    daily_rets = []
    turns = []
    rebal = month_first_dates(dates)
    for t in dates:
        g = by_date.get(t)
        r = 0.0
        if (not cash) and holdings and g is not None:
            for c, w in holdings.items():
                rv = g["ret"].get(c, 0.0)
                if rv is None or (isinstance(rv, float) and np.isnan(rv)):
                    rv = 0.0
                r += w * rv
        eq *= (1.0 + r)
        daily_rets.append(r)
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if (not cash) and dd <= -gate:
            cash = True
        if int(t.astype("int64")) in rebal:
            # 月度再平衡: 若上月触发过闸门(现金), 本月重新评估并回补(闸的作用=跳过崩月)
            holdings = select(t, by_date, first_date, hs300, universe,
                              liq_pct, top_frac)
            turns.append(turnover(holdings, prev))
            prev = holdings
            cash = False
        eq_series.append(eq)
    return np.array(eq_series), np.array(daily_rets), np.array(turns)


def metrics(dates, eq, daily_rets, label):
    n = len(eq)
    yrs = n / 252.0
    ann_ret = eq[-1] ** (252.0 / n) - 1.0 if n > 0 else 0.0
    vol = np.std(daily_rets) * np.sqrt(252.0)
    sharpe = ann_ret / vol if vol > 1e-9 else 0.0
    run_peak = np.maximum.accumulate(eq)
    dd = eq / run_peak - 1.0
    maxdd = dd.min()
    calmar = ann_ret / abs(maxdd) if abs(maxdd) > 1e-9 else 0.0
    # 2024-02 微盘崩专项
    m = pd.Series(eq, index=dates)
    feb = m[(m.index >= pd.Timestamp("2024-01-20")) & (m.index <= pd.Timestamp("2024-03-15"))]
    dd_2024 = (feb.min() / feb.iloc[0] - 1.0) if len(feb) > 1 else np.nan
    return {
        "label": label,
        "ann_ret": round(ann_ret, 4),
        "ann_vol": round(vol, 4),
        "sharpe": round(sharpe, 3),
        "maxdd": round(maxdd, 4),
        "calmar": round(calmar, 2),
        "dd_2024feb": round(float(dd_2024), 4) if not np.isnan(dd_2024) else None,
        "final_eq": round(float(eq[-1]), 3),
        "years": round(yrs, 1),
    }


# ───────────────────────── run ─────────────────────────
def run():
    log.info("载入 lean panel: %s", PANEL_OUT)
    df = pd.read_parquet(PANEL_OUT, columns=["date", "code", "low_att", "ret", "amount"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    # first_date 用全历史算(新上市过滤在窗口内仍生效)
    first_date = df.groupby("code")["date"].min().to_dict()
    # 限制近7年窗口
    df = df[df["date"] >= pd.Timestamp(START_DATE)].copy()
    df = df.dropna(subset=["low_att", "ret"])
    dates = np.array(sorted(df["date"].unique()), dtype="datetime64[ns]")
    log.info("窗口 %s+ : 截面 %d 交易日, %d 只", START_DATE, len(dates), df["code"].nunique())

    log.info("构建 by_date 索引(内存)...")
    by_date = {t: g.set_index("code")[["low_att", "ret", "amount"]]
               for t, g in df.groupby("date")}
    del df
    hs300 = set(pd.read_parquet(HS300_PRICE)["code"].astype(str).str.zfill(6).tolist())

    variants = [
        ("NoGate",         "all",      0.05, 0.999, 0.05, 0.10),
        ("Gate15",         "all",      0.05, 0.15,  0.05, 0.10),
        ("Gate30",         "all",      0.05, 0.30,  0.05, 0.10),
        ("SmallCapGate30", "nonhs300", 0.05, 0.30,  0.05, 0.10),
        ("HS300Gate",      "hs300",    0.05, 0.15,  0.05, 0.10),
    ]
    eqs = {}
    res = []
    for name, uni, liq, gate, reenter, top in variants:
        log.info("回测变体 %s (uni=%s liq=%.2f gate=%.2f)", name, uni, liq, gate)
        eq, dret, turns = run_variant(by_date, dates, first_date, hs300, uni,
                                      liq, gate, reenter, top)
        eqs[name] = eq
        m = metrics(dates, eq, dret, name)
        m["avg_turnover"] = round(float(np.mean(turns)) if len(turns) else 0.0, 3)
        res.append(m)
        log.info("  %s: 年化 %.1f%% Sharpe %.2f MaxDD %.1f%% dd2024feb %.1f%%",
                 name, m["ann_ret"]*100, m["sharpe"], m["maxdd"]*100,
                 (m["dd_2024feb"] or 0)*100)

    with open(METRICS_OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    # 权益曲线 CSV
    out = pd.DataFrame({"date": dates})
    for name, eq in eqs.items():
        out[name] = eq
    out.to_csv(EQ_OUT, index=False)

    print("\n" + "=" * 92)
    print(f"低关注(low_att)进池 · WFA 多头回测 (窗口 {START_DATE}+, 月度再平衡, 等权前10%, 流动性地板5%)")
    print("=" * 92)
    hdr = f"{'变体':14s} {'年化':>8s} {'波动':>8s} {'Sharpe':>8s} {'MaxDD':>9s} {'Calmar':>7s} {'2024-2':>9s} {'换手':>7s} {'终值':>9s}"
    print(hdr)
    for m in res:
        print(f"{m['label']:14s} {m['ann_ret']*100:7.1f}% {m['ann_vol']*100:7.1f}% "
              f"{m['sharpe']:8.2f} {m['maxdd']*100:8.1f}% {m['calmar']:7.2f} "
              f"{(m['dd_2024feb'] or 0)*100:8.1f}% {m['avg_turnover']:7.2f} {m['final_eq']:9.1f}")
    print("=" * 92)
    print("结论: 若 Gate 的 MaxDD/2024-2 显著低于 NoGate 且无大幅牺牲年化 -> 闸门该上.")
    print(f"权益曲线: {EQ_OUT}")
    print(f"指标: {METRICS_OUT}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_panel()
    else:
        build_panel()
        run()
