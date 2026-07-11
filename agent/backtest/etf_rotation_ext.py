"""ETF/指数轮动 · 扩展版(丰富宇宙 + 200日线闸门 + 20年+长历史 + 过拟合检验)
- 宇宙: CORE20Y(长历史指数+黄金期货+国债, 2003起) / RICH(宽基+风格+行业+黄金+国债+可转债+恒科, 无纳指)
- 策略: 纯动量 L20 top1(月度) × {MA200闸门 关/开}; 基准=宇宙等权 / 沪深300持有
- 防过拟合: 每套宇宙对最优配置做 前后半段 稳定性切分
数据: 指数 stock_zh_index_daily; ETF fund_etf_hist_sina; 黄金期货 futures_main_sina
"""
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from pathlib import Path
CACHE = Path("/workspace/stock_worm/data/etf_rotation_ext_cache.parquet")
FIG = Path("/workspace/VibeTradingPush/agent/backtest/screen_results/etf_rotation_ext_equity.png")
REP = Path("/workspace/VibeTradingPush/agent/backtest/screen_results/ETF轮动_扩展报告.md")

# (name, kind, symbol, is_equity)
MASTER = [
    ("上证综指", "index", "sh000001", True),
    ("沪深300", "index", "sh000300", True),
    ("中证500", "index", "sh000905", True),
    ("创业板指", "index", "sz399006", True),
    ("红利ETF", "etf_sina", "sh510880", True),
    ("券商ETF", "etf_sina", "sh512000", True),
    ("半导体ETF", "etf_sina", "sh512760", True),
    ("医药ETF", "etf_sina", "sh512010", True),
    ("军工ETF", "etf_sina", "sh512660", True),
    ("消费ETF", "etf_sina", "sh510150", True),
    ("煤炭ETF", "etf_sina", "sh515220", True),
    ("新能源ETF", "etf_sina", "sh515030", True),
    ("恒生科技ETF", "etf_sina", "sh513180", True),
    ("黄金ETF", "etf_sina", "sh518880", False),
    ("国债ETF", "etf_sina", "sh511010", False),
    ("货币ETF", "etf_sina", "sh511880", False),
    ("可转债ETF", "etf_sina", "sh511380", False),
    ("上证国债指", "index", "sh000012", False),
    ("黄金期货AU0", "futures", "AU0", False),
]
NAME2META = {m[0]: m for m in MASTER}

UNIVERSES = {
    "CORE20Y": ["上证综指", "沪深300", "中证500", "上证国债指", "黄金期货AU0"],
    "RICH": ["上证综指", "沪深300", "中证500", "创业板指", "红利ETF", "券商ETF",
             "半导体ETF", "医药ETF", "军工ETF", "消费ETF", "煤炭ETF", "新能源ETF",
             "恒生科技ETF", "黄金ETF", "国债ETF", "货币ETF", "可转债ETF"],
    "RICH13": ["上证综指", "沪深300", "中证500", "创业板指", "红利ETF", "券商ETF",
               "医药ETF", "军工ETF", "黄金ETF", "国债ETF", "货币ETF"],
}


def _fetch_one(name, kind, sym):
    if kind == "index":
        df = ak.stock_zh_index_daily(symbol=sym)
    elif kind == "etf_sina":
        df = ak.fund_etf_hist_sina(symbol=sym)
    elif kind == "futures":
        df = ak.futures_main_sina(symbol=sym)
    df = df.rename(columns={"日期": "date", "收盘价": "close"})
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename(name)


def load_prices(force=False):
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    frames = {}
    for name, kind, sym, _ in MASTER:
        try:
            s = _fetch_one(name, kind, sym)
            frames[name] = s
            print(f"[fetch] {name}: {s.index[0].date()} -> {s.index[-1].date()}")
        except Exception as e:
            print(f"[fetch] ERR {name}: {repr(e)[:90]}")
    px = pd.DataFrame(frames).sort_index().ffill().dropna(how="all")
    px.to_parquet(CACHE)
    print(f"[fetch] 合并 {px.shape}")
    return px


def _rebal_dates(idx, freq):
    p = idx.to_period(freq)
    return set(idx[p != p.shift(1)])


def backtest(px, lookback=20, top_n=1, rebal="M", cost=0.0003, trend=False, ma=200,
             regime=False, regime_ma=120, bench="沪深300",
             defensive=("国债ETF", "货币ETF", "上证国债指")):
    """动量轮动回测.

    trend : MA(ma) 个股闸门(权益跌破均线剔出排名)
    regime: 市场状态总开关 —— 以 bench(宽基) 相对其 MA(regime_ma) 判定 risk-on/off.
            risk-off 时整仓切防御资产(defensive 列表首个可用), 与个股 MA200 闸门正交叠加.
            注: bench 在 regime_ma 窗口前 MA 为 NaN, 视为 risk-on(不误杀).
    """
    eq_names = [c for c in px.columns if NAME2META[c][3]]
    ret = px.pct_change().fillna(0)
    sig = px.pct_change(lookback)
    ma_val = px.rolling(ma).mean() if trend else None
    bench_s = px[bench] if (regime and bench in px.columns) else None
    bench_ma = bench_s.rolling(regime_ma).mean() if bench_s is not None else None
    da = next((a for a in defensive if a in px.columns), None)
    rb = _rebal_dates(px.index, rebal)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w = pd.Series(0.0, index=px.columns)
    for i, d in enumerate(px.index):
        if (d in rb) or (i == 0):
            roff = (bench_ma is not None) and (bench_s.loc[d] <= bench_ma.loc[d])
            if roff and da is not None:
                w = pd.Series(0.0, index=px.columns); w[da] = 1.0
            else:
                s = sig.loc[d]
                if trend:
                    above = (px.loc[d] > ma_val.loc[d]) | ma_val.loc[d].isna()
                    cands = [c for c in px.columns
                             if (not NAME2META[c][3]) or bool(above[c])]
                    if not cands:
                        cands = ["国债ETF"] if "国债ETF" in px.columns else list(px.columns)
                else:
                    cands = list(px.columns)
                sc = s[cands].dropna().sort_values(ascending=False)
                if len(sc) == 0:
                    w = pd.Series(0.0, index=px.columns)
                    if "货币ETF" in px.columns:
                        w["货币ETF"] = 1.0
                else:
                    top = sc.index[:top_n]
                    w = pd.Series(0.0, index=px.columns)
                    w[top] = 1.0 / top_n
        weights.loc[d] = w
    port = pd.Series(0.0, index=px.index)
    prev = pd.Series(0.0, index=px.columns)
    for i, d in enumerate(px.index):
        if i == 0:
            prev = weights.loc[d]
            continue
        daily = (prev * ret.loc[d]).sum()
        if d in rb:
            turn = (weights.loc[d] - prev).abs().sum()
            daily -= cost * turn
            prev = weights.loc[d]
        port[d] = daily
    return (1 + port).cumprod(), port, weights


def metrics(equity, port):
    n = len(port)
    yrs = max(n / 252, 1e-9)
    tot = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1
    dd = (equity / equity.cummax() - 1).min()
    sharpe = port.mean() / (port.std() + 1e-12) * np.sqrt(252)
    return tot, cagr, dd, sharpe


def trim(px, min_assets=3):
    cnt = px.notna().sum(axis=1)
    start = cnt[cnt >= min_assets].index[0]
    return px.loc[start:]


def run_universe(px_full, univ_name):
    cols = UNIVERSES[univ_name]
    px = trim(px_full[cols].copy())
    print(f"\n=== 宇宙 {univ_name}: {px.index[0].date()} ~ {px.index[-1].date()} "
          f"({len(px)/252:.1f}年, {len(cols)}资产) ===")
    rows = []
    eqs = {}
    combos = [
        (False, False, "基准(MA关/regime关)"),
        (True,  False, "MA200开"),
        (False, True,  "regime开(MA关)"),
        (True,  True,  "MA200开+regime开"),
    ]
    for trend, regime, lab in combos:
        eq, port, _ = backtest(px, lookback=20, top_n=1, trend=trend, ma=200,
                               regime=regime, regime_ma=120)
        tot, cagr, dd, sh = metrics(eq, port)
        rows.append((f"{univ_name} 动量L20top1 {lab}", tot, cagr, dd, sh))
        eqs[f"{univ_name}|{'T' if trend else 't'}{'R' if regime else 'r'}"] = eq
        print(f"  [bt] 动量L20top1 {lab}: CAGR={cagr:+.2%} MaxDD={dd:+.2%} Sharpe={sh:.2f}")
        # 前后半段稳定性
        mid = px.index[len(px) // 2]
        for slab, sl in (("前半", px.index < mid), ("后半", px.index >= mid)):
            e2 = eq.loc[sl]
            if len(e2) > 50:
                t2, c2, d2, s2 = metrics(e2, port.loc[sl])
                print(f"        {slab}: CAGR={c2:+.2%} MaxDD={d2:+.2%} Sharpe={s2:.2f}")
    # 基准
    eq_ew, _, _ = backtest(px, lookback=20, top_n=len(cols), trend=False)
    t, c, d, s = metrics(eq_ew, eq_ew.pct_change().fillna(0))
    rows.append((f"{univ_name} 等权(月度)", t, c, d, s))
    eqs[f"{univ_name}|EW"] = eq_ew
    if "沪深300" in px.columns:
        hs = px["沪深300"]
        eq_hs = (1 + hs.pct_change().fillna(0)).cumprod()
        t, c, d, s = metrics(eq_hs, hs.pct_change().fillna(0))
        rows.append((f"{univ_name} 沪深300持有", t, c, d, s))
        eqs[f"{univ_name}|HS"] = eq_hs
    return rows, eqs


def main():
    px = load_prices()
    all_rows = []
    all_eqs = {}
    for univ in ("CORE20Y", "RICH13", "RICH"):
        rows, eqs = run_universe(px, univ)
        all_rows += rows
        all_eqs.update(eqs)

    df = pd.DataFrame(all_rows, columns=["策略", "累计收益", "年化", "最大回撤", "夏普"])
    print("\n", df.to_string(index=False))

    # 图: 各宇宙 4 种组合(MA×regime) 归一化净值
    combo_style = [("tr", "-", "基准(MA关/regime关)"), ("Tr", "-", "MA200开"),
                   ("tR", "--", "regime开(MA关)"), ("TR", "-.", "MA200开+regime开")]
    plt.figure(figsize=(13, 6))
    for univ in ("CORE20Y", "RICH13", "RICH"):
        for key, ls, _ in combo_style:
            k = f"{univ}|{key}"
            if k in all_eqs:
                plt.plot(all_eqs[k].index, all_eqs[k] / all_eqs[k].iloc[0],
                         ls, lw=1.0, label=f"{univ} {key}")
    plt.title("ETF Rotation - Normalized Equity (MA x regime combos)")
    plt.legend(fontsize=6, ncol=4)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG, dpi=110)
    plt.close()

    md = ["# ETF 轮动 · 扩展报告（丰富宇宙 + 200日闸门 + 20年+）", "",
          "- 数据: 指数(stock_zh_index_daily) + ETF(fund_etf_hist_sina) + 黄金期货(futures_main_sina); 纳指ETF 已排除(防过拟合)",
          "- 策略: 月度再平衡, 单边成本0.03%, 信号=20日收益率动量, 持top1; MA200闸门=权益跌破200日均线则剔出排名、全弃时切国债ETF",
          "- 宇宙: CORE20Y(长历史指数+国债+黄金期货, 2003起) / RICH13(2013+可用资产) / RICH(全量含恒科可转债, 上市晚)",
          "- 防过拟合: 每宇宙对最优配置做前后半段稳定性切分", "",
          "## 结果", "",
          "| 策略 | 累计 | 年化 | 最大回撤 | 夏普 |",
          "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['策略']} | {r['累计收益']:+.1%} | {r['年化']:+.2%} | {r['最大回撤']:+.2%} | {r['夏普']:.2f} |")
    md += ["", "![净值](etf_rotation_ext_equity.png)", "",
           "## 诚实解读", "",
           "- **MA200 闸门的价值**: 对比每宇宙 'MA200关' vs 'MA200开' 的回撤与夏普, 看趋势保护是否真正降低回撤(尤其股灾/熊市段).",
           "- **regime 总开关的价值**: 以沪深300 相对其 MA120 判定 risk-on/off, risk-off 时整仓切防御(国债ETF→货币ETF→上证国债指). "
           "对比 'MA200开' vs 'MA200开+regime开' 看市场状态层是否进一步压低回撤、平滑收益; 若 regime 在长历史(CORE20Y 跨20年)显著降低最大回撤而不牺牲太多收益, 说明'什么状态用什么策略'有效.",
           "- **丰富宇宙 vs 7资产**: RICH 比初始7资产多了一堆低相关行业/商品/海外资产, 看夏普/回撤是否因多样性改善, 还是只是数字游戏.",
           "- **过拟合检验**: 若某配置 '前半' 远好于 '后半', 说明是样本内巧合(尤其 RICH 窗口短、资产多). CORE20Y 跨20年多regime, 更可信.",
           "- **纳指已排除**: 避免用单一高收益海外资产把整个轮动'带偏'成美股beta.", "",
           f"*生成于 {datetime.now():%Y-%m-%d %H:%M}*"]
    REP.write_text("\n".join(md))
    print(f"\n报告: {REP}\n图: {FIG}")


if __name__ == "__main__":
    main()
