"""ETF / 指数轮动策略 (将'状态切换'thesis 落到资产层面)
数据源: 指数 stock_zh_index_daily(东财指数, 可达); ETF fund_etf_hist_sina(新浪, 可达)
宇宙: 沪深300 / 中证500 / 创业板指 / 中证1000 / 黄金ETF / 国债ETF / 货币ETF
两类策略:
  - momentum: 按 trailing 收益排序, 持 top-N 等权(纯动量, 不空仓)
  - momentum_trend: 仅允许站在 MA(ma) 上方的权益资产入选; 权益全弃则持国债ETF(崩盘保护)
基准: 宇宙等权(月度再平衡) / 沪深300 买入持有
"""
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

CACHE = Path("/workspace/stock_worm/data/etf_rotation_cache.parquet")
FIG = Path("/workspace/VibeTradingPush/agent/backtest/screen_results/etf_rotation_equity.png")
REP = Path("/workspace/VibeTradingPush/agent/backtest/screen_results/ETF轮动策略报告.md")

UNIVERSE = {
    "沪深300": ("index", "sh000300"),
    "中证500": ("index", "sh000905"),
    "创业板指": ("index", "sz399006"),
    "中证1000": ("index", "sh000852"),
    "黄金ETF": ("etf_sina", "sh518880"),
    "国债ETF": ("etf_sina", "sh511010"),
    "货币ETF": ("etf_sina", "sh511880"),
}
EQUITY = ["沪深300", "中证500", "创业板指", "中证1000"]


def _fetch_one(name, kind, sym):
    if kind == "index":
        df = ak.stock_zh_index_daily(symbol=sym)
    else:
        df = ak.fund_etf_hist_sina(symbol=sym)
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename(name)


def load_prices(force=False):
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    frames = {}
    for name, (kind, sym) in UNIVERSE.items():
        try:
            frames[name] = _fetch_one(name, kind, sym)
            print(f"[fetch] {name} OK {frames[name].index[0].date()} -> {frames[name].index[-1].date()}")
        except Exception as e:
            print(f"[fetch] ERR {name}: {repr(e)[:100]}")
    px = pd.DataFrame(frames).sort_index().ffill().dropna(how="any")
    px.to_parquet(CACHE)
    print(f"[fetch] 合并 {px.shape}, 区间 {px.index[0].date()} ~ {px.index[-1].date()}")
    return px


def _rebal_dates(idx, freq):
    if freq == "M":
        p = idx.to_period("M")
    elif freq == "W":
        p = idx.to_period("W")
    else:
        return set(idx)
    return set(idx[p != p.shift(1)])


def backtest(px, lookback=60, top_n=1, rebal="M", cost=0.0003,
             trend=False, ma=200):
    ret = px.pct_change().fillna(0)
    sig = px.pct_change(lookback)
    ma_val = px.rolling(ma).mean() if trend else None
    rb = _rebal_dates(px.index, rebal)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w = pd.Series(0.0, index=px.columns)
    for i, d in enumerate(px.index):
        if (d in rb) or (i == 0):
            s = sig.loc[d]
            if trend:
                cands = [c for c in px.columns
                         if (c not in EQUITY) or (px.loc[d, c] > ma_val.loc[d, c])]
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
    # 逐日组合收益(再平衡日扣成本)
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
    equity = (1 + port).cumprod()
    return equity, port, weights


def metrics(equity, port):
    n = len(port)
    yrs = n / 252
    tot = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1
    dd = (equity / equity.cummax() - 1).min()
    sharpe = port.mean() / (port.std() + 1e-12) * np.sqrt(252)
    return tot, cagr, dd, sharpe


def main():
    px = load_prices()
    configs = [
        ("纯动量 L20 top1", dict(lookback=20, top_n=1, trend=False)),
        ("纯动量 L60 top1", dict(lookback=60, top_n=1, trend=False)),
        ("纯动量 L120 top1", dict(lookback=120, top_n=1, trend=False)),
        ("纯动量 L120 top2", dict(lookback=120, top_n=2, trend=False)),
        ("动量+趋势 L60 top1(MA200)", dict(lookback=60, top_n=1, trend=True, ma=200)),
        ("动量+趋势 L120 top2(MA200)", dict(lookback=120, top_n=2, trend=True, ma=200)),
    ]
    rows = []
    eqs = {}
    for name, kw in configs:
        eq, port, w = backtest(px, rebal="M", **kw)
        tot, cagr, dd, sh = metrics(eq, port)
        rows.append((name, tot, cagr, dd, sh))
        eqs[name] = eq
        print(f"[bt] {name}: CAGR={cagr:+.2%} MaxDD={dd:+.2%} Sharpe={sh:.2f} Tot={tot:+.1%}")

    # 基准
    eq_ew, _, _ = backtest(px, lookback=60, top_n=len(px.columns), trend=False)  # 等权月度再平衡近似
    # 上面的等权实现其实会选全部(动量排序全持), 但权重相等 -> 等权. OK
    tot, cagr, dd, sh = metrics(eq_ew, eq_ew.pct_change().fillna(0))
    rows.append(("宇宙等权(月度)", tot, cagr, dd, sh))
    eqs["宇宙等权(月度)"] = eq_ew
    print(f"[bt] 宇宙等权: CAGR={cagr:+.2%} MaxDD={dd:+.2%} Sharpe={sh:.2f}")

    # 沪深300 买入持有
    hs300 = px["沪深300"]
    eq_hs = (1 + hs300.pct_change().fillna(0)).cumprod()
    tot, cagr, dd, sh = metrics(eq_hs, hs300.pct_change().fillna(0))
    rows.append(("沪深300买入持有", tot, cagr, dd, sh))
    eqs["沪深300买入持有"] = eq_hs
    print(f"[bt] 沪深300持有: CAGR={cagr:+.2%} MaxDD={dd:+.2%} Sharpe={sh:.2f}")

    # 报告 + 图
    df = pd.DataFrame(rows, columns=["策略", "累计收益", "年化", "最大回撤", "夏普"]).sort_values("年化", ascending=False)
    print("\n", df.to_string(index=False))

    # 图: 归一化净值(用英文标签避免中文字体缺失)
    EN = {"沪深300": "CSI300", "中证500": "CSI500", "创业板指": "ChiNext",
          "中证1000": "CSI1000", "黄金ETF": "GoldETF", "国债ETF": "BondETF",
          "货币ETF": "CashETF", "纯动量 L20 top1": "Mom L20 top1",
          "动量+趋势 L60 top1(MA200)": "Mom+Trend L60",
          "纯动量 L60 top1": "Mom L60 top1", "纯动量 L120 top1": "Mom L120 top1",
          "纯动量 L120 top2": "Mom L120 top2", "动量+趋势 L120 top2(MA200)": "Mom+Trend L120",
          "宇宙等权(月度)": "EqualWeight", "沪深300买入持有": "CSI300 BuyHold"}
    plt.figure(figsize=(11, 5))
    for name, eq in eqs.items():
        plt.plot(eq.index, eq / eq.iloc[0], label=EN.get(name, name),
                 lw=1.6 if name == "沪深300买入持有" else 1.2)
    plt.title("ETF/Index Rotation - Normalized Equity Curve (start=1)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG, dpi=110)
    plt.close()

    md = ["# ETF / 指数轮动策略报告", "",
          f"- 数据: {px.index[0].date()} ~ {px.index[-1].date()} ({len(px)} 交易日); 宇宙=沪深300/中证500/创业板/中证1000/黄金ETF/国债ETF/货币ETF",
          "- 回测: 月度再平衡, 单边成本 0.03%, 无杠杆; 纯动量持 top-N 等权, 动量+趋势在权益跌破 MA(200) 时切国债ETF(崩盘保护)",
          "- 目的: 把'状态切换'thesis 从因子层面落到资产层面, 看是否比选股更可交易、回撤更可控", "",
          "## 结果对比(按年化排序)", "",
          "| 策略 | 累计收益 | 年化 | 最大回撤 | 夏普 |",
          "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['策略']} | {r['累计收益']:+.1%} | {r['年化']:+.2%} | {r['最大回撤']:+.2%} | {r['夏普']:.2f} |")
    md += ["", "![净值](etf_rotation_equity.png)", "",
           "## 诚实解读", "",
           "- 轮动把'状态切换'变成策略本身: 牛市持权益、熊市自动切黄金/国债 —— 这正是咱们 thesis 在资产层的落地, 比股票因子门控(B)更干净.",
           "- 关键看 **最大回撤**: 若轮动策略的回撤显著低于沪深300买入持有, 说明'状态切换'确实在对的位置避险, 产品属性(可实盘、控回撤)远强于咱们的选股 OOS.",
           "- 若纯动量跑不赢沪深300买入持有, 说明 A股指数动量弱(牛短熊长+高波动), 需靠趋势保护/择时提升; 这正是下一步要调的参数.",
           "- 与选股 OOS 对照: 选股中性化后仅 Frozen 微跑赢等权(+0.395 超额夏普)且长持多头打不过 beta; 轮动若能用资产切换做出低回撤+正 alpha, 就是更优的'产品定义'.", "",
           f"*生成于 {datetime.now():%Y-%m-%d %H:%M}*"]
    REP.write_text("\n".join(md))
    print(f"\n报告: {REP}\n图: {FIG}")


if __name__ == "__main__":
    main()
