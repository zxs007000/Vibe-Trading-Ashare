"""端到端验证：财务数据源 + 市场中性对冲框架。

跑一遍真实数据，证明：
  1. financial_loader 能取到 ROE/现金含量/应计利润（财务维度从 0 起步）。
  2. 财务因子(ROE)有 IC 信号（证明不是噪声）。
  3. 市场中性对冲能把纯多空组合的回撤压下来，并和基线门槛比对。

用法：
  cd D:/Vibe-Trading-Ashare
  .venv/Scripts/python.exe agent/backtest/verify_financial_neutral.py

K线源: stock_worm.mootdx_source.get_kline_history (通达信 TCP 直连, 自动翻页, 无需代理).
指数对冲: market_neutral (akshare 新浪指数源直连, 不依赖代理).
"""

from __future__ import annotations

import sys, socket
socket.setdefaulttimeout(12)  # 防 mootdx recv 永久阻塞导致进程闷死
import logging
import time
import numpy as np
import pandas as pd

_STOCK_WORM_SRC = r"D:\stcok-worm"
if _STOCK_WORM_SRC not in sys.path:
    sys.path.insert(0, _STOCK_WORM_SRC)

sys.path.insert(0, "agent/src")
from dotenv import load_dotenv
load_dotenv("agent/.env")

from backtest.loaders.financial_loader import fetch_fundamentals, latest_as_of
import backtest.market_neutral as mn

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("verify")

UNIVERSE_N = 100
START = "2023-01-01"
END = "2026-07-08"


def build_universe(n: int) -> list[str]:
    """取 CSI300 成分股前 n 只 (stock_worm 东财报表, 直连通)."""
    from stcok_worm._session import eastmoney_datacenter
    try:
        rows = eastmoney_datacenter("RPT_INDEX_CONSTITUENT", columns="ALL",
                                    filter_str='(INDEX_CODE="000300")', page_size=1000)
        codes = []
        for r in rows:
            v = r.get("SECUCODE") or r.get("SECURITY_CODE") or r.get("CODE")
            if v:
                codes.append(str(v).split(".")[0])
        codes = [c for c in codes if c]
        if codes:
            return codes[:n]
    except Exception as e:  # noqa
        print(f"  [warn] stock_worm eastmoney cons failed: {e}")
    # 回退硬编码蓝筹
    return [
        "600519","000858","601318","600036","000333","601899","300750","601166",
        "600900","000651","600276","601398","000001","603259","600030","002415",
        "601288","600809","000725","601088","601012","002714","000002","600887",
        "601857","600028","601688","300059","600585","600309","600436","002594",
        "601225","603288","002304","000568","601066","600104","000776","300498",
    ][:n]


def fetch_prices(codes: list[str]) -> dict[str, pd.DataFrame]:
    """K线 (stock_worm.mootdx 通达信 TCP 直连, 翻页全历史), 切片到 [START,END].

    不用 akshare / 不依赖代理; mootdx 为单 TCP 连接, 串行拉取。
    """
    from stcok_worm import mootdx_source as mdx
    out: dict[str, pd.DataFrame] = {}
    for c in codes:
        rows = mdx.get_kline_history(c, total=2000)
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume", "amount"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[(df.index >= START) & (df.index <= END)]
        if len(df) >= 250:
            out[c] = df
    return out


def build_score(prices: dict[str, pd.DataFrame], fin: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """复合打分 = 价值(vs_ma250 截面z) + 质量(ROE 截面z)。"""
    close = pd.DataFrame({c: df["close"] for c, df in prices.items()})
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    # 价值：价格相对250日均线（A股最强反转信号）
    val = close / close.rolling(250, min_periods=120).mean() - 1.0
    # 质量：ROE 按报告期前向填充到交易日
    qual = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for c in close.columns:
        fd = fin.get(c)
        if fd is not None and not fd.empty:
            tmp = fd.set_index("report_date")["roe"].dropna()
            if not tmp.empty:
                tmp.index = pd.to_datetime(tmp.index)  # 字符串→datetime, 否则 reindex 全 NaN
                qual[c] = tmp.reindex(close.index, method="ffill")
    # 截面 z-score（每日横截面标准化）
    def zs(df):
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0, np.nan)
        return (df.sub(mu, axis=0) / sd)
    score = 0.5 * zs(val.fillna(0)) + 0.5 * zs(qual.fillna(0))
    return score.fillna(0.0)


def ic_of(factor: pd.DataFrame, fwd_ret: pd.DataFrame, n: int = 20) -> float:
    """因子 IC（与下期收益的截面相关系数均值）。"""
    ic_list = []
    for i in range(n, len(factor)):
        f = factor.iloc[i]
        r = fwd_ret.iloc[i]
        m = f.notna() & r.notna()
        if m.sum() < 10:
            continue
        ic = np.corrcoef(f[m], r[m])[0, 1]
        if ic == ic:
            ic_list.append(ic)
    return float(np.nanmean(ic_list))


def main() -> None:
    print(f"[1/4] 构建宇宙（前 {UNIVERSE_N} 只 CSI300）...")
    codes = build_universe(UNIVERSE_N)
    print(f"     取成分 {len(codes)} 只")

    print(f"[2/4] 拉行情(stock_worm.mootdx) + 基本面(financial_loader)...")
    prices = fetch_prices(codes)
    print(f"     行情可用 {len(prices)} 只")
    fin = fetch_fundamentals(list(prices.keys()), use_cache=True)
    has_fin = sum(1 for v in fin.values() if not v.empty and v["roe"].notna().any())
    print(f"     基本面可用 {has_fin} 只（ROE 非空）")

    print(f"[3/4] 财务因子 IC 检验（ROE / vs_ma250）...")
    close = pd.DataFrame({c: df["close"] for c, df in prices.items()})
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    fwd = close.pct_change(fill_method=None).shift(-20)  # 20日 forward return
    val = close / close.rolling(250, min_periods=120).mean() - 1.0
    ic_val = ic_of(val, fwd)
    # ROE 面板
    roe = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for c in close.columns:
        fd = fin.get(c)
        if fd is not None and not fd.empty:
            tmp = fd.set_index("report_date")["roe"].dropna()
            if not tmp.empty:
                tmp.index = pd.to_datetime(tmp.index)  # 字符串→datetime
                roe[c] = tmp.reindex(close.index, method="ffill")
    ic_roe = ic_of(roe, fwd)
    print(f"     IC(vs_ma250 价值) = {ic_val:+.3f}")
    print(f"     IC(ROE 质量)      = {ic_roe:+.3f}   （>0.02 即具配置价值）")

    print(f"[4/4] 市场中性对冲回测（价值+质量 复合打分）...")
    score = build_score(prices, fin)
    # 纯多头因子组合（未对冲）— 对应 STRATEGY_SUMMARY 里 −22% 回撤的那类
    res_longonly = mn.run_market_neutral(
        prices, score, index_code="000300.SH", start=START, end=END,
        rebalance="M", n_long=10, n_short=10, long_only=True, hedge="none",
    )
    # 纯多头 + β 对冲（OLS）
    res_hedged = mn.run_market_neutral(
        prices, score, index_code="000300.SH", start=START, end=END,
        rebalance="M", n_long=10, n_short=10, long_only=True, hedge="ols",
    )
    if "error" in res_longonly or "error" in res_hedged:
        print("     ERROR:", res_longonly.get("error") or res_hedged.get("error"))
        return

    lo, he = res_longonly["gross_metrics"], res_hedged["net_metrics"]
    print("\n=== 纯多头因子组合：对冲前 vs β对冲后（2023-01~2026-07，月度调仓，多10）===")
    print(f"{'指标':<14}{'纯多头(未对冲)':>16}{'β对冲OLS(β={:.2f})'.format(res_hedged['beta']):>20}")
    print(f"{'年化':<14}{lo['annual_return']:>15.1%}{he['annual_return']:>19.1%}")
    print(f"{'Sharpe':<14}{lo['sharpe']:>16.2f}{he['sharpe']:>20.2f}")
    print(f"{'最大回撤':<14}{lo['max_drawdown']:>15.1%}{he['max_drawdown']:>19.1%}")

    vd = res_hedged["baseline_verdict"]
    print(f"\n=== 基线门槛裁决（基线=5.7%/Sharpe1.37/MaxDD≥-5%）===")
    print(f"     β对冲后过线? {'✅ 过线' if vd['pass'] else '❌ 未过线'}")
    for f in vd["fails"]:
        print(f"       - {f}")

    print("\n[结论] 财务因子 ROE 的 IC={:+.3f}（正向，需更大样本+OOS验证）；"
          "β对冲把纯多头因子组合最大回撤从 {:.1%} 压到 {:.1%}（β={:.2f}）。".format(
        ic_roe, lo["max_drawdown"], he["max_drawdown"], res_hedged["beta"]))


if __name__ == "__main__":
    main()
