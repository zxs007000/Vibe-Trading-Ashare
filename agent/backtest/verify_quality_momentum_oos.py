"""quality_momentum 全市场 walk-forward 样本外验证 (纯 stock_worm 源).

数据源 (全部来自用户自己的 stock_worm, 不依赖外部代理 / 不用 akshare):
    - K线:    stock_worm.mootdx_source.get_kline_history  (通达信 TCP 直连, 自动翻页)
    - 成分股: stock_worm._session.eastmoney_datacenter     (东财报表, 直连)
    - 财务:    stock_worm.fundamentals (financial_loader 优先源)

目的:
    1. 广度检验: 在 CSI300 宽池上重跑 quality_momentum(vol-targeted), 对比 40 蓝筹的 0.830
    2. 时间样本外: 2019-2021 训练 / 2022-2024 测试, 看 0.830 是否窄样本虚高
"""

from __future__ import annotations

import sys, time, socket
socket.setdefaulttimeout(12)  # 防 mootdx recv 永久阻塞导致进程闷死
from pathlib import Path

import numpy as np
import pandas as pd

# 强制使用刚改过的 stock_worm 源码 (D:/stcok-worm), 确保 mootdx 翻页生效
_STOCK_WORM_SRC = r"D:\stcok-worm"
if _STOCK_WORM_SRC not in sys.path:
    sys.path.insert(0, _STOCK_WORM_SRC)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENT_DIR = _PROJECT_ROOT / "agent"
for _p in (str(_PROJECT_ROOT), str(_AGENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.backtest.loaders.financial_loader import fetch_fundamentals
from agent.backtest.metrics import calc_metrics

INITIAL = 1_000_000
TARGET_VOL = 0.15
VOL_WINDOW = 63
MAX_LEVERAGE = 2.0
MIN_LEVERAGE = 0.1


# ── 8 original factors (与 verify_phase3 完全一致) ──────────────
def f_value(c): return (-c.pct_change(252)+(c-c.rolling(250).mean())/c.rolling(250).mean()).fillna(0)
def f_momentum(c): h52=c.rolling(252).max(); return (c.pct_change(20)+(c-h52)/h52.replace(0,1)).fillna(0)
def f_quality(c,v): v60=v.rolling(60).mean(); dv=-np.log(v60/v60.shift(60).replace(0,1)); return (dv-c.pct_change().rolling(60).std()).fillna(0)
def f_volatility(c): r=c.pct_change(); return (-r.rolling(60).skew()-r.rolling(20).std()*np.sqrt(252)*0.1).fillna(0)
def f_liquidity(c,v): r=c.pct_change().abs(); return (-(r/(c*v+1)).rolling(21).mean()+v/v.rolling(20).mean()-1).fillna(0)
def f_reversal(c): return (-c.pct_change(5)-0.5*c.pct_change(60)).fillna(0)
def f_volume(c,v): vma=v.rolling(20).mean(); return (v.pct_change(5)+(v-vma)/vma.replace(0,1)).fillna(0)
def f_micro(c,h,l): return ((h-l)/c+(c-l)/(h-l+1e-10)).fillna(0)

FACTORS_ORIG = {
    "value":          lambda df: f_value(df["close"]),
    "momentum":       lambda df: f_momentum(df["close"]),
    "quality":        lambda df: f_quality(df["close"], df["volume"]),
    "volatility":     lambda df: f_volatility(df["close"]),
    "liquidity":      lambda df: f_liquidity(df["close"], df["volume"]),
    "reversal":       lambda df: f_reversal(df["close"]),
    "volume":         lambda df: f_volume(df["close"], df["volume"]),
    "microstructure": lambda df: f_micro(df["close"], df["high"], df["low"]),
}


# ── quality_momentum 三阶漏斗 (与 verify_phase3.run_pipeline 一致) ──
def run_pipeline(chains_dict, stocks, pre, common, keep_frac=0.50):
    results = {}
    for cid, stages in chains_dict.items():
        rets = []
        for i, d in enumerate(common):
            if i == len(common)-1: break
            nd = common[i+1]; pool = set(stocks.keys())
            for stage in stages:
                if stage not in next(iter(pre.values())): break
                scores = {s: pre[s][stage].get(d, np.nan) for s in pool}
                scores = {s: v for s, v in scores.items() if not np.isnan(v)}
                if len(scores) < 3: pool = set(); break
                keep = max(2, int(len(scores) * keep_frac))
                pool = {s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:keep]}
            if len(pool) >= 2:
                rets.append(np.mean([stocks[s]["return"].get(nd, 0) for s in pool]))
        if len(rets) >= 30:
            results[cid] = pd.Series(rets, index=common[:len(rets)])
    return results


def apply_vol_targeting(chain_rets: dict[str, pd.Series]) -> dict[str, pd.Series]:
    targeted = {}
    for cid, daily_rets in chain_rets.items():
        rolled = daily_rets.rolling(VOL_WINDOW, min_periods=20).std() * np.sqrt(252)
        raw_w = TARGET_VOL / rolled.replace(0, np.nan)
        raw_w = raw_w.clip(MIN_LEVERAGE, MAX_LEVERAGE).fillna(1.0)
        w = raw_w.ewm(alpha=0.06).mean()
        scaled = daily_rets * w.shift(1).fillna(1.0)
        targeted[cid] = scaled.dropna()
    return targeted


def calc_stats(rets: pd.Series) -> dict:
    eq = (1.0 + rets).cumprod() * INITIAL
    return calc_metrics(eq, trades=[], initial_cash=INITIAL, bars_per_year=252)


# ── stock_worm: 成分股 ──────────────────────────────────────────
def build_universe() -> list[str]:
    """CSI300 成分股 (stock_worm 东财报表, 直连无需代理)."""
    from stcok_worm._session import eastmoney_datacenter
    rows = eastmoney_datacenter("RPT_INDEX_CONSTITUENT", columns="ALL",
                                filter_str='(INDEX_CODE="000300")', page_size=1000)
    codes: set[str] = set()
    for r in rows:
        for k in ("SECURITY_CODE", "CODE", "SYMBOL", "SECUCODE"):
            v = r.get(k)
            if v:
                codes.add(str(v).split(".")[0])
                break
    if codes:
        print(f"    stock_worm 东财成分股: {len(codes)} 只 (CSI300)")
        return sorted(codes)
    return []


# ── stock_worm: K线 (mootdx TCP 直连, 翻页) ────────────────────
def pull_one(code: str) -> pd.DataFrame | None:
    from stcok_worm import mootdx_source as mdx
    rows = mdx.get_kline_history(code, total=2000)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["return"] = df["close"].pct_change()
    return df


def run_qm(stocks: dict, pre_full: dict, window_dates) -> tuple[pd.Series | None, pd.Series | None]:
    """在指定时间窗口上跑 quality_momentum, 返回 (ungated, vol-targeted)."""
    pre = {s: pre_full[s].reindex(window_dates) for s in stocks}
    rets = run_pipeline({"quality_momentum": ["quality", "momentum", "volume"]},
                         stocks, pre, list(window_dates), keep_frac=0.50)
    qm = rets.get("quality_momentum")
    if qm is None or len(qm) < 30:
        return None, None
    vt = apply_vol_targeting({"quality_momentum": qm})["quality_momentum"]
    return qm, vt


def main():
    print("=" * 72)
    print("  quality_momentum 全市场 OOS (纯 stock_worm 源: mootdx + 东财)")
    print("=" * 72)

    # 1. 成分股
    print("\n[1] 构建股票池 (CSI300)...")
    syms = build_universe()
    if not syms:
        print("    成分股为空, 退出"); return
    # 限 300 只即可 (mootdx 串行约 25min)
    syms = syms[:300]

    # 2. 拉 K线 (mootdx TCP 直连)
    print(f"\n[2] 拉 K线 (stock_worm.mootdx, 翻页全历史) x {len(syms)}...")
    stocks: dict = {}
    t0 = time.time()
    for i, s in enumerate(syms):
        df = pull_one(s)
        if df is not None and len(df) > 1500:
            stocks[s] = df
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(syms)}, valid {len(stocks)}  ({time.time()-t0:.0f}s)")
    print(f"    Got {len(stocks)} valid stocks  ({time.time()-t0:.0f}s)")

    # 3. 公共交易日 (交集)
    common = stocks[list(stocks.keys())[0]].index
    for d in stocks.values():
        common = common.intersection(d.index)
    common = common.sort_values()
    print(f"\n[3] 公共交易日: {len(common)}  ({common[0].date()} ~ {common[-1].date()})")

    # 4. 计算 8 原始因子
    print(f"\n[4] 计算因子面板...")
    pre_full = {}
    for sym, df in stocks.items():
        pf = pd.DataFrame(index=df.index)
        for th, fn in FACTORS_ORIG.items():
            pf[th] = fn(df)
        pre_full[sym] = pf.reindex(common)

    # 5. 三个窗口
    recent = common[-600:]
    train = common[(common >= "2019-01-01") & (common <= "2021-12-31")]
    test = common[(common >= "2022-01-01") & (common <= "2024-12-31")]
    print(f"\n[5] 窗口: recent600={len(recent)}  train(19-21)={len(train)}  test(22-24)={len(test)}")

    def _report(name, window):
        qm, vt = run_qm(stocks, pre_full, window)
        if vt is None:
            print(f"    {name}: 数据不足, 跳过"); return None
        mu = calc_stats(qm); mv = calc_stats(vt)
        print(f"    {name:<14} ungated Sharpe={mu['sharpe']:.3f}  "
              f"vol-tgt Sharpe={mv['sharpe']:.3f}  AnnRet={mv['annual_return']:.1%}  "
              f"MaxDD={mv['max_drawdown']:.1%}")
        return mv

    print(f"\n    ═══ quality_momentum 结果 (N={len(stocks)}) ═══")
    r_recent = _report("recent600d", recent)
    r_train = _report("train 19-21", train)
    r_test = _report("test 22-24", test)

    print("\n    ═══ 结论对照 ═══")
    print("    40 蓝筹基准 (verify_phase3): quality_momentum(vol-tgt) Sharpe ≈ 0.830")
    if r_recent and r_test:
        print(f"    CSI300 广度(recent600): Sharpe={r_recent['sharpe']:.3f}")
        print(f"    CSI300 样本外(test22-24): Sharpe={r_test['sharpe']:.3f}")
        print(f"    → 0.830 是否虚高: {'是(样本外明显低于0.83)' if r_test['sharpe'] < 0.65 else '否(样本外仍≥0.65)'}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
