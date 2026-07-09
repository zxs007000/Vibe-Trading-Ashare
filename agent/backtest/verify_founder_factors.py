"""verify_founder_factors.py — 方正金工因子真实数据回测

用 mootdx 取A股5分钟K线,对方正金工因子做月度多空回测。
因 mootdx 5分钟接口只能取最近约800根(约3-4天交易日),
本脚本用有限样本做因子有效性初验,非完整回测。

策略:
  1. 取50只CSI300成分股的5分钟数据
  2. 计算每个因子值(单日截面)
  3. 按因子值分5组,做多最高组/做空最低组
  4. 计算截面IC(因子值 vs 未来收益率)
"""

from __future__ import annotations

import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_PROJECT = Path(__file__).resolve().parents[2]
_AGENT = _PROJECT / "agent"
sys.path.insert(0, str(_PROJECT)); sys.path.insert(0, str(_AGENT))
sys.path.insert(0, str(_PROJECT / "agent" / "backtest" / "factors"))

from founder import (
    drip_water_stone, smart_money, moderate_risk,
    complete_tide, scaling_heights, clouds_disperse,
    moth_to_flame, wait_rescue, equal_treatment,
)

SYMS = [
    "600519","000858","601318","600036","000333","601899","300750","601166",
    "600900","000651","600276","601398","000001","603259","600030","002415",
    "601288","600809","000725","601088","601012","002714","000002","600887",
    "601857","600028","688981","002475","300059","601688","000063","300124",
    "600585","600309","600436","002594","601225","603288","002304","000568",
    "601995","600570","002352","300015","601066","600104","601628","000776",
    "300498","002230",
]


def pull_5min(code: str, count: int = 800) -> pd.DataFrame | None:
    """用 stcok_worm (通达信 TCP, 第一优先数据源) 取5分钟K线, 含 amount。

    经由 stcok_worm.mootdx_source.get_kline(freq=0) 直连通达信 7709 端口,
    返回完整时间戳 + 成交额(amount) 列, 直接满足 clouds_disperse /
    rapids_advance 对分钟级成交额的数据需求(无需 close*volume 近似)。
    """
    try:
        from stcok_worm import mootdx_source
        raw = mootdx_source.get_kline(code, 0, count)
        if not raw:
            return None
        df = pd.DataFrame({
            "open": [float(r["open"]) for r in raw],
            "high": [float(r["high"]) for r in raw],
            "low": [float(r["low"]) for r in raw],
            "close": [float(r["close"]) for r in raw],
            "volume": [float(r.get("volume", 0.0)) for r in raw],
            "amount": [float(r.get("amount", 0.0)) for r in raw],
        }, index=pd.to_datetime([str(r["date"]) for r in raw]))
        return df
    except Exception:
        return None


def pull_daily(code: str, days: int = 300) -> pd.DataFrame | None:
    """用腾讯接口取日频数据。"""
    import requests
    try:
        sym = f"sh{code}" if code.startswith("6") else f"sz{code}"
        r = requests.get(
            f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,{days},qfq",
            timeout=10
        )
        d = r.json()["data"][code]
        k = d.get("qfqday", d.get("day", []))
        k = [x[:6] for x in k]
        df = pd.DataFrame(k, columns=["date","open","close","high","low","volume"])
        for c in ["open","close","high","low","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["return"] = df["close"].pct_change()
        return df
    except Exception:
        return None


def compute_factor_values(stocks_5min: dict, factor_name: str) -> dict:
    """对每只股票计算因子值(取最近有效值)。"""
    factor_fn = {
        "drip_water_stone": drip_water_stone,
        "smart_money": smart_money,
        "moderate_risk": moderate_risk,
        "complete_tide": complete_tide,
        "scaling_heights": scaling_heights,
        "clouds_disperse": clouds_disperse,
        "moth_to_flame": moth_to_flame,
        "wait_rescue": wait_rescue,
        "equal_treatment": equal_treatment,
    }[factor_name]

    values = {}
    for code, bars in stocks_5min.items():
        try:
            f = factor_fn(bars)
            v = f.dropna()
            if len(v) > 0:
                values[code] = v.iloc[-1]
        except Exception:
            continue
    return values


def main():
    print("=" * 72)
    print("  方正金工因子真实数据回测 (mootdx 5分钟K线)")
    print("=" * 72)

    # ── 取5分钟数据 ──
    print("\n[1] 拉取50只股票5分钟数据...")
    stocks_5min = {}
    for i, s in enumerate(SYMS):
        df = pull_5min(s, 800)
        if df is not None and len(df) > 100:
            stocks_5min[s] = df
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/50, {len(stocks_5min)} valid")
        time.sleep(0.3)
    print(f"    {len(stocks_5min)} stocks OK")

    if len(stocks_5min) < 10:
        print("    ⚠ 数据不足, 退出")
        return

    # ── 从5分钟数据合成日频收益(未来1天收益) ──
    print("\n[2] 从5分钟数据合成日频收益...")
    future_rets = {}
    for code, df in stocks_5min.items():
        # 按日分组,取每日收盘价
        daily_close = df["close"].resample("D").last().dropna()
        if len(daily_close) >= 2:
            # 未来1天收益 = 明日收盘/今日收盘 - 1
            future_rets[code] = daily_close.iloc[-1] / daily_close.iloc[-2] - 1
    print(f"    {len(future_rets)} stocks有未来收益")
    factor_names = [
        "drip_water_stone", "smart_money", "moderate_risk",
        "complete_tide", "scaling_heights", "clouds_disperse",
        "moth_to_flame", "wait_rescue", "equal_treatment",
    ]

    print(f"\n[3] 截面因子分析 (最新交易日)...")
    print(f"  {'Factor':<20} {'N':>4} {'Mean':>12} {'Std':>12} {'IC_vs_ret':>12} {'Direction':>10}")
    print(f"  {'─'*20} {'─'*4} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")

    all_results = {}
    for fname in factor_names:
        fvals = compute_factor_values(stocks_5min, fname)
        if len(fvals) < 10:
            print(f"  {fname:<20} {len(fvals):>4}  (数据不足)")
            continue

        # 截面统计
        vals = np.array(list(fvals.values()))
        mean_v, std_v = np.nanmean(vals), np.nanstd(vals)

        # IC: 因子值 vs 未来收益
        paired = [(v, future_rets.get(c, np.nan)) for c, v in fvals.items()]
        paired = [(v, r) for v, r in paired if not np.isnan(r)]
        if len(paired) > 10:
            xs = [p[0] for p in paired]
            ys = [p[1] for p in paired]
            ic = np.corrcoef(xs, ys)[0, 1]
            direction = "正向" if ic > 0 else "反向"
        else:
            ic = np.nan
            direction = "N/A"

        all_results[fname] = {
            "values": fvals, "ic": ic, "n": len(fvals),
            "mean": mean_v, "std": std_v, "direction": direction,
        }
        print(f"  {fname:<20} {len(fvals):>4} {mean_v:>12.6f} {std_v:>12.6f} {ic:>12.4f} {direction:>10}")

    # ── 分组回测(对IC最显著的3个因子) ──
    print(f"\n[4] 分组回测 (5组, 最新截面)...")
    sorted_factors = sorted(all_results.items(), key=lambda x: abs(x[1]["ic"]) if not np.isnan(x[1]["ic"]) else 0, reverse=True)

    for fname, info in sorted_factors[:5]:
        vals = info["values"]
        if len(vals) < 15:
            continue
        ranked = sorted(vals.items(), key=lambda x: x[1])
        n = len(ranked)
        gsize = n // 5

        print(f"\n  ── {fname} (IC={info['ic']:.4f}) ──")
        print(f"  {'Group':<8} {'FactorRange':>20} {'MeanRet':>10} {'N':>4}")
        print(f"  {'─'*8} {'─'*20} {'─'*10} {'─'*4}")

        group_rets = []
        for g in range(5):
            start = g * gsize
            end = (g + 1) * gsize if g < 4 else n
            g_stocks = [s for s, _ in ranked[start:end]]
            g_factor = [v for _, v in ranked[start:end]]
            g_rets = [future_rets.get(s, np.nan) for s in g_stocks]
            g_rets_valid = [r for r in g_rets if not np.isnan(r)]
            mean_ret = np.mean(g_rets_valid) if g_rets_valid else np.nan
            group_rets.append(mean_ret)
            fr = f"[{min(g_factor):.4f}, {max(g_factor):.4f}]"
            print(f"  G{g:<7} {fr:>20} {mean_ret:>9.4f} {len(g_stocks):>4}")

        if not any(np.isnan(group_rets)):
            ls = group_rets[4] - group_rets[0]  # 最高组 - 最低组
            print(f"  Long-Short(G4-G0): {ls:.4f}")

    print(f"\n{'='*72}")
    print(f"  注意: mootdx 5分钟数据仅最近3-4天, 此为初验非完整回测。")
    print(f"  IC方向对照论文(负IC因子在A股做空低组=做多高组):")
    for fname, info in all_results.items():
        print(f"    {fname}: IC={info['ic']:.4f} ({info['direction']})")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
