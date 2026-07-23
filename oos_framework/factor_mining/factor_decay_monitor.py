#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factor_decay_monitor.py — 因子衰减监控 + 再冻结触发
====================================================

用户框架核心: "因子有寿命". P3 筛选是一次性的, 但因子会随时间衰减.
本模块持续监控入选因子池的 IC 健康度, 在衰减发生时触发"再冻结"标志.

功能:
  1) 滚动 IC 监控: 对每个入选因子, 算逐月滚动 Rank-IC + ICIR
  2) 衰减判定: 当因子近 60d IC 均值 < 历史均值 × DECAY_FRAC(30%) → 标记衰减
  3) 再冻结触发: 当衰减因子占比 > REFREEZE_THRESHOLD(40%) → 触发再冻结
  4) 报告: 输出因子健康度总览 + 衰减清单 + 再冻结建议

用法:
  python factor_decay_monitor.py [--stocks 1500] [--out FACTOR_DECAY_REPORT.md]

输入: factors_v2_selected.json (P3 入选因子集)
输出: FACTOR_DECAY_REPORT.md (因子健康度 + 衰减判定 + 再冻结建议)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from factor_mining.base_data import list_stocks, load_base_data, derive_variables, forward_returns
from factor_mining.operators import evaluate_expr
from factor_mining.factor_wfa import _l2t, wfa_folds
from factor_mining.universe import load_universe

# 监控参数
ROLL_IC_WIN = 60          # 滚动 IC 窗口(交易日, ≈ 3 月)
BASE_IC_WIN = 250         # 基准 IC 窗口(交易日, ≈ 1 年)
DECAY_FRAC = 0.30         # 衰减阈值: 近窗 IC < 历史 IC × 30% → 衰减
REFREEZE_THRESHOLD = 0.40 # 衰减因子占比 > 40% → 触发再冻结
MONTHLY_STEP = 21         # 月度采样步长(约 21 交易日)


def _daily_rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """逐日截面 Rank-IC(Spearman), 返回日频 IC 序列."""
    common = factor.index.intersection(fwd.index)
    ics = []
    for d in common:
        f = factor.loc[d]
        r = fwd.loc[d]
        m = f.notna() & r.notna()
        if m.sum() < 30:
            ics.append(np.nan)
            continue
        rho = spearmanr(f[m], r[m]).correlation
        ics.append(rho if np.isfinite(rho) else np.nan)
    return pd.Series(ics, index=common)


def monitor_factors(seen: dict, codes: list[str], fwd_horizon: int = 20) -> pd.DataFrame:
    """对入选因子池算逐月滚动 IC + 衰减判定.

    返回 DataFrame: 每行一个因子, 列含 is_ic/icir/recent_ic/decay_flag/verdict.
    """
    t0 = time.time()
    data = derive_variables(load_base_data(codes))
    fwd = forward_returns(data["close"], horizons=(fwd_horizon,))[fwd_horizon]

    rows = []
    for name, meta in seen.items():
        try:
            fac = evaluate_expr(meta["expr_tuple"], data)
        except Exception:
            rows.append({"name": name, "verdict": "求值失败", "decay_flag": True,
                         "is_ic": np.nan, "icir": np.nan, "recent_ic": np.nan,
                         "base_ic": np.nan, "decay_ratio": np.nan})
            continue

        ic_series = _daily_rank_ic(fac, fwd)
        if len(ic_series.dropna()) < 60:
            rows.append({"name": name, "verdict": "样本不足", "decay_flag": True,
                         "is_ic": float(ic_series.mean()) if len(ic_series) else np.nan,
                         "icir": np.nan, "recent_ic": np.nan,
                         "base_ic": np.nan, "decay_ratio": np.nan})
            continue

        is_ic = float(ic_series.mean())
        ic_std = float(ic_series.std())
        icir = is_ic / (ic_std + 1e-9) * np.sqrt(252) if ic_std > 1e-9 else 0.0

        # 滚动 IC
        roll_mean = ic_series.rolling(ROLL_IC_WIN, min_periods=20).mean()
        base_mean = ic_series.rolling(BASE_IC_WIN, min_periods=60).mean()

        recent_ic = float(roll_mean.iloc[-1]) if len(roll_mean) else np.nan
        base_ic = float(base_mean.iloc[-1]) if len(base_mean) else is_ic

        # 衰减判定: 近窗 IC < 历史 × DECAY_FRAC, 或近窗为负
        decay_ratio = recent_ic / (base_ic + 1e-9) if base_ic and abs(base_ic) > 1e-9 else np.nan
        decay_flag = (np.isfinite(recent_ic) and np.isfinite(base_ic) and base_ic > 0
                      and recent_ic < base_ic * DECAY_FRAC)
        if np.isfinite(recent_ic) and recent_ic < 0:
            decay_flag = True

        if decay_flag:
            verdict = "⚠️ 衰减(建议降权/剔除)"
        elif np.isfinite(recent_ic) and recent_ic > 0 and recent_ic > base_ic * 0.7:
            verdict = "✅ 健康"
        else:
            verdict = "⚠️ 弱化(观察)"

        rows.append({
            "name": name, "is_ic": round(is_ic, 4), "icir": round(icir, 2),
            "recent_ic": round(recent_ic, 4) if np.isfinite(recent_ic) else np.nan,
            "base_ic": round(base_ic, 4) if np.isfinite(base_ic) else np.nan,
            "decay_ratio": round(decay_ratio, 2) if np.isfinite(decay_ratio) else np.nan,
            "decay_flag": decay_flag, "verdict": verdict,
        })
        print(f"  [{len(rows)}/{len(seen)}] {name[:60]}: IC={is_ic:+.4f} "
              f"recent={recent_ic:+.4f} {'⚠️' if decay_flag else '✅'}", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n[decay] {len(df)} 因子监控完成 | {time.time()-t0:.0f}s", flush=True)
    return df


def build_report(df: pd.DataFrame, n_stocks: int, t0: float) -> str:
    n_total = len(df)
    n_decay = int(df["decay_flag"].sum())
    n_healthy = int((df["verdict"] == "✅ 健康").sum())
    n_weak = int((df["verdict"].startswith("⚠️ 弱化")).sum() if df["verdict"].dtype == object else 0)
    decay_rate = n_decay / n_total if n_total else 0
    refreeze = decay_rate > REFREEZE_THRESHOLD

    # 按衰减排序
    df_sorted = df.sort_values("decay_flag", ascending=False)

    lines = ["# 因子衰减监控 + 再冻结判定报告\n",
             f"- 监控样本: {n_stocks} 只 | 因子池: {n_total} 个 | 滚动IC窗: {ROLL_IC_WIN}d | "
             f"基准窗: {BASE_IC_WIN}d | 衰减阈值: IC < 历史×{DECAY_FRAC:.0%}",
             f"- **衰减因子: {n_decay}/{n_total} ({decay_rate:.0%})** | 健康: {n_healthy} | "
             f"弱化: {n_weak}",
             f"- 再冻结阈值: 衰减占比 > {REFREEZE_THRESHOLD:.0%} → "
             f"{'**🔥 触发再冻结**(用最近IS期重新算ICIR权重)' if refreeze else '未触发(因子池仍稳健)'}\n",
             "## 因子健康度总览(按衰减程度排序)\n",
             "| 因子 | IS_IC | ICIR | 近期IC | 历史IC | 衰减比 | 状态 |",
             "|---|---|---|---|---|---|---|"]
    for _, r in df_sorted.iterrows():
        ri = f"{r['recent_ic']:+.4f}" if pd.notna(r['recent_ic']) else "NaN"
        bi = f"{r['base_ic']:+.4f}" if pd.notna(r['base_ic']) else "NaN"
        dr = f"{r['decay_ratio']:.2f}" if pd.notna(r['decay_ratio']) else "NaN"
        lines.append(f"| `{r['name'][:60]}` | {r['is_ic']:+.4f} | {r['icir']:+.2f} | "
                     f"{ri} | {bi} | {dr} | {r['verdict']} |")

    lines.append(f"\n## 再冻结建议\n")
    if refreeze:
        decay_names = df[df["decay_flag"]]["name"].tolist()
        lines.append(f"- **触发再冻结**: 衰减占比 {decay_rate:.0%} > 阈值 {REFREEZE_THRESHOLD:.0%}")
        lines.append(f"- 建议动作: 用最近 {BASE_IC_WIN}d 作为新 IS 期, 重新算 ICIR 权重, "
                     f"剔除 {len(decay_names)} 个衰减因子, 重新冻结剩余因子集.")
        lines.append(f"- 衰减因子清单: {', '.join(n[:40] for n in decay_names[:10])}"
                     f"{'...' if len(decay_names) > 10 else ''}")
    else:
        lines.append(f"- 未触发再冻结: 衰减占比 {decay_rate:.0%} ≤ 阈值 {REFREEZE_THRESHOLD:.0%}")
        lines.append(f"- 因子池仍稳健, 继续使用当前冻结权重. 下次监控建议在 1 个月后.")

    lines.append(f"\n## 衰减因子详情\n")
    decay_df = df[df["decay_flag"]].sort_values("recent_ic")
    if len(decay_df):
        for _, r in decay_df.iterrows():
            ri = f"{r['recent_ic']:+.4f}" if pd.notna(r['recent_ic']) else "NaN"
            dr = f"{r['decay_ratio']:.2f}" if pd.notna(r['decay_ratio']) else "NaN"
            lines.append(f"- `{r['name'][:60]}`: IS IC={r['is_ic']:+.4f} → 近期 IC={ri} (衰减比 {dr})")
    else:
        lines.append("- 无衰减因子 ✅")

    lines.append(f"\n---\n*由 `factor_decay_monitor.py` 生成, 耗时 {time.time()-t0:.0f}s*")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=1500)
    ap.add_argument("--out", default="FACTOR_DECAY_REPORT.md")
    ap.add_argument("--factors_json", default="factors_v2_selected.json")
    ap.add_argument("--fwd_horizon", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()

    fj = args.factors_json
    if not os.path.isabs(fj):
        fj = os.path.join(HERE, fj)
    if not os.path.exists(fj):
        # 回退到 factors_discovered.json
        fj = os.path.join(HERE, "factors_discovered.json")
        if not os.path.exists(fj):
            print(f"未找到因子清单, 先运行 mine_v2.py 或 collect")
            return
    with open(fj, encoding="utf-8") as f:
        dump = json.load(f)
    # 兼容两种格式: list[str](仅因子名) 或 dict[name -> {expr_tuple_list, dir}]
    if isinstance(dump, list):
        # 因子名列表 → 需要从 factors_v2_3dim.json 加载 expr_tuple
        dim_fj = os.path.join(HERE, "factors_v2_3dim.json")
        if os.path.exists(dim_fj):
            with open(dim_fj, encoding="utf-8") as f2:
                dim = json.load(f2)
            seen = {k: {"expr_tuple": _l2t(v["expr_tuple_list"]), "dir": v.get("dir", 1)}
                    for k, v in dim.items() if k in dump}
        else:
            # 回退: 从 factors_discovered.json 加载
            disc_fj = os.path.join(HERE, "factors_discovered.json")
            with open(disc_fj, encoding="utf-8") as f2:
                disc = json.load(f2)
            seen = {k: {"expr_tuple": _l2t(v["expr_tuple_list"]), "dir": v.get("dir", 1)}
                    for k, v in disc.items() if k in dump}
    else:
        seen = {k: {"expr_tuple": _l2t(v.get("expr_tuple_list", v.get("expr_tuple"))),
                    "dir": v.get("dir", 1)} for k, v in dump.items()}
    print(f"[decay] 因子 {len(seen)} 个 | 复用 {fj}", flush=True)

    uni = load_universe()
    if uni:
        rng = np.random.default_rng(7)
        codes = sorted(rng.choice(uni, min(args.stocks, len(uni)), replace=False).tolist())
    else:
        codes = list_stocks(args.stocks)
    print(f"[decay] 监控样本 {len(codes)} 只", flush=True)

    df = monitor_factors(seen, codes, fwd_horizon=args.fwd_horizon)

    n_decay = int(df["decay_flag"].sum())
    decay_rate = n_decay / len(df) if len(df) else 0
    print(f"\n[decay] 衰减因子 {n_decay}/{len(df)} ({decay_rate:.0%}) | "
          f"再冻结: {'触发' if decay_rate > REFREEZE_THRESHOLD else '未触发'}", flush=True)

    md = build_report(df, len(codes), t0)
    outp = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    with open(outp, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n报告: {outp}  (耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
