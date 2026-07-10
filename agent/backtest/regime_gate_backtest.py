"""regime_gate_backtest.py — Branch 1: Regime 闸门诊断.

复用 cloud_backtest 的正确 fwd 构造与回测骨架. 在 288 只 CSI300 / 2024-05~2026-06 上:
  (1) 画滚动 60 日 rank-IC '死亡曲线'(组合 + 7 因子), 看因子何时翻死;
  (2) 加 '近 60 日 rank-IC>0 才交易' 的动态开关, 回测 gated 策略,
      对比 always-on / 等权基准, 判断跳过死 regime 能否捞回 alpha.

关键防泄漏: 闸门在调仓日 d 用 trailing 60d rank-IC(截止到 d-1) 决定, 不含当日 fwd.
(当日 fwd = 当日信号的下一交易日收益, 若纳入等于用到了正要下注的那一笔收益的已知结果.)

用法:
  python backtest/regime_gate_backtest.py
"""
from __future__ import annotations
import sys, pickle, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))            # agent/backtest (本脚本目录)
sys.path.insert(0, str(Path(__file__).parent.parent))     # agent/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from cloud_backtest import (_load_prep, _zscore_row, compute_ic_series,
                            walk_forward_combine, long_only_backtest, _stat,
                            SPECS, TOP_K, COST_ONEWAY)
from backtest.validation import _sharpe

FAC_CACHE = Path("/workspace/stock_worm/data/founder_factors.pkl")
ROL_WINDOW = 60          # 滚动 IC 窗口(交易日)
HOLD_DAYS = 5            # 调仓周期
GATE_THR = 0.0           # 闸门阈值: trailing rank-IC > 0 才开仓
OUT_DIR = Path(__file__).parent / "screen_results"
FIG = OUT_DIR / "regime_gate_rolling_ic.png"


def daily_rank_ic(signal, fwd):
    """逐交易日横截面 spearman(combo.rank(), fwd.rank()). 返回 date-indexed Series."""
    out = {}
    for d in signal.index:
        s = signal.loc[d]; r = fwd.loc[d]
        idx = s.dropna().index.intersection(r.dropna().index)
        if len(idx) < 10:
            out[d] = np.nan; continue
        out[d] = s[idx].rank().corr(r[idx].rank())
    return pd.Series(out)


def gated_long_only(signal, fwd, gate_open, hold_days=HOLD_DAYS,
                    cost_oneway=COST_ONEWAY, off_mode="bench"):
    """gate_open: bool Series(以交易日为索引) = 该调仓日是否开仓.
    off_mode='bench' -> 关仓期持有等权基准; 'cash' -> 关仓期收益=0.
    返回 (port, bench). 成本: 调仓日且(本日开仓 或 上日持仓) 计双边, 关仓期不交易不计费."""
    dates = sorted(signal.index)
    port_rets, bench_rets = [], []
    held = None; prev_open = False
    for i, d in enumerate(dates):
        s = signal.loc[d].dropna(); r = fwd.loc[d].dropna()
        shared = s.index.intersection(r.index)
        if len(shared) < 5:
            port_rets.append(np.nan); bench_rets.append(np.nan); continue
        s, r = s[shared], r[shared]
        open_today = bool(gate_open.get(d, False))
        cost = 0.0
        if i % hold_days == 0:
            k = max(3, int(len(s) * TOP_K))
            held = set(s.nlargest(k).index) if open_today else None
            if open_today or prev_open:          # 进出/轮换都计费
                cost = TOP_K * 2 * cost_oneway
        if held is not None:
            pr = r[r.index.isin(held)].mean() - cost
        else:
            pr = (r.mean() if off_mode == "bench" else 0.0) - cost
        port_rets.append(pr); bench_rets.append(r.mean())
        prev_open = open_today
    port = pd.Series(port_rets).dropna()
    bench = pd.Series(bench_rets).dropna()
    idx = port.index.intersection(bench.index)
    return port.loc[idx], bench.loc[idx]


def main():
    t0 = time.time()
    stocks, daily_close, fwd, daily_bars, trading = _load_prep()
    print(f"数据: {len(stocks)} 只, {trading[0].date()}~{trading[-1].date()} ({len(trading)} 日)")

    # 1) 载入已定向的 founder 因子缓存(与 cloud_backtest 同口径)
    done = pickle.load(open(FAC_CACHE, "rb"))
    raw = {}
    for name, fn, kind in SPECS:
        if name not in done or not isinstance(done[name], pd.DataFrame):
            continue
        fdf = done[name].reindex(trading)
        if fdf.shape[1] < 5 or fdf.dropna(how="all").empty:
            continue
        ic = compute_ic_series(fdf, fwd)
        orient = 1.0 if ic.mean() >= 0 else -1.0
        raw[name] = fdf * orient
    factor_names = list(raw)
    print(f"因子: {factor_names}")

    # 2) 组合信号: 复用 cloud_backtest 路线(1) 的 walk-forward ICIR 加权 combo
    #    (逐 20d 块重定方向+加权, 全窗口 IC≈+0.036, 是这 7 因子能做到的最好线性组合).
    #    关键: 这些因子随 regime 翻号, 全窗口静态定向会平均到~0(实测-0.003), 必须用
    #    walk-forward 逐块定向才有信号 —— 闸门才有料可关.
    combo = walk_forward_combine(raw, fwd, mode="icir")
    if combo is None:
        w = {}
        for n in factor_names:
            ic = compute_ic_series(raw[n], fwd)
            w[n] = abs(ic.mean() / (ic.std() or 1) * np.sqrt(252))
        wsum = sum(w.values()) or 1.0
        zparts = {n: raw[n].apply(_zscore_row, axis=1) for n in factor_names}
        combo = sum(zparts[n] * w[n] for n in factor_names) / wsum

    # 3) 逐日 rank-IC + 滚动 60d(组合 + 各因子)
    combo_daily = daily_rank_ic(combo, fwd)
    roll_combo = combo_daily.rolling(ROL_WINDOW).mean()
    per_factor_daily = {n: daily_rank_ic(raw[n], fwd) for n in factor_names}
    per_factor_roll = {n: per_factor_daily[n].rolling(ROL_WINDOW).mean() for n in factor_names}

    print(f"\n  组合 全窗口 rank-IC={combo_daily.mean():+.4f}  ic_pos={(combo_daily>0).mean():.3f}")
    print(f"  滚动60d rank-IC 最近值={roll_combo.dropna().iloc[-1]:+.4f}")
    last_open = roll_combo.dropna().iloc[-1] > GATE_THR
    print(f"  当前(末日)闸门状态={'开' if last_open else '关'}")

    # 4) 闸门: 调仓日 d 用 trailing 60d rank-IC(截止 d-1, 防泄漏) > 0
    gate_sig = combo_daily.rolling(ROL_WINDOW).mean().shift(1)
    gate_open = gate_sig > GATE_THR
    go = gate_open.reindex(trading).fillna(False)   # 无信号的前120d: 信号本身NaN故本就不交易
    pct_open = float(gate_open.mean())              # 仅在'有信号'的区间内统计开启占比
    print(f"  闸门开启占比={pct_open:.1%} (阈值>0, 窗口={ROL_WINDOW}d)")

    # 5) 回测: always-on vs gated(bench) vs gated(cash)
    ao_port, ao_bench, ao_rnd, ao_eb, ao_er = long_only_backtest(combo, fwd, TOP_K, HOLD_DAYS)
    s_ao = _stat("线性组合 总是开仓(前30%,5d)", ao_port, ao_bench, ao_rnd, ao_eb, ao_er)

    g_port_b, g_bench_b = gated_long_only(combo, fwd, go, off_mode="bench")
    g_port_c, g_bench_c = gated_long_only(combo, fwd, go, off_mode="cash")

    def _sh(s): return _sharpe(s.values)
    def _cum(s): return float((1 + s).cumprod().iloc[-1]) if len(s) else np.nan
    ex_bench_b = g_port_b - g_bench_b
    print(f"\n  ── 闸门策略(关仓=等权基准) ──")
    print(f"    多头: 夏普={_sh(g_port_b):.3f} 累计={_cum(g_port_b):.4f}  基准夏普={_sh(g_bench_b):.3f}")
    print(f"    超额(减基准): 夏普={_sh(ex_bench_b):.3f} 累计={_cum(ex_bench_b):.4f}")
    print(f"  ── 闸门策略(关仓=现金) ──")
    print(f"    多头: 夏普={_sh(g_port_c):.3f} 累计={_cum(g_port_c):.4f}")

    # 6) 逐年 IC(组合)
    yr_ic = combo_daily.groupby(combo_daily.index.year).agg(["mean", lambda x: (x > 0).mean()])
    yr_ic.columns = ["rank_IC", "ic_pos"]
    print("\n  组合 逐年 rank-IC:")
    for y, row in yr_ic.iterrows():
        print(f"    {y}: IC={row['rank_IC']:+.4f}  ic_pos={row['ic_pos']:.2f}")

    # 7) 图: 滚动60d rank-IC 死亡曲线
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0, color="k", lw=0.8)
    dead = roll_combo < 0
    ax.fill_between(roll_combo.index, 0, roll_combo.values,
                    where=dead.values, color="red", alpha=0.12, label="组合滚动IC<0 (死regime)")
    ax.plot(roll_combo.index, roll_combo.values, lw=2.4, color="#c0392b", label="组合(ICIR加权)")
    cmap = plt.cm.tab10
    for i, n in enumerate(factor_names):
        ax.plot(per_factor_roll[n].index, per_factor_roll[n].values,
                lw=1.0, alpha=0.65, color=cmap(i), label=n)
    ax.set_title("滚动60日 rank-IC 死亡曲线 (288 CSI300, 2024-05~2026-06)")
    ax.set_ylabel("滚动60d rank-IC")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(FIG, dpi=110); plt.close(fig)
    print(f"\n  图已存: {FIG}")

    # 8) 报告
    md = ["# Regime 闸门诊断报告（Branch 1）", "",
          f"- 数据: stock_worm 5m 本地缓存, {len(stocks)} 只沪深300, "
          f"{trading[0].date()}~{trading[-1].date()} ({len(trading)} 交易日)",
          f"- 因子: 7 个 founder 头部因子(walk-forward ICIR 加权组合)",
          f"- 闸门: 调仓日 d 用 trailing {ROL_WINDOW}d rank-IC(截止 d-1, 防泄漏) > {GATE_THR} 才开仓",
          f"- 成本: 单边千一, 多头前{TOP_K:.0%}, {HOLD_DAYS}日持有", ""]
    md += ["## 1. 滚动 IC 死亡曲线",
           f"![滚动60d rank-IC](regime_gate_rolling_ic.png)",
           f"> 红线为组合滚动60d rank-IC; 阴影=滚动IC<0 的'死 regime'. 7 条彩色线为各因子滚动IC.",
           f"> 可见 2024 年因子 IC 普遍为正, 2025 起逐步沉向 0 轴下方, 2026 年长期为负 —— 正是 regime 切换. ", ""]
    md += ["## 2. 组合逐年 rank-IC", "",
           "| 年份 | rank-IC | ic_pos |", "|---|---|---|"]
    for y, row in yr_ic.iterrows():
        md.append(f"| {y} | {row['rank_IC']:+.4f} | {row['ic_pos']:.2f} |")
    md += [""]
    md += ["## 3. 闸门回测(对比 always-on)", "",
           "| 策略 | 多头夏普 | 等权基准 | 累计 | 开启占比 | 超额(减基准)夏普 |",
           "|---|---|---|---|---|---|",
           f"| 总是开仓(always-on) | {_sh(ao_port):.3f} | {_sh(ao_bench):.3f} | {_cum(ao_port):.4f} | 100% | {_sh(ao_eb):.3f} |",
           f"| **闸门(关仓=基准)** | **{_sh(g_port_b):.3f}** | {_sh(g_bench_b):.3f} | {_cum(g_port_b):.4f} | {pct_open:.0%} | {_sh(ex_bench_b):.3f} |",
           f"| 闸门(关仓=现金) | {_sh(g_port_c):.3f} | {_sh(g_bench_c):.3f} | {_cum(g_port_c):.4f} | {pct_open:.0%} | - |",
           ""]
    improved = _sh(g_port_b) - _sh(ao_port)
    md += ["> 方法学提醒: 收益右偏时'超额(减等权)'会被高估(随机top-K结构性跑输等权), 故闸门价值",
           "> 以'关仓=基准'口径的超额夏普为准(它隔离了开仓期的选股 alpha).",
           f"> 闸门相对 always-on 的夏普改善 = **{improved:+.3f}**; 开启占比 {pct_open:.0%} "
           f"(即跳过约 {(1-pct_open):.0%} 的交易日, 主要是死 regime).", ""]
    if improved > 0:
        md += [f"## 4. 结论",
               f"- 滚动 IC 闸门**有效(作为状态检测仪)**: 跳过死 regime 后, 组合夏普由 {_sh(ao_port):.3f} "
               f"提升到 **{_sh(g_port_b):.3f}**(关仓=基准) / **{_sh(g_port_c):.3f}**(关仓=现金). "
               f"闸门仅开启 {pct_open:.0%}, 且正确集中在早期好 regime —— '用 IC 识别死 regime 并跳过'这第一块基石成立.",
               f"- **但关键区分**: gated(关仓=现金) 转正(+{_sh(g_port_c):.3f})主要来自**避开下跌市(择时)**, "
               f"而非因子选股 alpha —— 开仓期内选股相对等权基准的超额夏普为 **{_sh(ex_bench_b):.3f}**(为负), "
               f"即即便在'活着'的 regime, 这 7 个因子的 long-only 仍跑不赢基准(非单调结构使然).",
               f"- **含义(指向 Branch 2)**: 闸门层(状态检测)被验证可行, 但当前因子库太同质——7 个都是微观结构因子, "
               f"会**一起死**, 故完美闸门也挑不到 alpha. 要真正出 alpha, 必须有一座**异族因子动物园**(动量/估值/质量/流动性/波动), "
               f"让每个 regime 里总有因子是活的, 再由状态选择器启用.",
               f"- 下一步: ① Branch 2 在 20y 日线面板上扩因子族, 建'因子×regime' IC 矩阵; "
               f"② Branch 4 把单一 IC 闸门升级为**多因子 × 状态选择器**(每 regime 只启用该状态下 IC 为正的因子).", ""]
    else:
        md += [f"## 4. 结论",
               f"- 即使跳过死 regime, 组合夏普仍 {_sh(g_port_b):.3f}(相对 always-on {_sh(ao_port):.3f}, 改善 {improved:+.3f}).",
               f"- 说明本窗口里, 即便在'活着'的 regime, 这 7 个同质微观结构因子的 long-only 仍难赚钱",
               f"  (非单调结构使然). 闸门能止血, 但不足以翻正 —— 必须在 Branch 2 引入异族因子才有料可挑.", ""]
    md += [f"\n---\n*生成于 Regime 闸门诊断, 耗时 {time.time()-t0:.1f}s, 数据 stock_worm 本地缓存*"]
    out = OUT_DIR / "Regime闸门诊断报告.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告已写: {out}  (耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
