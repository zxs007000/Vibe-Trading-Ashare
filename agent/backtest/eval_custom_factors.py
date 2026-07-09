"""评估「咱们后加的因子」(自研 structured_reversal + 方正金工18因子) 的夏普率。

与 zoo 筛查同一套指标口径：五分位多空夏普(ls_sharpe) + RankIC/ICIR。
- structured_reversal 用 1D 面板 (2022-2026)，并按"反转方向取反"正确呈现。
- 方正金工因子用 5m 真实数据 (2025-01-01~2026-06-30)，日因子→IC/夏普。
  (clouds_disperse / rapids_advance 依赖 amount 列，mootdx 5m 无该列→跳过)
"""
from __future__ import annotations
import sys, time, logging, warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # 屏蔽 equal_treatment 等空切片警告

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
CACHE = Path(__file__).parent / "screen_results" / "_5m_cache_2025h1.pkl"

from run_factor_analysis import build_panel, compute_forward_returns, CSI300_SAMPLE
from backtest.loaders.astockdata_loader import DataLoader
from src.factors.factor_analysis_core import compute_ic_series, compute_group_equity
from backtest.validation import _sharpe
from backtest.factors.structured_reversal import compute_batch as sr_batch, structured_reversal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_custom")
UNIVERSE = sorted({c for c in CSI300_SAMPLE})


def metrics(factor_df: pd.DataFrame, return_df: pd.DataFrame) -> dict:
    ic = compute_ic_series(factor_df, return_df)
    if ic.empty:
        return dict(ic_mean=np.nan, icir=np.nan, ic_pos=np.nan, ic_tstat=np.nan,
                    ls_sharpe=np.nan, top_sharpe=np.nan, ls_ann_ret=np.nan)
    ic_mean, ic_std = float(ic.mean()), float(ic.std())
    icir = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else np.nan
    eq = compute_group_equity(factor_df, return_df, n_groups=5)
    if eq.empty:
        return dict(ic_mean=round(ic_mean, 4), icir=round(icir, 3),
                    ic_pos=round(float((ic > 0).mean()), 3),
                    ic_tstat=round(ic_mean / (ic_std / np.sqrt(len(ic))), 2) if ic_std > 0 else np.nan,
                    ls_sharpe=np.nan, top_sharpe=np.nan, ls_ann_ret=np.nan)
    gr = eq.pct_change().dropna()
    ls = _sharpe((gr["Group_5"] - gr["Group_1"]).values) if len(gr) > 20 else np.nan
    top = _sharpe(gr["Group_5"].values) if len(gr) > 20 else np.nan
    ls_ann = float(np.mean((gr["Group_5"] - gr["Group_1"]).values) * 252) if len(gr) > 20 else np.nan
    return dict(ic_mean=round(ic_mean, 4), icir=round(icir, 3),
                ic_pos=round(float((ic > 0).mean()), 3),
                ic_tstat=round(ic_mean / (ic_std / np.sqrt(len(ic))), 2) if ic_std > 0 else np.nan,
                ls_sharpe=round(ls, 3) if ls == ls else np.nan,
                top_sharpe=round(top, 3) if top == top else np.nan,
                ls_ann_ret=round(ls_ann, 4) if ls_ann == ls_ann else np.nan)


# ============================================================
# Part A: 自研 structured_reversal (1D, 2022-2026)
# ============================================================
def part_a():
    raw = {k: v for k, v in DataLoader().fetch(UNIVERSE, "2022-01-01", "2026-06-30", interval="1D").items()
           if v is not None and not v.empty}
    logger.info("PartA 1D: %d 只", len(raw))
    panel = build_panel(raw)
    return_df = compute_forward_returns(panel)

    rows = []
    # v2: volume, 多窗口+截面zscore
    combined = sr_batch(raw, method="volume", windows=(10, 21, 63), do_zscore=True)
    fdf = pd.DataFrame({c: s for c, s in combined.items() if len(s.dropna())})
    m = metrics(fdf, return_df)
    m["reversal_ls"] = round(-m["ls_sharpe"], 3) if m["ls_sharpe"] == m["ls_sharpe"] else np.nan
    rows.append(("structured_reversal_v2", "自研/反转(volume,多窗口+zscore)", m))

    # v1: equal, 单窗口21, 无zscore
    v1 = {c: structured_reversal(b, method="equal", window=21) for c, b in raw.items() if len(structured_reversal(b, method="equal", window=21).dropna())}
    fdf1 = pd.DataFrame(v1)
    m1 = metrics(fdf1, return_df)
    m1["reversal_ls"] = round(-m1["ls_sharpe"], 3) if m1["ls_sharpe"] == m1["ls_sharpe"] else np.nan
    rows.append(("structured_reversal_v1", "自研/反转(equal,21d)", m1))

    print("\n" + "=" * 78)
    print("自研 structured_reversal 体检（同 zoo 口径：2022-2026 五分位多空夏普）")
    print("=" * 78)
    cols = ["ic_mean", "icir", "ic_pos", "ic_tstat", "ls_sharpe", "reversal_ls", "top_sharpe"]
    out = pd.DataFrame({r[0]: r[2] for r in rows}).T[cols]
    out.insert(0, "说明", [r[1] for r in rows])
    print(out.round(3).to_string())
    print("\n  ls_sharpe = 库约定(做多高因子值); reversal_ls = 反转方向(做多低值=取反)")
    print("  ⚠ 原始值=近期收益率(高=近期赢家)。2022-26 动量占优→反转方向为负；")
    print("    仅 walk-forward(方向自适应)用法为正(~0.15, 见 verify_oos_overfit).")
    return out


# ============================================================
# Part B: 方正金工 18 因子 (5m 真实数据, 2025-01-01~2026-06-30)
# ============================================================
def part_b():
    from backtest.factors import founder as F
    from backtest.factors import huatai as H
    from backtest.factors import guosheng as G
    from backtest.factors import haitong as HA
    start, end = "2025-01-01", "2026-06-30"
    SUB = UNIVERSE[:30]  # 缩小截面以控制单轮耗时(<8min)
    # 5m 取数（单笔可覆盖 1.5 年），带缓存避免重复限流
    if CACHE.exists():
        logger.info("加载 5m 缓存 %s", CACHE)
        all_min = pickle.load(open(CACHE, "rb"))
        stocks_minute = {c: all_min[c] for c in SUB if c in all_min}
    else:
        stocks_minute = {}
        t0 = time.time()
        for code in SUB:
            r = DataLoader().fetch([code], start, end, interval="5m")
            for k, v in r.items():
                if v is not None and not v.empty:
                    stocks_minute[k] = v
        logger.info("PartB 5m: %d 只, 耗时 %.1fs", len(stocks_minute), time.time() - t0)

    # 日频前向收益（由 5m resample 得到日 close）
    daily_close = pd.DataFrame({c: d["close"].resample("D").last() for c, d in stocks_minute.items()}).sort_index()
    daily_ret = daily_close.pct_change()
    fwd = daily_ret.shift(-1)
    # 市场收益代理(截面均值), 供 panic_factor 计算真实"惊恐度" S_t=|r_t-m_t|/(...)
    market_ret = daily_ret.mean(axis=1)
    # 日频 bars（供 coin_team/panic_factor/withered_tree 使用）
    daily_bars = {c: pd.DataFrame({"open": d["open"].resample("D").first(),
                                   "high": d["high"].resample("D").max(),
                                   "low": d["low"].resample("D").min(),
                                   "close": d["close"].resample("D").last(),
                                   "volume": d["volume"].resample("D").sum()})
                  for c, d in stocks_minute.items()}

    # 方正各因子（clouds_disperse / rapids_advance 自 stock_worm 返回 5m amount 后已可用）
    # kind: 'minute' = _batch(stocks_minute); 'daily' = _batch(daily_bars);
    #       'single_daily' = 逐股调用单股函数(daily); 'panic' = _batch(daily, minute, mkt);
    #       'daily_mkt' = _batch(daily_bars, market_ret)
    specs = [
        ("drip_water_stone", F.drip_water_stone_batch, "minute"),
        ("smart_money", F.smart_money_batch, "minute"),
        ("withered_tree_blooms", F.withered_tree_blooms, "single_daily"),
        ("moderate_risk", F.moderate_risk_batch, "minute"),
        ("coin_team", F.coin_team_batch, "daily"),
        ("complete_tide", F.complete_tide_batch, "minute"),
        ("scaling_heights", F.scaling_heights_batch, "minute"),
        ("moth_to_flame", F.moth_to_flame_batch, "minute"),
        ("flower_hidden", F.flower_hidden_batch, "minute"),
        ("wait_rescue", F.wait_rescue_batch, "minute"),
        ("equal_treatment", F.equal_treatment_batch, "minute"),
        ("bull_bear_game", F.bull_bear_game_batch, "minute"),
        ("panic_factor", F.panic_factor_batch, "panic"),
        ("synergy_effect", F.synergy_effect_batch, "minute"),
        ("undercurrent", F.undercurrent_batch, "minute"),
        # ── 依赖分钟级成交额(amount), stock_worm 第一优先源已补齐 ──
        ("clouds_disperse", F.clouds_disperse_batch, "minute"),
        ("rapids_advance", F.rapids_advance_batch, "minute"),
        # ── 华泰金工(新复现) ──
        ("HT:idiosyncratic_volatility", H.idiosyncratic_volatility_batch, "daily_mkt"),
        ("HT:downside_deviation", H.downside_deviation_batch, "daily"),
        ("HT:historical_percentile", H.historical_percentile_batch, "daily"),
        ("HT:money_flow", H.money_flow_batch, "minute"),
        # ── 国盛金工(量价淘金) / 海通金工(流动性) ──
        ("GS:overnight_return", G.overnight_return_batch, "daily"),
        ("GS:volume_price_divergence", G.volume_price_divergence_batch, "daily"),
        ("HA:amihud_illiquidity", HA.amihud_illiquidity_batch, "daily"),
    ]

    rows = []
    for name, fn, kind in specs:
        try:
            if kind == "minute":
                res = fn(stocks_minute)
            elif kind == "daily":
                res = fn(daily_bars)
            elif kind == "panic":
                res = fn(daily_bars, stocks_minute, market_ret)
            elif kind == "daily_mkt":
                res = fn(daily_bars, market_ret)
            else:  # single_daily
                res = {c: fn(b) for c, b in daily_bars.items()}
            fdf = pd.DataFrame({c: s for c, s in res.items() if len(s.dropna()) > 5})
            if fdf.empty or fdf.shape[1] < 5:
                rows.append((name, "空/无效", dict(ic_mean=np.nan, icir=np.nan, ic_pos=np.nan,
                                                    ic_tstat=np.nan, ls_sharpe=np.nan, top_sharpe=np.nan)))
                continue
            m = metrics(fdf, fwd)
            rows.append((name, "ok", m))
        except Exception as e:
            rows.append((name, f"ERR:{type(e).__name__}:{e}", dict(ic_mean=np.nan, icir=np.nan, ic_pos=np.nan,
                                                                ic_tstat=np.nan, ls_sharpe=np.nan, top_sharpe=np.nan)))

    print("\n" + "=" * 78)
    print("方正金工因子体检（5m 真实数据 2025-01-01~2026-06-30，同口径 IC/多空夏普）")
    print("=" * 78)
    cols = ["ic_mean", "icir", "ic_pos", "ic_tstat", "ls_sharpe", "top_sharpe"]
    out = pd.DataFrame({r[0]: r[2] for r in rows}).T[cols]
    out.insert(0, "状态", [r[1] for r in rows])
    is_ht = out.index.str.startswith("HT:")
    is_gh = out.index.str.startswith(("GS:", "HA:"))
    founder_rows = out[~is_ht & ~is_gh]
    huatai_rows = out[is_ht]
    gh_rows = out[is_gh]
    print("\n── 方正金工 ──")
    print(founder_rows.round(3).to_string())
    print("\n── 华泰金工 ──")
    print(huatai_rows.round(3).to_string())
    print("\n── 国盛/海通金工(新复现) ──")
    print(gh_rows.round(3).to_string())
    # 落盘(分开存, 便于报告生成器分别读取)
    founder_rows.round(4).to_csv(Path(__file__).parent / "screen_results" / "custom_founder_eval.csv")
    huatai_rows.round(4).to_csv(Path(__file__).parent / "screen_results" / "custom_huatai_eval.csv")
    gh_rows.round(4).to_csv(Path(__file__).parent / "screen_results" / "custom_guosheng_haitong_eval.csv")
    print("\n  已保存 custom_founder_eval.csv / custom_huatai_eval.csv / custom_guosheng_haitong_eval.csv")
    print("\n  注: smart_money/drip_water_stone 等在 _batch 内对日因子做 rolling(20) 平滑(慢信号);")
    print("  clouds_disperse/rapids_advance 依赖 amount 列(mootdx 5m 无)→未纳入;")
    print("  ls_sharpe 为库约定(做多高因子值)，若 IC<0 则盈利方向为取反(=|ls_sharpe| 量级).")
    return out


if __name__ == "__main__":
    part_a()
    part_b()
