"""
因子库批量回测与夏普筛查 (Factor Zoo Sharpe Screening)

目标：对用户因子库内「能回测」的因子（注册表 466 个量价因子中，仅需
open/high/low/close/volume 的）做一次统一回测，按 **多空组合夏普率** 排序筛选。

流程：
  1. 数据：mootdx 拉取宽基成分股日线 OHLCV（覆盖牛熊多段，用于稳健 IC）。
  2. 面板：build_panel → compute_forward_returns（次日前瞻收益，无前视）。
  3. 遍历注册表全部因子，依赖 amount/vwap/sector 的会被 SkipAlpha 自动跳过。
  4. 每因子计算：RankIC 序列 + ICIR + 多空组合（Top−Bottom 分位）每日收益 → Sharpe。
  5. 按 long_short_sharpe 降序输出 Top-N，全量结果落盘 CSV。

用法：
  python screen_factor_zoo.py                 # 全量
  python screen_factor_zoo.py --limit 10      # 仅前 10 个（冒烟）
  python screen_factor_zoo.py --top 50        # 输出 Top 50
  python screen_factor_zoo.py --start 2022-01-01 --end 2026-06-30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # agent/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from backtest.loaders.astockdata_loader import DataLoader as AStockDataLoader
from run_factor_analysis import build_panel, compute_forward_returns, CSI300_SAMPLE
from src.factors.registry import get_default_registry, SkipAlpha
from src.factors.factor_analysis_core import compute_ic_series, compute_group_equity
from backtest.validation import _sharpe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("screen")


# ---------------------------------------------------------------------------
# 股票宇宙（去重后的 CSI300 代表样本；如需更大截面可在此扩充）
# ---------------------------------------------------------------------------
UNIVERSE = sorted({c for c in CSI300_SAMPLE})


def load_panel(start: str, end: str) -> dict[str, pd.DataFrame] | None:
    loader = AStockDataLoader()
    if not loader.is_available():
        logger.error("mootdx 不可用")
        return None
    t0 = time.time()
    raw = loader.fetch(UNIVERSE, start, end, interval="1D")
    raw = {k: v for k, v in raw.items() if v is not None and not v.empty}
    logger.info("拉取 %d/%d 只股票, 耗时 %.1fs", len(raw), len(UNIVERSE), time.time() - t0)
    if not raw:
        return None
    panel = build_panel(raw)
    return panel


def long_short_metrics(factor_df: pd.DataFrame, return_df: pd.DataFrame) -> dict:
    """多空（Top−Bottom 五分位）每日收益 → 夏普；并附带 Top 组夏普。"""
    eq = compute_group_equity(factor_df, return_df, n_groups=5)
    if eq.empty or "Group_5" not in eq.columns or "Group_1" not in eq.columns:
        return {"ls_sharpe": np.nan, "top_sharpe": np.nan, "ls_ann_ret": np.nan}
    grp_ret = eq.pct_change().dropna()
    ls_ret = (grp_ret["Group_5"] - grp_ret["Group_1"]).values
    top_ret = grp_ret["Group_5"].values
    ls_sharpe = _sharpe(ls_ret) if len(ls_ret) > 20 else np.nan
    top_sharpe = _sharpe(top_ret) if len(top_ret) > 20 else np.nan
    ls_ann_ret = float(np.mean(ls_ret) * 252) if len(ls_ret) else np.nan
    return {"ls_sharpe": ls_sharpe, "top_sharpe": top_sharpe, "ls_ann_ret": ls_ann_ret}


def screen(panel, return_df, alpha_ids, top_n=30) -> pd.DataFrame:
    reg = get_default_registry()
    rows = []
    n = len(alpha_ids)
    for i, aid in enumerate(alpha_ids, 1):
        alpha = reg.get(aid)
        meta = alpha.meta
        rec = {
            "alpha_id": aid,
            "zoo": alpha.zoo,
            "theme": ",".join(meta.get("theme", [])),
        }
        try:
            fval = reg.compute(aid, panel)
        except SkipAlpha as e:
            rec.update({"status": "skip", "reason": str(e), "ls_sharpe": np.nan,
                        "ic_mean": np.nan, "icir": np.nan})
            rows.append(rec)
            continue
        except Exception as e:  # RegistryError / 计算异常
            rec.update({"status": "error", "reason": f"{type(e).__name__}: {e}",
                        "ls_sharpe": np.nan, "ic_mean": np.nan, "icir": np.nan})
            rows.append(rec)
            continue
        try:
            ic = compute_ic_series(fval, return_df)
            if ic.empty:
                rec.update({"status": "no_ic", "reason": "IC 序列为空", "ls_sharpe": np.nan,
                            "ic_mean": np.nan, "icir": np.nan})
            else:
                ic_mean = float(ic.mean())
                ic_std = float(ic.std())
                icir = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else np.nan
                pos = float((ic > 0).mean())
                tstat = ic_mean / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else np.nan
                m = long_short_metrics(fval, return_df)
                rec.update({
                    "status": "ok",
                    "ic_mean": round(ic_mean, 4),
                    "ic_std": round(ic_std, 4),
                    "icir": round(icir, 3) if icir == icir else np.nan,
                    "ic_pos_ratio": round(pos, 3),
                    "ic_tstat": round(tstat, 2) if tstat == tstat else np.nan,
                    "ls_sharpe": round(m["ls_sharpe"], 3) if m["ls_sharpe"] == m["ls_sharpe"] else np.nan,
                    "top_sharpe": round(m["top_sharpe"], 3) if m["top_sharpe"] == m["top_sharpe"] else np.nan,
                    "ls_ann_ret": round(m["ls_ann_ret"], 4) if m["ls_ann_ret"] == m["ls_ann_ret"] else np.nan,
                })
        except Exception as e:
            rec.update({"status": "error", "reason": f"metric: {type(e).__name__}: {e}",
                        "ls_sharpe": np.nan, "ic_mean": np.nan, "icir": np.nan})
        rows.append(rec)
        if i % 25 == 0:
            logger.info("进度 %d/%d  最近 ls_sharpe=%.2f (%s)", i, n,
                        rec.get("ls_sharpe", np.nan), aid)
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="仅前 N 个因子（冒烟测试）")
    ap.add_argument("--top", type=int, default=30, help="输出 Top-N")
    ap.add_argument("--start", type=str, default="2022-01-01")
    ap.add_argument("--end", type=str, default="2026-06-30")
    ap.add_argument("--min-icir", type=float, default=0.0, help="筛选 IR 门槛（输出用）")
    args = ap.parse_args()

    end_date = args.end
    start_date = args.start
    logger.info("区间 %s ~ %s, 股票 %d 只", start_date, end_date, len(UNIVERSE))

    panel = load_panel(start_date, end_date)
    if panel is None:
        sys.exit(1)
    return_df = compute_forward_returns(panel)
    logger.info("面板 close=%s  前瞻收益=%s", panel["close"].shape, return_df.shape)

    reg = get_default_registry()
    all_ids = reg.list()
    logger.info("注册表共 %d 因子", len(all_ids))
    ids = all_ids[: args.limit] if args.limit else all_ids

    t0 = time.time()
    df = screen(panel, return_df, ids, top_n=args.top)
    logger.info("筛查完成 %d 因子, 耗时 %.1fs", len(df), time.time() - t0)

    # 落盘
    out_dir = Path(__file__).parent / "screen_results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"zoo_screen_{stamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("全量结果 → %s", csv_path)

    ok = df[df["status"] == "ok"].copy()
    skip = df[df["status"].isin(["skip", "no_ic"])].copy()
    err = df[df["status"] == "error"].copy()
    logger.info("统计: 成功 %d / 跳过 %d / 报错 %d", len(ok), len(skip), len(err))

    if not ok.empty:
        ok_sorted = ok.sort_values("ls_sharpe", ascending=False, na_position="last")
        print("\n" + "=" * 78)
        print(f"TOP {args.top} 因子（按多空夏普率降序，ICIR>= {args.min_icir}）")
        print("=" * 78)
        show = ok_sorted[ok_sorted["icir"].fillna(0) >= args.min_icir].head(args.top)
        cols = ["alpha_id", "zoo", "theme", "ic_mean", "icir", "ic_pos_ratio",
                "ic_tstat", "ls_sharpe", "top_sharpe", "ls_ann_ret"]
        print(show[cols].to_string(index=False))

    if not err.empty:
        print("\n报错因子（前 15）：")
        print(err.head(15)[["alpha_id", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
