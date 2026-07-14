"""
OOS 框架 · 监管事件因子族 (源自 regulatory_events.parquet)

经济动机:
    监管事件（立案/处罚/问询函等）是最强的负面事件信号之一，常领先于业绩爆雷、
    ST、退市。事件稀少但信息量大，适合做成「事件衰减」型因子：事件日赋值，
    之后按指数衰减铺到日线，捕捉其持续压抑效应。

处理流程:
    1. 从数据湖读监管事件 (code, event_date, severity)
    2. 每只股票、每个事件：在 [event_date, event_date+window] 窗口内按半衰期衰减注入严重度
    3. 截面 z-score + 行业中性化（可选）
    这些因子预期为负向信号（监管压力越高，未来收益越低）。

因子清单 (3 个):
    f_reg_sev_60d    近 60 日衰减严重度（半衰期 30 日）— 短期监管压力
    f_reg_sev_250d   近 250 日衰减严重度（半衰期 120 日）— 中长期监管压力
    f_reg_major_250d 近 250 日是否发生严重事件(立案/处罚/谴责, severity>=3) — 二元风险旗
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from factor_zoo_daily import _cross_sectional_zscore, neutralize_factors

logger = logging.getLogger(__name__)

LAKE = Path("D:/work Buddy GZ/Claw/stockworm")
FUND = LAKE / "fundamentals"
REG_PATH = FUND / "regulatory_events.parquet"

# 让 OOS 框架复用 stock_worm 的数据湖读取层（尊重 STOCKWORM_LAKE 环境变量）
_STOCKWORM = Path("D:/stcok-worm")
if str(_STOCKWORM) not in sys.path:
    sys.path.insert(0, str(_STOCKWORM))
try:
    from stcok_worm import datalake as _dl
except Exception:
    _dl = None


def _load_events() -> pd.DataFrame | None:
    if _dl is not None:
        df = _dl.load_regulatory_events()
        if not df.empty:
            return df
        logger.warning("stock_worm.datalake.load_regulatory_events() 返回空")
    if REG_PATH.exists():
        df = pd.read_parquet(REG_PATH)
        if not df.empty:
            return df
    logger.warning("regulatory_events 不可用（%s 不存在且 datalake 返回空），跳过监管因子", REG_PATH)
    return None


def _trailing_decay(events: pd.DataFrame, dates: pd.DatetimeIndex,
                    codes: list[str], half_life: float, window: int) -> pd.DataFrame:
    """把事件按指数衰减铺成 日线×code 面板（事件后 window 日内按半衰期衰减注入严重度）。"""
    res = pd.DataFrame(0.0, index=dates, columns=codes)
    ci = {c: i for i, c in enumerate(codes)}
    ev = events.sort_values("event_date")
    end = dates[-1]
    span = pd.Timedelta(days=window)
    for _, r in ev.iterrows():
        ed = r["event_date"]
        if ed > end:
            continue
        c = r["code"]
        j = ci.get(c)
        if j is None:
            continue
        sev = float(r["severity"])
        lo = dates.searchsorted(ed, side="left")
        hi = dates.searchsorted(ed + span, side="right")
        if lo >= hi:
            continue
        dts = np.array([(dates[k] - ed).days for k in range(lo, hi)], dtype=float)
        w = np.exp(-dts / half_life)
        res.iloc[lo:hi, j] += sev * w
    return res


def _trailing_flag(events: pd.DataFrame, dates: pd.DatetimeIndex,
                   codes: list[str], window: int, min_sev: int) -> pd.DataFrame:
    """近 window 日内是否发生 severity>=min_sev 的事件（二元旗）。"""
    res = pd.DataFrame(0.0, index=dates, columns=codes)
    ci = {c: i for i, c in enumerate(codes)}
    ev = events[events["severity"] >= min_sev].sort_values("event_date")
    end = dates[-1]
    span = pd.Timedelta(days=window)
    for _, r in ev.iterrows():
        ed = r["event_date"]
        if ed > end:
            continue
        j = ci.get(r["code"])
        if j is None:
            continue
        lo = dates.searchsorted(ed, side="left")
        hi = dates.searchsorted(ed + span, side="right")
        if lo < hi:
            res.iloc[lo:hi, j] = 1.0
    return res


def build_regulatory_factors(panel: dict[str, pd.DataFrame],
                             industry_map: dict[str, str] = None) -> dict[str, pd.DataFrame]:
    """从监管事件构建 3 个因子。返回 {factor_name: DataFrame(日期×code)}。"""
    if "close" not in panel:
        raise ValueError("panel 必须含 'close'")
    close = panel["close"]
    dates = close.index
    codes = list(close.columns)

    df = _load_events()
    if df is None:
        return {}

    out = {}
    specs = [
        ("f_reg_sev_60d",   dict(kind="decay", half_life=30,  window=60)),
        ("f_reg_sev_250d",  dict(kind="decay", half_life=120, window=250)),
        ("f_reg_major_250d", dict(kind="flag",  window=250, min_sev=3)),
    ]
    for fname, spec in specs:
        if spec["kind"] == "decay":
            daily = _trailing_decay(df, dates, codes, spec["half_life"], spec["window"])
        else:
            daily = _trailing_flag(df, dates, codes, spec["window"], spec["min_sev"])
        if daily.isna().all().all() or (daily == 0).all().all():
            continue
        z = _cross_sectional_zscore(daily)
        if industry_map:
            z = neutralize_factors(z, industry_map)
        out[fname] = z

    logger.info("监管因子构建完成: %d 个", len(out))
    return out


def list_regulatory_factors() -> list[str]:
    return ["f_reg_sev_60d", "f_reg_sev_250d", "f_reg_major_250d"]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from panel_builder import load_panel, load_codes_from_lake
    codes = load_codes_from_lake(min_bars=250)[:50]
    panel = load_panel(codes, start_date="2018-01-01", end_date="2026-06-30")
    fac = build_regulatory_factors(panel)
    for name, f in fac.items():
        print(f"{name}: shape={f.shape} nonnull={int((f != 0).sum())}")
