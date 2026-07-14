"""
OOS 框架 · 经典组合型基本面因子族 (factor_zoo_fundamental 的增强补充)

动机:
    现有 factor_zoo_fundamental.py 的 25 个因子全部是"单指标 z-score"，在 400 只
    真实个股池上 OOS IC 均值仅 ~0.007，没有一个 >0.02，缺少学术界/业界公认的
    "组合/衍生型"强因子。本模块补上这批经过 A股实证的经典因子，它们与现有单指标
    低相关，是理想的分散化 alpha 来源。

因子清单 (7 个组合因子):
    f_gpa        毛利资产比 GP/A (Novy-Marx 2013)  = 毛利率 × 资产周转率
                 "The Other Side of Value: The Gross Profitability Premium"
                 学术界公认最强质量因子之一，与价值因子低/负相关，组合后 Sharpe 大增。
    f_accruals   应计利润质量 (Sloan 1996, 负向修正为正)
                 = 经营现金流率 − 净利率 (同为 %/营收)。高值=盈利含金量高(现金>利润)。
                 A股会计操纵更普遍，应计异象显著；与价值/动量低相关。
    f_cfoa       现金流资产比 CF/A = 经营现金流率 × 资产周转率
                 Piotroski / 质量体系核心，高 CF/A = 真实造血能力强。
    f_fscore     Piotroski F-Score (0~8 分，缺"未增发"项)
                 A股全市场 high-low 组合年化超额 ~7.6%，IR ~1.07 (海通/多家研报实证)。
                 8 项二元财务健康信号求和：盈利 4 + 杠杆流动性 2 + 效率 2。
    f_garp       成长调整估值 GARP (PEG 的正向形式) = 盈利收益率 × 净利同比
                 低 PEG = 好，等价于 (eps/close)×growth 高 = 好 (仅 growth>0)。
    f_htsec_f    海通"历史财务综合因子"Factor_F 变体
                 8 指标横截面 z 等权加总：ROE / ΔROE / ΔCFO / EPS / -ΔLEV /
                 流动比率 / Δ毛利率 / 周转率。研报口径的 FSCORE 连续化版本。
    f_earnings_q 盈余质量 = ROE × 应计质量 (盈利能力 × 现金含量) 的交互增强

数据来源: 与 factor_zoo_fundamental 共用 fin_indicators.parquet。
处理: 报告期级算原始值 → point-in-time ffill 到日频 → winsorize → 截面 z → 行业中性化。
"""

import logging
import numpy as np
import pandas as pd

from factor_zoo_daily import _cross_sectional_zscore, neutralize_factors
from factor_zoo_fundamental import _load_fin, _ffill_to_dates

logger = logging.getLogger(__name__)


def _winsorize(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    """按分位数截尾，压制财报里的 inf / 极端异常值。"""
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 20:
        return s
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(ql, qh)


def _build_report_level(df: pd.DataFrame) -> pd.DataFrame:
    """在报告期级 (code,date) 计算所有组合因子的原始列。"""
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    df = df.sort_values(["code", "date"])
    g = df.groupby("code", group_keys=False)

    # 相邻报告期变化 (F-Score 方向项用)
    df["d_roa"] = g["roa"].diff()
    df["d_turn"] = g["asset_turnover"].diff()
    df["d_curr"] = g["current_ratio"].diff()
    df["d_lev"] = g["debt_to_asset"].diff()
    df["d_gm"] = g["gross_margin"].diff()

    # ---- 连续型组合因子 ----
    df["x_gpa"] = df["gross_margin"] * df["asset_turnover"]          # GP/A
    df["x_accruals"] = df["ocf_to_revenue"] - df["net_margin"]       # 低应计=高分
    df["x_cfoa"] = df["ocf_to_revenue"] * df["asset_turnover"]       # CF/A
    df["x_earnings_q"] = df["roe"] * (df["ocf_to_revenue"] - df["net_margin"])  # 盈余质量交互

    # ---- Piotroski F-Score (8 项，缺"未增发普通股") ----
    f1 = (df["roa"] > 0).astype(float)                              # ROA>0
    f2 = (df["ocf_to_revenue"] > 0).astype(float)                   # OCF>0
    f3 = (df["d_roa"] > 0).astype(float)                            # ΔROA>0
    f4 = (df["ocf_to_revenue"] > df["net_margin"]).astype(float)    # 应计: OCF% > 净利率%
    f5 = (df["d_lev"] < 0).astype(float)                            # 杠杆下降
    f6 = (df["d_curr"] > 0).astype(float)                           # 流动比率上升
    f8 = (df["d_gm"] > 0).astype(float)                             # 毛利率上升
    f9 = (df["d_turn"] > 0).astype(float)                           # 资产周转率上升
    fscore = f1 + f2 + f3 + f4 + f5 + f6 + f8 + f9
    valid = df["roa"].notna() & df["ocf_to_revenue"].notna()
    fscore[~valid] = np.nan
    df["x_fscore"] = fscore

    return df


# 报告期级连续因子 (原始列 -> 因子名)
_REPORT_FACTORS = {
    "x_gpa": "f_gpa",
    "x_accruals": "f_accruals",
    "x_cfoa": "f_cfoa",
    "x_fscore": "f_fscore",
    "x_earnings_q": "f_earnings_q",
}

# 海通综合因子分量: 原始列 -> (因子内部名, 方向 +1/-1)
_HTSEC_PARTS = {
    "roe": ("roe", +1),
    "roe_yoy": ("d_roe", +1),
    "ocf_ps_yoy": ("d_cfo", +1),
    "eps": ("eps", +1),
    "debt_to_asset_yoy": ("d_lev", -1),   # 杠杆上升为负
    "current_ratio": ("liquid", +1),
    "gross_margin_yoy": ("d_margin", +1),
    "asset_turnover": ("turn", +1),       # 周转率水平 (缺 yoy 用水平代理)
}


def build_fundamental_plus_factors(
    panel: dict[str, pd.DataFrame],
    industry_map: dict[str, str] = None,
) -> dict[str, pd.DataFrame]:
    """构建 7 个经典组合型基本面因子。

    Returns:
        {factor_name: DataFrame(index=date, columns=code)}  已 z-score (+可选中性化)
    """
    if "close" not in panel:
        raise ValueError("panel 必须含 'close'")
    close = panel["close"]
    dates = close.index
    codes = list(close.columns)

    df = _load_fin()
    if df is None:
        return {}

    rep = _build_report_level(df)
    out = {}

    def _finalize(daily: pd.DataFrame, fname: str):
        if daily is None or daily.isna().all().all():
            return
        daily = daily.replace([np.inf, -np.inf], np.nan)
        z = _cross_sectional_zscore(daily)
        if industry_map:
            z = neutralize_factors(z, industry_map)
        out[fname] = z

    # ---- 报告期级连续因子 ----
    for col, fname in _REPORT_FACTORS.items():
        sub = rep[["code", "date", col]].copy()
        # 报告期级 winsorize (对每个报告期截面压极端)
        sub[col] = sub.groupby("date")[col].transform(_winsorize)
        daily = _ffill_to_dates(sub, col, dates, codes)
        _finalize(daily, fname)

    # ---- GARP (需收盘价): 盈利收益率 × 净利同比 (仅 growth>0) ----
    if "eps" in df.columns and "net_profit_yoy" in df.columns:
        eps_daily = _ffill_to_dates(df[["code", "date", "eps"]], "eps", dates, codes)
        g_daily = _ffill_to_dates(df[["code", "date", "net_profit_yoy"]],
                                  "net_profit_yoy", dates, codes)
        with np.errstate(divide="ignore", invalid="ignore"):
            ep = eps_daily / close.replace(0, np.nan)          # 盈利收益率
        g_pos = g_daily.where(g_daily > 0, np.nan)             # 仅正增长
        garp = ep * g_pos
        _finalize(garp, "f_garp")

    # ---- 海通综合 Factor_F: 分量各自 z 后等权加总 ----
    zparts = []
    for col, (nm, sign) in _HTSEC_PARTS.items():
        if col not in df.columns:
            continue
        sub = df[["code", "date", col]].copy()
        sub[col] = sub.groupby("date")[col].transform(_winsorize)
        daily = _ffill_to_dates(sub, col, dates, codes)
        if daily.isna().all().all():
            continue
        z = _cross_sectional_zscore(daily) * sign
        zparts.append(z)
    if zparts:
        stacked = pd.concat(zparts, axis=0, keys=range(len(zparts)))
        comp = stacked.groupby(level=1).mean()
        comp = comp.reindex(index=dates, columns=codes)
        if industry_map:
            comp = neutralize_factors(comp, industry_map)
        out["f_htsec_f"] = comp

    logger.info("经典组合基本面因子构建完成: %d 个", len(out))
    return out


def list_fundamental_plus_factors() -> list[str]:
    return list(_REPORT_FACTORS.values()) + ["f_garp", "f_htsec_f"]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from panel_builder import load_panel, load_codes_from_lake

    codes = load_codes_from_lake(min_bars=400)[:100]
    panel = load_panel(codes, start_date="2020-01-01", end_date="2026-07-14")
    fac = build_fundamental_plus_factors(panel)
    for name, f in fac.items():
        print(f"{name}: shape={f.shape} nonnull={int(f.notna().sum().sum())}")
