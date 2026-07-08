"""基本面财务数据加载器。

设计原则（遵用户偏好"优先用 stock_worm 单一源"）：
    1. 主源：stock_worm.fundamentals.quarterly_history（东财，多期季报，含
       WEIGHTAVG_ROE/BPS/MGJYXJJE/BASIC_EPS 等，可构建时间序列因子）
    2. 回退：akshare（stock_financial_analysis_indicator + stock_financial_abstract），
       当 stock_worm 不可达/被限流时使用
    3. 落盘缓存：~/.vibe-trading/financial_cache/{code}.parquet

对外统一产出 DataFrame，列：
    report_date, roe, net_margin, debt_to_assets, eps, op_cash_flow_ps,
    bvps, cash_content, accruals, cash_ratio_proxy

其中 roe / bvps / accruals 被 zoo 财务因子直接消费；其余为兼容/扩展列。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

CACHE_DIR = os.path.expanduser("~/.vibe-trading/financial_cache")


# --------------------------------------------------------------------------- #
# 主源：stock_worm（优先）
# --------------------------------------------------------------------------- #
def _stcok_worm_available() -> bool:
    try:
        import stcok_worm  # noqa: F401
        return True
    except Exception:
        return False


def _fetch_sw(codes: List[str], periods: int = 12) -> Dict[str, pd.DataFrame]:
    """经由 stock_worm 取多期财报，构造统一 DataFrame。"""
    import stcok_worm
    from stcok_worm import fundamentals as sw_fund

    out: Dict[str, pd.DataFrame] = {}
    for code in codes:
        try:
            rows = sw_fund.quarterly_history(code, periods=periods)
        except Exception:
            rows = []
        if not rows:
            continue
        recs = []
        for r in rows:
            rd = str(r.get("REPORTDATE", "")).split(" ")[0]
            if not rd:
                continue
            roe = _num(r.get("WEIGHTAVG_ROE"))
            eps = _num(r.get("BASIC_EPS"))
            bvps = _num(r.get("BPS"))
            ocf_ps = _num(r.get("MGJYXJJE"))  # 每股经营现金净金额
            income = _num(r.get("TOTAL_OPERATE_INCOME"))
            netprofit = _num(r.get("PARENT_NETPROFIT"))
            # 净利率 = 归母净利 / 营业总收入（stock_worm 季报无直接净利率字段）
            net_margin = (netprofit / income) if (income and income != 0) else None
            # 现金含量 = 每股经营现金流 / EPS
            cash_content = (ocf_ps / eps) if (eps and eps != 0) else None
            # 应计利润代理 = (EPS - 每股OCF) / 每股净资产 ≈ (NI-OCF)/权益
            accruals = ((eps - ocf_ps) / bvps) if (bvps and bvps != 0) else None
            recs.append({
                "report_date": rd,
                "roe": roe,
                "net_margin": net_margin,
                "debt_to_assets": None,  # stock_worm 季报快照无直接负债率字段
                "eps": eps,
                "op_cash_flow_ps": ocf_ps,
                "bvps": bvps,
                "cash_content": cash_content,
                "accruals": accruals,
                "cash_ratio_proxy": None,
            })
        if recs:
            df = pd.DataFrame(recs).drop_duplicates("report_date").sort_values("report_date")
            out[code] = df
    return out


# --------------------------------------------------------------------------- #
# 回退源：akshare（无 token）
# --------------------------------------------------------------------------- #
def _akshare_available() -> bool:
    try:
        import akshare  # noqa: F401
        return True
    except Exception:
        return False


def _fetch_akshare(codes: List[str]) -> Dict[str, pd.DataFrame]:
    """原 akshare 路径（东财无 token），作为 stock_worm 不可达时的回退。"""
    import akshare as ak

    out: Dict[str, pd.DataFrame] = {}
    for code in codes:
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code.split(".")[0], start_year="2019")
            if df is None or df.empty:
                continue
            df = df.copy()
            df["report_date"] = pd.to_datetime(df["报告期"], errors="coerce")
            df = df.dropna(subset=["report_date"]).sort_values("report_date")
            keep = {
                "净资产收益率(%)": "roe",
                "销售净利率(%)": "net_margin",
                "资产负债率(%)": "debt_to_assets",
                "基本每股收益": "eps",
                "每股经营现金流量净额": "op_cash_flow_ps",
                "每股净资产": "bvps",
            }
            sub = pd.DataFrame(index=df.index)
            for src, dst in keep.items():
                sub[dst] = pd.to_numeric(df.get(src), errors="coerce")
            sub["report_date"] = df["report_date"].dt.strftime("%Y-%m-%d")
            # 现金含量 / 应计利润 / 现金率代理
            eps = sub["eps"]; ocf = sub["op_cash_flow_ps"]; bvps = sub["bvps"]
            sub["cash_content"] = (ocf / eps.replace(0, pd.NA)).replace([pd.NA, float("inf"), -float("inf")], None)
            sub["accruals"] = ((eps - ocf) / bvps.replace(0, pd.NA)).replace([pd.NA, float("inf"), -float("inf")], None)
            sub["cash_ratio_proxy"] = (ocf / bvps.replace(0, pd.NA)).replace([pd.NA, float("inf"), -float("inf")], None)
            out[code] = sub[["report_date", "roe", "net_margin", "debt_to_assets", "eps",
                              "op_cash_flow_ps", "bvps", "cash_content", "accruals", "cash_ratio_proxy"]]
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# 公共接口
# --------------------------------------------------------------------------- #
def fetch_fundamentals(
    codes: List[str],
    *,
    use_cache: bool = True,
    prefer: str = "stock_worm",
    periods: int = 12,
) -> Dict[str, pd.DataFrame]:
    """取一组股票的基本面财报。

    Args:
        codes: 股票代码列表（带或不带交易所后缀均可）。
        use_cache: 是否读写本地缓存（按 code 缓存最新结果）。
        prefer: 主源，"stock_worm" 或 "akshare"。
        periods: 主源拉取的季报期数。

    Returns:
        {code: DataFrame(report_date, roe, net_margin, debt_to_assets, eps,
                         op_cash_flow_ps, bvps, cash_content, accruals, cash_ratio_proxy)}
        仅包含成功取到数据的 code。
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    result: Dict[str, pd.DataFrame] = {}

    for code in codes:
        ckey = code.split(".")[0]
        cache_path = os.path.join(CACHE_DIR, f"{ckey}.parquet")
        if use_cache and os.path.exists(cache_path):
            try:
                result[code] = pd.read_parquet(cache_path)
                continue
            except Exception:
                pass

        # 主源
        primary = _fetch_sw([code], periods=periods) if (
            prefer == "stock_worm" and _stcok_worm_available()
        ) else _fetch_akshare([code])
        if code in primary and not primary[code].empty:
            result[code] = primary[code]
        else:
            # 回退到另一个源
            fallback = _fetch_akshare([code]) if prefer == "stock_worm" else _fetch_sw([code], periods=periods)
            if code in fallback and not fallback[code].empty:
                result[code] = fallback[code]

        if code in result and use_cache:
            try:
                result[code].to_parquet(cache_path, index=False)
            except Exception:
                pass

    return result


def latest_as_of(df: pd.DataFrame, as_of: str) -> Dict[str, Optional[float]]:
    """取报告期不晚于 as_of 的最近一条记录，拼成字典。"""
    if df is None or df.empty:
        return {}
    d = df[df["report_date"] <= as_of]
    if d.empty:
        d = df  # 数据都比 as_of 新，退而取最新
    row = d.iloc[-1]
    out = {}
    for col in ["roe", "net_margin", "debt_to_assets", "cash_ratio", "accruals"]:
        if col in row and pd.notna(row[col]):
            out[col] = float(row[col])
    return out


def _num(x):
    try:
        if x is None or x == "" or x == "--":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None
