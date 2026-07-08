"""龙虎榜事件因子 — datacenter-web API（东财数据中心）。

信号逻辑：
  - 取最近 N 个交易日的龙虎榜数据
  - 对每只股票计算：净买入金额(归一化) × 上榜频率
  - 正数 = 机构/游资净买入，负数 = 净卖出
  - 未上榜股票 = 0.0（中性）

用途：当日截面选股信号，不适合历史回测（API 无历史数据）。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

__alpha_meta__ = {
    "id": "dragon_tiger",
    "nickname": "龙虎榜因子",
    "theme": ["sentiment"],
    "formula_latex": r"S = \frac{\sum \text{BILLBOARD\_NET\_AMT}}{\sum \text{BILLBOARD\_DEAL\_AMT}} \times \log(1+freq)",
    "columns_required": ["close"],
    "extras_required": [],
    "requires_sector": False,
    "universe": ["equity_cn"],
    "frequency": ["1D"],
    "decay_horizon": 5,
    "min_warmup_bars": 1,
    "notes": (
        "龙虎榜净买入占比因子。排名越高=机构游资越看好。"
        "数据源：datacenter-web.eastmoney.com。仅当日可用，不适于历史回测。"
    ),
}

logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 20  # 回溯 20 个交易日
_CACHE_TTL = 7200  # 2h 缓存


def _fetch_recent_dragon_tiger(
    lookback_days: int = _LOOKBACK_DAYS,
) -> dict[str, dict]:
    """取最近 N 个交易日的全市场龙虎榜，返回 {code: {net_amt, deal_amt, count, dates}}."""
    s = requests.Session()
    s.headers.update(
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0"}
    )

    stock_data: dict[str, dict] = {}
    base_url = "https://datacenter-web.eastmoney.com/api/data/v1/get"

    collected_days = 0
    max_offset = lookback_days + 10  # 周末/假期留 buffer

    for offset in range(max_offset):
        if collected_days >= lookback_days:
            break
        dt = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            r = s.get(
                base_url,
                params={
                    "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
                    "columns": "SECUCODE,SECURITY_NAME_ABBR,BILLBOARD_NET_AMT,BILLBOARD_DEAL_AMT,BILLBOARD_BUY_AMT,BILLBOARD_SELL_AMT,CHANGE_RATE,TRADE_DATE",
                    "filter": f"(TRADE_DATE='{dt}')",
                    "pageNumber": "1",
                    "pageSize": "200",
                    "sortColumns": "BILLBOARD_NET_AMT",
                    "sortTypes": "-1",
                    "source": "WEB",
                    "client": "WEB",
                },
                timeout=15,
            )
            data = r.json().get("result", {}).get("data", [])
            if data:
                collected_days += 1
                for row in data:
                    secucode = (row.get("SECUCODE", "") or "").split(".")[0]
                    if not secucode:
                        continue
                    if secucode not in stock_data:
                        stock_data[secucode] = {
                            "net_amt": 0.0,
                            "deal_amt": 0.0,
                            "count": 0,
                            "dates": [],
                        }
                    stock_data[secucode]["net_amt"] += float(
                        row.get("BILLBOARD_NET_AMT", 0) or 0
                    )
                    stock_data[secucode]["deal_amt"] += float(
                        row.get("BILLBOARD_DEAL_AMT", 0) or 0
                    )
                    stock_data[secucode]["count"] += 1
                    stock_data[secucode]["dates"].append(dt)
            time.sleep(0.15)
        except Exception:
            continue

    return stock_data


# ── Lazy cache ─────────────────────────────────────
_cache: dict = {"ts": 0, "data": {}}


def _get_cached() -> dict[str, dict]:
    now = time.time()
    if now - _cache["ts"] < _CACHE_TTL and _cache["data"]:
        return _cache["data"]
    _cache["data"] = _fetch_recent_dragon_tiger()
    _cache["ts"] = now
    return _cache["data"]


def compute(panel: dict) -> pd.DataFrame:
    """计算龙虎榜信号面板。返回与 close 同形的 DataFrame。

    对历史日期全部填 0.0（无历史龙虎榜数据）—— 此因子仅做当日截面。
    """
    close = panel["close"]
    codes = list(close.columns)

    raw = _get_cached()
    logger.info(
        "dragon_tiger fetched: %d stocks with recent board appearances",
        len(raw),
    )

    result = pd.DataFrame(
        np.zeros(close.shape),
        index=close.index,
        columns=close.columns,
        dtype=float,
    )

    for code in codes:
        if code not in raw or code not in result.columns:
            continue
        entry = raw[code]
        net = entry["net_amt"]
        deal = entry["deal_amt"]
        cnt = entry["count"]

        # 信号 = 净买入占比 × log(1+次数)
        ratio = net / deal if deal > 0 else 0.0
        freq_bonus = np.log1p(cnt)
        signal = np.clip(ratio * freq_bonus, -1, 1)

        result[code] = signal

    nonzero = (result.iloc[-1] != 0).sum()
    logger.info(
        "dragon_tiger computed: %d/%d stocks on board, range [%.3f, %.3f]",
        nonzero,
        len(codes),
        float(result.min().min()),
        float(result.max().max()),
    )
    return result
