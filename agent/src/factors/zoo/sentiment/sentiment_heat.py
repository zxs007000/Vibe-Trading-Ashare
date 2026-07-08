"""情绪因子：讨论热度（新闻量 + 互动量）。

逻辑：高讨论量 = 市场关注度异常，可能伴随波动放大。
不做情感方向判断，纯度量讨论热度。

数据源：stcok_worm.news.stock_news()，统计近期新闻条数和来源数。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

__alpha_meta__ = {
    "id": "sentiment_heat",
    "nickname": "讨论热度因子",
    "theme": ["sentiment"],
    "formula_latex": r"H_t = \log(1 + N_{\mathrm{news}})",
    "columns_required": ["close"],
    "extras_required": [],
    "requires_sector": False,
    "universe": ["equity_cn"],
    "frequency": ["1D"],
    "decay_horizon": 3,
    "min_warmup_bars": 1,
    "notes": (
        "基于 stock_worm 新闻条数 + 来源数的讨论热度度量。"
        "log(1+N) 压缩极端值。适合截面识别热点股，历史回测有前视偏差。"
    ),
}

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".vibe-trading" / "sentiment_cache"
_CACHE_TTL = 86400


def _cache_key(code: str) -> str:
    return hashlib.md5(f"heat_{code}".encode()).hexdigest()[:12]


def _load_cache(code: str) -> dict | None:
    p = _CACHE_DIR / f"{_cache_key(code)}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if time.time() - data.get("ts", 0) > _CACHE_TTL:
            return None
        return data
    except Exception:
        return None


def _save_cache(code: str, data: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data["ts"] = time.time()
    p = _CACHE_DIR / f"{_cache_key(code)}.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _get_heat(code: str) -> float:
    cached = _load_cache(code)
    if cached is not None:
        return float(cached.get("heat", 0.0))

    try:
        from stcok_worm import news as sw_news

        articles = sw_news.stock_news(code, page_size=30)
    except Exception:
        logger.debug("stock_worm news fetch failed for heat %s", code, exc_info=True)
        articles = []

    if not articles:
        _save_cache(code, {"heat": 0.0, "count": 0, "sources": 0})
        return 0.0

    n_articles = len(articles)
    sources = len(set(a.get("source", "") for a in articles if a.get("source")))

    # 热度 = log(1 + 条数) * (1 + 来源多样性 bonus)
    diversity_bonus = min(sources / 5.0, 1.0)
    heat = np.log1p(n_articles) * (1.0 + diversity_bonus * 0.5)

    _save_cache(code, {"heat": heat, "count": n_articles, "sources": sources})
    return heat


def compute(panel: dict) -> pd.DataFrame:
    close = panel["close"]
    codes = list(close.columns)

    heat_values: dict[str, float] = {}
    for code in codes:
        heat_values[code] = _get_heat(code)

    result = pd.DataFrame(
        np.zeros(close.shape),
        index=close.index,
        columns=close.columns,
        dtype=float,
    )

    for code, h in heat_values.items():
        if code in result.columns:
            result[code] = h

    logger.info(
        "sentiment_heat computed: %d stocks, max_heat=%.2f",
        len(codes),
        float(result.max().max() or 0),
    )
    return result
