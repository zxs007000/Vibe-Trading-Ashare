"""情绪因子：新闻情感分歧度（标准差）。

逻辑：对每只股票取多条新闻标题，计算情感得分的标准差。
高分歧度 = 市场对该股票的看法严重分裂，通常伴随：
  - 即将公布的重大消息（财报/重组/处罚）
  - 多空激烈博弈
  - 潜在的趋势转折点

学术依据：Diether, Malloy & Scherbina (2002) — 分析师分歧度与未来收益负相关。
此处用新闻情感分歧度代理同样的经济直觉。

数据源：stock_worm.news.stock_news() + DictionaryAnalyzer。
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
    "id": "sentiment_divergence",
    "nickname": "情绪分歧度因子",
    "theme": ["sentiment"],
    "formula_latex": r"D_t = \mathrm{std}\left[\mathrm{sentiment}(\mathrm{news}_i)\right]",
    "columns_required": ["close"],
    "extras_required": [],
    "requires_sector": False,
    "universe": ["equity_cn"],
    "frequency": ["1D"],
    "decay_horizon": 3,
    "min_warmup_bars": 1,
    "notes": (
        "新闻情感得分的截面标准差。高分歧=多空激烈=潜在转折。"
        "学术依据: 分析师分歧度与未来收益负相关。"
    ),
}

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".vibe-trading" / "sentiment_cache"
_CACHE_TTL = 86400


def _cache_key(code: str) -> str:
    return hashlib.md5(f"div_{code}".encode()).hexdigest()[:12]


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


def _get_sentiment_divergence(code: str) -> float:
    """获取新闻情感分歧度（标准差）。

    返回 [0, 2]：0=完全一致，>1=高度分歧。
    """
    cached = _load_cache(code)
    if cached is not None:
        return float(cached.get("divergence", 0.0))

    try:
        from stcok_worm import news as sw_news
        articles = sw_news.stock_news(code, page_size=30)
    except Exception:
        logger.debug("stock_worm news fetch failed for div %s", code, exc_info=True)
        articles = []

    if len(articles) < 3:
        _save_cache(code, {"divergence": 0.0, "count": len(articles)})
        return 0.0

    try:
        from stcok_worm.sentiment.analyzers.dictionary import DictionaryAnalyzer
        analyzer = DictionaryAnalyzer()
        scores = []
        for a in articles:
            text = a.get("title", "") or ""
            result = analyzer.analyze(text)
            scores.append(result.get("sentiment", 0.0))

        div = float(np.std(scores)) if len(scores) >= 2 else 0.0
    except Exception:
        logger.debug("sentiment divergence calc failed for %s", code, exc_info=True)
        div = 0.0

    _save_cache(code, {"divergence": div, "count": len(articles)})
    return div


def compute(panel: dict) -> pd.DataFrame:
    close = panel["close"]
    codes = list(close.columns)

    div_values: dict[str, float] = {}
    for code in codes:
        div_values[code] = _get_sentiment_divergence(code)

    result = pd.DataFrame(
        np.zeros(close.shape),
        index=close.index,
        columns=close.columns,
        dtype=float,
    )

    for code, div in div_values.items():
        if code in result.columns:
            result[code] = div

    nonzero = sum(1 for v in div_values.values() if v > 0)
    logger.info(
        "sentiment_divergence computed: %d/%d stocks had multi-article data, "
        "max_div=%.3f",
        nonzero, len(codes),
        float(result.max().max() or 0),
    )
    return result
