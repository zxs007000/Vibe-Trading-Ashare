"""情绪因子：新闻情感得分（stock_worm 新闻 + 词典分析器）。

数据源：stcok_worm.news.stock_news() + DictionaryAnalyzer（无外部依赖）。
不需要 LLM key，不需要爬虫，纯文本词典匹配。

策略瓶颈：原系统 sentiment 维度为 0 个因子，本因子建立第 4 个 alpha 维度。

注意事项：
  - 情绪数据来自近期新闻（不是历史时序），存在前视偏差风险。
  - 适合用于当日截面信号，不适用于历史回测（回测时全期填入同一值）。
  - 缓存到 ~/.vibe-trading/sentiment_cache/，24h 内不重复请求。

compute 约定：返回与 close 同形的 DataFrame，信号值范围 [-1, 1]。
正数 = 市场情绪偏多，负数 = 偏空。
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
    "id": "sentiment_score",
    "nickname": "新闻情感因子",
    "theme": ["sentiment"],
    "formula_latex": r"S_t = \frac{1}{N}\sum_{i=1}^{N}\mathrm{sentiment}(\mathrm{news}_i)",
    "columns_required": ["close"],
    "extras_required": [],
    "requires_sector": False,
    "universe": ["equity_cn"],
    "frequency": ["1D"],
    "decay_horizon": 5,
    "min_warmup_bars": 1,
    "notes": (
        "基于 stock_worm 新闻 + 词典情感分析（正面词/负面词/否定词匹配）。"
        "信号范围 [-1, 1]，每天新闻情感均值。适合截面选股，历史回测有前视偏差。"
    ),
}

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".vibe-trading" / "sentiment_cache"
_CACHE_TTL = 86400  # 24h 秒


def _cache_key(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()[:12]


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


def _get_sentiment(code: str) -> float:
    """获取单只股票的情绪分数。

    优先走缓存，缓存未命中时调 stock_worm 取新闻 + 词典分析。
    返回值范围 [-1, 1]，0 表示中性/无数据。
    """
    cached = _load_cache(code)
    if cached is not None:
        return float(cached.get("sentiment", 0.0))

    try:
        # 延迟导入，避免阻塞 Registry 扫描
        from stcok_worm import news as sw_news

        articles = sw_news.stock_news(code, page_size=20)
    except Exception:
        logger.debug("stock_worm news fetch failed for %s", code, exc_info=True)
        articles = []

    if not articles:
        _save_cache(code, {"sentiment": 0.0, "count": 0, "source": "none"})
        return 0.0

    try:
        from stcok_worm.sentiment.analyzers.dictionary import DictionaryAnalyzer

        analyzer = DictionaryAnalyzer()
        scores = []
        for a in articles:
            text = a.get("title", "") or ""
            result = analyzer.analyze(text)
            scores.append(result.get("sentiment", 0.0))

        raw = float(np.mean(scores)) if scores else 0.0
    except Exception:
        logger.debug("DictionaryAnalyzer failed for %s", code, exc_info=True)
        raw = 0.0

    _save_cache(code, {"sentiment": raw, "count": len(articles), "source": "dictionary"})
    return raw


def compute(panel: dict) -> pd.DataFrame:
    """计算新闻情感因子面板。

    Args:
        panel: 标准面板 dict，至少含 "close" (pd.DataFrame, columns=股票代码, index=日期).

    Returns:
        与 close 同形的 pd.DataFrame，每列填同一情感分数（跨时间重复）。
    """
    close = panel["close"]
    codes = list(close.columns)
    shape = close.shape

    sentiment_series: dict[str, float] = {}
    for code in codes:
        sentiment_series[code] = _get_sentiment(code)

    # 初始填 0.0（无新闻数据 = 中性），避免全 NaN 导致 Registry 校验失败
    result = pd.DataFrame(
        np.zeros(shape),
        index=close.index,
        columns=close.columns,
        dtype=float,
    )

    for code, score in sentiment_series.items():
        if code in result.columns:
            result[code] = score

    logger.info(
        "sentiment_score computed: %d stocks, range [%.3f, %.3f]",
        len(codes),
        float(result.min().min() or 0),
        float(result.max().max() or 0),
    )
    return result
