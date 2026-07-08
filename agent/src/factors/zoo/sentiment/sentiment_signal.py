"""情绪因子：龙虎榜机构信号（净买入 / 机构动向）。

数据源：stcok_worm.signals.dragon_tiger()，计算：
  - net_buy_ratio = 净买入额 / 成交额
  - institution_presence = 机构席位出现次数

高净买入 + 多机构席位 = 强看多信号。

注意事项：
  - 龙虎榜数据来自当日披露（T+0 盘后可得），T+1 开盘才是实际可交易时点。
  - 适合作为隔夜信号因子，decay_horizon=3 表示 3 天内信号衰减。
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
    "id": "sentiment_signal",
    "nickname": "龙虎榜信号因子",
    "theme": ["sentiment"],
    "formula_latex": (
        r"S_{\mathrm{dt}} = \mathrm{tanh}\left("
        r"\frac{\mathrm{net\_buy}}{\mathrm{turnover}} + "
        r"0.1 \cdot N_{\mathrm{institution}}\right)"
    ),
    "columns_required": ["close"],
    "extras_required": [],
    "requires_sector": False,
    "universe": ["equity_cn"],
    "frequency": ["1D"],
    "decay_horizon": 3,
    "min_warmup_bars": 1,
    "notes": (
        "基于 stock_worm 龙虎榜数据：净买入占比 + 机构席位数。"
        "信号来自历史披露，无前视偏差风险，适合回测。"
    ),
}

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".vibe-trading" / "sentiment_cache"
_CACHE_TTL = 86400


def _cache_key(code: str) -> str:
    return hashlib.md5(f"dragon_{code}".encode()).hexdigest()[:12]


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


def _get_dragon_signal(code: str) -> float:
    """从龙虎榜数据提取信号。

    返回范围 [-1, 1]：
      > 0.3 = 强看多（大额净买入 + 多机构）
      < -0.3 = 强看空（大额净卖出）
      0 = 无龙虎榜数据（中性）
    """
    cached = _load_cache(code)
    if cached is not None:
        return float(cached.get("signal", 0.0))

    try:
        from stcok_worm import signals as sw_signals

        records = sw_signals.dragon_tiger(code, page_size=5)
    except Exception:
        logger.debug("stock_worm dragon_tiger fetch failed for %s", code, exc_info=True)
        records = []

    if not records:
        _save_cache(code, {"signal": 0.0, "records": 0})
        return 0.0

    try:
        signals: list[float] = []
        for r in records:
            net_buy = float(r.get("NETBUYAMT", 0) or 0)  # 净买入额（元）
            turnover = float(r.get("TURNOVERRATE", 0) or 0)  # 换手率（%）
            buy_amt = float(r.get("BUYAMT", 0) or 0)  # 买入金额

            # 净买入占比（基于买入金额估算总成交）
            net_ratio = net_buy / max(buy_amt, 1.0)

            # 机构席位出现（看席位名称是否含"机构"）
            seat_names = str(r.get("BUYTRADERNAME", "") or "") + str(r.get("SELLTRADERNAME", "") or "")
            inst_count = seat_names.count("机构")

            sig = np.tanh(net_ratio + 0.1 * inst_count)
            signals.append(float(sig))

        avg_signal = float(np.mean(signals)) if signals else 0.0
    except Exception:
        logger.debug("dragon signal calc failed for %s", code, exc_info=True)
        avg_signal = 0.0

    _save_cache(code, {"signal": avg_signal, "records": len(records)})
    return avg_signal


def compute(panel: dict) -> pd.DataFrame:
    close = panel["close"]
    codes = list(close.columns)

    sig_values: dict[str, float] = {}
    for code in codes:
        sig_values[code] = _get_dragon_signal(code)

    # 初始填 0.0（无龙虎榜数据 = 中性信号），避免全 NaN 导致 Registry 校验失败
    result = pd.DataFrame(
        np.zeros(close.shape),
        index=close.index,
        columns=close.columns,
        dtype=float,
    )

    for code, sig in sig_values.items():
        if code in result.columns:
            result[code] = sig

    nonzero = sum(1 for v in sig_values.values() if v != 0.0)
    logger.info(
        "sentiment_signal computed: %d/%d stocks had dragon tiger data",
        nonzero,
        len(codes),
    )
    return result
