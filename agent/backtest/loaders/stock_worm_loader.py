"""stock-worm loader: local A-share data source package (通达信/mootdx 优先).

Wraps the user's local ``stcok_worm`` package as a Vibe-Trading data-source
loader. The package is installed in the backend venv; ``is_available`` checks
that the package imports (its optional 通达信/tdxpy dependency is needed for
the TDX path, otherwise the in-package 腾讯 source is used as fallback).

Data source: 通达信 (mootdx/TDX TCP) is preferred; falls back to 腾讯 within
the same package when TDX is unreachable. No API token required.

Markets: A-shares (SH/SZ/BJ).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders.base import cached_loader_fetch, validate_date_range
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

# Vibe-Trading interval codes -> (mootdx freq code, tencent period code)
_INTERVAL_MAP = {
    "1D": (9, "day"),
    "1W": (5, "week"),
    "1M": (6, "month"),
    "D": (9, "day"),
    "W": (5, "week"),
    "M": (6, "month"),
}


@register
class DataLoader:
    """Local stock-worm A-share OHLCV loader (通达信优先, 腾讯兜底)."""

    name = "stock_worm"
    markets = {"a_share"}
    requires_auth = False

    def is_available(self) -> bool:
        try:
            import stcok_worm  # noqa: F401

            return True
        except Exception as exc:  # pragma: no cover - depends on env
            logger.debug("stock_worm loader unavailable: %s", exc)
            return False

    def __init__(self) -> None:
        pass

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        validate_date_range(start_date, end_date)

        freq, period = _INTERVAL_MAP.get(interval, (9, "day"))

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            try:
                df = cached_loader_fetch(
                    source=self.name,
                    symbol=code,
                    timeframe=interval,
                    start_date=start_date,
                    end_date=end_date,
                    fields=None,
                    fetch=lambda code=code: self._fetch_one(code, freq, period),
                )
                if df is None or df.empty:
                    continue
                # Trim to the requested window (sources may return full history).
                mask = (df.index >= pd.Timestamp(start_date)) & (
                    df.index <= pd.Timestamp(end_date)
                )
                df = df.loc[mask]
                if not df.empty:
                    result[code] = df
            except Exception as exc:
                logger.warning("stock_worm failed for %s: %s", code, exc)
        return result

    def _fetch_one(self, code: str, freq: int, period: str) -> Optional[pd.DataFrame]:
        try:
            import stcok_worm
        except Exception:
            return None

        # Prefer 通达信 (TDX TCP); fall back to 腾讯 within the same package.
        raw = stcok_worm.mootdx_source.get_kline(code, freq)
        if not raw:
            logger.debug(
                "stock_worm mootdx returned nothing for %s; falling back to tencent", code
            )
            raw = stcok_worm.tencent.get_kline(code, period)
        if not raw:
            return None

        rows = []
        for r in raw:
            try:
                rows.append(
                    {
                        "trade_date": pd.Timestamp(str(r["date"])[:10]),
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "volume": float(r.get("volume", 0.0)),
                    }
                )
            except (KeyError, ValueError, TypeError):
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows).set_index("trade_date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].dropna(
            subset=["open", "high", "low", "close"]
        )
        return df
