"""A-Stock-Data 数据加载器: A股 OHLCV 数据，基于通达信 TCP 直连协议.

基于开源项目 a-stock-data (github.com/simonlin1212/a-stock-data)，
底层使用 mootdx 的通达信原生二进制协议，不走 HTTP，无需 token，不会被封 IP。

支持范围: A股日/周/月/分钟级 OHLCV（沪/深自动识别，北交所暂不支持）。

在 registry.py 的 a_share 回退链中排第一:
  astockdata → tencent → mootdx → eastmoney → baostock → akshare → tushare → local
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders.base import cached_loader_fetch, validate_date_range
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

# 通达信频率编码（分钟级）
_INTRADAY_FREQ: dict[str, int] = {
    "1m": 8,    # 1分钟
    "5m": 0,    # 5分钟
    "15m": 1,   # 15分钟
    "30m": 2,   # 30分钟
    "1H": 3,    # 1小时
}
# 通达信频率编码（日级及以上）
_DAILY_FREQ: dict[str, int] = {
    "1D": 4,    # 日线
    "1W": 5,    # 周线
    "1M": 6,    # 月线
}

# bars() 每次返回最多 800 行，需要分页取更老的数据
_BARS_PAGE = 800
# 最多翻 25 页（25×800=20000根K线，约覆盖日线10年、1小时5年）
_MAX_PAGES = 25


def _is_a_share(code: str) -> bool:
    """判断是否为A股代码。支持 600519.SH / 000858.SZ / 600519 两种格式。"""
    upper = code.upper()
    if upper.endswith((".SH", ".SZ", ".BJ")):
        return True
    return len(code) == 6 and code.isdigit()


def _is_bj(code: str) -> bool:
    """判断是否为北交所代码。通达信 std 不支持北交所，需要跳过。"""
    upper = code.upper()
    if upper.endswith(".BJ"):
        return True
    return len(code) == 6 and code.isdigit() and code[0] in ("4", "8")


@register
class DataLoader:
    """A-Stock-Data A股数据加载器（TCP直连通达信，免费，无需认证）。"""

    name = "astockdata"
    markets = {"a_share"}
    requires_auth = False

    def __init__(self) -> None:
        self._client = None  # 延迟初始化通达信客户端

    def is_available(self) -> bool:
        """检查 mootdx 是否已安装。"""
        try:
            import mootdx  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        """懒加载通达信 Quotes 客户端，只连接一次。"""
        if self._client is None:
            from mootdx.quotes import Quotes
            self._client = Quotes.factory(market="std")
        return self._client

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """批量拉取A股 OHLCV 数据。

        Args:
            codes: 股票代码列表，如 ["600519.SH", "000858.SZ"]
            start_date: 起始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            interval: K线周期（1D/1W/1M/1m/5m/15m/30m/1H）
            fields: 忽略，保留接口兼容

        Returns:
            {股票代码: OHLCV DataFrame}，index=日期，columns=[open,high,low,close,volume]
        """
        validate_date_range(start_date, end_date)
        if interval not in _DAILY_FREQ and interval not in _INTRADAY_FREQ:
            raise ValueError(
                f"Unsupported interval for astockdata: {interval!r}. "
                f"Supported: {sorted(_DAILY_FREQ) + sorted(_INTRADAY_FREQ)}"
            )

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            if not _is_a_share(code):
                continue  # 跳过非A股代码
            if _is_bj(code):
                logger.warning("astockdata: 北交所 (%s) not supported, use akshare/tushare", code)
                continue  # 通达信不支持北交所
            try:
                # 使用缓存层：如果之前拉过同样的数据就直接读本地缓存
                df = cached_loader_fetch(
                    source=self.name,
                    symbol=code,
                    timeframe=interval,
                    start_date=start_date,
                    end_date=end_date,
                    fields=None,
                    fetch=lambda code=code: self._fetch_one(code, start_date, end_date, interval),
                )
                if df is not None and not df.empty:
                    result[code] = df
            except Exception as exc:
                logger.warning("astockdata failed for %s: %s", code, exc)
        return result

    def _fetch_one(
        self, code: str, start_date: str, end_date: str, interval: str,
    ) -> Optional[pd.DataFrame]:
        """拉取单只股票数据。日线用 get_k_data（支持日期范围），其他周期用 bars 分页。"""
        symbol = code.split(".")[0]
        client = self._get_client()

        # 日线有原生的日期范围 API，直接传起止日期
        if interval == "1D":
            df = client.get_k_data(code=symbol, start_date=start_date, end_date=end_date)
            return self._normalize_daily(df)

        # 分钟线/周线/月线只能从最新往前翻页
        freq = _DAILY_FREQ.get(interval) or _INTRADAY_FREQ[interval]
        return self._fetch_bars_paginated(client, symbol, freq, start_date, end_date)

    @staticmethod
    def _fetch_bars_paginated(
        client, symbol: str, freq: int, start_date: str, end_date: str,
    ) -> Optional[pd.DataFrame]:
        """分页拉取 bars() 数据，从最新往前翻，直到覆盖 start_date。

        通达信 bars() 每次返回最近 800 根K线。通过 start 参数跳过最新 N 根，
        实现翻页。最多翻 25 页，防止无限循环。
        """
        start_ts = pd.Timestamp(start_date)
        chunks: list[pd.DataFrame] = []
        for page in range(_MAX_PAGES):
            df = client.bars(
                symbol=symbol,
                frequency=freq,
                start=page * _BARS_PAGE,  # 跳过最新 N 根
                offset=_BARS_PAGE,         # 每页取 800 根
            )
            if df is None or df.empty:
                break
            chunks.append(df)
            # 如果本页最早一根K线已经早于 start_date，不需要再翻了
            first_dt = pd.to_datetime(df["datetime"].iloc[0])
            if first_dt <= start_ts:
                break
        if not chunks:
            return None
        combined = pd.concat(chunks, ignore_index=False)
        return DataLoader._normalize_bars(combined, start_date, end_date)

    @staticmethod
    def _normalize_daily(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """将 get_k_data() 的输出标准化为统一 OHLCV 格式。

        get_k_data 返回: [open, close, high, low, vol, amount, date, code]
        需要: index=trade_date(DatetimeIndex), columns=[open,high,low,close,volume]
        """
        if df is None or df.empty:
            return None
        out = df.rename(columns={"vol": "volume"}).copy()
        out.index = pd.to_datetime(out.index)
        out.index.name = "trade_date"
        for col in ("open", "high", "low", "close", "volume"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        keep = ["open", "high", "low", "close", "volume"]
        if "amount" in out.columns:
            out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
            keep = keep + ["amount"]
        out = out[keep].dropna(subset=["open", "high", "low", "close"])
        return out.sort_index() if not out.empty else None

    @staticmethod
    def _normalize_bars(
        df: Optional[pd.DataFrame], start_date: str, end_date: str,
    ) -> Optional[pd.DataFrame]:
        """将 bars() 的输出标准化并裁剪到指定日期范围。

        bars() 返回: [open, close, high, low, vol, amount, year, month, day, hour, minute, datetime]
        需要和 _normalize_daily 相同的输出格式。
        """
        if df is None or df.empty:
            return None
        out = df.copy()
        if "datetime" in out.columns:
            out["trade_date"] = pd.to_datetime(out["datetime"])
            out = out.set_index("trade_date")
        else:
            out.index = pd.to_datetime(out.index)
            out.index.name = "trade_date"
        out = out.sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        keep = ["open", "high", "low", "close", "volume"]
        if "amount" in out.columns:
            out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
            keep = keep + ["amount"]
        out = out[keep].dropna(subset=["open", "high", "low", "close"])
        # 截止到 end_date 当天的最后一根K线（含收盘）
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out.loc[pd.Timestamp(start_date):end_ts]
        return out if not out.empty else None
