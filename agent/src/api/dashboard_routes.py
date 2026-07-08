"""Dashboard data routes — Web UI 首页仪表盘.

提供三组数据, 全部优先走本地 ``stcok_worm`` (通达信 TDX TCP, 失败回退腾讯),
任何数据源都拿不到时回退到内置演示数据, 保证前端界面始终能渲染:

- ``GET /dashboard/market``   — A 股主要指数实时(日频)行情
- ``GET /dashboard/kline``    — 单只指数/个股的 OHLCV 日 K
- ``GET /dashboard/portfolio``— 内置示范(纸账户)组合: 持仓市值 / 当日盈亏 /
                                累计盈亏 / 权益曲线, 价格优先用真实最新收盘价
- ``GET /dashboard/market-state``— 大盘 8 态择时 (CSI300): 当前状态 / 置信度 /
                                推荐逻辑链 / 近期状态切换

所有耗时取数都在线程池里并行执行, 并带 60s 内存缓存, 避免重复打 TDX 服务器。
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# 大盘 8 态择时引擎 (因子策略的决策层, 来自云端策略包)
from src.factors.market_state import classify, state_history  # noqa: E402

# ---------------------------------------------------------------------------
# 默认标的
# ---------------------------------------------------------------------------

# A 股主要指数 (code 用通达信/腾讯通用 6 位 + 市场后缀)
DEFAULT_INDICES: List[Dict[str, Any]] = [
    {"code": "000001.SH", "name": "上证指数", "is_index": True},
    {"code": "399001.SZ", "name": "深证成指", "is_index": True},
    {"code": "399006.SZ", "name": "创业板指", "is_index": True},
    {"code": "000300.SH", "name": "沪深300", "is_index": True},
    {"code": "000905.SH", "name": "中证500", "is_index": True},
    {"code": "000688.SH", "name": "科创50", "is_index": True},
]

# 内置示范组合 (纸账户). shares=持仓股数, cost=持仓成本均价(仅用于算累计盈亏).
DEFAULT_PORTFOLIO: List[Dict[str, Any]] = [
    {"code": "600519.SH", "name": "贵州茅台", "shares": 100, "cost": 1500.0},
    {"code": "300750.SZ", "name": "宁德时代", "shares": 500, "cost": 180.0},
    {"code": "600036.SH", "name": "招商银行", "shares": 2000, "cost": 35.0},
    {"code": "002594.SZ", "name": "比亚迪", "shares": 300, "cost": 240.0},
    {"code": "601318.SH", "name": "中国平安", "shares": 1500, "cost": 48.0},
    {"code": "000858.SZ", "name": "五粮液", "shares": 400, "cost": 150.0},
]

# 纸账户现金 (演示)
PAPER_CASH = 500_000.0

# 演示兜底行情 (code -> (last, prev_close, open))
_DEMO_INDEX_QUOTES: Dict[str, tuple] = {
    "000001.SH": (3210.5, 3195.2, 3198.0),
    "399001.SZ": (10120.3, 10050.8, 10060.0),
    "399006.SZ": (2030.7, 2005.4, 2010.0),
    "000300.SH": (3745.9, 3728.1, 3730.0),
    "000905.SH": (5480.2, 5455.6, 5460.0),
    "000688.SH": (892.4, 880.1, 883.0),
}
_DEMO_STOCK_QUOTES: Dict[str, tuple] = {
    "600519.SH": (1482.0, 1501.0, 1498.0),
    "300750.SZ": (176.3, 181.2, 179.0),
    "600036.SH": (36.2, 35.8, 35.9),
    "002594.SZ": (245.6, 240.1, 242.0),
    "601318.SH": (47.1, 48.3, 47.6),
    "000858.SZ": (142.5, 148.0, 145.0),
}

# 内存缓存
_CACHE: Dict[str, tuple] = {}  # key -> (expire_ts, value)
_CACHE_TTL = 60.0
_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 取数 (带缓存 + 线程池)
# ---------------------------------------------------------------------------

def _cache_get(key: str) -> Optional[Any]:
    with _CACHE_LOCK:
        item = _CACHE.get(key)
    if not item:
        return None
    expire, value = item
    if time.time() > expire:
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time() + _CACHE_TTL, value)


def _norm_bar(r: Any) -> Dict[str, float]:
    """把单根 K 线归一为 dict. 兼容 dict-like 与位置列表两种返回格式."""
    if isinstance(r, (list, tuple)):
        # 部分版本返回位置列表: [date, open, close, high, low, volume]
        return {
            "date": str(r[0])[:10] if len(r) > 0 else "",
            "open": float(r[1]) if len(r) > 1 else 0.0,
            "close": float(r[2]) if len(r) > 2 else 0.0,
            "high": float(r[3]) if len(r) > 3 else 0.0,
            "low": float(r[4]) if len(r) > 4 else 0.0,
            "volume": float(r[5]) if len(r) > 5 else 0.0,
        }
    return {
        "date": str(r.get("date", ""))[:10],
        "open": float(r.get("open", 0.0)),
        "high": float(r.get("high", 0.0)),
        "low": float(r.get("low", 0.0)),
        "close": float(r.get("close", 0.0)),
        "volume": float(r.get("volume", r.get("vol", 0.0))),
    }


def _fetch_kline_raw(code: str, is_index: bool, count: int = 240) -> Optional[List[Dict[str, float]]]:
    """拉单标的日 K, 返回 [{date,open,high,low,close,volume}] 或 None.

    优先通达信(TDX), 个股失败再回退腾讯.
    """
    try:
        import stcok_worm  # noqa: F401
    except Exception as exc:  # pragma: no cover - 依赖环境
        logger.debug("stcok_worm 不可用: %s", exc)
        return None

    raw = None
    try:
        if is_index:
            raw = stcok_worm.mootdx_source.get_index_kline(code, 9, count)
        else:
            raw = stcok_worm.mootdx_source.get_kline(code, 9, count)
            if not raw:
                raw = stcok_worm.tencent.get_kline(code, "day")
    except Exception as exc:
        logger.warning("拉 %s K线失败: %s", code, exc)
        return None

    if not raw:
        return None

    out: List[Dict[str, float]] = []
    for r in raw:
        try:
            out.append(_norm_bar(r))
        except (ValueError, TypeError, IndexError):
            continue
    return out if out else None


def _demo_kline(last: float, bars: int = 120) -> List[Dict[str, float]]:
    """生成一段用于兜底演示的日 K (随机游走, 末值≈last)."""
    today = __import__("datetime").date.today()
    out: List[Dict[str, float]] = []
    v = last * (0.82 + random.random() * 0.06)
    for i in range(bars - 1, -1, -1):
        v += (last - v) * 0.04 + (random.random() - 0.5) * last * 0.02
        d = today.fromordinal(today.toordinal() - i)
        out.append(
            {
                "date": d.isoformat(),
                "open": round(v * (1 + (random.random() - 0.5) * 0.01), 2),
                "high": round(v * (1 + random.random() * 0.012), 2),
                "low": round(v * (1 - random.random() * 0.012), 2),
                "close": round(v, 2),
                "volume": round(random.random() * 1e6, 0),
            }
        )
    out[-1]["close"] = round(last, 2)
    return out


def _quote_from_kline(kline: List[Dict[str, float]], name: str, code: str) -> Dict[str, Any]:
    last = kline[-1]["close"]
    prev = kline[-2]["close"] if len(kline) > 1 else last
    opn = kline[-1]["open"]
    change = last - prev
    pct = (change / prev * 100.0) if prev else 0.0
    return {
        "code": code,
        "name": name,
        "last": round(last, 2),
        "prev_close": round(prev, 2),
        "open": round(opn, 2),
        "change": round(change, 2),
        "pct": round(pct, 2),
        "source": "realtime",
    }


# ---------------------------------------------------------------------------
# 业务组装
# ---------------------------------------------------------------------------

def _build_market(codes: Optional[List[str]] = None) -> Dict[str, Any]:
    cache_key = "market:" + (",".join(codes) if codes else "default")
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    indices = DEFAULT_INDICES
    if codes:
        # 允许 ?codes= 覆盖默认指数集合 (仍标记 is_index=True)
        indices = [{"code": c, "name": c, "is_index": True} for c in codes]

    def _one(idx: Dict[str, Any]) -> Dict[str, Any]:
        code = idx["code"]
        name = idx["name"]
        kl = _fetch_kline_raw(code, idx.get("is_index", True), count=5)
        if kl and len(kl) >= 1:
            return _quote_from_kline(kl, name, code)
        # 兜底演示
        demo = _DEMO_INDEX_QUOTES.get(code)
        if demo:
            last, prev, opn = demo
            change = last - prev
            return {
                "code": code,
                "name": name,
                "last": last,
                "prev_close": prev,
                "open": opn,
                "change": round(change, 2),
                "pct": round(change / prev * 100.0, 2),
                "source": "demo",
            }
        return {"code": code, "name": name, "last": 0, "prev_close": 0,
                "open": 0, "change": 0, "pct": 0, "source": "demo"}

    quotes: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(8, len(indices) or 1)) as ex:
        futs = {ex.submit(_one, idx): idx for idx in indices}
        for fut in as_completed(futs):
            quotes.append(fut.result())

    # 维持原始顺序
    quotes.sort(key=lambda q: [i["code"] for i in indices].index(q["code"]))
    real = any(q["source"] == "realtime" for q in quotes)
    result = {"indices": quotes, "source": "realtime" if real else "demo"}
    _cache_set(cache_key, result)
    return result


def _build_kline(code: str, is_index: bool, count: int) -> Dict[str, Any]:
    cache_key = f"kline:{code}:{is_index}:{count}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    kl = _fetch_kline_raw(code, is_index, count=max(count, 10))
    source = "realtime"
    if not kl:
        # 兜底演示
        demo = _DEMO_INDEX_QUOTES.get(code) or _DEMO_STOCK_QUOTES.get(code)
        last = demo[0] if demo else 100.0
        kl = _demo_kline(last, bars=count)
        source = "demo"

    bars = [
        {
            "time": r["date"],
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["volume"],
        }
        for r in kl[-count:]
    ]
    result = {"code": code, "is_index": is_index, "bars": bars, "source": source}
    _cache_set(cache_key, result)
    return result


def _build_portfolio() -> Dict[str, Any]:
    cache_key = "portfolio"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    def _one(h: Dict[str, Any]) -> Dict[str, Any]:
        code = h["code"]
        name = h["name"]
        shares = h["shares"]
        cost = h["cost"]
        kl = _fetch_kline_raw(code, False, count=240)
        if kl and len(kl) >= 1:
            last = kl[-1]["close"]
            prev = kl[-2]["close"] if len(kl) > 1 else last
            series = kl
            src = "realtime"
        else:
            demo = _DEMO_STOCK_QUOTES.get(code)
            last = demo[0] if demo else cost
            prev = demo[1] if demo else cost
            series = _demo_kline(last, bars=240)
            src = "demo"
        value = shares * last
        day_pnl = shares * (last - prev)
        total_pnl = value - shares * cost
        return {
            "code": code,
            "name": name,
            "shares": shares,
            "cost": cost,
            "last": round(last, 2),
            "prev_close": round(prev, 2),
            "value": round(value, 2),
            "day_pnl": round(day_pnl, 2),
            "day_pct": round((last - prev) / prev * 100.0, 2) if prev else 0.0,
            "total_pnl": round(total_pnl, 2),
            "total_pct": round(total_pnl / (shares * cost) * 100.0, 2) if shares * cost else 0.0,
            "series": series,
            "source": src,
        }

    holdings_raw: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(8, len(DEFAULT_PORTFOLIO) or 1)) as ex:
        futs = {ex.submit(_one, h): h for h in DEFAULT_PORTFOLIO}
        for fut in as_completed(futs):
            holdings_raw.append(fut.result())

    holdings_raw.sort(key=lambda x: [h["code"] for h in DEFAULT_PORTFOLIO].index(x["code"]))

    # 持仓市值 / 盈亏
    positions_value = sum(h["value"] for h in holdings_raw)
    day_pnl = sum(h["day_pnl"] for h in holdings_raw)
    total_pnl = sum(h["total_pnl"] for h in holdings_raw)
    cash = PAPER_CASH
    total_value = positions_value + cash

    # 权益曲线: 取各持仓序列尾部公共长度, 按股数加权
    min_len = min((len(h["series"]) for h in holdings_raw), default=0)
    equity_curve: List[Dict[str, Any]] = []
    if min_len > 1:
        # 用第一个持仓的日期轴
        base_series = holdings_raw[0]["series"][-min_len:]
        dates = [b["date"] for b in base_series]
        for i in range(min_len):
            v = cash
            for h in holdings_raw:
                v += h["shares"] * h["series"][-min_len:][i]["close"]
            equity_curve.append({"date": dates[i], "value": round(v, 2)})
    else:
        # 兜底: 平滑上行曲线
        base = total_value * 0.8
        dates = [__import__("datetime").date.today().fromordinal(__import__("datetime").date.today().toordinal() - (120 - i)).isoformat() for i in range(120)]
        for i, d in enumerate(dates):
            v = base + (total_value - base) * (i / 119) + (random.random() - 0.5) * total_value * 0.01
            equity_curve.append({"date": d, "value": round(v, 2)})

    holdings = [
        {k: v for k, v in h.items() if k != "series"} for h in holdings_raw
    ]
    real = any(h["source"] == "realtime" for h in holdings_raw)
    result = {
        "cash": cash,
        "positions_value": round(positions_value, 2),
        "total_value": round(total_value, 2),
        "day_pnl": round(day_pnl, 2),
        "day_pct": round(day_pnl / positions_value * 100.0, 2) if positions_value else 0.0,
        "total_pnl": round(total_pnl, 2),
        "total_pct": round(total_pnl / (total_value - cash) * 100.0, 2) if (total_value - cash) else 0.0,
        "equity_curve": equity_curve,
        "holdings": holdings,
        "source": "realtime" if real else "demo",
    }
    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# 大盘 8 态择时 (CSI300) —— 因子策略的决策层
# ---------------------------------------------------------------------------

# 演示回退: 对齐 STRATEGY_SUMMARY 中 2026.07.07 的"冲高回落"快照
_DEMO_MARKET_STATE: Dict[str, Any] = {
    "state": "pullback",
    "label_zh": "冲高回落",
    "confidence": 0.72,
    "since_date": "2026-06-19",
    "duration_days": 19,
    "snapshot": {
        "close": 4792.0,
        "ma60": 4843.0,
        "ma250": 4576.0,
        "ret20_pct": 1.4,
        "ret5_pct": -3.8,
        "vol20_annual_pct": 26.0,
    },
    "recommended": [
        {"chain_id": "value_qlowvol", "name_zh": "价值质量低波", "match": "neutral", "score": 0.5},
        {"chain_id": "value_momentum", "name_zh": "价值动量", "match": "neutral", "score": 0.5},
        {"chain_id": "value_stable", "name_zh": "价值稳定", "match": "neutral", "score": 0.5},
        {"chain_id": "reversal_momentum", "name_zh": "反转接力", "match": "neutral", "score": 0.5},
        {"chain_id": "vol_reversal", "name_zh": "波动回归", "match": "neutral", "score": 0.5},
        {"chain_id": "quality_momentum", "name_zh": "质量动量", "match": "avoid", "score": 0.0},
        {"chain_id": "liq_momentum", "name_zh": "放量动量", "match": "avoid", "score": 0.0},
        {"chain_id": "micro_reversal", "name_zh": "微观反转(已降级)", "match": "avoid", "score": 0.0},
    ],
    "recent_transitions": [
        {"date": "2026-02-10", "label_zh": "震荡上行"},
        {"date": "2026-03-21", "label_zh": "单边上涨"},
        {"date": "2026-05-08", "label_zh": "震荡上行"},
        {"date": "2026-06-03", "label_zh": "窄幅盘整"},
        {"date": "2026-06-19", "label_zh": "冲高回落"},
    ],
}


def _build_market_state() -> Dict[str, Any]:
    """大盘 8 态判定 (CSI300) + 推荐逻辑链 + 近期状态切换.

    优先用真实 CSI300 日 K (经 ``_fetch_kline_raw`` 走 TDX/腾讯); 拿不到则回退演示.
    """
    cache_key = "market_state"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    source = "realtime"
    try:
        import pandas as pd

        bars = _fetch_kline_raw("000300.SH", True, 300)
        if not bars or len(bars) < 120:
            raise RuntimeError("CSI300 kline unavailable / too short")
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"]).set_index("date").sort_index()
        if len(df) < 120:
            raise RuntimeError("CSI300 insufficient rows after cleaning")
        close = df["close"]

        result = classify(close)
        hist = state_history(close, min_duration=5)
        recent = hist[hist["state"] != hist["state"].shift()].tail(8)
        transitions = [
            {"date": str(r.name.date()), "label_zh": r["label_zh"]}
            for _, r in recent.iterrows()
        ]

        data: Dict[str, Any] = {
            "state": result.state,
            "label_zh": result.label_zh,
            "confidence": result.confidence,
            "since_date": result.since_date,
            "duration_days": result.duration_days,
            "snapshot": {
                "close": float(result.feature_snapshot.get("close", 0.0)),
                "ma60": float(result.feature_snapshot.get("ma60", 0.0)),
                "ma250": float(result.feature_snapshot.get("ma250", 0.0)),
                "ret20_pct": float(result.feature_snapshot.get("ret20_pct", 0.0)),
                "ret5_pct": float(result.feature_snapshot.get("ret5_pct", 0.0)),
                "vol20_annual_pct": float(result.feature_snapshot.get("vol20_annual_pct", 0.0)),
            },
            "recommended": [
                {
                    "chain_id": c["chain_id"],
                    "name_zh": c["name_zh"],
                    "match": c["match"],
                    "score": float(c["score"]),
                }
                for c in result.recommended_chains
            ],
            "recent_transitions": transitions,
        }
    except Exception as exc:
        logger.warning("market_state realtime failed (%s), falling back to demo", exc)
        source = "demo"
        data = _DEMO_MARKET_STATE

    out = {"market_state": data, "source": source}
    _cache_set(cache_key, out)
    return out


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------

def register_dashboard_routes(app: FastAPI) -> None:
    """Mount dashboard routes onto ``app``."""

    @app.get("/dashboard/market")
    def dashboard_market(codes: Optional[str] = None):
        """A 股主要指数行情. ?codes=000001.SH,399001.SZ 可覆盖默认集合."""
        code_list = [c.strip() for c in codes.split(",") if c.strip()] if codes else None
        return _build_market(code_list)

    @app.get("/dashboard/kline")
    def dashboard_kline(code: str, is_index: bool = True, count: int = 120):
        """单标的日 K. is_index=true 走指数接口."""
        count = max(20, min(count, 800))
        return _build_kline(code, bool(is_index), count)

    @app.get("/dashboard/portfolio")
    def dashboard_portfolio():
        """内置示范(纸账户)组合快照: 市值 / 盈亏 / 权益曲线 / 持仓."""
        return _build_portfolio()

    @app.get("/dashboard/market-state")
    def dashboard_market_state():
        """大盘 8 态择时 (CSI300): 当前状态 / 置信度 / 推荐逻辑链 / 近期切换."""
        return _build_market_state()
