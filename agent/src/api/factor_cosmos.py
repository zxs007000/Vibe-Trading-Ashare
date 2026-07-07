"""Factor Cosmos HTTP route — 因子星空图的后端数据端点。

Mounted by ``agent/api_server.py`` via ``register_factor_cosmos_routes(app, ...)``.

- ``POST /factor-cosmos`` — 运行完整的因子生命周期分析(拉数据 → 滚动 IC →
  相关矩阵 → 多因子筛选),返回星空图所需的 JSON:
  ``factors``(每个因子的状态/IR/Theme)、``logical_chain``(匹配的逻辑链)、
  ``correlations``(高相关对,用于"浅色星")、``themes``、``stats``(alive/decaying/dead 计数)。

数据拉取走 a-stock-data / mootdx 的 TCP 直连通达信协议,**需要国内 IP**。
海外/沙箱环境下 fetch 会失败,此时端点返回 503 + 清晰提示,前端会回退到
内置示例数据渲染星空。

错误处理原则同 alpha_routes:用户输入错误(非法 period/universe)直接 400;
内部异常记录日志并返回固定短语,绝不泄露堆栈/路径。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 项目根路径:agent/src/api/factor_cosmos.py → 上三层即 Vibe-Trading-main
# (run_factor_analysis.py 在其根目录)。把根目录插入 sys.path 以便复用
# run_factor_analysis 的 fetch / build_panel / compute_forward_returns。
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_error(_exc: BaseException) -> str:
    return "internal error; see server logs"


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

_VALID_PERIODS = {"1y", "2y", "3y", "6m", "1m"}
_VALID_UNIVERSES = {
    # 别名 → registry universe
    "csi300": "equity_cn",
    "csi500": "equity_cn",
    "hs300": "equity_cn",
    "sp500": "equity_us",
    "sz50": "equity_cn",
}


class FactorCosmosRequest(BaseModel):
    """POST /factor-cosmos body."""

    universe: str = Field("csi300", min_length=1, max_length=32)
    period: str = Field("2y", min_length=2, max_length=16)
    limit: int = Field(600, ge=1, le=2000)

    @field_validator("period")
    @classmethod
    def _period_known(cls, v: str) -> str:
        if v not in _VALID_PERIODS:
            raise ValueError(f"unknown period {v!r}; expected one of {sorted(_VALID_PERIODS)}")
        return v

    @field_validator("universe")
    @classmethod
    def _universe_known(cls, v: str) -> str:
        if v not in _VALID_UNIVERSES:
            raise ValueError(f"unknown universe {v!r}; expected one of {sorted(_VALID_UNIVERSES)}")
        return v


# ---------------------------------------------------------------------------
# Sync worker (runs in a thread; data-heavy)
# ---------------------------------------------------------------------------


def _parse_period_years(period: str) -> int:
    """'2y' → 2, '6m' → 0.5(向上取整到 1 年窗口需要)。"""
    if period.endswith("y"):
        return max(1, int(period[:-1]))
    if period.endswith("m"):
        months = int(period[:-1])
        return max(1, months // 12)  # 6m → 0, clamp to 1
    return 2


def _run_cosmos_blocking(payload: FactorCosmosRequest) -> dict[str, Any]:
    """Synchronous worker — called via ``asyncio.to_thread``."""
    from run_factor_analysis import (  # local import: heavy + needs sys.path root
        fetch_ashare_data,
        build_panel,
        compute_forward_returns,
        CSI300_SAMPLE,
    )
    from src.factors.registry import get_default_registry
    from src.factors.rolling_ic import batch_rolling_analysis
    from src.factors.multi_factor_selector import select_factors, SelectionConfig
    from src.factors.factor_correlation import (
        compute_factor_values,
        compute_correlation_matrix,
        find_correlated_pairs,
    )

    years = _parse_period_years(payload.period)
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    # --- Step 1: 拉数据(需国内 IP) ---
    logger.info("[factor-cosmos] fetching %s data %s→%s", payload.universe, start_date, end_date)
    stock_data = fetch_ashare_data(CSI300_SAMPLE, start_date, end_date)
    if not stock_data:
        raise RuntimeError(
            "no market data returned; mootdx requires mainland-China network access "
            "to reach TDX quote servers"
        )

    panel = build_panel(stock_data)
    return_df = compute_forward_returns(panel)

    # --- Step 2: 滚动 IC 全量分析 ---
    registry = get_default_registry()
    batch = batch_rolling_analysis(registry, panel, return_df, window=60)

    # --- Step 3: 多因子筛选(含逻辑链匹配) ---
    cfg = SelectionConfig(
        rolling_window=60,
        max_per_theme=2,
        min_rolling_ir=0.15,
        corr_threshold=0.7,
        max_total_factors=8,
    )
    sel = select_factors(registry, panel, return_df, cfg)
    selected_ids = set(sel.selected)

    # --- Step 4: 组装 factors 列表(star 数据源) ---
    factors: list[dict[str, Any]] = []
    for r in batch:
        s = r.summary
        factors.append(
            {
                "id": r.alpha_id,
                "theme": list(r.themes),
                "status": r.current_status,  # alive | decaying | dead
                "ir": round(float(s.get("last_rolling_ir", 0.0) or 0.0), 4),
                "ic_mean": round(float(s.get("ic_mean", 0.0) or 0.0), 6),
                "alive_days": s.get("alive_days"),
                "selected": r.alpha_id in selected_ids,
                "zoo": s.get("zoo"),
            }
        )

    # --- Step 5: 高相关对(浅色星 / 相关线) ---
    correlations: list[dict[str, Any]] = []
    factor_ids = [r.alpha_id for r in batch]
    factor_dfs = compute_factor_values(registry, panel, factor_ids)
    if len(factor_dfs) >= 2:
        try:
            corr_matrix = compute_correlation_matrix(factor_dfs)
            pairs = find_correlated_pairs(corr_matrix, threshold=0.7)
            correlations = [
                {"a": a, "b": b, "r": round(float(rho), 4)}
                for a, b, rho in pairs[:300]
            ]
        except Exception:  # noqa: BLE001 — correlation is best-effort; don't fail the whole call
            logger.exception("[factor-cosmos] correlation matrix failed; continuing without it")

    # --- Step 6: 主题集合 + 统计 ---
    themes_set: set[str] = set()
    for f in factors:
        for t in f["theme"]:
            themes_set.add(t)
    stats = {
        "total": len(factors),
        "alive": sum(1 for f in factors if f["status"] == "alive"),
        "decaying": sum(1 for f in factors if f["status"] == "decaying"),
        "dead": sum(1 for f in factors if f["status"] == "dead"),
    }

    return {
        "status": "ok",
        "generated_at": _now_iso(),
        "universe": payload.universe,
        "period": payload.period,
        "factors": factors,
        "logical_chain": {
            "name": sel.logical_chain,
            "score": round(float(sel.chain_score), 4),
            "themes_covered": list(sel.themes_covered),
        },
        "correlations": correlations,
        "themes": sorted(themes_set),
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AuthDep = Callable[..., Awaitable[Any] | Any]


def register_factor_cosmos_routes(
    app: FastAPI,
    require_auth: AuthDep | None = None,
    require_event_stream_auth: AuthDep | None = None,
) -> None:
    """Mount the factor-cosmos route onto ``app``."""
    if require_auth is None or require_event_stream_auth is None:
        import sys as _sys

        host = _sys.modules.get("api_server") or _sys.modules.get("agent.api_server")
        if host is None:
            raise RuntimeError(
                "register_factor_cosmos_routes: api_server module not in sys.modules; "
                "pass require_auth/require_event_stream_auth explicitly"
            )
        if require_auth is None:
            require_auth = host.require_auth
        if require_event_stream_auth is None:
            require_event_stream_auth = host.require_event_stream_auth

    @app.post(
        "/factor-cosmos",
        status_code=200,
        dependencies=[Depends(require_auth)],
    )
    async def factor_cosmos(payload: FactorCosmosRequest) -> dict[str, Any]:
        """Run the full factor lifecycle analysis and return cosmos JSON."""
        try:
            result = await asyncio.to_thread(_run_cosmos_blocking, payload)
            return result
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            # 数据拉取失败(无国内 IP / 网络不通)→ 503 + 清晰信息
            logger.warning("[factor-cosmos] data unavailable: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("[factor-cosmos] worker crashed")
            raise HTTPException(status_code=500, detail=_safe_error(exc))
