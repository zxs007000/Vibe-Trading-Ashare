"""LLM and data-source settings HTTP routes.

Mounted by ``agent/api_server.py`` via ``register_settings_routes(app, ...)``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys as _sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Agent root (agent/) — resolved from this file's location (agent/src/api/).
_AGENT_DIR = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Pydantic models (defined locally -- NO shared modules, per maintainer rule)
# ---------------------------------------------------------------------------


class LLMProviderOption(BaseModel):
    """Supported LLM provider metadata for the settings UI."""

    name: str
    label: str
    api_key_env: Optional[str] = None
    base_url_env: str
    default_model: str
    default_base_url: str
    api_key_required: bool = True
    auth_type: str = "api_key"
    login_command: Optional[str] = None


class LLMSettingsResponse(BaseModel):
    """Current LLM runtime settings."""

    provider: str
    model_name: str
    base_url: str
    api_key_env: Optional[str] = None
    api_key_configured: bool
    api_key_hint: Optional[str] = None
    api_key_required: bool
    temperature: float
    timeout_seconds: int
    max_retries: int
    reasoning_effort: str
    sse_timeout_seconds: int
    env_path: str
    providers: List[LLMProviderOption]


class UpdateLLMSettingsRequest(BaseModel):
    """Update LLM settings persisted to agent/.env."""

    provider: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    clear_api_key: bool = False
    temperature: float = 0.0
    timeout_seconds: int = Field(120, ge=1, le=3600)
    max_retries: int = Field(2, ge=0, le=20)
    reasoning_effort: Optional[str] = None


class DataSourceSettingsResponse(BaseModel):
    """Current data source credential settings."""

    tushare_token_configured: bool
    tushare_token_hint: Optional[str] = None
    baostock_supported: bool
    baostock_installed: bool
    baostock_message: str
    env_path: str


class UpdateDataSourceSettingsRequest(BaseModel):
    """Update project-local data source credentials."""

    tushare_token: Optional[str] = None
    clear_tushare_token: bool = False


class SetPreferredDataSourceRequest(BaseModel):
    """Persist the user's preferred backtest data source."""

    preferred_source: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Provider metadata (settings-exclusive)
# ---------------------------------------------------------------------------

LLM_PROVIDER_CONFIG_PATH = _AGENT_DIR / "src" / "providers" / "llm_providers.json"


def _load_llm_providers() -> List[LLMProviderOption]:
    """Load provider metadata from JSON so additions stay data-driven."""
    try:
        raw = json.loads(LLM_PROVIDER_CONFIG_PATH.read_text(encoding="utf-8"))
        providers = [LLMProviderOption(**item) for item in raw]
    except Exception as exc:
        raise RuntimeError(f"Failed to load LLM provider config: {LLM_PROVIDER_CONFIG_PATH}") from exc

    seen: set[str] = set()
    for provider in providers:
        if provider.name in seen:
            raise RuntimeError(f"Duplicate LLM provider name: {provider.name}")
        seen.add(provider.name)
    if not providers:
        raise RuntimeError("LLM provider config must not be empty")
    return providers


LLM_PROVIDERS = _load_llm_providers()
LLM_PROVIDER_BY_NAME = {provider.name: provider for provider in LLM_PROVIDERS}
LLM_REASONING_EFFORTS = {"", "low", "medium", "high", "max"}
LLM_API_KEY_PLACEHOLDERS = {"", "sk-or-v1-your-key-here", "sk-xxx", "xxx", "gsk_xxx"}
TUSHARE_TOKEN_PLACEHOLDERS = {"", "your-tushare-token"}

# Friendly, UI-facing descriptions for the known backtest data sources. The
# canonical name set lives in ``backtest.loaders.registry.VALID_SOURCES``; this
# map is display-only and falls back to the raw name for anything unknown.
_DATA_SOURCE_DESCRIPTIONS: Dict[str, str] = {
    "auto": "自动选择：按市场回退链挑选可用源",
    "tushare": "Tushare（需 token）",
    "akshare": "AKShare（免费公共接口）",
    "tencent": "腾讯财经行情（免费）",
    "eastmoney": "东方财富行情（免费）",
    "sina": "新浪财经行情（免费）",
    "baostock": "BaoStock（免费，TCP 协议）",
    "mootdx": "通达信 TDX（需 tdxpy）",
    "local": "本地数据桥（CSV/Parquet，需配置）",
    "yfinance": "Yahoo Finance（美股/全球）",
    "yahoo": "Yahoo Finance（免费）",
    "stooq": "Stooq（免费 EOD）",
    "okx": "OKX（加密货币）",
    "ccxt": "CCXT（加密货币交易所）",
    "futu": "富途（需账号）",
    "finnhub": "Finnhub（需 token）",
    "alphavantage": "Alpha Vantage（需 token）",
    "tiingo": "Tiingo（需 token）",
    "fmp": "Financial Modeling Prep（需 token）",
    "sec_edgar": "SEC EDGAR（美股基本面）",
    "rsshub_events": "RSSHub 事件流（需自建实例）",
    "stock_worm": "Stock-Worm 本地库（通达信优先，免 token）",
}


def _list_data_sources(preferred: Optional[str]) -> List[Dict[str, Any]]:
    """Return every valid backtest data source with availability + preferred flag.

    ``VALID_SOURCES`` is the single source of truth shared by the backtest
    config schema and the agent backtest tool, so this endpoint automatically
    includes any newly registered loader (e.g. ``stock_worm``) without a frontend
    change.
    """
    from backtest.loaders.registry import VALID_SOURCES, LOADER_REGISTRY, _ensure_registered

    _ensure_registered()
    items: List[Dict[str, Any]] = []
    for name in sorted(VALID_SOURCES):
        available = False
        if name == "auto":
            available = True
        else:
            try:
                cls = LOADER_REGISTRY.get(name)
                if cls is not None:
                    available = bool(cls().is_available())
            except Exception:
                available = False
        items.append(
            {
                "name": name,
                "available": available,
                "description": _DATA_SOURCE_DESCRIPTIONS.get(name, name),
                "is_preferred": bool(preferred and name == preferred),
            }
        )
    return items


# ---------------------------------------------------------------------------
# Host access helpers (late-binding for test monkeypatch compat)
# ---------------------------------------------------------------------------


def _host():
    """Return the ``api_server`` module for late-access attribute reads.

    Tests monkeypatch ``ENV_PATH``, ``ENV_EXAMPLE_PATH``, ``_baostock_supported``
    and ``_baostock_installed`` directly on the ``api_server`` module; every
    function that reads these symbols goes through ``_host()`` so monkeypatched
    values take effect.
    """
    return _sys.modules.get("api_server") or _sys.modules.get("agent.api_server")


# ---------------------------------------------------------------------------
# Settings-exclusive helpers
# ---------------------------------------------------------------------------


def _baostock_supported() -> bool:
    """Check whether the project has a BaoStock loader implementation."""
    host = _host()
    agent_dir = host.AGENT_DIR if host is not None else _AGENT_DIR
    loader_dir = agent_dir / "backtest" / "loaders"
    return any((loader_dir / name).exists() for name in ("baostock.py", "baostock_loader.py"))


def _baostock_installed() -> bool:
    """Check whether the optional BaoStock package is importable."""
    return importlib.util.find_spec("baostock") is not None


def _read_settings_env_values() -> Dict[str, str]:
    """Read settings without creating agent/.env.

    Prefer the user's active agent/.env.  If it does not exist yet, fall back
    to agent/.env.example for display defaults only.
    """
    host = _host()
    env_path = host.ENV_PATH
    env_example_path = host.ENV_EXAMPLE_PATH
    read_env = host._read_env_values
    if env_path.exists():
        return read_env(env_path)
    if env_example_path.exists():
        return read_env(env_example_path)
    return {}


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _build_llm_settings_response(
    values: Optional[Dict[str, str]] = None,
) -> LLMSettingsResponse:
    """Build the public settings payload from dotenv values."""
    host = _host()
    env_values = values if values is not None else _read_settings_env_values()
    provider_name = env_values.get("LANGCHAIN_PROVIDER", "openai").strip().lower()
    provider = LLM_PROVIDER_BY_NAME.get(provider_name, LLM_PROVIDER_BY_NAME["openai"])
    api_key = env_values.get(provider.api_key_env or "", "") if provider.api_key_env else ""
    api_key_configured = host._is_configured_secret(api_key, LLM_API_KEY_PLACEHOLDERS)
    api_key_hint = None
    if provider.auth_type == "oauth":
        try:
            from src.providers.openai_codex import get_openai_codex_login_status

            token = get_openai_codex_login_status()
        except Exception:
            token = None
        api_key_configured = bool(token)
        api_key_hint = None
    return LLMSettingsResponse(
        provider=provider.name,
        model_name=env_values.get("LANGCHAIN_MODEL_NAME", provider.default_model),
        base_url=env_values.get(provider.base_url_env, provider.default_base_url),
        api_key_env=provider.api_key_env,
        api_key_configured=api_key_configured,
        api_key_hint=api_key_hint,
        api_key_required=provider.api_key_required,
        temperature=host._coerce_float(env_values.get("LANGCHAIN_TEMPERATURE", "0.0"), 0.0),
        timeout_seconds=host._coerce_int(env_values.get("TIMEOUT_SECONDS", "120"), 120),
        max_retries=host._coerce_int(env_values.get("MAX_RETRIES", "2"), 2),
        reasoning_effort=env_values.get("LANGCHAIN_REASONING_EFFORT", "").strip().lower(),
        sse_timeout_seconds=host._coerce_int(env_values.get("VIBE_TRADING_SSE_TIMEOUT", "90"), 90),
        env_path=host._project_relative_path(host.ENV_PATH),
        providers=LLM_PROVIDERS,
    )


def _build_data_source_settings_response(
    values: Optional[Dict[str, str]] = None,
) -> DataSourceSettingsResponse:
    """Build the public data source settings payload."""
    host = _host()
    env_values = values if values is not None else _read_settings_env_values()
    token = env_values.get("TUSHARE_TOKEN", "")
    token_configured = host._is_configured_secret(token, TUSHARE_TOKEN_PLACEHOLDERS)
    # Late-access baostock helpers for monkeypatch compat.
    baostock_sup = getattr(host, "_baostock_supported", _baostock_supported)
    baostock_ins = getattr(host, "_baostock_installed", _baostock_installed)
    supported = baostock_sup()
    installed = baostock_ins()
    if supported:
        baostock_message = "BaoStock loader is available."
    elif installed:
        baostock_message = "BaoStock package is installed, but this project has no BaoStock loader."
    else:
        baostock_message = "No BaoStock loader is registered in this project."
    return DataSourceSettingsResponse(
        tushare_token_configured=token_configured,
        tushare_token_hint=None,
        baostock_supported=supported,
        baostock_installed=installed,
        baostock_message=baostock_message,
        env_path=host._project_relative_path(host.ENV_PATH),
    )


def _sync_runtime_env(provider: LLMProviderOption, updates: Dict[str, str]) -> None:
    """Apply saved LLM settings to the running API process."""
    host = _host()
    for key, value in updates.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

    if provider.api_key_env:
        key_value = os.environ.get(provider.api_key_env, "")
        if host._is_configured_secret(key_value, LLM_API_KEY_PLACEHOLDERS):
            os.environ["OPENAI_API_KEY"] = key_value
        else:
            os.environ.pop("OPENAI_API_KEY", None)
    elif provider.auth_type == "oauth":
        os.environ.pop("OPENAI_API_KEY", None)
    else:
        os.environ["OPENAI_API_KEY"] = "ollama"

    base_url = os.environ.get(provider.base_url_env, "")
    if base_url:
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
    else:
        os.environ.pop("OPENAI_API_BASE", None)
        os.environ.pop("OPENAI_BASE_URL", None)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AuthDep = Callable[..., Awaitable[Any] | Any]


def register_settings_routes(
    app: FastAPI,
    require_local_or_auth: AuthDep | None = None,
    require_settings_write_auth: AuthDep | None = None,
) -> None:
    """Mount the settings routes onto ``app``."""
    host = _sys.modules.get("api_server") or _sys.modules.get("agent.api_server")

    if host is None:
        raise RuntimeError(
            "register_settings_routes: api_server module not in sys.modules; "
            "ensure api_server is imported before calling this function"
        )

    if require_local_or_auth is None:
        require_local_or_auth = host.require_local_or_auth
    if require_settings_write_auth is None:
        require_settings_write_auth = host.require_settings_write_auth

    # --- Routes ---

    @app.get(
        "/settings/llm",
        response_model=LLMSettingsResponse,
        dependencies=[Depends(require_local_or_auth)],
    )
    async def get_llm_settings():
        """Return project-local LLM settings for the Web UI."""
        return _build_llm_settings_response()

    @app.put(
        "/settings/llm",
        response_model=LLMSettingsResponse,
        dependencies=[Depends(require_settings_write_auth)],
    )
    async def update_llm_settings(payload: UpdateLLMSettingsRequest):
        """Persist project-local LLM settings and update the running process."""
        host_ref = _host()
        provider_name = payload.provider.strip().lower()
        provider = LLM_PROVIDER_BY_NAME.get(provider_name)
        if provider is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported LLM provider"
            )

        model_name = payload.model_name.strip()
        if not model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Model name is required"
            )

        if payload.temperature < 0 or payload.temperature > 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Temperature must be between 0 and 2",
            )

        reasoning_effort = (payload.reasoning_effort or "").strip().lower()
        if reasoning_effort not in LLM_REASONING_EFFORTS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reasoning effort must be low, medium, high, or max",
            )

        current_values = _read_settings_env_values()
        base_url = (
            payload.base_url if payload.base_url is not None else provider.default_base_url
        ).strip()
        if provider.auth_type == "oauth":
            try:
                from src.providers.openai_codex import validate_codex_base_url

                base_url = validate_codex_base_url(base_url)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
                ) from exc
        updates: Dict[str, str] = {
            "LANGCHAIN_PROVIDER": provider.name,
            "LANGCHAIN_MODEL_NAME": model_name,
            provider.base_url_env: base_url,
            "LANGCHAIN_TEMPERATURE": str(payload.temperature),
            "TIMEOUT_SECONDS": str(payload.timeout_seconds),
            "MAX_RETRIES": str(payload.max_retries),
        }
        if reasoning_effort or "LANGCHAIN_REASONING_EFFORT" in current_values:
            updates["LANGCHAIN_REASONING_EFFORT"] = reasoning_effort

        if provider.api_key_env:
            if payload.clear_api_key:
                updates[provider.api_key_env] = ""
            elif payload.api_key is not None and payload.api_key.strip():
                api_key = payload.api_key.strip()
                updates[provider.api_key_env] = (
                    api_key
                    if host_ref._is_configured_secret(api_key, LLM_API_KEY_PLACEHOLDERS)
                    else ""
                )
            elif provider.api_key_env in current_values and host_ref._is_configured_secret(
                current_values[provider.api_key_env],
                LLM_API_KEY_PLACEHOLDERS,
            ):
                updates[provider.api_key_env] = current_values[provider.api_key_env]
        elif payload.clear_api_key:
            os.environ.pop("OPENAI_API_KEY", None)

        host_ref._write_env_values(host_ref.ENV_PATH, updates)
        _sync_runtime_env(provider, updates)
        return _build_llm_settings_response(host_ref._read_env_values(host_ref.ENV_PATH))

    @app.get(
        "/settings/data-sources",
        response_model=DataSourceSettingsResponse,
        dependencies=[Depends(require_local_or_auth)],
    )
    async def get_data_source_settings():
        """Return project-local data source credentials for the Web UI."""
        return _build_data_source_settings_response()

    @app.put(
        "/settings/data-sources",
        response_model=DataSourceSettingsResponse,
        dependencies=[Depends(require_settings_write_auth)],
    )
    async def update_data_source_settings(payload: UpdateDataSourceSettingsRequest):
        """Persist project-local data source credentials and update the running process."""
        host_ref = _host()
        current_values = _read_settings_env_values()
        updates: Dict[str, str] = {}

        if payload.clear_tushare_token:
            updates["TUSHARE_TOKEN"] = ""
        elif payload.tushare_token is not None and payload.tushare_token.strip():
            updates["TUSHARE_TOKEN"] = payload.tushare_token.strip()
        elif "TUSHARE_TOKEN" in current_values:
            updates["TUSHARE_TOKEN"] = current_values["TUSHARE_TOKEN"]

        if updates:
            host_ref._write_env_values(host_ref.ENV_PATH, updates)
            token = updates.get("TUSHARE_TOKEN", "").strip()
            if host_ref._is_configured_secret(token, TUSHARE_TOKEN_PLACEHOLDERS):
                os.environ["TUSHARE_TOKEN"] = token
            else:
                os.environ.pop("TUSHARE_TOKEN", None)

        return _build_data_source_settings_response(
            host_ref._read_env_values(host_ref.ENV_PATH)
        )

    @app.get(
        "/backtest/data-sources",
        dependencies=[Depends(require_local_or_auth)],
    )
    async def list_backtest_data_sources():
        """List all valid backtest data sources with availability (single source of truth)."""
        host_ref = _host()
        values = host_ref._read_env_values(host_ref.ENV_PATH)
        preferred = (values.get("PREFERRED_DATA_SOURCE") or "").strip() or None
        return {"preferred_source": preferred, "sources": _list_data_sources(preferred)}

    @app.put(
        "/backtest/data-sources",
        dependencies=[Depends(require_settings_write_auth)],
    )
    async def set_backtest_data_source(payload: SetPreferredDataSourceRequest):
        """Save the preferred backtest data source to agent/.env."""
        from backtest.loaders.registry import VALID_SOURCES

        if payload.preferred_source not in VALID_SOURCES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unknown data source {payload.preferred_source!r}; must be one of {sorted(VALID_SOURCES)}",
            )
        host_ref = _host()
        host_ref._write_env_values(host_ref.ENV_PATH, {"PREFERRED_DATA_SOURCE": payload.preferred_source})
        os.environ["PREFERRED_DATA_SOURCE"] = payload.preferred_source
        return {
            "preferred_source": payload.preferred_source,
            "sources": _list_data_sources(payload.preferred_source),
        }
