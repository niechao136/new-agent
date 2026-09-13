"""Authentication tests: API-key middleware, config loading and card security."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from news_agent.a2a.card import build_agent_card
from news_agent.a2a.server import create_app
from news_agent.config import AuthSettings, LLMSettings, Settings, SourceConfig

BASE_URL = "http://test"
API_KEY = "secret-key-1"
OTHER_KEY = "secret-key-2"


def auth_settings(tmp_path, *, enabled: bool = True, **auth_kwargs: Any) -> Settings:
    return Settings(
        agent_url=BASE_URL,
        sources=[SourceConfig(name="mock", type="mock")],
        llm=LLMSettings(enabled=False),
        cache_enabled=False,
        cache_path=str(tmp_path / "auth.sqlite3"),
        log_level="WARNING",
        auth=AuthSettings(enabled=enabled, api_keys=[API_KEY, OTHER_KEY], **auth_kwargs),
    )


@pytest.fixture
def protected_app(tmp_path):
    return create_app(auth_settings(tmp_path))


@pytest.fixture
async def http(protected_app):
    transport = httpx.ASGITransport(app=protected_app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        yield client


# ---------------------------------------------------------------------------
# disabled by default: existing deployments keep working
# ---------------------------------------------------------------------------
async def test_disabled_allows_anonymous(tmp_path):
    app = create_app(auth_settings(tmp_path, enabled=False))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.get("/skills")
    assert response.status_code == 200


async def test_enabled_without_keys_disables_enforcement(tmp_path):
    settings = auth_settings(tmp_path)
    settings = settings.model_copy(
        update={"auth": AuthSettings(enabled=True, api_keys=[])}
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        response = await client.get("/skills")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# enforcement on protected endpoints
# ---------------------------------------------------------------------------
async def test_protected_endpoint_rejects_missing_key(http: httpx.AsyncClient):
    response = await http.get("/skills")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert "detail" in response.json()


async def test_protected_endpoint_rejects_invalid_key(http: httpx.AsyncClient):
    response = await http.get("/skills", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


async def test_protected_endpoint_accepts_api_key_header(http: httpx.AsyncClient):
    response = await http.get("/skills", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert response.json()["skills"]


async def test_protected_endpoint_accepts_second_key(http: httpx.AsyncClient):
    response = await http.get("/skills", headers={"X-API-Key": OTHER_KEY})
    assert response.status_code == 200


async def test_protected_endpoint_accepts_bearer(http: httpx.AsyncClient):
    response = await http.get("/skills", headers={"Authorization": f"Bearer {API_KEY}"})
    assert response.status_code == 200


async def test_custom_header_name(tmp_path):
    settings = auth_settings(tmp_path, api_key_header="X-Custom-Key")
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
        default_header = await client.get(
            "/skills", headers={"X-API-Key": API_KEY}
        )
        custom_header = await client.get(
            "/skills", headers={"X-Custom-Key": API_KEY}
        )
    assert default_header.status_code == 401
    assert custom_header.status_code == 200


# ---------------------------------------------------------------------------
# public endpoints stay open (health probes + A2A discovery)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path", ["/healthz", "/readyz", "/.well-known/agent-card.json", "/.well-known/agent.json"]
)
async def test_public_paths_need_no_key(http: httpx.AsyncClient, path: str):
    response = await http.get(path)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Agent Card advertises the security requirements
# ---------------------------------------------------------------------------
async def test_agent_card_declares_security(tmp_path):
    card = build_agent_card(auth_settings(tmp_path))
    assert set(card.security_schemes) == {"bearer", "api_key"}
    api_key_scheme = card.security_schemes["api_key"].api_key_security_scheme
    assert api_key_scheme.location == "header"
    assert api_key_scheme.name == "X-API-Key"
    assert card.security_requirements


async def test_agent_card_without_auth_has_no_security(tmp_path):
    card = build_agent_card(auth_settings(tmp_path, enabled=False))
    assert not card.security_schemes
    assert not card.security_requirements


# ---------------------------------------------------------------------------
# env-driven configuration
# ---------------------------------------------------------------------------
def test_settings_load_from_env(monkeypatch):
    import os

    monkeypatch.setenv("NEWS_AGENT_AUTH_ENABLED", "1")
    monkeypatch.setenv("NEWS_AGENT_AUTH_API_KEYS", f"{API_KEY}, {OTHER_KEY}")
    monkeypatch.setenv("NEWS_AGENT_AUTH_API_KEY_HEADER", "X-My-Key")

    settings = Settings.load(env=dict(os.environ))

    assert settings.auth.enabled
    assert settings.auth.api_keys == [API_KEY, OTHER_KEY]
    assert settings.auth.api_key_header == "X-My-Key"
    assert settings.auth.configured


def test_settings_default_auth_disabled():
    settings = Settings.load(env={})
    assert not settings.auth.enabled
    assert not settings.auth.configured
