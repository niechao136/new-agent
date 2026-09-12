"""CLI tests: command wiring and the container health probe."""

from __future__ import annotations

import argparse
from typing import Any

import httpx
import pytest

from news_agent.cli import _run_healthcheck, build_parser, main


def _args(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(list(argv))


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("serve", ("serve",)),
        ("card", ("card",)),
        ("skills", ("skills",)),
        ("sources", ("sources",)),
        ("run", ("run", "人形机器人")),
        ("call", ("call", "人形机器人")),
        ("task", ("task", "task-id")),
        ("healthcheck", ("healthcheck",)),
    ],
)
def test_parser_accepts_each_command(command: str, argv: tuple[str, ...]) -> None:
    assert _args(*argv).command == command


def test_unknown_command_exits() -> None:
    with pytest.raises(SystemExit):
        _args("does-not-exist")


# ---------------------------------------------------------------------------
# health probe (used by the container HEALTHCHECK)
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {"status": "ok"}

    def json(self) -> Any:
        return self._payload


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, calls: list[str], **kwargs: Any) -> None:
    status = kwargs.get("status_code", 200)

    class _Client:
        def __init__(self, **client_kwargs: Any) -> None:
            self.kwargs = client_kwargs

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def get(self, url: str) -> _FakeResponse:
            calls.append(url)
            return _FakeResponse(status)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


async def test_healthcheck_is_ok_and_targets_healthz(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    _install_fake_client(monkeypatch, calls)

    assert await _run_healthcheck(_args("healthcheck", "--base-url", "http://agent:9000")) == 0
    assert calls == ["http://agent:9000/healthz"]
    assert '"status": "ok"' in capsys.readouterr().out


async def test_healthcheck_defaults_to_agent_port(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NEWS_AGENT_PORT", "8123")
    calls: list[str] = []
    _install_fake_client(monkeypatch, calls)

    assert await _run_healthcheck(_args("healthcheck")) == 0
    assert calls == ["http://127.0.0.1:8123/healthz"]

    # --ready probes /readyz (and therefore warms the agent up)
    assert await _run_healthcheck(_args("healthcheck", "--ready")) == 0
    assert calls[-1] == "http://127.0.0.1:8123/readyz"
    capsys.readouterr()


async def test_healthcheck_reports_non_200(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    _install_fake_client(monkeypatch, calls, status_code=503)

    assert await _run_healthcheck(_args("healthcheck", "--ready")) == 1
    assert "HTTP 503" in capsys.readouterr().err


async def test_healthcheck_reports_unreachable_agent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # nothing listens on the discard port -> transport error -> exit code 1
    args = _args("healthcheck", "--base-url", "http://127.0.0.1:9", "--timeout", "1")

    assert await _run_healthcheck(args) == 1

    err = capsys.readouterr().err
    assert "unhealthy" in err
    assert "http://127.0.0.1:9/healthz" in err


def test_main_dispatches_healthcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _install_fake_client(monkeypatch, calls)

    assert main(["healthcheck", "--base-url", "http://agent:9000", "--ready"]) == 0
    assert calls == ["http://agent:9000/readyz"]


def test_main_prints_agent_card(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["card"]) == 0
    out = capsys.readouterr().out
    assert '"name": "news-agent"' in out
    assert "supportedInterfaces" in out
    assert "fetch_news" in out
