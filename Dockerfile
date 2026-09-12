# syntax=docker/dockerfile:1
#
# news-agent container image
#
#   docker build -t news-agent:0.1.0 .
#   docker run --rm -p 9901:9901 news-agent:0.1.0
#
# The image is built in two stages: dependencies are resolved from the committed
# uv.lock into a self-contained virtualenv, and only that virtualenv is copied
# into the runtime layer (no compiler toolchain, no uv, no tests).

# ---------------------------------------------------------------------------
# Stage 1 - builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Pin uv so rebuilds stay reproducible (matches the version that generated uv.lock)
# and use the interpreter shipped by the base image instead of a downloaded one.
ARG UV_VERSION=0.11.3
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3.11 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app

# Dependency layer: only the manifests, so application code changes do not
# invalidate the (slow) dependency install.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Project layer: install the agent itself as a real wheel (--no-editable), so the
# runtime image needs neither the sources nor a src/ path hack.
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# ---------------------------------------------------------------------------
# Stage 2 - runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="news-agent" \
      org.opencontainers.image.description="A2A news agent (LangGraph + a2a-sdk)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    NEWS_AGENT_HOST=0.0.0.0 \
    NEWS_AGENT_PORT=9901 \
    NEWS_AGENT_CACHE_PATH=/data/news_agent.sqlite3 \
    NEWS_AGENT_LOG_LEVEL=INFO

COPY --from=builder /opt/venv /opt/venv

# Run unprivileged; /data holds the SQLite cache and is meant to be a volume.
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
 && install -d -o app -g app /data

WORKDIR /app
USER app

EXPOSE 9901

# Probes the agent's own /healthz endpoint (exit 0 = healthy). Toggle to
# readiness with NEWS_AGENT_HEALTHCHECK_ARGS="--ready" if you prefer /readyz.
ENV NEWS_AGENT_HEALTHCHECK_ARGS=""
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD news-agent healthcheck ${NEWS_AGENT_HEALTHCHECK_ARGS}

# `news-agent serve` reads NEWS_AGENT_HOST / NEWS_AGENT_PORT, so the same image
# can be re-pointed without rebuilding, and uvicorn handles SIGTERM gracefully.
CMD ["news-agent", "serve"]
