# Linux x86_64 packaging. GPU/device acceptance is a separate release gate.
FROM node:24-bookworm-slim AS frontend
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS wheel
WORKDIR /build
COPY requirements/build.txt /tmp/build.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/build.txt
COPY pyproject.toml ./
COPY src/ src/
RUN python -m hatchling build -t wheel

# Copy only the upstream streaming executable, not its web UI or services.
FROM alexxit/go2rtc:1.9.14 AS streaming

FROM python:3.11-slim-bookworm AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 argus \
    && useradd --uid 10001 --gid argus --create-home argus \
    && mkdir -p /app/data /app/configs /app/.local-validation \
    && chown -R argus:argus /app/data /app/configs /app/.local-validation
COPY --from=frontend /web/dist /app/web/dist
COPY --from=streaming /usr/local/bin/go2rtc /usr/local/bin/go2rtc
COPY --chown=argus:argus configs/ /app/configs/
COPY --from=wheel /build/dist/*.whl /tmp/wheels/

# This target contains no model engines and never starts capture/inference.
FROM base AS functional-check
COPY requirements/functional.txt /tmp/functional.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/functional.txt \
    && pip install --no-deps /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels
COPY scripts/ scripts/
COPY tests/ tests/
COPY src/ src/
COPY pyproject.toml ./
USER argus
CMD ["python", "scripts/run_container_functional_checks.py"]

FROM base AS runtime
ENV ARGUS__DASHBOARD__FRONTEND_DIST_DIR=/app/web/dist
COPY requirements/build.txt /tmp/build.txt
COPY requirements/runtime-linux-x86_64.txt /tmp/runtime.txt
RUN test "$(uname -m)" = "x86_64" \
    && pip install --no-cache-dir --require-hashes -r /tmp/build.txt \
    && pip install --no-cache-dir --no-build-isolation --require-hashes -r /tmp/runtime.txt \
    && pip install --no-deps /tmp/wheels/*.whl \
    && pip check \
    && rm -rf /tmp/wheels
USER argus
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://127.0.0.1:8080/api/system/health', timeout=3); r.raise_for_status()" || exit 1
ENTRYPOINT ["python", "-m", "argus"]
CMD ["--config", "/app/configs/runtime.yaml"]
