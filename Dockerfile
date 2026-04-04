# Argus - Nuclear FOE Visual Detection System
# Multi-stage build for minimal image size

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install the wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Install heavy ML dependencies separately for better caching
RUN pip install --no-cache-dir \
    ultralytics>=8.3.0 \
    anomalib>=1.2.0 \
    openvino>=2024.0.0

# Copy configs
COPY configs/ configs/

# Create data directories
RUN mkdir -p data/baselines data/models data/exports data/db data/alerts data/logs

# Non-root user for security
RUN useradd --create-home argus
USER argus

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8080/api/system/health'); r.raise_for_status()" || exit 1

ENTRYPOINT ["python", "-m", "argus"]
CMD ["--config", "configs/default.yaml"]
