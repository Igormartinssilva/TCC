# ─── Stage 1: download kubectl + promtool binaries ───────────────────────────
FROM debian:bookworm-slim AS binaries

ARG KUBECTL_VERSION=v1.30.2
ARG PROMETHEUS_VERSION=2.53.0
ARG TARGETARCH=amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates tar \
    && rm -rf /var/lib/apt/lists/*


# kubectl
RUN curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" \
        -o /usr/local/bin/kubectl \
    && chmod +x /usr/local/bin/kubectl

# promtool (comes bundled inside the Prometheus release tarball)
RUN curl -fsSL \
    "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-${TARGETARCH}.tar.gz" \
    | tar -xz --strip-components=1 -C /usr/local/bin \
        "prometheus-${PROMETHEUS_VERSION}.linux-${TARGETARCH}/promtool"


# ─── Stage 2: final runtime image ─────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.title="prometheus-promql-mcp" \
      org.opencontainers.image.description="MCP server for Prometheus PromQL queries" \
      org.opencontainers.image.licenses="MIT"

# Non-root user for K8s security best-practices
RUN groupadd --gid 1001 mcp \
    && useradd --uid 1001 --gid mcp --shell /bin/sh --create-home mcp

# Copy binaries from build stage
COPY --from=binaries /usr/local/bin/kubectl   /usr/local/bin/kubectl
COPY --from=binaries /usr/local/bin/promtool  /usr/local/bin/promtool

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# App
WORKDIR /app
COPY prometheus-promql-mcp.py .
RUN chown -R mcp:mcp /app

USER mcp

EXPOSE 8001

# HEALTHCHECK com socket check (sem endpoint específico necessário)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8001), timeout=1)"


CMD ["python", "prometheus-promql-mcp.py"]