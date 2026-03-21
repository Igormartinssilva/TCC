FROM python:3.12-slim

# Install kubectl
RUN apt-get update && apt-get install -y curl ca-certificates && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && mv kubectl /usr/local/bin/kubectl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir mcp[cli] fastmcp pydantic uvicorn

WORKDIR /app
COPY k8s_mcp.py .

# Expose SSE port
EXPOSE 8080

# Run FastMCP directly with native SSE transport (no supergateway needed)
CMD ["python", "/app/k8s_mcp.py"]
