from fastapi import FastAPI
import subprocess
import logging

app = FastAPI(
    title="Kubernetes Observability MCP",
    version="1.0.0",
    description="MCP para observabilidade Kubernetes via kubectl"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
def root():
    """Endpoint raiz para health check"""
    return {"status": "ok", "message": "Kubernetes Observability MCP is running"}


def run_kubectl(command: list) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return result.stderr
        return result.stdout
    except Exception as e:
        return str(e)


@app.post("/tools/list_pods", operation_id="list_pods")
def list_pods(namespace: str = "default"):
    """
    List pods in a Kubernetes namespace
    """
    logger.info(f"Listando pods no namespace {namespace}")
    output = run_kubectl(
        ["kubectl", "get", "pods", "-n", namespace, "-o", "wide"]
    )
    return {"output": output}


@app.post("/tools/get_pod_logs", operation_id="get_pod_logs")
def get_pod_logs(
    pod_name: str,
    namespace: str = "default",
    lines: int = 50
):
    """
    Retorna logs de um pod
    """
    logger.info(f"Logs do pod {pod_name}")
    output = run_kubectl(
        ["kubectl", "logs", pod_name, "-n", namespace, f"--tail={lines}"]
    )
    return {"output": output}


@app.post("/tools/describe_pod", operation_id="describe_pod")
def describe_pod(pod_name: str, namespace: str = "default"):
    """
    Descreve um pod
    """
    logger.info(f"Describe pod {pod_name}")
    output = run_kubectl(
        ["kubectl", "describe", "pod", pod_name, "-n", namespace]
    )
    return {"output": output}


@app.post("/tools/cluster_info", operation_id="cluster_info")
def cluster_info():
    """
    Informações do cluster Kubernetes
    """
    logger.info("Cluster info")
    output = run_kubectl(["kubectl", "cluster-info"])
    return {"output": output}

@app.post("/tools/metrics", operation_id="metrics")
def metrics():
    """
    Métricas do cluster Kubernetes
    """
    logger.info("Cluster metrics")
    output = run_kubectl(["kubectl", "top", "nodes"])
    return {"output": output}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)