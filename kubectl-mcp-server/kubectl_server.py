#!/usr/bin/env python3
import subprocess
import sys
import logging
from mcp.server.fastmcp import FastMCP

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Create an MCP server
mcp = FastMCP("Kubectl")

def run_kubectl(command: list) -> dict:
    """Executa comando kubectl e retorna resultado."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout if result.returncode == 0 else result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "Comando expirou (timeout 30s)"
        }
    except Exception as e:
        return {
            "success": False,
            "output": f"Erro: {str(e)}"
        }


@mcp.tool()
def list_pods(namespace: str = "default") -> str:
    """Lista todos os pods em um namespace específico."""
    logger.info(f"Listando pods em namespace: {namespace}")
    result = run_kubectl(["kubectl", "get", "pods", "-n", namespace, "-o", "wide"])
    return result["output"]


@mcp.tool()
def get_pod_logs(pod_name: str, namespace: str = "default", lines: int = 50) -> str:
    """Obtém logs de um pod específico."""
    logger.info(f"Obtendo logs do pod: {pod_name}")
    result = run_kubectl(["kubectl", "logs", pod_name, "-n", namespace, f"--tail={lines}"])
    return result["output"]


@mcp.tool()
def describe_pod(pod_name: str, namespace: str = "default") -> str:
    """Obtém descrição detalhada de um pod."""
    logger.info(f"Descrevendo pod: {pod_name}")
    result = run_kubectl(["kubectl", "describe", "pod", pod_name, "-n", namespace])
    return result["output"]


@mcp.tool()
def cluster_info() -> str:
    """Obtém informações do cluster Kubernetes."""
    logger.info("Obtendo informações do cluster")
    result = run_kubectl(["kubectl", "cluster-info"])
    return result["output"]