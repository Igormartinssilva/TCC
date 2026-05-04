import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prometheus-promql-mcp")

# ---------------------------------------------------------------------------
# Config — override via K8s ConfigMap / env vars
# ---------------------------------------------------------------------------
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-operated.monitoring.svc.cluster.local:9090")
KUBECTL_PATH   = os.getenv("KUBECTL_PATH", "/usr/local/bin/kubectl")
PROMTOOL_PATH  = os.getenv("PROMTOOL_PATH", "/usr/local/bin/promtool")
KUBECONFIG     = os.getenv("KUBECONFIG", "")
DEFAULT_NS     = os.getenv("DEFAULT_NAMESPACE", "default")
MCP_HOST       = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT       = int(os.getenv("MCP_PORT", "8001"))
MCP_TRANSPORT  = os.getenv("MCP_TRANSPORT", "sse")   # 'sse' or 'stdio'

mcp = FastMCP("prometheus-promql-mcp")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, timeout: int = 30) -> dict:
    try:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, **({"KUBECONFIG": KUBECONFIG} if KUBECONFIG else {})},
        )
        return {"stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s", "returncode": -1}
    except FileNotFoundError as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


async def _prom_get(path: str,  params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(f"{PROMETHEUS_URL}{path}", params=params or {})
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            return {"status": "error", "error": str(exc)}


def _kubectl(args: str, timeout: int = 30) -> str:
    kc = f"--kubeconfig {KUBECONFIG}" if KUBECONFIG else ""
    r  = _run(f"{KUBECTL_PATH} {kc} {args}", timeout=timeout)
    if r["returncode"] != 0 and r["stderr"]:
        return f"ERROR: {r['stderr']}\n{r['stdout']}"
    return r["stdout"] or r["stderr"]


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


async def _instant(expr: str, ) -> dict:
    return await _prom_get("/api/v1/query", {"query": expr})


async def _range(  expr: str, minutes: int = 30, step: str = "60s") -> dict:
    now   = datetime.utcnow()
    start = (now - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end   = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return await _prom_get("/api/v1/query_range", {"query": expr, "start": start, "end": end, "step": step})


# ===========================================================================
# 1) TOOL DISCOVERY
# ===========================================================================

@mcp.tool()
def list_available_tools() -> str:
    """
    Returns the full catalog of available tools grouped by category.
    ALWAYS call this first to understand what you can do before acting.
    """
    catalog = {
        "instruction": "Call list_available_tools() first. Then use pre-built tools — they require only pod_name/namespace, no PromQL.",
        "categories": {
            "pre_built_pod_metrics": [
                "get_pod_cpu", "get_pod_memory", "get_pod_restarts",
                "get_pod_network_io", "get_pod_ready_status", "get_pod_full_metrics",
            ],
            "pre_built_node_metrics": [
                "get_node_cpu", "get_node_memory", "get_node_disk",
                "get_node_network", "get_node_full_metrics",
            ],
            "pre_built_cluster_metrics": [
                "get_cluster_cpu_usage", "get_cluster_memory_usage",
                "get_firing_alerts", "get_oomkilled_pods",
                "get_crashloopbackoff_pods", "get_pending_pods",
                "get_deployment_replica_status", "get_api_server_health",
                "get_etcd_health", "get_pvc_usage",
            ],
            "generic_prometheus": [
                "query_prometheus", "prometheus_list_metrics",
                "prometheus_get_labels", "prometheus_get_label_values",
                "prometheus_get_alerts", "prometheus_get_rules",
                "prometheus_get_targets", "prometheus_health_check",
            ],
            "promtool": [
                "promtool_check_rules", "promtool_check_config",
                "promtool_lint_rules", "promtool_query_instant",
                "promtool_query_range", "analyze_promql", "suggest_promql_for",
            ],
            "compound_troubleshoot": [
                "troubleshoot_pod", "troubleshoot_node",
                "troubleshoot_namespace", "cluster_health_snapshot",
                "get_top_resource_consumers",
            ],
        },
        "quick_start_examples": {
            "check_pod_cpu":    'get_pod_cpu(pod_name="my-pod", namespace="default")',
            "check_pod_memory": 'get_pod_memory(pod_name="my-pod", namespace="default")',
            "all_pod_metrics":  'get_pod_full_metrics(pod_name="my-pod", namespace="default")',
            "cluster_health":   "cluster_health_snapshot()",
            "firing_alerts":    "get_firing_alerts()",
            "full_diagnosis":   'troubleshoot_pod(pod_name="my-pod", namespace="default")',
            "suggest_queries":  'suggest_promql_for(scenario="pod_cpu", pod_name="my-pod", namespace="default")',
        },
    }
    return json.dumps(catalog, indent=2)


# ===========================================================================
# 2) PRE-BUILT POD METRICS
# ===========================================================================

@mcp.tool()
async def get_pod_cpu(
    
    pod_name: str,
    namespace: str = DEFAULT_NS,
    mode: str = "instant",
    minutes: int = 15,
    step: str = "30s",
) -> str:
    """
    Get CPU usage in cores for a specific pod. No PromQL needed.

    Args:
        pod_name:  Exact pod name, e.g. 'nginx-deployment-647677fc66-4lzqh'
        namespace: Kubernetes namespace, e.g. 'default'
        mode:      'instant' for current value, 'range' for time series
        minutes:   Range lookback in minutes (default 15)
        step:      Range resolution e.g. '30s', '1m'
    """
    expr = (
        f'sum(rate(container_cpu_usage_seconds_total'
        f'{{pod="{pod_name}",namespace="{namespace}",container!=""}}[5m])) by (pod, container)'
    )
    data = await _range(expr, minutes, step) if mode == "range" else await _instant(expr)
    return json.dumps({"tool": "get_pod_cpu", "pod": pod_name, "namespace": namespace,
                       "unit": "cores", "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_pod_memory(
    
    pod_name: str,
    namespace: str = DEFAULT_NS,
    mode: str = "instant",
    minutes: int = 15,
    step: str = "30s",
) -> str:
    """
    Get memory working-set in bytes for a specific pod. No PromQL needed.

    Args:
        pod_name:  Exact pod name
        namespace: Kubernetes namespace
        mode:      'instant' or 'range'
        minutes:   Range lookback in minutes
        step:      Range resolution
    """
    expr = (
        f'sum(container_memory_working_set_bytes'
        f'{{pod="{pod_name}",namespace="{namespace}",container!=""}}) by (pod, container)'
    )
    data = await _range(expr, minutes, step) if mode == "range" else await _instant(expr)
    return json.dumps({"tool": "get_pod_memory", "pod": pod_name, "namespace": namespace,
                       "unit": "bytes", "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_pod_restarts(
    
    pod_name: str,
    namespace: str = DEFAULT_NS,
    window_minutes: int = 60,
) -> str:
    """
    Get container restart count for a pod over the last N minutes. No PromQL needed.

    Args:
        pod_name:       Exact pod name
        namespace:      Kubernetes namespace
        window_minutes: Lookback window (default 60)
    """
    expr = (
        f'increase(kube_pod_container_status_restarts_total'
        f'{{pod="{pod_name}",namespace="{namespace}"}}[{window_minutes}m])'
    )
    data = await _instant(expr)
    return json.dumps({"tool": "get_pod_restarts", "pod": pod_name, "namespace": namespace,
                       "window_minutes": window_minutes, "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_pod_network_io(
     
    pod_name: str,
    namespace: str = DEFAULT_NS,
    minutes: int = 15,
    step: str = "30s",
) -> str:
    """
    Get network RX/TX bytes per second for a pod. No PromQL needed.

    Args:
        pod_name:  Exact pod name
        namespace: Kubernetes namespace
        minutes:   Lookback window
        step:      Resolution
    """
    rx = f'sum(rate(container_network_receive_bytes_total{{pod="{pod_name}",namespace="{namespace}"}}[5m])) by (pod)'
    tx = f'sum(rate(container_network_transmit_bytes_total{{pod="{pod_name}",namespace="{namespace}"}}[5m])) by (pod)'
    rx_data, tx_data = await asyncio.gather(_range(rx, minutes, step), _range(tx, minutes, step))
    return json.dumps({"tool": "get_pod_network_io", "pod": pod_name, "namespace": namespace,
                       "rx_bytes_per_sec": rx_data, "tx_bytes_per_sec": tx_data}, indent=2)


@mcp.tool()
async def get_pod_ready_status(pod_name: str, namespace: str = DEFAULT_NS) -> str:
    """
    Check if pod containers are ready (1=ready, 0=not ready). No PromQL needed.

    Args:
        pod_name:  Exact pod name
        namespace: Kubernetes namespace
    """
    expr = f'kube_pod_container_status_ready{{pod="{pod_name}",namespace="{namespace}"}}'
    data = await _instant(expr)
    return json.dumps({"tool": "get_pod_ready_status", "pod": pod_name, "namespace": namespace,
                       "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_pod_full_metrics(
    
    pod_name: str,
    namespace: str = DEFAULT_NS,
    minutes: int = 15,
    step: str = "30s",
) -> str:
    """
    All-in-one pod metrics: CPU, memory, limits, restarts, readiness, exit reason, network.
    USE THIS when you need a complete picture of a pod health. No PromQL needed.

    Args:
        pod_name:  Exact pod name
        namespace: Kubernetes namespace
        minutes:   Range window in minutes
        step:      Resolution
    """
    queries = {
        "cpu_cores":        f'sum(rate(container_cpu_usage_seconds_total{{pod="{pod_name}",namespace="{namespace}",container!=""}}[5m])) by (container)',
        "memory_bytes":     f'sum(container_memory_working_set_bytes{{pod="{pod_name}",namespace="{namespace}",container!=""}}) by (container)',
        "memory_limit":     f'sum(kube_pod_container_resource_limits{{pod="{pod_name}",namespace="{namespace}",resource="memory"}}) by (container)',
        "cpu_limit":        f'sum(kube_pod_container_resource_limits{{pod="{pod_name}",namespace="{namespace}",resource="cpu"}}) by (container)',
        "restarts_1h":      f'increase(kube_pod_container_status_restarts_total{{pod="{pod_name}",namespace="{namespace}"}}[1h])',
        "ready":            f'kube_pod_container_status_ready{{pod="{pod_name}",namespace="{namespace}"}}',
        "waiting_reason":   f'kube_pod_container_status_waiting_reason{{pod="{pod_name}",namespace="{namespace}"}}',
        "last_exit_reason": f'kube_pod_container_status_last_terminated_reason{{pod="{pod_name}",namespace="{namespace}"}}',
        "net_rx_bps":       f'sum(rate(container_network_receive_bytes_total{{pod="{pod_name}",namespace="{namespace}"}}[5m]))',
        "net_tx_bps":       f'sum(rate(container_network_transmit_bytes_total{{pod="{pod_name}",namespace="{namespace}"}}[5m]))',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    return json.dumps({"tool": "get_pod_full_metrics", "pod": pod_name, "namespace": namespace,
                       "timestamp": _now_iso(), "promql_used": queries, "metrics": results}, indent=2)


# ===========================================================================
# 3) PRE-BUILT NODE METRICS
# ===========================================================================

@mcp.tool()
async def get_node_cpu(node_name: str,  minutes: int = 15, step: str = "30s") -> str:
    """
    Get CPU usage percentage for a specific node. No PromQL needed.

    Args:
        node_name: Node name as shown by 'kubectl get nodes'
        minutes:   Lookback window
        step:      Resolution
    """
    expr = f'100 - (avg by(instance)(rate(node_cpu_seconds_total{{mode="idle",instance=~"{node_name}.*"}}[5m])) * 100)'
    data = await _range(expr, minutes, step)
    return json.dumps({"tool": "get_node_cpu", "node": node_name, "unit": "percent",
                       "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_node_memory(node_name: str) -> str:
    """
    Get total, available and used memory % for a node. No PromQL needed.

    Args:
        node_name: Node name as shown by 'kubectl get nodes'
    """
    queries = {
        "total_bytes":     f'node_memory_MemTotal_bytes{{instance=~"{node_name}.*"}}',
        "available_bytes": f'node_memory_MemAvailable_bytes{{instance=~"{node_name}.*"}}',
        "used_percent":    f'(1 - node_memory_MemAvailable_bytes{{instance=~"{node_name}.*"}} / node_memory_MemTotal_bytes{{instance=~"{node_name}.*"}}) * 100',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    return json.dumps({"tool": "get_node_memory", "node": node_name,
                       "promql_used": queries, "result": results}, indent=2)


@mcp.tool()
async def get_node_disk(node_name: str,  mountpoint: str = "/") -> str:
    """
    Get disk usage for a node and mountpoint. No PromQL needed.

    Args:
        node_name:  Node name
        mountpoint: Filesystem mountpoint (default '/')
    """
    queries = {
        "avail_bytes":  f'node_filesystem_avail_bytes{{instance=~"{node_name}.*",mountpoint="{mountpoint}"}}',
        "size_bytes":   f'node_filesystem_size_bytes{{instance=~"{node_name}.*",mountpoint="{mountpoint}"}}',
        "used_percent": f'(1 - node_filesystem_avail_bytes{{instance=~"{node_name}.*",mountpoint="{mountpoint}"}} / node_filesystem_size_bytes{{instance=~"{node_name}.*",mountpoint="{mountpoint}"}}) * 100',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    return json.dumps({"tool": "get_node_disk", "node": node_name, "mountpoint": mountpoint,
                       "promql_used": queries, "result": results}, indent=2)


@mcp.tool()
async def get_node_network(node_name: str,  minutes: int = 15, step: str = "30s") -> str:
    """
    Get network RX/TX bytes per second for a node. No PromQL needed.

    Args:
        node_name: Node name
        minutes:   Lookback window
        step:      Resolution
    """
    queries = {
        "rx_bps": f'rate(node_network_receive_bytes_total{{instance=~"{node_name}.*",device!~"lo|veth.*|docker.*|br.*"}}[5m])',
        "tx_bps": f'rate(node_network_transmit_bytes_total{{instance=~"{node_name}.*",device!~"lo|veth.*|docker.*|br.*"}}[5m])',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_range(e, minutes, step) for e in queries.values()])))
    return json.dumps({"tool": "get_node_network", "node": node_name,
                       "promql_used": queries, "result": results}, indent=2)


@mcp.tool()
async def get_node_full_metrics( node_name: str) -> str:
    """
    All-in-one node metrics: CPU, memory, disk, load, network, pod count. No PromQL needed.

    Args:
        node_name: Node name as shown by 'kubectl get nodes'
    """
    queries = {
        "cpu_pct":      f'100 - (avg by(instance)(rate(node_cpu_seconds_total{{mode="idle",instance=~"{node_name}.*"}}[5m])) * 100)',
        "mem_pct":      f'(1 - node_memory_MemAvailable_bytes{{instance=~"{node_name}.*"}} / node_memory_MemTotal_bytes{{instance=~"{node_name}.*"}}) * 100',
        "mem_total":    f'node_memory_MemTotal_bytes{{instance=~"{node_name}.*"}}',
        "disk_pct":     f'(1 - node_filesystem_avail_bytes{{instance=~"{node_name}.*",mountpoint="/"}} / node_filesystem_size_bytes{{instance=~"{node_name}.*",mountpoint="/"}}) * 100',
        "load_1m":      f'node_load1{{instance=~"{node_name}.*"}}',
        "load_5m":      f'node_load5{{instance=~"{node_name}.*"}}',
        "load_15m":     f'node_load15{{instance=~"{node_name}.*"}}',
        "net_rx_bps":   f'sum(rate(node_network_receive_bytes_total{{instance=~"{node_name}.*",device!~"lo"}}[5m]))',
        "net_tx_bps":   f'sum(rate(node_network_transmit_bytes_total{{instance=~"{node_name}.*",device!~"lo"}}[5m]))',
        "pod_count":    f'count(kube_pod_info{{node="{node_name}"}})',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    return json.dumps({"tool": "get_node_full_metrics", "node": node_name,
                       "timestamp": _now_iso(), "promql_used": queries, "metrics": results}, indent=2)


# ===========================================================================
# 4) PRE-BUILT CLUSTER-WIDE METRICS
# ===========================================================================

@mcp.tool()
async def get_cluster_cpu_usage() -> str:
    """Get CPU usage percentage per node across the entire cluster. No PromQL needed."""
    expr = 'sort_desc(100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100))'
    data = await _instant(expr)
    return json.dumps({"tool": "get_cluster_cpu_usage", "unit": "percent",
                       "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_cluster_memory_usage() -> str:
    """Get memory usage percentage per node across the entire cluster. No PromQL needed."""
    expr = 'sort_desc((1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)'
    data = await _instant(expr)
    return json.dumps({"tool": "get_cluster_memory_usage", "unit": "percent",
                       "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_firing_alerts(severity: str = "") -> str:
    """
    Get all currently firing Prometheus alerts. No PromQL needed.

    Args:
        severity: Optional filter: 'critical', 'warning', 'info' — leave empty for all
    """
    expr = f'ALERTS{{alertstate="firing",severity="{severity}"}}' if severity else 'ALERTS{alertstate="firing"}'
    data = await _instant(expr)
    return json.dumps({"tool": "get_firing_alerts", "severity_filter": severity or "all",
                       "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_oomkilled_pods(namespace: str = "", window_minutes: int = 60) -> str:
    """
    Find pods OOMKilled in the last N minutes. No PromQL needed.

    Args:
        namespace:      Kubernetes namespace — leave empty for all
        window_minutes: Lookback window (default 60)
    """
    ns   = f',namespace="{namespace}"' if namespace else ""
    expr = f'increase(kube_pod_container_status_last_terminated_reason{{reason="OOMKilled"{ns}}}[{window_minutes}m]) > 0'
    data = await _instant(expr)
    return json.dumps({"tool": "get_oomkilled_pods", "namespace": namespace or "all",
                       "window_minutes": window_minutes, "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_crashloopbackoff_pods(namespace: str = "") -> str:
    """
    Find all pods currently in CrashLoopBackOff. No PromQL needed.

    Args:
        namespace: Kubernetes namespace — leave empty for all
    """
    ns   = f',namespace="{namespace}"' if namespace else ""
    expr = f'kube_pod_container_status_waiting_reason{{reason="CrashLoopBackOff"{ns}}} == 1'
    data = await _instant(expr)
    return json.dumps({"tool": "get_crashloopbackoff_pods", "namespace": namespace or "all",
                       "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_pending_pods(namespace: str = "") -> str:
    """
    Find all pods stuck in Pending state. No PromQL needed.

    Args:
        namespace: Kubernetes namespace — leave empty for all
    """
    ns   = f',namespace="{namespace}"' if namespace else ""
    expr = f'kube_pod_status_phase{{phase="Pending"{ns}}} == 1'
    data = await _instant(expr)
    return json.dumps({"tool": "get_pending_pods", "namespace": namespace or "all",
                       "promql": expr, "result": data}, indent=2)


@mcp.tool()
async def get_deployment_replica_status(namespace: str = "", deployment_name: str = "") -> str:
    """
    Get desired vs available replicas for deployments. No PromQL needed.

    Args:
        namespace:       Kubernetes namespace — leave empty for all
        deployment_name: Specific deployment — leave empty for all
    """
    filters = ""
    if namespace:
        filters += f'namespace="{namespace}"'
    if deployment_name:
        filters += ("," if filters else "") + f'deployment="{deployment_name}"'
    f_block = f"{{{filters}}}" if filters else ""
    queries = {
        "desired":     f'kube_deployment_spec_replicas{f_block}',
        "available":   f'kube_deployment_status_replicas_available{f_block}',
        "unavailable": f'kube_deployment_status_replicas_unavailable{f_block} > 0',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    return json.dumps({"tool": "get_deployment_replica_status",
                       "namespace": namespace or "all", "deployment": deployment_name or "all",
                       "promql_used": queries, "result": results}, indent=2)


@mcp.tool()
async def get_api_server_health() -> str:
    """Get Kubernetes API server request rate, error rate and P99 latency. No PromQL needed."""
    queries = {
        "request_rate_5m":     'sum(rate(apiserver_request_total[5m])) by (verb, code)',
        "error_rate_5xx":      'sum(rate(apiserver_request_total{code=~"5.."}[5m])) by (verb)',
        "latency_p99_seconds": 'histogram_quantile(0.99, sum(rate(apiserver_request_duration_seconds_bucket[5m])) by (le, verb))',
        "inflight_requests":   'sum(apiserver_current_inflight_requests) by (request_kind)',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    return json.dumps({"tool": "get_api_server_health", "timestamp": _now_iso(),
                       "promql_used": queries, "result": results}, indent=2)


@mcp.tool()
async def get_etcd_health() -> str:
    """Get etcd cluster health: leader, failed proposals, DB size, latency. No PromQL needed."""
    queries = {
        "has_leader":           'etcd_server_has_leader',
        "leader_changes_1h":    'increase(etcd_server_leader_changes_seen_total[1h])',
        "failed_proposals_5m":  'increase(etcd_server_proposals_failed_total[5m])',
        "db_size_bytes":        'etcd_mvcc_db_total_size_in_bytes',
        "grpc_latency_p99":     'histogram_quantile(0.99, sum(rate(grpc_server_handling_seconds_bucket{grpc_type="unary"}[5m])) by (le))',
        "disk_wal_sync_p99":    'histogram_quantile(0.99, sum(rate(etcd_disk_wal_fsync_duration_seconds_bucket[5m])) by (le))',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    return json.dumps({"tool": "get_etcd_health", "timestamp": _now_iso(),
                       "promql_used": queries, "result": results}, indent=2)


@mcp.tool()
async def get_pvc_usage(namespace: str = "", threshold_percent: float = 0) -> str:
    """
    Get PersistentVolumeClaim disk usage. No PromQL needed.

    Args:
        namespace:         Namespace filter — leave empty for all
        threshold_percent: Only return PVCs above this % (e.g. 80 for >80% full, 0 = all)
    """
    ns_f  = f'namespace="{namespace}"' if namespace else ""
    block = f"{{{ns_f}}}" if ns_f else ""
    pct   = f'(kubelet_volume_stats_used_bytes{block} / kubelet_volume_stats_capacity_bytes{block}) * 100'
    if threshold_percent > 0:
        pct += f" > {threshold_percent}"
    queries = {
        "used_bytes":    f'kubelet_volume_stats_used_bytes{block}',
        "capacity_bytes":f'kubelet_volume_stats_capacity_bytes{block}',
        "used_percent":  pct,
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    return json.dumps({"tool": "get_pvc_usage", "namespace": namespace or "all",
                       "threshold_percent": threshold_percent, "promql_used": queries, "result": results}, indent=2)


# ===========================================================================
# 5) GENERIC PROMETHEUS
# ===========================================================================

@mcp.tool()
async def query_prometheus(
     
    expr: str,
    mode: str = "instant",
    minutes: int = 30,
    step: str = "60s",
) -> str:
    """
    Generic PromQL executor. Prefer pre-built tools above when possible.
    Use only when you know the exact PromQL expression.

    Args:
        expr:    Valid PromQL expression
        mode:    'instant' (current) or 'range' (time series)
        minutes: Lookback for range mode (default 30)
        step:    Resolution for range mode (default '60s')
    """
    data = await _range(expr, minutes, step) if mode == "range" else await _instant(expr)
    return json.dumps({"promql": expr, "mode": mode, "result": data}, indent=2)


@mcp.tool()
async def prometheus_list_metrics(match: Optional[str] = None) -> str:
    """
    List all metric names in Prometheus.

    Args:
        match: Substring filter, e.g. 'container_cpu' — leave empty for all
    """
    data  = await _prom_get("/api/v1/label/__name__/values")
    names: list = data.get("data", [])
    if match:
        names = [n for n in names if match.lower() in n.lower()]
    return json.dumps({"total": len(names), "metrics": names}, indent=2)


@mcp.tool()
async def prometheus_get_labels(metric: str) -> str:
    """Return all label names for a metric."""
    return json.dumps(await _prom_get("/api/v1/labels", {"match[]": metric}), indent=2)


@mcp.tool()
async def prometheus_get_label_values(label: str, metric: Optional[str] = None) -> str:
    """
    Return all values for a label, optionally scoped to a metric.

    Args:
        label:  Label name, e.g. 'namespace', 'pod', 'node'
        metric: Optional metric to scope results
    """
    params = {"match[]": metric} if metric else {}
    return json.dumps(await _prom_get(f"/api/v1/label/{label}/values", params), indent=2)


@mcp.tool()
async def prometheus_get_alerts() -> str:
    """Return all currently firing alerts from Prometheus."""
    return json.dumps(await _prom_get("/api/v1/alerts"), indent=2)


@mcp.tool()
async def prometheus_get_rules() -> str:
    """Return all loaded alerting and recording rules."""
    return json.dumps(await _prom_get("/api/v1/rules"), indent=2)


@mcp.tool()
async def prometheus_get_targets() -> str:
    """Return all scrape targets and their UP/DOWN health."""
    return json.dumps(await _prom_get("/api/v1/targets"), indent=2)


@mcp.tool()
async def prometheus_health_check() -> str:
    """Check Prometheus /-/healthy and /-/ready endpoints."""
    results = {}
    async with httpx.AsyncClient(timeout=10) as client:
        for path in ("/-/healthy", "/-/ready"):
            try:
                r = await client.get(f"{PROMETHEUS_URL}{path}")
                results[path] = {"status_code": r.status_code, "body": r.text.strip()}
            except Exception as exc:
                results[path] = {"error": str(exc)}
    return json.dumps(results, indent=2)


# ===========================================================================
# 6) PROMTOOL + PROMQL HELPERS
# ===========================================================================

@mcp.tool()
def promtool_check_rules(rules_file_path: str) -> str:
    """Validate a Prometheus rules YAML file using promtool."""
    return json.dumps(_run(f"{PROMTOOL_PATH} check rules {rules_file_path}"), indent=2)


@mcp.tool()
def promtool_lint_rules(rules_file_path: str) -> str:
    """Lint a Prometheus rules YAML file for style issues."""
    return json.dumps(_run(f"{PROMTOOL_PATH} check rules --lint {rules_file_path}"), indent=2)


@mcp.tool()
def promtool_check_config(config_file_path: str) -> str:
    """Validate a prometheus.yml configuration file."""
    return json.dumps(_run(f"{PROMTOOL_PATH} check config {config_file_path}"), indent=2)


@mcp.tool()
def promtool_query_instant(expr: str) -> str:
    """Run an instant PromQL query via promtool CLI."""
    clean_expr = expr.replace("\\", "") #The agent sometimes adds extra backslashes when passing the query through the CLI, so we remove them here for promtool to work correctly.
    return json.dumps(_run(f"promtool query instant {PROMETHEUS_URL} {shlex.quote(clean_expr)}"), indent=2)


@mcp.tool()
def promtool_query_range(expr: str, start: str, end: str, step: str = "1m") -> str:
    """
    Run a range PromQL query via promtool CLI.

    Args:
        expr:  PromQL expression
        start: Start RFC3339, e.g. '2024-01-01T00:00:00Z'
        end:   End RFC3339
        step:  Resolution e.g. '1m', '5m'
    """
    cmd = f"{PROMTOOL_PATH} query range --url={PROMETHEUS_URL} --start={start} --end={end} --step={step} {shlex.quote(expr)}"
    return json.dumps(_run(cmd), indent=2)


@mcp.tool()
def analyze_promql(query: str) -> str:
    """
    Analyze a PromQL query: complexity, functions, operators, nesting, common issues.

    Args:
        query: PromQL expression to analyze
    """
    operators = re.findall(r'\b(and|or|unless|on|by|without|group_left|group_right|bool)\b', query)
    functions = re.findall(r'(\w+)\s*\(', query)
    max_nest = cur = 0
    for ch in query:
        if ch == '(':
            cur += 1; max_nest = max(max_nest, cur)
        elif ch == ')':
            cur -= 1
    score  = round(len(query) / 100 + len(operators) * 2 + len(functions) * 3 + max_nest * 4, 2)
    rating = "Simple" if score < 5 else "Moderate" if score < 15 else "Complex" if score < 30 else "Very Complex"
    issues = []
    if ".*" in query:
        issues.append("'.*' wildcard is expensive — prefer specific regex")
    if max_nest > 5:
        issues.append(f"High nesting ({max_nest}) — consider recording rules")
    if re.search(r'(sum|count|avg|max|min)\s*\(', query) and not re.search(r'\b(by|without)\b', query):
        issues.append("Aggregation without 'by'/'without' — all labels are dropped")
    return json.dumps({"query": query, "complexity_score": score, "complexity_rating": rating,
                       "operators": operators, "functions": functions, "nesting_level": max_nest,
                       "issues": issues}, indent=2)


@mcp.tool()
def suggest_promql_for(
    scenario: str,
    pod_name: str = "",
    namespace: str = "",
    node_name: str = "",
) -> str:
    """
    Get ready-to-use PromQL examples for common troubleshooting scenarios.
    Call this to LEARN what queries exist before running them.

    Args:
        scenario:  One of: pod_cpu, pod_memory, pod_restarts, pod_oom, pod_crash,
                   node_cpu, node_memory, node_disk, cluster_alerts, etcd,
                   apiserver, network_errors, pvc_full, hpa_pressure
        pod_name:  Optional — pre-fills pod label
        namespace: Optional — pre-fills namespace label
        node_name: Optional — pre-fills instance/node label
    """
    p  = f'pod="{pod_name}"'      if pod_name  else ""
    ns = f'namespace="{namespace}"' if namespace else ""
    nd = f'instance=~"{node_name}.*"' if node_name else 'instance=~".*"'
    pns_parts = [x for x in [p, ns] if x]
    pns = ",".join(pns_parts)

    templates: dict[str, dict] = {
        "pod_cpu": {
            "description": "CPU usage in cores for a pod",
            "recommended_tool": f'get_pod_cpu(pod_name="{pod_name or "POD_NAME"}", namespace="{namespace or "NAMESPACE"}")',
            "instant_query": f'sum(rate(container_cpu_usage_seconds_total{{{pns},container!=""}}[5m])) by (pod, container)' if pns else 'sum(rate(container_cpu_usage_seconds_total{container!=""}[5m])) by (pod, container)',
            "tip": "Divide result by CPU limit to get utilization percentage",
        },
        "pod_memory": {
            "description": "Memory working-set bytes for a pod",
            "recommended_tool": f'get_pod_memory(pod_name="{pod_name or "POD_NAME"}", namespace="{namespace or "NAMESPACE"}")',
            "instant_query": f'sum(container_memory_working_set_bytes{{{pns},container!=""}}) by (pod, container)' if pns else 'sum(container_memory_working_set_bytes{container!=""}) by (pod, container)',
            "tip": "1 GiB = 1073741824 bytes",
        },
        "pod_restarts": {
            "description": "Container restart count over 1 hour",
            "recommended_tool": f'get_pod_restarts(pod_name="{pod_name or "POD_NAME"}", namespace="{namespace or "NAMESPACE"}")',
            "instant_query": f'increase(kube_pod_container_status_restarts_total{{{pns}}}[1h])' if pns else 'increase(kube_pod_container_status_restarts_total[1h])',
            "tip": "Value > 0 means pod restarted in the last hour",
        },
        "pod_oom": {
            "description": "Pods killed by OOM killer in the last hour",
            "recommended_tool": f'get_oomkilled_pods(namespace="{namespace or ""}")',
            "instant_query": f'increase(kube_pod_container_status_last_terminated_reason{{reason="OOMKilled"{("," + ns) if ns else ""}}}[1h]) > 0',
            "tip": "Fix: increase memory limits or optimize app memory usage",
        },
        "pod_crash": {
            "description": "Pods currently in CrashLoopBackOff",
            "recommended_tool": f'get_crashloopbackoff_pods(namespace="{namespace or ""}")',
            "instant_query": f'kube_pod_container_status_waiting_reason{{reason="CrashLoopBackOff"{("," + ns) if ns else ""}}} == 1',
        },
        "node_cpu": {
            "description": "Node CPU usage percentage",
            "recommended_tool": f'get_node_cpu(node_name="{node_name or "NODE_NAME"}")',
            "instant_query": f'100 - (avg by(instance)(rate(node_cpu_seconds_total{{mode="idle",{nd}}}[5m])) * 100)',
            "tip": "> 80% sustained is high, investigate processes",
        },
        "node_memory": {
            "description": "Node memory used percentage",
            "recommended_tool": f'get_node_memory(node_name="{node_name or "NODE_NAME"}")',
            "instant_query": f'(1 - node_memory_MemAvailable_bytes{{{nd}}} / node_memory_MemTotal_bytes{{{nd}}}) * 100',
            "tip": "> 90% may cause OOM kills on node",
        },
        "node_disk": {
            "description": "Node root filesystem used percentage",
            "recommended_tool": f'get_node_disk(node_name="{node_name or "NODE_NAME"}")',
            "instant_query": f'(1 - node_filesystem_avail_bytes{{{nd},mountpoint="/"}} / node_filesystem_size_bytes{{{nd},mountpoint="/"}}) * 100',
            "tip": "> 85% is dangerous, > 95% will cause pod evictions",
        },
        "cluster_alerts": {
            "description": "All currently firing alerts",
            "recommended_tool": "get_firing_alerts()",
            "instant_query": 'ALERTS{alertstate="firing"}',
            "tip": "Filter critical: get_firing_alerts(severity='critical')",
        },
        "etcd": {
            "description": "etcd cluster health indicators",
            "recommended_tool": "get_etcd_health()",
            "instant_query": 'etcd_server_has_leader',
            "extra_queries": [
                'increase(etcd_server_leader_changes_seen_total[1h])',
                'etcd_mvcc_db_total_size_in_bytes',
            ],
            "tip": "etcd_server_has_leader == 0 means cluster is broken",
        },
        "apiserver": {
            "description": "Kubernetes API server error and latency",
            "recommended_tool": "get_api_server_health()",
            "instant_query": 'sum(rate(apiserver_request_total{code=~"5.."}[5m])) by (verb)',
            "tip": "Any 5xx rate > 0 needs investigation",
        },
        "network_errors": {
            "description": "Network errors on nodes",
            "recommended_tool": f'get_node_network(node_name="{node_name or "NODE_NAME"}")',
            "instant_query": f'rate(node_network_receive_errs_total{{{nd}}}[5m]) > 0',
            "extra_queries": [f'rate(node_network_transmit_errs_total{{{nd}}}[5m]) > 0'],
        },
        "pvc_full": {
            "description": "PVCs over 80% full",
            "recommended_tool": f'get_pvc_usage(namespace="{namespace or ""}", threshold_percent=80)',
            "instant_query": f'(kubelet_volume_stats_used_bytes{{{ns}}} / kubelet_volume_stats_capacity_bytes{{{ns}}}) * 100 > 80' if ns else '(kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes) * 100 > 80',
        },
        "hpa_pressure": {
            "description": "HPA current vs desired vs max replicas",
            "recommended_tool": f'get_deployment_replica_status(namespace="{namespace or ""}")',
            "instant_query": f'kube_horizontalpodautoscaler_status_current_replicas{{{ns}}}' if ns else 'kube_horizontalpodautoscaler_status_current_replicas',
            "extra_queries": [
                f'kube_horizontalpodautoscaler_spec_max_replicas{{{ns}}}' if ns else 'kube_horizontalpodautoscaler_spec_max_replicas',
            ],
            "tip": "If current == max, HPA is at limit and cannot scale further",
        },
    }

    available = list(templates.keys())
    if scenario not in templates:
        return json.dumps({"error": f"Unknown scenario '{scenario}'",
                           "available_scenarios": available}, indent=2)
    return json.dumps({"scenario": scenario, **templates[scenario]}, indent=2)


# ===========================================================================
# 8) COMPOUND TROUBLESHOOTING
# ===========================================================================

@mcp.tool()
async def troubleshoot_pod(  pod_name: str, namespace: str = DEFAULT_NS) -> str:
    """
    Complete pod diagnosis: kubectl describe + logs + events + all Prometheus metrics.
    Best single tool to call when a pod is misbehaving.

    Args:
        pod_name:  Exact pod name
        namespace: Kubernetes namespace
    """
    report: dict[str, Any] = {"pod": pod_name, "namespace": namespace, "timestamp": _now_iso()}
    report["kubectl_describe"] = _kubectl(f"describe pod {pod_name} -n {namespace}")
    report["kubectl_logs"]     = _kubectl(f"logs {pod_name} -n {namespace} --tail=50 2>&1")
    report["kubectl_logs_prev"]= _kubectl(f"logs {pod_name} -n {namespace} --previous --tail=50 2>&1")
    report["kubectl_events"]   = _kubectl(
        f"get events -n {namespace} --field-selector=involvedObject.name={pod_name},type=Warning"
    )
    queries = {
        "cpu_cores":        f'sum(rate(container_cpu_usage_seconds_total{{pod="{pod_name}",namespace="{namespace}",container!=""}}[5m])) by (container)',
        "memory_bytes":     f'sum(container_memory_working_set_bytes{{pod="{pod_name}",namespace="{namespace}",container!=""}}) by (container)',
        "restarts_1h":      f'increase(kube_pod_container_status_restarts_total{{pod="{pod_name}",namespace="{namespace}"}}[1h])',
        "ready":            f'kube_pod_container_status_ready{{pod="{pod_name}",namespace="{namespace}"}}',
        "waiting_reason":   f'kube_pod_container_status_waiting_reason{{pod="{pod_name}",namespace="{namespace}"}}',
        "last_exit_reason": f'kube_pod_container_status_last_terminated_reason{{pod="{pod_name}",namespace="{namespace}"}}',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    report["prometheus_metrics"] = results
    return json.dumps(report, indent=2)


@mcp.tool()
async def troubleshoot_node(  node_name: str) -> str:
    """
    Complete node diagnosis: kubectl describe + pod list + top + Prometheus metrics.

    Args:
        PROMETHEUS_URL: URL of the Prometheus instance
        node_name: Node name as shown by 'kubectl get nodes'
    """
    report: dict[str, Any] = {"node": node_name, "timestamp": _now_iso()}
    report["kubectl_describe"] = _kubectl(f"describe node {node_name}")
    report["pods_on_node"]     = _kubectl(f"get pods --all-namespaces --field-selector=spec.nodeName={node_name}")
    report["kubectl_top"]      = _kubectl(f"top node {node_name} --no-headers 2>&1")
    queries = {
        "cpu_pct":    f'100 - (avg by(instance)(rate(node_cpu_seconds_total{{mode="idle",instance=~"{node_name}.*"}}[5m])) * 100)',
        "mem_pct":    f'(1 - node_memory_MemAvailable_bytes{{instance=~"{node_name}.*"}} / node_memory_MemTotal_bytes{{instance=~"{node_name}.*"}}) * 100',
        "disk_pct":   f'(1 - node_filesystem_avail_bytes{{instance=~"{node_name}.*",mountpoint="/"}} / node_filesystem_size_bytes{{instance=~"{node_name}.*",mountpoint="/"}}) * 100',
        "load5":      f'node_load5{{instance=~"{node_name}.*"}}',
        "net_rx_bps": f'sum(rate(node_network_receive_bytes_total{{instance=~"{node_name}.*",device!~"lo"}}[5m]))',
        "net_tx_bps": f'sum(rate(node_network_transmit_bytes_total{{instance=~"{node_name}.*",device!~"lo"}}[5m]))',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    report["prometheus_metrics"] = results
    return json.dumps(report, indent=2)


@mcp.tool()
async def troubleshoot_namespace( namespace: str) -> str:
    """
    Namespace-level diagnosis: pods, deployments, services, events, HPA, quotas + Prometheus.

    Args:
        namespace: Target namespace
    """
    report: dict[str, Any] = {"namespace": namespace, "timestamp": _now_iso()}
    report["pods"]        = _kubectl(f"get pods -n {namespace} -o wide")
    report["deployments"] = _kubectl(f"get deployments -n {namespace}")
    report["services"]    = _kubectl(f"get services -n {namespace}")
    report["hpa"]         = _kubectl(f"get hpa -n {namespace}")
    report["events"]      = _kubectl(f"get events -n {namespace} --sort-by=.lastTimestamp")
    report["quotas"]      = _kubectl(f"describe resourcequota -n {namespace} 2>&1 || echo 'No quotas'")
    queries = {
        "pod_phases":   f'sum(kube_pod_status_phase{{namespace="{namespace}"}}) by (phase)',
        "cpu_total":    f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}",container!=""}}[5m]))',
        "memory_total": f'sum(container_memory_working_set_bytes{{namespace="{namespace}",container!=""}})',
        "restarts_1h":  f'increase(kube_pod_container_status_restarts_total{{namespace="{namespace}"}}[1h]) > 0',
        "crashloop":    f'kube_pod_container_status_waiting_reason{{reason="CrashLoopBackOff",namespace="{namespace}"}} == 1',
    }
    results = dict(zip(queries.keys(), await asyncio.gather(*[_instant(e) for e in queries.values()])))
    report["prometheus_metrics"] = results
    return json.dumps(report, indent=2)


@mcp.tool()
async def cluster_health_snapshot() -> str:
    """
    Full cluster health snapshot — best starting point for any troubleshooting session.
    Covers: node status, unhealthy pods, warning events, firing alerts, top consumers.
    """
    report: dict[str, Any] = {"timestamp": _now_iso()}
    report["nodes"]            = _kubectl("get nodes -o wide")
    report["not_running_pods"] = _kubectl(
        "get pods --all-namespaces --no-headers --field-selector=status.phase!=Running,status.phase!=Succeeded 2>&1 | head -50"
    )
    report["warning_events"]   = _kubectl(
        "get events --all-namespaces --field-selector=type=Warning --sort-by=.lastTimestamp"
    )
    report["top_nodes"]        = _kubectl("top nodes --no-headers 2>&1")
    prom_queries = {
        "firing_alerts":    "ALERTS{alertstate='firing'}",
        "node_cpu_pct":     "sort_desc(100 - (avg by(instance)(rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100))",
        "node_mem_pct":     "sort_desc((1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)",
        "pod_restarts_5m":  "sort_desc(increase(kube_pod_container_status_restarts_total[5m])) > 0",
        "oomkilled_30m":    "increase(kube_pod_container_status_last_terminated_reason{reason='OOMKilled'}[30m]) > 0",
        "crashloop_pods":   "kube_pod_container_status_waiting_reason{reason='CrashLoopBackOff'} == 1",
        "pending_pods":     "kube_pod_status_phase{phase='Pending'} == 1",
        "pvc_above_85pct":  "kubelet_volume_stats_used_bytes / kubelet_volume_stats_capacity_bytes > 0.85",
        "api_errors_5m":    "sum(rate(apiserver_request_total{code=~'5..'}[5m])) by (verb) > 0",
        "etcd_has_leader":  "etcd_server_has_leader",
    }
    results = dict(zip(prom_queries.keys(), await asyncio.gather(*[_instant(e) for e in prom_queries.values()])))
    report["prometheus"] = results
    return json.dumps(report, indent=2)


@mcp.tool()
async def get_top_resource_consumers(
    resource: str = "cpu",
    namespace: str = "all",
    top_n: int = 10,
) -> str:
    """
    Return the top N pods by CPU or memory consumption.

    Args:
        resource:  'cpu' or 'memory'
        namespace: Namespace or 'all'
        top_n:     Number of results (default 10)
    """
    ns   = "--all-namespaces" if namespace == "all" else f"-n {namespace}"
    sort = "--sort-by=cpu" if resource == "cpu" else "--sort-by=memory"
    out  = _kubectl(f"top pods {ns} {sort} --no-headers")
    return "\n".join(out.splitlines()[:top_n])


# ===========================================================================
# Entrypoint
# ===========================================================================

if __name__ == "__main__":
    logger.info(f"Starting Prometheus PromQL MCP | transport={MCP_TRANSPORT} host={MCP_HOST} port={MCP_PORT}")
    logger.info(f"Prometheus URL: {PROMETHEUS_URL} | Default namespace: {DEFAULT_NS}")

    if MCP_TRANSPORT == "stdio":
        mcp.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run(mcp.http_app(), host=MCP_HOST, port=MCP_PORT)