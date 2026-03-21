import os
import sys
from fastmcp import FastMCP
from pydantic import BaseModel
from typing import Optional
import subprocess
import logging
import json
import re
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("k8s_mcp")

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def run_kubectl(command: list, timeout: int = 60) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return result.stderr
        return result.stdout
    except subprocess.TimeoutExpired:
        return "ERROR: kubectl command timed out after 60s"
    except Exception as e:
        return str(e)


def run_kubectl_json(command: list) -> dict:
    command += ["-o", "json"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"error": result.stderr}
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}


def parse_resource_value(value: str) -> float:
    """Converts values like 250m, 1Gi, 512Mi to float (CPU in cores, memory in MiB)"""
    if not value or value == "<unknown>":
        return 0.0
    if value.endswith("m"):
        return float(value[:-1]) / 1000
    if value.endswith("Ki"):
        return float(value[:-2]) / 1024
    if value.endswith("Mi"):
        return float(value[:-2])
    if value.endswith("Gi"):
        return float(value[:-2]) * 1024
    try:
        return float(value)
    except Exception:
        return 0.0


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@mcp.tool()
def root():
    """Health check — returns server status"""
    return {"status": "ok", "message": "Kubernetes Observability MCP v2 is running", "timestamp": now_iso()}


# ─────────────────────────────────────────────
# 1. BASIC
# ─────────────────────────────────────────────

@mcp.tool()
def list_pods(namespace: str = "default"):
    """List pods in a namespace with detailed status"""
    logger.info(f"Listing pods in namespace {namespace}")
    output = run_kubectl(["kubectl", "get", "pods", "-n", namespace, "-o", "wide"])
    return {"output": output, "namespace": namespace, "timestamp": now_iso()}


@mcp.tool()
def get_pod_logs(pod_name: str, namespace: str = "default", lines: int = 50, container: Optional[str] = None):
    """Returns logs from a pod (supports multiple containers)"""
    logger.info(f"Fetching logs for pod {pod_name}")
    cmd = ["kubectl", "logs", pod_name, "-n", namespace, f"--tail={lines}"]
    if container:
        cmd += ["-c", container]
    output = run_kubectl(cmd)
    return {"output": output, "pod": pod_name, "namespace": namespace, "lines": lines}


@mcp.tool()
def describe_pod(pod_name: str, namespace: str = "default"):
    """Describes a pod with all events and conditions"""
    logger.info(f"Describing pod {pod_name}")
    output = run_kubectl(["kubectl", "describe", "pod", pod_name, "-n", namespace])
    return {"output": output}


@mcp.tool()
def cluster_info():
    """General cluster information"""
    output = run_kubectl(["kubectl", "cluster-info"])
    version = run_kubectl(["kubectl", "version", "--short"])
    nodes = run_kubectl(["kubectl", "get", "nodes", "-o", "wide"])
    return {"cluster_info": output, "version": version, "nodes": nodes, "timestamp": now_iso()}


@mcp.tool()
def metrics():
    """Current CPU and memory metrics for nodes and pods"""
    nodes = run_kubectl(["kubectl", "top", "nodes"])
    pods = run_kubectl(["kubectl", "top", "pods", "--all-namespaces"])
    return {"nodes": nodes, "pods": pods, "timestamp": now_iso()}


# ─────────────────────────────────────────────
# 2. POD RESOURCE SNAPSHOTS
# ─────────────────────────────────────────────

@mcp.tool()
def get_pod_resource_history(namespace: str = "default", hours: int = 1):
    """
    Current CPU/memory snapshot for all pods in the namespace.
    Returns structured data for time-series analysis.
    Includes limits, requests, and current usage to detect throttling/OOM risk.
    """
    logger.info(f"Resource history for namespace {namespace}")

    top_raw = run_kubectl(["kubectl", "top", "pods", "-n", namespace, "--no-headers"])
    pods_json = run_kubectl_json(["kubectl", "get", "pods", "-n", namespace])

    usage_map = {}
    for line in top_raw.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            usage_map[parts[0]] = {"cpu_usage": parts[1], "mem_usage": parts[2]}

    result = []
    if "items" in pods_json:
        for pod in pods_json["items"]:
            pod_name = pod["metadata"]["name"]
            status = pod.get("status", {}).get("phase", "Unknown")
            containers = pod["spec"].get("containers", [])

            cpu_req = mem_req = cpu_lim = mem_lim = "N/A"
            for c in containers:
                res = c.get("resources", {})
                cpu_req = res.get("requests", {}).get("cpu", "N/A")
                mem_req = res.get("requests", {}).get("memory", "N/A")
                cpu_lim = res.get("limits", {}).get("cpu", "N/A")
                mem_lim = res.get("limits", {}).get("memory", "N/A")

            usage = usage_map.get(pod_name, {})
            cpu_use_val = parse_resource_value(usage.get("cpu_usage", "0"))
            cpu_lim_val = parse_resource_value(cpu_lim)

            throttle_risk = "unknown"
            if cpu_lim_val > 0:
                ratio = cpu_use_val / cpu_lim_val
                throttle_risk = "high" if ratio > 0.85 else "medium" if ratio > 0.60 else "low"

            result.append({
                "pod": pod_name,
                "status": status,
                "cpu_usage": usage.get("cpu_usage", "N/A"),
                "mem_usage": usage.get("mem_usage", "N/A"),
                "cpu_request": cpu_req,
                "cpu_limit": cpu_lim,
                "mem_request": mem_req,
                "mem_limit": mem_lim,
                "throttle_risk": throttle_risk,
                "snapshot_time": now_iso()
            })

    return {
        "namespace": namespace,
        "pod_count": len(result),
        "resources": result,
        "interpretation_hint": (
            "throttle_risk=high means usage is above 85% of CPU limit — pod likely suffers throttling. "
            "OOM risk when mem_usage approaches mem_limit."
        )
    }


# ─────────────────────────────────────────────
# 3. RESTART HISTORY
# ─────────────────────────────────────────────

@mcp.tool()
def get_restart_timeline(namespace: str = "default"):
    """
    Returns restart history for all pods in the namespace.
    Pods with many restarts indicate crashloops — important pattern for time-series analysis.
    """
    logger.info(f"Restart timeline for namespace {namespace}")
    pods_json = run_kubectl_json(["kubectl", "get", "pods", "-n", namespace])

    restarts = []
    if "items" in pods_json:
        for pod in pods_json["items"]:
            name = pod["metadata"]["name"]
            phase = pod.get("status", {}).get("phase", "Unknown")
            container_statuses = pod.get("status", {}).get("containerStatuses", [])

            total_restarts = 0
            last_state_info = []
            for cs in container_statuses:
                r = cs.get("restartCount", 0)
                total_restarts += r
                last = cs.get("lastState", {}).get("terminated", {})
                if last:
                    last_state_info.append({
                        "container": cs.get("name"),
                        "exit_code": last.get("exitCode"),
                        "reason": last.get("reason"),
                        "finished_at": last.get("finishedAt")
                    })

            severity = "ok"
            if total_restarts >= 10:
                severity = "critical"
            elif total_restarts >= 3:
                severity = "warning"

            restarts.append({
                "pod": name,
                "phase": phase,
                "total_restarts": total_restarts,
                "severity": severity,
                "last_termination": last_state_info
            })

    restarts.sort(key=lambda x: x["total_restarts"], reverse=True)

    return {
        "namespace": namespace,
        "timestamp": now_iso(),
        "pods": restarts,
        "summary": {
            "critical_pods": sum(1 for p in restarts if p["severity"] == "critical"),
            "warning_pods": sum(1 for p in restarts if p["severity"] == "warning"),
            "healthy_pods": sum(1 for p in restarts if p["severity"] == "ok"),
        },
        "interpretation_hint": (
            "exit_code=137 = OOMKill (out of memory). "
            "exit_code=1 = application error. "
            "exit_code=143 = SIGTERM (normal shutdown or eviction)."
        )
    }


# ─────────────────────────────────────────────
# 4. CLUSTER EVENTS
# ─────────────────────────────────────────────

@mcp.tool()
def get_events_timeline(namespace: str = "default", event_type: Optional[str] = None):
    """
    Lists recent cluster events sorted by timestamp.
    Essential for correlating metric spikes with events (deploys, OOMKills, evictions).
    event_type: 'Warning' or 'Normal'
    """
    logger.info(f"Events timeline for namespace {namespace}")
    events_json = run_kubectl_json(["kubectl", "get", "events", "-n", namespace, "--sort-by=.lastTimestamp"])

    events = []
    if "items" in events_json:
        for ev in events_json["items"]:
            etype = ev.get("type", "")
            if event_type and etype != event_type:
                continue

            events.append({
                "type": etype,
                "reason": ev.get("reason", ""),
                "message": ev.get("message", ""),
                "object": ev.get("involvedObject", {}).get("name", ""),
                "kind": ev.get("involvedObject", {}).get("kind", ""),
                "count": ev.get("count", 1),
                "first_time": ev.get("firstTimestamp", ""),
                "last_time": ev.get("lastTimestamp", ""),
            })

    warnings = [e for e in events if e["type"] == "Warning"]
    return {
        "namespace": namespace,
        "total_events": len(events),
        "warning_count": len(warnings),
        "events": events,
        "top_warnings": warnings[:10],
        "interpretation_hint": (
            "Correlate 'last_time' of warnings with CPU/memory spikes. "
            "Critical reasons: OOMKilling, Evicted, BackOff, FailedScheduling."
        )
    }


# ─────────────────────────────────────────────
# 5. HPA STATUS
# ─────────────────────────────────────────────

@mcp.tool()
def get_hpa_status(namespace: str = "default"):
    """
    Returns current HPA (Horizontal Pod Autoscaler) status.
    Shows current vs min/max replicas and scaling metrics.
    """
    logger.info(f"HPA status for namespace {namespace}")
    hpa_json = run_kubectl_json(["kubectl", "get", "hpa", "-n", namespace])

    hpas = []
    if "items" in hpa_json:
        for hpa in hpa_json["items"]:
            name = hpa["metadata"]["name"]
            spec = hpa.get("spec", {})
            status = hpa.get("status", {})

            current = status.get("currentReplicas", 0)
            desired = status.get("desiredReplicas", 0)
            minr = spec.get("minReplicas", 1)
            maxr = spec.get("maxReplicas", 1)

            at_max = current >= maxr
            scaling_pressure = "none"
            if desired > current:
                scaling_pressure = "scaling_up"
            elif desired < current:
                scaling_pressure = "scaling_down"
            elif at_max:
                scaling_pressure = "saturated_at_max"

            metrics_status = []
            for m in status.get("currentMetrics", []):
                mtype = m.get("type", "")
                if mtype == "Resource":
                    r = m.get("resource", {})
                    metrics_status.append({
                        "metric": r.get("name"),
                        "current": r.get("current", {}).get("averageUtilization"),
                        "target": None
                    })

            hpas.append({
                "name": name,
                "min_replicas": minr,
                "max_replicas": maxr,
                "current_replicas": current,
                "desired_replicas": desired,
                "scaling_pressure": scaling_pressure,
                "at_maximum": at_max,
                "current_metrics": metrics_status,
                "last_scale_time": status.get("lastScaleTime", "N/A")
            })

    return {
        "namespace": namespace,
        "timestamp": now_iso(),
        "hpas": hpas,
        "saturated_hpas": [h for h in hpas if h["at_maximum"]],
        "interpretation_hint": (
            "scaling_pressure=saturated_at_max means the HPA wants more replicas but has hit its limit — "
            "likely a capacity bottleneck. Check last_scale_time to see when the last scale event occurred."
        )
    }


# ─────────────────────────────────────────────
# 6. NODE PRESSURE
# ─────────────────────────────────────────────

@mcp.tool()
def get_node_pressure():
    """
    Checks node pressure conditions (MemoryPressure, DiskPressure, PIDPressure).
    Essential for understanding evictions and time-series instability.
    """
    logger.info("Node pressure check")
    nodes_json = run_kubectl_json(["kubectl", "get", "nodes"])
    top_raw = run_kubectl(["kubectl", "top", "nodes", "--no-headers"])

    top_map = {}
    for line in top_raw.strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            top_map[parts[0]] = {
                "cpu_usage": parts[1],
                "cpu_pct": parts[2],
                "mem_usage": parts[3],
                "mem_pct": parts[4]
            }

    nodes = []
    if "items" in nodes_json:
        for node in nodes_json["items"]:
            name = node["metadata"]["name"]
            conditions = node.get("status", {}).get("conditions", [])

            pressures = {}
            ready = "Unknown"
            for cond in conditions:
                ctype = cond.get("type", "")
                status_val = cond.get("status", "False")
                if ctype == "Ready":
                    ready = status_val
                elif ctype in ("MemoryPressure", "DiskPressure", "PIDPressure"):
                    pressures[ctype] = status_val == "True"

            has_pressure = any(pressures.values())
            usage = top_map.get(name, {})

            nodes.append({
                "node": name,
                "ready": ready,
                "memory_pressure": pressures.get("MemoryPressure", False),
                "disk_pressure": pressures.get("DiskPressure", False),
                "pid_pressure": pressures.get("PIDPressure", False),
                "has_any_pressure": has_pressure,
                "cpu_usage": usage.get("cpu_usage", "N/A"),
                "cpu_pct": usage.get("cpu_pct", "N/A"),
                "mem_usage": usage.get("mem_usage", "N/A"),
                "mem_pct": usage.get("mem_pct", "N/A"),
            })

    return {
        "timestamp": now_iso(),
        "nodes": nodes,
        "nodes_with_pressure": [n for n in nodes if n["has_any_pressure"]],
        "nodes_not_ready": [n for n in nodes if n["ready"] != "True"],
        "interpretation_hint": (
            "MemoryPressure=True causes pod eviction — correlate with 'Evicted' events. "
            "DiskPressure can cause image pull failures and log write errors."
        )
    }


# ─────────────────────────────────────────────
# 7. RESOURCE PATTERN ANALYSIS
# ─────────────────────────────────────────────

@mcp.tool()
def analyze_resource_patterns(namespace: str = "default"):
    """
    Analyzes resource usage patterns for all pods in the namespace.
    Detects: idle pods, over-provisioned pods, OOM/throttling risk pods.
    Returns insights ready for LLM interpretation.
    """
    logger.info(f"Analyzing resource patterns for namespace {namespace}")
    top_raw = run_kubectl(["kubectl", "top", "pods", "-n", namespace, "--no-headers"])
    pods_json = run_kubectl_json(["kubectl", "get", "pods", "-n", namespace])

    limits_map = {}
    if "items" in pods_json:
        for pod in pods_json["items"]:
            name = pod["metadata"]["name"]
            containers = pod["spec"].get("containers", [])
            cpu_lim = mem_lim = cpu_req = mem_req = "0"
            for c in containers:
                res = c.get("resources", {})
                cpu_lim = res.get("limits", {}).get("cpu", "0")
                mem_lim = res.get("limits", {}).get("memory", "0")
                cpu_req = res.get("requests", {}).get("cpu", "0")
                mem_req = res.get("requests", {}).get("memory", "0")
            limits_map[name] = {
                "cpu_limit": cpu_lim, "mem_limit": mem_lim,
                "cpu_request": cpu_req, "mem_request": mem_req
            }

    over_provisioned = []
    oom_risk = []
    throttle_risk = []
    idle = []
    healthy = []

    for line in top_raw.strip().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        pod_name, cpu_use, mem_use = parts[0], parts[1], parts[2]
        lims = limits_map.get(pod_name, {})

        cpu_use_val = parse_resource_value(cpu_use)
        mem_use_val = parse_resource_value(mem_use)
        cpu_lim_val = parse_resource_value(lims.get("cpu_limit", "0"))
        mem_lim_val = parse_resource_value(lims.get("mem_limit", "0"))
        cpu_req_val = parse_resource_value(lims.get("cpu_request", "0"))

        entry = {"pod": pod_name, "cpu_usage": cpu_use, "mem_usage": mem_use}

        if cpu_lim_val > 0 and (cpu_use_val / cpu_lim_val) > 0.85:
            entry["issue"] = f"Throttling risk: using {cpu_use} of {lims.get('cpu_limit')} limit"
            throttle_risk.append(entry)
        elif mem_lim_val > 0 and (mem_use_val / mem_lim_val) > 0.85:
            entry["issue"] = f"OOM risk: using {mem_use} of {lims.get('mem_limit')} limit"
            oom_risk.append(entry)
        elif cpu_req_val > 0 and cpu_use_val < (cpu_req_val * 0.10):
            entry["issue"] = f"Idle: only using {cpu_use} with request of {lims.get('cpu_request')}"
            idle.append(entry)
        elif cpu_lim_val > 0 and cpu_req_val > 0 and (cpu_lim_val / max(cpu_req_val, 0.001)) > 10:
            entry["issue"] = f"Over-provisioned: request={lims.get('cpu_request')} limit={lims.get('cpu_limit')}"
            over_provisioned.append(entry)
        else:
            healthy.append(entry)

    return {
        "namespace": namespace,
        "timestamp": now_iso(),
        "summary": {
            "total_pods_analyzed": len(throttle_risk) + len(oom_risk) + len(idle) + len(over_provisioned) + len(healthy),
            "throttle_risk_count": len(throttle_risk),
            "oom_risk_count": len(oom_risk),
            "idle_count": len(idle),
            "over_provisioned_count": len(over_provisioned),
            "healthy_count": len(healthy),
        },
        "throttle_risk_pods": throttle_risk,
        "oom_risk_pods": oom_risk,
        "idle_pods": idle,
        "over_provisioned_pods": over_provisioned,
        "recommendations": _generate_recommendations(throttle_risk, oom_risk, idle, over_provisioned)
    }


def _generate_recommendations(throttle, oom, idle, over_prov) -> list:
    recs = []
    if throttle:
        recs.append(f"{len(throttle)} pod(s) at CPU throttling risk — consider increasing cpu.limit or optimizing the application.")
    if oom:
        recs.append(f"{len(oom)} pod(s) at OOMKill risk — increase memory.limit or investigate memory leaks.")
    if idle:
        recs.append(f"{len(idle)} pod(s) consuming <10% of cpu.request — reduce requests to free cluster capacity.")
    if over_prov:
        recs.append(f"{len(over_prov)} over-provisioned pod(s) — request/limit ratio too high, adjust for better bin-packing.")
    if not recs:
        recs.append("Resources appear well-sized. Keep monitoring.")
    return recs


# ─────────────────────────────────────────────
# 8. EVENT + METRIC CORRELATION (Root Cause)
# ─────────────────────────────────────────────

@mcp.tool()
def correlate_events_and_resources(namespace: str = "default"):
    """
    Correlates Warning events with current resource usage.
    Returns a consolidated view for root cause analysis.
    """
    logger.info(f"Correlating events and resources in namespace {namespace}")

    events_data = get_events_timeline(namespace=namespace, event_type="Warning")
    resources_data = get_pod_resource_history(namespace=namespace)
    restarts_data = get_restart_timeline(namespace=namespace)

    problem_pods = set()
    for ev in events_data.get("events", []):
        if ev["type"] == "Warning":
            problem_pods.add(ev["object"])
    for pod in restarts_data.get("pods", []):
        if pod["severity"] in ("warning", "critical"):
            problem_pods.add(pod["pod"])

    correlated = []
    for pod_res in resources_data.get("resources", []):
        pod_name = pod_res["pod"]
        pod_events = [e for e in events_data.get("events", []) if e["object"] == pod_name]
        pod_restarts = next((p for p in restarts_data.get("pods", []) if p["pod"] == pod_name), {})

        if pod_name in problem_pods or pod_res.get("throttle_risk") in ("high", "medium"):
            correlated.append({
                "pod": pod_name,
                "cpu_usage": pod_res.get("cpu_usage"),
                "mem_usage": pod_res.get("mem_usage"),
                "throttle_risk": pod_res.get("throttle_risk"),
                "restart_count": pod_restarts.get("total_restarts", 0),
                "restart_severity": pod_restarts.get("severity", "ok"),
                "last_termination_reason": pod_restarts.get("last_termination", [{}])[0].get("reason") if pod_restarts.get("last_termination") else None,
                "recent_warnings": [
                    {"reason": e["reason"], "message": e["message"][:100], "time": e["last_time"]}
                    for e in pod_events[:5]
                ]
            })

    correlated.sort(key=lambda x: (
        x["restart_count"] * -1,
        0 if x["throttle_risk"] == "high" else 1
    ))

    return {
        "namespace": namespace,
        "timestamp": now_iso(),
        "investigation_summary": {
            "pods_with_issues": len(correlated),
            "total_warnings": events_data.get("warning_count", 0),
            "critical_restart_pods": restarts_data.get("summary", {}).get("critical_pods", 0),
        },
        "correlated_issues": correlated,
        "interpretation_hint": (
            "This endpoint is the starting point for root cause analysis. "
            "Pods with high restart_count + OOMKill = memory leak or limit too low. "
            "Pods with throttle_risk=high + BackOff events = CPU overload. "
            "Correlate 'time' of warnings with reported spike timestamps."
        )
    }


# ─────────────────────────────────────────────
# 9. DEPLOYMENTS & ROLLOUTS
# ─────────────────────────────────────────────

@mcp.tool()
def get_deployment_status(namespace: str = "default"):
    """
    Status of all deployments: available replicas, rollout in progress, images in use.
    Useful for correlating deploys with changes in time-series data.
    """
    logger.info(f"Deployment status for namespace {namespace}")
    dep_json = run_kubectl_json(["kubectl", "get", "deployments", "-n", namespace])

    deployments = []
    if "items" in dep_json:
        for dep in dep_json["items"]:
            name = dep["metadata"]["name"]
            spec = dep.get("spec", {})
            status = dep.get("status", {})
            containers = dep["spec"]["template"]["spec"].get("containers", [])

            desired = spec.get("replicas", 0)
            ready = status.get("readyReplicas", 0)
            available = status.get("availableReplicas", 0)
            updated = status.get("updatedReplicas", 0)

            rollout_in_progress = updated != desired or ready != desired
            images = [c.get("image", "") for c in containers]

            annotations = dep["metadata"].get("annotations", {})
            last_deploy = annotations.get("deployment.kubernetes.io/revision", "N/A")

            deployments.append({
                "name": name,
                "desired_replicas": desired,
                "ready_replicas": ready,
                "available_replicas": available,
                "updated_replicas": updated,
                "rollout_in_progress": rollout_in_progress,
                "health": "degraded" if ready < desired else "healthy",
                "images": images,
                "revision": last_deploy
            })

    return {
        "namespace": namespace,
        "timestamp": now_iso(),
        "deployments": deployments,
        "degraded": [d for d in deployments if d["health"] == "degraded"],
        "rolling_out": [d for d in deployments if d["rollout_in_progress"]],
        "interpretation_hint": (
            "rollout_in_progress=True during an error spike indicates a deploy may be the root cause. "
            "Check 'revision' and correlate with event timestamps."
        )
    }


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    transport = os.getenv("MCP_TRANSPORT", "sse")
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8080"))

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # FastMCP 2.x expõe o app Starlette assim:
        uvicorn.run(mcp.http_app(), host=host, port=port)