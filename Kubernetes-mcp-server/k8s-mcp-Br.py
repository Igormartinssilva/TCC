import sys
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import Optional
import subprocess
import logging
import json
import re
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Create an MCP server
mcp = FastMCP("k8s")

# ─────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────
def run_kubectl(command: list, timeout: int = 60) -> str:  # Aumentado de 30 para 60
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
    """Converte valores como 250m, 1Gi, 512Mi para float (CPU em cores, memória em MiB)"""
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
    return {"status": "ok", "message": "Kubernetes Observability MCP v2 is running", "timestamp": now_iso()}


# ─────────────────────────────────────────────
# 1. BÁSICO (herdado + melhorado)
# ─────────────────────────────────────────────

@mcp.tool()
def list_pods(namespace: str = "default"):
    """Lista pods em um namespace com status detalhado"""
    logger.info(f"Listando pods no namespace {namespace}")
    output = run_kubectl(["kubectl", "get", "pods", "-n", namespace, "-o", "wide"])
    return {"output": output, "namespace": namespace, "timestamp": now_iso()}


@mcp.tool()
def get_pod_logs(pod_name: str, namespace: str = "default", lines: int = 50, container: Optional[str] = None):
    """Retorna logs de um pod (suporta múltiplos containers)"""
    logger.info(f"Logs do pod {pod_name}")
    cmd = ["kubectl", "logs", pod_name, "-n", namespace, f"--tail={lines}"]
    if container:
        cmd += ["-c", container]
    output = run_kubectl(cmd)
    return {"output": output, "pod": pod_name, "namespace": namespace, "lines": lines}


@mcp.tool()
def describe_pod(pod_name: str, namespace: str = "default"):
    """Descreve um pod com todos os eventos e condições"""
    logger.info(f"Describe pod {pod_name}")
    output = run_kubectl(["kubectl", "describe", "pod", pod_name, "-n", namespace])
    return {"output": output}


@mcp.tool()
def cluster_info():
    """Informações gerais do cluster"""
    output = run_kubectl(["kubectl", "cluster-info"])
    version = run_kubectl(["kubectl", "version", "--short"])
    nodes = run_kubectl(["kubectl", "get", "nodes", "-o", "wide"])
    return {"cluster_info": output, "version": version, "nodes": nodes, "timestamp": now_iso()}


@mcp.tool()
def metrics():
    """Métricas atuais de CPU e memória dos nós"""
    nodes = run_kubectl(["kubectl", "top", "nodes"])
    pods = run_kubectl(["kubectl", "top", "pods", "--all-namespaces"])
    return {"nodes": nodes, "pods": pods, "timestamp": now_iso()}


# ─────────────────────────────────────────────
# 2. SÉRIES TEMPORAIS DE PODS
# ─────────────────────────────────────────────

@mcp.tool()
def get_pod_resource_history(namespace: str = "default", hours: int = 1):
    """
    Snapshot atual de CPU/memória de todos os pods no namespace.
    Retorna dados estruturados para análise de séries temporais.
    Inclui limites, requests e uso atual para detectar throttling/OOM risk.
    """
    logger.info(f"Resource history para namespace {namespace}")

    # Uso atual via top
    top_raw = run_kubectl(["kubectl", "top", "pods", "-n", namespace, "--no-headers"])

    # Requests e limits via JSON
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
            "throttle_risk=high significa uso acima de 85% do limit de CPU — "
            "pod provavelmente sofre throttling. OOM risk quando mem_usage se aproxima de mem_limit."
        )
    }


# ─────────────────────────────────────────────
# 3. HISTÓRICO DE RESTARTS (proxy de instabilidade)
# ─────────────────────────────────────────────

@mcp.tool()
def get_restart_timeline(namespace: str = "default"):
    """
    Retorna histórico de restarts de todos os pods no namespace.
    Pods com muitos restarts indicam crashloops — padrão importante em séries temporais.
    """
    logger.info(f"Restart timeline para {namespace}")
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
            "exit_code=137 = OOMKill (sem memória). "
            "exit_code=1 = erro de aplicação. "
            "exit_code=143 = SIGTERM (shutdown normal ou eviction)."
        )
    }


# ─────────────────────────────────────────────
# 4. EVENTOS DO CLUSTER (correlação com picos)
# ─────────────────────────────────────────────

@mcp.tool()
def get_events_timeline(namespace: str = "default", event_type: Optional[str] = None):
    """
    Lista eventos recentes do cluster ordenados por timestamp.
    Essencial para correlacionar picos de métricas com eventos (deploys, OOMKills, evictions).
    event_type: 'Warning' ou 'Normal'
    """
    logger.info(f"Events timeline para {namespace}")
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
            "Correlacione 'last_time' dos warnings com picos de CPU/memória. "
            "Razões críticas: OOMKilling, Evicted, BackOff, FailedScheduling."
        )
    }


# ─────────────────────────────────────────────
# 5. HPA — HISTÓRICO DE ESCALONAMENTO
# ─────────────────────────────────────────────

@mcp.tool()
def get_hpa_status(namespace: str = "default"):
    """
    Retorna status atual dos HPAs (Horizontal Pod Autoscalers).
    Mostra réplicas atuais vs mín/máx e métricas que disparam o scale.
    """
    logger.info(f"HPA status para {namespace}")
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
            "scaling_pressure=saturated_at_max indica que o HPA quer mais réplicas mas já atingiu o limite — "
            "provável gargalo de capacidade. Verifique last_scale_time para ver quando foi o último evento de scale."
        )
    }


# ─────────────────────────────────────────────
# 6. PRESSÃO DE NÓS
# ─────────────────────────────────────────────

@mcp.tool()
def get_node_pressure():
    """
    Verifica condições de pressão nos nós (MemoryPressure, DiskPressure, PIDPressure).
    Fundamental para entender evictions e instabilidade em séries temporais.
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
            "MemoryPressure=True causa eviction de pods — correlacione com eventos 'Evicted'. "
            "DiskPressure pode causar falhas de pull de imagem e escrita de logs."
        )
    }


# ─────────────────────────────────────────────
# 7. ANÁLISE DE PADRÕES E TENDÊNCIAS
# ─────────────────────────────────────────────

@mcp.tool()
def analyze_resource_patterns(namespace: str = "default"):
    """
    Analisa padrões de uso de recursos em todos os pods do namespace.
    Detecta: pods ociosos, pods super-provisionados, pods com risco de OOM/throttling.
    Retorna insights prontos para interpretação da LLM.
    """
    logger.info(f"Analisando padrões para {namespace}")
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
            entry["issue"] = f"Throttling risk: usando {cpu_use} de {lims.get('cpu_limit')} limit"
            throttle_risk.append(entry)
        elif mem_lim_val > 0 and (mem_use_val / mem_lim_val) > 0.85:
            entry["issue"] = f"OOM risk: usando {mem_use} de {lims.get('mem_limit')} limit"
            oom_risk.append(entry)
        elif cpu_req_val > 0 and cpu_use_val < (cpu_req_val * 0.10):
            entry["issue"] = f"Ocioso: usando apenas {cpu_use} com request de {lims.get('cpu_request')}"
            idle.append(entry)
        elif cpu_lim_val > 0 and cpu_req_val > 0 and (cpu_lim_val / max(cpu_req_val, 0.001)) > 10:
            entry["issue"] = f"Super-provisionado: request={lims.get('cpu_request')} limit={lims.get('cpu_limit')}"
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
        recs.append(f"{len(throttle)} pod(s) com risco de CPU throttling — considere aumentar cpu.limit ou otimizar a aplicação.")
    if oom:
        recs.append(f"{len(oom)} pod(s) com risco de OOMKill — aumente memory.limit ou investigue memory leaks.")
    if idle:
        recs.append(f"{len(idle)} pod(s) consumindo <10% do cpu.request — reduza requests para liberar capacidade do cluster.")
    if over_prov:
        recs.append(f"{len(over_prov)} pod(s) super-provisionados — ratio request/limit muito alto, ajuste para melhor bin-packing.")
    if not recs:
        recs.append("Recursos parecem bem dimensionados. Continue monitorando.")
    return recs


# ─────────────────────────────────────────────
# 8. CORRELAÇÃO EVENTO + MÉTRICA (Root Cause)
# ─────────────────────────────────────────────

@mcp.tool()
def correlate_events_and_resources(namespace: str = "default"):
    """
    Correlaciona eventos de Warning com uso atual de recursos.
    Retorna uma visão consolidada para root cause analysis — o 'painel de investigação' da LLM.
    """
    logger.info(f"Correlacionando eventos e recursos em {namespace}")

    # Coleta paralela
    events_data = get_events_timeline(namespace=namespace, event_type="Warning")
    resources_data = get_pod_resource_history(namespace=namespace)
    restarts_data = get_restart_timeline(namespace=namespace)

    # Mapa de pods com problemas
    problem_pods = set()
    for ev in events_data.get("events", []):
        if ev["type"] == "Warning":
            problem_pods.add(ev["object"])
    for pod in restarts_data.get("pods", []):
        if pod["severity"] in ("warning", "critical"):
            problem_pods.add(pod["pod"])

    # Correlação
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
            "Este endpoint é o ponto de partida para root cause analysis. "
            "Pods com restart_count alto + OOMKill = memory leak ou limit muito baixo. "
            "Pods com throttle_risk=high + BackOff events = sobrecarga de CPU. "
            "Correlacione 'time' dos warnings com horários de picos reportados."
        )
    }


# ─────────────────────────────────────────────
# 9. DEPLOYMENTS E ROLLOUTS
# ─────────────────────────────────────────────

@mcp.tool()
def get_deployment_status(namespace: str = "default"):
    """
    Status de todos os deployments: réplicas disponíveis, rollout em progresso, imagens em uso.
    Útil para correlacionar deploys com mudanças em séries temporais.
    """
    logger.info(f"Deployment status para {namespace}")
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
            "rollout_in_progress=True durante um pico de erros indica que um deploy pode ser a causa raiz. "
            "Verifique 'revision' e correlacione com timestamps de eventos."
        )
    }


# ─────────────────────────────────────────────
# 10. PAINEL COMPLETO (snapshot diagnóstico)
# ─────────────────────────────────────────────


