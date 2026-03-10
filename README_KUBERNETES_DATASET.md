# 🚀 Guia Completo: Criando Dataset de Kubernetes para Time-LLM

## 📌 Índice
1. [Conceitos Fundamentais](#conceitos)
2. [Métricas que Você Precisa](#métricas)
3. [Ferramentas para Recolher Dados](#ferramentas)
4. [Scripts Práticos](#scripts)
5. [Estrutura do Dataset](#estrutura)
6. [Validação e Limpeza](#validação)
7. [Treinando o Modelo](#treinamento)

---

## <a name="conceitos"></a>1. Conceitos Fundamentais

### Por que um Dataset de Kubernetes?

Time-LLM foi treinado em **dados financeiros e energéticos** (ETTh1 = temperatura de óleo em subestação). Para prever/entender **métricas do seu cluster Kubernetes**, precisamos:

1. **Especializar o modelo** em padrões Kubernetes
2. **Aprender correlações** (CPU ↑ → Memória ↑ → Erro ↑)
3. **Detectar anomalias** (comportamento fora do padrão)

### Diferença: Kubernetes vs ETTh1

| Aspecto | ETTh1 | Kubernetes |
|---------|-------|-----------|
| **Sazonalidade** | Diária (24h) | Diária + Semanal |
| **Features** | 7 (fixas) | 10-50+ (dinâmicas) |
| **Volatilidade** | Baja | Alta (auto-scaling) |
| **Padrões** | Trend claro | Eventos discretos |
| **Frequência** | 1 ponto/hora | 1 ponto/minuto ou 5 minutos |

### O que o Modelo Aprenderá?

```
INPUT: [cpu, memory, disk, network, errors, latency] × últimas 336 horas
                              ↓
                      (Patch Embedding)
                              ↓
                          (GPT-2)
                              ↓
OUTPUT: Predição dos próximos 96 horas (4 dias)
```

**Exemplos de padrões que aprenderá**:
- ✅ Sábado-domingo tem menos tráfego
- ✅ Manhã tem mais requests
- ✅ Quando CPU sobe rapidamente, erro_rate sobe em ~5 minutos
- ✅ Padrão normal vs padrão anômalo

---

## <a name="métricas"></a>2. Métricas que Você Precisa

### 2.1 Métricas Essenciais (Minimum Viable)

Se tiver poucos dados ou quer começar simples, recolha **estes 8 campos**:

```csv
timestamp,cpu_usage,memory_usage,disk_io,network_in,error_rate,latency_p99,pod_restarts
2026-01-01 00:00:00,-45.2,62.1,1024,5120,0.1,125,0
2026-01-01 01:00:00,48.5,64.3,2048,6144,0.15,135,0
```

| Métrica | Fonte | Unidade | O que Significa |
|---------|-------|---------|-----------------|
| **cpu_usage** | `container_cpu_usage_seconds_total` | % (0-100) | Uso de CPU (agregado) |
| **memory_usage** | `container_memory_usage_bytes` | % (0-100) | Uso de memória |
| **disk_io** | `node_disk_io_reads_bytes_total + writes` | MB/s | I/O de disco |
| **network_in** | `container_network_receive_bytes_total` | MB/s | Dados entrados |
| **error_rate** | `http_requests_total{status=~"5.."}` / total | % | Taxa de erros 5xx |
| **latency_p99** | `http_request_duration_seconds` | ms | Latência 99º percentil |
| **pod_restarts** | `container_last_seen` + events | count | Quantos pods restartaram |

### 2.2 Métricas Recomendadas (Mais Completo)

Se quer performance melhor, recolha **12-15 campos**:

```csv
timestamp,cpu_usage,memory_usage,disk_io,network_in,network_out,error_rate,latency_p50,latency_p99,latency_p95,pod_restarts,container_count,queue_depth,gc_pause_ms,requests_per_sec
```

| Métrica Extra | Fonte | Por quê |
|---------------|-------|---------|
| **network_out** | `container_network_transmit_bytes_total` | Identifica outbound requests |
| **latency_p50/p95** | Percentile analysis | Entender distribuição |
| **container_count** | `kube_pod_info` | Quando auto-scaling ativa |
| **queue_depth** | App metrics | Gargalos internos |
| **gc_pause_ms** | JVM metrics (se Java) | Correlação com latência |
| **requests_per_sec** | HTTP total | Carga absoluta |

### 2.3 Métricas Avançadas (Se Quiser Ser Completo)

```csv
...,node_cpu_percent,node_mem_percent,network_drops,network_errors,disk_iops,disk_utilization,context_switches,interrupts,load_average,cache_miss_rate,page_faults
```

---

## <a name="ferramentas"></a>3. Ferramentas para Recolher Dados

### 3.1 Opção A: Prometheus (RECOMENDADA)

**Se seu cluster já tem Prometheus rodando**, é a forma mais fácil.

#### Instalação rápida (se não tiver)

```bash
# Helm (recomendado)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# Validar
kubectl port-forward -n monitoring svc/prometheus-stack-operated 9090:9090
# Abrir: http://localhost:9090
```

#### Queries úteis no Prometheus

```sql
-- CPU Usage (%)
rate(container_cpu_usage_seconds_total[5m]) * 100

-- Memory Usage (%)
(container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100

-- Network In (bytes/s)
rate(container_network_receive_bytes_total[5m])

-- Network Out (bytes/s)
rate(container_network_transmit_bytes_total[5m])

-- Error Rate (%)
(rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])) * 100

-- Latency p99 (ms)
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) * 1000

-- Pod Restarts
increase(container_last_seen[1h])
```

#### Script Python para exportar Prometheus → CSV

```python
# prometheus_export.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

# Configuração
PROMETHEUS_URL = "http://localhost:9090"
START_TIME = (datetime.now() - timedelta(days=60)).isoformat()  # últimos 60 dias
END_TIME = datetime.now().isoformat()
STEP = "1h"  # 1 hora (ajuste para 5m, 10m conforme quiser)

def query_prometheus(query, start_time, end_time, step):
    """Query Prometheus e retorna valores"""
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params = {
        "query": query,
        "start": start_time,
        "end": end_time,
        "step": step
    }
    response = requests.get(url, params=params)
    return response.json()

# Definir queries
queries = {
    "cpu_usage": 'rate(container_cpu_usage_seconds_total{pod!=""}[5m]) * 100',
    "memory_usage": '(container_memory_usage_bytes{pod!=""} / container_spec_memory_limit_bytes{pod!=""}) * 100',
    "network_in": 'rate(container_network_receive_bytes_total{pod!=""}[5m]) / 1024 / 1024',  # MB/s
    "network_out": 'rate(container_network_transmit_bytes_total{pod!=""}[5m]) / 1024 / 1024',
    "error_rate": '(rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])) * 100',
    "latency_p99": 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) * 1000',
    "pod_restarts": 'increase(kube_pod_container_status_restarts_total[1h])',
}

# Recolher dados
print("Recolhendo dados do Prometheus...")
data_dict = {"timestamp": []}

for metric_name, query in queries.items():
    print(f"  Query: {metric_name}...")
    result = query_prometheus(query, START_TIME, END_TIME, STEP)
    
    values = []
    timestamps = []
    
    if result["status"] == "success" and result["data"]["result"]:
        # Agregar todos os valores se houver múltiplas séries
        for series in result["data"]["result"]:
            for ts, value in series["values"]:
                ts_dt = datetime.fromtimestamp(int(ts))
                if ts_dt not in timestamps:
                    timestamps.append(ts_dt)
                # Converter para número
                try:
                    values.append(float(value))
                except:
                    values.append(None)
        
        # Média ou agregação (ajuste conforme necessário)
        if values:
            avg_value = sum(v for v in values if v is not None) / len([v for v in values if v is not None])
            data_dict[metric_name] = [avg_value] * len(timestamps)
    else:
        print(f"    ⚠️ Sem dados para {metric_name}")
        data_dict[metric_name] = [None] * len(data_dict["timestamp"])

# Se for primeira query, preencher timestamps
if not data_dict["timestamp"]:
    result = query_prometheus(list(queries.values())[0], START_TIME, END_TIME, STEP)
    if result["status"] == "success" and result["data"]["result"]:
        for ts, value in result["data"]["result"][0]["values"]:
            data_dict["timestamp"].append(datetime.fromtimestamp(int(ts)))

# Criar DataFrame
df = pd.DataFrame(data_dict)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Salvar
output_path = "./dataset/kubernetes/kubernetes_metrics.csv"
df.to_csv(output_path, index=False)
print(f"✅ Dados salvos em: {output_path}")
print(f"   Total de pontos: {len(df)}")
print(f"   Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
```

**Como usar**:
```bash
# 1. Instalar dependências
pip install requests pandas

# 2. Rodar script
python prometheus_export.py

# 3. Validar CSV criado
head -5 ./dataset/kubernetes/kubernetes_metrics.csv
```

---

### 3.2 Opção B: Kubelet Metrics (Alternativa sem Prometheus)

Se **não tem Prometheus**, pode extrair direto do Kubelet:

```bash
# Conectar ao Kubelet de um node
kubectl debug node/NODE_NAME -it --image=ubuntu

# Dentro do container
curl -k https://localhost:10250/metrics | grep container_
```

---

### 3.3 Opção C: kubectl + Events (Manual)

Se quer recolher manualmente (menos automatizado):

```bash
# ✅ Recolher resource usage
kubectl top nodes > nodes.txt
kubectl top pods -A > pods.txt

# ✅ Recolher events
kubectl get events -A --sort-by='.lastTimestamp' > events.txt

# ✅ Recolher logs de erros
kubectl logs -l app=myapp --tail=1000 | grep ERROR > errors.txt
```

Script Python para agregar:

```python
# extract_kubectl_metrics.py
import subprocess
import json
from datetime import datetime
import pandas as pd

def get_pod_metrics():
    """Get metrics from kubectl top pods"""
    result = subprocess.run(["kubectl", "top", "pods", "-A", "--no-headers"], 
                          capture_output=True, text=True)
    
    pods = []
    for line in result.stdout.strip().split('\n'):
        if line:
            parts = line.split()
            pods.append({
                "namespace": parts[0],
                "name": parts[1],
                "cpu": int(parts[2].replace('m', '')),  # Convert milli-CPU
                "memory": int(parts[3].replace('Mi', ''))  # Convert MiB
            })
    return pods

def get_event_summary():
    """Count recent events/errors"""
    result = subprocess.run(
        ["kubectl", "get", "events", "-A", "--sort-by=.metadata.creationTimestamp"],
        capture_output=True, text=True
    )
    
    events = result.stdout.strip().split('\n')
    error_count = len([e for e in events if 'Error' in e or 'Failed' in e])
    restart_count = len([e for e in events if 'Restarted' in e])
    
    return {
        "error_events": error_count,
        "restart_events": restart_count,
        "total_events": len(events)
    }

# Recolher
pods = get_pod_metrics()
events = get_event_summary()

print(f"Pods: {len(pods)}")
print(f"CPU total: {sum(p['cpu'] for p in pods)} m")
print(f"Memória total: {sum(p['memory'] for p in pods)} Mi")
print(f"Eventos com erro: {events['error_events']}")

# Salvar em arquivo (roda periodicamente via cron)
with open(f"./metrics/{datetime.now().isoformat()}.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "pods": pods,
        "events": events
    }, f)
```

---

## <a name="scripts"></a>4. Scripts Práticos

### 4.1 Script automático: Recolher dados por 60 dias

Este script roda continuamente e salva 1 ponto por hora:

```python
# collect_k8s_metrics.py
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

class KubernetesMetricsCollector:
    def __init__(self, prometheus_url="http://localhost:9090", 
                 output_dir="./dataset/kubernetes/"):
        self.prometheus_url = prometheus_url
        self.output_dir = output_dir
        self.csv_file = os.path.join(output_dir, "kubernetes_metrics.csv")
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Inicializar CSV se não existir
        if not os.path.exists(self.csv_file):
            self._init_csv()
    
    def _init_csv(self):
        """Criar arquivo CSV com headers"""
        df = pd.DataFrame(columns=[
            "timestamp", "cpu_usage", "memory_usage", "disk_io",
            "network_in", "network_out", "error_rate", "latency_p99",
            "pod_restarts"
        ])
        df.to_csv(self.csv_file, index=False)
        print(f"✅ CSV inicializado: {self.csv_file}")
    
    def query_prometheus(self, query):
        """Query Prometheus"""
        url = f"{self.prometheus_url}/api/v1/query"
        params = {"query": query}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            result = response.json()
            
            if result["status"] == "success" and result["data"]["result"]:
                # Retornar valor da primeira série
                value = float(result["data"]["result"][0]["value"][1])
                return value
            else:
                return None
        except Exception as e:
            print(f"❌ Erro ao recolher métrica: {e}")
            return None
    
    def collect_metrics(self):
        """Recolher um ponto de dados"""
        timestamp = datetime.now()
        
        # Define as queries
        metrics = {
            "cpu_usage": 'avg(rate(container_cpu_usage_seconds_total{pod!=""}[5m])) * 100',
            "memory_usage": 'avg(container_memory_usage_bytes{pod!=""} / container_spec_memory_limit_bytes{pod!=""}) * 100 or (avg(container_memory_usage_bytes{pod!=""}) / (1024*1024*1024)) * 100',
            "disk_io": 'avg(rate(node_disk_io_reads_bytes_total[5m]) + rate(node_disk_io_writes_bytes_total[5m])) / 1024 / 1024',
            "network_in": 'avg(rate(container_network_receive_bytes_total{pod!=""}[5m])) / 1024 / 1024',
            "network_out": 'avg(rate(container_network_transmit_bytes_total{pod!=""}[5m])) / 1024 / 1024',
            "error_rate": 'avg((rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])) * 100) or 0',
            "latency_p99": 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) * 1000 or 0',
            "pod_restarts": 'increase(kube_pod_container_status_restarts_total[1h]) or 0',
        }
        
        # Recolher todas as métricas
        data_point = {"timestamp": timestamp}
        print(f"\n📊 Recolhendo métricas em {timestamp.isoformat()}...")
        
        for metric_name, query in metrics.items():
            value = self.query_prometheus(query)
            data_point[metric_name] = value if value is not None else 0
            status = "✅" if value is not None else "⚠️"
            print(f"  {status} {metric_name}: {value}")
        
        # Salvar no CSV
        df_new = pd.DataFrame([data_point])
        df_existing = pd.read_csv(self.csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(self.csv_file, index=False)
        
        print(f"✅ Salvo! Total de pontos: {len(df_combined)}")
        
        return data_point
    
    def run_continuous(self, interval_hours=1):
        """Rodar continuamente a cada N horas"""
        print(f"🚀 Iniciando coleta de métricas a cada {interval_hours} hora(s)...")
        print(f"📁 Salvando em: {self.csv_file}")
        
        while True:
            try:
                self.collect_metrics()
            except Exception as e:
                print(f"❌ Erro na coleta: {e}")
            
            # Aguardar até próximo ciclo
            sleep_seconds = interval_hours * 3600
            print(f"⏳ Próxima coleta em {interval_hours}h...")
            time.sleep(sleep_seconds)

if __name__ == "__main__":
    collector = KubernetesMetricsCollector()
    collector.run_continuous(interval_hours=1)  # Recolher a cada 1 hora
```

**Como usar**:

```bash
# Terminal 1: Iniciar coleta (roda para sempre)
python collect_k8s_metrics.py

# Terminal 2: Verificar progresso
tail -f ./dataset/kubernetes/kubernetes_metrics.csv

# Ou em background (Linux/Mac)
nohup python collect_k8s_metrics.py > collection.log 2>&1 &
```

**Cron schedule** (Linux/Mac - coleta automática a cada 1 hora):

```bash
# Editar crontab
crontab -e

# Adicionar linha:
0 * * * * cd /path/to/project && python collect_k8s_metrics.py
```

---

### 4.2 Script rápido: Recolher 60 dias de dados históricos

Se Prometheus já tem dados históricos, extraia tudo:

```python
# backfill_prometheus_data.py
import requests
import pandas as pd
from datetime import datetime, timedelta

PROMETHEUS_URL = "http://localhost:9090"
END_TIME = datetime.now()
START_TIME = END_TIME - timedelta(days=60)

# Converter para timestamp Unix
start_ts = int(START_TIME.timestamp())
end_ts = int(END_TIME.timestamp())
step = "1h"  # 1 ponto por hora

print(f"⏳ Baixando dados de {START_TIME} a {END_TIME}...")

# Queries para recolher
queries = {
    "cpu_usage": 'avg(rate(container_cpu_usage_seconds_total[5m])) * 100 by (pod)',
    "memory_usage": 'avg(container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100 by (pod)',
    "network_in": 'avg(rate(container_network_receive_bytes_total[5m])) by (pod)',
    "error_rate": '(rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])) * 100',
    "pod_restarts": 'rate(kube_pod_container_status_restarts_total[1h])',
}

# Recolher cada query
all_data = []

for metric_name, query in queries.items():
    print(f"📥 Query: {metric_name}...")
    
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params = {
        "query": query,
        "start": start_ts,
        "end": end_ts,
        "step": step
    }
    
    response = requests.get(url, params=params)
    result = response.json()
    
    if result["status"] == "success":
        for series in result["data"]["result"]:
            for ts, value in series["values"]:
                ts_dt = datetime.fromtimestamp(int(ts))
                all_data.append({
                    "timestamp": ts_dt,
                    metric_name: float(value)
                })
        print(f"  ✅ {len(result['data']['result'])} séries coletadas")
    else:
        print(f"  ❌ Erro ao recolher {metric_name}")

# Agregar por timestamp
df = pd.DataFrame(all_data)
df_agg = df.groupby("timestamp").mean().reset_index()

# Salvar
output_file = "./dataset/kubernetes/kubernetes_metrics_backfill.csv"
df_agg.to_csv(output_file, index=False)

print(f"\n✅ Dados salvos em: {output_file}")
print(f"   Total de pontos: {len(df_agg)}")
print(f"   Período: {df_agg['timestamp'].min()} a {df_agg['timestamp'].max()}")
print(f"\nPrimeiros dados:")
print(df_agg.head())
```

**Como usar**:
```bash
python backfill_prometheus_data.py
```

---

## <a name="estrutura"></a>5. Estrutura do Dataset

### 5.1 Formato CSV esperado

```csv
timestamp,cpu_usage,memory_usage,disk_io,network_in,network_out,error_rate,latency_p99,pod_restarts
2026-01-01 00:00:00,35.2,62.1,512.5,2048.3,1024.1,0.10,125,0
2026-01-01 01:00:00,38.5,64.3,768.2,2560.4,1280.2,0.15,135,0
2026-01-01 02:00:00,42.1,58.9,256.8,4096.5,1536.3,0.08,118,0
2026-01-01 03:00:00,45.3,60.2,512.1,2048.2,1024.0,0.12,128,1
...
```

### 5.2 Intervalo de tempo

| Intervalo | Pontos/dia | Arquivo/60 dias | Recomendação |
|-----------|-----------|-----------------|--------------|
| **1 minuto** | 1,440 | ~215 MB | ❌ Muito grande |
| **5 minutos** | 288 | ~43 MB | ✅ Ideal (muitos padrões) |
| **10 minutos** | 144 | ~22 MB | ✅ Bom (balanço) |
| **1 hora** | 24 | ~1.5 MB | ✅ Simples (menos padrões) |

**Recomendação**: Use **10-60 minutos** para Time-LLM.

### 5.3 Normalizar / Limpar dados

```python
# clean_kubernetes_data.py
import pandas as pd
import numpy as np

# Recolher dados
df = pd.read_csv('./dataset/kubernetes/kubernetes_metrics.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("📊 Validação inicial:")
print(f"  - Linhas: {len(df)}")
print(f"  - Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
print(f"  - Colunas: {list(df.columns)}")

# 1️⃣ Verificar valores faltantes
print("\n🔍 Valores faltantes:")
missing = df.isnull().sum()
print(missing)

# Preencher NaN com interpolação linear (recomendado)
df_filled = df.copy()
numeric_cols = df_filled.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    # Preencher com forward fill + backward fill
    df_filled[col] = df_filled[col].fillna(method='ffill').fillna(method='bfill')
    
    # Se ainda tiver NaN, usar média
    mean_val = df_filled[col].mean()
    df_filled[col] = df_filled[col].fillna(mean_val)

print(f"✅ Valores faltantes preenchidos")

# 2️⃣ Normalizar para 0-100 (percentual)
for col in ['cpu_usage', 'memory_usage', 'error_rate']:
    min_val = df_filled[col].min()
    max_val = df_filled[col].max()
    
    # Clamp to 0-100
    df_filled[col] = df_filled[col].clip(0, 100)
    
    print(f"  - {col}: min={min_val:.2f}, max={max_val:.2f}")

# 3️⃣ Detectar outliers (3 sigma)
print("\n🚨 Outliers detectados (3 sigma):")
for col in numeric_cols:
    mean = df_filled[col].mean()
    std = df_filled[col].std()
    
    outliers = df_filled[
        (df_filled[col] > mean + 3*std) | (df_filled[col] < mean - 3*std)
    ]
    
    if len(outliers) > 0:
        print(f"  - {col}: {len(outliers)} outliers")
        # Substituir outliers pela mediana local (janela de 5 pontos)
        df_filled[col] = df_filled[col].rolling(window=5, center=True).median()

# 4️⃣ Ordenar por timestamp
df_clean = df_filled.sort_values('timestamp').reset_index(drop=True)

# 5️⃣ Salvar
output_file = './dataset/kubernetes/kubernetes_metrics_clean.csv'
df_clean.to_csv(output_file, index=False)

print(f"\n✅ Dados limpos e salvos em: {output_file}")
print(f"\nEstatísticas finais:")
print(df_clean.describe())

# 6️⃣ Plot para visualizar
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 3, idx % 3]
        df_clean.plot(x='timestamp', y=col, ax=ax)
        ax.set_title(col)
        ax.set_xlabel('')
    
    plt.tight_layout()
    plt.savefig('./dataset/kubernetes/metrics_distribution.png', dpi=100)
    print("📊 Gráfico salvo em: ./dataset/kubernetes/metrics_distribution.png")
except Exception as e:
    print(f"⚠️ Não foi possível gerar gráfico: {e}")
```

**Como usar**:
```bash
python clean_kubernetes_data.py
```

---

## <a name="validação"></a>6. Validação e Limpeza

### 6.1 Checklist de validação

Antes de treinar, verifique:

```python
# validate_dataset.py
import pandas as pd
import numpy as np

def validate_kubernetes_dataset(csv_path):
    """Validar dataset antes de treinar"""
    
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("=" * 60)
    print("VALIDAÇÃO DO DATASET KUBERNETES")
    print("=" * 60)
    
    # 1. Tamanho do dataset
    print(f"\n✓ TAMANHO")
    print(f"  Pontos de dados: {len(df)} (recomendado: > 8000)")
    if len(df) < 8000:
        print(f"  ⚠️ Poucos dados! Colete mais.")
    else:
        print(f"  ✅ OK")
    
    # 2. Período temporal
    print(f"\n✓ PERÍODO TEMPORAL")
    date_range = df['timestamp'].max() - df['timestamp'].min()
    days = date_range.days
    print(f"  Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print(f"  Duração: {days} dias (recomendado: > 30 dias)")
    if days < 30:
        print(f"  ⚠️ Período pequeno! Colete pelo menos 1 mês.")
    else:
        print(f"  ✅ OK")
    
    # 3. Timestamps uniformes
    print(f"\n✓ TIMESTAMPS")
    # Calcular intervalo médio entre pontos
    time_diffs = df['timestamp'].diff().dt.total_seconds()
    median_diff = time_diffs.median()
    min_diff = time_diffs.min()
    max_diff = time_diffs.max()
    
    print(f"  Intervalo median: {median_diff:.0f}s ({median_diff/60:.1f} min)")
    print(f"  Intervalo min: {min_diff:.0f}s")
    print(f"  Intervalo max: {max_diff:.0f}s")
    
    # Alertar se intervalo muito variável
    if max_diff / median_diff > 2:
        print(f"  ⚠️ Timestamps muito irregulares! (max/median = {max_diff/median_diff:.1f}x)")
    else:
        print(f"  ✅ Timestamps regulares")
    
    # 4. Valores faltantes
    print(f"\n✓ VALORES FALTANTES")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("  ⚠️ Campos com NaN:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"    - {col}: {count} ({pct:.1f}%)")
    else:
        print(f"  ✅ Sem valores faltantes")
    
    # 5. Distribuição de valores
    print(f"\n✓ DISTRIBUIÇÃO DE VALORES")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        print(f"\n  {col}:")
        print(f"    Min: {df[col].min():.2f}")
        print(f"    Max: {df[col].max():.2f}")
        print(f"    Média: {df[col].mean():.2f}")
        print(f"    Std: {df[col].std():.2f}")
        
        # Verificar se está em range esperado
        if col == 'cpu_usage' or col == 'memory_usage' or col == 'error_rate':
            if df[col].min() < 0 or df[col].max() > 100:
                print(f"    ⚠️ Fora do range 0-100!")
        
        # Verificar se é zero sempre
        if df[col].nunique() == 1:
            print(f"    ⚠️ Valores constantes! (sempre {df[col].iloc[0]})")
    
    # 6. Correlação entre variáveis
    print(f"\n✓ CORRELAÇÃO")
    corr_matrix = df[numeric_cols].corr()
    print("  Correlação esperada:")
    print("    - cpu_usage ↔ memory_usage: 0.3-0.7 (tipicamente +)")
    print("    - error_rate ↔ latency: 0.2-0.6 (tipicamente +)")
    print("\n  Matriz de correlação:")
    print(corr_matrix.round(2))
    
    # 7. Detecção de anomalias
    print(f"\n✓ ANOMALIAS")
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        
        outliers = df[
            (df[col] > mean + 3*std) | (df[col] < mean - 3*std)
        ]
        
        if len(outliers) > 0:
            pct = (len(outliers) / len(df)) * 100
            print(f"  - {col}: {len(outliers)} outliers ({pct:.1f}%)")
    
    # Resumo final
    print(f"\n" + "=" * 60)
    issues = []
    if len(df) < 8000:
        issues.append("Dataset pequeno")
    if days < 30:
        issues.append("Período pequeno")
    if missing.sum() > 0:
        issues.append("Valores faltantes")
    
    if not issues:
        print("✅ DATASET PRONTO PARA TREINAR!")
    else:
        print(f"⚠️ PROBLEMAS DETECTADOS:")
        for issue in issues:
            print(f"   - {issue}")
    print("=" * 60)

if __name__ == "__main__":
    validate_kubernetes_dataset('./dataset/kubernetes/kubernetes_metrics_clean.csv')
```

**Como usar**:
```bash
python validate_dataset.py
```

---

## <a name="treinamento"></a>7. Treinando o Modelo

### 7.1 Preparar ambiente

```bash
cd /path/to/Time-LLM

# 1. Copiar dados para a pasta correta
mkdir -p ./dataset/kubernetes
cp ./dataset/kubernetes/kubernetes_metrics_clean.csv ./dataset/kubernetes/kubernetes_metrics.csv

# 2. Atualizar código para suportar novo dataset
# (Ver arquivo data_provider - próxima seção)
```

### 7.2 Modificar data_provider (registrar novo dataset)

Editar `data_provider/data_loader.py` para reconhecer dataset Kubernetes:

```python
# Em data_provider/data_loader.py, procurar por:
class Dataset_ETT_hourly(Dataset):
    def __init__(self, ...):
        ...
        if set_type == 0:
            self.data_path = os.path.join(self.root_path,
                                         'ETTh1.csv')

# Adicionar APÓS essa classe:
class Dataset_Kubernetes(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='kubernetes_metrics.csv',
                 target='error_rate', scale=True, timeenc=0, freq='h'):
        
        # type 0: train, 1: val, 2: test
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = os.path.join(self.root_path, data_path)
        
        self.data_x = None
        self.data_y = None
        self.scaler = None
        
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)
        
        # Features a usar (excluir timestamp)
        cols = [col for col in df_raw.columns if col != 'timestamp']
        df_raw = df_raw[['timestamp'] + cols]
        
        num_train = int(len(df_raw) * 0.7)       # 70% train
        num_test = int(len(df_raw) * 0.2)        # 20% test
        num_vali = len(df_raw) - num_train - num_test  # 10% val
        
        if self.set_type == 0:
            df_data = df_raw[:num_train]
        elif self.set_type == 1:
            df_data = df_raw[num_train:num_train+num_vali]
        else:
            df_data = df_raw[num_train+num_vali:]
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_data.columns[1:]
            df_x = df_data[cols_data]
        elif self.features == 'S':
            df_x = df_data[[self.target]]
        
        if self.scale:
            data = self.scaler.fit_transform(df_x)
        else:
            data = df_x.values
        
        self.data_x = data
        self.data_y = data
    
    def __getitem__(self, index):
        seq_len = 336  # 14 dias em horas
        pred_len = 96  # 4 dias
        
        s_begin = max(0, index)
        s_end = s_begin + seq_len
        r_begin = s_end - pred_len
        r_end = r_begin + pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        if self.set_type == 2:
            seq_y = self.data_y[r_begin:r_end]
        else:
            seq_y = self.data_y[r_begin:r_end]
        
        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - 336 - 96 + 1
```

### 7.3 Comando para treinar

#### Opção 1: Transfer Learning (RECOMENDADO - 2-4 horas)

```powershell
# Primeira vez: Fine-tuning do modelo ETTh1
uv run python run_main_windows_fixed.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --model_id kubernetes_finetuned_v1 `
  --model TimeLLM `
  --data kubernetes `
  --features M `
  --root_path ./dataset/kubernetes/ `
  --data_path kubernetes_metrics.csv `
  --seq_len 336 `
  --label_len 168 `
  --pred_len 96 `
  --batch_size 16 `
  --train_epochs 15 `
  --learning_rate 0.00001 `
  --d_model 32 `
  --d_ff 128 `
  --n_heads 8 `
  --e_layers 2 `
  --d_layers 1 `
  --dropout 0.1 `
  --enc_in 9 `
  --dec_in 9 `
  --c_out 9 `
  --llm_model GPT2 `
  --llm_dim 768 `
  --llm_layers 12 `
  --patience 5 `
  --model_comment kubernetes-finetuned-v1
```

**Explicação dos parâmetros**:
- `--seq_len 336`: 2 semanas de histórico (14 dias × 24 horas)
- `--pred_len 96`: prever 4 dias adiante
- `--learning_rate 0.00001`: 10x menor (transfer learning)
- `--train_epochs 15`: poucos epochs (convergência rápida)
- `--enc_in 9`: correspondente ao número de features no CSV
- `--batch_size 16`: pequeno (CPU Windows)

#### Opção 2: Treino do Zero (MAIS ROBUSTO - 50-100 horas)

```powershell
uv run python run_main_windows_fixed.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --model_id kubernetes_scratch_v1 `
  --model TimeLLM `
  --data kubernetes `
  --features M `
  --root_path ./dataset/kubernetes/ `
  --data_path kubernetes_metrics.csv `
  --seq_len 336 `
  --label_len 168 `
  --pred_len 96 `
  --batch_size 16 `
  --train_epochs 50 `
  --learning_rate 0.0001 `
  --d_model 32 `
  --d_ff 128 `
  --enc_in 9 `
  --dec_in 9 `
  --c_out 9 `
  --llm_model GPT2 `
  --llm_layers 12 `
  --model_comment kubernetes-fullretrain-v1
```

### 7.4 Durante o treinamento

```bash
# Terminal 2: Monitorar loss
tail -f ./run_logs/long_term_forecast_kubernetes_...

# Esperado:
# Epoch 1 of 15: train_loss=0.85, val_loss=0.82
# Epoch 2 of 15: train_loss=0.78, val_loss=0.75
# ... (loss deve descer)
```

### 7.5 Depois do treinamento

#### Testar em novos dados

```powershell
# Fazer inferência
uv run python run_main_windows_fixed.py `
  --task_name long_term_forecast `
  --is_training 0 `
  --model_id kubernetes_inference `
  --model TimeLLM `
  --data kubernetes `
  --features M `
  --root_path ./dataset/kubernetes/ `
  --data_path kubernetes_metrics.csv `
  --seq_len 336 `
  --pred_len 96 `
  --batch_size 1 `
  --enc_in 9 `
  --dec_in 9 `
  --c_out 9 `
  --llm_model GPT2 `
  --checkpoints ./checkpoints/long_term_forecast_kubernetes_...
```

#### Evaluar performance

```python
# evaluate_kubernetes_model.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Carregar predictions e actuals
predictions = np.load('./predictions.npy')  # Output do modelo
actuals = np.load('./actuals.npy')

# Calcular métricas
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
mape = mean_absolute_percentage_error(actuals, predictions)
rmse = np.sqrt(mse)

print("=" * 50)
print("PERFORMANCE DO MODELO")
print("=" * 50)
print(f"\nMSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# Comparar com baseline (modelo simples)
baseline = np.repeat(actuals[:-1].mean(), len(actuals))
baseline_mae = mean_absolute_error(actuals, baseline)

print(f"\nMelhoria vs baseline:")
print(f"  MAE model: {mae:.4f}")
print(f"  MAE baseline: {baseline_mae:.4f}")
print(f"  Melhoria: {((baseline_mae - mae) / baseline_mae * 100):.1f}%")
```

---

## 📋 Resumo Prático

### Timeline sugerida

```
DIA 1:
  ├─ Instalar Prometheus (se não tiver)
  └─ Começar coleta de dados

DIAS 1-60:
  └─ Deixar coletando 60+ dias (pode ser paralelo)

DIA 61:
  ├─ Validar dataset
  ├─ Limpar dados
  └─ Começar treinamento

DIA 62 (se Transfer Learning):
  └─ ✅ Modelo pronto!

DIA 1-4 (se Full Retrain):
  └─ Completar 50 epochs
```

### Checklist de implementação

- [ ] Prometheus instalado
- [ ] Scripts de coleta rodando
- [ ] 60+ dias de dados históricos
- [ ] CSV validado (`validate_dataset.py`)
- [ ] Dataset registrado em `data_provider/data_loader.py`
- [ ] Treino iniciado
- [ ] Performance avaliada
- [ ] Modelo pronto para produçãoo

---

## 🎯 Próximos Passos

1. **Coleta**: Rodar scripts por 2 meses
2. **Treino**: Fine-tuning (2-4h) ou Full (50-100h)
3. **Deploy**: Fazer inferência em tempo real em seu cluster
4. **Monitoramento**: Usar modelo para detecção de anomalias

Tem dúvidas? Posso criar scripts adicionais para:
- Integração com Grafana
- API para inferência em tempo real
- Dashboard de monitoramento
- Alertas automáticos

