# collector.py
import subprocess
import csv
import time
from datetime import datetime

POD_NAME = "meu-pod-teste"
NAMESPACE = "default"
INTERVAL_SECONDS = 30  # coleta a cada 30s
OUTPUT_FILE = "cpu_metrics.csv"

def get_pod_cpu():
    result = subprocess.run(
        ["kubectl", "top", "pod", POD_NAME, "-n", NAMESPACE, "--no-headers"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        parts = result.stdout.split()
        # Output: meu-pod-teste   5m   128Mi
        cpu_str = parts[1]  # ex: "5m" = 5 millicores
        cpu_val = int(cpu_str.replace("m", ""))
        return cpu_val
    return None

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "cpu_millicores"])
    
    while True:
        ts = datetime.now().isoformat()
        cpu = get_pod_cpu()
        if cpu is not None:
            writer.writerow([ts, cpu])
            f.flush()
            print(f"{ts} -> {cpu}m CPU")
        time.sleep(INTERVAL_SECONDS)