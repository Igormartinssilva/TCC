# Prometheus PromQL MCP - Installation and Kubernetes Deployment

## Local Setup

### Build Docker Image

Navigate to the directory containing the Dockerfile and run:

```bash
docker build -t prometheus-promql-mcp:latest .
```

### Test Locally

Run the container to test it locally:

```bash
docker run --rm -p 8001:8001 -e PROMETHEUS_URL=http://your-prometheus:9090 prometheus-promql-mcp:latest
```

Replace `http://your-prometheus:9090` with your Prometheus URL.

### Push to Docker Hub

Tag the image:

```bash
docker tag prometheus-promql-mcp:latest igormsilva/prometheus-promql-mcp:latest
```

Push to Docker Hub:

```bash
docker push igormsilva/prometheus-promql-mcp:latest
```

## Kubernetes Deployment

### Prerequisites

- Access to a Kubernetes cluster (INF cluster)
- `kubectl` configured and authenticated
- A Prometheus instance accessible from the cluster

### Deploy Steps

1. **Edit Kubernetes Files**

   Update the namespace in both YAML files to match your deployment namespace:

   - `Service-prometheus-promql-mcp.yaml`
   - `Deployment-prometheus-promql-mcp.yaml`

   Set the `PROMETHEUS_URL` environment variable in the Deployment file to point to your Prometheus instance.

2. **SSH into the Cluster**

   ```bash
   ssh your-user@cluster-host
   ```

3. **Apply Service**

   ```bash
   kubectl apply -f Service-prometheus-promql-mcp.yaml
   ```

4. **Apply Deployment**

   ```bash
   kubectl apply -f Deployment-prometheus-promql-mcp.yaml
   ```

### Important Notes

- Ensure the Service port matches the port where MCP is running (default: 8001)
- Update the `PROMETHEUS_URL` environment variable in the Deployment to match your Prometheus instance
- Update the namespace in both YAML files according to your cluster configuration

## Verify Deployment

Check if the pod is running:

```bash
kubectl get pods -n <your-namespace>
```

Check service status:

```bash
kubectl get svc -n <your-namespace>
```

View pod logs:

```bash
kubectl logs -f <pod-name> -n <your-namespace>
```
