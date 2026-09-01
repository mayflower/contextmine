# ContextMine Helm Chart

This Helm chart deploys ContextMine, a documentation/code indexing system with MCP (Model Context Protocol) support, on Kubernetes.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.0+
- PV provisioner support (for PostgreSQL and Worker persistence)
- (Optional) Ingress controller (nginx-ingress, traefik, etc.)
- (Optional) cert-manager for TLS certificates

## Quick Start

### 1. Build and Push Images

Before deploying, build and push the Docker images to your registry:

```bash
# Build and tag images with the source revision
docker build -t ghcr.io/your-org/contextmine-api:sha-<git-sha> -f apps/api/Dockerfile .
docker build -t ghcr.io/your-org/contextmine-worker:sha-<git-sha> -f apps/worker/Dockerfile .
docker build --target analyzer -t ghcr.io/your-org/contextmine-analyzer:sha-<git-sha> -f apps/worker/Dockerfile .

# Push to registry
docker push ghcr.io/your-org/contextmine-api:sha-<git-sha>
docker push ghcr.io/your-org/contextmine-worker:sha-<git-sha>
docker push ghcr.io/your-org/contextmine-analyzer:sha-<git-sha>
```

### 2. Create Values Override

Create a `my-values.yaml` file with your configuration:

```yaml
api:
  image:
    repository: ghcr.io/your-org/contextmine-api
    tag: sha-<git-sha>
    digest: sha256:<registry-digest>

worker:
  image:
    repository: ghcr.io/your-org/contextmine-worker
    tag: sha-<git-sha>
    digest: sha256:<registry-digest>

config:
  appMode: "production"
  publicBaseUrl: "http://localhost:8000"
  mcpOauthBaseUrl: "http://localhost:8000"
  sandbox:
    apiUrl: "https://agent-sandbox-platform-api.data.mayflower.zone"
    analyzerSnapshot: "contextmine-analyzer"
  scip:
    installDepsMode: "never"

secrets:
  sandboxApiKey: "your-platform-api-key"
  github:
    clientId: "your-github-client-id"
    clientSecret: "your-github-client-secret"
  sessionSecret: "generate-a-secure-random-value"
  tokenEncryptionKey: "generate-a-secure-random-value"
  openaiApiKey: "sk-..."  # Optional
```

### 3. Install the Chart

```bash
# Install with custom values
helm install contextmine ./deploy/helm/contextmine -f my-values.yaml

# Or install with inline overrides
helm install contextmine ./deploy/helm/contextmine \
  --set secrets.github.clientId=your-client-id \
  --set secrets.github.clientSecret=your-client-secret
```

### 4. Access the Application

```bash
# Port forward the API service
kubectl port-forward svc/contextmine-api 8000:8000

# Port forward Prefect UI (optional)
kubectl port-forward svc/contextmine-prefect 4200:4200
```

Open http://localhost:8000 in your browser.

## Configuration

### Global Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imagePullPolicy` | Image pull policy | `IfNotPresent` |
| `global.imagePullSecrets` | Image pull secrets | `[]` |

### PostgreSQL Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `postgresql.enabled` | Enable built-in PostgreSQL | `true` |
| `postgresql.external.host` | External PostgreSQL host | `""` |
| `postgresql.external.port` | External PostgreSQL port | `5432` |
| `postgresql.auth.username` | PostgreSQL username | `contextmine` |
| `postgresql.auth.password` | PostgreSQL password | `contextmine` |
| `postgresql.auth.database` | Main database name | `contextmine` |
| `postgresql.auth.prefectDatabase` | Prefect database name | `prefect` |
| `postgresql.persistence.enabled` | Enable persistence | `true` |
| `postgresql.persistence.size` | Storage size | `10Gi` |
| `postgresql.persistence.storageClass` | Storage class | `""` |

### API Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api.image.repository` | API image repository | `ghcr.io/your-org/contextmine-api` |
| `api.image.tag` | API image tag | `sha-35e5920` |
| `api.image.digest` | API image digest | See values.yaml |
| `api.replicaCount` | Number of replicas | `1` |
| `api.service.type` | Service type | `ClusterIP` |
| `api.service.port` | Service port | `8000` |
| `api.resources` | Resource limits/requests | See values.yaml |

### Prefect Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `prefect.image.repository` | Prefect image | `prefecthq/prefect` |
| `prefect.image.tag` | Prefect image tag | `3.8.4-python3.14` |
| `prefect.image.digest` | Prefect image digest | See values.yaml |
| `prefect.replicaCount` | Number of replicas | `1` |
| `prefect.service.type` | Service type | `ClusterIP` |
| `prefect.service.port` | Service port | `4200` |

### Worker Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `worker.image.repository` | Worker image repository | `ghcr.io/your-org/contextmine-worker` |
| `worker.image.tag` | Worker image tag | `sha-35e5920` |
| `worker.image.digest` | Worker image digest | See values.yaml |
| `worker.replicaCount` | Number of replicas | `1` |
| `worker.persistence.enabled` | Enable persistence | `true` |
| `worker.persistence.size` | Storage size | `20Gi` |

### Ingress Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `false` |
| `ingress.className` | Ingress class name | `""` |
| `ingress.annotations` | Ingress annotations | `{}` |
| `ingress.hosts` | Ingress hosts | See values.yaml |
| `ingress.tls` | TLS configuration | `[]` |

### Application Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `config.debug` | Enable debug mode | `"true"` |
| `config.appMode` | Runtime mode; production disables local repository analysis | `"development"` |
| `config.publicBaseUrl` | Public URL for OAuth | `"http://localhost:8000"` |
| `config.mcpOauthBaseUrl` | MCP OAuth base URL | `"http://localhost:8000"` |
| `config.mcpAllowedOrigins` | Allowed CORS origins | `""` |
| `config.sandbox.apiUrl` | Mayflower Agent Sandbox platform API | `""` |
| `config.sandbox.analyzerSnapshot` | Published analyzer snapshot name | `""` |

### Secrets

| Parameter | Description | Default |
|-----------|-------------|---------|
| `secrets.github.clientId` | GitHub OAuth client ID | `""` |
| `secrets.sandboxApiKey` | Agent Sandbox platform API key | `""` |
| `secrets.github.clientSecret` | GitHub OAuth client secret | `""` |
| `secrets.sessionSecret` | Session encryption key | `"dev-..."` |
| `secrets.tokenEncryptionKey` | Token encryption key | `"dev-..."` |
| `secrets.openaiApiKey` | OpenAI API key | `""` |
| `secrets.anthropicApiKey` | Anthropic API key | `""` |
| `secrets.geminiApiKey` | Gemini API key | `""` |

## Using External PostgreSQL

To use an external PostgreSQL database (e.g., AWS RDS, Cloud SQL):

```yaml
postgresql:
  enabled: false
  external:
    host: "your-rds-instance.region.rds.amazonaws.com"
    port: 5432
  auth:
    username: "contextmine"
    password: "your-secure-password"
    database: "contextmine"
    prefectDatabase: "prefect"
```

**Note**: You must create both the `contextmine` and `prefect` databases manually, and ensure required extensions are installed (or use `ghcr.io/mayflower/pg4ai`):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
```

## Production Deployment

For production deployments, use the provided `values-production.yaml` as a starting point:

```bash
helm install contextmine ./deploy/helm/contextmine \
  -f ./deploy/helm/contextmine/values-production.yaml \
  -f my-production-secrets.yaml
```

### Production Checklist

- [ ] Change all default passwords and secrets
- [ ] Configure proper resource limits
- [ ] Enable ingress with TLS
- [ ] Set up external PostgreSQL (recommended)
- [ ] Configure proper storage classes
- [ ] Set `config.debug` to `"false"`
- [ ] Restrict `config.mcpAllowedOrigins`
- [ ] Publish the non-root `analyzer` image as the configured sandbox snapshot
- [ ] Configure the sandbox API key and verify one create/run/result/delete lifecycle
- [ ] Set up monitoring and alerting
- [ ] Configure backup for PostgreSQL

## Upgrading

```bash
helm upgrade contextmine ./deploy/helm/contextmine -f my-values.yaml
```

## Uninstalling

```bash
helm uninstall contextmine
```

**Note**: This will not delete PersistentVolumeClaims by default. To delete all data:

```bash
kubectl delete pvc -l app.kubernetes.io/instance=contextmine
```

## Troubleshooting

### Pods not starting

Check pod status and logs:

```bash
kubectl get pods -l app.kubernetes.io/instance=contextmine
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### Database connection issues

Verify PostgreSQL is running:

```bash
kubectl get pods -l app.kubernetes.io/component=postgres
kubectl logs <postgres-pod-name>
```

### Prefect UI not accessible

Access Prefect UI via port-forward:

```bash
kubectl port-forward svc/contextmine-prefect 4200:4200
```

Then open http://localhost:4200 in your browser.
