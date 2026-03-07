# 🚀 LLM Gateway: Unified AI Routing

A powerful, production-ready AI gateway designed to unify LLM providers (Ollama Cloud) into a single, OpenAI-compatible API. Optimized for performance and high availability.

## 🛠 Features

- **Effectiveness-Based Aliases**: Use descriptive, capability-driven model names (`logic-supreme`, `code-architect`).
- **Smart Key Rotation**: Automatically rotates API keys every **2 hours** to avoid limits.
- **Alias-Aware Permissions**: Filter keys by friendly alias OR technical model name.
- **Extreme Performance**: Optimized for code editors (OpenCode/VSCode) with a TTFT of **~0.003s**.
- **Monitoring Stack**: Integrated Grafana and Prometheus for real-time traffic analysis.
- **Failover & Resilience**: Built-in circuit breakers and automatic provider fallback.

## 📖 Deployment (Docker)

The easiest way to deploy is using Docker. We use custom, high ports (48xxx) to ensure no conflicts with standard services.

```bash
cd docker

# Launch the Gateway only
docker-compose up -d

# Launch with full monitoring (Grafana & Prometheus)
docker-compose --profile monitoring up -d --build
```

### Access Ports:
- **API Gateway**: `http://localhost:48001/v1`
- **Grafana**: `http://localhost:48002` (Admin: `admin` / `admin`)
- **Prometheus**: `http://localhost:48003`

## 🧠 Model Nomenclature

The gateway uses a standard nomenclature based on task effectiveness:

| Alias | Target Usage | Primary Model Examples |
|-------|--------------|------------------------|
| `logic-supreme` | Reasoning / Logic | Kimi K2.5, DeepSeek-V3 |
| `code-architect`| Coding / Architecture| Qwen3-Coder-480B |
| `author-pro`   | Writing / Content | DeepSeek-V3, GLM-5 |
| `omni-think`   | General Intelligence | Llama-3.1-405B |

## ⚙️ Configuration

1. Copy the example configuration:
   ```bash
   cp config/providers.yml.example config/providers.yml
   ```
2. Edit `config/providers.yml` and add your API keys.

## 🧪 Usage Examples

### Chat Completions
```bash
curl http://localhost:48001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "logic-supreme",
    "messages": [{"role": "user", "content": "Explain quantum computing."}],
    "stream": true
  }'
```

### Integration with IDEs (OpenCode / VSCode)
Set your base URL to `http://localhost:48001/v1` and use any string as the API key.

## 📈 Monitoring
Metrics are automatically collected. Access Grafana at `http://localhost:48002` to view real-time throughput and status codes.

## ⚠️ Troubleshooting

### Docker "Permission Denied" Error (Zombie Container)
When trying to update or stop the container, you might encounter an error like:
`Error response from daemon: cannot remove container "llm-gateway": could not kill container: permission denied`

**Why this happens:**
A process inside the container (like the Python web server) has turned into a "zombie" process. The Docker daemon loses the ability to kill it, leaving the container in an unbreakable locked state. Since Docker relies on immutable images, you cannot simply update running code; you *must* destroy the old container and start a new one, which this bug prevents.

**How to fix it:**
You have two options to break the lock and apply your updates:

1. **Restart the Docker Daemon** (Recommended):
   This flushes the zombie processes from the hypervisor/daemon level.
   ```bash
   sudo systemctl restart docker
   sudo docker-compose -f docker/docker-compose.yml up -d --build
   ```

2. **Kill the Zombie Process Manually**:
   If you cannot restart the daemon, find the stuck Python process on your host and kill it directly:
   ```bash
   ps -ef | grep gateway.server | grep -v grep | awk '{print $2}' | xargs -r kill -9
   docker rm -f llm-gateway
   docker-compose -f docker/docker-compose.yml up -d --build
   ```

---
*Developed for high-performance agentic workflows.*
