---
type: ssot
owner: governance-team
last_updated: 2026-09-03
last-reviewed: 2026-09-18
---

# 远程机器接入 aetherforge 教程

> 目标：让 Mac mini 的 aetherforge gateway 能调用 MacBook Pro 和 Y7000P 上的 LM Studio / Ollama

## 步骤 1：让 LM Studio 监听 Tailscale 网络

### MacBook Pro（macOS）

1. 打开 LM Studio
2. **Settings → Local Inference Server**
3. **Host** 改成 `0.0.0.0`（默认是 `localhost`）
4. **Start Server**

验证（在 Mac mini 上）：
```bash
curl http://100.96.126.35:1234/v1/models
```

### Y7000P（Windows 11）

1. 打开 LM Studio
2. **Settings → Local Inference Server**
3. **Host** 改成 `0.0.0.0`
4. **Start Server**

验证（在 Mac mini 上）：
```bash
curl http://100.64.43.36:1234/v1/models
```

## 步骤 2：让 Ollama 监听 Tailscale 网络

### MacBook Pro

```bash
# 设置环境变量（永久生效加到 ~/.zshrc）
export OLLAMA_HOST=0.0.0.0:11434

# 重启 Ollama
brew services restart ollama
```

验证：
```bash
# 在 Mac mini 上
curl http://100.96.126.35:11434/api/tags
```

### Y7000P（Windows 11）

```powershell
# 设置环境变量（管理员 PowerShell）
[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST','0.0.0.0:11434','Machine')

# 重启 Ollama（在服务管理器里重启）
```

验证：
```bash
# 在 Mac mini 上
curl http://100.64.43.36:11434/api/tags
```

## 步骤 3：在 Mac mini 上验证接入

```bash
cd ~/Workspace/projects/aetherforge
uv run python -m llm_gateway.cli list --ssot
```

预期能看到类似这样：

```
  🟢 ENG-LMSTUDIO-MACBOOKPRO/gemma-4-31b-it-mlx     ← MBP 的 LM Studio
  🟢 ENG-OLLAMA-MACBOOKPRO/qwen3.6-35b-a3b           ← MBP 的 Ollama
  🟢 ENG-LMSTUDIO-Y7000P/gemma-4-12b-it-qat          ← Y7000P 的 LM Studio
  🟢 ENG-OLLAMA-Y7000P/llama3                         ← Y7000P 的 Ollama
  ...
```

## 步骤 4：使用

打通后 aetherforge 自动发现全部模型：
```bash
# 查看所有可用模型
cd ~/Workspace/projects/aetherforge
uv run python -m llm_gateway.cli list --ssot

# 通过 SSOT 路由生成（会自动选模型）
uv run python -m llm_gateway.cli generate --ssot -m gemma-4-12b-qat "你好"
```

## 如果连不通

1. **检查 Tailscale**：`tailscale status` 确认都显示 `-`（已连接）
2. **检查端口监听**：
   - macOS: `lsof -i :1234` → 确认显示 `*:1234` 不是 `127.0.0.1:1234`
   - Windows: `netstat -ano | findstr :1234`
3. **Windows 防火墙**：可能需要允许 LM Studio / Ollama 通过防火墙
4. **Mac mini 上直 ping**：`ping 100.96.126.35`
