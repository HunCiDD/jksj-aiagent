# LLM Unified Gateway

第一周作业的可运行实现。网关向后兼容 OpenAI Chat Completions 的主要响应结构，内部通过两个 Adapter 调用不同的上游协议：

| 网关模型别名 | 上游协议 |
| --- | --- |
| `deepseek-v4-pro` | OpenAI Responses API |
| `deepseek-v4-flash` | Anthropic Messages API |

能力范围：

- 根据请求 `model` 动态路由。
- 普通响应和 SSE 流式响应。
- `response_format={"type":"json_object"}` 结构化输出，并在网关侧校验合法 JSON。
- SQLite 持久化 Prompt 模板和不可变版本。
- 输入、输出、缓存、推理和总 Token 统计。
- 流式 TTFT 和总延迟。
- 统一错误码、最多 3 次指数退避重试。
- 每个模型独立的 token bucket 限流，超限返回 HTTP 429。
- 调用记录查询、健康检查和真实服务 smoke 脚本。

完整设计说明见 [assignment-minimum-gateway.md](assignment-minimum-gateway.md)，能力全景见 [full-gateway-capabilities.md](full-gateway-capabilities.md)。

## 1. 安装与配置

仓库根目录执行：

```powershell
uv sync
Copy-Item week01/llm-gateway/.env.example week01/llm-gateway/.env
```

项目不会自动读取 `.env`。可以将变量设置到当前 PowerShell 会话，或按自己的方式加载 `.env`。

```powershell
$env:GATEWAY_API_KEY = "dev-gateway-key"

$env:DEEPSEEK_RESPONSES_BASE_URL = "https://your-responses-endpoint/v1"
$env:DEEPSEEK_RESPONSES_API_KEY = "your-responses-key"
$env:DEEPSEEK_RESPONSES_MODEL = "deepseek-v4-pro"

$env:DEEPSEEK_MESSAGES_BASE_URL = "https://your-messages-endpoint/v1"
$env:DEEPSEEK_MESSAGES_API_KEY = "your-messages-key"
$env:DEEPSEEK_MESSAGES_MODEL = "deepseek-v4-flash"
```

`BASE_URL` 应包含版本路径，例如以 `/v1` 结尾。网关会分别追加 `/responses` 和 `/messages`。

启动服务：

```powershell
uv run uvicorn app.main:app --app-dir week01/llm-gateway --host 127.0.0.1 --port 8000 --reload
```

OpenAPI 文档位于 `http://127.0.0.1:8000/docs`。

## 2. 普通模型调用

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [
      {"role": "user", "content": "用一句话介绍 LLM Gateway"}
    ]
  }'
```

切换为另一个协议只需修改 `model`：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "用一句话介绍 LLM Gateway"}
    ]
  }'
```

响应中的 `gateway` 字段包含实际协议、请求 ID、重试次数和延迟。非流式请求没有首 Token，`ttft_ms` 为 `null`。

## 3. 流式输出

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "分几段介绍 LLM Gateway"}
    ],
    "stream": true
  }'
```

客户端会收到 `data: {...}` 事件，最后一条为：

```text
data: [DONE]
```

最后一个内容 chunk 包含 usage、`ttft_ms`、`total_latency_ms` 和 `request_id`。

## 4. 结构化 JSON 输出

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [
      {"role": "user", "content": "返回一个包含 name 和 age 的用户对象"}
    ],
    "response_format": {"type": "json_object"}
  }'
```

Assistant 的 `content` 必须是 JSON object 字符串。若上游返回普通文本或数组，网关返回 `422 INVALID_STRUCTURED_OUTPUT`。

两个 Adapter 都通过系统指令约束 JSON 输出；没有把 `response_format` 盲目透传给不支持该字段的协议。最终合法性由网关统一校验。

## 5. Prompt 模板与版本

创建版本 1：

```bash
curl http://127.0.0.1:8000/v1/prompts \
  -H "Authorization: Bearer dev-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "summarize",
    "template": "请总结：{{ text }}"
  }'
```

创建版本 2：

```bash
curl http://127.0.0.1:8000/v1/prompts/summarize/versions \
  -H "Authorization: Bearer dev-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "template": "请用三个要点总结：{{ text }}"
  }'
```

查询版本 1：

```bash
curl http://127.0.0.1:8000/v1/prompts/summarize/versions/1 \
  -H "Authorization: Bearer dev-gateway-key"
```

引用版本的模型调用：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "prompt": {
      "name": "summarize",
      "version": 1,
      "variables": {"text": "需要总结的内容"}
    },
    "messages": [
      {"role": "user", "content": "请执行"}
    ]
  }'
```

变量使用 `{{ variable }}`。缺少变量返回 `422 PROMPT_VARIABLE_MISSING`。版本创建后不可修改；如果请求没有 `messages`，渲染后的模板会作为用户消息。

## 6. 可观测性

每次调用都可以通过 `gateway.request_id` 查询：

```bash
curl http://127.0.0.1:8000/v1/calls/req-your-request-id \
  -H "Authorization: Bearer dev-gateway-key"
```

查看最近调用：

```bash
curl "http://127.0.0.1:8000/v1/calls?limit=20" \
  -H "Authorization: Bearer dev-gateway-key"
```

记录字段包括：

- `input_tokens`
- `output_tokens`
- `cached_input_tokens`
- `reasoning_tokens`
- `total_tokens`
- `ttft_ms`
- `total_latency_ms`
- `attempt_count`
- `retry_count`
- `provider` 和 `protocol`
- Prompt 名称和版本

上游没有返回的分类 token 记为 0，不从字符串长度伪造统计。

## 7. 重试与限流

只对超时、连接错误、上游 429 和上游 5xx 重试。默认执行 `0.5s -> 1s -> 2s` 指数退避，首次调用之外最多重试 3 次。

每个模型拥有独立 token bucket：

```text
DEEPSEEK_RESPONSES_RPM / DEEPSEEK_RESPONSES_BURST
DEEPSEEK_MESSAGES_RPM  / DEEPSEEK_MESSAGES_BURST
```

本地限流超限立即返回 `429 MODEL_RATE_LIMITED` 和 `Retry-After`，不会重复重试。

## 8. 验证

自动化验证不依赖真实供应商：

```powershell
uv run pytest -q
```

或运行 PowerShell 验证入口：

```powershell
pwsh week01/llm-gateway/scripts/verify.ps1
```

自动化测试覆盖：

- Gateway Key 鉴权和健康检查。
- 两个模型分别路由到 Responses 和 Messages Adapter。
- SSE 流式内容和结束事件。
- 两种协议的结构化 JSON。
- Prompt v1/v2、变量渲染、版本不可变和缺变量错误。
- Token 分类、TTFT、总延迟和调用查询。
- 统一结构化输出错误。
- 3 次指数退避重试和退避序列。
- 模型 A 被限流时模型 B 仍可调用。

真实供应商 smoke 需要先配置环境变量并启动服务，然后执行：

```powershell
uv run python week01/llm-gateway/scripts/smoke.py
```

也可以在自动化测试后直接验证真实服务：

```powershell
pwsh week01/llm-gateway/scripts/verify.ps1 -Live
```

## 9. 最小版本限制

- 限流器位于单进程内存；多 worker 部署需要 Redis。
- SQLite 适合作业验收，不用于高并发生产。
- 当前结构化输出支持 `json_object`；JSON Schema 字段校验未作为最小范围实现。
- 没有跨模型 fallback、熔断、语义缓存和 Web 管理后台。
- 流式响应在客户端已经收到内容后不会透明重试。
