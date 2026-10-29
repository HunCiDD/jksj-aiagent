# 第一周作业：LLM 统一模型调用服务最小实现

## 1. 目标与范围

本项目实现一个最小可运行、可验证的 LLM Gateway，严格满足本周作业的六项验收能力：

1. 通过适配器模式统一两种不同 API 协议，并按 `model` 动态路由。
2. 支持 SSE 流式输出。
3. 支持 `response_format` 结构化 JSON 输出。
4. 支持 Prompt 模板存储、变量替换和版本引用。
5. 记录分类 Token 消耗、首 Token 延迟和总延迟。
6. 统一错误码，支持指数退避重试，并按模型独立限流。

同时提供完整源码、README、curl 示例和覆盖全部功能点的验证脚本。

本实现是单机学习版，不包含多租户控制台、语义缓存、熔断、跨模型 fallback、多区域部署和完整安全合规平台。

## 2. 目标架构

```text
Client
  |
  v
POST /v1/chat/completions
  |
  +--> Gateway Key 鉴权
  +--> 按 model 独立限流
  +--> Prompt 模板解析与变量替换
  +--> 根据 model 选择 Adapter
  |
  +-----------------------------+
  |                             |
  v                             v
OpenAI Responses Adapter    Anthropic Messages Adapter
  |                             |
  v                             v
DeepSeek V4 Pro             DeepSeek V4 Flash
  |                             |
  +-----------------------------+
  |
  v
统一响应 / 统一 SSE / 统一错误
  |
  v
SQLite 调用记录与限额配置
```

核心要求是两种上游协议真正不同，不能把两个 OpenAI-compatible 接口当作“双协议适配”的证据。

## 3. 模型路由

作业指定的模型与协议对应关系如下：

| 对客户端暴露的模型 | 上游协议 | Adapter |
| --- | --- | --- |
| `deepseek-v4-pro` | OpenAI Responses API | `ResponsesAdapter` |
| `deepseek-v4-flash` | Anthropic Messages API | `MessagesAdapter` |

上游模型 ID、Base URL、API Key 和超时时间通过环境变量或本地配置提供，不能硬编码在代码中。

示例配置：

```yaml
models:
  deepseek-v4-pro:
    provider: deepseek
    protocol: openai_responses
    upstream_model: ${DEEPSEEK_PRO_RESPONSES_MODEL}
    base_url: ${DEEPSEEK_PRO_RESPONSES_BASE_URL}
    api_key: ${DEEPSEEK_PRO_RESPONSES_API_KEY}
    timeout_seconds: 60

  deepseek-v4-flash:
    provider: deepseek
    protocol: anthropic_messages
    upstream_model: ${DEEPSEEK_FLASH_MESSAGES_MODEL}
    base_url: ${DEEPSEEK_FLASH_MESSAGES_BASE_URL}
    api_key: ${DEEPSEEK_FLASH_MESSAGES_API_KEY}
    timeout_seconds: 60
```

网关必须根据请求中的 `model` 查询路由表，不得用 `if/else` 把供应商差异散落在 API 处理层。

## 4. 统一抽象层

### 4.1 统一请求

内部统一请求至少包含：

```python
class UnifiedRequest:
    model: str
    messages: list[Message]
    stream: bool
    response_format: ResponseFormat | None
    prompt: PromptReference | None
    temperature: float | None
    max_tokens: int | None
```

### 4.2 统一响应

```python
class UnifiedResponse:
    request_id: str
    model: str
    provider: str
    protocol: str
    content: str
    finish_reason: str | None
    usage: TokenUsage
    latency: Latency
```

### 4.3 Adapter 接口

两个 Adapter 必须实现相同接口：

```python
class ProviderAdapter(Protocol):
    async def complete(self, request: UnifiedRequest) -> UnifiedResponse: ...
    async def stream(self, request: UnifiedRequest) -> AsyncIterator[StreamEvent]: ...
```

Adapter 负责：

- 鉴权头和上游 URL。
- 消息结构和请求体转换。
- OpenAI Responses 与 Anthropic Messages 的协议差异。
- 普通响应和 SSE 事件归一化。
- usage、结束原因和错误码映射。

Adapter 不负责 Gateway Key、按模型限流、调用记录、重试策略和 Prompt 版本存储。

## 5. 对外 API

### 5.1 模型调用

```text
POST /v1/chat/completions
Authorization: Bearer <GATEWAY_API_KEY>
```

普通请求：

```json
{
  "model": "deepseek-v4-pro",
  "messages": [
    {"role": "user", "content": "用一句话介绍 LLM Gateway"}
  ],
  "stream": false
}
```

响应保持 OpenAI-compatible 的主要结构，并附加网关元数据：

```json
{
  "id": "chatcmpl-gw-...",
  "object": "chat.completion",
  "model": "deepseek-v4-pro",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "input_tokens": 20,
    "output_tokens": 30,
    "total_tokens": 50,
    "cached_input_tokens": 0,
    "reasoning_tokens": 0
  },
  "gateway": {
    "request_id": "req-...",
    "provider": "deepseek",
    "protocol": "openai_responses",
    "retry_count": 0,
    "ttft_ms": null,
    "total_latency_ms": 850
  }
}
```

非流式请求没有可观测的首个增量 token，`ttft_ms` 可为 `null`。流式请求必须给出真实 TTFT。

### 5.2 流式调用

请求体使用 `"stream": true`，响应类型为 `text/event-stream`。

统一 SSE 事件采用 OpenAI Chat Completions 风格：

```text
data: {"id":"chatcmpl-gw-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-gw-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"你好"}}]}

data: {"id":"chatcmpl-gw-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...}}

data: [DONE]

```

OpenAI Responses 和 Anthropic Messages 的原始事件由各自 Adapter 解析，对客户端只暴露统一格式。

### 5.3 Prompt 版本管理

创建模板并形成版本 1：

```text
POST /v1/prompts
```

```json
{
  "name": "summarize",
  "template": "请总结下面的内容：{{ text }}"
}
```

创建新版本：

```text
POST /v1/prompts/summarize/versions
```

```json
{
  "template": "请用三个要点总结下面的内容：{{ text }}"
}
```

查询指定版本：

```text
GET /v1/prompts/summarize/versions/1
```

调用时引用具体版本：

```json
{
  "model": "deepseek-v4-flash",
  "prompt": {
    "name": "summarize",
    "version": 1,
    "variables": {
      "text": "需要总结的内容"
    }
  },
  "messages": [
    {"role": "user", "content": "请执行"}
  ]
}
```

约束：

- 模板版本创建后不可修改，只能新增版本。
- 必须支持显式版本号，`latest` 为可选增强。
- 变量占位符使用 `{{ variable }}`。
- 缺少变量时返回 `422`，不允许静默替换为空字符串。
- 渲染后的模板作为系统消息放在请求消息前；如果请求没有 `messages`，则将渲染结果作为用户消息。
- 调用记录保存实际引用的模板名称和版本。

### 5.4 调用记录查询

```text
GET /v1/calls/{request_id}
```

用于验证 Token、TTFT、总延迟、重试次数、实际路由结果和错误码。验证脚本不得只依赖控制台日志。

### 5.5 健康检查

```text
GET /health
GET /ready
```

`/health` 表示进程存活，`/ready` 表示配置和必要依赖可用。

## 6. 结构化输出

统一入口至少支持：

```json
{
  "model": "deepseek-v4-pro",
  "messages": [
    {"role": "user", "content": "返回一个包含 name 和 age 的用户对象"}
  ],
  "response_format": {
    "type": "json_object"
  }
}
```

实现要求：

1. 网关解析统一 `response_format`。
2. Adapter 根据上游协议选择原生参数、Prompt 约束或两者组合。
3. 对上游返回的 Assistant 内容执行 `json.loads()`。
4. 只有合法 JSON 才视为成功。
5. 非法 JSON 转换为统一错误 `INVALID_STRUCTURED_OUTPUT`，并记录原请求 ID。
6. 两个模型协议都必须具备可验证的结构化输出路径。

为保证作业稳定，验收可以使用 `json_object`。如增加 `json_schema`，必须实际校验字段、类型和必填项，不能只检查字符串能否解析。

结构化输出和流式输出可以作为两个独立测试点。若组合使用，网关应在完整流结束后执行最终 JSON 校验。

## 7. 可观测性

每次成功或最终失败的逻辑调用都写入 SQLite 调用记录，至少包含：

```text
request_id
tenant_id
requested_model
resolved_model
provider
protocol
prompt_name
prompt_version
status
error_code
input_tokens
output_tokens
cached_input_tokens
reasoning_tokens
total_tokens
ttft_ms
total_latency_ms
attempt_count
retry_count
created_at
```

### 7.1 Token 分类

必须统一为以下字段：

- `input_tokens`
- `output_tokens`
- `cached_input_tokens`
- `reasoning_tokens`
- `total_tokens`

供应商没有提供的分类记录为 `0` 或 `null`，不得根据字符串长度伪造精确 token 数。若供应商只在流末尾返回 usage，Adapter 应收集最终 usage 并交给调用记录层。

### 7.2 延迟

- `ttft_ms`：网关收到请求到收到第一个实际内容增量的时间。
- `total_latency_ms`：网关收到请求到生成完成或最终失败的时间。
- 工具参数增量、思考事件或空心跳是否计入 TTFT，必须在 README 中统一说明；建议只以第一个可见内容或工具参数增量为准。

### 7.3 日志安全

- 不记录上游 API Key 和 Gateway Key。
- 默认不记录完整消息、完整响应和 Prompt 变量。
- 日志必须包含 `request_id`，可通过该 ID 查询结构化调用记录。

## 8. 统一错误

统一错误格式：

```json
{
  "error": {
    "code": "UPSTREAM_TIMEOUT",
    "message": "Upstream model request timed out",
    "request_id": "req-...",
    "retryable": true,
    "details": {}
  }
}
```

最小错误码集合：

| HTTP | 错误码 | 是否重试 |
| --- | --- | --- |
| 400 | `INVALID_REQUEST` | 否 |
| 401 | `INVALID_GATEWAY_KEY` | 否 |
| 404 | `MODEL_NOT_FOUND` | 否 |
| 422 | `PROMPT_VARIABLE_MISSING` | 否 |
| 422 | `INVALID_STRUCTURED_OUTPUT` | 否 |
| 429 | `MODEL_RATE_LIMITED` | 否 |
| 429 | `UPSTREAM_RATE_LIMITED` | 是 |
| 502 | `UPSTREAM_BAD_RESPONSE` | 视情况 |
| 503 | `UPSTREAM_UNAVAILABLE` | 是 |
| 504 | `UPSTREAM_TIMEOUT` | 是 |
| 500 | `INTERNAL_ERROR` | 否 |

内部限流返回的 429 不应再次重试，否则会形成无效的自我重试循环。

## 9. 指数退避重试

同一模型调用失败时，仅对可重试错误执行指数退避：

```text
首次尝试
第 1 次重试：等待 0.5 秒
第 2 次重试：等待 1.0 秒
第 3 次重试：等待 2.0 秒
```

要求：

- 配置 `max_retries = 3`。
- 按作业字面语义，首次调用之外最多再重试 3 次，并在 README 明示。
- 使用 `base_delay * 2 ** (retry_index - 1)` 计算退避时间。
- 设置最大退避上限，例如 2 秒或 5 秒。
- 不得重试 `400`、`401`、`404`、本地 `429` 和结构化输出校验错误。
- 流式请求只有在首个客户端可见增量发出之前才能透明重试。
- 测试中通过 mock 或注入等待函数验证退避序列，不实际制造长时间等待。

本最小版本可以只重试当前选定模型，不要求跨模型 fallback。

## 10. 按模型独立限流

每个模型维护独立限流器：

```text
deepseek-v4-pro   -> limiter A
deepseek-v4-flash -> limiter B
```

最小实现采用进程内 token bucket 或固定窗口即可：

- 限制每分钟请求数。
- 支持有限突发量。
- 超限立即返回 HTTP 429。
- 响应包含 `Retry-After`。
- 模型 A 超限不得影响模型 B。
- 通过低阈值配置使验证脚本可以稳定复现 429。

请求进入时消耗一次限流配额，后续重试不重复消耗该逻辑请求的入口配额。多次 worker 部署需要 Redis 等分布式限流器，但不属于本次作业最小范围。

## 11. 数据存储

最小版本使用 SQLite 即可：

### prompt_templates

```text
id
name
version
template
created_at
```

对 `(name, version)` 建唯一约束。

### llm_calls

```text
request_id              PRIMARY KEY
tenant_id
requested_model
resolved_model
provider
protocol
prompt_name
prompt_version
status
error_code
input_tokens
output_tokens
cached_input_tokens
reasoning_tokens
total_tokens
ttft_ms
total_latency_ms
attempt_count
retry_count
created_at
```

SQLite 足以提供持久、可查询的验收证据，不要求部署 PostgreSQL 或 Redis。

## 12. 最小安全边界

- 使用一个静态 Gateway API Key 完成鉴权，密钥通过环境变量注入。
- Gateway Key 与上游供应商 Key 分离。
- 上游 Base URL 和密钥只允许来自服务端配置。
- 日志不记录密钥和完整用户内容。
- 输入消息、`model`、模板变量和 `response_format` 使用 Pydantic 校验。
- 设置最大输入长度、最大输出 token 和上游超时，避免单请求无限消耗。

## 13. 建议目录结构

```text
week01/llm-gateway/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── chat.py
│   │   ├── prompts.py
│   │   └── calls.py
│   ├── adapters/
│   │   ├── base.py
│   │   ├── responses.py
│   │   └── messages.py
│   ├── core/
│   │   ├── config.py
│   │   ├── errors.py
│   │   ├── auth.py
│   │   └── routing.py
│   ├── services/
│   │   ├── gateway.py
│   │   ├── prompts.py
│   │   ├── rate_limit.py
│   │   └── observability.py
│   ├── db.py
│   └── schemas.py
├── config/
│   └── models.yaml
├── scripts/
│   ├── smoke.py
│   ├── verify.sh
│   └── verify.ps1
├── tests/
│   ├── test_adapters.py
│   ├── test_streaming.py
│   ├── test_structured_output.py
│   ├── test_prompts.py
│   ├── test_retry.py
│   └── test_rate_limit.py
├── .env.example
├── README.md
├── assignment-minimum-gateway.md
└── full-gateway-capabilities.md
```

## 14. 验证证据矩阵

| 验收点 | 自动化验证 | 真实 smoke 证据 |
| --- | --- | --- |
| 双协议动态路由 | 断言两个 `model` 分别选择 Responses 和 Messages Adapter | 分别调用两个模型并返回成功 |
| SSE 流式 | 断言多个统一 chunk 和 `[DONE]` | curl 展示原始流 |
| 结构化输出 | mock 上游输出合法和非法 JSON，断言解析与错误 | 两个模型各调用一次 JSON 输出 |
| Prompt 版本 | 创建 v1/v2，断言 v1 不跟随 v2 变化 | curl 引用指定版本并查看上游渲染结果 |
| Token 和延迟 | 断言两类 usage、TTFT 和总延迟写入记录 | 通过 `request_id` 查询调用记录 |
| 统一错误 | 参数、鉴权、模板、上游错误码测试 | curl 展示统一错误响应 |
| 指数重试 | mock 连续失败，记录重试次数和退避序列 | 展示重试日志和最终成功/失败 |
| 按模型限流 | 耗尽模型 A，断言 A 为 429 且 B 仍为 200 | 使用低阈值 curl 验证 |

外部供应商网络和额度不能成为唯一验收依据。`pytest` 应通过 mock 上游覆盖全部核心逻辑，真实模型 smoke 脚本用于最终演示。

## 15. 最小版本的完成定义

满足以下条件即可认为作业完成：

1. `uv run uvicorn app.main:app` 一条命令可以启动服务。
2. 两个模型分别通过各自真实协议调用成功。
3. 流式请求返回连续 SSE chunk 和结束事件。
4. `response_format` 请求返回的 Assistant 内容是合法 JSON。
5. 可以创建至少两个 Prompt 版本，并引用指定版本得到不同渲染结果。
6. 每个调用均可查询输入/输出/缓存/推理/总 token、TTFT 和总延迟。
7. 可重试错误表现为最多 3 次指数退避重试。
8. 模型 A 超限返回 429 时，模型 B 的独立限流不受影响。
9. README 包含环境变量、启动命令和全部 curl 示例。
10. 验证脚本和 pytest 覆盖上述全部行为，并提供可复现输出。

## 16. 明确不做的能力

以下能力不影响本次作业验收，最小版可以不做：

- Web 管理后台和完整多租户体系。
- Redis 分布式限流。
- 语义缓存和响应缓存。
- 熔断器、跨模型 fallback 和多区域路由。
- 图片、音频和视频的跨协议转换。
- 完整 PII、内容审核和合规策略引擎。
- 动态价格计算、预算预留和财务对账。
- 配置热更新、灰度发布和影子流量。

先完成可验证的纵向链路，再扩展成本、安全和运维能力。
