# Claude Code CLI: Models and API Integration

## Supported Models

### Model References Found in Source

| Model ID | Family | Notes |
|----------|--------|-------|
| `claude-opus-4-6` | Opus | Latest (current default for complex) |
| `claude-opus-4-5` | Opus | |
| `claude-opus-4-5-20251101` | Opus | Dated release |
| `claude-opus-4-1` | Opus | |
| `claude-opus-4-1-20250805` | Opus | Dated release |
| `claude-opus-4-0` | Opus | |
| `claude-opus-4` | Opus | Alias |
| `claude-opus-4-20250514` | Opus | Dated release |
| `claude-4-opus-20250514` | Opus | Legacy naming |
| `claude-sonnet-4-6` | Sonnet | Latest Sonnet |
| `claude-sonnet-4-5` | Sonnet | |
| `claude-sonnet-4-5-20250929` | Sonnet | Dated release |
| `claude-sonnet-4` | Sonnet | Alias |
| `claude-sonnet-4-20250514` | Sonnet | Dated release |
| `claude-sonnet-3-7` | Sonnet | Legacy |
| `claude-3-7-sonnet-20250219` | Sonnet | Legacy naming |
| `claude-3-5-sonnet-20241022` | Sonnet | Legacy |
| `claude-3-sonnet-20240229` | Sonnet | Legacy |
| `claude-haiku-4-5` | Haiku | |
| `claude-haiku-4-5-20251001` | Haiku | Dated release |
| `claude-haiku-4` | Haiku | Alias |
| `claude-haiku-3-5` | Haiku | Legacy |
| `claude-3-5-haiku-20241022` | Haiku | Legacy naming |
| `claude-instant-1.1` | Instant | Legacy |
| `claude-instant-1.2` | Instant | Legacy |
| `claude-code-20250219` | Code | Specialized code model |
| `claude-3-opus-20240229` | Opus | Legacy |

### Model Selection

| Mechanism | Priority | Description |
|-----------|----------|-------------|
| `--model` CLI flag | Highest | Runtime override |
| `ANTHROPIC_MODEL` env var | High | Environment override |
| `model` in settings | Medium | Persistent config |
| `availableModels` allowlist | - | Restricts options |
| `mainLoopModel` | - | Internal selection |
| Built-in default | Lowest | Fallback |

### Model Aliases

Users can specify short aliases: `"opus"`, `"sonnet"`, `"haiku"`.
The `availableModels` allowlist accepts these aliases, which map to
the latest model in each family.

### Model Overrides

`modelOverrides` maps Anthropic model IDs to provider-specific IDs:

```json
{
  "modelOverrides": {
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6-v1:0"
  }
}
```

## API Integration

### Anthropic Direct API

- Endpoint: `ANTHROPIC_BASE_URL` (default: `https://api.anthropic.com`)
- Authentication: `ANTHROPIC_API_KEY` or OAuth token
- Unix socket: `ANTHROPIC_UNIX_SOCKET` for local proxying

### API Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `/v1/messages` | Main conversation API (streaming) |
| `/v1/messages/count_tokens` | Token counting |
| `/v1/messages/batches` | Batch processing |
| `/v1/models` | Model listing |
| `/v1/complete` | Legacy completion |
| `/v1/token` | Token validation |
| `/v1/files` | File management |
| `/v1/code/upstreamproxy/ws` | WebSocket proxy |
| `/v2/session_ingress/shttp/mcp/` | MCP session ingress |

### Provider Backends

| Provider | Client | Auth |
|----------|--------|------|
| Anthropic Direct | `Anthropic` (SDK) | API key / OAuth |
| AWS Bedrock | `BedrockClient` / `BedrockRuntimeClient` | AWS IAM |
| Google Vertex AI | Native HTTP | GCP credentials |
| Azure Foundry | Native HTTP | Azure credentials |
| Anthropic AWS | `AnthropicAws` | Hybrid auth |

### Prompt Caching

Anthropic's prompt caching reduces repeat token costs:

- `cache_control: { type: "ephemeral" }` -- Standard cache
- 1-hour cache on Bedrock (`ENABLE_PROMPT_CACHING_1H_BEDROCK`)
- Per-model disable: `DISABLE_PROMPT_CACHING_HAIKU/SONNET/OPUS`
- Cache sharing: `promptCacheSharingEnabled`
- Token tracking: `promptCacheReadTokens`, `promptCacheWriteTokens`

### Retry and Fallback

- `CLAUDE_CODE_MAX_RETRIES` -- Max API retry count
- `--fallback-model` -- Fallback model for overload (print mode only)
- `FALLBACK_FOR_ALL_PRIMARY_MODELS` -- Universal fallback
- `CLAUDE_CODE_SKIP_FAST_MODE_NETWORK_ERRORS` -- Skip on network errors
- `CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK` -- Disable non-streaming fallback

### Request Customization

- `ANTHROPIC_BETAS` / `--betas` -- Beta feature headers
- `ANTHROPIC_CUSTOM_HEADERS` -- Custom request headers
- `CLAUDE_CODE_EXTRA_BODY` -- Extra request body fields
- `CLAUDE_CODE_EXTRA_METADATA` -- Extra metadata
- `API_TIMEOUT_MS` -- Request timeout
- `CLAUDE_CODE_ATTRIBUTION_HEADER` -- Attribution header
- `CLAUDE_CODE_STALL_TIMEOUT_MS_FOR_TESTING` -- Stall detection

### Structured Output

- `--json-schema` -- Enforce JSON schema on output
- `MAX_STRUCTURED_OUTPUT_RETRIES` -- Retry on schema validation failure

### Token Management

- `MAX_THINKING_TOKENS` -- Cap thinking tokens
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS` -- Cap output tokens
- `CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS` -- File read budget
- `CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE` -- Rate limit override
- Token counting via `/v1/messages/count_tokens`

### Effort Level

- `--effort` CLI flag: `low`, `medium`, `high`, `max`
- `CLAUDE_CODE_EFFORT_LEVEL` env var
- `effortLevel` in settings
- `CLAUDE_CODE_ALWAYS_ENABLE_EFFORT` -- Always apply effort level
- `/effort` slash command -- Interactive change

### Thinking/Reasoning

- `alwaysThinkingEnabled` -- Enable extended thinking
- `CLAUDE_CODE_DISABLE_THINKING` -- Disable thinking
- `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING` -- Disable adaptive mode
- `DISABLE_INTERLEAVED_THINKING` -- Disable interleaved thinking
- `MAX_THINKING_TOKENS` -- Token budget for thinking
- `CLAUDE_CODE_DISABLE_FAST_MODE` -- Disable fast mode shortcut

### Fast Mode

Fast mode uses smaller/faster models for simple operations:
- `fastMode` setting (boolean)
- `fastModePerSessionOptIn` -- Opt-in per session
- `/fast` slash command -- Toggle
- `ANTHROPIC_SMALL_FAST_MODEL` -- Custom small model
- `ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION` -- Region for small model
