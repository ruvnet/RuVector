# Claude Code CLI: Telemetry and Observability

## Telemetry System

### OpenTelemetry Integration

Claude Code has deep OpenTelemetry (OTEL) integration for tracing,
metrics, and logging:

**OTEL Environment Variables**:

| Variable | Purpose |
|----------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint URL |
| `OTEL_EXPORTER_OTLP_HEADERS` | Authentication headers |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | Protocol (grpc/http) |
| `OTEL_EXPORTER_OTLP_INSECURE` | Allow insecure connections |
| `OTEL_TRACES_EXPORTER` | Trace exporter type |
| `OTEL_METRICS_EXPORTER` | Metrics exporter type |
| `OTEL_LOGS_EXPORTER` | Log exporter type |
| `OTEL_TRACES_EXPORT_INTERVAL` | Trace export interval |
| `OTEL_METRICS_EXPORT_INTERVAL` | Metrics export interval |
| `OTEL_LOGS_EXPORT_INTERVAL` | Log export interval |

**Batch processing**:
- `OTEL_BSP_MAX_QUEUE_SIZE` -- Span batch queue
- `OTEL_BSP_MAX_EXPORT_BATCH_SIZE` -- Span batch size
- `OTEL_BSP_EXPORT_TIMEOUT` -- Span export timeout
- `OTEL_BSP_SCHEDULE_DELAY` -- Span schedule delay
- `OTEL_BLRP_*` -- Log record processor equivalents

**Limits**:
- `OTEL_ATTRIBUTE_COUNT_LIMIT` -- Max attributes per span
- `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT` -- Max attribute value length

**Shutdown**:
- `CLAUDE_CODE_OTEL_FLUSH_TIMEOUT_MS` -- Flush timeout
- `CLAUDE_CODE_OTEL_SHUTDOWN_TIMEOUT_MS` -- Shutdown timeout

**Headers helper**:
- `otelHeadersHelper` setting -- Script to generate OTEL headers
- `CLAUDE_CODE_OTEL_HEADERS_HELPER_DEBOUNCE_MS` -- Debounce

### Datadog Integration

`Datadog` type referenced -- native Datadog APM support.
- `CLAUDE_CODE_DATADOG_FLUSH_INTERVAL_MS` -- Flush interval

### Perfetto Tracing

`CLAUDE_CODE_PERFETTO_TRACE` -- Enable Perfetto trace output for
Chrome DevTools-compatible performance profiling.

### CPU Profiling

`CLAUDE_CODE_PROFILE_STARTUP` -- Profile CLI startup performance.

### What Gets Tracked

**Telemetry flags**:
- `CLAUDE_CODE_ENABLE_TELEMETRY` -- Master telemetry switch
- `ENABLE_ENHANCED_TELEMETRY_BETA` -- Enhanced telemetry
- `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA` -- Enhanced telemetry (alt)
- `DISABLE_TELEMETRY` -- Disable all telemetry
- `OTEL_LOG_TOOL_CONTENT` -- Log tool content
- `OTEL_LOG_TOOL_DETAILS` -- Log tool details
- `OTEL_LOG_USER_PROMPTS` -- Log user prompts
- `BETA_TRACING_ENDPOINT` -- Beta tracing
- `ENABLE_BETA_TRACING_DETAILED` -- Detailed beta traces

**Metrics tracked** (inferred from patterns):
- Session lifecycle events
- Tool execution duration and outcome
- API call latency and token usage
- Model selection and fallback events
- Permission decisions
- Context compaction frequency
- Error rates and types
- Plugin/MCP server health

## Debugging

### Debug Mode

`-d, --debug [filter]` -- Enable debug with optional category filter.

Filter categories: `api`, `hooks`, `mcp`, `file`, `1p`, etc.
Prefix with `!` to exclude: `"!1p,!file"`.

### Debug Output

- `CLAUDE_CODE_DEBUG_LOGS_DIR` -- Custom debug log directory
- `--debug-file <path>` -- Write to specific file
- `CLAUDE_CODE_DEBUG_LOG_LEVEL` -- Log level
- `CLAUDE_DEBUG` -- Legacy debug flag
- `DEBUG` -- Node.js debug namespace
- `DEBUG_AUTH` -- Authentication debugging
- `DEBUG_SDK` -- SDK debugging

### Diagnostics

`claude doctor` / `/doctor` -- Built-in diagnostics:
- Check MCP server connections
- Verify authentication
- Test API connectivity
- Validate configuration
- Check permissions

### Frame Timing

`CLAUDE_CODE_FRAME_TIMING_LOG` -- Log UI frame timing for
rendering performance analysis.

### Debug Repaints

`CLAUDE_CODE_DEBUG_REPAINTS` -- Highlight UI repaints for
debugging rendering performance.

## Error Reporting

- `DISABLE_ERROR_REPORTING` -- Disable crash/error reporting
- Error context captured with session info
- Stack traces sent with telemetry (when enabled)

## Cost Tracking

- `/cost` slash command -- View session costs
- `/extra-usage` -- Detailed usage breakdown
- Token usage per turn tracked
- Cost counter: `getCostCounter`
- Commit counter: `getCommitCounter`
- Budget enforcement: `--max-budget-usd`

## Stream Watchdog

`CLAUDE_ENABLE_STREAM_WATCHDOG` -- Monitor streaming health.
`CLAUDE_STREAM_IDLE_TIMEOUT_MS` -- Detect stalled streams.

## Slow Operation Detection

`CLAUDE_CODE_SLOW_OPERATION_THRESHOLD_MS` -- Flag operations
exceeding this threshold for investigation.
