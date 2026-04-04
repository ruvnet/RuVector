# Claude Code CLI: MCP (Model Context Protocol) Integration

## Overview

Claude Code acts as an MCP **client**, connecting to external MCP servers
to extend its tool, resource, and prompt capabilities.

## Transport Layer

Four transport types supported:

| Transport | Protocol | Use Case |
|-----------|----------|----------|
| `stdio` | stdin/stdout | Local process (most common) |
| `sse` | Server-Sent Events | Remote persistent connection |
| `streamable-http` | HTTP streaming | Modern remote transport |
| `websocket` | WebSocket | Bidirectional real-time |

## MCP Protocol Methods

Claude Code implements these MCP protocol methods:

### Core
- `initialize` -- Handshake and capability negotiation
- `ping` -- Keepalive

### Tools
- `tools/list` -- Discover available tools
- `tools/call` -- Execute a tool

### Resources
- `resources/list` -- List available resources
- `resources/read` -- Read a resource

### Prompts
- `prompts/list` -- List available prompt templates
- `prompts/get` -- Get a specific prompt

### Completions
- `completion/complete` -- Auto-completion support

### Notifications
- `notifications/initialized` -- Client initialized
- `notifications/cancelled` -- Operation cancelled
- `notifications/message` -- Status messages
- `notifications/progress` -- Progress updates

## Connection Management

### MCPConnectionManager

Central manager for all MCP server connections:

```
MCPConnectionManager
  |-- MCPConnections (active connection pool)
  |-- MCPServer definitions (from config)
  |-- MCPTool registry (discovered tools)
  |-- MCPToolOutput handling
  |-- MCPToolOverrides (user overrides)
```

### Configuration Sources

MCP servers can be defined in multiple places (priority order):

1. `--mcp-config` CLI argument (JSON file or string)
2. `--strict-mcp-config` -- Use ONLY --mcp-config, ignore all else
3. `.mcp.json` in project root
4. `.claude/settings.json` (user level)
5. `.claude/settings.local.json` (local level)
6. Managed settings (enterprise)

### Server Definition Format

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "package-name"],
      "env": { "KEY": "value" }
    }
  }
}
```

Or for remote servers:
```json
{
  "mcpServers": {
    "remote-server": {
      "url": "https://example.com/mcp",
      "transport": "sse"
    }
  }
}
```

## Tool Discovery and Deferred Loading

### Eager vs Deferred Tools

- **Eager**: Tools from MCP servers with few tools are loaded immediately
- **Deferred**: Tools from servers with many tools (like claude-flow) are
  loaded on-demand via the `ToolSearch` built-in tool

The `ToolSearch` tool allows Claude to discover deferred tools by:
1. Querying with a name or keyword
2. Getting back full JSON Schema definitions
3. Then calling those tools normally

Controlled by `ENABLE_TOOL_SEARCH` env var.

### Tool Naming Convention

MCP tools are prefixed: `mcp__<server-name>__<tool-name>`

Example: `mcp__claude-flow_alpha__swarm_init`

The `MCP_NO_PREFIX` / `CLAUDE_AGENT_SDK_MCP_NO_PREFIX` env vars can
disable this prefixing.

## OAuth for Remote MCP

Remote MCP servers can use OAuth authentication:

- `MCP_CLIENT_SECRET` -- Client secret for OAuth
- `MCP_OAUTH_CALLBACK_PORT` -- Local callback port
- `MCP_OAUTH_CLIENT_METADATA_URL` -- Client metadata URL
- `MCP_CLIENT_METADATA_URL` -- Alternative metadata URL
- `OAuthClientConfig` -- Full OAuth configuration
- `OAuthClientAuthHandler` -- Authentication handler

## Batch Connection

For performance, MCP server connections are batched:

- `MCP_SERVER_CONNECTION_BATCH_SIZE` -- Max concurrent connections
- `MCP_REMOTE_SERVER_CONNECTION_BATCH_SIZE` -- Max concurrent remote connections
- `MCP_CONNECTION_NONBLOCKING` -- Non-blocking connection mode

## MCP Output Handling

- `MCP_OUTPUT_TOKENS` -- Token budget for MCP output
- `MAX_MCP_OUTPUT_TOKENS` -- Maximum output tokens
- `MCP_LARGE_OUTPUT_FILES` -- Save large outputs to files
- `MCP_TRUNCATION_PROMPT_OVERRIDE` -- Custom truncation message

## MCP Debug

- `--mcp-debug` flag (deprecated, use `--debug`)
- `MCPDebug` -- Debug mode state
- `MCPLogsPath` -- MCP debug log location

## MCP Proxy

Claude Code can proxy MCP connections:

- `MCP_PROXY_URL` -- Upstream proxy URL
- `MCP_PROXY_PATH` -- Proxy path prefix
- Used for session ingress: `/v2/session_ingress/shttp/mcp/`

## Server Approval

Users must approve MCP servers from `.mcp.json`:

- `enabledMcpjsonServers` -- Approved server list
- `disabledMcpjsonServers` -- Rejected server list
- `enableAllProjectMcpServers` -- Auto-approve all
- `deniedMcpServers` -- Enterprise denylist
- `allowedMcpServers` -- Enterprise allowlist

## MCP Instrumentation

- `MCP_INSTR_DELTA` / `CLAUDE_CODE_MCP_INSTR_DELTA` -- Instrumentation interval
- Connected to OpenTelemetry for observability
