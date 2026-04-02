# Claude Code CLI: Dependency and Module Graph

## High-Level Architecture Graph

```
+------------------------------------------------------------------+
|                        Claude Code CLI                            |
+------------------------------------------------------------------+
|                                                                    |
|  +-----------+    +-------------+    +-----------+                |
|  | CLI Entry |    | VS Code Ext |    | Agent SDK |                |
|  | (Bun SEA) |    | (extension) |    | (library) |                |
|  +-----+-----+    +------+------+    +-----+-----+                |
|        |                  |                 |                      |
|        +------------------+-----------------+                     |
|                           |                                        |
|                    +------v------+                                 |
|                    | Main Loop   |                                 |
|                    | Engine      |                                 |
|                    +------+------+                                 |
|                           |                                        |
|        +------------------+------------------+                    |
|        |                  |                  |                     |
|  +-----v-----+    +------v------+    +------v------+             |
|  | Tool       |    | Context     |    | Permission  |             |
|  | System     |    | Manager     |    | System      |             |
|  +-----+-----+    +------+------+    +------+------+             |
|        |                  |                  |                     |
|  +-----v-----+    +------v------+    +------v------+             |
|  | MCP Client |    | Session     |    | Sandbox     |             |
|  | Framework  |    | Persistence |    | Manager     |             |
|  +-----+-----+    +------+------+    +------+------+             |
|        |                  |                  |                     |
|  +-----v-----+    +------v------+    +------v------+             |
|  | Hook       |    | Prompt      |    | Auth        |             |
|  | Engine     |    | Cache       |    | Providers   |             |
|  +-----------+    +-------------+    +-------------+             |
|                                                                    |
+------------------------------------------------------------------+
```

## Module Dependency Graph

### Core Modules

```
Main Loop Engine
  depends on:
    -> Tool System (tool dispatch, execution)
    -> Context Manager (token tracking, compaction)
    -> Permission System (tool approval)
    -> API Client (Anthropic Messages API)
    -> Session Manager (persistence, resume)
    -> Hook Engine (lifecycle events)
    -> Agent System (multi-agent coordination)
    -> Streaming Handler (SSE processing)
```

### Tool System Dependencies

```
Tool System
  depends on:
    -> Permission System (pre-execution check)
    -> Sandbox Manager (Bash tool isolation)
    -> MCP Client (external tool execution)
    -> File System (Read/Write/Edit tools)
    -> Process Manager (Bash tool, Task tool)
    -> Hook Engine (PreToolUse/PostToolUse)
  provides:
    -> Tool results to Main Loop
    -> Tool schemas to API Client
```

### MCP Client Dependencies

```
MCP Client Framework
  depends on:
    -> Transport Layer (stdio/SSE/HTTP/WebSocket)
    -> OAuth Handler (remote server auth)
    -> Configuration System (server definitions)
    -> Tool Registry (tool discovery)
  provides:
    -> Extended tools to Tool System
    -> Resources to Context Manager
    -> Prompts to Main Loop
```

### Permission System Dependencies

```
Permission System
  depends on:
    -> Configuration (rules, mode)
    -> Hook Engine (PreToolUse can override)
    -> Sandbox Manager (OS-level enforcement)
    -> UI Layer (permission prompts)
  provides:
    -> Allow/Deny decisions to Tool System
    -> Audit log to Telemetry
```

### Agent System Dependencies

```
Agent System
  depends on:
    -> Main Loop Engine (conversation execution)
    -> Tool System (tool subsets per agent)
    -> Session Manager (subagent persistence)
    -> Configuration (agent definitions)
    -> Plugin System (plugin agents)
  provides:
    -> Subagent execution to Task tool
    -> Agent selection to Main Loop
```

## External Dependencies

### Node.js Built-ins (via Bun)

```
node:fs / node:fs/promises    -- File operations
node:child_process             -- Process spawning (Bash tool)
node:path                      -- Path manipulation
node:os                        -- OS detection
node:stream                    -- Streaming
node:events                    -- Event emitter
node:crypto                    -- Cryptography
node:http / node:https         -- HTTP client
node:tls                       -- TLS
node:net                       -- TCP/Unix sockets
node:async_hooks               -- Async context tracking
node:readline                  -- Terminal input
node:tty                       -- Terminal detection
node:worker_threads            -- Worker threads
node:buffer                    -- Buffer handling
node:dns                       -- DNS resolution
node:timers/promises           -- Async timers
node:util/types                -- Type checking
node:inspector                 -- V8 inspector
node:perf_hooks                -- Performance hooks
node:string_decoder            -- String decoding
```

### Bundled Libraries

| Library | Purpose |
|---------|---------|
| Anthropic SDK | API client (`new Anthropic()`) |
| AWS SDK (Bedrock) | `BedrockClient`, `BedrockRuntimeClient` |
| AWS SDK (Cognito/SSO/STS) | Authentication |
| Google Auth Library | GCP authentication |
| Azure Identity | Azure authentication |
| Ajv | JSON Schema validation |
| Commander.js | CLI argument parsing |
| proper-lockfile | File locking |
| fast-xml-parser | XML processing |
| gRPC | gRPC client (for Google APIs) |
| OpenTelemetry SDK | Observability |
| Datadog SDK | APM integration |

### Bun-Specific APIs

```
bun:sqlite    -- SQLite (for local storage)
bun:test      -- Testing framework
Bun.serve()   -- HTTP server
Bun.file()    -- File handling
```

## Data Flow Graph

```
User Input
     |
     v
[CLI Parser / VS Code Extension]
     |
     v
[System Prompt Builder]
  - CLAUDE.md files
  - Auto-memory
  - Agent instructions
  - Tool descriptions
  - Git status
     |
     v
[Message Array Builder]
  - System prompt
  - Conversation history
  - Tool results
     |
     v
[API Client] ---> Anthropic Messages API
     |                    |
     v                    v
[Stream Handler]    [Prompt Cache]
     |
     v
[Response Processor]
  |         |           |
  v         v           v
[Text]  [Thinking]  [Tool Use]
                        |
                        v
                  [Permission Check]
                        |
                  +-----+-----+
                  |           |
                  v           v
              [Approve]   [Deny]
                  |           |
                  v           v
            [Tool Execute] [Error Response]
                  |
                  v
            [Tool Result]
                  |
                  v
            [Back to API Client for next turn]
```

## State Management Graph

```
Session State
  |-- Conversation History (messages array)
  |-- Tool Permission Cache (approved patterns)
  |-- MCP Connections (active servers)
  |-- File Checkpoints (edit history)
  |-- Auto-Memory (MEMORY.md)
  |-- Token Counter (usage tracking)
  |-- Cost Counter (spend tracking)
  |-- Agent State (active agent config)
  |-- Compact State (compaction history)
  |-- Plugin State (active plugins)
```

## Initialization Sequence

```
1. Binary boot (Bun runtime init)
2. CLI argument parsing (Commander.js)
3. Configuration loading (settings cascade)
4. Authentication (API key / OAuth / AWS / GCP / Azure)
5. CLAUDE.md discovery and loading
6. MCP server connections (batch, parallel)
7. Tool registry build (built-in + MCP + deferred)
8. Session init or resume
9. Git status collection (if in repo)
10. System prompt assembly
11. Hook system initialization
12. Plugin sync (if enabled)
13. Main loop start
```
