# ADR-112: rvAgent MCP Server with SSE and stdio Transports

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Implemented                                     |
| **Date**    | 2026-03-15                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents), ADR-108 (rvAgent-ruvbot), ADR-111 (RuVocal) |
| **Related** | ADR-104 (MCP Skills), ADR-105 (MCP Implementation) |

## Context

The rvAgent framework requires a standalone MCP server binary that:

1. **Supports multiple transports**: stdio (for Claude Code) and SSE (for web clients)
2. **Provides tool groups**: Organize 46+ tools into logical categories
3. **Offers flexible filtering**: CLI args for selecting tool groups or all tools
4. **Integrates with RuVocal**: Direct MCP connection for ADR-111

### Current State

- `rvagent-mcp` crate exists with:
  - ✅ `StdioTransport` - Basic implementation
  - ✅ `MemoryTransport` - Testing
  - ✅ `McpServer` - Request handling
  - ✅ `McpToolRegistry` - Tool registration
  - ❌ `SseTransport` - Missing
  - ❌ CLI binary - Missing
  - ❌ Tool groups - Missing

### Requirements

1. **SSE Transport**: HTTP Server-Sent Events for web clients
2. **stdio Transport**: NDJSON over stdin/stdout for CLI integration
3. **Tool Groups**: Categorize tools for selective exposure
4. **CLI Arguments**: Transport selection, port, tool filtering
5. **All Tools Option**: Expose entire registry without filtering

---

## Decision

Implement a full-featured MCP server binary with:

### 1. Transport Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    rvagent-mcp binary                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CLI Args: --transport <stdio|sse> --port 9000 --groups file,sh │
│                                                                  │
│  ┌─────────────────┐        ┌─────────────────┐                 │
│  │  StdioTransport │        │   SseTransport  │                 │
│  │                 │        │                 │                 │
│  │  stdin ──▶ req  │        │  HTTP POST ──▶  │                 │
│  │  stdout ◀── res │        │  SSE stream ◀── │                 │
│  └────────┬────────┘        └────────┬────────┘                 │
│           │                          │                           │
│           └──────────┬───────────────┘                           │
│                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    McpServer                                 ││
│  │                                                              ││
│  │   • initialize / ping                                        ││
│  │   • tools/list / tools/call                                  ││
│  │   • resources/list / resources/read                          ││
│  │   • prompts/list / prompts/get                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                      │                                           │
│                      ▼                                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              McpToolRegistry (grouped)                       ││
│  │                                                              ││
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ ││
│  │   │  file    │  │  shell   │  │  memory  │  │   agent     │ ││
│  │   │  group   │  │  group   │  │  group   │  │   group     │ ││
│  │   │          │  │          │  │          │  │             │ ││
│  │   │ read     │  │ execute  │  │ search   │  │ spawn       │ ││
│  │   │ write    │  │ bash     │  │ store    │  │ orchestrate │ ││
│  │   │ edit     │  │ run      │  │ retrieve │  │ status      │ ││
│  │   │ ls       │  │          │  │          │  │             │ ││
│  │   │ glob     │  │          │  │          │  │             │ ││
│  │   │ grep     │  │          │  │          │  │             │ ││
│  │   └──────────┘  └──────────┘  └──────────┘  └─────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2. CLI Interface

```bash
# stdio mode (default for Claude Code integration)
rvagent-mcp --transport stdio

# SSE mode with port
rvagent-mcp --transport sse --port 9000

# Filter by tool groups
rvagent-mcp --transport sse --groups file,shell,memory

# Expose all tools
rvagent-mcp --transport sse --all

# With logging
rvagent-mcp --transport sse --port 9000 --log-level debug

# Help
rvagent-mcp --help
```

### 3. Tool Groups

| Group | Tools | Description |
|-------|-------|-------------|
| `file` | read_file, write_file, edit_file, ls, glob, grep | File system operations |
| `shell` | execute, bash | Command execution |
| `memory` | semantic_search, store_memory, retrieve_memory | Vector memory |
| `agent` | spawn_agent, agent_status, orchestrate | Multi-agent |
| `git` | git_status, git_commit, git_diff, git_log | Version control |
| `web` | web_fetch, web_search | Web operations |
| `brain` | brain_search, brain_share, brain_vote | π Brain integration |
| `task` | create_task, list_tasks, complete_task | Task management |

### 4. SSE Protocol

```
# Client connects
GET /sse HTTP/1.1
Accept: text/event-stream

# Server sends events
event: message
data: {"jsonrpc":"2.0","id":1,"result":{...}}

# Client sends requests via POST
POST /message HTTP/1.1
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
```

### 5. Implementation

#### Cargo.toml additions

```toml
[[bin]]
name = "rvagent-mcp"
path = "src/main.rs"

[dependencies]
clap = { version = "4.4", features = ["derive"] }
axum = { version = "0.7", features = ["tokio"] }
tokio-stream = "0.1"
tower-http = { version = "0.5", features = ["cors"] }
```

#### main.rs structure

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "rvagent-mcp")]
#[command(about = "rvAgent MCP Server")]
struct Cli {
    /// Transport type
    #[arg(short, long, default_value = "stdio")]
    transport: Transport,

    /// Port for SSE server
    #[arg(short, long, default_value = "9000")]
    port: u16,

    /// Tool groups to expose
    #[arg(short, long, value_delimiter = ',')]
    groups: Option<Vec<String>>,

    /// Expose all tools
    #[arg(long)]
    all: bool,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}
```

---

## Tool Group Definitions

```rust
pub enum ToolGroup {
    File,    // read, write, edit, ls, glob, grep
    Shell,   // execute, bash
    Memory,  // semantic_search, store, retrieve
    Agent,   // spawn, status, orchestrate
    Git,     // status, commit, diff, log
    Web,     // fetch, search
    Brain,   // search, share, vote
    Task,    // create, list, complete
    All,     // Everything
}

impl ToolGroup {
    pub fn tools(&self) -> &[&str] {
        match self {
            Self::File => &["read_file", "write_file", "edit_file", "ls", "glob", "grep"],
            Self::Shell => &["execute", "bash"],
            Self::Memory => &["semantic_search", "store_memory", "retrieve_memory"],
            Self::Agent => &["spawn_agent", "agent_status", "orchestrate"],
            Self::Git => &["git_status", "git_commit", "git_diff", "git_log"],
            Self::Web => &["web_fetch", "web_search"],
            Self::Brain => &["brain_search", "brain_share", "brain_vote"],
            Self::Task => &["create_task", "list_tasks", "complete_task"],
            Self::All => &[], // Special case: include everything
        }
    }
}
```

---

## Consequences

### Positive

1. **Claude Code Integration**: stdio transport works natively
2. **Web Client Support**: SSE enables RuVocal direct connection
3. **Selective Exposure**: Tool groups limit attack surface
4. **Flexibility**: CLI args for different deployment scenarios
5. **Standards Compliance**: MCP protocol compatible

### Negative

1. **Binary Size**: axum adds ~2MB to binary
2. **Complexity**: Two transport implementations to maintain
3. **Port Allocation**: SSE requires available port

### Risks

1. **SSE Timeout**: Long-running connections may disconnect
2. **CORS Issues**: Browser security may block SSE
3. **Memory**: Many concurrent SSE clients consume RAM

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| CLI binary | ✅ Complete | `src/main.rs` |
| SseTransport | ✅ Complete | `src/transport.rs` |
| Tool groups | ✅ Complete | `src/groups.rs` |
| stdio mode | ✅ Complete | `src/transport.rs` |
| Integration tests | ✅ Complete | `tests/` |

---

## Usage Examples

### Claude Code Integration

```json
{
  "mcpServers": {
    "rvagent": {
      "command": "rvagent-mcp",
      "args": ["--transport", "stdio", "--groups", "file,shell"]
    }
  }
}
```

### RuVocal Connection

```typescript
const sse = new EventSource('http://localhost:9000/sse');
sse.onmessage = (event) => {
  const response = JSON.parse(event.data);
  handleMcpResponse(response);
};

// Send request
fetch('http://localhost:9000/message', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'tools/list',
    params: {}
  })
});
```

### Docker Deployment

```dockerfile
FROM rust:1.75-slim
COPY --from=builder /app/target/release/rvagent-mcp /usr/local/bin/
EXPOSE 9000
CMD ["rvagent-mcp", "--transport", "sse", "--port", "9000", "--all"]
```

---

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [rvAgent Tools](../crates/rvAgent/rvagent-tools/)
- [ADR-111 RuVocal Integration](./ADR-111-ruvocal-ui-rvagent-integration.md)
