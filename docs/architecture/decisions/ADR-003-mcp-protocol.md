# ADR-003: MCP Server Architecture and Transport Choices

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-13 |
| **Authors** | RuVector Architecture Team |
| **Reviewers** | Architecture Review Board |
| **Supersedes** | - |
| **Related** | ADR-004-rvf-format, ADR-005-cross-platform-bindings |

## 1. Context

### 1.1 Problem Statement

RuVector needs to expose its vector database, neural computation, and cognitive capabilities to AI agents and LLM orchestrators. The challenge is choosing a protocol that:

- Enables seamless integration with AI assistants (Claude, GPT, etc.)
- Supports both local and remote deployment
- Provides a consistent API across different environments
- Maintains low latency for interactive use
- Supports streaming for long-running operations

### 1.2 Protocol Landscape

| Protocol | Latency | Streaming | Adoption | AI Integration |
|----------|---------|-----------|----------|----------------|
| REST/HTTP | Medium | Limited | Universal | Good |
| gRPC | Low | Native | Growing | Medium |
| WebSocket | Low | Native | Good | Medium |
| **MCP (stdio)** | Very Low | Native | AI-native | **Excellent** |
| GraphQL | Medium | Limited | Growing | Medium |

### 1.3 What is MCP?

The **Model Context Protocol (MCP)** is an open protocol enabling:

- **Tool Discovery**: AI agents discover available tools dynamically
- **Structured Invocation**: Type-safe tool calls with JSON schemas
- **Bidirectional Streaming**: Server can push updates to clients
- **Multi-modal Content**: Support for text, images, and structured data

MCP uses JSON-RPC 2.0 over various transports:
- **stdio**: Local process communication (lowest latency)
- **SSE (Server-Sent Events)**: HTTP-based streaming
- **WebSocket**: Bidirectional streaming

## 2. Decision

### 2.1 Adopt MCP as Primary Protocol

We implement MCP as the primary protocol for AI agent integration with three transport options:

```
                    MCP Protocol Layer (JSON-RPC 2.0)
                              |
           +------------------+------------------+
           |                  |                  |
    [STDIO Transport]   [SSE Transport]   [WebSocket Transport]
           |                  |                  |
    Local processes     HTTP clients      Real-time apps
    Claude Code, CLI    Web browsers      Dashboards
```

### 2.2 Transport Selection

| Transport | Use Case | Latency | Deployment |
|-----------|----------|---------|------------|
| **STDIO** | Claude Code, local CLI | <1ms | Process spawn |
| **SSE** | Web apps, remote agents | 5-20ms | HTTP server |
| **WebSocket** | Real-time dashboards | 2-10ms | WS server |

**Default**: STDIO for local, SSE for remote

### 2.3 Server Architecture

```
+------------------------------------------------------------------+
|                        MCP Server                                  |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+    +------------------+    +---------------+ |
|  | Protocol Layer   |    | Tool Registry    |    | Auth/Security | |
|  | (JSON-RPC 2.0)   |    | (91 tools)       |    | (API keys,    | |
|  +------------------+    +------------------+    | rate limits)  | |
|           |                       |              +---------------+ |
|           v                       v                      |         |
|  +------------------+    +------------------+            |         |
|  | Transport Layer  |    | Handler Layer    |<-----------+         |
|  | stdio/SSE/WS     |    | (business logic) |                      |
|  +------------------+    +------------------+                      |
|           |                       |                                |
|           v                       v                                |
|  +----------------------------------------------------------+     |
|  |                     Core RuVector                          |     |
|  | VectorDB | HNSW Index | Embeddings | Neural | Brain        |     |
|  +----------------------------------------------------------+     |
+------------------------------------------------------------------+
```

### 2.4 Tool Categories

The MCP server exposes 91 tools across categories:

| Category | Tool Count | Examples |
|----------|------------|----------|
| Vector Operations | 15 | `vector_insert`, `vector_search`, `vector_delete` |
| Index Management | 8 | `index_create`, `index_rebuild`, `index_stats` |
| Embeddings | 12 | `embed_text`, `embed_batch`, `similarity` |
| Brain (Collective) | 18 | `brain_share`, `brain_search`, `brain_vote` |
| Neural/SONA | 10 | `sona_adapt`, `neural_predict`, `pattern_learn` |
| RVF Format | 8 | `rvf_create`, `rvf_load`, `rvf_search` |
| Admin | 10 | `status`, `metrics`, `config` |
| Security | 10 | `audit`, `verify`, `sign` |

## 3. Implementation

### 3.1 STDIO Transport

```rust
//! MCP STDIO transport for local communication

pub struct StdioTransport {
    handler: Arc<McpHandler>,
}

impl StdioTransport {
    pub async fn run(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        tracing::info!("MCP STDIO transport started");

        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;

            if n == 0 {
                break; // EOF
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Parse JSON-RPC request
            let request: McpRequest = match serde_json::from_str(trimmed) {
                Ok(req) => req,
                Err(e) => {
                    let error_response = McpResponse::error(
                        None,
                        McpError::new(PARSE_ERROR, e.to_string()),
                    );
                    let response_json = serde_json::to_string(&error_response)?;
                    stdout.write_all(response_json.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                    continue;
                }
            };

            // Handle request
            let response = self.handler.handle_request(request).await;

            // Send response
            let response_json = serde_json::to_string(&response)?;
            stdout.write_all(response_json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }

        Ok(())
    }
}
```

### 3.2 SSE Transport

```rust
//! MCP SSE transport for HTTP-based streaming

pub struct SseTransport {
    handler: Arc<McpHandler>,
    host: String,
    port: u16,
}

impl SseTransport {
    pub async fn run(&self) -> Result<()> {
        let cors = CorsLayer::new()
            .allow_origin(AllowOrigin::predicate(|origin, _| {
                if let Ok(origin_str) = origin.to_str() {
                    origin_str.starts_with("http://127.0.0.1")
                        || origin_str.starts_with("http://localhost")
                        || origin_str.starts_with("https://")
                } else {
                    false
                }
            }))
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([CONTENT_TYPE, AUTHORIZATION]);

        let app = Router::new()
            .route("/", get(root))
            .route("/mcp", post(mcp_handler))
            .route("/mcp/sse", get(mcp_sse_stream))
            .layer(cors)
            .with_state(self.handler.clone());

        let addr = format!("{}:{}", self.host, self.port);
        let listener = TcpListener::bind(&addr).await?;

        tracing::info!("MCP SSE transport listening on http://{}", addr);
        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn mcp_sse_stream(
    State(handler): State<Arc<McpHandler>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        yield Ok(Event::default().data("connected"));

        // Keep-alive pings
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            yield Ok(Event::default().event("ping").data("keep-alive"));
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(30))
            .text("keep-alive"),
    )
}
```

### 3.3 Protocol Messages

#### Initialize Handshake

```json
// Request
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": { "listChanged": true }
    },
    "clientInfo": {
      "name": "claude-code",
      "version": "1.0.0"
    }
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": { "listChanged": false }
    },
    "serverInfo": {
      "name": "ruvector-mcp",
      "version": "2.0.0"
    }
  }
}
```

#### Tool Listing

```json
// Request
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}

// Response
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "vector_search",
        "description": "Search for similar vectors using HNSW index",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "array",
              "items": { "type": "number" },
              "description": "Query vector"
            },
            "k": {
              "type": "integer",
              "default": 10,
              "description": "Number of results"
            },
            "metric": {
              "type": "string",
              "enum": ["cosine", "euclidean", "dot"],
              "default": "cosine"
            }
          },
          "required": ["query"]
        }
      },
      // ... 90 more tools
    ]
  }
}
```

#### Tool Invocation

```json
// Request
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "brain_search",
    "arguments": {
      "query": "authentication patterns in Rust",
      "limit": 10,
      "min_quality": 0.7
    }
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"results\": [{\"id\": \"...\", \"score\": 0.95, ...}]}"
      }
    ]
  }
}
```

### 3.4 Handler Implementation

```rust
pub struct McpHandler {
    brain: Arc<Brain>,
    vector_db: Arc<VectorDB>,
    config: Config,
}

impl McpHandler {
    pub async fn handle_request(&self, request: McpRequest) -> McpResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(&request),
            "initialized" => McpResponse::success(request.id, json!({})),
            "tools/list" => self.handle_tools_list(&request),
            "tools/call" => self.handle_tools_call(&request).await,
            "shutdown" => McpResponse::success(request.id, json!({})),
            _ => McpResponse::error(
                request.id,
                McpError::new(METHOD_NOT_FOUND, format!("Unknown: {}", request.method)),
            ),
        }
    }

    async fn handle_tools_call(&self, request: &McpRequest) -> McpResponse {
        let tool_call: ToolCall = match serde_json::from_value(request.params.clone()) {
            Ok(tc) => tc,
            Err(e) => return McpResponse::error(
                request.id.clone(),
                McpError::new(INVALID_PARAMS, e.to_string()),
            ),
        };

        match tool_call.name.as_str() {
            // Vector operations
            "vector_insert" => self.vector_insert(&tool_call.arguments).await,
            "vector_search" => self.vector_search(&tool_call.arguments).await,
            "vector_delete" => self.vector_delete(&tool_call.arguments).await,

            // Brain operations
            "brain_share" => self.brain_share(&tool_call.arguments).await,
            "brain_search" => self.brain_search(&tool_call.arguments).await,
            "brain_vote" => self.brain_vote(&tool_call.arguments).await,

            // ... 85 more handlers

            _ => McpResponse::error(
                request.id.clone(),
                McpError::new(METHOD_NOT_FOUND, format!("Unknown tool: {}", tool_call.name)),
            ),
        }
    }
}
```

### 3.5 Security Considerations

```rust
pub struct McpSecurity {
    api_keys: HashMap<String, ApiKeyConfig>,
    rate_limiter: RateLimiter,
    audit_log: AuditLog,
}

impl McpSecurity {
    /// Validate API key and check permissions
    pub fn authenticate(&self, key: &str) -> Result<Identity, AuthError> {
        let config = self.api_keys.get(key)
            .ok_or(AuthError::InvalidKey)?;

        if config.is_expired() {
            return Err(AuthError::KeyExpired);
        }

        Ok(Identity {
            id: config.owner.clone(),
            permissions: config.permissions.clone(),
        })
    }

    /// Check rate limits
    pub fn check_rate_limit(&self, identity: &Identity) -> Result<(), RateLimitError> {
        self.rate_limiter.check(&identity.id)
    }

    /// Log tool invocation for audit
    pub fn audit(&self, identity: &Identity, tool: &str, success: bool) {
        self.audit_log.record(AuditEntry {
            timestamp: Utc::now(),
            identity: identity.id.clone(),
            tool: tool.to_string(),
            success,
        });
    }
}
```

## 4. Consequences

### 4.1 Benefits

1. **AI-Native Integration**: Direct support for Claude, GPT, and other LLMs
2. **Low Latency**: STDIO transport adds <1ms overhead
3. **Streaming Support**: SSE enables real-time updates for long operations
4. **Type Safety**: JSON Schema for all tools ensures valid inputs
5. **Discoverability**: Tools/list enables dynamic capability discovery
6. **Security**: Built-in auth, rate limiting, and audit logging

### 4.2 Costs

1. **Protocol Lock-in**: MCP is relatively new, may evolve
2. **Complexity**: Three transport implementations to maintain
3. **JSON Overhead**: Not as efficient as binary protocols for large data
4. **Testing**: Need to test all transports independently

### 4.3 Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| REST API | No streaming, higher latency, less AI integration |
| gRPC | Heavier, requires protobuf, overkill for tool calls |
| GraphQL | Query complexity overkill, no streaming subscription |
| Custom protocol | No ecosystem, need custom clients |
| WebSocket only | SSE simpler for request-response |

## 5. Performance

### 5.1 Latency Benchmarks

| Transport | Request-Response | Streaming Start |
|-----------|------------------|-----------------|
| STDIO | 0.3ms | N/A |
| SSE (localhost) | 2.1ms | 5ms |
| SSE (remote) | 15-50ms | 20-80ms |
| WebSocket | 1.5ms | 3ms |

### 5.2 Throughput

| Configuration | Requests/sec | Notes |
|---------------|--------------|-------|
| STDIO (single) | 3,000 | Sequential |
| SSE (10 connections) | 8,000 | Parallel |
| SSE (100 connections) | 25,000 | Axum async |

## 6. Related Decisions

- **ADR-004-rvf-format**: RVF files can be accessed via MCP tools
- **ADR-005-cross-platform-bindings**: MCP server available in WASM
- **ADR-007-differential-privacy**: Brain tools respect privacy budgets

## 7. References

1. MCP Specification: https://modelcontextprotocol.io/
2. JSON-RPC 2.0: https://www.jsonrpc.org/specification
3. SSE Specification: https://html.spec.whatwg.org/multipage/server-sent-events.html
4. Implementation: `/crates/mcp-brain/`, `/crates/ruvector-cli/src/mcp/`

## 8. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-13 | Architecture Team | Initial decision record |
