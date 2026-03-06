# ADR-066: SSE MCP Transport

**Status**: Accepted, Deployed
**Date**: 2026-02-28
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-059 (Shared Brain Google Cloud), ADR-060 (Shared Brain Capabilities), ADR-062 (Brainpedia), ADR-063 (WASM Executable Nodes), ADR-064 (Pi Brain Infrastructure)

## 1. Context

The Shared Brain's primary MCP interface is stdio-based: the `mcp-brain` crate runs as a local process within a Claude Code session, communicating via JSON-RPC over stdin/stdout. This works well for local sessions but cannot serve remote clients — browsers, CI pipelines, or Claude Code sessions that want to connect without installing a local binary.

MCP over Server-Sent Events (SSE) solves this. The client opens a long-lived `GET /sse` connection to receive server-pushed events, and sends tool calls via `POST /messages?sessionId=<uuid>`. This is a standard MCP transport that Claude Code supports natively via `claude mcp add <name> --url <sse-url>`.

The SSE transport is hosted on the same `mcp-brain-server` binary that serves the REST API (ADR-064), at `pi.ruv.io/sse`. No additional infrastructure is required.

## 2. Decision

Implement MCP SSE transport as two routes (`/sse` and `/messages`) on the existing `mcp-brain-server` axum router. Tool calls received via SSE are proxied to the REST API endpoints via HTTP loopback, reusing all existing authentication, rate limiting, and verification logic. Session state is managed via `DashMap<String, mpsc::Sender<String>>`.

## 3. Architecture

### 3.1 Protocol Flow

```
Client                                Server (pi.ruv.io)
  |                                       |
  |  GET /sse                             |
  |-------------------------------------->|
  |  event: endpoint                      |
  |  data: /messages?sessionId=<uuid>     |
  |<--------------------------------------|
  |                                       |
  |  POST /messages?sessionId=<uuid>      |
  |  {"jsonrpc":"2.0","method":"initialize",...}
  |-------------------------------------->|
  |  event: message                       |
  |  data: {"jsonrpc":"2.0","result":{...}}
  |<--------------------------------------|
  |                                       |
  |  POST /messages?sessionId=<uuid>      |
  |  {"jsonrpc":"2.0","method":"tools/call","params":{"name":"brain_search",...}}
  |-------------------------------------->|
  |  event: message                       |
  |  data: {"jsonrpc":"2.0","result":{...}}
  |<--------------------------------------|
  |                                       |
  |  (keepalive comments)                 |
  |<--------------------------------------|
```

1. Client opens `GET /sse`. Server generates a UUID session ID, creates an `mpsc::channel`, stores the sender in the session map, and returns an SSE stream.
2. The first SSE event (`endpoint`) tells the client where to POST messages.
3. Client sends JSON-RPC requests to `POST /messages?sessionId=<uuid>`.
4. Server parses the request, dispatches to the appropriate handler, and sends the response back through the SSE channel.
5. KeepAlive comments prevent connection timeouts on proxies and load balancers.
6. On disconnect, the session is cleaned up from the `DashMap`.

### 3.2 Session Management

```rust
// In AppState
sessions: Arc<DashMap<String, tokio::sync::mpsc::Sender<String>>>
```

Each SSE connection creates a session with:
- A UUID session ID
- An `mpsc::channel(64)` for buffering responses
- A `DashMap` entry mapping session ID to the channel sender

The SSE stream reads from the channel receiver. When the client disconnects (stream drops), the cleanup closure removes the session from the map. The channel buffer of 64 prevents slow clients from blocking the server while providing backpressure.

### 3.3 MCP Protocol Implementation

The `/messages` handler implements the MCP protocol methods:

| Method | Handler | Description |
|--------|---------|-------------|
| `initialize` | Inline | Returns protocol version `2024-11-05`, server name `pi-brain`, capabilities |
| `initialized` | Inline | Acknowledgment, returns empty result |
| `notifications/initialized` | Inline | Notification acknowledgment |
| `tools/list` | `mcp_tool_definitions()` | Returns the full tool catalog |
| `tools/call` | `handle_mcp_tool_call()` | Dispatches to HTTP loopback proxy |

### 3.4 HTTP Loopback Proxy

Tool calls are not handled directly in the SSE message handler. Instead, they are proxied to the server's own REST API via HTTP loopback. This ensures that:

1. All existing middleware (CORS, rate limiting, body size limits, tracing) applies uniformly
2. Authentication and verification logic is not duplicated
3. The REST API and MCP SSE surface expose identical behavior
4. Testing is simplified — REST endpoint tests cover both transport paths

The proxy constructs an HTTP request to `http://127.0.0.1:{PORT}/v1/...` with the appropriate method, path, and body derived from the MCP tool call arguments.

### 3.5 Tool Catalog

22 tools are exposed via the SSE transport, grouped into four categories:

**Core Brain (10)**: `brain_share`, `brain_search`, `brain_get`, `brain_vote`, `brain_transfer`, `brain_drift`, `brain_partition`, `brain_list`, `brain_delete`, `brain_status`

**LoRA Sync (1)**: `brain_sync`

**Brainpedia (6, ADR-062)**: `brain_page_create`, `brain_page_get`, `brain_page_delta`, `brain_page_deltas`, `brain_page_evidence`, `brain_page_promote`

**WASM Nodes (5, ADR-063)**: `brain_node_list`, `brain_node_publish`, `brain_node_get`, `brain_node_wasm`, `brain_node_revoke`

Each tool definition includes a JSON Schema `inputSchema` specifying required and optional parameters, types, and descriptions.

## 4. Implementation

### 4.1 SSE Handler

```rust
async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    state.sessions.insert(session_id.clone(), tx);

    let stream = async_stream::stream! {
        // First event: tell client where to POST
        yield Ok(Event::default()
            .event("endpoint")
            .data(format!("/messages?sessionId={session_id}")));

        // Stream responses from the channel
        let mut rx = rx;
        while let Some(msg) = rx.recv().await {
            yield Ok(Event::default().event("message").data(msg));
        }

        // Cleanup on disconnect
        sessions_cleanup.remove(&session_id_cleanup);
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
```

### 4.2 Message Handler

```rust
async fn messages_handler(
    State(state): State<AppState>,
    Query(query): Query<McpMessageQuery>,
    body: String,
) -> StatusCode {
    let sender = match state.sessions.get(&query.session_id) {
        Some(s) => s.clone(),
        None => return StatusCode::NOT_FOUND,
    };

    let request: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            // Send JSON-RPC parse error through the SSE channel
            let _ = sender.send(error_response(-32700, e)).await;
            return StatusCode::ACCEPTED;
        }
    };

    let response = match method {
        "initialize" => { /* protocol handshake */ },
        "tools/list" => { /* return tool catalog */ },
        "tools/call" => { /* HTTP loopback proxy */ },
        _ => { /* method not found */ },
    };

    let _ = sender.send(serde_json::to_string(&response).unwrap()).await;
    StatusCode::ACCEPTED
}
```

### 4.3 Connection

Claude Code clients connect with:

```bash
claude mcp add pi --url https://pi.ruv.io/sse
```

This registers the SSE endpoint. Claude Code opens the SSE connection, reads the `endpoint` event, and sends all subsequent MCP requests to the `/messages` URL with the provided session ID.

### 4.4 CORS Configuration

The SSE endpoint shares the same CORS layer as the REST API:

| Allowed Origin | Purpose |
|----------------|---------|
| `https://brain.ruv.io` | Legacy brain domain |
| `https://pi.ruv.io` | Primary domain |
| `https://ruvbrain-875130704813.us-central1.run.app` | Direct Cloud Run URL |
| `http://localhost:8080` | Local development |
| `http://127.0.0.1:8080` | Local development (IP) |

Allowed methods: GET, POST, DELETE, OPTIONS. Allowed headers: Authorization, Content-Type, Accept.

### 4.5 KeepAlive

`Sse::new(stream).keep_alive(KeepAlive::default())` sends periodic SSE comments (`:keepalive` lines) to prevent intermediate proxies, load balancers, and Cloud Run from closing idle connections. The default interval is 15 seconds.

## 5. Consequences

### Positive

- **Zero additional infrastructure**: SSE runs on the same binary and port as the REST API
- **Native Claude Code support**: `claude mcp add --url` is the standard way to connect remote MCP servers
- **Full tool parity**: All 22 tools available via both stdio (local `mcp-brain`) and SSE (remote `pi.ruv.io`)
- **Reused middleware**: Rate limiting, CORS, authentication, and verification apply uniformly via the loopback proxy

### Negative

- **Single-direction streaming**: SSE is server-to-client only. The client must POST to a separate endpoint. WebSocket would allow bidirectional messaging but is not part of the MCP spec.
- **Session memory**: Each active SSE connection holds a channel and a `DashMap` entry. Under high concurrent connections this grows linearly. The 64-message buffer per session bounds memory per connection.
- **Cloud Run timeout**: Cloud Run has a maximum request timeout (default 5 minutes, configurable up to 60 minutes). Long-lived SSE connections that exceed this timeout are terminated. KeepAlive prevents idle disconnects, but the maximum lifetime is bounded by Cloud Run's configuration.

### Neutral

- The loopback proxy adds one local HTTP hop per tool call. On the same machine this adds sub-millisecond latency, which is negligible compared to Firestore round-trips and embedding computation.
- Error responses from the REST API are translated into JSON-RPC error format before being sent through the SSE channel.
