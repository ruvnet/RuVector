//! Transport abstraction for MCP message exchange.
//!
//! Defines the [`Transport`] trait for sending and receiving JSON-RPC messages,
//! with concrete implementations for stdio, SSE, and in-memory (testing) transports.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex};

use crate::protocol::{JsonRpcRequest, JsonRpcResponse};
use crate::{McpError, Result};

// ---------------------------------------------------------------------------
// Transport trait
// ---------------------------------------------------------------------------

/// Async transport for bidirectional JSON-RPC message exchange.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a JSON-RPC response.
    async fn send_response(&self, response: JsonRpcResponse) -> Result<()>;

    /// Send a JSON-RPC request (used by client).
    async fn send_request(&self, request: JsonRpcRequest) -> Result<()>;

    /// Receive the next incoming JSON-RPC request. Returns `None` on EOF.
    async fn receive_request(&self) -> Result<Option<JsonRpcRequest>>;

    /// Receive the next incoming JSON-RPC response. Returns `None` on EOF.
    async fn receive_response(&self) -> Result<Option<JsonRpcResponse>>;

    /// Close the transport.
    async fn close(&self) -> Result<()>;

    /// Send a request and wait for the corresponding response.
    ///
    /// This is a convenience method used by the MCP client.
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        self.send_request(request).await?;
        self.receive_response()
            .await?
            .ok_or_else(|| McpError::transport("connection closed before response"))
    }
}

// ---------------------------------------------------------------------------
// TransportConfig
// ---------------------------------------------------------------------------

/// Configuration for transport initialization.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Maximum message size in bytes (0 = unlimited).
    pub max_message_size: usize,
    /// Read timeout in milliseconds (0 = no timeout).
    pub read_timeout_ms: u64,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            max_message_size: 4 * 1024 * 1024, // 4MB
            read_timeout_ms: 30_000,           // 30s
        }
    }
}

// ---------------------------------------------------------------------------
// StdioTransport
// ---------------------------------------------------------------------------

/// Transport that reads JSON-RPC from stdin and writes to stdout.
///
/// Messages are newline-delimited JSON (NDJSON).
pub struct StdioTransport {
    _config: TransportConfig,
}

impl StdioTransport {
    /// Create a new stdio transport.
    pub fn new(config: TransportConfig) -> Self {
        Self { _config: config }
    }
}

#[async_trait]
impl Transport for StdioTransport {
    async fn send_response(&self, response: JsonRpcResponse) -> Result<()> {
        let json = serde_json::to_string(&response)?;
        use tokio::io::AsyncWriteExt;
        let mut stdout = tokio::io::stdout();
        stdout
            .write_all(json.as_bytes())
            .await
            .map_err(|e| McpError::transport(format!("stdout write: {}", e)))?;
        stdout
            .write_all(b"\n")
            .await
            .map_err(|e| McpError::transport(format!("stdout write: {}", e)))?;
        stdout
            .flush()
            .await
            .map_err(|e| McpError::transport(format!("stdout flush: {}", e)))?;
        Ok(())
    }

    async fn send_request(&self, request: JsonRpcRequest) -> Result<()> {
        let json = serde_json::to_string(&request)?;
        use tokio::io::AsyncWriteExt;
        let mut stdout = tokio::io::stdout();
        stdout
            .write_all(json.as_bytes())
            .await
            .map_err(|e| McpError::transport(format!("stdout write: {}", e)))?;
        stdout
            .write_all(b"\n")
            .await
            .map_err(|e| McpError::transport(format!("stdout write: {}", e)))?;
        stdout
            .flush()
            .await
            .map_err(|e| McpError::transport(format!("stdout flush: {}", e)))?;
        Ok(())
    }

    async fn receive_request(&self) -> Result<Option<JsonRpcRequest>> {
        use tokio::io::{AsyncBufReadExt, BufReader};
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .await
            .map_err(|e| McpError::transport(format!("stdin read: {}", e)))?;
        if n == 0 {
            return Ok(None);
        }
        let request: JsonRpcRequest = serde_json::from_str(line.trim())?;
        Ok(Some(request))
    }

    async fn receive_response(&self) -> Result<Option<JsonRpcResponse>> {
        use tokio::io::{AsyncBufReadExt, BufReader};
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .await
            .map_err(|e| McpError::transport(format!("stdin read: {}", e)))?;
        if n == 0 {
            return Ok(None);
        }
        let response: JsonRpcResponse = serde_json::from_str(line.trim())?;
        Ok(Some(response))
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MemoryTransport
// ---------------------------------------------------------------------------

/// In-memory transport for testing — uses tokio channels.
///
/// Create a pair with [`MemoryTransport::pair`] for client/server testing.
pub struct MemoryTransport {
    req_tx: mpsc::Sender<JsonRpcRequest>,
    req_rx: tokio::sync::Mutex<mpsc::Receiver<JsonRpcRequest>>,
    resp_tx: mpsc::Sender<JsonRpcResponse>,
    resp_rx: tokio::sync::Mutex<mpsc::Receiver<JsonRpcResponse>>,
}

impl MemoryTransport {
    /// Create a connected pair of memory transports.
    ///
    /// Messages sent as requests on `a` are received as requests on `b`,
    /// and responses sent on `b` are received as responses on `a`.
    pub fn pair(buffer: usize) -> (Self, Self) {
        let (req_tx_a, req_rx_b) = mpsc::channel(buffer);
        let (req_tx_b, req_rx_a) = mpsc::channel(buffer);
        let (resp_tx_a, resp_rx_b) = mpsc::channel(buffer);
        let (resp_tx_b, resp_rx_a) = mpsc::channel(buffer);

        let a = Self {
            req_tx: req_tx_a,
            req_rx: tokio::sync::Mutex::new(req_rx_a),
            resp_tx: resp_tx_a,
            resp_rx: tokio::sync::Mutex::new(resp_rx_a),
        };
        let b = Self {
            req_tx: req_tx_b,
            req_rx: tokio::sync::Mutex::new(req_rx_b),
            resp_tx: resp_tx_b,
            resp_rx: tokio::sync::Mutex::new(resp_rx_b),
        };
        (a, b)
    }
}

#[async_trait]
impl Transport for MemoryTransport {
    async fn send_response(&self, response: JsonRpcResponse) -> Result<()> {
        self.resp_tx
            .send(response)
            .await
            .map_err(|_| McpError::transport("response channel closed"))
    }

    async fn send_request(&self, request: JsonRpcRequest) -> Result<()> {
        self.req_tx
            .send(request)
            .await
            .map_err(|_| McpError::transport("request channel closed"))
    }

    async fn receive_request(&self) -> Result<Option<JsonRpcRequest>> {
        let mut rx = self.req_rx.lock().await;
        Ok(rx.recv().await)
    }

    async fn receive_response(&self) -> Result<Option<JsonRpcResponse>> {
        let mut rx = self.resp_rx.lock().await;
        Ok(rx.recv().await)
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SseTransport
// ---------------------------------------------------------------------------

/// SSE (Server-Sent Events) transport configuration.
#[derive(Debug, Clone)]
pub struct SseConfig {
    /// Port to listen on.
    pub port: u16,
    /// Host to bind to.
    pub host: String,
    /// Enable CORS.
    pub enable_cors: bool,
    /// Heartbeat interval in seconds.
    pub heartbeat_interval_secs: u64,
}

impl Default for SseConfig {
    fn default() -> Self {
        Self {
            port: 9000,
            host: "127.0.0.1".into(),
            enable_cors: true,
            heartbeat_interval_secs: 30,
        }
    }
}

/// SSE transport for HTTP-based MCP communication.
///
/// Clients connect via GET /sse for event stream and POST /message for requests.
pub struct SseTransport {
    config: SseConfig,
    /// Broadcast channel for sending responses to SSE clients.
    response_tx: broadcast::Sender<JsonRpcResponse>,
    /// Channel for receiving requests from POST /message.
    request_rx: Arc<Mutex<mpsc::Receiver<JsonRpcRequest>>>,
    /// Sender for incoming requests (used by HTTP handler).
    request_tx: mpsc::Sender<JsonRpcRequest>,
    /// Shutdown signal.
    shutdown_tx: Option<broadcast::Sender<()>>,
}

impl SseTransport {
    /// Create a new SSE transport.
    pub fn new(config: SseConfig) -> Self {
        let (response_tx, _) = broadcast::channel(256);
        let (request_tx, request_rx) = mpsc::channel(256);
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            config,
            response_tx,
            request_rx: Arc::new(Mutex::new(request_rx)),
            request_tx,
            shutdown_tx: Some(shutdown_tx),
        }
    }

    /// Get the response broadcast sender (for SSE event stream).
    pub fn response_sender(&self) -> broadcast::Sender<JsonRpcResponse> {
        self.response_tx.clone()
    }

    /// Get the request sender (for POST /message handler).
    pub fn request_sender(&self) -> mpsc::Sender<JsonRpcRequest> {
        self.request_tx.clone()
    }

    /// Get the configuration.
    pub fn config(&self) -> &SseConfig {
        &self.config
    }

    /// Get the shutdown sender.
    pub fn shutdown_sender(&self) -> Option<broadcast::Sender<()>> {
        self.shutdown_tx.clone()
    }
}

#[async_trait]
impl Transport for SseTransport {
    async fn send_response(&self, response: JsonRpcResponse) -> Result<()> {
        self.response_tx
            .send(response)
            .map_err(|_| McpError::transport("no SSE subscribers"))?;
        Ok(())
    }

    async fn send_request(&self, _request: JsonRpcRequest) -> Result<()> {
        // SSE transport is server-side only; clients send requests via POST
        Err(McpError::transport("SSE transport does not send requests"))
    }

    async fn receive_request(&self) -> Result<Option<JsonRpcRequest>> {
        let mut rx = self.request_rx.lock().await;
        Ok(rx.recv().await)
    }

    async fn receive_response(&self) -> Result<Option<JsonRpcResponse>> {
        // SSE transport is server-side only
        Err(McpError::transport(
            "SSE transport does not receive responses",
        ))
    }

    async fn close(&self) -> Result<()> {
        if let Some(ref shutdown_tx) = self.shutdown_tx {
            let _ = shutdown_tx.send(());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TransportType enum
// ---------------------------------------------------------------------------

/// Transport type for CLI selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    /// stdio transport (NDJSON over stdin/stdout).
    Stdio,
    /// SSE transport (HTTP Server-Sent Events).
    Sse,
}

impl std::str::FromStr for TransportType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "stdio" | "std" => Ok(Self::Stdio),
            "sse" | "http" | "web" => Ok(Self::Sse),
            _ => Err(format!("unknown transport: {}", s)),
        }
    }
}

impl std::fmt::Display for TransportType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stdio => write!(f, "stdio"),
            Self::Sse => write!(f, "sse"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_transport_request_roundtrip() {
        let (client, server) = MemoryTransport::pair(16);
        let req = JsonRpcRequest::new(1, "tools/list");
        client.send_request(req).await.unwrap();
        let received = server.receive_request().await.unwrap().unwrap();
        assert_eq!(received.method, "tools/list");
    }

    #[tokio::test]
    async fn test_memory_transport_response_roundtrip() {
        let (client, server) = MemoryTransport::pair(16);
        let resp = JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({"tools": []}));
        server.send_response(resp).await.unwrap();
        let received = client.receive_response().await.unwrap().unwrap();
        assert!(received.result.is_some());
    }

    #[tokio::test]
    async fn test_memory_transport_multiple_messages() {
        let (client, server) = MemoryTransport::pair(16);
        for i in 0..5 {
            client
                .send_request(JsonRpcRequest::new(i, "ping"))
                .await
                .unwrap();
        }
        for i in 0..5 {
            let req = server.receive_request().await.unwrap().unwrap();
            assert_eq!(req.id, serde_json::json!(i));
        }
    }

    #[tokio::test]
    async fn test_memory_transport_bidirectional() {
        let (a, b) = MemoryTransport::pair(16);
        a.send_request(JsonRpcRequest::new(1, "ping"))
            .await
            .unwrap();
        let req = b.receive_request().await.unwrap().unwrap();
        assert_eq!(req.method, "ping");
        b.send_response(JsonRpcResponse::success(
            serde_json::json!(1),
            serde_json::json!("pong"),
        ))
        .await
        .unwrap();
        let resp = a.receive_response().await.unwrap().unwrap();
        assert_eq!(resp.result.unwrap(), serde_json::json!("pong"));
    }

    #[tokio::test]
    async fn test_memory_transport_close() {
        let (a, _b) = MemoryTransport::pair(16);
        assert!(a.close().await.is_ok());
    }

    #[test]
    fn test_transport_config_default() {
        let config = TransportConfig::default();
        assert_eq!(config.max_message_size, 4 * 1024 * 1024);
        assert_eq!(config.read_timeout_ms, 30_000);
    }

    #[tokio::test]
    async fn test_memory_transport_drop_sender_returns_none() {
        let (client, server) = MemoryTransport::pair(16);
        drop(client);
        let result = server.receive_request().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_memory_transport_request_with_params() {
        let (client, server) = MemoryTransport::pair(16);
        let req =
            JsonRpcRequest::new(42, "tools/call").with_params(serde_json::json!({"name": "echo"}));
        client.send_request(req).await.unwrap();
        let received = server.receive_request().await.unwrap().unwrap();
        assert_eq!(received.method, "tools/call");
        assert!(received.params.is_some());
    }

    #[tokio::test]
    async fn test_memory_transport_error_response() {
        let (client, server) = MemoryTransport::pair(16);
        let resp = JsonRpcResponse::error(
            serde_json::json!(1),
            crate::protocol::JsonRpcError::method_not_found("nope"),
        );
        server.send_response(resp).await.unwrap();
        let received = client.receive_response().await.unwrap().unwrap();
        assert!(received.error.is_some());
        assert_eq!(received.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_stdio_transport_creation() {
        let _transport = StdioTransport::new(TransportConfig::default());
    }

    #[tokio::test]
    async fn test_stdio_transport_close() {
        let transport = StdioTransport::new(TransportConfig::default());
        assert!(transport.close().await.is_ok());
    }

    #[test]
    fn test_sse_config_default() {
        let config = SseConfig::default();
        assert_eq!(config.port, 9000);
        assert_eq!(config.host, "127.0.0.1");
        assert!(config.enable_cors);
        assert_eq!(config.heartbeat_interval_secs, 30);
    }

    #[tokio::test]
    async fn test_sse_transport_creation() {
        let transport = SseTransport::new(SseConfig::default());
        assert_eq!(transport.config().port, 9000);
    }

    #[tokio::test]
    async fn test_sse_transport_send_response() {
        let transport = SseTransport::new(SseConfig::default());
        let mut rx = transport.response_sender().subscribe();

        let resp =
            JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({"status": "ok"}));
        transport.send_response(resp.clone()).await.unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.id, serde_json::json!(1));
    }

    #[tokio::test]
    async fn test_sse_transport_receive_request() {
        let transport = SseTransport::new(SseConfig::default());
        let req_tx = transport.request_sender();

        let req = JsonRpcRequest::new(42, "ping");
        req_tx.send(req).await.unwrap();

        let received = transport.receive_request().await.unwrap().unwrap();
        assert_eq!(received.method, "ping");
        assert_eq!(received.id, serde_json::json!(42));
    }

    #[tokio::test]
    async fn test_sse_transport_send_request_fails() {
        let transport = SseTransport::new(SseConfig::default());
        let req = JsonRpcRequest::new(1, "ping");
        assert!(transport.send_request(req).await.is_err());
    }

    #[tokio::test]
    async fn test_sse_transport_receive_response_fails() {
        let transport = SseTransport::new(SseConfig::default());
        assert!(transport.receive_response().await.is_err());
    }

    #[tokio::test]
    async fn test_sse_transport_close() {
        let transport = SseTransport::new(SseConfig::default());
        assert!(transport.close().await.is_ok());
    }

    #[test]
    fn test_transport_type_from_str() {
        assert_eq!(
            "stdio".parse::<TransportType>().unwrap(),
            TransportType::Stdio
        );
        assert_eq!("sse".parse::<TransportType>().unwrap(), TransportType::Sse);
        assert_eq!("http".parse::<TransportType>().unwrap(), TransportType::Sse);
        assert_eq!("web".parse::<TransportType>().unwrap(), TransportType::Sse);
    }

    #[test]
    fn test_transport_type_from_str_invalid() {
        assert!("invalid".parse::<TransportType>().is_err());
    }

    #[test]
    fn test_transport_type_display() {
        assert_eq!(format!("{}", TransportType::Stdio), "stdio");
        assert_eq!(format!("{}", TransportType::Sse), "sse");
    }
}
