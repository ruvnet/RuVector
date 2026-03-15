//! Transport abstraction for MCP message exchange.
//!
//! Defines the [`Transport`] trait for sending and receiving JSON-RPC messages,
//! with concrete implementations for stdio and in-memory (testing) transports.

use async_trait::async_trait;
use tokio::sync::mpsc;

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
            read_timeout_ms: 30_000,            // 30s
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
        let resp = JsonRpcResponse::success(
            serde_json::json!(1),
            serde_json::json!({"tools": []}),
        );
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
        let req = JsonRpcRequest::new(42, "tools/call")
            .with_params(serde_json::json!({"name": "echo"}));
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
}
