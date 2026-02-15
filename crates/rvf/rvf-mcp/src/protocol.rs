//! JSON-RPC 2.0 protocol handling for MCP.

use serde_json::{json, Value};

use crate::{RvfMcpServer, ToolCall};

/// JSON-RPC error codes.
#[derive(Debug)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

/// Handle a raw JSON-RPC request string. Returns a JSON-RPC response string.
pub fn handle_jsonrpc(server: &mut RvfMcpServer, request: &str) -> Result<String, String> {
    let req: Value = serde_json::from_str(request).map_err(|e| format!("parse error: {e}"))?;

    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let params = req.get("params").cloned().unwrap_or(json!({}));

    match method {
        "initialize" => {
            let response = json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "rvf-mcp",
                        "version": "0.1.0"
                    },
                    "capabilities": {
                        "tools": {}
                    }
                }
            });
            Ok(response.to_string())
        }

        "notifications/initialized" => {
            // Client acknowledgment — no response needed, but return empty for non-notification
            Ok(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {}
            })
            .to_string())
        }

        "tools/list" => {
            let tools = server.list_tools();
            let tool_list: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    })
                })
                .collect();

            Ok(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "tools": tool_list }
            })
            .to_string())
        }

        "tools/call" => {
            let name = params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

            // Check if tool exists
            let known = server.list_tools();
            if !known.iter().any(|t| t.name == name) {
                return Ok(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": format!("unknown tool: {name}")
                    }
                })
                .to_string());
            }

            let call = ToolCall { name, arguments };
            match server.handle_tool_call(call) {
                Ok(result) => {
                    let content = json!([{
                        "type": "text",
                        "text": result.text,
                    }]);
                    Ok(json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": content,
                            "isError": result.is_error,
                        }
                    })
                    .to_string())
                }
                Err(e) => Ok(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32603, "message": e }
                })
                .to_string()),
            }
        }

        _ => Ok(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32601,
                "message": format!("unknown method: {method}")
            }
        })
        .to_string()),
    }
}
