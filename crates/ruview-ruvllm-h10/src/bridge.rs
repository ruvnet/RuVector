//! hailo-ollama subprocess bridge.
//!
//! Spawns and supervises `hailo-ollama` as a child process, then proxies
//! requests via the ollama-compatible REST API.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::{Error, Result};

const OLLAMA_BASE: &str = "http://127.0.0.1:8000";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const STARTUP_POLL: Duration = Duration::from_millis(300);

// ──────────────────────────────────────────────────────── types

#[derive(Debug, Serialize)]
pub struct OllamaGenerateReq<'a> {
    pub model:   &'a str,
    pub prompt:  &'a str,
    pub stream:  bool,
    pub options: OllamaOptions,
}

#[derive(Debug, Serialize)]
pub struct OllamaOptions {
    pub num_predict: i32,
    pub temperature: f32,
}

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateChunk {
    pub response:  String,
    pub done:      bool,
    #[serde(default)]
    pub eval_count: u64,      // total tokens generated (final chunk only)
    #[serde(default)]
    pub eval_duration: u64,   // nanoseconds (final chunk only)
}

#[derive(Debug, Deserialize)]
pub struct OllamaPullResp {
    pub status: String,
}

// ──────────────────────────────────────────────────────── stats

#[derive(Default)]
pub struct BridgeStats {
    pub tokens_generated: AtomicU64,
    pub requests:         AtomicU64,
}

impl BridgeStats {
    pub fn tok_per_sec_window(&self, elapsed: Duration) -> f32 {
        let toks = self.tokens_generated.load(Ordering::Relaxed);
        let secs = elapsed.as_secs_f32();
        if secs > 0.0 { toks as f32 / secs } else { 0.0 }
    }
}

// ──────────────────────────────────────────────────────── bridge

pub struct HailoOllamaBridge {
    _child:  Mutex<Child>,
    client:  Client,
    model:   String,
    started: Instant,
    pub stats: Arc<BridgeStats>,
}

impl HailoOllamaBridge {
    /// Spawn hailo-ollama and wait for it to become ready.
    pub async fn spawn(model: impl Into<String>) -> Result<Self> {
        let model = model.into();
        tracing::info!(%model, "spawning hailo-ollama");

        let child = Command::new("hailo-ollama")
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| Error::Bridge(format!("hailo-ollama spawn failed: {e}")))?;

        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| Error::Bridge(e.to_string()))?;

        // Wait for hailo-ollama to open its HTTP port.
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            if Instant::now() > deadline {
                return Err(Error::Bridge("hailo-ollama did not start in 30s".into()));
            }
            match client.get(format!("{OLLAMA_BASE}/api/tags")).send().await {
                Ok(r) if r.status().is_success() => break,
                _ => tokio::time::sleep(STARTUP_POLL).await,
            }
        }

        tracing::info!("hailo-ollama ready");

        Ok(Self {
            _child: Mutex::new(child),
            client,
            model,
            started: Instant::now(),
            stats: Arc::new(BridgeStats::default()),
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }

    /// Pull a model (idempotent; no-op if already present).
    /// hailo-ollama requires `{"model": "name:tag", "insecure": false}` (not the standard ollama format).
    pub async fn pull(&self, model: &str) -> Result<()> {
        tracing::info!(%model, "pulling model from hailo library");
        // Drain the streaming progress response; last line is {"status":"success"}.
        let mut resp = self
            .client
            .post(format!("{OLLAMA_BASE}/api/pull"))
            .json(&serde_json::json!({"model": model, "insecure": false}))
            .send()
            .await
            .map_err(|e| Error::Bridge(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(Error::Bridge(format!(
                "pull failed HTTP {}",
                resp.status()
            )));
        }

        // Stream and discard progress chunks; log periodic updates.
        let mut total_bytes = 0u64;
        while let Some(chunk) = resp.chunk().await.map_err(|e| Error::Bridge(e.to_string()))? {
            total_bytes += chunk.len() as u64;
            if total_bytes % (50 * 1024 * 1024) == 0 {
                tracing::info!(%model, mb = total_bytes / (1024 * 1024), "pull progress");
            }
        }
        tracing::info!(%model, "model pull complete");
        Ok(())
    }

    /// Stream tokens from hailo-ollama.
    /// Yields `(token_text, done, latency_us_from_request_start)`.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: i32,
        temperature: f32,
        tx: tokio::sync::mpsc::Sender<(String, bool, i64)>,
    ) -> Result<()> {
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();

        let body = OllamaGenerateReq {
            model:   &self.model,
            prompt,
            stream:  true,
            options: OllamaOptions { num_predict: max_tokens, temperature },
        };

        let mut resp = self
            .client
            .post(format!("{OLLAMA_BASE}/api/generate"))
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Bridge(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(Error::Bridge(format!(
                "generate HTTP {}",
                resp.status()
            )));
        }

        // hailo-ollama streams one JSON object per line.
        let mut buf = Vec::new();
        loop {
            let Some(chunk) = resp.chunk().await.map_err(|e| Error::Bridge(e.to_string()))? else {
                break;
            };
            buf.extend_from_slice(&chunk);

            // Process complete newline-delimited JSON objects.
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let trimmed = line.trim_ascii();
                if trimmed.is_empty() { continue; }

                match serde_json::from_slice::<OllamaGenerateChunk>(trimmed) {
                    Ok(c) => {
                        let latency = t0.elapsed().as_micros() as i64;
                        if c.done && c.eval_count > 0 {
                            self.stats.tokens_generated.fetch_add(c.eval_count, Ordering::Relaxed);
                        }
                        if tx.send((c.response, c.done, latency)).await.is_err() {
                            return Ok(()); // receiver dropped
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "hailo-ollama chunk parse error");
                    }
                }
            }
        }
        Ok(())
    }

    /// Check if /dev/hailo0 is present and hailo-ollama is healthy.
    pub async fn health_check(&self) -> (bool, String) {
        let dev_ok = std::path::Path::new("/dev/hailo0").exists();
        let api_ok = self
            .client
            .get(format!("{OLLAMA_BASE}/api/tags"))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        (dev_ok && api_ok, "hailo10h".to_string())
    }
}
