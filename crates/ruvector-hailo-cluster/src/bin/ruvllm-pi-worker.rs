//! `ruvllm-pi-worker` — per-Pi LLM completion worker (ADR-179).
//!
//! Sibling worker on each Pi 5 to `ruvector-hailo-worker` (ADR-167):
//! - hailo worker  → :50051  → embeddings via Hailo-8 NPU
//! - ruvllm worker → :50053  → completions via Cortex-A76 + pi_quant
//!
//! ## Build
//!
//! Pi 5 cross-build (workstation):
//! ```text
//! RUSTFLAGS= cargo build \
//!   --target aarch64-unknown-linux-gnu --release \
//!   -p ruvector-hailo-cluster \
//!   --no-default-features --features ruvllm-engine \
//!   --bin ruvllm-pi-worker
//! ```
//!
//! Without `--features ruvllm-engine`, the bin still builds but only
//! exposes the iter-3 scaffold (TCP banner, no LLM). Useful for the
//! deploy-pipeline tests we ran in iter 4.
//!
//! ## Env vars
//!
//! ```text
//! RUVLLM_WORKER_BIND          listen socket   (default 0.0.0.0:50053)
//! RUVLLM_MODEL_PATH           local model directory containing
//!                             config.json + tokenizer.json + model.safetensors
//!                             (e.g. /var/lib/ruvllm/models/qwen2.5-0.5b)
//!                             — no hf-hub download (cross-build constraint, ADR-179 iter 8)
//! RUVLLM_QUANTIZE             pi_quant | turbo_quant | quip | bitnet158 | none
//!                             (iter 9 wires `none` only — fp16 reference; quant lands iter 10+)
//! RUVLLM_KV_QUANTIZE          rabitq | none  (iter 9: none)
//! RUVLLM_MAX_INFLIGHT         scheduler concurrent requests (default 4)
//! RUVLLM_MAX_SEQ              max prompt+completion tokens (default 2048)
//! RUVLLM_LOG_PROMPT_AUDIT     none | hash | full  (default none)
//! ```
//!
//! ## Wire surface (iter 9)
//!
//! Plain TCP request/response (no gRPC yet — that's iter 11):
//! - newline-delimited JSON request: `{"prompt":"...", "max_tokens":N}`
//! - newline-delimited JSON response: `{"text":"...", "tokens":N, "ms":N}`
//!
//! Lets the iter-12 `ruvllm-cluster-bench` (mirror of hailo bench) drive
//! the cluster with simple line-oriented IO; gRPC `Completion` proto
//! lands in iter 11 once we lock the request shape.

use std::env;
use std::net::SocketAddr;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "ruvllm-engine")]
const GATE: &str = "ADR-179 iter 9 — engine wired (qwen2.5-0.5b fp16 first)";
#[cfg(not(feature = "ruvllm-engine"))]
const GATE: &str = "ADR-179 iter 3 — scaffold (no engine; build with --features ruvllm-engine)";

fn parse_bind() -> SocketAddr {
    env::var("RUVLLM_WORKER_BIND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| "0.0.0.0:50053".parse().unwrap())
}

fn read_optional_env(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| "<unset>".to_string())
}

#[cfg(feature = "ruvllm-engine")]
mod engine {
    use ruvllm::backends::{CandleBackend, GenerateParams, LlmBackend, ModelConfig};
    use std::sync::Mutex;

    /// Thread-safe wrapper around CandleBackend. ServingEngine has its
    /// own scheduler+batcher, but iter 9 ships a simpler single-shot
    /// generate path — one request at a time — so we can prove the
    /// loop closes end-to-end. Iter 10 swaps this for ServingEngine.
    pub struct PiEngine {
        inner: Mutex<CandleBackend>,
    }

    impl PiEngine {
        pub fn load(model_path: &str) -> anyhow::Result<Self> {
            let mut backend = CandleBackend::new()
                .map_err(|e| anyhow::anyhow!("CandleBackend::new failed: {e:?}"))?;
            // ADR-179 iter 10: candle-transformers' Llama path panics
            // with "compile with '--features flash-attn'" on CPU when
            // use_flash_attn=true. The flag is intended to gate
            // CUDA-only kernels — set false so the model uses the
            // standard candle attention. We also disable quantization
            // here so iter 10 first-light is fp16; pi_quant lands iter 11.
            let config = ModelConfig {
                use_flash_attention: false,
                quantization: None,
                ..ModelConfig::default()
            };
            backend
                .load_model(model_path, config)
                .map_err(|e| anyhow::anyhow!("load_model({model_path}) failed: {e:?}"))?;
            Ok(Self {
                inner: Mutex::new(backend),
            })
        }

        pub fn generate(&self, prompt: &str, max_tokens: usize) -> anyhow::Result<String> {
            let backend = self
                .inner
                .lock()
                .map_err(|_| anyhow::anyhow!("engine mutex poisoned"))?;
            let params = GenerateParams {
                max_tokens,
                ..Default::default()
            };
            backend
                .generate(prompt, params)
                .map_err(|e| anyhow::anyhow!("generate failed: {e:?}"))
        }
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ruvllm_pi_worker=info".into()),
        )
        .init();

    let bind = parse_bind();
    let model_path = env::var("RUVLLM_MODEL_PATH").unwrap_or_default();

    tracing::info!(
        version = VERSION,
        gate = GATE,
        %bind,
        model_path = %if model_path.is_empty() { "<unset>".into() } else { model_path.clone() },
        quantize = %read_optional_env("RUVLLM_QUANTIZE"),
        max_inflight = %read_optional_env("RUVLLM_MAX_INFLIGHT"),
        "ruvllm-pi-worker starting"
    );

    #[cfg(feature = "ruvllm-engine")]
    let engine = if model_path.is_empty() {
        tracing::warn!("RUVLLM_MODEL_PATH is unset; refusing to start engine, falling back to scaffold mode");
        None
    } else {
        tracing::info!("loading model from {} ...", model_path);
        match engine::PiEngine::load(&model_path) {
            Ok(e) => {
                tracing::info!("model loaded, ready to serve");
                Some(std::sync::Arc::new(e))
            }
            Err(e) => {
                tracing::error!(error = %e, "model load failed; falling back to scaffold mode");
                None
            }
        }
    };

    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(%bind, "ruvllm-pi-worker listening");

    loop {
        let (sock, peer) = listener.accept().await?;
        #[cfg(feature = "ruvllm-engine")]
        let engine_clone = engine.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_conn(
                sock,
                peer,
                #[cfg(feature = "ruvllm-engine")]
                engine_clone,
            )
            .await
            {
                tracing::warn!(%peer, error = %e, "conn handler error");
            }
        });
    }
}

async fn handle_conn(
    mut sock: tokio::net::TcpStream,
    peer: std::net::SocketAddr,
    #[cfg(feature = "ruvllm-engine")] engine: Option<std::sync::Arc<engine::PiEngine>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    tracing::debug!(%peer, "conn open");
    let (rx, mut tx) = sock.split();
    let mut lines = BufReader::new(rx).lines();

    while let Some(line) = lines.next_line().await? {
        if line.is_empty() {
            continue;
        }
        let req: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                let banner = format!(
                    "ruvllm-pi-worker v{} — {}\n",
                    VERSION, GATE
                );
                tx.write_all(banner.as_bytes()).await?;
                continue;
            }
        };
        let prompt = req.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
        let max_tokens = req
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;

        let started = std::time::Instant::now();
        #[cfg(feature = "ruvllm-engine")]
        let resp = match &engine {
            Some(e) => match e.generate(prompt, max_tokens) {
                Ok(text) => {
                    let ms = started.elapsed().as_millis() as u64;
                    serde_json::json!({"text": text, "tokens": text.len(), "ms": ms})
                }
                Err(e) => serde_json::json!({"error": format!("{e:#}")}),
            },
            None => {
                serde_json::json!({"error": "engine not loaded; check RUVLLM_MODEL_PATH"})
            }
        };
        #[cfg(not(feature = "ruvllm-engine"))]
        let resp = serde_json::json!({
            "error": "binary built without ruvllm-engine feature",
            "prompt": prompt,
            "max_tokens": max_tokens,
        });

        let mut out = serde_json::to_vec(&resp)?;
        out.push(b'\n');
        tx.write_all(&out).await?;
    }
    tracing::debug!(%peer, "conn closed");
    Ok(())
}
