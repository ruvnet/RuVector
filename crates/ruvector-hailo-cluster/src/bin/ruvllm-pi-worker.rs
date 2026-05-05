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
const GATE: &str = "ADR-180 iter 3 — N-backend pool + semaphore parallelism";
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
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    /// ADR-180 iter 3: pool of `Mutex<CandleBackend>` instances + a
    /// tokio Semaphore for capacity control. N independent backends =
    /// N parallel requests *actually running concurrently* on different
    /// threads, each holding its own model weights + KV cache.
    ///
    /// **Why not ServingEngine?** Iter-2 found that ruvllm 2.2.0's
    /// `ServingEngine::generate_next_token` is a per-token text-mode
    /// dispatcher that still serializes on `model.generate(text, 1)` —
    /// no actual batched forward pass against CandleBackend. It's a
    /// scaffold for true continuous batching, not a working impl.
    ///
    /// **Cost of pool**: N × ~640 MB weights = 4 backends ≈ 2.5 GB on
    /// each Pi. 8 GB Pi 5 has plenty of headroom (the embed worker
    /// + system already use ~1 GB, leaving ~5 GB for our pool + KV).
    pub struct PiEngine {
        backends: Vec<Arc<tokio::sync::Mutex<CandleBackend>>>,
        sem: Arc<Semaphore>,
    }

    impl PiEngine {
        pub fn load(model_path: &str, pool_size: usize) -> anyhow::Result<Self> {
            // Load ONE backend first to fail fast on bad model paths,
            // then load the rest. (Loads are independent; we could
            // parallelize, but Pi 5 disk IO + tokenizer init is the
            // serial bottleneck either way.)
            let config = ModelConfig {
                use_flash_attention: false,
                quantization: None,
                ..ModelConfig::default()
            };

            let mut backends = Vec::with_capacity(pool_size);
            for i in 0..pool_size {
                let mut backend = CandleBackend::new()
                    .map_err(|e| anyhow::anyhow!("CandleBackend::new[{i}] failed: {e:?}"))?;
                backend
                    .load_model(model_path, config.clone())
                    .map_err(|e| anyhow::anyhow!("load_model[{i}]({model_path}) failed: {e:?}"))?;
                backends.push(Arc::new(tokio::sync::Mutex::new(backend)));
                tracing::info!(
                    "loaded backend slot {}/{} for {}",
                    i + 1,
                    pool_size,
                    model_path
                );
            }

            Ok(Self {
                backends,
                sem: Arc::new(Semaphore::new(pool_size)),
            })
        }

        /// Pick a free backend slot (round-robin under semaphore) and
        /// drive the request to completion. Multiple concurrent calls
        /// run on different slots — true request-level parallelism.
        pub async fn generate(&self, prompt: &str, max_tokens: usize) -> anyhow::Result<String> {
            let _permit = self
                .sem
                .acquire()
                .await
                .map_err(|e| anyhow::anyhow!("semaphore closed: {e:?}"))?;

            // Find the first un-locked backend (try-lock walk).
            let backend = {
                let mut chosen = None;
                for b in &self.backends {
                    if let Ok(_g) = b.try_lock() {
                        chosen = Some(Arc::clone(b));
                        break;
                    }
                }
                chosen.unwrap_or_else(|| Arc::clone(&self.backends[0]))
            };

            let prompt = prompt.to_string();
            // Move the actual generate() call to a blocking thread —
            // candle's CPU forward pass is sync + compute-heavy.
            tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                let backend = backend.blocking_lock();
                let params = GenerateParams {
                    max_tokens,
                    ..Default::default()
                };
                backend
                    .generate(&prompt, params)
                    .map_err(|e| anyhow::anyhow!("generate failed: {e:?}"))
            })
            .await
            .map_err(|e| anyhow::anyhow!("spawn_blocking join: {e:?}"))?
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
        tracing::warn!(
            "RUVLLM_MODEL_PATH is unset; refusing to start engine, falling back to scaffold mode"
        );
        None
    } else {
        // ADR-180 iter 2: max_inflight controls ServingEngine's
        // continuous-batching capacity. Default 4; bump via env to sweep.
        let max_inflight: usize = env::var("RUVLLM_MAX_INFLIGHT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        tracing::info!(
            "loading model from {} (max_inflight={}) ...",
            model_path,
            max_inflight
        );
        match engine::PiEngine::load(&model_path, max_inflight) {
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
                let banner = format!("ruvllm-pi-worker v{} — {}\n", VERSION, GATE);
                tx.write_all(banner.as_bytes()).await?;
                continue;
            }
        };
        let prompt = req.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
        let max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(64) as usize;

        #[cfg(feature = "ruvllm-engine")]
        let resp = {
            let started = std::time::Instant::now();
            match &engine {
                // ADR-180 iter 2: generate() is now async (ServingEngine submit_async).
                Some(e) => match e.generate(prompt, max_tokens).await {
                    Ok(text) => {
                        let ms = started.elapsed().as_millis() as u64;
                        serde_json::json!({"text": text, "tokens": text.len(), "ms": ms})
                    }
                    Err(e) => serde_json::json!({"error": format!("{e:#}")}),
                },
                None => {
                    serde_json::json!({"error": "engine not loaded; check RUVLLM_MODEL_PATH"})
                }
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
