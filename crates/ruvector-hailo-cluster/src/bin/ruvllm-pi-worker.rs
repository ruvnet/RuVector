//! `ruvllm-pi-worker` — per-Pi LLM completion worker (ADR-179).
//!
//! ## Status (iter 3, scaffold only)
//!
//! This is the skeleton bin: env-var contract, TCP listener, version
//! probe. Engine wiring (`ruvllm::serving::ServingEngine` +
//! quantized model load + gRPC completion RPC) lands in iter 4–6.
//!
//! Sibling worker on each Pi 5 to `ruvector-hailo-worker` (ADR-167):
//! - hailo worker  → :50051  → embeddings via Hailo-8 NPU
//! - ruvllm worker → :50053  → completions via Cortex-A76 + pi_quant
//!
//! ## Env vars (forward-compatible — most are placeholders for now)
//!
//! ```text
//! RUVLLM_WORKER_BIND          listen socket   (default 0.0.0.0:50053)
//! RUVLLM_MODEL_PATH           local path to .safetensors|.gguf|.qm
//!                             (no hf-hub download — out-of-band rsync;
//!                              ADR-179 §risks: avoids native-tls cross-link)
//! RUVLLM_QUANTIZE             pi_quant | turbo_quant | quip | none
//!                             (default pi_quant; pi_quant_simd path
//!                              picked at runtime when NEON dotprod ok)
//! RUVLLM_KV_QUANTIZE          rabitq | none  (default none for iter 3)
//! RUVLLM_MAX_INFLIGHT         scheduler concurrent requests (default 4)
//! RUVLLM_MAX_SEQ              max prompt+completion tokens (default 2048)
//! RUVLLM_LOG_PROMPT_AUDIT     none | hash | full  (default none — no leak)
//! ```
//!
//! ## Wire (iter 4+)
//!
//! gRPC on `:50053` exposing a small completion service mirroring the
//! hailo cluster's pattern (Embedding service → unary + streaming +
//! Health + GetStats). Proto goes at
//! `crates/ruvector-hailo-cluster/proto/completion.proto`.
//!
//! ## Why not reuse `ruvllm-cli serve`
//!
//! `ruvllm-cli` cross-builds halt on `hf_hub::api::sync` (needs ureq +
//! native-tls). This bin uses ruvllm as a library + loads from local
//! paths only, dodging hf-hub entirely. See ADR-179 iter 2 for the
//! feature-tree forensics.

use std::env;
use std::net::SocketAddr;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const COMMIT_GATE: &str = "ADR-179 iter 3 — scaffold (no engine yet)";

fn parse_bind() -> SocketAddr {
    env::var("RUVLLM_WORKER_BIND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| "0.0.0.0:50053".parse().unwrap())
}

fn read_optional_env(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| "<unset>".to_string())
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
    tracing::info!(version = VERSION, gate = COMMIT_GATE, "ruvllm-pi-worker starting");
    tracing::info!(
        model_path = %read_optional_env("RUVLLM_MODEL_PATH"),
        quantize = %read_optional_env("RUVLLM_QUANTIZE"),
        kv_quantize = %read_optional_env("RUVLLM_KV_QUANTIZE"),
        max_inflight = %read_optional_env("RUVLLM_MAX_INFLIGHT"),
        max_seq = %read_optional_env("RUVLLM_MAX_SEQ"),
        "iter-3 scaffold: env contract logged"
    );

    // Iter 3: just bind, accept, and print a "hello" line per connection.
    // Iter 4 swaps this for a tonic Server::builder()
    //   .add_service(CompletionServer::new(impl))
    //   .serve(bind);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(%bind, "ruvllm-pi-worker listening (TCP echo placeholder)");

    loop {
        let (mut sock, peer) = listener.accept().await?;
        tracing::info!(%peer, "accepted connection");
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let banner = format!(
                "ruvllm-pi-worker v{} — {}\nbind={}\n",
                VERSION,
                COMMIT_GATE,
                sock.local_addr().map(|a| a.to_string()).unwrap_or_default()
            );
            let _ = sock.write_all(banner.as_bytes()).await;
        });
    }
}
