#![allow(clippy::all)]
//! Sanity check for the sparse-attention integration
//! (`patches/candle-transformers`'s `forward_sparse` / `forward_attn_sparse`,
//! wired via `CandleBackend::enable_sparse_attention`).
//!
//! Loads one real GGUF model, generates the same prompt with dense attention
//! and then with sparse attention enabled, and prints both outputs so a human
//! can eyeball that sparse attention produces coherent (not garbled) text —
//! this is not a numerical equivalence check (sparse attention is a genuinely
//! different, approximate algorithm; the tensor bridging is the thing under
//! test here, not exact output parity).
//!
//! ```bash
//! cargo run -p ruvllm --release --features candle,metal,sparse-attention \
//!     --example sparse_attention_check -- --model ./model.gguf
//! ```

use std::env;
use std::path::PathBuf;

#[cfg(feature = "sparse-attention")]
fn main() {
    use ruvllm::{CandleBackend, GenerateParams, LlmBackend, ModelConfig};

    let args: Vec<String> = env::args().collect();
    let mut model_path = PathBuf::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model" || args[i] == "-m" {
            i += 1;
            if i < args.len() {
                model_path = PathBuf::from(&args[i]);
            }
        }
        i += 1;
    }
    if model_path.as_os_str().is_empty() {
        eprintln!("Usage: --model <GGUF_PATH> (tokenizer.json expected alongside it)");
        std::process::exit(1);
    }
    let tokenizer_path = model_path
        .parent()
        .expect("model path has no parent")
        .join("tokenizer.json");

    let params = GenerateParams {
        max_tokens: 40,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        ..Default::default()
    };
    let long = args.iter().any(|a| a == "--long");
    let prompt = if long {
        // Deliberately longer than SparseAttentionConfig::default()'s
        // 128-token window, so the local-window/landmark sparsity actually
        // activates instead of degenerating to full attention.
        "The history of the city of Paris stretches back over two thousand years. \
         Founded by a Celtic tribe known as the Parisii around the 3rd century BC, \
         the settlement was originally located on the Ile de la Cite, an island in \
         the Seine river. The Romans conquered the area in 52 BC and renamed it \
         Lutetia, building roads, baths, and an amphitheater. Over the following \
         centuries the city grew steadily, becoming the capital of the Kingdom of \
         France under the Capetian dynasty. During the Middle Ages, Paris became a \
         major center of learning, commerce, and religion in Europe. The capital of \
         France is"
    } else {
        "The capital of France is"
    };

    let mut dense = CandleBackend::new().expect("create backend");
    dense
        .load_gguf(&model_path, &ModelConfig::default())
        .expect("load gguf (dense)");
    dense
        .load_tokenizer(&tokenizer_path)
        .expect("load tokenizer (dense)");
    let dense_output = dense
        .generate(prompt, params.clone())
        .expect("dense generate");
    println!("[dense]  {dense_output:?}");

    let mut sparse = CandleBackend::new().expect("create backend");
    sparse
        .load_gguf(&model_path, &ModelConfig::default())
        .expect("load gguf (sparse)");
    sparse
        .load_tokenizer(&tokenizer_path)
        .expect("load tokenizer (sparse)");
    sparse
        .enable_sparse_attention(ruvllm_sparse_attention::SparseAttentionConfig::default())
        .expect("enable sparse attention");
    assert!(sparse.sparse_attention_enabled());
    let sparse_output = sparse.generate(prompt, params).expect("sparse generate");
    println!("[sparse] {sparse_output:?}");

    if dense_output.trim().is_empty() || sparse_output.trim().is_empty() {
        eprintln!("FAIL: empty output from one of the paths");
        std::process::exit(1);
    }
    println!("\nBoth paths produced non-empty output; inspect them above for coherence.");
}

#[cfg(not(feature = "sparse-attention"))]
fn main() {
    eprintln!("This example requires --features candle,metal,sparse-attention");
    std::process::exit(1);
}
