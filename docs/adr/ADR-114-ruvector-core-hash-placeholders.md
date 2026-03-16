# ADR-114: Ruvector-Core Hash Placeholder Embeddings

**Status**: Accepted
**Date**: 2026-03-16
**Authors**: ruv.io, RuVector Architecture Team
**Deciders**: Architecture Review Board
**SDK**: Claude-Flow
**Relates to**: ADR-058 (Hash Security Hardening), ADR-029 (RVF Canonical Format)

## Context

### Current Embedding Implementation

The `ruvector-core` crate provides a pluggable embedding system via the `EmbeddingProvider` trait. The default implementation, `HashEmbedding`, uses a **non-semantic hash-based approach** that is explicitly marked as a placeholder.

**Critical Warning in lib.rs (lines 15-20)**:
```rust
//! - **AgenticDB**: ⚠️⚠️⚠️ **CRITICAL WARNING** ⚠️⚠️⚠️
//!   - Uses PLACEHOLDER hash-based embeddings, NOT real semantic embeddings
//!   - "dog" and "cat" will NOT be similar (different characters)
//!   - "dog" and "god" WILL be similar (same characters) - **This is wrong!**
//!   - **MUST integrate real embedding model for production** (ONNX, Candle, or API)
```

### Hash Placeholders Identified

| Component | Location | Type | Status |
|-----------|----------|------|--------|
| `HashEmbedding` | `embeddings.rs:44-93` | Byte-level hash embedding | Placeholder - NOT semantic |
| `CandleEmbedding` | `embeddings.rs:107-178` | Transformer stub | Stub - returns error |
| Deprecation warning | `lib.rs:100-106` | Compile-time | Active warning |

### HashEmbedding Algorithm (embeddings.rs:67-83)

```rust
fn embed(&self, text: &str) -> Result<Vec<f32>> {
    let mut embedding = vec![0.0; self.dimensions];
    let bytes = text.as_bytes();

    for (i, byte) in bytes.iter().enumerate() {
        embedding[i % self.dimensions] += (*byte as f32) / 255.0;
    }

    // Normalize to unit vector
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding { *val /= norm; }
    }
    Ok(embedding)
}
```

**Why This Is Wrong for Semantic Search**:
- Operates on raw byte values, not meaning
- "dog" (100,111,103) and "cat" (99,97,116) share no similarity
- "dog" and "god" (103,111,100) are highly similar (same bytes, different order)
- No understanding of synonyms, context, or language

### Distinction from ADR-058

ADR-058 addresses **content integrity hashing** in the RVF wire format:
- XXH3-128 for segment checksums
- SHAKE-256 for cryptographic integrity
- Timing-safe verification

This ADR addresses **semantic embedding hashing** in ruvector-core:
- Vector representations of text meaning
- Similarity search and nearest-neighbor queries
- Production embedding model integration

These are orthogonal concerns with different security and functionality requirements.

## Decision

### 1. Explicit Placeholder Naming

The `HashEmbedding::name()` method returns `"HashEmbedding (placeholder)"` to ensure visibility in logs and debugging. This naming convention must be preserved.

### 2. Compile-Time Deprecation Warning

Maintain the compile-time warning (lib.rs:100-106) that triggers when the `storage` feature is enabled:

```rust
#[deprecated(
    since = "0.1.0",
    note = "AgenticDB uses placeholder hash-based embeddings. For semantic search,
            integrate a real embedding model (ONNX, Candle, or API).
            See /examples/onnx-embeddings for production setup."
)]
const AGENTICDB_EMBEDDING_WARNING: () = ();
```

### 3. Supported Production Alternatives

Five production paths are documented and supported:

| Provider | Location | Use Case | Recommended |
|----------|----------|----------|-------------|
| **`OnnxEmbedding`** | `ruvector-core` (feature: `onnx-embeddings`) | Direct ONNX Runtime embeddings (all-MiniLM-L6-v2) | ✅ **PRIMARY** |
| **RuvLLM ONNX** | `ruvllm` crate / `@ruvector/ruvllm` | Real semantic embeddings with SONA learning | ✅ Yes |
| `RuvectorIntegration` | `ruvllm::ruvector_integration` | Full HNSW + SONA learning stack | ✅ Yes |
| `ApiEmbedding` | `ruvector-core` (feature: `api-embeddings`) | External APIs (OpenAI, Cohere, Voyage) | ✅ Yes |
| Custom `EmbeddingProvider` | N/A | User-implemented models | Yes |

### 4. RuvLLM as Primary Replacement (RECOMMENDED)

**RuvLLM provides real semantic embeddings** via ONNX runtime — this is the recommended replacement for `HashEmbedding`:

**Locations**:
- TypeScript: `npm/packages/ruvector/src/core/onnx-embedder.ts`
- Rust: `crates/ruvllm/src/ruvector_integration.rs`

**Capabilities**:
- **Model**: `all-MiniLM-L6-v2` (384-dimensional transformer embeddings)
- **Type**: Real semantic embeddings via ONNX WASM runtime
- **NOT hash-based** — actual transformer model inference
- Batch processing with parallel workers (3.8x speedup)
- Optional SIMD acceleration
- Model auto-download from HuggingFace on first use

**TypeScript/JavaScript** (npm):
```typescript
import { OnnxEmbedder } from 'ruvector';

const embedder = new OnnxEmbedder({ modelId: 'all-MiniLM-L6-v2' });
await embedder.initialize();

const result = await embedder.embed("hello world");
// result.embedding: number[] (384-dimensional REAL semantic embedding)
// "dog" and "cat" WILL be similar (semantic understanding)
```

**Rust** (via ruvllm crate):
```rust
use ruvllm::ruvector_integration::{RuvectorIntegration, IntegrationConfig};

let config = IntegrationConfig::default(); // 384-dim embeddings
let integration = RuvectorIntegration::new(config)?;

// Full stack: HNSW indexing + SONA learning + semantic search
let decision = integration.route_with_intelligence("implement auth", &embedding)?;
integration.learn_from_outcome(&task, decision.agent, true)?;
```

**Why RuvLLM over ApiEmbedding**:

| Aspect | RuvLLM ONNX | ApiEmbedding |
|--------|-------------|--------------|
| Latency | 5-50ms (local) | 50-200ms (network) |
| Cost | Free | $0.02-0.13/1M tokens |
| Privacy | Data stays local | Data sent to API |
| Offline | ✅ Works offline | ❌ Requires internet |
| SONA Learning | ✅ Integrated | ❌ Separate setup |
| Batch Processing | ✅ Parallel workers | ❌ Sequential API calls |

### 5. CandleEmbedding Stub Behavior

The `CandleEmbedding::from_pretrained()` method intentionally returns an error:

```rust
Err(RuvectorError::ModelLoadError(format!(
    "Candle embedding support is a stub. Please:\n\
     1. Use RuvLLM ONNX embeddings (recommended)\n\
     2. Or use ApiEmbedding for external APIs\n\
     3. See docs for ruvllm integration examples",
    model_id
)))
```

This ensures users cannot accidentally use a non-functional embedding provider.

### 6. ApiEmbedding as Fallback

For deployments where RuvLLM is not available, `ApiEmbedding` provides external API access:
- **OpenAI**: `text-embedding-3-small` (1536 dims), `text-embedding-3-large` (3072 dims)
- **Cohere**: `embed-english-v3.0` (1024 dims)
- **Voyage**: `voyage-2` (1024 dims), `voyage-large-2` (1536 dims)

## Consequences

### Positive

- Clear documentation prevents accidental production use of placeholder embeddings
- Pluggable architecture allows drop-in replacement
- Compile-time warnings surface issues during development
- Multiple integration paths support diverse deployment scenarios

### Negative

- Default behavior is intentionally broken for semantic search
- Users must take explicit action to enable real embeddings
- API-based embeddings add latency and cost
- Local model support (Candle) requires additional implementation

### Trade-offs

| Approach | Latency | Cost | Quality | Complexity |
|----------|---------|------|---------|------------|
| HashEmbedding | <1ms | Free | Poor (non-semantic) | None |
| **`OnnxEmbedding`** | **5-50ms** | **Free** | **High** | **Model download (~90MB)** |
| RuvLLM ONNX | 5-50ms | Free | High | Model download (~90MB) |
| RuvectorIntegration | 5-50ms | Free | High + Learning | SONA setup |
| ApiEmbedding | 50-200ms | $0.02-0.13/1M tokens | High | API key management |
| Candle (future) | 10-100ms | Free | High | Heavy dependencies |

## Implementation Checklist

### Completed
- [x] `HashEmbedding` with explicit placeholder naming
- [x] `EmbeddingProvider` trait for pluggable providers
- [x] `ApiEmbedding` with OpenAI, Cohere, Voyage support
- [x] Compile-time deprecation warning
- [x] Documentation in lib.rs and embeddings.rs
- [x] **RuvLLM ONNX embeddings** (`OnnxEmbedder` in npm, `RuvectorIntegration` in Rust)
- [x] **SONA learning integration** via `ruvllm::ruvector_integration`
- [x] **Parallel batch processing** with worker threads
- [x] **`OnnxEmbedding` in ruvector-core** (feature: `onnx-embeddings`) — Direct ONNX Runtime integration using ort 2.0

### Pending (Future PRs)
- [ ] Full Candle implementation (replace stub)
- [ ] Benchmark suite comparing provider performance

## Usage Examples

### Production (RuvLLM - RECOMMENDED)

**TypeScript/JavaScript:**
```typescript
import { OnnxEmbedder } from 'ruvector';

// Initialize once (downloads model on first use)
const embedder = new OnnxEmbedder({
  modelId: 'all-MiniLM-L6-v2',
  enableParallel: true,  // 3.8x speedup for batches
});
await embedder.initialize();

// Single embedding
const result = await embedder.embed("hello world");
console.log(result.embedding.length); // 384

// Batch embedding (parallel)
const batch = await embedder.embedBatch(["dog", "cat", "puppy"]);
// "dog" and "puppy" WILL be similar (semantic!)
```

**Rust:**
```rust
use ruvllm::ruvector_integration::{RuvectorIntegration, IntegrationConfig};

let integration = RuvectorIntegration::new(IntegrationConfig::default())?;

// Intelligent routing with SONA learning
let decision = integration.route_with_intelligence("implement auth", &embedding)?;

// Learn from outcome (improves future routing)
integration.learn_from_outcome(&task, decision.agent, true)?;
```

### ruvector-core OnnxEmbedding (Native Rust)

**Direct ONNX Runtime integration** (requires feature: `onnx-embeddings`):
```rust
use ruvector_core::embeddings::{EmbeddingProvider, OnnxEmbedding};

// Downloads and caches model from HuggingFace on first use (~90MB)
let provider = OnnxEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;

// Real semantic embeddings
let embedding = provider.embed("hello world")?;
assert_eq!(embedding.len(), 384);

// "dog" and "cat" WILL be similar (semantic understanding!)
let dog = provider.embed("dog")?;
let cat = provider.embed("cat")?;
let sim: f32 = dog.iter().zip(&cat).map(|(a, b)| a * b).sum();
println!("dog-cat similarity: {}", sim); // ~0.8
```

### Testing/Prototyping (Placeholder)
```rust
use ruvector_core::embeddings::{EmbeddingProvider, HashEmbedding};

let provider = HashEmbedding::new(384);
let embedding = provider.embed("hello world")?; // Fast but NOT semantic
assert_eq!(provider.name(), "HashEmbedding (placeholder)");
```

### Fallback (API-Based)
```rust
use ruvector_core::embeddings::{EmbeddingProvider, ApiEmbedding};

let provider = ApiEmbedding::openai("sk-...", "text-embedding-3-small");
let embedding = provider.embed("hello world")?; // Real semantic embeddings
```

## Security Considerations

### Hash Collision Risk (HashEmbedding)

The byte-level hashing creates predictable collisions:
- Anagrams always collide ("dog" ≈ "god")
- Repeated patterns concentrate in specific dimensions
- NOT suitable for any security-sensitive application

### API Key Management (ApiEmbedding)

When using external APIs:
- Store keys in environment variables or secret managers
- Rotate keys periodically
- Monitor usage for anomalies
- Consider rate limiting and caching

## Related ADRs

- **ADR-058**: Hash Security Hardening (RVF wire format checksums)
- **ADR-029**: RVF Canonical Format
- **ADR-042**: Security-RVF-AIDefence-TEE

## References

- Sentence Transformers: https://sbert.net/
- ONNX Runtime: https://onnxruntime.ai/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- Candle: https://github.com/huggingface/candle
