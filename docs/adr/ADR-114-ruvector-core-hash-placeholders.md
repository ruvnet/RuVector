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

Three production paths are documented and supported:

| Provider | Feature Flag | Use Case |
|----------|--------------|----------|
| `ApiEmbedding` | `api-embeddings` | External APIs (OpenAI, Cohere, Voyage) |
| `CandleEmbedding` | `real-embeddings` | Local transformer models (stub) |
| Custom `EmbeddingProvider` | N/A | User-implemented ONNX, custom models |

### 4. CandleEmbedding Stub Behavior

The `CandleEmbedding::from_pretrained()` method intentionally returns an error:

```rust
Err(RuvectorError::ModelLoadError(format!(
    "Candle embedding support is a stub. Please:\n\
     1. Use ApiEmbedding for production (recommended)\n\
     2. Or implement CandleEmbedding for model: {}\n\
     3. See docs for ONNX Runtime integration examples",
    model_id
)))
```

This ensures users cannot accidentally use a non-functional embedding provider.

### 5. ApiEmbedding as Recommended Default

For production deployments, `ApiEmbedding` is the recommended path:
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
| ApiEmbedding | 50-200ms | $0.02-0.13/1M tokens | High | API key management |
| ONNX Runtime | 5-50ms | Free | High | Model bundling |
| Candle (future) | 10-100ms | Free | High | Heavy dependencies |

## Implementation Checklist

### Completed
- [x] `HashEmbedding` with explicit placeholder naming
- [x] `EmbeddingProvider` trait for pluggable providers
- [x] `ApiEmbedding` with OpenAI, Cohere, Voyage support
- [x] Compile-time deprecation warning
- [x] Documentation in lib.rs and embeddings.rs

### Pending (Future PRs)
- [ ] ONNX Runtime integration example in `/examples/onnx-embeddings`
- [ ] Full Candle implementation (replace stub)
- [ ] Benchmark suite comparing provider performance
- [ ] Caching layer for API-based embeddings

## Usage Examples

### Testing/Prototyping (Placeholder)
```rust
use ruvector_core::embeddings::{EmbeddingProvider, HashEmbedding};

let provider = HashEmbedding::new(384);
let embedding = provider.embed("hello world")?; // Fast but NOT semantic
assert_eq!(provider.name(), "HashEmbedding (placeholder)");
```

### Production (API-Based)
```rust
use ruvector_core::embeddings::{EmbeddingProvider, ApiEmbedding};

let provider = ApiEmbedding::openai("sk-...", "text-embedding-3-small");
let embedding = provider.embed("hello world")?; // Real semantic embeddings
```

### Production (Custom ONNX)
```rust
use ruvector_core::embeddings::EmbeddingProvider;

struct OnnxEmbedding { /* ... */ }

impl EmbeddingProvider for OnnxEmbedding {
    fn embed(&self, text: &str) -> ruvector_core::Result<Vec<f32>> {
        // Implement ONNX inference
    }
    fn dimensions(&self) -> usize { 384 }
    fn name(&self) -> &str { "OnnxEmbedding (all-MiniLM-L6-v2)" }
}
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
