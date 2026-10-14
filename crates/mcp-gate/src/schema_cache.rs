//! ReCache-pattern content-addressed schema-resource cache (PIR WP26, ADR-329).
//!
//! Every tool/skill/schema gets a *stable resource identity*: the SHA-256 of
//! its canonical JSON encoding (object keys sorted recursively, no
//! insignificant whitespace). The identity survives key reordering and
//! formatting churn, so a schema is compiled into its context block exactly
//! once per content, and agent context is assembled by concatenating cached
//! blocks in the requested order instead of recompiling the full schema set
//! whenever the ordering changes (the reuse pattern from ReCache,
//! arXiv:2608.19662; implemented from the paper's description — no upstream
//! code was consulted or copied).
//!
//! # Honest scope
//!
//! This is **context-assembly-level** reuse: the stable resource identity and
//! the once-per-content compiled block are the substrate. True KV-level
//! position-independent reuse (caching per-resource key/value tensors inside
//! the serving stack, ReCache's headline TTFT/memory numbers) lives in
//! ruvllm's KV-cache layer and is follow-up work — it is **not** claimed
//! here. What this layer guarantees is byte-identity: assembling cached
//! blocks produces exactly the bytes a from-scratch compile of the same
//! ordered set would, so downstream behaviour (and hence invocation
//! accuracy) is unchanged by construction.
//!
//! # Guards
//!
//! Canonicalization is fail-loud: schemas above the configured size ceiling
//! are refused *while encoding* (the encoder bails as soon as the output
//! exceeds the ceiling, so an oversize schema cannot force a full canonical
//! allocation before refusal), nesting deeper than [`MAX_CANONICAL_DEPTH`]
//! is refused, assembly is bounded by a configurable total-bytes ceiling
//! (so a long order repeating large identities cannot amplify memory), and
//! assembling with an identity the cache no longer holds (evicted) is an
//! error rather than a silent recompile — the caller re-inserts and retries,
//! keeping cache behaviour observable. (Non-finite numbers cannot occur:
//! `serde_json::Number` cannot represent NaN/infinity, so every number that
//! reaches canonicalization is finite by construction.)
//!
//! # Memory sizing
//!
//! Worst-case residency is `capacity × max_schema_bytes` — with the
//! defaults, 2048 × 256 KiB ≈ **512 MiB**. That is a deliberate
//! upper bound, not an expected footprint (typical tool schemas are 1–4 KB,
//! ≈ 8 MiB at full capacity), but wiring this cache into a long-lived
//! server should size [`SchemaCacheConfig`] consciously rather than
//! defaulting blindly.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;

use crate::types::McpTool;

/// Maximum nesting depth accepted by the canonical encoder.
pub const MAX_CANONICAL_DEPTH: usize = 128;

/// Default per-schema size ceiling (bytes of canonical encoding), 256 KiB.
pub const DEFAULT_MAX_SCHEMA_BYTES: usize = 256 * 1024;

/// Default cache capacity, matching the ≤2048-entry discipline of
/// `ruvector-query-cache` (ADR-301).
pub const DEFAULT_CAPACITY: usize = 2048;

/// Default ceiling on one assembled context, 64 MiB. Bounds the
/// amplification a long `order` repeating large identities could otherwise
/// produce (an unbounded order of 256 KiB blocks grows linearly with its
/// length).
pub const DEFAULT_MAX_ASSEMBLED_BYTES: usize = 64 * 1024 * 1024;

/// Errors from canonicalization, compilation, or assembly.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SchemaCacheError {
    /// The schema's canonical encoding exceeds the configured ceiling.
    /// `actual` is the encoded length at the point the encoder bailed — a
    /// lower bound on the full canonical size, since encoding stops as soon
    /// as the ceiling is crossed.
    #[error(
        "schema too large: canonical encoding is at least {actual} bytes, ceiling is {ceiling}"
    )]
    SchemaTooLarge { actual: usize, ceiling: usize },
    /// The requested assembly would exceed the configured total-bytes
    /// ceiling (guards against amplification via long orders repeating
    /// large identities).
    #[error("assembly too large: {actual} bytes requested, ceiling is {ceiling}")]
    AssemblyTooLarge { actual: usize, ceiling: usize },
    /// The schema nests deeper than [`MAX_CANONICAL_DEPTH`].
    #[error("schema nesting exceeds maximum depth {0}")]
    DepthExceeded(usize),
    /// The value could not be serialized to JSON at all.
    #[error("schema is not serializable: {0}")]
    NotSerializable(String),
    /// An identity requested during assembly is not resident (never inserted
    /// or evicted since). The caller must re-insert and retry.
    #[error("resource {0} is not resident in the cache")]
    NotResident(ResourceId),
    /// A cache with capacity 0 cannot hold any block.
    #[error("cache capacity is zero")]
    ZeroCapacity,
}

/// Stable content-addressed identity of a schema resource: SHA-256 of the
/// canonical encoding. Two schemas that differ only in key order or
/// formatting share one identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceId([u8; 32]);

impl ResourceId {
    /// The raw digest bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

/// Recursively write the canonical encoding of `value`: object keys sorted
/// bytewise, arrays in order, no insignificant whitespace, string/number
/// atoms rendered by `serde_json` (stable for a given `serde_json` version).
///
/// The `ceiling` is enforced *while encoding*: as soon as `out` exceeds it
/// the encoder bails with [`SchemaCacheError::SchemaTooLarge`], so an
/// oversize schema cannot force a full canonical allocation before refusal.
fn write_canonical(
    value: &serde_json::Value,
    out: &mut String,
    depth: usize,
    ceiling: usize,
) -> Result<(), SchemaCacheError> {
    if depth > MAX_CANONICAL_DEPTH {
        return Err(SchemaCacheError::DepthExceeded(MAX_CANONICAL_DEPTH));
    }
    if out.len() > ceiling {
        return Err(SchemaCacheError::SchemaTooLarge {
            actual: out.len(),
            ceiling,
        });
    }
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::String(s) => write_quoted(s, out, ceiling)?,
        serde_json::Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out, depth + 1, ceiling)?;
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            // Sort keys explicitly rather than relying on the map's iteration
            // order, so the encoding is canonical even if some crate in the
            // build graph enables serde_json's `preserve_order` feature.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_quoted(key, out, ceiling)?;
                out.push(':');
                write_canonical(&map[key.as_str()], out, depth + 1, ceiling)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

fn write_quoted(s: &str, out: &mut String, ceiling: usize) -> Result<(), SchemaCacheError> {
    // A single string atom can be arbitrarily large: refuse from its raw
    // length before rendering the (at-least-as-large) quoted form.
    if out.len().saturating_add(s.len()) > ceiling {
        return Err(SchemaCacheError::SchemaTooLarge {
            actual: out.len().saturating_add(s.len()),
            ceiling,
        });
    }
    // serde_json string escaping is deterministic. A control character
    // expands to six bytes, so the raw-length check above can undercount.
    let quoted =
        serde_json::to_string(s).map_err(|e| SchemaCacheError::NotSerializable(e.to_string()))?;
    if out.len().saturating_add(quoted.len()) > ceiling {
        return Err(SchemaCacheError::SchemaTooLarge {
            actual: out.len().saturating_add(quoted.len()),
            ceiling,
        });
    }
    out.push_str(&quoted);
    Ok(())
}

/// Canonicalize a JSON value under the default per-schema ceiling,
/// [`DEFAULT_MAX_SCHEMA_BYTES`]. Use [`canonicalize_bounded`] to pick a
/// different ceiling.
pub fn canonicalize(value: &serde_json::Value) -> Result<String, SchemaCacheError> {
    canonicalize_bounded(value, DEFAULT_MAX_SCHEMA_BYTES)
}

/// Canonicalize with a streaming size ceiling: encoding bails as soon as the
/// output exceeds `ceiling`, so an oversize input cannot force a full
/// canonical allocation before refusal.
pub fn canonicalize_bounded(
    value: &serde_json::Value,
    ceiling: usize,
) -> Result<String, SchemaCacheError> {
    let mut out = String::new();
    write_canonical(value, &mut out, 0, ceiling)?;
    if out.len() > ceiling {
        return Err(SchemaCacheError::SchemaTooLarge {
            actual: out.len(),
            ceiling,
        });
    }
    Ok(out)
}

/// Compile one canonical encoding into its context block. The block format
/// is a single line — canonical JSON plus a trailing newline — so that
/// concatenating blocks in order is byte-identical to a from-scratch compile
/// of the same ordered set.
fn compile_block(canonical: &str) -> String {
    let mut block = String::with_capacity(canonical.len() + 1);
    block.push_str(canonical);
    block.push('\n');
    block
}

/// Compile an ordered set of schemas from scratch, with no cache involved.
/// This is the reference the cache's assembly must match byte-for-byte, and
/// the cold baseline in the benchmark. Applies the same `max_schema_bytes`
/// and `max_assembled_bytes` ceilings as a cache built from `config`.
pub fn compile_fresh(
    values: &[&serde_json::Value],
    config: &SchemaCacheConfig,
) -> Result<String, SchemaCacheError> {
    let mut out = String::new();
    for value in values {
        let block = compile_block(&canonicalize_bounded(value, config.max_schema_bytes)?);
        let total = out.len().saturating_add(block.len());
        if total > config.max_assembled_bytes {
            return Err(SchemaCacheError::AssemblyTooLarge {
                actual: total,
                ceiling: config.max_assembled_bytes,
            });
        }
        out.push_str(&block);
    }
    Ok(out)
}

struct CacheEntry {
    block: String,
    last_used: u64,
}

/// Configuration for [`SchemaResourceCache`].
///
/// Worst-case memory residency is `capacity × max_schema_bytes` — the
/// defaults bound at 2048 × 256 KiB ≈ 512 MiB. Long-lived servers should
/// size these consciously (see the module-level "Memory sizing" note).
#[derive(Debug, Clone)]
pub struct SchemaCacheConfig {
    /// Maximum resident compiled blocks; least-recently-used is evicted.
    pub capacity: usize,
    /// Per-schema ceiling on canonical encoding size, in bytes.
    pub max_schema_bytes: usize,
    /// Ceiling on one assembled context, in bytes. Guards against
    /// amplification via long orders repeating large identities.
    pub max_assembled_bytes: usize,
}

impl Default for SchemaCacheConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_CAPACITY,
            max_schema_bytes: DEFAULT_MAX_SCHEMA_BYTES,
            max_assembled_bytes: DEFAULT_MAX_ASSEMBLED_BYTES,
        }
    }
}

/// Hit/miss/eviction totals.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SchemaCacheStats {
    /// Inserts that found the identity already resident (compile skipped).
    pub hits: u64,
    /// Inserts that compiled a new block.
    pub misses: u64,
    /// Blocks evicted to make room.
    pub evictions: u64,
}

/// Bounded LRU cache of compiled schema blocks, keyed by content identity.
pub struct SchemaResourceCache {
    config: SchemaCacheConfig,
    entries: HashMap<ResourceId, CacheEntry>,
    tick: u64,
    stats: SchemaCacheStats,
}

impl SchemaResourceCache {
    /// Create a cache with the default configuration.
    pub fn new() -> Self {
        Self::with_config(SchemaCacheConfig::default())
    }

    /// Create a cache with an explicit configuration.
    pub fn with_config(config: SchemaCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            tick: 0,
            stats: SchemaCacheStats::default(),
        }
    }

    /// Canonicalize, identify, and (if new) compile and store the schema.
    /// Idempotent: the same content always yields the same [`ResourceId`],
    /// and a resident identity skips recompilation.
    pub fn insert(&mut self, value: &serde_json::Value) -> Result<ResourceId, SchemaCacheError> {
        if self.config.capacity == 0 {
            return Err(SchemaCacheError::ZeroCapacity);
        }
        // Streaming ceiling: refusal happens during encoding, before a full
        // canonical allocation for an oversize schema can exist.
        let canonical = canonicalize_bounded(value, self.config.max_schema_bytes)?;
        let id = ResourceId(Sha256::digest(canonical.as_bytes()).into());
        self.tick += 1;
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.last_used = self.tick;
            self.stats.hits += 1;
            return Ok(id);
        }
        if self.entries.len() >= self.config.capacity {
            self.evict_lru();
        }
        self.entries.insert(
            id,
            CacheEntry {
                block: compile_block(&canonical),
                last_used: self.tick,
            },
        );
        self.stats.misses += 1;
        Ok(id)
    }

    /// Convenience: insert an MCP tool definition (name, description, and
    /// input schema all participate in the identity).
    pub fn insert_tool(&mut self, tool: &McpTool) -> Result<ResourceId, SchemaCacheError> {
        let value = to_value(tool)?;
        self.insert(&value)
    }

    /// Assemble a context from resident blocks in the given order. The
    /// result is byte-identical to [`compile_fresh`] over the same ordered
    /// schemas and configuration. Errors (rather than silently recompiling)
    /// if any identity is not resident, and refuses assemblies whose total
    /// size would exceed `max_assembled_bytes` (a long order repeating large
    /// identities would otherwise amplify memory linearly with its length).
    pub fn assemble(&mut self, order: &[ResourceId]) -> Result<String, SchemaCacheError> {
        // Validate residency and total size first so a partial or oversized
        // assembly is never observable.
        let mut total = 0usize;
        for id in order {
            match self.entries.get(id) {
                Some(entry) => total = total.saturating_add(entry.block.len()),
                None => return Err(SchemaCacheError::NotResident(*id)),
            }
        }
        if total > self.config.max_assembled_bytes {
            return Err(SchemaCacheError::AssemblyTooLarge {
                actual: total,
                ceiling: self.config.max_assembled_bytes,
            });
        }
        let mut out = String::with_capacity(total);
        for id in order {
            self.tick += 1;
            let entry = self.entries.get_mut(id).expect("residency validated above");
            entry.last_used = self.tick;
            out.push_str(&entry.block);
        }
        Ok(out)
    }

    /// Whether an identity is currently resident.
    pub fn contains(&self, id: &ResourceId) -> bool {
        self.entries.contains_key(id)
    }

    /// Number of resident blocks.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Running totals.
    pub fn stats(&self) -> &SchemaCacheStats {
        &self.stats
    }

    fn evict_lru(&mut self) {
        if let Some(oldest) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(id, _)| *id)
        {
            self.entries.remove(&oldest);
            self.stats.evictions += 1;
        }
    }
}

impl Default for SchemaResourceCache {
    fn default() -> Self {
        Self::new()
    }
}

fn to_value<T: Serialize>(value: &T) -> Result<serde_json::Value, SchemaCacheError> {
    serde_json::to_value(value).map_err(|e| SchemaCacheError::NotSerializable(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_a() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action_id": { "type": "string" },
                "target": { "type": "object", "properties": { "device": { "type": "string" } } }
            },
            "required": ["action_id"]
        })
    }

    #[test]
    fn identity_stable_under_key_reordering_and_whitespace() {
        // Same content, different textual key order and formatting.
        let v1: serde_json::Value =
            serde_json::from_str(r#"{"b": [1, 2], "a": {"y": true, "x": null}}"#).unwrap();
        let v2: serde_json::Value =
            serde_json::from_str("{\n  \"a\": {\"x\": null, \"y\": true},\n  \"b\": [1,2]\n}")
                .unwrap();
        let mut cache = SchemaResourceCache::new();
        let id1 = cache.insert(&v1).unwrap();
        let id2 = cache.insert(&v2).unwrap();
        assert_eq!(id1, id2);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn assembly_is_byte_identical_to_fresh_compile() {
        let schemas: Vec<serde_json::Value> = (0..20)
            .map(|i| {
                serde_json::json!({
                    "name": format!("tool_{i}"),
                    "input_schema": schema_a(),
                    "index": i
                })
            })
            .collect();
        let mut cache = SchemaResourceCache::new();
        let ids: Vec<ResourceId> = schemas.iter().map(|s| cache.insert(s).unwrap()).collect();
        // A permuted order must match fresh compilation of the same order.
        let order: Vec<usize> = (0..schemas.len()).rev().collect();
        let permuted_ids: Vec<ResourceId> = order.iter().map(|&i| ids[i]).collect();
        let permuted_refs: Vec<&serde_json::Value> = order.iter().map(|&i| &schemas[i]).collect();
        let assembled = cache.assemble(&permuted_ids).unwrap();
        let fresh = compile_fresh(&permuted_refs, &SchemaCacheConfig::default()).unwrap();
        assert_eq!(assembled, fresh);
    }

    #[test]
    fn lru_eviction_and_not_resident_error() {
        let mut cache = SchemaResourceCache::with_config(SchemaCacheConfig {
            capacity: 2,
            ..SchemaCacheConfig::default()
        });
        let a = cache.insert(&serde_json::json!({"n": 1})).unwrap();
        let b = cache.insert(&serde_json::json!({"n": 2})).unwrap();
        // Touch `a` so `b` becomes the least recently used.
        cache.assemble(&[a]).unwrap();
        let c = cache.insert(&serde_json::json!({"n": 3})).unwrap();
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.contains(&a));
        assert!(!cache.contains(&b));
        assert!(cache.contains(&c));
        // Assembling with the evicted identity is a loud error, not a
        // silent recompile.
        assert_eq!(cache.assemble(&[b]), Err(SchemaCacheError::NotResident(b)));
    }

    #[test]
    fn oversize_schema_is_refused() {
        let mut cache = SchemaResourceCache::with_config(SchemaCacheConfig {
            max_schema_bytes: 64,
            ..SchemaCacheConfig::default()
        });
        let big = serde_json::json!({ "blob": "x".repeat(128) });
        match cache.insert(&big) {
            Err(SchemaCacheError::SchemaTooLarge { actual, ceiling }) => {
                assert!(actual > ceiling);
                assert_eq!(ceiling, 64);
            }
            other => panic!("expected SchemaTooLarge, got {other:?}"),
        }
        assert!(cache.is_empty());
    }

    #[test]
    fn oversize_bails_during_encoding_not_after() {
        // A schema vastly larger than the ceiling: refusal must report a
        // bailed-at length near the ceiling, proving the encoder stopped
        // early rather than canonicalizing the whole value first.
        let ceiling = 1024;
        let huge = serde_json::json!({
            "items": (0..5000)
                .map(|i| serde_json::json!({ "name": format!("field_{i}"), "blob": "y".repeat(256) }))
                .collect::<Vec<_>>()
        });
        let full_len = canonicalize_bounded(&huge, usize::MAX).unwrap().len();
        assert!(full_len > 1_000_000, "fixture should dwarf the ceiling");

        match canonicalize_bounded(&huge, ceiling) {
            Err(SchemaCacheError::SchemaTooLarge { actual, ceiling: c }) => {
                assert_eq!(c, ceiling);
                // Bailed close to the ceiling, nowhere near the full size.
                assert!(
                    actual < ceiling * 2,
                    "expected an early bail near {ceiling}, got {actual}"
                );
                assert!(actual * 100 < full_len, "bail was not early");
            }
            other => panic!("expected SchemaTooLarge, got {other:?}"),
        }

        // Same refusal through the cache, and nothing is retained.
        let mut cache = SchemaResourceCache::with_config(SchemaCacheConfig {
            max_schema_bytes: ceiling,
            ..SchemaCacheConfig::default()
        });
        assert!(matches!(
            cache.insert(&huge),
            Err(SchemaCacheError::SchemaTooLarge { .. })
        ));
        assert!(cache.is_empty());
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn amplified_order_is_refused() {
        let mut cache = SchemaResourceCache::with_config(SchemaCacheConfig {
            max_assembled_bytes: 4096,
            ..SchemaCacheConfig::default()
        });
        let id = cache
            .insert(&serde_json::json!({ "blob": "z".repeat(512) }))
            .unwrap();
        // One block is fine.
        assert!(cache.assemble(&[id]).is_ok());
        // The same identity repeated far enough exceeds the ceiling.
        let amplified = vec![id; 512];
        match cache.assemble(&amplified) {
            Err(SchemaCacheError::AssemblyTooLarge { actual, ceiling }) => {
                assert!(actual > ceiling);
                assert_eq!(ceiling, 4096);
            }
            other => panic!("expected AssemblyTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn compile_fresh_applies_cache_ceilings() {
        let config = SchemaCacheConfig {
            max_schema_bytes: 64,
            max_assembled_bytes: 4096,
            ..SchemaCacheConfig::default()
        };
        let big = serde_json::json!({ "blob": "x".repeat(128) });
        assert!(matches!(
            compile_fresh(&[&big], &config),
            Err(SchemaCacheError::SchemaTooLarge { ceiling: 64, .. })
        ));
        let small = serde_json::json!({ "blob": "z".repeat(32) });
        let amplified = vec![&small; 512];
        match compile_fresh(&amplified, &config) {
            Err(SchemaCacheError::AssemblyTooLarge { actual, ceiling }) => {
                assert_eq!(ceiling, 4096);
                // Refused at the first block that crosses the ceiling.
                assert!(actual <= ceiling + 64);
            }
            other => panic!("expected AssemblyTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn canonicalize_applies_default_ceiling() {
        let big = serde_json::json!("x".repeat(DEFAULT_MAX_SCHEMA_BYTES + 1));
        assert!(matches!(
            canonicalize(&big),
            Err(SchemaCacheError::SchemaTooLarge { .. })
        ));
    }

    #[test]
    fn escaped_string_is_refused_before_append() {
        // 100 raw bytes pass the raw-length check but escape to 602 bytes.
        let ctrl = serde_json::json!(["\u{1}".repeat(100)]);
        match canonicalize_bounded(&ctrl, 128) {
            Err(SchemaCacheError::SchemaTooLarge { actual, ceiling }) => {
                assert_eq!(ceiling, 128);
                assert_eq!(actual, 1 + 602);
            }
            other => panic!("expected SchemaTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn depth_overflow_fails_loud() {
        let mut nested = serde_json::json!(0);
        for _ in 0..(MAX_CANONICAL_DEPTH + 2) {
            nested = serde_json::json!([nested]);
        }
        assert_eq!(
            canonicalize(&nested),
            Err(SchemaCacheError::DepthExceeded(MAX_CANONICAL_DEPTH))
        );
    }

    #[test]
    fn cross_identity_isolation() {
        let mut cache = SchemaResourceCache::new();
        let a = cache.insert(&serde_json::json!({"name": "a"})).unwrap();
        let b = cache.insert(&serde_json::json!({"name": "b"})).unwrap();
        assert_ne!(a, b);
        let block_a = cache.assemble(&[a]).unwrap();
        let block_b = cache.assemble(&[b]).unwrap();
        assert_ne!(block_a, block_b);
        assert!(block_a.contains("\"a\""));
        assert!(block_b.contains("\"b\""));
    }

    #[test]
    fn zero_capacity_is_refused() {
        let mut cache = SchemaResourceCache::with_config(SchemaCacheConfig {
            capacity: 0,
            ..SchemaCacheConfig::default()
        });
        assert_eq!(
            cache.insert(&serde_json::json!({})),
            Err(SchemaCacheError::ZeroCapacity)
        );
    }

    #[test]
    fn insert_tool_uses_full_definition_identity() {
        use crate::tools::McpGateTools;
        let mut cache = SchemaResourceCache::new();
        let tools = McpGateTools::list_tools();
        let ids: Vec<ResourceId> = tools
            .iter()
            .map(|t| cache.insert_tool(t).unwrap())
            .collect();
        // All gate tools are distinct resources.
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), tools.len());
        // Re-inserting is a hit, not a recompile.
        let before = cache.stats().misses;
        cache.insert_tool(&tools[0]).unwrap();
        assert_eq!(cache.stats().misses, before);
    }
}
