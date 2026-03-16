# ADR-045: Signal Extensions Plugin Architecture

| Field       | Value                                          |
|-------------|------------------------------------------------|
| Status      | Proposed                                       |
| Date        | 2026-02-21                                     |
| Authors     | @grparry (proposal)                            |
| Supersedes  | —                                              |
| Extends     | ADR-043 (External Intelligence Providers)      |

## Context

ADR-043 established the `IntelligenceProvider` trait and `QualitySignal` type for feeding external quality data into ruvLLM. The current `QualitySignal` carries a composite score, an outcome, and 9 optional quality factors. This covers the common case well.

However, different consumers of ruvLLM operate in very different environments — workflow engines, CI/CD pipelines, coding assistants, multi-agent frameworks, IDE plugins — and each has domain-specific context that could improve learning quality if it could be attached to signals. Today, that context is lost because `QualitySignal` has a fixed schema with no extension point.

### Examples of Consumer-Specific Context

These are illustrative, not prescriptive — the point is that each consumer has *different* structured data:

- **Workflow engines** may want to attach execution step sequences, timing data, or parent-child execution relationships.
- **CI/CD pipelines** may want to attach pipeline stage, test matrix configuration, or build environment details.
- **Coding assistants** may want to attach file paths modified, AST diff summaries, or language-specific metrics.
- **Multi-agent systems** may want to attach agent role assignments, inter-agent message counts, or coordination overhead.

No single set of fields can serve all of these. Adding consumer-specific fields to `QualitySignal` would create a maintenance burden upstream and couple ruvLLM to external systems' data models.

### Design Principle

Rather than prescribing what structured data looks like, give consumers a generic, typed extension mechanism. RuvLLM provides the hooks; consumers define the content.

## Decision

**Add an optional `extensions` field to `QualitySignal` and a `SignalExtensionHandler` trait for consumers that want ruvLLM to act on their extension data.**

This is a two-layer design:

1. **Transport layer** — `QualitySignal.extensions` carries arbitrary typed JSON, keyed by namespace. No ruvLLM code changes needed to add new extension types. Any provider can attach any data.

2. **Processing layer** — consumers that want ruvLLM to *use* their extension data (e.g., for richer SONA trajectories or HNSW features) register a `SignalExtensionHandler`. This is optional — unhandled extensions are preserved but not processed.

### Changes to QualitySignal

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySignal {
    // ── Existing fields (unchanged) ──────────────────────────────
    pub id: String,
    pub task_description: String,
    pub outcome: Outcome,
    pub quality_score: f32,
    #[serde(default)]
    pub human_verdict: Option<HumanVerdict>,
    #[serde(default)]
    pub quality_factors: Option<QualityFactors>,
    pub completed_at: String,

    // ── New: generic extension data ──────────────────────────────
    /// Provider-specific structured data, keyed by namespace.
    ///
    /// Namespaces prevent collisions between providers. Convention:
    /// use your provider name or organization as the namespace key
    /// (e.g., "my-pipeline", "ci-system", "code-assistant").
    ///
    /// RuvLLM preserves extension data through the signal pipeline.
    /// To have ruvLLM act on extension data (extract SONA features,
    /// generate embeddings, etc.), register a `SignalExtensionHandler`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<HashMap<String, serde_json::Value>>,
}
```

### SignalExtensionHandler Trait

```rust
/// Handler for processing provider-specific extension data on signals.
///
/// Consumers register handlers with `IntelligenceLoader` to extract
/// features from their extension data. Handlers are called during
/// signal ingestion, after the core signal fields have been processed.
///
/// This is optional — signals with unhandled extensions are still
/// ingested normally using their core fields.
pub trait SignalExtensionHandler: Send + Sync {
    /// The namespace this handler processes (must match extension key).
    fn namespace(&self) -> &str;

    /// Extract additional SONA trajectory features from extension data.
    ///
    /// Returns key-value pairs that are merged into the trajectory's
    /// metadata. Called once per signal during ingestion.
    fn extract_trajectory_features(
        &self,
        extension_data: &serde_json::Value,
        signal: &QualitySignal,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let _ = (extension_data, signal);
        Ok(HashMap::new())
    }

    /// Extract additional embedding features from extension data.
    ///
    /// Returns text fragments that are appended to the task description
    /// before embedding generation. This allows extension data to
    /// influence HNSW clustering without changing the embedding model.
    fn extract_embedding_context(
        &self,
        extension_data: &serde_json::Value,
        signal: &QualitySignal,
    ) -> Result<Vec<String>> {
        let _ = (extension_data, signal);
        Ok(vec![])
    }

    /// Extract additional router calibration features.
    ///
    /// Returns key-value pairs used as features when the model router
    /// estimates task complexity. Called during calibration updates.
    fn extract_router_features(
        &self,
        extension_data: &serde_json::Value,
        signal: &QualitySignal,
    ) -> Result<HashMap<String, f32>> {
        let _ = (extension_data, signal);
        Ok(HashMap::new())
    }
}
```

### Changes to IntelligenceLoader

```rust
impl IntelligenceLoader {
    // Existing methods unchanged

    /// Register a handler for processing extension data in a specific namespace.
    pub fn register_extension_handler(
        &mut self,
        handler: Box<dyn SignalExtensionHandler>,
    ) {
        self.extension_handlers.insert(
            handler.namespace().to_string(),
            handler,
        );
    }
}
```

During `load_all_signals()`, after processing core signal fields, the loader checks each signal's `extensions` map. For each namespace key that has a registered handler, it calls the handler's extraction methods and merges the results into the trajectory/embedding/router data.

## Architecture

```
Provider writes signal          Consumer registers handler
       │                                    │
       ▼                                    ▼
┌──────────────┐                ┌─────────────────────────┐
│ QualitySignal │                │ SignalExtensionHandler   │
│   .extensions │──namespace──▶ │   .namespace()           │
│     {"my-ns": │    match      │   .extract_trajectory_*  │
│       {...}}  │               │   .extract_embedding_*   │
└──────────────┘                │   .extract_router_*      │
       │                        └─────────────────────────┘
       ▼                                    │
┌──────────────┐                            ▼
│ Intelligence │              ┌───────────────────────┐
│   Loader     │─── calls ──▶│ Merge extracted features│
│              │              │ into SONA / HNSW /     │
│              │              │ router pipelines       │
└──────────────┘              └───────────────────────┘
```

**No handler registered?** The signal is ingested normally using its core fields. The extension data is preserved in the signal record but not acted upon. Zero overhead.

## Design Constraints

- **Backward compatible.** `extensions` is `Option` with `#[serde(default)]`. Existing signals deserialize without error. Existing providers compile and run without changes.
- **No prescribed schema.** ruvLLM does not define what goes in extension data. Each consumer defines their own namespace and schema.
- **Namespace isolation.** Multiple providers can attach extension data to the same signal without collision. Each handler only processes its own namespace.
- **All handler methods have defaults.** Consumers implement only the extraction methods relevant to their data. A handler that only cares about SONA trajectories can skip embedding and router methods.
- **Payload size is consumer-controlled.** ruvLLM does not validate extension data size. Consumers are responsible for keeping their extensions reasonable. The file-based provider's existing 10 MiB limit provides a natural ceiling.

## Existing Code References

| Item | Status | Location |
|------|--------|----------|
| `QualitySignal` struct | EXISTS | `crates/ruvllm/src/intelligence/mod.rs` |
| `QualityFactors` struct | EXISTS | `crates/ruvllm/src/intelligence/mod.rs` |
| `IntelligenceProvider` trait | EXISTS | `crates/ruvllm/src/intelligence/mod.rs` |
| `FileSignalProvider` | EXISTS | `crates/ruvllm/src/intelligence/mod.rs` |
| `IntelligenceLoader` | EXISTS | `crates/ruvllm/src/intelligence/mod.rs` |
| `SignalExtensionHandler` trait | NEW | `crates/ruvllm/src/intelligence/mod.rs` |

## Implementation

### Files Modified

| # | Path | Changes |
|---|------|---------|
| 1 | `crates/ruvllm/src/intelligence/mod.rs` | Add `extensions` field to `QualitySignal`. Add `SignalExtensionHandler` trait. Add `register_extension_handler()` to `IntelligenceLoader`. Add handler dispatch in signal ingestion. |
| 2 | `npm/packages/ruvllm/src/intelligence.ts` | Add `extensions?: Record<string, unknown>` to `QualitySignal` interface. Add `SignalExtensionHandler` interface. |

### Files Created

| # | Path | Description |
|---|------|-------------|
| 1 | `docs/adr/ADR-045-structured-quality-signals.md` | This ADR |

## Example: Consumer Implementation

A workflow engine that wants ruvLLM to learn from execution step sequences would:

1. **Populate extensions when emitting signals:**

```json
{
  "id": "task-123",
  "task_description": "Refactor authentication module",
  "outcome": "success",
  "quality_score": 0.87,
  "extensions": {
    "my-workflow-engine": {
      "steps": [
        { "name": "plan", "duration_ms": 2300 },
        { "name": "implement", "duration_ms": 15000 },
        { "name": "review", "duration_ms": 4200 }
      ],
      "total_nodes": 5,
      "parent_execution": "exec-456"
    }
  }
}
```

2. **Register a handler to extract features:**

```rust
struct MyWorkflowHandler;

impl SignalExtensionHandler for MyWorkflowHandler {
    fn namespace(&self) -> &str { "my-workflow-engine" }

    fn extract_trajectory_features(
        &self,
        data: &serde_json::Value,
        _signal: &QualitySignal,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let mut features = HashMap::new();
        if let Some(steps) = data.get("steps") {
            features.insert("step_count".into(), json!(steps.as_array().map(|a| a.len()).unwrap_or(0)));
        }
        if let Some(parent) = data.get("parent_execution") {
            features.insert("parent_execution".into(), parent.clone());
        }
        Ok(features)
    }

    fn extract_router_features(
        &self,
        data: &serde_json::Value,
        _signal: &QualitySignal,
    ) -> Result<HashMap<String, f32>> {
        let mut features = HashMap::new();
        if let Some(n) = data.get("total_nodes").and_then(|v| v.as_f64()) {
            features.insert("pipeline_complexity".into(), n as f32);
        }
        Ok(features)
    }
}

// Registration:
loader.register_extension_handler(Box::new(MyWorkflowHandler));
```

A CI/CD system would do the same with its own namespace and its own data shape — without any ruvLLM changes.

## Consequences

### Positive

1. **Consumers own their data model.** No upstream PRs needed to add new structured data. Define a namespace, populate it, optionally register a handler.
2. **ruvLLM stays generic.** The core signal type doesn't accumulate domain-specific fields from every consumer.
3. **Incremental adoption.** Start by attaching extension data (preserved but not processed). Add a handler later when you want ruvLLM to act on it.
4. **Fully backward compatible.** Existing signals, providers, and consumers work unchanged.
5. **Composable.** Multiple consumers can attach extensions to the same signal. Multiple handlers can be registered. They don't interfere with each other.

### Negative

1. **No schema validation.** Extension data is `serde_json::Value` — ruvLLM cannot validate its structure. Schema enforcement is the handler's responsibility.
2. **Handler registration requires Rust.** Non-Rust consumers can attach extension data via JSON files, but processing that data requires a Rust handler. (Mitigation: a future ADR could add a declarative handler config for common patterns like "extract field X as router feature Y".)
3. **Testing surface.** Each handler needs its own tests. ruvLLM's test suite covers the dispatch mechanism; consumers test their handlers.

## Related Decisions

- **ADR-002**: RuvLLM Integration with Ruvector — Witness Log schema with `quality_score: f32`
- **ADR-043**: External Intelligence Providers — established `IntelligenceProvider` trait and `QualitySignal` type (this ADR extends the signal type with a generic extension point)
- **ADR-CE-021**: Shared SONA — multiple external systems contributing trajectories
- **ADR-004**: KV Cache Management — tiered approach benefiting from richer calibration signals
