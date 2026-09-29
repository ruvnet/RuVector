//! Gateway ↔ Durable Object wire protocol (ADR-351 §4.3, §6).
//!
//! The gateway is the only caller of both DO classes. Every request names
//! the tenant (from the verified token, never from request input) and the
//! DO asserts it against its stored identity (`LedgerMeta` / `DoMeta`
//! `check`, mismatch → `404`). Each call is handled synchronously inside
//! the DO — no `await` between reading resident state and issuing SQL — so
//! one call is one coalesced SQLite commit and the DO is the consistency
//! boundary.

use ruvector_edge_auth::Capability;
use ruvector_edge_store::shard::{UpsertRow, UsageDelta};
use ruvector_edge_store::{ErrorCode, Metric, OpError, QueryRequest, ShardConfig};
use ruvector_edge_tenancy::quota::limits::M1_SHARD_FLOAT_CAP;
use ruvector_edge_tenancy::QuotaDelta;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

/// Error across the DO boundary: stable code plus the step-up scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireErr {
    /// Stable code.
    pub code: ErrorCode,
    /// `insufficient_scope` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

impl WireErr {
    /// From an executor error.
    pub fn from_op(e: &OpError) -> Self {
        WireErr {
            code: e.code,
            scope: e.scope.map(str::to_string),
        }
    }

    /// Back to an [`OpError`]. `detail` is static and never echoes input,
    /// so it is not transported (the code stands in for it); `scope` is re-derived from the fixed
    /// vocabulary (an unknown value is dropped).
    pub fn into_op(self) -> OpError {
        let scope = self.scope.and_then(|s| {
            [
                Capability::Read,
                Capability::Write,
                Capability::Admin,
                Capability::PublishPublic,
            ]
            .into_iter()
            .map(Capability::satisfying_scope)
            .find(|k| *k == s)
        });
        OpError {
            code: self.code,
            detail: self.code.as_str(),
            scope,
        }
    }
}

/// Transport failure (DO unreachable, undecodable reply): retryable.
pub fn unavailable() -> OpError {
    OpError::new(ErrorCode::ShardUnavailable, "durable object unavailable")
}

/// Shard configuration as the ledger's catalog defines it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CfgWire {
    /// Dimension.
    pub dim: u32,
    /// Metric.
    pub metric: Metric,
    /// Declared filterable keys.
    pub filterable_keys: Vec<String>,
}

impl CfgWire {
    /// The store's shard config (M1 float cap).
    pub fn to_config(&self) -> ShardConfig {
        ShardConfig {
            dim: self.dim,
            metric: self.metric,
            filterable_keys: self.filterable_keys.clone(),
            float_cap: M1_SHARD_FLOAT_CAP,
        }
    }
}

/// A live catalog entry as the gateway needs it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionWire {
    /// 32 lowercase hex chars.
    pub uid: String,
    /// `1..=6`.
    pub shard_count: u32,
    /// Shard configuration.
    pub cfg: CfgWire,
    /// Public JSON view (`CatalogEntry::to_json`).
    pub view: Json,
}

/// Signed usage change of a shard write.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeltaWire {
    /// Net vectors.
    pub vectors: i64,
    /// Net floats.
    pub floats: i64,
    /// Net stored bytes.
    pub bytes: i64,
}

impl From<UsageDelta> for DeltaWire {
    fn from(d: UsageDelta) -> Self {
        DeltaWire {
            vectors: d.vectors,
            floats: d.floats,
            bytes: d.bytes,
        }
    }
}

impl DeltaWire {
    /// `self + o`, saturating.
    pub fn plus(self, o: DeltaWire) -> DeltaWire {
        DeltaWire {
            vectors: self.vectors.saturating_add(o.vectors),
            floats: self.floats.saturating_add(o.floats),
            bytes: self.bytes.saturating_add(o.bytes),
        }
    }
    /// `self - o`, saturating.
    pub fn minus(self, o: DeltaWire) -> DeltaWire {
        self.plus(o.neg())
    }
    /// `-self`, saturating.
    pub fn neg(self) -> DeltaWire {
        DeltaWire {
            vectors: self.vectors.saturating_neg(),
            floats: self.floats.saturating_neg(),
            bytes: self.bytes.saturating_neg(),
        }
    }
    /// `true` if any component of `self` grows beyond `limit`.
    pub fn exceeds(self, limit: DeltaWire) -> bool {
        self.vectors > limit.vectors || self.floats > limit.floats || self.bytes > limit.bytes
    }
    /// The ledger quota delta of this change (no op count).
    pub fn quota(self) -> QuotaDelta {
        QuotaDelta {
            collections: 0,
            vectors: self.vectors,
            float_budget: self.floats,
            bytes: self.bytes,
            ops: 0,
        }
    }
    /// JSON form.
    pub fn to_json(self) -> Json {
        json!({ "vectors": self.vectors, "floats": self.floats, "bytes": self.bytes })
    }
}

/// Who wrote (recorded in the shard `ops` log).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActorWire {
    /// Edge subject.
    pub sub: String,
    /// Token id.
    pub jti: String,
    /// Grant family.
    pub family_id: String,
    /// `act.sub` of an exchanged token (the adapter that acted, ADR-351
    /// §5.6/§16.3), recorded as `ops.act_sub`; `None` for direct calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub act_sub: Option<String>,
}

/// One match with its exact rank key (`f64` bits, so the cross-shard merge
/// in the gateway is bit-exact).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchWire {
    /// Vector id.
    pub id: String,
    /// Wire distance.
    pub distance: f32,
    /// Metadata, when requested / present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Json>,
    /// Values, when requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f32>>,
    /// `f64::to_bits` of the rank key.
    pub score_bits: u64,
}

impl MatchWire {
    /// Merge order: `(score, id)` ascending (the store's `rank_cmp`).
    pub fn rank_cmp(&self, o: &MatchWire) -> std::cmp::Ordering {
        f64::from_bits(self.score_bits)
            .total_cmp(&f64::from_bits(o.score_bits))
            .then_with(|| self.id.cmp(&o.id))
    }
    /// Public JSON view (as the store's `Match` serialises).
    pub fn to_public(&self) -> Json {
        let mut m = json!({ "id": self.id, "distance": self.distance });
        if let Some(md) = &self.metadata {
            m["metadata"] = md.clone();
        }
        if let Some(v) = &self.values {
            m["values"] = json!(v);
        }
        m
    }
}

/// A `TenantLedger` DO request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerRequest {
    /// Tenant key from the verified token.
    pub tenant_key: String,
    /// The call.
    pub call: LedgerCall,
}

/// `TenantLedger` calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum LedgerCall {
    /// Role of `sub` and whether the tenant is claimed.
    Access { sub: String },
    /// `tenant:claim`.
    Claim { sub: String, now: u64 },
    /// One live collection by name.
    Collection { name: String },
    /// Every live collection.
    Collections,
    /// Validate a create spec (raw JSON args) without writing.
    ValidateCreate { spec: String },
    /// Create a collection.
    CreateCollection { spec: String, sub: String, now: u64 },
    /// Admit growth (limits enforced).
    Admit {
        delta: QuotaDelta,
        work_units: u64,
        now: u64,
    },
    /// Internal correction (no limits, clamped).
    Adjust { delta: QuotaDelta, now: u64 },
    /// Usage, limits, work units.
    Usage { now: u64 },
    /// `op_id` lookup (`sha256` of the raw body).
    IdemLookup {
        sub: String,
        key: String,
        sha256: [u8; 32],
        now: u64,
    },
    /// `op_id` lookup that, on a miss, reserves the key for this request in
    /// the same DO turn (concurrent twins see `in_flight`).
    IdemReserve {
        sub: String,
        key: String,
        sha256: [u8; 32],
        now: u64,
    },
    /// Clear this request's reservation (the op failed / is not remembered).
    IdemRelease {
        sub: String,
        key: String,
        sha256: [u8; 32],
    },
    /// Remember a response under `op_id`.
    IdemStore {
        sub: String,
        key: String,
        sha256: [u8; 32],
        response: String,
        now: u64,
    },
}

/// `TenantLedger` results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "out", rename_all = "snake_case")]
pub enum LedgerOut {
    /// For `Access`.
    Access { role: Option<String>, claimed: bool },
    /// For `Claim`.
    Role { role: String },
    /// For `Collection`.
    Collection { entry: Option<CollectionWire> },
    /// For `Collections` / `CreateCollection` (one entry).
    Collections { entries: Vec<CollectionWire> },
    /// For `ValidateCreate`.
    Validated { name: String, shards: u32 },
    /// For `Admit` / `Adjust` / `IdemStore` / `IdemRelease`.
    Done,
    /// For `Usage`.
    Usage { report: Json },
    /// For `IdemLookup` / `IdemReserve`: `None` miss, `Some(body)` replay;
    /// `in_flight`: reserved by an unfinished request with the same body.
    Idem {
        replay: Option<String>,
        conflict: bool,
        #[serde(default)]
        in_flight: bool,
    },
}

/// A `VectorShard` DO request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardRequest {
    /// Tenant key from the verified token.
    pub tenant_key: String,
    /// `collection_uid` (32 hex).
    pub uid: String,
    /// Decimal shard index.
    pub shard: u32,
    /// The call.
    pub call: ShardCall,
}

/// `VectorShard` calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum ShardCall {
    /// Validate an upsert, report its delta (writes nothing).
    Plan { cfg: CfgWire, rows: Vec<UpsertRow> },
    /// Re-plan and apply atomically; refused (`409`) if the fresh delta
    /// grows beyond what the ledger admitted.
    Apply {
        cfg: CfgWire,
        rows: Vec<UpsertRow>,
        admitted: DeltaWire,
        actor: ActorWire,
        now: u64,
    },
    /// Exact top-k; `steps_before` is the fan-out's running scan budget.
    Query {
        cfg: CfgWire,
        req: QueryRequest,
        steps_before: u64,
    },
    /// Fetch ids.
    Fetch {
        ids: Vec<String>,
        include_values: bool,
    },
    /// Delete ids.
    Delete {
        ids: Vec<String>,
        actor: ActorWire,
        dry_run: bool,
        now: u64,
    },
    /// Size and sequence counters.
    Stats,
}

/// `VectorShard` results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "out", rename_all = "snake_case")]
pub enum ShardOut {
    /// For `Plan`.
    Planned { delta: DeltaWire },
    /// For `Apply` / `Delete`.
    Written {
        count: u64,
        write_seq: u64,
        delta: DeltaWire,
    },
    /// A write failed after issuing statements: the shard was reopened from
    /// storage and `applied` is what really changed (`None`: unknowable).
    Failed {
        err: WireErr,
        applied: Option<DeltaWire>,
    },
    /// For `Query`.
    Matches {
        matches: Vec<MatchWire>,
        scanned: u64,
        steps: u64,
    },
    /// For `Fetch`.
    Fetched { matches: Vec<MatchWire> },
    /// For `Stats`.
    Stats {
        count: u64,
        resident_bytes: u64,
        write_seq: u64,
        snapshot_seq: u64,
    },
}

/// A DO reply body.
pub type Reply<T> = Result<T, WireErr>;
