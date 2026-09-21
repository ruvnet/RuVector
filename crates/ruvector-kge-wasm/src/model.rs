//! Pure, target-independent glue between the binding surface and
//! `ruvector-kge` core. This file is duplicated VERBATIM in
//! `ruvector-kge-ffi/src/model.rs` and `ruvector-kge-wasm/src/model.rs`
//! (`cmp` them to verify); the napi / wasm-bindgen wrappers in each crate's
//! `lib.rs` are the only per-target code.
//!
//! The binding owns the vocabulary (label <-> id): the core never sees strings,
//! so nothing here or below it can log or leak triple text (ADR-005). Request
//! errors are returned as `{"error":{"kind","message"}}`; only programmer
//! errors (bad options JSON, odd/zero `dims`, a tampered model) surface as an
//! `Err(String)` that the wrapper throws.

use ruvector_kge::{AnnIndex, KgeError, ScorerKind, Tables, Triple};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Serialized-model schema version, bumped if the on-disk shape changes.
const SCHEMA_VERSION: u32 = 1;

// ---- error envelope -------------------------------------------------------

/// The closed set of request-error kinds, mirrored in the TypeScript
/// `errors.ts` guard and `index.d.ts`: `limit | invalid | unavailable |
/// unsupported | scorer`.
pub fn err_json(kind: &str, message: &str) -> String {
    serde_json::json!({ "error": { "kind": kind, "message": message } }).to_string()
}

/// Map a core `KgeError` onto the envelope. Ids are numeric (no label text).
pub fn kge_error_json(e: &KgeError) -> String {
    let (kind, msg) = match e {
        KgeError::Limit(m) => ("limit", (*m).to_string()),
        KgeError::Invalid(m) => ("invalid", m.clone()),
        KgeError::Dims { expected, got } => (
            "invalid",
            format!("dimension mismatch: expected {expected}, got {got}"),
        ),
        KgeError::UnknownEntity(id) => ("invalid", format!("unknown entity id {id}")),
        KgeError::UnknownRelation(id) => ("invalid", format!("unknown relation id {id}")),
        KgeError::Scorer(m) => ("scorer", m.clone()),
    };
    err_json(kind, &msg)
}

// ---- vocabulary -----------------------------------------------------------

/// Insertion-ordered label table. `labels[id]` is the string; `index` is the
/// reverse map, rebuilt after load (it is `#[serde(skip)]`, so the serialized
/// model carries only the label list — compact and deterministic).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Vocab {
    labels: Vec<String>,
    #[serde(skip)]
    index: HashMap<String, u32>,
}

impl Vocab {
    fn intern(&mut self, label: &str) -> u32 {
        if let Some(&id) = self.index.get(label) {
            return id;
        }
        let id = self.labels.len() as u32;
        self.labels.push(label.to_string());
        self.index.insert(label.to_string(), id);
        id
    }
    pub fn get(&self, label: &str) -> Option<u32> {
        self.index.get(label).copied()
    }
    pub fn label(&self, id: u32) -> Option<&str> {
        self.labels.get(id as usize).map(String::as_str)
    }
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    fn rebuild_index(&mut self) {
        self.index = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i as u32))
            .collect();
    }
}

// ---- config ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Config {
    pub scorer: ScorerKind,
    pub dims: usize,
    pub seed: u64,
}

fn default_scorer() -> ScorerKind {
    ScorerKind::Hole
}
fn default_dims() -> usize {
    256
}
fn default_seed() -> u64 {
    42
}

#[derive(Deserialize)]
struct Options {
    #[serde(default = "default_scorer")]
    scorer: ScorerKind,
    #[serde(default = "default_dims")]
    dims: usize,
    #[serde(default = "default_seed")]
    seed: u64,
}

// ---- the model ------------------------------------------------------------

/// One knowledge-graph embedding model: config + vocab + triples + lazily
/// built tables, plus an optional ANN index. Adding triples *grows* the tables
/// (old rows — trained or not — are copied, new rows get seed-init), so a later
/// `train` is not silently discarded. The ANN index is transient
/// (`#[serde(skip)]`): it is dropped on any table change and is not saved, so
/// after `fromJson` `predict` is exhaustive until `buildIndex` runs again.
#[derive(Serialize, Deserialize)]
pub struct KgeModel {
    version: u32,
    pub config: Config,
    pub entities: Vocab,
    pub relations: Vocab,
    pub triples: Vec<Triple>,
    /// Split label per triple, parallel to `triples`. Lets a caller (e.g. the
    /// bench harness) pin a frozen split so `train`/`eval` honour it instead of
    /// deriving one.
    #[serde(default)]
    pub splits: Vec<SplitLabel>,
    pub tables: Option<Tables>,
    #[serde(skip)]
    pub ann: Option<AnnIndex>,
}

/// A triple's split membership (ADR-006 frozen splits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitLabel {
    Unlabelled,
    Train,
    Valid,
    Transfer,
    Test,
}

impl SplitLabel {
    /// Parse a split name; `None` for an unknown name. `unlabelled` is internal
    /// only — a triple gets it by omitting `split`, never by naming it.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "train" => Some(Self::Train),
            "valid" => Some(Self::Valid),
            "transfer" => Some(Self::Transfer),
            "test" => Some(Self::Test),
            _ => None,
        }
    }
}

impl KgeModel {
    /// `optionsJson`: `{"scorer":"hole"|"rotate","dims":256,"seed":42}`. Throws
    /// (returns `Err`) on malformed JSON or a `dims` that is zero or odd —
    /// ADR-001 requires an even `d` for the HolE real-FFT pairing.
    pub fn new(options_json: &str) -> Result<Self, String> {
        let o: Options =
            serde_json::from_str(options_json).map_err(|e| format!("invalid options JSON: {e}"))?;
        if o.dims == 0 || !o.dims.is_multiple_of(2) {
            return Err(format!(
                "dims must be a positive even number, got {}",
                o.dims
            ));
        }
        Ok(Self {
            version: SCHEMA_VERSION,
            config: Config {
                scorer: o.scorer,
                dims: o.dims,
                seed: o.seed,
            },
            entities: Vocab::default(),
            relations: Vocab::default(),
            triples: Vec::new(),
            splits: Vec::new(),
            tables: None,
            ann: None,
        })
    }

    /// Parse a `{"sha256","model"}` envelope, fail closed on a hash mismatch
    /// (ADR-005 manifest discipline). The hash covers the canonical struct
    /// serialization, so any edit to the payload — or to the hash — throws.
    pub fn from_json(json: &str) -> Result<Self, String> {
        #[derive(Deserialize)]
        struct Envelope {
            sha256: String,
            model: serde_json::Value,
        }
        let env: Envelope =
            serde_json::from_str(json).map_err(|e| format!("invalid model JSON: {e}"))?;
        let mut model: KgeModel =
            serde_json::from_value(env.model).map_err(|e| format!("invalid model payload: {e}"))?;
        let body = serde_json::to_string(&model).map_err(|e| e.to_string())?;
        if sha256_hex(body.as_bytes()) != env.sha256 {
            return Err("model hash mismatch: payload is tampered or corrupt".to_string());
        }
        model.entities.rebuild_index();
        model.relations.rebuild_index();
        Ok(model)
    }

    /// Serialize to a hash-carrying `{"sha256","model"}` envelope.
    pub fn to_json(&self) -> String {
        let body = serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string());
        let hash = sha256_hex(body.as_bytes());
        format!("{{\"sha256\":\"{hash}\",\"model\":{body}}}")
    }

    /// `{"scorer","dims","seed","entities","relations","triples","indexed"}`.
    pub fn stats_json(&self) -> String {
        serde_json::json!({
            "scorer": self.config.scorer,
            "dims": self.config.dims,
            "seed": self.config.seed,
            "entities": self.entities.len(),
            "relations": self.relations.len(),
            "triples": self.triples.len(),
            "indexed": self.ann.is_some(),
            "splits": self.split_counts(),
        })
        .to_string()
    }

    /// Build the tables from the current vocab if absent. `max(1)` guards the
    /// empty case so `Tables::new` never allocates a zero-row matrix.
    pub fn ensure_built(&mut self) {
        if self.tables.is_none() {
            let ne = self.entities.len().max(1);
            let nr = self.relations.len().max(1);
            self.tables = Some(Tables::new(ne, nr, self.config.dims, self.config.seed));
        }
    }

    pub(crate) fn intern_entity(&mut self, label: &str) -> u32 {
        self.entities.intern(label)
    }
    pub(crate) fn intern_relation(&mut self, label: &str) -> u32 {
        self.relations.intern(label)
    }

    /// Drop the ANN index (call whenever the tables change).
    pub(crate) fn invalidate_index(&mut self) {
        self.ann = None;
    }

    /// Install a freshly built ANN index.
    pub(crate) fn set_index(&mut self, index: AnnIndex) {
        self.ann = Some(index);
    }

    /// Record a triple's split label (parallel to `triples`).
    pub(crate) fn push_split(&mut self, label: SplitLabel) {
        self.splits.push(label);
    }

    /// True if any triple carries a real (non-`Unlabelled`) split label.
    pub(crate) fn has_split_tags(&self) -> bool {
        self.splits.iter().any(|&l| l != SplitLabel::Unlabelled)
    }

    /// The triples labelled `label`.
    pub(crate) fn triples_with_split(&self, label: SplitLabel) -> Vec<Triple> {
        self.triples
            .iter()
            .zip(self.splits.iter())
            .filter(|(_, &l)| l == label)
            .map(|(t, _)| *t)
            .collect()
    }

    /// `{train,valid,transfer,test,unlabelled}` triple counts.
    pub(crate) fn split_counts(&self) -> serde_json::Value {
        let mut c = [0usize; 5]; // Unlabelled, Train, Valid, Transfer, Test
        for &l in &self.splits {
            c[l as usize] += 1;
        }
        serde_json::json!({
            "train": c[SplitLabel::Train as usize],
            "valid": c[SplitLabel::Valid as usize],
            "transfer": c[SplitLabel::Transfer as usize],
            "test": c[SplitLabel::Test as usize],
            "unlabelled": c[SplitLabel::Unlabelled as usize],
        })
    }

    /// Grow the tables to the current vocab size, preserving existing rows
    /// (trained weights are not discarded); new rows keep their seed-init. A
    /// no-op when the tables have not been built yet — `ensure_built` will size
    /// them to the full current vocab on first use.
    pub(crate) fn grow_tables(&mut self) {
        let Some(old) = self.tables.take() else {
            return;
        };
        let ne = self.entities.len().max(1);
        let nr = self.relations.len().max(1);
        if old.num_entities() == ne && old.num_relations() == nr {
            self.tables = Some(old);
            return;
        }
        let mut grown = Tables::new(ne, nr, self.config.dims, self.config.seed);
        for i in 0..old.num_entities() as u32 {
            let src = old.entity(i).unwrap().to_vec();
            grown.entity_mut(i).unwrap().copy_from_slice(&src);
        }
        for i in 0..old.num_relations() as u32 {
            let src = old.relation(i).unwrap().to_vec();
            grown.relation_mut(i).unwrap().copy_from_slice(&src);
        }
        self.tables = Some(grown);
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}
