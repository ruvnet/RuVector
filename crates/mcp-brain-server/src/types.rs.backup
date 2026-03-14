//! Shared types for the brain server

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Brain memory categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BrainCategory {
    Architecture,
    Pattern,
    Solution,
    Convention,
    Security,
    Performance,
    Tooling,
    Debug,
    Custom(String),
}

impl std::fmt::Display for BrainCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Architecture => write!(f, "architecture"),
            Self::Pattern => write!(f, "pattern"),
            Self::Solution => write!(f, "solution"),
            Self::Convention => write!(f, "convention"),
            Self::Security => write!(f, "security"),
            Self::Performance => write!(f, "performance"),
            Self::Tooling => write!(f, "tooling"),
            Self::Debug => write!(f, "debug"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// A shared brain memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainMemory {
    pub id: Uuid,
    pub category: BrainCategory,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub code_snippet: Option<String>,
    pub embedding: Vec<f32>,
    pub contributor_id: String,
    pub quality_score: BetaParams,
    pub partition_id: Option<u32>,
    pub witness_hash: String,
    pub rvf_gcs_path: Option<String>,
    /// JSON-serialized RedactionLog from rvf-federation PiiStripper (Phase 2, ADR-075)
    #[serde(default)]
    pub redaction_log: Option<String>,
    /// JSON-serialized DiffPrivacyProof from rvf-federation DiffPrivacyEngine (Phase 3, ADR-075)
    #[serde(default)]
    pub dp_proof: Option<String>,
    /// Raw witness chain bytes from rvf-crypto create_witness_chain (Phase 4, ADR-075)
    #[serde(default)]
    pub witness_chain: Option<Vec<u8>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Bayesian quality scoring (Beta distribution)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaParams {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaParams {
    pub fn new() -> Self {
        Self { alpha: 1.0, beta: 1.0 }
    }

    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    pub fn observations(&self) -> f64 {
        self.alpha + self.beta - 2.0
    }

    pub fn upvote(&mut self) {
        self.alpha += 1.0;
    }

    pub fn downvote(&mut self) {
        self.beta += 1.0;
    }
}

impl Default for BetaParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Contributor info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributorInfo {
    pub pseudonym: String,
    pub reputation: ReputationScore,
    pub contribution_count: u64,
    pub created_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
    pub is_system: bool,
}

/// Multi-factor reputation score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    pub accuracy: f64,
    pub uptime: f64,
    pub stake: f64,
    pub composite: f64,
}

impl ReputationScore {
    pub fn cold_start() -> Self {
        Self {
            accuracy: 0.5,
            uptime: 0.2,
            stake: 0.0,
            composite: 0.1,
        }
    }

    pub fn compute_composite(&mut self) {
        let stake_weight = (self.stake + 1.0).log10().min(6.0) / 6.0;
        // Use max(0.3) so non-staked contributors can still reach threshold
        // via accuracy and uptime alone
        self.composite = self.accuracy.powi(2) * self.uptime * stake_weight.max(0.3);
    }

    pub fn apply_decay(&mut self, months_inactive: f64) {
        let decay = 0.95_f64.powf(months_inactive);
        self.composite *= decay;
    }

    pub fn apply_poisoning_penalty(&mut self) {
        self.composite *= 0.5;
        self.accuracy *= 0.5;
    }
}

impl Default for ReputationScore {
    fn default() -> Self {
        Self::cold_start()
    }
}

/// Drift report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    pub domain: Option<String>,
    pub coefficient_of_variation: f64,
    pub is_drifting: bool,
    pub delta_sparsity: f64,
    pub trend: String,
    pub suggested_action: String,
    pub window_size: usize,
}

/// Partition result from MinCut
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionResult {
    pub clusters: Vec<KnowledgeCluster>,
    pub cut_value: f64,
    pub edge_strengths: Vec<EdgeStrengthInfo>,
    pub total_memories: usize,
}

/// A knowledge cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeCluster {
    pub id: u32,
    pub memory_ids: Vec<Uuid>,
    pub centroid: Vec<f32>,
    pub dominant_category: BrainCategory,
    pub size: usize,
    pub coherence: f64,
}

/// Edge strength info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeStrengthInfo {
    pub source_cluster: u32,
    pub target_cluster: u32,
    pub strength: f64,
}

/// Vote direction
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteDirection {
    Up,
    Down,
}

/// API request/response types
#[derive(Debug, Deserialize)]
pub struct ShareRequest {
    pub category: BrainCategory,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub code_snippet: Option<String>,
    /// Client-provided embedding. If omitted, server generates via ruvllm.
    #[serde(default)]
    pub embedding: Vec<f32>,
    pub rvf_bytes: Option<String>, // base64-encoded RVF container
    /// Witness hash for integrity. If omitted, server computes from content.
    #[serde(default)]
    pub witness_hash: String,
    /// Optional challenge nonce for replay protection
    pub nonce: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub category: Option<BrainCategory>,
    pub tags: Option<String>,
    pub limit: Option<usize>,
    pub min_quality: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct VoteRequest {
    pub direction: VoteDirection,
}

#[derive(Debug, Deserialize)]
pub struct TransferRequest {
    pub source_domain: String,
    pub target_domain: String,
}

#[derive(Debug, Deserialize)]
pub struct DriftQuery {
    pub domain: Option<String>,
    pub since: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
pub struct PartitionQuery {
    pub domain: Option<String>,
    pub min_cluster_size: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub domain: String,
    pub uptime_seconds: u64,
    pub persistence_mode: String,
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub total_memories: usize,
    pub total_contributors: usize,
    pub graph_nodes: usize,
    pub graph_edges: usize,
    pub cluster_count: usize,
    pub avg_quality: f64,
    pub drift_status: String,
    pub lora_epoch: u64,
    pub lora_pending_submissions: usize,
    pub total_pages: usize,
    pub total_nodes: usize,
    pub total_votes: u64,
    pub embedding_engine: String,
    pub embedding_dim: usize,
    pub embedding_corpus: usize,
    pub dp_epsilon: f64,
    pub dp_budget_used: f64,
    pub rvf_segments_per_memory: f64,
    /// Global Workspace Theory attention load (0.0-1.0)
    pub gwt_workspace_load: f32,
    /// Global Workspace Theory average salience of current representations
    pub gwt_avg_salience: f32,
    /// Knowledge velocity: embedding deltas per hour (temporal tracking)
    pub knowledge_velocity: f64,
    /// Total temporal deltas recorded
    pub temporal_deltas: usize,
    pub sona_patterns: usize,
    pub sona_trajectories: usize,
    /// Meta-learning average regret (lower = better)
    pub meta_avg_regret: f64,
    /// Meta-learning plateau status
    pub meta_plateau_status: String,
    // ── Midstream Platform (ADR-077) ──
    /// Nanosecond scheduler total ticks
    pub midstream_scheduler_ticks: u64,
    /// Categories with Lyapunov attractor analysis
    pub midstream_attractor_categories: usize,
    /// Strange-loop engine version
    pub midstream_strange_loop_version: String,
}

/// Response for GET /v1/temporal — temporal delta tracking stats
#[derive(Debug, Serialize)]
pub struct TemporalResponse {
    pub total_deltas: usize,
    pub recent_hour_deltas: usize,
    pub knowledge_velocity: f64,
    pub trend: String,
}

#[derive(Debug, Serialize)]
pub struct ShareResponse {
    pub id: Uuid,
    pub partition_id: Option<u32>,
    pub quality_score: f64,
    pub witness_hash: String,
    pub rvf_segments: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct TransferResponse {
    pub source_domain: String,
    pub target_domain: String,
    pub acceleration_factor: f64,
    pub transfer_success: bool,
    pub message: String,
    pub source_memory_count: usize,
    pub target_memory_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// A brain memory with its search relevance score
#[derive(Debug, Clone, Serialize)]
pub struct ScoredBrainMemory {
    #[serde(flatten)]
    pub memory: BrainMemory,
    pub score: f64,
}

/// Query parameters for paginated list endpoint
#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub category: Option<BrainCategory>,
    pub tags: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub sort: Option<ListSort>,
}

/// Sort options for list endpoint
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListSort {
    UpdatedAt,
    Quality,
    Votes,
}

impl Default for ListSort {
    fn default() -> Self {
        Self::UpdatedAt
    }
}

/// Paginated list response envelope
#[derive(Debug, Serialize)]
pub struct ListResponse {
    pub memories: Vec<BrainMemory>,
    pub total_count: usize,
    pub offset: usize,
    pub limit: usize,
}

/// Challenge nonce for replay protection
#[derive(Debug, Serialize)]
pub struct ChallengeResponse {
    pub nonce: String,
    pub expires_at: DateTime<Utc>,
}

/// Request for POST /v1/verify — verify integrity of memories and witness chains
#[derive(Debug, Deserialize)]
pub struct VerifyRequest {
    /// Witness chain step labels for hash verification
    pub witness_steps: Option<Vec<String>>,
    /// Expected SHAKE-256 hash of the witness chain
    pub witness_hash: Option<String>,
    /// Memory ID for lookup-based verification
    pub memory_id: Option<uuid::Uuid>,
    /// Expected content hash (SHAKE-256 hex)
    pub content_hash: Option<String>,
    /// Raw content data to hash-verify (UTF-8 string)
    pub content_data: Option<String>,
    /// Base64-encoded binary witness chain bytes
    pub witness_chain_bytes: Option<String>,
}

/// Response for POST /v1/verify
#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub method: String,
    pub message: String,
}

/// LoRA weights submitted by a session for federation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraSubmission {
    pub down_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub rank: usize,
    pub hidden_dim: usize,
    pub evidence_count: u64,
}

impl LoraSubmission {
    /// Gate A: policy validity check
    pub fn validate(&self) -> Result<(), String> {
        let expected_down = self.hidden_dim * self.rank;
        let expected_up = self.rank * self.hidden_dim;
        if self.down_proj.len() != expected_down {
            return Err(format!("down_proj shape: expected {expected_down}, got {}", self.down_proj.len()));
        }
        if self.up_proj.len() != expected_up {
            return Err(format!("up_proj shape: expected {expected_up}, got {}", self.up_proj.len()));
        }
        for (i, &v) in self.down_proj.iter().chain(self.up_proj.iter()).enumerate() {
            if v.is_nan() || v.is_infinite() {
                return Err(format!("NaN/Inf at index {i}"));
            }
            if v.abs() > 2.0 {
                return Err(format!("Weight out of [-2, 2] at index {i}: {v}"));
            }
        }
        let down_norm: f32 = self.down_proj.iter().map(|x| x * x).sum::<f32>().sqrt();
        let up_norm: f32 = self.up_proj.iter().map(|x| x * x).sum::<f32>().sqrt();
        if down_norm > 100.0 || up_norm > 100.0 {
            return Err(format!("Norm too large: down={down_norm:.1}, up={up_norm:.1}"));
        }
        if self.evidence_count < 5 {
            return Err(format!("Insufficient evidence: {}", self.evidence_count));
        }
        Ok(())
    }
}

/// Consensus LoRA weights (aggregated from multiple sessions)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusLoraWeights {
    pub down_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub rank: usize,
    pub hidden_dim: usize,
    pub epoch: u64,
    pub contributor_count: usize,
    pub total_evidence: u64,
}

/// Response for GET /v1/lora/latest
#[derive(Debug, Serialize)]
pub struct LoraLatestResponse {
    pub weights: Option<ConsensusLoraWeights>,
    pub epoch: u64,
}

/// Response for POST /v1/lora/submit
#[derive(Debug, Serialize)]
pub struct LoraSubmitResponse {
    pub accepted: bool,
    pub pending_submissions: usize,
    pub current_epoch: u64,
}

// ──────────────────────────────────────────────────────────────────────
// Brainpedia types (ADR-062)
// ──────────────────────────────────────────────────────────────────────

/// Page lifecycle status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PageStatus {
    Draft,
    Canonical,
    Contested,
    Archived,
}

impl std::fmt::Display for PageStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Draft => write!(f, "draft"),
            Self::Canonical => write!(f, "canonical"),
            Self::Contested => write!(f, "contested"),
            Self::Archived => write!(f, "archived"),
        }
    }
}

/// Delta type for page modifications
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeltaType {
    Correction,
    Extension,
    Evidence,
    Deprecation,
}

/// Evidence linking a claim to a verifiable outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceLink {
    pub evidence_type: EvidenceType,
    pub description: String,
    pub contributor_id: String,
    pub verified: bool,
    pub created_at: DateTime<Utc>,
}

/// Types of verifiable evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum EvidenceType {
    TestPass { test_name: String, repo: String, commit_hash: String },
    BuildSuccess { pipeline_url: String, commit_hash: String },
    MetricImproval { metric_name: String, before: f64, after: f64 },
    PeerReview { reviewer: String, direction: VoteDirection, score: f64 },
}

/// A delta entry modifying a canonical page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageDelta {
    pub id: Uuid,
    pub page_id: Uuid,
    pub delta_type: DeltaType,
    pub content_diff: serde_json::Value,
    pub evidence_links: Vec<EvidenceLink>,
    pub contributor_id: String,
    pub quality_score: BetaParams,
    pub witness_hash: String,
    pub created_at: DateTime<Utc>,
}

/// Request to create a new Brainpedia page (Draft)
#[derive(Debug, Deserialize)]
pub struct CreatePageRequest {
    pub category: BrainCategory,
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub code_snippet: Option<String>,
    /// Client-provided embedding. If omitted, server generates via ruvllm.
    #[serde(default)]
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub evidence_links: Vec<EvidenceLink>,
    /// Witness hash for integrity. If omitted, server computes from content.
    #[serde(default)]
    pub witness_hash: String,
}

/// Request to submit a delta to a page
#[derive(Debug, Deserialize)]
pub struct SubmitDeltaRequest {
    pub delta_type: DeltaType,
    pub content_diff: serde_json::Value,
    pub evidence_links: Vec<EvidenceLink>,
    pub witness_hash: String,
}

/// Request to add evidence to a page
#[derive(Debug, Deserialize)]
pub struct AddEvidenceRequest {
    pub evidence: EvidenceLink,
}

/// Response for page creation
#[derive(Debug, Serialize)]
pub struct PageResponse {
    pub id: Uuid,
    pub status: PageStatus,
    pub quality_score: f64,
    pub evidence_count: u32,
    pub delta_count: u32,
}

/// Response for page get (includes delta log)
#[derive(Debug, Serialize)]
pub struct PageDetailResponse {
    pub memory: BrainMemory,
    pub status: PageStatus,
    pub evidence_count: u32,
    pub delta_count: u32,
    pub deltas: Vec<PageDelta>,
    pub evidence_links: Vec<EvidenceLink>,
}

/// Summary for page listing (lighter than PageDetailResponse)
#[derive(Debug, Serialize)]
pub struct PageSummary {
    pub id: Uuid,
    pub title: String,
    pub category: BrainCategory,
    pub status: PageStatus,
    pub quality_score: f64,
    pub delta_count: u32,
    pub evidence_count: u32,
    pub updated_at: DateTime<Utc>,
}

/// Response envelope for paginated page listing
#[derive(Debug, Serialize)]
pub struct ListPagesResponse {
    pub pages: Vec<PageSummary>,
    pub total_count: usize,
    pub offset: usize,
    pub limit: usize,
}

// ──────────────────────────────────────────────────────────────────────
// WASM Executable Nodes (ADR-063)
// ──────────────────────────────────────────────────────────────────────

/// A conformance test vector for deterministic verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceVector {
    pub input: String,
    /// SHA-256 of the raw output f32 bytes (little-endian)
    pub expected_output_sha256: String,
}

/// A published WASM node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmNode {
    pub id: String,
    pub name: String,
    pub version: String,
    pub abi_version: u8,
    pub dim: u16,
    pub sha256: String,
    pub size_bytes: usize,
    pub exports: Vec<String>,
    pub contributor_id: String,
    pub interface: serde_json::Value,
    pub conformance: Vec<ConformanceVector>,
    pub compiler_tag: String,
    pub revoked: bool,
    pub created_at: DateTime<Utc>,
}

/// Summary for GET /v1/nodes listing
#[derive(Debug, Serialize)]
pub struct WasmNodeSummary {
    pub id: String,
    pub name: String,
    pub version: String,
    pub abi_version: u8,
    pub dim: u16,
    pub sha256: String,
    pub size_bytes: usize,
    pub exports: Vec<String>,
    pub contributor_id: String,
    pub revoked: bool,
    pub created_at: DateTime<Utc>,
}

impl From<&WasmNode> for WasmNodeSummary {
    fn from(n: &WasmNode) -> Self {
        Self {
            id: n.id.clone(),
            name: n.name.clone(),
            version: n.version.clone(),
            abi_version: n.abi_version,
            dim: n.dim,
            sha256: n.sha256.clone(),
            size_bytes: n.size_bytes,
            exports: n.exports.clone(),
            contributor_id: n.contributor_id.clone(),
            revoked: n.revoked,
            created_at: n.created_at,
        }
    }
}

/// Request to publish a WASM node
#[derive(Debug, Deserialize)]
pub struct PublishNodeRequest {
    pub id: String,
    pub name: String,
    pub version: String,
    pub dim: Option<u16>,
    pub exports: Vec<String>,
    pub interface: serde_json::Value,
    pub conformance: Vec<ConformanceVector>,
    pub compiler_tag: Option<String>,
    /// Base64-encoded WASM binary
    pub wasm_bytes: String,
    /// Ed25519 signature over canonical binary manifest (hex)
    pub signature: String,
    /// Optional SHA-256 claim — server verifies against computed hash
    pub sha256: Option<String>,
}

/// Query for GET /v1/training/preferences
#[derive(Debug, Deserialize)]
pub struct TrainingQuery {
    pub since_index: Option<u64>,
    pub limit: Option<usize>,
}

/// Response for GET /v1/training/preferences
#[derive(Debug, Serialize)]
pub struct TrainingPreferencesResponse {
    pub pairs: Vec<crate::store::PreferencePair>,
    pub next_index: u64,
    pub total_votes: u64,
}

/// Result of an explicit or background training cycle.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingCycleResult {
    pub sona_message: String,
    pub sona_patterns: usize,
    pub pareto_before: usize,
    pub pareto_after: usize,
    pub memory_count: usize,
    pub vote_count: u64,
}

/// Federated LoRA store for accumulating submissions and producing consensus
pub struct LoraFederationStore {
    /// Pending submissions waiting for next aggregation round
    pub pending: Vec<(LoraSubmission, String, f64)>, // (weights, contributor, reputation)
    /// Current consensus weights
    pub consensus: Option<ConsensusLoraWeights>,
    /// Current epoch number
    pub epoch: u64,
    /// Previous consensus for rollback
    pub previous_consensus: Option<ConsensusLoraWeights>,
    /// Minimum submissions before aggregation
    pub min_submissions: usize,
    /// Expected rank for validation
    pub expected_rank: usize,
    /// Expected hidden dim for validation
    pub expected_hidden_dim: usize,
}

impl LoraFederationStore {
    pub fn new(rank: usize, hidden_dim: usize) -> Self {
        Self {
            pending: Vec::new(),
            consensus: None,
            epoch: 0,
            previous_consensus: None,
            min_submissions: 3,
            expected_rank: rank,
            expected_hidden_dim: hidden_dim,
        }
    }

    /// Submit weights from a session (after Gate A validation)
    pub fn submit(&mut self, submission: LoraSubmission, contributor: String, reputation: f64) {
        self.pending.push((submission, contributor, reputation));

        // Auto-aggregate when we have enough submissions
        if self.pending.len() >= self.min_submissions {
            self.aggregate();
        }
    }

    /// Run Gate B aggregation: per-parameter median + reputation-weighted trimmed mean
    pub fn aggregate(&mut self) {
        if self.pending.len() < self.min_submissions {
            return;
        }

        let dim = self.expected_hidden_dim * self.expected_rank;
        let total_params = dim * 2; // down + up

        // Per-parameter median for outlier detection
        let mut all_params: Vec<Vec<f32>> = vec![Vec::new(); total_params];
        let mut weights: Vec<f64> = Vec::new();
        // Track which pending submissions were accepted (for correct index mapping)
        let mut accepted_indices: Vec<usize> = Vec::new();

        for (idx, (sub, _, rep)) in self.pending.iter().enumerate() {
            let params: Vec<f32> = sub.down_proj.iter()
                .chain(sub.up_proj.iter())
                .copied()
                .collect();
            if params.len() != total_params {
                continue;
            }
            for (i, &v) in params.iter().enumerate() {
                all_params[i].push(v);
            }
            weights.push(*rep * sub.evidence_count as f64);
            accepted_indices.push(idx);
        }

        let n = weights.len();
        if n < self.min_submissions {
            return;
        }

        // Compute per-parameter median
        let medians: Vec<f32> = all_params.iter().map(|vals| {
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            }
        }).collect();

        // Compute MAD (Median Absolute Deviation) per parameter
        let mads: Vec<f32> = all_params.iter().zip(medians.iter()).map(|(vals, &med)| {
            let mut devs: Vec<f32> = vals.iter().map(|&v| (v - med).abs()).collect();
            devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if devs.len() % 2 == 0 {
                (devs[devs.len() / 2 - 1] + devs[devs.len() / 2]) / 2.0
            } else {
                devs[devs.len() / 2]
            }
        }).collect();

        // Reputation-weighted trimmed mean: exclude params >3*MAD from median
        let mut result = vec![0.0f32; total_params];
        let mut result_weight = vec![0.0f64; total_params];

        // Iterate only over accepted submissions with correct weight indexing
        for (weight_idx, &pending_idx) in accepted_indices.iter().enumerate() {
            let (sub, _, _) = &self.pending[pending_idx];
            let params: Vec<f32> = sub.down_proj.iter()
                .chain(sub.up_proj.iter())
                .copied()
                .collect();
            if params.len() != total_params {
                continue;
            }
            let w = weights[weight_idx];
            for (i, &v) in params.iter().enumerate() {
                let dev = (v - medians[i]).abs();
                let threshold = (mads[i] * 3.0).max(0.01);
                if dev <= threshold {
                    result[i] += v * w as f32;
                    result_weight[i] += w;
                }
            }
        }

        // Normalize
        for i in 0..total_params {
            if result_weight[i] > 1e-10 {
                result[i] /= result_weight[i] as f32;
            }
        }

        let total_evidence: u64 = self.pending.iter().map(|(s, _, _)| s.evidence_count).sum();

        // Save previous for rollback
        self.previous_consensus = self.consensus.take();

        let (down, up) = result.split_at(dim);
        self.consensus = Some(ConsensusLoraWeights {
            down_proj: down.to_vec(),
            up_proj: up.to_vec(),
            rank: self.expected_rank,
            hidden_dim: self.expected_hidden_dim,
            epoch: self.epoch + 1,
            contributor_count: n,
            total_evidence,
        });

        self.epoch += 1;
        self.pending.clear();
    }

    /// Rollback to previous consensus (if drift is too high)
    pub fn rollback(&mut self) -> bool {
        if let Some(prev) = self.previous_consensus.take() {
            self.consensus = Some(prev);
            true
        } else {
            false
        }
    }

    /// Compute L2 distance between current and previous consensus
    pub fn consensus_drift(&self) -> Option<f32> {
        let curr = self.consensus.as_ref()?;
        let prev = self.previous_consensus.as_ref()?;

        let d: f32 = curr.down_proj.iter().zip(prev.down_proj.iter())
            .chain(curr.up_proj.iter().zip(prev.up_proj.iter()))
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        Some(d.sqrt())
    }

    /// Save consensus weights to Firestore for persistence across restarts
    pub async fn save_to_store(&self, store: &crate::store::FirestoreClient) {
        if let Some(ref consensus) = self.consensus {
            let doc = serde_json::json!({
                "epoch": self.epoch,
                "consensus": consensus,
            });
            store.firestore_put_public("brain_lora", "consensus", &doc).await;
        }
    }

    /// Load consensus weights from Firestore on startup
    pub async fn load_from_store(&mut self, store: &crate::store::FirestoreClient) {
        let docs = store.firestore_list_public("brain_lora").await;
        for doc in docs {
            if let Some(epoch) = doc.get("epoch").and_then(|v| v.as_u64()) {
                self.epoch = epoch;
            }
            if let Some(consensus) = doc.get("consensus") {
                if let Ok(c) = serde_json::from_value::<ConsensusLoraWeights>(consensus.clone()) {
                    self.consensus = Some(c);
                }
            }
        }
        if self.epoch > 0 {
            tracing::info!("LoRA state loaded from Firestore: epoch {}", self.epoch);
        }
    }
}

impl Default for LoraFederationStore {
    fn default() -> Self {
        Self::new(2, 128) // Rank-2, 128-dim
    }
}

/// Nonce store for replay protection.
/// Tracks issued nonces with expiry to prevent replay attacks.
/// Nonces are optional on requests for backward compatibility.
pub struct NonceStore {
    /// Issued nonces: nonce → expiry time
    nonces: dashmap::DashMap<String, chrono::DateTime<chrono::Utc>>,
    /// Counter for periodic expired-nonce cleanup
    ops_counter: std::sync::atomic::AtomicU64,
}

impl NonceStore {
    pub fn new() -> Self {
        Self {
            nonces: dashmap::DashMap::new(),
            ops_counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Issue a new nonce with 5-minute TTL
    pub fn issue(&self) -> (String, chrono::DateTime<chrono::Utc>) {
        let nonce = uuid::Uuid::new_v4().to_string();
        let expires_at = chrono::Utc::now() + chrono::Duration::minutes(5);
        self.nonces.insert(nonce.clone(), expires_at);
        self.maybe_cleanup();
        (nonce, expires_at)
    }

    /// Consume a nonce: returns true if valid and not expired, removes it to prevent reuse.
    /// Returns false if nonce was never issued, already used, or expired.
    pub fn consume(&self, nonce: &str) -> bool {
        self.maybe_cleanup();
        if let Some((_, expires_at)) = self.nonces.remove(nonce) {
            expires_at > chrono::Utc::now()
        } else {
            false
        }
    }

    /// Periodic cleanup of expired nonces
    fn maybe_cleanup(&self) {
        let count = self.ops_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count % 500 != 0 {
            return;
        }
        let now = chrono::Utc::now();
        self.nonces.retain(|_, expires_at| *expires_at > now);
    }
}

/// RVF feature flags, read once at startup to avoid per-request env::var calls.
#[derive(Debug, Clone)]
pub struct RvfFeatureFlags {
    pub pii_strip: bool,
    pub dp_enabled: bool,
    pub dp_epsilon: f64,
    pub witness: bool,
    pub container: bool,
    pub adversarial: bool,
    pub neg_cache: bool,
    pub sona_enabled: bool,
    /// Global Workspace Theory attention layer for search ranking (ADR-075 AGI)
    pub gwt_enabled: bool,
    /// Temporal delta tracking for knowledge evolution (ADR-075 AGI)
    pub temporal_enabled: bool,
    /// Meta-learning exploration via domain expansion engine (ADR-075 AGI)
    pub meta_learning_enabled: bool,
    // ── Midstream Platform (ADR-077) ──
    /// Nanosecond-scheduler background task management
    pub midstream_scheduler: bool,
    /// Temporal-attractor-studio Lyapunov exponent analysis
    pub midstream_attractor: bool,
    /// Temporal-neural-solver certified prediction
    pub midstream_solver: bool,
    /// Strange-loop recursive meta-cognition
    pub midstream_strange_loop: bool,
}

impl RvfFeatureFlags {
    /// Read all RVF_* env vars once and cache the results.
    pub fn from_env() -> Self {
        Self {
            pii_strip: std::env::var("RVF_PII_STRIP")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            dp_enabled: std::env::var("RVF_DP_ENABLED")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            dp_epsilon: std::env::var("RVF_DP_EPSILON")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            witness: std::env::var("RVF_WITNESS")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            container: std::env::var("RVF_CONTAINER")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            adversarial: std::env::var("RVF_ADVERSARIAL")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            neg_cache: std::env::var("RVF_NEG_CACHE")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            sona_enabled: std::env::var("SONA_ENABLED")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            gwt_enabled: std::env::var("GWT_ENABLED")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            temporal_enabled: std::env::var("TEMPORAL_ENABLED")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            meta_learning_enabled: std::env::var("META_LEARNING_ENABLED")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            // Midstream flags default to false (opt-in per ADR-077)
            midstream_scheduler: std::env::var("MIDSTREAM_SCHEDULER")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            midstream_attractor: std::env::var("MIDSTREAM_ATTRACTOR")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            midstream_solver: std::env::var("MIDSTREAM_SOLVER")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            midstream_strange_loop: std::env::var("MIDSTREAM_STRANGE_LOOP")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
        }
    }
}

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub store: std::sync::Arc<crate::store::FirestoreClient>,
    pub gcs: std::sync::Arc<crate::gcs::GcsClient>,
    pub graph: std::sync::Arc<parking_lot::RwLock<crate::graph::KnowledgeGraph>>,
    pub rate_limiter: std::sync::Arc<crate::rate_limit::RateLimiter>,
    pub ranking: std::sync::Arc<parking_lot::RwLock<crate::ranking::RankingEngine>>,
    pub cognitive: std::sync::Arc<parking_lot::RwLock<crate::cognitive::CognitiveEngine>>,
    pub drift: std::sync::Arc<parking_lot::RwLock<crate::drift::DriftMonitor>>,
    pub aggregator: std::sync::Arc<crate::aggregate::ByzantineAggregator>,
    pub domain_engine: std::sync::Arc<parking_lot::RwLock<ruvector_domain_expansion::DomainExpansionEngine>>,
    pub sona: std::sync::Arc<parking_lot::RwLock<sona::SonaEngine>>,
    pub lora_federation: std::sync::Arc<parking_lot::RwLock<LoraFederationStore>>,
    /// RuvLLM embedding engine (HashEmbedder + RlmEmbedder)
    pub embedding_engine: std::sync::Arc<parking_lot::RwLock<crate::embeddings::EmbeddingEngine>>,
    /// Nonce store for replay protection on write endpoints
    pub nonce_store: std::sync::Arc<NonceStore>,
    /// Differential privacy engine for embedding noise injection (ADR-075 Phase 3)
    pub dp_engine: std::sync::Arc<parking_lot::Mutex<rvf_federation::DiffPrivacyEngine>>,
    /// Negative cache for degenerate query signatures (ADR-075 Phase 6)
    pub negative_cache: std::sync::Arc<parking_lot::Mutex<rvf_runtime::NegativeCache>>,
    /// RVF feature flags read once at startup (avoids per-request env::var calls)
    pub rvf_flags: RvfFeatureFlags,
    /// Global Workspace Theory attention layer for memory selection (ADR-075 AGI)
    pub workspace: std::sync::Arc<parking_lot::RwLock<ruvector_nervous_system::routing::workspace::GlobalWorkspace>>,
    /// Temporal delta tracking for knowledge evolution (ADR-075 AGI)
    pub delta_stream: std::sync::Arc<parking_lot::RwLock<ruvector_delta_core::DeltaStream<ruvector_delta_core::VectorDelta>>>,
    /// Cached verifier (holds compiled PiiStripper regexes — avoids recompiling per request)
    pub verifier: std::sync::Arc<parking_lot::RwLock<crate::verify::Verifier>>,
    /// Negative cost fuse: when true, reject all writes (Firestore/GCS errors spiking)
    pub read_only: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub start_time: std::time::Instant,
    // ── Midstream Platform (ADR-077) ──
    /// Nanosecond-precision background scheduler (Phase 9b)
    pub nano_scheduler: std::sync::Arc<nanosecond_scheduler::Scheduler>,
    /// Per-category Lyapunov exponent results from attractor analysis (Phase 9c)
    pub attractor_results: std::sync::Arc<parking_lot::RwLock<std::collections::HashMap<String, temporal_attractor_studio::LyapunovResult>>>,
    /// Temporal neural solver with certified predictions (Phase 9d)
    pub temporal_solver: std::sync::Arc<parking_lot::RwLock<temporal_neural_solver::TemporalSolver>>,
    /// Meta-cognitive recursive learning with safety bounds (Phase 9e)
    pub strange_loop: std::sync::Arc<parking_lot::RwLock<strange_loop::StrangeLoop<strange_loop::ScalarReasoner, strange_loop::SimpleCritic, strange_loop::SafeReflector>>>,
    /// Active SSE sessions: session ID -> sender channel for streaming responses
    pub sessions: std::sync::Arc<dashmap::DashMap<String, tokio::sync::mpsc::Sender<String>>>,
}
