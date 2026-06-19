//! OpenMythos configuration and the honest-boundary metadata gate.

use std::collections::BTreeMap;

use crate::error::{Result, RuvLLMError};

/// Attention variant used by every transformer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttnType {
    /// Grouped-Query Attention (`n_kv_heads < n_heads`).
    Gqa,
    /// Multi-Latent Attention (DeepSeek-V2 style, compressed KV cache).
    Mla,
}

/// Configuration for an OpenMythos model. Defaults mirror the reference
/// `MythosConfig`; [`MythosConfig::tiny`] is a scaled-down test config.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MythosConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub max_seq_len: usize,
    /// Recurrent loop ceiling (depth) used at training time.
    pub max_loop_iters: usize,
    pub prelude_layers: usize,
    pub coda_layers: usize,

    // --- Attention ---
    pub attn_type: AttnType,
    /// MLA: rank of the compressed KV latent.
    pub kv_lora_rank: usize,
    /// MLA: rank of the compressed query latent (0 = no query compression).
    pub q_lora_rank: usize,
    /// MLA: RoPE-carrying head dimension.
    pub qk_rope_head_dim: usize,
    /// MLA: non-RoPE head dimension.
    pub qk_nope_head_dim: usize,
    /// MLA: value head dimension.
    pub v_head_dim: usize,

    // --- MoE FFN (recurrent block) ---
    pub expert_dim: usize,
    pub n_experts: usize,
    pub n_shared_experts: usize,
    pub n_experts_per_tok: usize,
    /// Use MoE FFN in the recurrent block (prelude/coda always use dense FFN).
    pub use_moe: bool,

    // --- Recurrent stabilization ---
    pub act_threshold: f32,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    /// Channels covered by the loop-index positional embedding (even, <= dim).
    pub loop_dim: usize,
    /// Rank of the per-depth LoRA adapter applied in the recurrent block.
    pub lora_rank: usize,
}

impl Default for MythosConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            dim: 2048,
            n_heads: 16,
            n_kv_heads: 4,
            max_seq_len: 4096,
            max_loop_iters: 16,
            prelude_layers: 2,
            coda_layers: 2,
            attn_type: AttnType::Gqa,
            kv_lora_rank: 512,
            q_lora_rank: 1536,
            qk_rope_head_dim: 64,
            qk_nope_head_dim: 128,
            v_head_dim: 128,
            expert_dim: 512,
            n_experts: 64,
            n_shared_experts: 2,
            n_experts_per_tok: 4,
            use_moe: true,
            act_threshold: 0.99,
            rope_theta: 500_000.0,
            rms_norm_eps: 1e-5,
            loop_dim: 64,
            lora_rank: 16,
        }
    }
}

impl MythosConfig {
    /// A tiny config for tests / smoke runs (GQA).
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            dim: 32,
            n_heads: 4,
            n_kv_heads: 2,
            max_seq_len: 64,
            max_loop_iters: 6,
            prelude_layers: 1,
            coda_layers: 1,
            attn_type: AttnType::Gqa,
            kv_lora_rank: 16,
            q_lora_rank: 0,
            qk_rope_head_dim: 4,
            qk_nope_head_dim: 4,
            v_head_dim: 8,
            expert_dim: 24,
            n_experts: 4,
            n_shared_experts: 1,
            n_experts_per_tok: 2,
            use_moe: true,
            act_threshold: 0.99,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            loop_dim: 8,
            lora_rank: 4,
        }
    }

    /// A tiny config exercising the MLA attention path.
    pub fn tiny_mla() -> Self {
        Self {
            attn_type: AttnType::Mla,
            q_lora_rank: 16,
            ..Self::tiny()
        }
    }

    /// Head dimension for GQA (`dim / n_heads`).
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Combined per-head query/key dimension for MLA.
    pub fn mla_qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Validate structural invariants. Weight-sharing compatibility of a source
    /// checkpoint is enforced separately by [`validate_mythos_metadata`].
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.n_heads == 0 || self.dim % self.n_heads != 0 {
            return Err(RuvLLMError::Config(
                "OpenMythos: dim must be a non-zero multiple of n_heads".into(),
            ));
        }
        match self.attn_type {
            AttnType::Gqa => {
                if self.n_kv_heads == 0 || self.n_heads % self.n_kv_heads != 0 {
                    return Err(RuvLLMError::Config(
                        "OpenMythos: n_heads must be a multiple of n_kv_heads".into(),
                    ));
                }
            }
            AttnType::Mla => {
                if self.kv_lora_rank == 0 || self.qk_rope_head_dim == 0 || self.v_head_dim == 0 {
                    return Err(RuvLLMError::Config(
                        "OpenMythos(MLA): kv_lora_rank, qk_rope_head_dim, v_head_dim must be > 0"
                            .into(),
                    ));
                }
                if self.qk_rope_head_dim % 2 != 0 {
                    return Err(RuvLLMError::Config(
                        "OpenMythos(MLA): qk_rope_head_dim must be even".into(),
                    ));
                }
            }
        }
        if self.max_loop_iters == 0 {
            return Err(RuvLLMError::Config(
                "OpenMythos: max_loop_iters must be >= 1".into(),
            ));
        }
        if self.use_moe && self.n_experts_per_tok > self.n_experts {
            return Err(RuvLLMError::Config(
                "OpenMythos: n_experts_per_tok must be <= n_experts".into(),
            ));
        }
        if self.loop_dim > self.dim || self.loop_dim % 2 != 0 {
            return Err(RuvLLMError::Config(
                "OpenMythos: loop_dim must be even and <= dim".into(),
            ));
        }
        if !(self.act_threshold > 0.0 && self.act_threshold <= 1.0) {
            return Err(RuvLLMError::Config(
                "OpenMythos: act_threshold must be in (0, 1]".into(),
            ));
        }
        Ok(())
    }

    /// Build a config from GGUF-style metadata, falling back to defaults for any
    /// missing keys. Recognized keys use the `mythos.*` namespace.
    pub fn from_metadata(meta: &BTreeMap<String, String>) -> Self {
        let mut c = Self::default();
        let u = |k: &str| meta.get(k).and_then(|v| v.trim().parse::<usize>().ok());
        let f = |k: &str| meta.get(k).and_then(|v| v.trim().parse::<f32>().ok());
        if let Some(v) = u("mythos.vocab_size") {
            c.vocab_size = v;
        }
        if let Some(v) = u("mythos.dim") {
            c.dim = v;
        }
        if let Some(v) = u("mythos.n_heads") {
            c.n_heads = v;
        }
        if let Some(v) = u("mythos.n_kv_heads") {
            c.n_kv_heads = v;
        }
        if let Some(v) = u("mythos.max_loop_iters") {
            c.max_loop_iters = v;
        }
        if let Some(v) = u("mythos.prelude_layers") {
            c.prelude_layers = v;
        }
        if let Some(v) = u("mythos.coda_layers") {
            c.coda_layers = v;
        }
        if let Some(v) = u("mythos.n_experts") {
            c.n_experts = v;
        }
        if let Some(v) = u("mythos.n_experts_per_tok") {
            c.n_experts_per_tok = v;
        }
        if let Some(v) = f("mythos.act_threshold") {
            c.act_threshold = v;
        }
        if let Some(v) = f("mythos.rope_theta") {
            c.rope_theta = v;
        }
        if let Some(arch) = meta.get("general.architecture") {
            if arch.to_lowercase().contains("mla") {
                c.attn_type = AttnType::Mla;
            }
        }
        c
    }
}

/// Error when a checkpoint is not a weight-sharing / recurrent-depth model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MythosCompatibilityError {
    pub detected_architecture: String,
    pub reason: String,
}

impl std::fmt::Display for MythosCompatibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "non-recurrent-depth checkpoint rejected (architecture='{}'): {}. \
             OpenMythos requires weights natively trained for recurrent-depth \
             weight-sharing; standard weights produce garbage tokens.",
            self.detected_architecture, self.reason
        )
    }
}

impl std::error::Error for MythosCompatibilityError {}

impl From<MythosCompatibilityError> for RuvLLMError {
    fn from(e: MythosCompatibilityError) -> Self {
        RuvLLMError::Model(e.to_string())
    }
}

/// `general.architecture` values recognized as OpenMythos-compatible.
pub const MYTHOS_ARCHITECTURES: &[&str] = &["openmythos", "mythos", "rdt", "recurrent_depth"];

/// Metadata flags that mark a checkpoint as recurrent-depth.
pub const MYTHOS_RECURRENCE_KEYS: &[&str] = &[
    "mythos.recurrent",
    "rdt.recurrent",
    "recurrent_depth.enabled",
];

/// Validate that metadata describes an OpenMythos-compatible checkpoint.
///
/// Accepts iff `general.architecture` is one of [`MYTHOS_ARCHITECTURES`] (the
/// `mythos`/`openmythos` value may carry an `-mla` suffix), or a recurrence flag
/// in [`MYTHOS_RECURRENCE_KEYS`] is truthy. This is the OpenMythos honest
/// boundary — see [`crate::models::rdt::validate_rdt_metadata`].
pub fn validate_mythos_metadata(
    meta: &BTreeMap<String, String>,
) -> std::result::Result<(), MythosCompatibilityError> {
    let arch = meta
        .get("general.architecture")
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_default();

    let arch_base = arch.split(['-', '.']).next().unwrap_or("");
    if MYTHOS_ARCHITECTURES.contains(&arch.as_str()) || MYTHOS_ARCHITECTURES.contains(&arch_base) {
        return Ok(());
    }

    for key in MYTHOS_RECURRENCE_KEYS {
        if let Some(raw) = meta.get(*key) {
            if matches!(
                raw.trim().to_lowercase().as_str(),
                "true" | "1" | "yes" | "on"
            ) {
                return Ok(());
            }
        }
    }

    Err(MythosCompatibilityError {
        detected_architecture: if arch.is_empty() {
            "<unknown>".into()
        } else {
            arch
        },
        reason: "architecture is not recurrent-depth and no recurrence flag was set".into(),
    })
}
