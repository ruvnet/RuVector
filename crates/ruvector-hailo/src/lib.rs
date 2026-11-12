//! ruvector embedding backend for the Hailo-8 NPU.
//!
//! ADR-167 (`hailo-backend` branch). Implements
//! `ruvector_core::embeddings::EmbeddingProvider` (iter 218 closed
//! ADR-178 Gap B by landing the path dep + impl block).
//!
//! Default build (no `hailo` feature): every API call returns
//! `Err(HailoError::FeatureDisabled)`. Lets non-Pi machines run
//! `cargo check -p ruvector-hailo` without HailoRT installed.

pub mod device;
pub mod error;
pub mod hef_verify;
pub mod inference;
pub mod tokenizer;

#[cfg(feature = "cpu-fallback")]
pub mod cpu_embedder;

#[cfg(feature = "cpu-fallback")]
pub mod host_embeddings;

#[cfg(feature = "hailo")]
pub mod hef_pipeline;

#[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
pub mod hef_embedder;

#[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
pub mod hef_embedder_pool;

pub use device::HailoDevice;
pub use error::HailoError;
pub use inference::{l2_normalize, mean_pool, EmbeddingPipeline, DEFAULT_MAX_SEQ, MINI_LM_DIM};
pub use tokenizer::{EncodedInput, SpecialIds, WordPieceTokenizer};

#[cfg(feature = "cpu-fallback")]
pub use cpu_embedder::CpuEmbedder;

use ruvector_core::EmbeddingSpaceIdentity;
use sha2::{Digest, Sha256};
use std::path::Path;
#[cfg(feature = "hailo")]
use std::sync::Mutex;

/// Convenience alias matching ruvector-core's `Result<T> = Result<T, Error>`.
pub type Result<T> = std::result::Result<T, HailoError>;

/// Embedding inference engine backed by the Hailo-8 NPU.
///
/// Uses interior mutability so the public API is `&self` — that lets
/// `HailoEmbedder` implement `ruvector_core::embeddings::EmbeddingProvider`
/// (which takes `&self`) without forcing every caller to manage a `&mut`.
///
/// Phase 1 step 1 (this iteration): scaffold + signature parity. Open
/// returns `FeatureDisabled` until iteration 4 brings device enumeration
/// online.
pub struct HailoEmbedder {
    /// Embedding dimensionality from the loaded HEF. Set when an HEF is
    /// loaded; 0 in stub.
    dimensions: usize,
    /// Human-readable name for logging — e.g. `"hailo:all-MiniLM-L6-v2"`.
    name: String,
    identity: EmbeddingSpaceIdentity,
    /// PCIe BDF of the underlying device once opened, e.g. `0001:01:00.0`.
    device_id: String,
    /// Held-open vdevice handle. Iter-95: kept across the embedder's
    /// lifetime so `chip_temperature()` can read the on-die NPU
    /// thermal sensors without re-opening (which is expensive — each
    /// `hailo_create_vdevice` does a firmware handshake).
    /// Wrapped in `Mutex` so concurrent reads serialize safely; the
    /// libhailort vdevice is documented thread-safe for inference but
    /// thermal reads + future config writes still want serial access.
    /// Iter 137 — gated on `feature = "hailo"` AND wrapped in Option
    /// so the cpu-fallback path can ship on hosts that *built* the
    /// hailo feature in but happen to lack a HAT at runtime.
    #[cfg(feature = "hailo")]
    device: Option<Mutex<crate::device::HailoDevice>>,
    /// Iter 133 — Path C CPU fallback. `Some(_)` when the operator
    /// has model.safetensors + tokenizer.json + config.json in the
    /// model dir but no HEF (yet). When set, `embed()` dispatches to
    /// real BERT-6 inference on the host CPU via candle. NPU stays
    /// idle — fallback only. Only present when built with
    /// `--features cpu-fallback`.
    #[cfg(feature = "cpu-fallback")]
    cpu_fallback: Option<crate::cpu_embedder::CpuEmbedder>,
    /// Iter 162 (ADR-176 P4) — NPU acceleration via the iter-156b HEF
    /// plus iter-160 host-side embeddings plus iter-161 end-to-end
    /// pipeline. `Some(_)` when both `model.hef` and the safetensors
    /// trio are present in `model_dir`. Takes precedence over
    /// `cpu_fallback` in `embed()` dispatch.
    ///
    /// Iter 235 — wraps a `HefBackend` enum so the dispatch can pick
    /// between a single-pipeline `HefEmbedder` and a multi-pipeline
    /// `HefEmbedderPool` based on `RUVECTOR_NPU_POOL_SIZE`. Default
    /// (env unset or = 1) keeps the iter-162 single-pipeline path.
    #[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
    hef_embedder: Option<HefBackend>,
}

/// Iter 235 — switch between single-pipeline and pool-of-pipelines
/// NPU dispatch. Exposed as a private enum because the choice is
/// driven by `RUVECTOR_NPU_POOL_SIZE` at `HailoEmbedder::open` time;
/// callers see a uniform `embed()` regardless.
#[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
enum HefBackend {
    /// Single Mutex<HefPipeline>. Cheaper init + RAM, hard ~70 RPS
    /// ceiling per cognitum-v0 iter-227 baseline.
    Single(crate::hef_embedder::HefEmbedder),
    /// N independent pipelines on the same vdevice; HailoRT's
    /// network-group scheduler arbitrates NPU access. Targets the
    /// PCIe-DMA-overlap throughput unlock (iter 234 design doc).
    Pool(crate::hef_embedder_pool::HefEmbedderPool),
}

#[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
impl HefBackend {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Single(s) => s.embed(text),
            Self::Pool(p) => p.embed(text),
        }
    }
    fn output_dim(&self) -> usize {
        match self {
            Self::Single(s) => s.output_dim(),
            Self::Pool(p) => p.output_dim(),
        }
    }
}

/// Artifact filenames this backend knows how to fingerprint, in preference
/// order. Provenance verification is only meaningful for these layouts, so an
/// `open_with_identity` against a directory holding none of them fails closed.
///
/// # Layouts and what the two model hashes mean
/// `EmbeddingSpaceIdentity` has separate `model_artifact_sha256` (the weights
/// the operator deployed) and `model_graph_sha256` (the executable graph those
/// weights run as). This backend serves three layouts, and the two hashes are
/// only distinct in the first:
///
/// | layout | artifact | graph |
/// |---|---|---|
/// | NPU + CPU fallback (`model.safetensors` + `model.hef`) | safetensors | hef |
/// | CPU fallback only (`model.safetensors`) | safetensors | safetensors |
/// | NPU only (`model.hef`) | hef | hef |
///
/// In the single-file layouts the two hashes are deliberately equal: there is
/// no separately-compiled graph to fingerprint, and claiming a different value
/// would be inventing provenance. Anyone producing an identity to hand to
/// [`HailoEmbedder::open_with_identity`] should build the hash fields with
/// [`HailoEmbedder::artifact_digests`] rather than guessing, which is why that
/// helper is public.
const MODEL_ARTIFACT_FILES: &[&str] = &["model.safetensors", "model.hef"];
const MODEL_GRAPH_FILES: &[&str] = &["model.hef", "model.safetensors"];
const TOKENIZER_FILES: &[&str] = &["tokenizer.json", "vocab.txt"];

/// The model this backend's HEF and cpu-fallback weights are built for
/// (ADR-167). Used to look up the registry fixture for the default identity.
const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// SHA-256 digests of the artifacts found in a model directory.
///
/// `None` for a class the directory does not provide. Produced by
/// [`HailoEmbedder::artifact_digests`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactDigests {
    pub model_artifact: Option<String>,
    pub model_graph: Option<String>,
    pub tokenizer: Option<String>,
}

impl HailoEmbedder {
    /// Open a Hailo NPU device and load the HEF + tokenizer artifacts found
    /// at `model_dir`.
    ///
    /// Expected layout under `model_dir`:
    ///
    /// ```text
    /// model_dir/
    ///   model.hef             # compiled by Hailo Dataflow Compiler
    ///   vocab.txt             # WordPiece vocab (one token per line)
    ///   special_tokens.json   # CLS/SEP/PAD ids
    /// ```
    ///
    /// # Embedding-space identity
    /// The caller supplies none, so this path derives a default one (ADR-281
    /// §2): the registry fixture for `all-MiniLM-L6-v2` when the loaded model
    /// has that dimensionality — what the ADR-167 HEF and the cpu-fallback
    /// safetensors are compiled for — and otherwise a provider-scoped
    /// symmetric identity naming this model directory. Both are `Symmetric` /
    /// `PrefixPolicy::None`, which is what this backend actually does: it
    /// applies no prompt template on either side. The artifact hash fields are
    /// digests of whatever known artifacts the directory holds, so the
    /// identity still changes when the model bytes change. Callers that need
    /// an externally attested identity use
    /// [`open_with_identity`](Self::open_with_identity), which verifies the
    /// supplied hashes against the files on disk and fails closed.
    pub fn open(model_dir: &Path) -> Result<Self> {
        Self::open_inner(model_dir, None)
    }

    /// Open only when the caller supplies complete identity and the artifact
    /// hashes agree with the files selected by the backend.
    pub fn open_with_identity(model_dir: &Path, identity: EmbeddingSpaceIdentity) -> Result<Self> {
        identity
            .validate()
            .map_err(|e| HailoError::EmbeddingProvenance(e.to_string()))?;
        Self::verify_artifact_provenance(model_dir, &identity)?;
        Self::open_inner(model_dir, Some(identity))
    }

    /// Paths of the known artifacts present under `model_dir`, in
    /// `(model artifact, model graph, tokenizer)` order.
    fn artifact_paths(
        model_dir: &Path,
    ) -> (
        Option<std::path::PathBuf>,
        Option<std::path::PathBuf>,
        Option<std::path::PathBuf>,
    ) {
        let first_existing = |names: &[&str]| -> Option<std::path::PathBuf> {
            names
                .iter()
                .map(|name| model_dir.join(name))
                .find(|path| path.exists())
        };
        (
            first_existing(MODEL_ARTIFACT_FILES),
            first_existing(MODEL_GRAPH_FILES),
            first_existing(TOKENIZER_FILES),
        )
    }

    fn digest_file(model_dir: &Path, path: &Path) -> Result<String> {
        let bytes = std::fs::read(path).map_err(|_| HailoError::BadModelDir {
            path: model_dir.display().to_string(),
            what: "artifact required by EmbeddingSpaceIdentity",
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// SHA-256 digests of the artifacts in `model_dir`, ready to drop into the
    /// three hash fields of an [`EmbeddingSpaceIdentity`].
    ///
    /// This is the supported way to build an identity that
    /// [`open_with_identity`](Self::open_with_identity) will accept: it applies
    /// the same file-preference rules the verifier uses, including the
    /// single-file layouts where the artifact and graph digests are equal (see
    /// [`MODEL_ARTIFACT_FILES`]). Errors when the directory holds none of the
    /// artifacts this backend knows how to fingerprint, since there is nothing
    /// to attest.
    pub fn artifact_digests(model_dir: &Path) -> Result<ArtifactDigests> {
        let (model_artifact, model_graph, tokenizer) = Self::artifact_paths(model_dir);
        if model_artifact.is_none() && model_graph.is_none() && tokenizer.is_none() {
            return Err(HailoError::EmbeddingProvenance(format!(
                "cannot fingerprint `{}`: none of the known artifacts were found (model \
                 artifact: {}; model graph: {}; tokenizer: {})",
                model_dir.display(),
                MODEL_ARTIFACT_FILES.join(", "),
                MODEL_GRAPH_FILES.join(", "),
                TOKENIZER_FILES.join(", "),
            )));
        }
        Ok(ArtifactDigests {
            model_artifact: model_artifact
                .map(|p| Self::digest_file(model_dir, &p))
                .transpose()?,
            model_graph: model_graph
                .map(|p| Self::digest_file(model_dir, &p))
                .transpose()?,
            tokenizer: tokenizer
                .map(|p| Self::digest_file(model_dir, &p))
                .transpose()?,
        })
    }

    /// Check every hash a caller-supplied identity declares against the bytes
    /// on disk.
    ///
    /// Fails closed when the directory holds none of the artifact layouts this
    /// backend knows how to fingerprint: an unrecognized layout must not mean
    /// "provenance verified" just because no file matched a known name. There
    /// is no opt-out — callers that cannot supply artifact hashes use
    /// [`open`](Self::open), whose default identity is documented as
    /// unattested.
    fn verify_artifact_provenance(
        model_dir: &Path,
        identity: &EmbeddingSpaceIdentity,
    ) -> Result<()> {
        let digests = Self::artifact_digests(model_dir)?;
        for (label, found, expected) in [
            (
                "model artifact",
                digests.model_artifact.as_deref(),
                &identity.model_artifact_sha256,
            ),
            (
                "model graph",
                digests.model_graph.as_deref(),
                &identity.model_graph_sha256,
            ),
            (
                "tokenizer",
                digests.tokenizer.as_deref(),
                &identity.tokenizer_sha256,
            ),
        ] {
            // A missing individual file is still a verification gap: the
            // identity declares a hash for it, so refuse rather than skip.
            let Some(found) = found else {
                return Err(HailoError::EmbeddingProvenance(format!(
                    "identity declares a {label} hash but `{}` holds no {label} file",
                    model_dir.display()
                )));
            };
            if found != expected {
                return Err(HailoError::EmbeddingProvenance(format!(
                    "{label} SHA-256 mismatch in `{}`: on-disk {found}, identity declares {expected}",
                    model_dir.display()
                )));
            }
        }
        Ok(())
    }

    /// `identity == None` derives the documented default identity; `Some(_)`
    /// verifies caller-supplied provenance against the artifacts on disk.
    fn open_inner(model_dir: &Path, identity: Option<EmbeddingSpaceIdentity>) -> Result<Self> {
        // Iter 137: combinatorial feature gating. Build matrix:
        //   * neither feature      → FeatureDisabled (default x86 dev)
        //   * hailo only           → device-only (HAT host, no Python deps)
        //   * cpu-fallback only    → CPU-only (dev box, no HailoRT installed)
        //   * hailo + cpu-fallback → device + CPU fallback (production Pi)
        // Default no-features build: short-circuit. Returning here also
        // makes the constructor below dead code, so we provide stub
        // values for `device_id` etc. so the cfg lattice still compiles.
        #[cfg(all(not(feature = "hailo"), not(feature = "cpu-fallback")))]
        {
            let _ = (model_dir, &identity);
            return Err(HailoError::FeatureDisabled);
        }
        #[cfg(all(not(feature = "hailo"), not(feature = "cpu-fallback")))]
        #[allow(unreachable_code)]
        let device_id = String::new();

        // Try to open the Hailo device when the feature is on. If the
        // host has no HAT we still want CPU fallback to succeed — only
        // surface the device error if we can't fall back.
        #[cfg(feature = "hailo")]
        let (device_opt, device_id) = match crate::device::HailoDevice::open() {
            Ok(device) => {
                let v = device.version().unwrap_or((0, 0, 0));
                let device_id = format!("hailort:{}.{}.{}", v.0, v.1, v.2);
                (Some(device), device_id)
            }
            #[cfg(feature = "cpu-fallback")]
            Err(_) => (None, "cpu-fallback:no-device".to_string()),
            #[cfg(not(feature = "cpu-fallback"))]
            Err(e) => return Err(e),
        };

        #[cfg(all(not(feature = "hailo"), feature = "cpu-fallback"))]
        let device_id = "cpu-fallback:no-hailo-feature".to_string();

        // Iter 162 (ADR-176 P4) — open priority:
        //   1. HEF + safetensors trio (NPU acceleration)
        //   2. safetensors trio only (cpu-fallback)
        //   3. neither (NoModelLoaded — health probe still serves)
        //
        // HEF requires both `hailo` (for HefPipeline) and `cpu-fallback`
        // (for HostEmbeddings + tokenizer). When the feature lattice
        // doesn't enable both, we fall straight through to cpu-fallback
        // (or no model).
        // Both paths are only consulted under `feature = "cpu-fallback"`
        // (HEF requires it for HostEmbeddings, cpu-fallback obviously);
        // gate to silence unused-var warnings on `--features hailo` alone.
        #[cfg(feature = "cpu-fallback")]
        let hef_path = model_dir.join("model.hef");
        #[cfg(feature = "cpu-fallback")]
        let safetensors = model_dir.join("model.safetensors");

        #[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
        let hef_embedder: Option<HefBackend> = {
            if hef_path.exists() && safetensors.exists() {
                if let Some(dev) = device_opt.as_ref() {
                    // Iter 235 — pick single vs pool based on env var.
                    // Unset / = 1 → Single (preserves iter-162 default).
                    // >= 2     → Pool with N pipelines on the shared vdevice.
                    // Bad value (non-numeric) → log + fall back to Single
                    // rather than fail boot — the worker stays alive on the
                    // single-pipeline path and operators get a recovery
                    // window without a forced restart.
                    let pool_size = std::env::var("RUVECTOR_NPU_POOL_SIZE")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(1);
                    if pool_size >= 2 {
                        Some(HefBackend::Pool(
                            crate::hef_embedder_pool::HefEmbedderPool::open(
                                dev, model_dir, pool_size,
                            )?,
                        ))
                    } else {
                        Some(HefBackend::Single(crate::hef_embedder::HefEmbedder::open(
                            dev, model_dir,
                        )?))
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        // cpu-fallback: load only if HEF wasn't loaded (avoid duplicate
        // 90 MB safetensors mmap when both could load).
        #[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
        let cpu_fallback = if hef_embedder.is_some() {
            None
        } else if !hef_path.exists() && safetensors.exists() {
            Some(crate::cpu_embedder::CpuEmbedder::open(model_dir)?)
        } else {
            None
        };

        #[cfg(all(not(feature = "hailo"), feature = "cpu-fallback"))]
        let cpu_fallback = if !hef_path.exists() && safetensors.exists() {
            Some(crate::cpu_embedder::CpuEmbedder::open(model_dir)?)
        } else {
            None
        };

        // Dimension priority: HEF output dim > cpu-fallback BERT config
        // > MINI_LM_DIM constant. The HEF was compiled for hidden_size
        // 384 in iter-156b; this gate makes any future HEF with a
        // different hidden_size automatically picked up.
        #[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
        let dimensions = hef_embedder
            .as_ref()
            .map(HefBackend::output_dim)
            .or_else(|| cpu_fallback.as_ref().map(|c| c.output_dim()))
            .unwrap_or(crate::inference::MINI_LM_DIM);
        #[cfg(all(not(feature = "hailo"), feature = "cpu-fallback"))]
        let dimensions = cpu_fallback
            .as_ref()
            .map(|c| c.output_dim())
            .unwrap_or(crate::inference::MINI_LM_DIM);
        #[cfg(not(feature = "cpu-fallback"))]
        let dimensions = crate::inference::MINI_LM_DIM;

        // Artifact hashes were already verified in `open_with_identity`, which
        // runs before any model is loaded; only the dimension can be checked
        // here, once the backend knows what it loaded.
        let identity = match identity {
            Some(identity) => {
                if identity.output_dimension as usize != dimensions {
                    return Err(HailoError::EmbeddingProvenance(format!(
                        "identity dimension {} does not match loaded model dimension {dimensions}",
                        identity.output_dimension
                    )));
                }
                identity
            }
            None => Self::default_identity(model_dir, dimensions)?,
        };

        Ok(Self {
            dimensions,
            identity,
            name: Self::model_name(model_dir),
            device_id,
            #[cfg(feature = "hailo")]
            device: device_opt.map(Mutex::new),
            #[cfg(feature = "cpu-fallback")]
            cpu_fallback,
            #[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
            hef_embedder,
        })
    }

    /// Human-readable model name derived from the model directory.
    fn model_name(model_dir: &Path) -> String {
        format!(
            "hailo:{}",
            model_dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown-model")
        )
    }

    /// Identity used by [`open`](Self::open), which has no caller-supplied
    /// provenance. See that method's docs for what it does and does not
    /// attest.
    ///
    /// Nothing here depends on the filesystem path. `model_dir` is where the
    /// bytes happen to live today; moving or renaming that directory does not
    /// change which vectors the model produces, so it must not change the
    /// embedding-space id. An earlier version put the directory basename in the
    /// fallback `model_id`, which meant renaming a model directory made every
    /// persisted corpus reject on reopen. The fallback now keys off the model
    /// artifact digest instead, which moves with the bytes.
    fn default_identity(model_dir: &Path, dimensions: usize) -> Result<EmbeddingSpaceIdentity> {
        let digests = Self::artifact_digests(model_dir).ok();
        let mut identity = match ruvector_core::embeddings::registry_identity(
            "ruvector-hailo",
            DEFAULT_MODEL_ID,
            dimensions,
        ) {
            Ok(identity) => identity,
            // Not the model this backend ships for (a HEF compiled at some
            // other hidden size). Name the space after the artifact bytes.
            Err(_) => {
                let fallback_id = digests
                    .as_ref()
                    .and_then(|d| d.model_artifact.as_deref())
                    .map(|digest| format!("hailo-artifact-sha256:{digest}"))
                    .unwrap_or_else(|| "hailo-artifact-sha256:absent".to_string());
                ruvector_core::embeddings::default_space_identity(
                    "ruvector-hailo",
                    &fallback_id,
                    dimensions,
                )
            }
        };
        // Bind whatever artifacts are actually present into the identity, so
        // swapping model bytes changes the embedding-space id even on this
        // unattested path. Absent files keep the derived placeholder digest.
        if let Some(digests) = digests {
            if let Some(digest) = digests.model_artifact {
                identity.model_artifact_sha256 = digest;
            }
            if let Some(digest) = digests.model_graph {
                identity.model_graph_sha256 = digest;
            }
            if let Some(digest) = digests.tokenizer {
                identity.tokenizer_sha256 = digest;
            }
        }
        identity
            .validate()
            .map_err(|e| HailoError::EmbeddingProvenance(e.to_string()))?;
        Ok(identity)
    }

    /// Read the on-die NPU temperature(s) from the held-open vdevice.
    /// Returns `(ts0_celsius, ts1_celsius)` — Hailo-8 has two thermal
    /// sensors on the chip. `None` if the read failed (cluster
    /// callers treat that as "skip the npu_temp gauge for this tick").
    ///
    /// Iter 95 deliverable from ADR-174 §93.
    pub fn chip_temperature(&self) -> Option<(f32, f32)> {
        #[cfg(not(feature = "hailo"))]
        {
            None
        }
        #[cfg(feature = "hailo")]
        {
            // None when no HAT was present at open time — cpu-fallback
            // path with no NPU. Caller treats this the same as a failed
            // sensor read, which is the correct semantic.
            let g = self
                .device
                .as_ref()?
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            g.chip_temperature()
        }
    }

    /// Embed a single piece of text into a `dimensions()`-element f32 vector.
    ///
    /// Embed `text` into a `dim`-length unit vector.
    ///
    /// **Iter 130 — placeholder removed.** Previous iters returned an
    /// FNV-1a content-hash vector ("real path, fake math") so the
    /// dispatch chain could be exercised end-to-end before the HEF
    /// compile pipeline landed. That was misleading — operators saw
    /// vectors come back and reasonably assumed they were embeddings.
    /// Now `embed` returns `HailoError::NoModelLoaded` until a real
    /// model graph is wired in, so the cluster's failure mode honestly
    /// reflects "no inference happening."
    ///
    /// **What still works without a model:** open / dimensions / device
    /// id / chip_temperature / the entire gRPC stack. The worker boots,
    /// reports ready=false (since dimensions=0 is the gate, but iter 87
    /// pre-declared 384 to keep the path testable; iter 130 keeps that
    /// pre-declaration so health probes succeed and the operator-side
    /// `--validate-fleet` flow can detect "model missing" via a clean
    /// embed failure rather than a connection-refused).
    ///
    /// **To make `embed` work end-to-end:** see the iter-130 commit
    /// message and ADR-167's "What's still unimplemented" section —
    /// drop a compiled `model.hef` into the worker's model dir and
    /// restart. The existing `HailoEmbedder::open` path picks it up;
    /// the ModelLoaded gate trips and `embed` starts dispatching to
    /// the NPU's vstream API.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Iter 162 (ADR-176 P4): dispatch order:
        //   1. NPU HEF (real NPU acceleration, ~73 FPS encoder)
        //   2. CPU fallback (host CPU BERT-6, ~7 FPS / Pi worker)
        //   3. NoModelLoaded — health probes still serve
        //   4. FeatureDisabled if neither feature is built in
        #[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
        if let Some(hef) = &self.hef_embedder {
            return hef.embed(text);
        }

        #[cfg(feature = "cpu-fallback")]
        if let Some(cpu) = &self.cpu_fallback {
            return cpu.embed(text);
        }

        #[cfg(feature = "hailo")]
        {
            let _ = text;
            return Err(HailoError::NoModelLoaded);
        }

        #[allow(unreachable_code)]
        {
            let _ = text;
            Err(HailoError::FeatureDisabled)
        }
    }

    /// Embed a batch of texts. Default impl loops; iteration 7 replaces
    /// with batched-vstream feed when the HEF is compiled with batch>1.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t)?);
        }
        Ok(out)
    }

    /// Vector dimensionality (e.g. 384 for `all-MiniLM-L6-v2`).
    /// Mirrors `EmbeddingProvider::dimensions()`.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Iter 130: honest "is a model graph actually loaded?" gate.
    /// Returns `true` only when `embed()` would do real semantic
    /// inference (either NPU via HEF or host CPU via the cpu-fallback
    /// candle path). The worker's `health()` uses this to set the
    /// `ready` flag so the cluster's `validate_fleet` correctly
    /// identifies model-less workers as not-ready instead of
    /// false-healthy.
    ///
    /// Iter 163 made this canonically `true` for the production NPU
    /// path (cognitum-v0 + iter-156b HEF); iter-176 added the
    /// cpu-fallback automatic failover. Iter 223 — corrected this
    /// doc comment, which still claimed "always false" from the
    /// iter-130-era (same stale-stratigraphy class iter-217 fixed
    /// in ADR-167).
    pub fn has_model(&self) -> bool {
        // Iter 162 (ADR-176 P4): HEF + cpu-fallback both count.
        #[cfg(all(feature = "hailo", feature = "cpu-fallback"))]
        {
            if self.hef_embedder.is_some() {
                return true;
            }
        }
        #[cfg(feature = "cpu-fallback")]
        {
            if self.cpu_fallback.is_some() {
                return true;
            }
        }
        false
    }

    /// Human-readable provider name. Mirrors `EmbeddingProvider::name()`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// PCIe BDF, e.g. `"0001:01:00.0"`. Empty before `open()` succeeds.
    /// Hailo-specific extension — not on the EmbeddingProvider trait.
    pub fn device_id(&self) -> &str {
        &self.device_id
    }
}

// SAFETY: HailoEmbedder owns `Option<Mutex<HailoDevice>>` (iter 137,
// see field declaration). The HailoRT C library is documented thread-
// safe per device handle when accessed under a single configuration;
// the Mutex wrapper enforces the rest. The HEF backend behind
// `hef_embedder` (iter 234+) carries its own per-slot Mutex<Inner>,
// either via the single-pipeline `HefEmbedder` or the multi-pipeline
// `HefEmbedderPool`. Send+Sync are required by `EmbeddingProvider`.
unsafe impl Send for HailoEmbedder {}
unsafe impl Sync for HailoEmbedder {}

/// Iter 218 — closes ADR-178 Gap B (HIGH) part 1. Implements
/// `ruvector_core::embeddings::EmbeddingProvider` for `HailoEmbedder`,
/// the headline integration ADR-167 §2.5 promised but never delivered.
///
/// All three methods delegate to the existing inherent methods; the
/// only translation is `HailoError → ruvector_core::RuvectorError`,
/// folded into `ModelInferenceError(String)` for non-dim failures and
/// `DimensionMismatch` for the (unreachable but well-typed) dim
/// path.
///
/// Effect: `Arc<dyn EmbeddingProvider>` callers (the recommended
/// ruvector-core consumer pattern) can now hold a `HailoEmbedder`
/// without rewriting around the inherent-method API.
impl ruvector_core::embeddings::EmbeddingProvider for HailoEmbedder {
    fn embed_for(
        &self,
        role: ruvector_core::EmbeddingRole,
        text: &str,
    ) -> ruvector_core::Result<Vec<f32>> {
        if self.identity.role_policy == ruvector_core::EmbeddingRolePolicy::Asymmetric {
            return Err(ruvector_core::RuvectorError::EmbeddingRoleUnsupported(
                format!(
                    "Hailo backend has no verified {:?} role transform for this identity",
                    role
                ),
            ));
        }
        HailoEmbedder::embed(self, text)
            .map_err(|e| ruvector_core::RuvectorError::ModelInferenceError(e.to_string()))
    }

    fn embedding_space(&self) -> &ruvector_core::EmbeddingSpaceIdentity {
        &self.identity
    }

    fn dimensions(&self) -> usize {
        HailoEmbedder::dimensions(self)
    }

    fn name(&self) -> &str {
        HailoEmbedder::name(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_on_missing_dir_resolves_without_panic() {
        // Across all feature combos, opening against a nonexistent dir
        // must resolve to either:
        //   * Err(FeatureDisabled / NoDevice / BadModelDir / ...) —
        //     hard failure modes the operator can act on
        //   * Ok(embedder) with has_model() == false — the iter-130
        //     "model not yet present" path that lets health probes
        //     report ready=false instead of connection-refused
        let r = HailoEmbedder::open(Path::new("/nonexistent"));
        match r {
            Ok(e) => assert!(
                !e.has_model(),
                "open(missing dir) returned Ok but has_model=true — should be ready=false"
            ),
            Err(
                HailoError::FeatureDisabled
                | HailoError::NotYetImplemented(_)
                | HailoError::BadModelDir { .. }
                | HailoError::NoDevice(_)
                | HailoError::Tokenizer(_)
                | HailoError::EmbeddingProvenance(_),
            ) => {}
            Err(other) => panic!("unexpected open() error: {:?}", other),
        }
    }

    /// Smoke test for the identity-free entry point across every feature
    /// combo: it must never be the ADR-281 provenance path that stops it.
    /// Feature/device failures are expected on hosts without a HAT; a
    /// provenance failure would mean `open` is unusable again, which is what
    /// broke the worker daemon.
    #[test]
    fn open_never_fails_on_provenance() {
        let dir = tempfile::tempdir().expect("tempdir");
        match HailoEmbedder::open(dir.path()) {
            Ok(embedder) => {
                assert!(
                    embedder.dimensions() > 0,
                    "default identity must carry a usable dimension"
                );
                assert_eq!(
                    ruvector_core::EmbeddingProvider::embedding_space(&embedder).output_dimension
                        as usize,
                    embedder.dimensions()
                );
            }
            Err(HailoError::EmbeddingProvenance(e)) => {
                panic!("open must not require caller-supplied provenance: {e}")
            }
            Err(_) => {}
        }
    }

    /// Provenance verification fails closed: a directory whose layout holds
    /// none of the artifacts this backend knows how to fingerprint must be
    /// refused, not silently accepted as verified.
    #[test]
    fn open_with_identity_fails_closed_on_unknown_layout() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("weights.bin"), b"not a known artifact").unwrap();
        let identity = ruvector_core::embeddings::registry_identity(
            "test",
            "sentence-transformers/all-MiniLM-L6-v2",
            crate::inference::MINI_LM_DIM,
        )
        .unwrap();

        let err = match HailoEmbedder::open_with_identity(dir.path(), identity) {
            Ok(_) => panic!("unverifiable artifact layout must not open"),
            Err(e) => e,
        };
        let message = err.to_string();
        assert!(
            matches!(err, HailoError::EmbeddingProvenance(_)),
            "expected a provenance error, got {err:?}"
        );
        for expected in [
            "model.safetensors",
            "model.hef",
            "tokenizer.json",
            "vocab.txt",
        ] {
            assert!(
                message.contains(expected),
                "error must list the artifacts looked for, missing {expected}: {message}"
            );
        }
    }

    /// The two model hashes must be independently useful: on the production
    /// layout (weights + compiled graph) they fingerprint different files, and
    /// on the single-file layouts they are equal by the documented convention.
    /// If they were always equal, no real NPU deployment could ever be attested.
    #[test]
    fn artifact_digests_follow_the_documented_layout_table() {
        let both = tempfile::tempdir().expect("tempdir");
        std::fs::write(both.path().join("model.safetensors"), b"weights").unwrap();
        std::fs::write(both.path().join("model.hef"), b"compiled graph").unwrap();
        std::fs::write(both.path().join("vocab.txt"), b"[PAD]\n").unwrap();
        let d = HailoEmbedder::artifact_digests(both.path()).expect("known layout");
        assert_ne!(
            d.model_artifact, d.model_graph,
            "NPU + fallback layout must fingerprint safetensors and hef separately"
        );
        assert!(d.tokenizer.is_some());

        for single in ["model.safetensors", "model.hef"] {
            let dir = tempfile::tempdir().expect("tempdir");
            std::fs::write(dir.path().join(single), b"only artifact").unwrap();
            let d = HailoEmbedder::artifact_digests(dir.path()).expect("known layout");
            assert_eq!(
                d.model_artifact, d.model_graph,
                "{single}-only layout has no separately compiled graph"
            );
        }

        let unknown = tempfile::tempdir().expect("tempdir");
        std::fs::write(unknown.path().join("weights.bin"), b"x").unwrap();
        assert!(
            HailoEmbedder::artifact_digests(unknown.path()).is_err(),
            "an unfingerprintable layout must not report digests"
        );
    }

    /// An identity built from `artifact_digests` is exactly what the verifier
    /// accepts — otherwise `open_with_identity` would be unusable in practice.
    #[test]
    fn digests_from_helper_satisfy_verification() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        std::fs::write(dir.path().join("model.hef"), b"compiled graph").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();

        let digests = HailoEmbedder::artifact_digests(dir.path()).unwrap();
        let mut identity = ruvector_core::embeddings::registry_identity(
            "test",
            "sentence-transformers/all-MiniLM-L6-v2",
            crate::inference::MINI_LM_DIM,
        )
        .unwrap();
        identity.model_artifact_sha256 = digests.model_artifact.unwrap();
        identity.model_graph_sha256 = digests.model_graph.unwrap();
        identity.tokenizer_sha256 = digests.tokenizer.unwrap();

        HailoEmbedder::verify_artifact_provenance(dir.path(), &identity)
            .expect("digests produced by artifact_digests must verify");
    }

    /// Where the bytes live is not part of what they embed: moving or renaming
    /// a model directory must not change the embedding-space id, or every
    /// persisted corpus would reject on reopen after a routine path change.
    #[test]
    fn default_identity_is_stable_across_directory_renames() {
        // Both branches: the registry fixture (MiniLM dimensionality) and the
        // artifact-keyed fallback (any other dimensionality).
        for dimensions in [crate::inference::MINI_LM_DIM, 512] {
            let parent = tempfile::tempdir().expect("tempdir");
            let before = parent.path().join("all-minilm-l6-v2");
            std::fs::create_dir(&before).unwrap();
            std::fs::write(before.join("model.safetensors"), b"weights").unwrap();
            std::fs::write(before.join("tokenizer.json"), b"{}").unwrap();
            let id_before = HailoEmbedder::default_identity(&before, dimensions)
                .unwrap()
                .embedding_space_id()
                .unwrap();

            let after = parent.path().join("models-2026-08-02");
            std::fs::rename(&before, &after).unwrap();
            let id_after = HailoEmbedder::default_identity(&after, dimensions)
                .unwrap()
                .embedding_space_id()
                .unwrap();

            assert_eq!(
                id_before, id_after,
                "renaming a model dir changed the {dimensions}D embedding-space id"
            );

            // The bytes, in contrast, must change it.
            std::fs::write(after.join("model.safetensors"), b"different weights").unwrap();
            let id_new_bytes = HailoEmbedder::default_identity(&after, dimensions)
                .unwrap()
                .embedding_space_id()
                .unwrap();
            assert_ne!(
                id_after, id_new_bytes,
                "swapping model bytes must change the {dimensions}D embedding-space id"
            );
        }
    }

    /// A known layout whose bytes disagree with the declared hashes is
    /// rejected too.
    #[test]
    fn open_with_identity_rejects_hash_mismatch() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        let identity = ruvector_core::embeddings::registry_identity(
            "test",
            "sentence-transformers/all-MiniLM-L6-v2",
            crate::inference::MINI_LM_DIM,
        )
        .unwrap();

        match HailoEmbedder::open_with_identity(dir.path(), identity) {
            Ok(_) => panic!("mismatched artifact hashes must not open"),
            Err(HailoError::EmbeddingProvenance(msg)) => {
                assert!(
                    msg.contains("SHA-256 mismatch"),
                    "unexpected message: {msg}"
                )
            }
            Err(other) => panic!("expected a provenance error, got {other:?}"),
        }
    }

    #[test]
    fn embedding_provider_signature_parity() {
        // Iter 218 — closes ADR-178 Gap B part 1. Was a no-op (only
        // `T: Send + Sync`). Now asserts the real
        // `impl EmbeddingProvider for HailoEmbedder` block compiles
        // — if the trait drifts and the impl breaks, this test fails
        // at the bound check. Catches the same regression class
        // ADR-178 flagged: a future trait-shape change to
        // `EmbeddingProvider` that the hailo crate doesn't propagate.
        fn assert_impl<T: ruvector_core::embeddings::EmbeddingProvider>() {}
        assert_impl::<HailoEmbedder>();
    }
}
