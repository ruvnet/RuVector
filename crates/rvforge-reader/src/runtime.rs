//! Runtime selection (FR004, ADR-289 §2).
//!
//! The reader picks the strongest compatible runtime by walking the
//! `selectionOrder` in the compatibility matrix and taking the first entry the
//! host can actually provide. `Unsupported` is terminal: the reader refuses to
//! execute rather than falling back to something weaker.
//!
//! The order is read from a vendored copy of the canonical matrix and is not
//! configurable at runtime. ADR-289 permits reordering only by signed policy;
//! signed policy is not implemented yet, so [`PolicySource`] always reports
//! [`PolicySource::EmbeddedDefault`].

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Vendored copy of `docs/research/rvf-forge/compatibility-matrix.json`.
///
/// Vendored rather than read from disk so that a file swap on the installed
/// machine cannot reorder the selection list.
pub const COMPATIBILITY_MATRIX_JSON: &str = include_str!("../assets/compatibility-matrix.json");

#[derive(Debug, Clone, Deserialize)]
pub struct CompatibilityMatrix {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    pub updated: String,
    #[serde(rename = "runtimeProfiles")]
    pub runtime_profiles: BTreeMap<String, MatrixProfile>,
    #[serde(rename = "selectionOrder")]
    pub selection_order: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MatrixProfile {
    /// `supported` or `planned`. Only `supported` profiles are selectable.
    pub status: String,
    #[serde(rename = "isolationClaim")]
    pub isolation_claim: String,
    #[serde(default)]
    pub platforms: Vec<PlatformEntry>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlatformEntry {
    pub os: String,
    pub arch: Vec<String>,
    #[serde(default)]
    pub min: String,
    #[serde(default)]
    pub mechanisms: Vec<String>,
}

/// What the host adapter can actually provide.
///
/// Every field is a claim the adapter must be able to back. ADR-285 §3 requires
/// an adapter that could not engage a mechanism to report that rather than
/// silently running with a weaker stack, so these default to `false`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostProfile {
    /// `windows` | `macos` | `linux` | `browser` | `rvm`.
    pub os: String,
    /// `x64` | `aarch64` | `wasm32`.
    pub arch: String,
    /// The adapter can engage this platform's OS confinement stack
    /// (job objects / app sandbox / namespaces+seccomp).
    pub os_isolation: bool,
    /// A KVM-capable Linux host is available for the microVM path.
    pub kvm: bool,
    /// Bare-metal RVM with seven-phase measured boot.
    pub rvm_measured_boot: bool,
}

impl HostProfile {
    /// The host as this build can honestly describe it.
    ///
    /// `rvm-host` adapters do not exist yet, so no OS confinement, KVM, or
    /// measured boot is claimed. Selection therefore lands on plain `wasm`
    /// today; it will move up the order as adapters land.
    pub fn detect() -> Self {
        let os = if cfg!(target_os = "windows") {
            "windows"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else if cfg!(target_arch = "wasm32") {
            "browser"
        } else {
            "linux"
        };
        let arch = if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else if cfg!(target_arch = "wasm32") {
            "wasm32"
        } else {
            "x64"
        };
        Self {
            os: os.to_string(),
            arch: arch.to_string(),
            os_isolation: false,
            kvm: false,
            rvm_measured_boot: false,
        }
    }
}

/// Where the selection order came from. Only signed policy may replace the
/// embedded default (ADR-289 §2); that path is not implemented.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PolicySource {
    EmbeddedDefault,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SelectionStatus {
    Selected,
    /// Terminal. The reader refuses to execute.
    Unsupported,
}

/// Why a profile was or was not eligible. Surfaced in the UI so the boundary
/// is observable rather than assumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Evaluation {
    pub profile: String,
    pub eligible: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeChoice {
    pub status: SelectionStatus,
    /// The selected profile id, or `None` when unsupported.
    pub profile: Option<String>,
    /// The isolation class the selected profile actually claims. Hosted RVM
    /// claims `os-sandbox+wasm`, never bare-metal isolation (ADR-285).
    pub isolation_claim: Option<String>,
    /// Mechanisms the matrix lists for this host under the selected profile.
    pub mechanisms: Vec<String>,
    pub order: Vec<String>,
    pub evaluated: Vec<Evaluation>,
    pub policy_source: PolicySource,
    pub matrix_schema_version: u32,
    pub matrix_updated: String,
    pub host: HostProfile,
    /// Path this selection was reported for, when a subject was supplied.
    pub subject: Option<String>,
}

#[derive(Debug)]
pub enum RuntimeError {
    Matrix(String),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::Matrix(m) => write!(f, "compatibility matrix error: {m}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Parse the vendored matrix.
pub fn embedded_matrix() -> Result<CompatibilityMatrix, RuntimeError> {
    serde_json::from_str(COMPATIBILITY_MATRIX_JSON).map_err(|e| RuntimeError::Matrix(e.to_string()))
}

/// Apply the FR004 order against a host, using the vendored matrix.
pub fn select_for_host(host: &HostProfile) -> Result<RuntimeChoice, RuntimeError> {
    let matrix = embedded_matrix()?;
    Ok(select(&matrix, host))
}

/// Apply the FR004 order against a host and an explicit matrix.
pub fn select(matrix: &CompatibilityMatrix, host: &HostProfile) -> RuntimeChoice {
    let mut evaluated = Vec::new();
    let mut chosen: Option<(String, String, Vec<String>)> = None;

    for id in &matrix.selection_order {
        let Some(profile) = matrix.runtime_profiles.get(id) else {
            evaluated.push(Evaluation {
                profile: id.clone(),
                eligible: false,
                reason: "profile absent from matrix".to_string(),
            });
            continue;
        };

        match eligibility(id, profile, host) {
            Ok(mechanisms) => {
                if chosen.is_none() {
                    chosen = Some((id.clone(), profile.isolation_claim.clone(), mechanisms));
                    evaluated.push(Evaluation {
                        profile: id.clone(),
                        eligible: true,
                        reason: "selected: strongest compatible runtime".to_string(),
                    });
                } else {
                    evaluated.push(Evaluation {
                        profile: id.clone(),
                        eligible: true,
                        reason: "eligible but weaker than the selected runtime".to_string(),
                    });
                }
            }
            Err(reason) => evaluated.push(Evaluation {
                profile: id.clone(),
                eligible: false,
                reason,
            }),
        }
    }

    match chosen {
        Some((profile, isolation_claim, mechanisms)) => RuntimeChoice {
            status: SelectionStatus::Selected,
            profile: Some(profile),
            isolation_claim: Some(isolation_claim),
            mechanisms,
            order: matrix.selection_order.clone(),
            evaluated,
            policy_source: PolicySource::EmbeddedDefault,
            matrix_schema_version: matrix.schema_version,
            matrix_updated: matrix.updated.clone(),
            host: host.clone(),
            subject: None,
        },
        None => RuntimeChoice {
            status: SelectionStatus::Unsupported,
            profile: None,
            isolation_claim: None,
            mechanisms: Vec::new(),
            order: matrix.selection_order.clone(),
            evaluated,
            policy_source: PolicySource::EmbeddedDefault,
            matrix_schema_version: matrix.schema_version,
            matrix_updated: matrix.updated.clone(),
            host: host.clone(),
            subject: None,
        },
    }
}

/// `Ok(mechanisms)` when the host can provide this profile, `Err(reason)`
/// otherwise. Reasons are user-facing.
fn eligibility(
    id: &str,
    profile: &MatrixProfile,
    host: &HostProfile,
) -> Result<Vec<String>, String> {
    if profile.status != "supported" {
        return Err(format!(
            "matrix status is '{}', not selectable",
            profile.status
        ));
    }

    let entry = profile
        .platforms
        .iter()
        .find(|p| p.os == host.os && p.arch.iter().any(|a| arch_matches(a, &host.arch)))
        .ok_or_else(|| format!("no platform entry for {}/{}", host.os, host.arch))?;

    // Capability gates beyond the platform table. Each corresponds to a
    // mechanism the host adapter must actually engage (ADR-285 §3).
    match id {
        "rvm-native" => {
            if !host.rvm_measured_boot {
                return Err("host does not provide RVM measured boot".to_string());
            }
        }
        "os-isolation+wasm" => {
            if !host.os_isolation {
                return Err("host adapter cannot engage OS confinement".to_string());
            }
        }
        "linux-microvm" => {
            if !host.kvm {
                return Err("host has no KVM-capable hypervisor".to_string());
            }
        }
        "wasm" => {}
        other => return Err(format!("unknown runtime profile '{other}'")),
    }

    Ok(entry.mechanisms.clone())
}

/// The matrix spells the same architecture differently per OS
/// (`arm64` on Windows, `aarch64` elsewhere).
fn arch_matches(matrix_arch: &str, host_arch: &str) -> bool {
    fn canon(a: &str) -> Option<&'static str> {
        match a {
            "x86_64" | "amd64" | "x64" => Some("x64"),
            "arm64" | "aarch64" => Some("aarch64"),
            "wasm32" => Some("wasm32"),
            _ => None,
        }
    }
    match (canon(matrix_arch), canon(host_arch)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}
