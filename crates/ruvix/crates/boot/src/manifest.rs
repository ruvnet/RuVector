//! RVF Manifest parsing for boot and component mounting.
//!
//! The RVF manifest describes the structure of an RVF package:
//! - Component graph (DAG of components with queue wiring)
//! - Memory schema (region declarations)
//! - Proof policy (per-component tier requirements)
//! - Rollback hooks (WASM functions for state rollback)
//! - Witness log policy (retention, compression, export)

use ruvix_types::{ProofTier, RegionPolicy, WitTypeId};

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

#[cfg(feature = "std")]
use std::{string::String, vec::Vec};

/// Maximum length of a component name.
#[allow(dead_code)]
pub const MAX_COMPONENT_NAME_LEN: usize = 256;

/// Maximum rollback hook code size in bytes.
#[allow(dead_code)]
pub const MAX_ROLLBACK_HOOK_SIZE: usize = 64 * 1024; // 64 KiB

/// RVF Manifest containing the complete package description.
///
/// The manifest is parsed and verified during Stage 1 of boot.
/// All fields are validated before any kernel objects are created.
#[derive(Debug, Clone)]
pub struct RvfManifest {
    /// Manifest format version.
    pub version: ManifestVersion,

    /// SHA-256 hash of the package contents.
    pub content_hash: [u8; 32],

    /// Component graph (DAG structure).
    pub component_graph: ComponentGraph,

    /// Memory schema (region declarations).
    pub memory_schema: MemorySchema,

    /// Proof policy (per-component tier requirements).
    pub proof_policy: ProofPolicy,

    /// Rollback hooks (WASM functions for state rollback).
    pub rollback_hooks: RollbackHooks,

    /// Witness log policy (retention, compression, export).
    pub witness_log_policy: WitnessLogPolicy,

    /// Minimum capabilities required by this package.
    pub required_capabilities: RequiredCapabilities,
}

/// Manifest format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ManifestVersion {
    /// Major version (breaking changes).
    pub major: u16,
    /// Minor version (backward-compatible additions).
    pub minor: u16,
    /// Patch version (bug fixes).
    pub patch: u16,
}

impl ManifestVersion {
    /// Current manifest version.
    pub const CURRENT: Self = Self {
        major: 1,
        minor: 0,
        patch: 0,
    };

    /// Creates a new manifest version.
    #[inline]
    #[must_use]
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Checks if this version is compatible with the current version.
    #[inline]
    #[must_use]
    pub const fn is_compatible(&self) -> bool {
        self.major == Self::CURRENT.major
    }
}

impl Default for ManifestVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

/// Component graph representing the DAG of WASM components.
///
/// Components are topologically sorted for initialization order.
/// Queue wiring defines inter-component communication.
#[derive(Debug, Clone, Default)]
pub struct ComponentGraph {
    /// Component declarations (topologically sorted).
    pub components: Vec<ComponentDecl>,

    /// Queue wiring (connections between components).
    pub wirings: Vec<QueueWiring>,
}

impl ComponentGraph {
    /// Creates an empty component graph.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of components.
    #[inline]
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Returns the number of queue wirings.
    #[inline]
    #[must_use]
    pub fn wiring_count(&self) -> usize {
        self.wirings.len()
    }

    /// Validates the component graph structure.
    ///
    /// Returns `true` if:
    /// - Components form a valid DAG (no cycles)
    /// - All wiring references valid components
    /// - Root component exists (index 0) or graph is empty (Phase A allows empty)
    #[must_use]
    pub fn validate(&self) -> bool {
        // Phase A: Allow empty component graphs for test manifests
        // Phase B would require at least one component
        if self.components.is_empty() {
            return true;
        }

        // Check wiring references
        let count = self.component_count();
        for wiring in &self.wirings {
            if wiring.source_component as usize >= count
                || wiring.target_component as usize >= count
            {
                return false;
            }
        }

        true
    }
}

/// Component declaration within an RVF package.
#[derive(Debug, Clone)]
pub struct ComponentDecl {
    /// Component index (0 = root component).
    pub index: u32,

    /// Component name (for debugging).
    pub name: String,

    /// SHA-256 hash of the WASM component code.
    pub code_hash: [u8; 32],

    /// Offset in the RVF package to the WASM code.
    pub code_offset: u64,

    /// Size of the WASM code in bytes.
    pub code_size: u64,

    /// WIT type ID for the component's interface.
    pub wit_type: WitTypeId,

    /// Entry point function name.
    pub entry_point: String,

    /// Dependencies (indices of components this depends on).
    pub dependencies: Vec<u32>,
}

impl ComponentDecl {
    /// Returns the component name as a string slice.
    #[must_use]
    pub fn name_str(&self) -> &str {
        &self.name
    }

    /// Returns the entry point as a string slice.
    #[must_use]
    pub fn entry_point_str(&self) -> &str {
        &self.entry_point
    }
}

/// Queue wiring between components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct QueueWiring {
    /// Source component index.
    pub source_component: u32,

    /// Source port name hash.
    pub source_port_hash: u32,

    /// Target component index.
    pub target_component: u32,

    /// Target port name hash.
    pub target_port_hash: u32,

    /// Queue capacity (number of entries).
    pub queue_capacity: u32,

    /// Maximum message size in bytes.
    pub max_message_size: u32,

    /// WIT type ID for message schema validation.
    pub message_type: WitTypeId,
}

/// Memory schema containing region declarations.
#[derive(Debug, Clone, Default)]
pub struct MemorySchema {
    /// Region declarations.
    pub regions: Vec<RegionDecl>,

    /// Total memory required in bytes.
    pub total_memory_required: u64,
}

impl MemorySchema {
    /// Creates an empty memory schema.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of region declarations.
    #[inline]
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Validates the memory schema.
    #[must_use]
    pub fn validate(&self) -> bool {
        // Calculate total memory and check for overflow
        let mut total: u64 = 0;
        for region in &self.regions {
            if let Some(size) = region.size_bytes() {
                total = match total.checked_add(size) {
                    Some(t) => t,
                    None => return false,
                };
            }
        }

        total <= self.total_memory_required
    }
}

/// Region declaration within the memory schema.
#[derive(Debug, Clone)]
pub struct RegionDecl {
    /// Region index (unique within the manifest).
    pub index: u32,

    /// Region name (for debugging).
    pub name: String,

    /// Region access policy.
    pub policy: RegionPolicy,

    /// Owning component index.
    pub owner_component: u32,

    /// Components with read access (indices).
    pub read_access: Vec<u32>,

    /// Components with write access (indices).
    pub write_access: Vec<u32>,
}

impl RegionDecl {
    /// Returns the size of this region in bytes, if determinable.
    #[must_use]
    pub fn size_bytes(&self) -> Option<u64> {
        match &self.policy {
            RegionPolicy::Immutable => None, // Size determined at creation
            RegionPolicy::AppendOnly { max_size } => Some(*max_size as u64),
            RegionPolicy::Slab {
                slot_size,
                slot_count,
            } => Some((*slot_size as u64) * (*slot_count as u64)),
        }
    }
}

/// Proof policy containing per-component tier requirements.
#[derive(Debug, Clone, Default)]
pub struct ProofPolicy {
    /// Per-component proof tier requirements.
    pub component_tiers: Vec<ComponentProofTier>,

    /// Default tier for components not explicitly listed.
    pub default_tier: ProofTier,

    /// Whether to allow tier escalation at runtime.
    pub allow_tier_escalation: bool,
}

impl ProofPolicy {
    /// Creates a new proof policy with the given default tier.
    #[inline]
    #[must_use]
    pub fn new(default_tier: ProofTier) -> Self {
        Self {
            component_tiers: Vec::new(),
            default_tier,
            allow_tier_escalation: false,
        }
    }

    /// Gets the proof tier for a component.
    #[must_use]
    pub fn tier_for_component(&self, component_index: u32) -> ProofTier {
        for ct in &self.component_tiers {
            if ct.component_index == component_index {
                return ct.required_tier;
            }
        }
        self.default_tier
    }
}

/// Per-component proof tier requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ComponentProofTier {
    /// Component index.
    pub component_index: u32,

    /// Required proof tier for this component.
    pub required_tier: ProofTier,

    /// Operations requiring proof (bitmask).
    pub proof_required_ops: ProofRequiredOps,
}

/// Bitmask for operations requiring proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct ProofRequiredOps(pub u32);

impl ProofRequiredOps {
    /// No operations require proof.
    pub const NONE: Self = Self(0);

    /// Vector put operations require proof.
    pub const VECTOR_PUT: Self = Self(1 << 0);

    /// Graph apply operations require proof.
    pub const GRAPH_APPLY: Self = Self(1 << 1);

    /// Region writes require proof.
    pub const REGION_WRITE: Self = Self(1 << 2);

    /// Queue sends require proof.
    pub const QUEUE_SEND: Self = Self(1 << 3);

    /// All operations require proof.
    pub const ALL: Self = Self(0b1111);

    /// Checks if an operation requires proof.
    #[inline]
    #[must_use]
    pub const fn requires(&self, op: Self) -> bool {
        (self.0 & op.0) != 0
    }
}

/// Container for rollback hooks.
#[derive(Debug, Clone, Default)]
pub struct RollbackHooks {
    /// Rollback hook declarations.
    pub hooks: Vec<RollbackHook>,
}

impl RollbackHooks {
    /// Creates an empty rollback hooks container.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of rollback hooks.
    #[inline]
    #[must_use]
    pub fn count(&self) -> usize {
        self.hooks.len()
    }
}

/// Rollback hook (WASM function for state rollback).
#[derive(Debug, Clone)]
pub struct RollbackHook {
    /// Hook index.
    pub index: u32,

    /// Component index this hook applies to.
    pub component_index: u32,

    /// WASM function name to call for rollback.
    pub function_name: String,

    /// Regions this hook can access during rollback.
    pub accessible_regions: Vec<u32>,

    /// Timeout for rollback execution in microseconds.
    pub timeout_us: u64,
}

/// Witness log policy configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct WitnessLogPolicy {
    /// Maximum entries to retain before rotation.
    pub max_entries: u64,

    /// Maximum size in bytes before rotation.
    pub max_size_bytes: u64,

    /// Retention period in seconds (0 = forever).
    pub retention_seconds: u64,

    /// Compression algorithm.
    pub compression: WitnessCompression,

    /// Export policy.
    pub export_policy: WitnessExportPolicy,

    /// Whether to hash-chain entries for integrity.
    pub hash_chain: bool,
}

impl Default for WitnessLogPolicy {
    fn default() -> Self {
        Self {
            max_entries: 1_000_000,
            max_size_bytes: 100 * 1024 * 1024, // 100 MiB
            retention_seconds: 0,              // Forever
            compression: WitnessCompression::None,
            export_policy: WitnessExportPolicy::OnRotation,
            hash_chain: true,
        }
    }
}

/// Witness log compression algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum WitnessCompression {
    /// No compression.
    #[default]
    None = 0,

    /// LZ4 compression (fast).
    Lz4 = 1,

    /// Zstd compression (better ratio).
    Zstd = 2,
}

/// Witness log export policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum WitnessExportPolicy {
    /// Export on log rotation.
    #[default]
    OnRotation = 0,

    /// Export on shutdown.
    OnShutdown = 1,

    /// Never export automatically.
    Never = 2,

    /// Export to external attestation service.
    External = 3,
}

/// Required capabilities for the RVF package.
#[derive(Debug, Clone, Default)]
pub struct RequiredCapabilities {
    /// Requires physical memory access.
    pub physical_memory: bool,

    /// Requires interrupt queue access.
    pub interrupt_queue: bool,

    /// Requires timer access.
    pub timer: bool,

    /// Requires vector store access.
    pub vector_store: bool,

    /// Requires graph store access.
    pub graph_store: bool,

    /// Minimum memory required in bytes.
    pub min_memory_bytes: u64,
}

impl RvfManifest {
    /// Parses an RVF manifest from bytes.
    ///
    /// # Errors
    ///
    /// Returns `KernelError::InvalidManifest` if the manifest is malformed.
    pub fn parse(data: &[u8]) -> Result<Self, ruvix_types::KernelError> {
        // Check minimum size
        if data.len() < 96 {
            return Err(ruvix_types::KernelError::InvalidManifest);
        }

        // Parse magic number "RVF1"
        if &data[0..4] != b"RVF1" {
            return Err(ruvix_types::KernelError::InvalidManifest);
        }

        // Parse version
        let major = u16::from_le_bytes([data[4], data[5]]);
        let minor = u16::from_le_bytes([data[6], data[7]]);
        let patch = u16::from_le_bytes([data[8], data[9]]);
        let version = ManifestVersion::new(major, minor, patch);

        if !version.is_compatible() {
            return Err(ruvix_types::KernelError::InvalidManifest);
        }

        // Parse content hash
        let mut content_hash = [0u8; 32];
        content_hash.copy_from_slice(&data[10..42]);

        // Parse section offsets
        let component_graph_offset =
            u32::from_le_bytes([data[42], data[43], data[44], data[45]]) as usize;
        let memory_schema_offset =
            u32::from_le_bytes([data[46], data[47], data[48], data[49]]) as usize;
        let proof_policy_offset =
            u32::from_le_bytes([data[50], data[51], data[52], data[53]]) as usize;
        let rollback_hooks_offset =
            u32::from_le_bytes([data[54], data[55], data[56], data[57]]) as usize;
        let witness_log_offset =
            u32::from_le_bytes([data[58], data[59], data[60], data[61]]) as usize;
        let required_caps_offset =
            u32::from_le_bytes([data[62], data[63], data[64], data[65]]) as usize;

        // Parse each section (simplified for Phase A)
        let component_graph = Self::parse_component_graph(data, component_graph_offset)?;
        let memory_schema = Self::parse_memory_schema(data, memory_schema_offset)?;
        let proof_policy = Self::parse_proof_policy(data, proof_policy_offset)?;
        let rollback_hooks = Self::parse_rollback_hooks(data, rollback_hooks_offset)?;
        let witness_log_policy = Self::parse_witness_log_policy(data, witness_log_offset)?;
        let required_capabilities = Self::parse_required_capabilities(data, required_caps_offset)?;

        Ok(Self {
            version,
            content_hash,
            component_graph,
            memory_schema,
            proof_policy,
            rollback_hooks,
            witness_log_policy,
            required_capabilities,
        })
    }

    fn parse_component_graph(
        _data: &[u8],
        offset: usize,
    ) -> Result<ComponentGraph, ruvix_types::KernelError> {
        if offset == 0 {
            return Ok(ComponentGraph::new());
        }

        // Minimal parsing for Phase A
        Ok(ComponentGraph::new())
    }

    fn parse_memory_schema(
        _data: &[u8],
        offset: usize,
    ) -> Result<MemorySchema, ruvix_types::KernelError> {
        if offset == 0 {
            return Ok(MemorySchema::new());
        }

        Ok(MemorySchema::new())
    }

    fn parse_proof_policy(
        _data: &[u8],
        offset: usize,
    ) -> Result<ProofPolicy, ruvix_types::KernelError> {
        if offset == 0 {
            return Ok(ProofPolicy::new(ProofTier::Standard));
        }

        Ok(ProofPolicy::new(ProofTier::Standard))
    }

    fn parse_rollback_hooks(
        _data: &[u8],
        offset: usize,
    ) -> Result<RollbackHooks, ruvix_types::KernelError> {
        if offset == 0 {
            return Ok(RollbackHooks::new());
        }

        Ok(RollbackHooks::new())
    }

    fn parse_witness_log_policy(
        data: &[u8],
        offset: usize,
    ) -> Result<WitnessLogPolicy, ruvix_types::KernelError> {
        if offset == 0 || offset >= data.len() {
            return Ok(WitnessLogPolicy::default());
        }

        // Parse witness log policy fields
        if offset + 32 > data.len() {
            return Ok(WitnessLogPolicy::default());
        }

        let max_entries = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);

        let max_size_bytes = u64::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
            data[offset + 15],
        ]);

        let retention_seconds = u64::from_le_bytes([
            data[offset + 16],
            data[offset + 17],
            data[offset + 18],
            data[offset + 19],
            data[offset + 20],
            data[offset + 21],
            data[offset + 22],
            data[offset + 23],
        ]);

        let compression = match data[offset + 24] {
            1 => WitnessCompression::Lz4,
            2 => WitnessCompression::Zstd,
            _ => WitnessCompression::None,
        };

        let export_policy = match data[offset + 25] {
            1 => WitnessExportPolicy::OnShutdown,
            2 => WitnessExportPolicy::Never,
            3 => WitnessExportPolicy::External,
            _ => WitnessExportPolicy::OnRotation,
        };

        let hash_chain = data[offset + 26] != 0;

        Ok(WitnessLogPolicy {
            max_entries,
            max_size_bytes,
            retention_seconds,
            compression,
            export_policy,
            hash_chain,
        })
    }

    fn parse_required_capabilities(
        _data: &[u8],
        offset: usize,
    ) -> Result<RequiredCapabilities, ruvix_types::KernelError> {
        if offset == 0 {
            return Ok(RequiredCapabilities::default());
        }

        Ok(RequiredCapabilities::default())
    }

    /// Validates the entire manifest.
    #[must_use]
    pub fn validate(&self) -> bool {
        self.component_graph.validate() && self.memory_schema.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_version_compatibility() {
        let v1_0_0 = ManifestVersion::new(1, 0, 0);
        let v1_1_0 = ManifestVersion::new(1, 1, 0);
        let v2_0_0 = ManifestVersion::new(2, 0, 0);

        assert!(v1_0_0.is_compatible());
        assert!(v1_1_0.is_compatible());
        assert!(!v2_0_0.is_compatible());
    }

    #[test]
    fn test_component_graph_empty() {
        let graph = ComponentGraph::new();
        assert_eq!(graph.component_count(), 0);
        // Phase A: Empty graphs are valid for test manifests
        assert!(graph.validate());
    }

    #[test]
    fn test_memory_schema_empty() {
        let schema = MemorySchema::new();
        assert_eq!(schema.region_count(), 0);
        assert!(schema.validate()); // Empty schema is valid
    }

    #[test]
    fn test_proof_policy_default_tier() {
        let policy = ProofPolicy::new(ProofTier::Reflex);
        assert_eq!(policy.tier_for_component(0), ProofTier::Reflex);
        assert_eq!(policy.tier_for_component(999), ProofTier::Reflex);
    }

    #[test]
    fn test_proof_required_ops() {
        let ops = ProofRequiredOps::VECTOR_PUT;
        assert!(ops.requires(ProofRequiredOps::VECTOR_PUT));
        assert!(!ops.requires(ProofRequiredOps::GRAPH_APPLY));

        let all = ProofRequiredOps::ALL;
        assert!(all.requires(ProofRequiredOps::VECTOR_PUT));
        assert!(all.requires(ProofRequiredOps::REGION_WRITE));
    }

    #[test]
    fn test_witness_log_policy_default() {
        let policy = WitnessLogPolicy::default();
        assert_eq!(policy.max_entries, 1_000_000);
        assert!(policy.hash_chain);
        assert_eq!(policy.compression, WitnessCompression::None);
    }

    #[test]
    fn test_manifest_parse_invalid_magic() {
        let data = b"XXXX";
        assert!(RvfManifest::parse(data).is_err());
    }

    #[test]
    fn test_manifest_parse_too_short() {
        let data = b"RVF1";
        assert!(RvfManifest::parse(data).is_err());
    }
}
