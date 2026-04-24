//! RVF Bridge — ADR-106 Layer 1 shared wire types adapter for rvAgent.
//!
//! This module bridges the rvAgent framework with the RVF (RuVector Format) type system,
//! implementing the shared-types architecture specified in ADR-106. It re-exports canonical
//! RVF wire types and provides conversion utilities between rvAgent's runtime types and
//! RVF's wire-format types.
//!
//! # Feature gating
//!
//! This module is always available, but the `rvf-compat` feature enables direct
//! re-exports from `rvf-types` instead of local definitions. Without the feature,
//! compatible local types are used.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Mount types (ADR-106 Layer 1 — mirrors ruvix-types::rvf)
// ---------------------------------------------------------------------------

/// Handle to a mounted RVF package within the agent runtime.
///
/// Maps to `ruvix-types::rvf::RvfMountHandle` and will be unified with
/// `rvf-types::mount::RvfMountHandle` when that module is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct RvfMountHandle {
    /// Slot index in the mount table.
    pub id: u32,
    /// Generation counter for ABA protection.
    pub generation: u32,
}

impl RvfMountHandle {
    /// Create a new mount handle.
    #[inline]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self { id, generation }
    }

    /// Create a null (invalid) handle.
    #[inline]
    pub const fn null() -> Self {
        Self {
            id: 0,
            generation: 0,
        }
    }

    /// Check if this handle is null.
    #[inline]
    pub const fn is_null(&self) -> bool {
        self.id == 0 && self.generation == 0
    }
}

impl Default for RvfMountHandle {
    fn default() -> Self {
        Self::null()
    }
}

/// Identifier for a component within a mounted RVF package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct RvfComponentId {
    /// The mount handle of the containing package.
    pub mount: RvfMountHandle,
    /// Component index within the package (0-based).
    pub component_index: u32,
}

impl RvfComponentId {
    /// Create a new component ID.
    #[inline]
    pub const fn new(mount: RvfMountHandle, component_index: u32) -> Self {
        Self {
            mount,
            component_index,
        }
    }

    /// Create a component ID for the root component (index 0).
    #[inline]
    pub const fn root(mount: RvfMountHandle) -> Self {
        Self {
            mount,
            component_index: 0,
        }
    }
}

/// RVF package verification status.
///
/// Mirrors `ruvix-types::rvf::RvfVerifyStatus` and `rvf-types` verification codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RvfVerifyStatus {
    /// Package signature is valid (ML-DSA-65 verified).
    SignatureValid = 0,
    /// Signature verification failed.
    SignatureInvalid = 1,
    /// Package manifest is malformed.
    ManifestInvalid = 2,
    /// Required component is missing.
    ComponentMissing = 3,
    /// Proof policy cannot be satisfied.
    ProofPolicyInvalid = 4,
    /// Package requires capabilities not available.
    CapabilitiesInsufficient = 5,
}

impl RvfVerifyStatus {
    /// Returns true if the package is valid for mounting.
    #[inline]
    pub const fn is_valid(&self) -> bool {
        matches!(self, Self::SignatureValid)
    }

    /// Returns a human-readable description.
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SignatureValid => "Signature valid",
            Self::SignatureInvalid => "Signature invalid",
            Self::ManifestInvalid => "Manifest invalid",
            Self::ComponentMissing => "Component missing",
            Self::ProofPolicyInvalid => "Proof policy invalid",
            Self::CapabilitiesInsufficient => "Capabilities insufficient",
        }
    }
}

impl TryFrom<u8> for RvfVerifyStatus {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::SignatureValid),
            1 => Ok(Self::SignatureInvalid),
            2 => Ok(Self::ManifestInvalid),
            3 => Ok(Self::ComponentMissing),
            4 => Ok(Self::ProofPolicyInvalid),
            5 => Ok(Self::CapabilitiesInsufficient),
            other => Err(other),
        }
    }
}

/// WIT (WASM Interface Types) type identifier for message schema validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct WitTypeId(pub u32);

impl WitTypeId {
    /// No schema (raw bytes).
    pub const NONE: Self = Self(0);

    /// Create a new WIT type ID.
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns true if this is the NONE type.
    #[inline]
    pub const fn is_none(&self) -> bool {
        self.0 == 0
    }
}

impl Default for WitTypeId {
    fn default() -> Self {
        Self::NONE
    }
}

// ---------------------------------------------------------------------------
// Witness types (ADR-106 Phase 4 — unified witness format)
// ---------------------------------------------------------------------------

/// Task execution outcome — matches `rvf-types::witness::TaskOutcome`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TaskOutcome {
    /// Task completed with passing tests.
    Solved = 0,
    /// Task attempted but tests fail.
    Failed = 1,
    /// Task skipped (precondition not met).
    Skipped = 2,
    /// Task errored (infrastructure failure).
    Errored = 3,
}

impl TryFrom<u8> for TaskOutcome {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Solved),
            1 => Ok(Self::Failed),
            2 => Ok(Self::Skipped),
            3 => Ok(Self::Errored),
            other => Err(other),
        }
    }
}

/// Governance mode — matches `rvf-types::witness::GovernanceMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum GovernanceMode {
    /// Read-only plus suggestions.
    Restricted = 0,
    /// Writes allowed with human confirmation gates.
    Approved = 1,
    /// Bounded authority with automatic rollback.
    Autonomous = 2,
}

impl Default for GovernanceMode {
    fn default() -> Self {
        Self::Approved
    }
}

impl TryFrom<u8> for GovernanceMode {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Restricted),
            1 => Ok(Self::Approved),
            2 => Ok(Self::Autonomous),
            other => Err(other),
        }
    }
}

/// Policy check result — matches `rvf-types::witness::PolicyCheck`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PolicyCheck {
    /// Tool call allowed by policy.
    Allowed = 0,
    /// Tool call denied by policy.
    Denied = 1,
    /// Tool call required human confirmation.
    Confirmed = 2,
}

impl Default for PolicyCheck {
    fn default() -> Self {
        Self::Allowed
    }
}

impl TryFrom<u8> for PolicyCheck {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Allowed),
            1 => Ok(Self::Denied),
            2 => Ok(Self::Confirmed),
            other => Err(other),
        }
    }
}

/// Witness header constants matching `rvf-types::witness`.
pub const WITNESS_MAGIC: u32 = 0x5257_5657; // "RVWW"
pub const WITNESS_HEADER_SIZE: usize = 64;

/// Flags for witness bundle.
pub const WIT_SIGNED: u16 = 0x0001;
pub const WIT_HAS_SPEC: u16 = 0x0002;
pub const WIT_HAS_PLAN: u16 = 0x0004;
pub const WIT_HAS_TRACE: u16 = 0x0008;
pub const WIT_HAS_DIFF: u16 = 0x0010;
pub const WIT_HAS_TEST_LOG: u16 = 0x0020;

/// A tool call record in RVF witness wire format.
///
/// Compatible with `rvf-types::witness::ToolCallEntry` for serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RvfToolCallEntry {
    /// Tool name / action.
    pub action: String,
    /// SHA-256 of arguments, truncated to 8 bytes.
    pub args_hash: [u8; 8],
    /// SHA-256 of result, truncated to 8 bytes.
    pub result_hash: [u8; 8],
    /// Wall-clock latency in milliseconds.
    pub latency_ms: u32,
    /// Cost in microdollars.
    pub cost_microdollars: u32,
    /// Tokens consumed.
    pub tokens: u32,
    /// Policy check result.
    pub policy_check: PolicyCheck,
}

/// Witness bundle header — RVF wire-format compatible.
///
/// This is the rvAgent-side representation that can be serialized to
/// match `rvf-types::witness::WitnessHeader`'s 64-byte wire format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RvfWitnessHeader {
    /// Format version (currently 1).
    pub version: u16,
    /// Bitfield flags.
    pub flags: u16,
    /// Unique task identifier (UUID bytes).
    pub task_id: [u8; 16],
    /// SHA-256 of the policy, truncated to 8 bytes.
    pub policy_hash: [u8; 8],
    /// Creation timestamp (nanoseconds since UNIX epoch).
    pub created_ns: u64,
    /// Task outcome.
    pub outcome: TaskOutcome,
    /// Governance mode.
    pub governance_mode: GovernanceMode,
    /// Number of tool calls recorded.
    pub tool_call_count: u16,
    /// Total cost in microdollars.
    pub total_cost_microdollars: u32,
    /// Total wall-clock latency in milliseconds.
    pub total_latency_ms: u32,
    /// Total tokens consumed.
    pub total_tokens: u32,
    /// Number of retries.
    pub retry_count: u16,
    /// Number of TLV sections.
    pub section_count: u16,
    /// Total bundle size.
    pub total_bundle_size: u32,
}

impl RvfWitnessHeader {
    /// Serialize to a 64-byte wire-format array matching `rvf-types::WitnessHeader`.
    pub fn to_bytes(&self) -> [u8; WITNESS_HEADER_SIZE] {
        let mut buf = [0u8; WITNESS_HEADER_SIZE];
        buf[0..4].copy_from_slice(&WITNESS_MAGIC.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..24].copy_from_slice(&self.task_id);
        buf[24..32].copy_from_slice(&self.policy_hash);
        buf[32..40].copy_from_slice(&self.created_ns.to_le_bytes());
        buf[40] = self.outcome as u8;
        buf[41] = self.governance_mode as u8;
        buf[42..44].copy_from_slice(&self.tool_call_count.to_le_bytes());
        buf[44..48].copy_from_slice(&self.total_cost_microdollars.to_le_bytes());
        buf[48..52].copy_from_slice(&self.total_latency_ms.to_le_bytes());
        buf[52..56].copy_from_slice(&self.total_tokens.to_le_bytes());
        buf[56..58].copy_from_slice(&self.retry_count.to_le_bytes());
        buf[58..60].copy_from_slice(&self.section_count.to_le_bytes());
        buf[60..64].copy_from_slice(&self.total_bundle_size.to_le_bytes());
        buf
    }

    /// Deserialize from a 64-byte wire-format array.
    pub fn from_bytes(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < WITNESS_HEADER_SIZE {
            return Err("data too short for witness header");
        }
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != WITNESS_MAGIC {
            return Err("invalid witness magic bytes");
        }
        let mut task_id = [0u8; 16];
        task_id.copy_from_slice(&data[8..24]);
        let mut policy_hash = [0u8; 8];
        policy_hash.copy_from_slice(&data[24..32]);

        Ok(Self {
            version: u16::from_le_bytes([data[4], data[5]]),
            flags: u16::from_le_bytes([data[6], data[7]]),
            task_id,
            policy_hash,
            created_ns: u64::from_le_bytes([
                data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
            ]),
            outcome: TaskOutcome::try_from(data[40]).map_err(|_| "invalid outcome")?,
            governance_mode: GovernanceMode::try_from(data[41])
                .map_err(|_| "invalid governance mode")?,
            tool_call_count: u16::from_le_bytes([data[42], data[43]]),
            total_cost_microdollars: u32::from_le_bytes([data[44], data[45], data[46], data[47]]),
            total_latency_ms: u32::from_le_bytes([data[48], data[49], data[50], data[51]]),
            total_tokens: u32::from_le_bytes([data[52], data[53], data[54], data[55]]),
            retry_count: u16::from_le_bytes([data[56], data[57]]),
            section_count: u16::from_le_bytes([data[58], data[59]]),
            total_bundle_size: u32::from_le_bytes([data[60], data[61], data[62], data[63]]),
        })
    }
}

// ---------------------------------------------------------------------------
// Manifest types (ADR-106 Layer 2)
// ---------------------------------------------------------------------------

/// Parsed RVF manifest entry describing a tool or skill.
///
/// When `rvf-manifest` is available (via `rvf-compat` feature), this will
/// delegate to `rvf-manifest::ManifestEntry`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfManifestEntry {
    /// Entry name (tool or skill identifier).
    pub name: String,
    /// Entry type.
    pub entry_type: RvfManifestEntryType,
    /// Human-readable description.
    pub description: String,
    /// Version string (semver).
    pub version: String,
    /// JSON schema for parameters (if applicable).
    #[serde(default)]
    pub parameters_schema: Option<serde_json::Value>,
    /// SHA-256 hash of the entry's content.
    #[serde(default)]
    pub content_hash: Option<String>,
    /// Required capabilities for this entry.
    #[serde(default)]
    pub required_capabilities: Vec<String>,
}

/// Type of entry in an RVF manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RvfManifestEntryType {
    /// A tool that can be invoked by the agent.
    Tool,
    /// A skill (prompt template + tool set).
    Skill,
    /// A WASM component.
    WasmComponent,
    /// A data segment (vectors, embeddings).
    DataSegment,
    /// A middleware plugin.
    Middleware,
}

/// Parsed RVF manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfManifest {
    /// Manifest format version.
    pub version: u16,
    /// Package name.
    pub name: String,
    /// Package version.
    pub package_version: String,
    /// Entries in this manifest.
    pub entries: Vec<RvfManifestEntry>,
    /// Signature algorithm used (e.g., "ML-DSA-65").
    #[serde(default)]
    pub signature_algo: Option<String>,
    /// Package-level metadata.
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl RvfManifest {
    /// Create an empty manifest.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            version: 1,
            name: name.into(),
            package_version: version.into(),
            entries: Vec::new(),
            signature_algo: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Get all tool entries.
    pub fn tools(&self) -> Vec<&RvfManifestEntry> {
        self.entries
            .iter()
            .filter(|e| e.entry_type == RvfManifestEntryType::Tool)
            .collect()
    }

    /// Get all skill entries.
    pub fn skills(&self) -> Vec<&RvfManifestEntry> {
        self.entries
            .iter()
            .filter(|e| e.entry_type == RvfManifestEntryType::Skill)
            .collect()
    }

    /// Get all WASM component entries.
    pub fn wasm_components(&self) -> Vec<&RvfManifestEntry> {
        self.entries
            .iter()
            .filter(|e| e.entry_type == RvfManifestEntryType::WasmComponent)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RVF Bridge Configuration
// ---------------------------------------------------------------------------

/// Configuration for the RVF bridge in rvAgent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfBridgeConfig {
    /// Whether RVF integration is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Path to the RVF package directory.
    #[serde(default)]
    pub package_dir: Option<String>,
    /// Whether to verify package signatures.
    #[serde(default = "default_true")]
    pub verify_signatures: bool,
    /// Whether to produce RVF wire-format witness bundles.
    #[serde(default)]
    pub rvf_witness: bool,
    /// Governance mode for the agent.
    #[serde(default)]
    pub governance_mode: GovernanceMode,
}

fn default_true() -> bool {
    true
}

impl Default for RvfBridgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            package_dir: None,
            verify_signatures: true,
            rvf_witness: false,
            governance_mode: GovernanceMode::Approved,
        }
    }
}

// ---------------------------------------------------------------------------
// Mount table
// ---------------------------------------------------------------------------

/// An entry in the agent's RVF mount table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountTableEntry {
    /// Mount handle for this package.
    pub handle: RvfMountHandle,
    /// Package name from the manifest.
    pub package_name: String,
    /// Package version.
    pub package_version: String,
    /// Verification status.
    pub verify_status: RvfVerifyStatus,
    /// Parsed manifest.
    pub manifest: RvfManifest,
    /// Timestamp when mounted (millis since UNIX epoch).
    pub mounted_at_ms: u64,
}

/// The agent's mount table — tracks all mounted RVF packages.
///
/// Uses a `HashMap` index by package name for O(1) lookups by name,
/// and a `HashMap` by handle ID for O(1) lookups by handle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MountTable {
    entries: Vec<MountTableEntry>,
    /// Index: handle.id → position in `entries` vec.
    #[serde(skip)]
    handle_index: std::collections::HashMap<u32, usize>,
    /// Index: package_name → position in `entries` vec.
    #[serde(skip)]
    name_index: std::collections::HashMap<String, usize>,
    next_id: u32,
    generation: u32,
}

impl MountTable {
    /// Create a new empty mount table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild indices after deserialization.
    fn rebuild_indices(&mut self) {
        self.handle_index.clear();
        self.name_index.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            self.handle_index.insert(entry.handle.id, i);
            self.name_index.insert(entry.package_name.clone(), i);
        }
    }

    /// Mount an RVF package and return its handle.
    pub fn mount(
        &mut self,
        manifest: RvfManifest,
        verify_status: RvfVerifyStatus,
    ) -> RvfMountHandle {
        self.next_id += 1;
        self.generation += 1;
        let handle = RvfMountHandle::new(self.next_id, self.generation);
        let pkg_name = manifest.name.clone();
        let entry = MountTableEntry {
            handle,
            package_name: pkg_name.clone(),
            package_version: manifest.package_version.clone(),
            verify_status,
            manifest,
            mounted_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        let idx = self.entries.len();
        self.entries.push(entry);
        self.handle_index.insert(handle.id, idx);
        self.name_index.insert(pkg_name, idx);
        handle
    }

    /// Unmount a package by handle.
    pub fn unmount(&mut self, handle: RvfMountHandle) -> bool {
        let len = self.entries.len();
        self.entries.retain(|e| e.handle != handle);
        if self.entries.len() < len {
            // Rebuild indices after removal (compact operation)
            self.rebuild_indices();
            true
        } else {
            false
        }
    }

    /// Look up a mounted package by handle (O(1) via index).
    pub fn get(&self, handle: RvfMountHandle) -> Option<&MountTableEntry> {
        self.handle_index
            .get(&handle.id)
            .and_then(|&idx| self.entries.get(idx))
            .filter(|e| e.handle == handle) // Generation check
    }

    /// Look up a mounted package by name (O(1) via index).
    pub fn get_by_name(&self, name: &str) -> Option<&MountTableEntry> {
        self.name_index
            .get(name)
            .and_then(|&idx| self.entries.get(idx))
    }

    /// List all mounted packages.
    pub fn list(&self) -> &[MountTableEntry] {
        &self.entries
    }

    /// Collect all tools from all mounted packages.
    ///
    /// Avoids the intermediate `Vec` allocation from `manifest.tools()`
    /// by directly filtering entries inline.
    pub fn all_tools(&self) -> Vec<(&RvfMountHandle, &RvfManifestEntry)> {
        self.entries
            .iter()
            .flat_map(|entry| {
                entry
                    .manifest
                    .entries
                    .iter()
                    .filter(|e| e.entry_type == RvfManifestEntryType::Tool)
                    .map(move |tool| (&entry.handle, tool))
            })
            .collect()
    }

    /// Number of mounted packages.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no packages are mounted.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mount_handle() {
        let h = RvfMountHandle::new(1, 2);
        assert!(!h.is_null());
        assert_eq!(h.id, 1);
        assert_eq!(h.generation, 2);

        let null = RvfMountHandle::null();
        assert!(null.is_null());
        assert_eq!(null, RvfMountHandle::default());
    }

    #[test]
    fn test_component_id() {
        let mount = RvfMountHandle::new(1, 0);
        let comp = RvfComponentId::new(mount, 3);
        assert_eq!(comp.component_index, 3);

        let root = RvfComponentId::root(mount);
        assert_eq!(root.component_index, 0);
    }

    #[test]
    fn test_verify_status() {
        assert!(RvfVerifyStatus::SignatureValid.is_valid());
        assert!(!RvfVerifyStatus::SignatureInvalid.is_valid());
        assert!(!RvfVerifyStatus::ManifestInvalid.is_valid());
        assert_eq!(RvfVerifyStatus::SignatureValid.as_str(), "Signature valid");

        assert_eq!(
            RvfVerifyStatus::try_from(0),
            Ok(RvfVerifyStatus::SignatureValid)
        );
        assert_eq!(
            RvfVerifyStatus::try_from(5),
            Ok(RvfVerifyStatus::CapabilitiesInsufficient)
        );
        assert!(RvfVerifyStatus::try_from(6).is_err());
    }

    #[test]
    fn test_wit_type_id() {
        assert!(WitTypeId::NONE.is_none());
        assert!(!WitTypeId::new(42).is_none());
        assert_eq!(WitTypeId::default(), WitTypeId::NONE);
    }

    #[test]
    fn test_task_outcome_roundtrip() {
        for raw in 0..=3u8 {
            let o = TaskOutcome::try_from(raw).unwrap();
            assert_eq!(o as u8, raw);
        }
        assert!(TaskOutcome::try_from(4).is_err());
    }

    #[test]
    fn test_governance_mode_roundtrip() {
        for raw in 0..=2u8 {
            let g = GovernanceMode::try_from(raw).unwrap();
            assert_eq!(g as u8, raw);
        }
        assert!(GovernanceMode::try_from(3).is_err());
    }

    #[test]
    fn test_policy_check_roundtrip() {
        for raw in 0..=2u8 {
            let p = PolicyCheck::try_from(raw).unwrap();
            assert_eq!(p as u8, raw);
        }
        assert!(PolicyCheck::try_from(3).is_err());
    }

    #[test]
    fn test_witness_header_roundtrip() {
        let hdr = RvfWitnessHeader {
            version: 1,
            flags: WIT_SIGNED | WIT_HAS_TRACE,
            task_id: [0x42; 16],
            policy_hash: [0xAA; 8],
            created_ns: 1_700_000_000_000_000_000,
            outcome: TaskOutcome::Solved,
            governance_mode: GovernanceMode::Approved,
            tool_call_count: 5,
            total_cost_microdollars: 15_000,
            total_latency_ms: 4_500,
            total_tokens: 8_000,
            retry_count: 1,
            section_count: 2,
            total_bundle_size: 1024,
        };
        let bytes = hdr.to_bytes();
        assert_eq!(bytes.len(), WITNESS_HEADER_SIZE);

        // Verify magic
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, WITNESS_MAGIC);

        let decoded = RvfWitnessHeader::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, hdr);
    }

    #[test]
    fn test_witness_header_bad_magic() {
        let mut bytes = [0u8; WITNESS_HEADER_SIZE];
        bytes[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        assert!(RvfWitnessHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_witness_header_too_short() {
        assert!(RvfWitnessHeader::from_bytes(&[0u8; 32]).is_err());
    }

    #[test]
    fn test_manifest_creation() {
        let mut manifest = RvfManifest::new("test-pkg", "0.1.0");
        manifest.entries.push(RvfManifestEntry {
            name: "read_file".into(),
            entry_type: RvfManifestEntryType::Tool,
            description: "Read a file".into(),
            version: "0.1.0".into(),
            parameters_schema: Some(serde_json::json!({"type": "object"})),
            content_hash: None,
            required_capabilities: vec![],
        });
        manifest.entries.push(RvfManifestEntry {
            name: "deploy".into(),
            entry_type: RvfManifestEntryType::Skill,
            description: "Deploy the app".into(),
            version: "0.1.0".into(),
            parameters_schema: None,
            content_hash: None,
            required_capabilities: vec!["execute".into()],
        });
        manifest.entries.push(RvfManifestEntry {
            name: "processor".into(),
            entry_type: RvfManifestEntryType::WasmComponent,
            description: "Data processor".into(),
            version: "0.1.0".into(),
            parameters_schema: None,
            content_hash: Some("abc123".into()),
            required_capabilities: vec![],
        });

        assert_eq!(manifest.tools().len(), 1);
        assert_eq!(manifest.skills().len(), 1);
        assert_eq!(manifest.wasm_components().len(), 1);
    }

    #[test]
    fn test_manifest_serde_roundtrip() {
        let manifest = RvfManifest::new("test", "1.0.0");
        let json = serde_json::to_string(&manifest).unwrap();
        let back: RvfManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "test");
        assert_eq!(back.package_version, "1.0.0");
    }

    #[test]
    fn test_bridge_config_defaults() {
        let cfg = RvfBridgeConfig::default();
        assert!(!cfg.enabled);
        assert!(cfg.verify_signatures);
        assert!(!cfg.rvf_witness);
        assert_eq!(cfg.governance_mode, GovernanceMode::Approved);
    }

    #[test]
    fn test_mount_table() {
        let mut table = MountTable::new();
        assert!(table.is_empty());

        let manifest = RvfManifest::new("pkg-a", "0.1.0");
        let handle = table.mount(manifest, RvfVerifyStatus::SignatureValid);
        assert!(!handle.is_null());
        assert_eq!(table.len(), 1);

        let entry = table.get(handle).unwrap();
        assert_eq!(entry.package_name, "pkg-a");
        assert_eq!(entry.verify_status, RvfVerifyStatus::SignatureValid);

        // Second mount
        let manifest2 = RvfManifest::new("pkg-b", "0.2.0");
        let handle2 = table.mount(manifest2, RvfVerifyStatus::SignatureValid);
        assert_ne!(handle, handle2);
        assert_eq!(table.len(), 2);

        // Unmount first
        assert!(table.unmount(handle));
        assert_eq!(table.len(), 1);
        assert!(table.get(handle).is_none());
        assert!(table.get(handle2).is_some());
    }

    #[test]
    fn test_mount_table_all_tools() {
        let mut table = MountTable::new();
        let mut manifest = RvfManifest::new("tools-pkg", "0.1.0");
        manifest.entries.push(RvfManifestEntry {
            name: "tool_a".into(),
            entry_type: RvfManifestEntryType::Tool,
            description: "Tool A".into(),
            version: "0.1.0".into(),
            parameters_schema: None,
            content_hash: None,
            required_capabilities: vec![],
        });
        manifest.entries.push(RvfManifestEntry {
            name: "tool_b".into(),
            entry_type: RvfManifestEntryType::Tool,
            description: "Tool B".into(),
            version: "0.1.0".into(),
            parameters_schema: None,
            content_hash: None,
            required_capabilities: vec![],
        });
        table.mount(manifest, RvfVerifyStatus::SignatureValid);

        let tools = table.all_tools();
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_tool_call_entry_serde() {
        let entry = RvfToolCallEntry {
            action: "read_file".into(),
            args_hash: [0x11; 8],
            result_hash: [0x22; 8],
            latency_ms: 150,
            cost_microdollars: 500,
            tokens: 200,
            policy_check: PolicyCheck::Allowed,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: RvfToolCallEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(entry, back);
    }
}
