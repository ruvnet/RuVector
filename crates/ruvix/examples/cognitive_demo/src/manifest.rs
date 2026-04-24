//! RVF Manifest definition for the cognitive demo package.
//!
//! This module defines the complete manifest structure for the cognitive_demo.rvf
//! package, demonstrating all ADR-087 Section 9.2 manifest features.

use crate::config;
use ruvix_types::{ProofTier, RegionPolicy, WitTypeId};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::{string::String, vec, vec::Vec};

#[cfg(feature = "std")]
use std::{string::String, vec, vec::Vec};

/// Demo manifest structure matching ADR-087 Section 9.2.
///
/// This manifest describes the complete cognitive pipeline RVF package.
#[derive(Debug, Clone)]
pub struct DemoManifest {
    /// Manifest version.
    pub version: ManifestVersion,

    /// Package identifier.
    pub package_id: String,

    /// Package content hash (SHA-256).
    pub content_hash: [u8; 32],

    /// Component graph - 5 components demonstrating the pipeline.
    pub components: Vec<DemoComponent>,

    /// Memory schema - all 3 region types.
    pub regions: Vec<DemoRegion>,

    /// Proof policy - all 3 tiers.
    pub proof_policy: ProofPolicyConfig,

    /// Rollback hooks for state recovery.
    pub rollback_hooks: Vec<RollbackHookConfig>,

    /// Witness log configuration.
    pub witness_log: WitnessLogConfig,
}

/// Manifest format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestVersion {
    /// Major version.
    pub major: u16,
    /// Minor version.
    pub minor: u16,
    /// Patch version.
    pub patch: u16,
}

impl ManifestVersion {
    /// Current manifest version.
    pub const CURRENT: Self = Self {
        major: 1,
        minor: 0,
        patch: 0,
    };

    /// Creates a new version.
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl Default for ManifestVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

/// Component declaration in the manifest.
#[derive(Debug, Clone)]
pub struct DemoComponent {
    /// Component index (0-4).
    pub index: u32,

    /// Component name.
    pub name: String,

    /// Component type.
    pub component_type: ComponentType,

    /// Entry point function name.
    pub entry_point: String,

    /// Syscalls this component uses.
    pub syscalls: Vec<SyscallUsage>,

    /// Dependencies (other component indices).
    pub dependencies: Vec<u32>,

    /// Required capabilities.
    pub required_caps: Vec<CapabilityRequirement>,

    /// WIT type for interface.
    pub wit_type: WitTypeId,
}

/// Component types in the cognitive pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ComponentType {
    /// SensorAdapter - sensor_subscribe, queue_send.
    SensorAdapter = 0,

    /// FeatureExtractor - queue_recv, vector_put_proved.
    FeatureExtractor = 1,

    /// ReasoningEngine - vector_get, graph_apply_proved.
    ReasoningEngine = 2,

    /// Attestor - attest_emit.
    Attestor = 3,

    /// Coordinator - task_spawn, cap_grant, timer_wait.
    Coordinator = 4,
}

impl ComponentType {
    /// Returns the component name.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::SensorAdapter => "SensorAdapter",
            Self::FeatureExtractor => "FeatureExtractor",
            Self::ReasoningEngine => "ReasoningEngine",
            Self::Attestor => "Attestor",
            Self::Coordinator => "Coordinator",
        }
    }

    /// Returns the entry point name.
    pub const fn entry_point(&self) -> &'static str {
        match self {
            Self::SensorAdapter => "run_sensor_adapter",
            Self::FeatureExtractor => "run_feature_extractor",
            Self::ReasoningEngine => "run_reasoning_engine",
            Self::Attestor => "run_attestor",
            Self::Coordinator => "run_coordinator",
        }
    }
}

/// Syscall usage declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyscallUsage {
    /// Syscall number (0-11).
    pub syscall_number: u8,

    /// Syscall name for documentation.
    pub syscall_name: &'static str,

    /// Expected call count in full pipeline.
    pub expected_count: u32,
}

impl SyscallUsage {
    /// Creates a new syscall usage.
    pub const fn new(syscall_number: u8, syscall_name: &'static str, expected_count: u32) -> Self {
        Self {
            syscall_number,
            syscall_name,
            expected_count,
        }
    }
}

/// Capability requirement for a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityRequirement {
    /// Capability type.
    pub cap_type: CapType,

    /// Required rights.
    pub rights: u32,

    /// Target resource (region index, queue index, etc.).
    pub target: u32,
}

/// Capability types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CapType {
    /// Region access capability.
    Region = 0,

    /// Queue access capability.
    Queue = 1,

    /// Vector store access capability.
    VectorStore = 2,

    /// Graph store access capability.
    GraphStore = 3,

    /// Timer capability.
    Timer = 4,

    /// Sensor subscription capability.
    Sensor = 5,
}

/// Memory region declaration in the manifest.
#[derive(Debug, Clone)]
pub struct DemoRegion {
    /// Region index.
    pub index: u32,

    /// Region name.
    pub name: String,

    /// Region type.
    pub region_type: DemoRegionType,

    /// Region policy.
    pub policy: RegionPolicy,

    /// Owning component.
    pub owner: ComponentType,

    /// Components with read access.
    pub read_access: Vec<ComponentType>,

    /// Components with write access.
    pub write_access: Vec<ComponentType>,
}

/// Region types in the demo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoRegionType {
    /// Immutable region for model weights.
    Immutable { size: usize },

    /// Append-only region for witness log.
    AppendOnly { max_size: usize },

    /// Slab region for vector store.
    Slab { slot_size: usize, slots: usize },
}

impl DemoRegionType {
    /// Converts to RegionPolicy.
    pub fn to_policy(&self) -> RegionPolicy {
        match self {
            Self::Immutable { .. } => RegionPolicy::Immutable,
            Self::AppendOnly { max_size } => RegionPolicy::AppendOnly {
                max_size: *max_size,
            },
            Self::Slab { slot_size, slots } => RegionPolicy::Slab {
                slot_size: *slot_size,
                slot_count: *slots,
            },
        }
    }

    /// Returns the total size in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Immutable { size } => *size,
            Self::AppendOnly { max_size } => *max_size,
            Self::Slab { slot_size, slots } => *slot_size * *slots,
        }
    }
}

/// Proof policy configuration.
#[derive(Debug, Clone)]
pub struct ProofPolicyConfig {
    /// Default proof tier.
    pub default_tier: ProofTier,

    /// Vector mutations tier.
    pub vector_mutations: ProofTier,

    /// Graph mutations tier.
    pub graph_mutations: ProofTier,

    /// Structural changes tier.
    pub structural_changes: ProofTier,

    /// Per-component tier overrides.
    pub component_tiers: Vec<ComponentProofTier>,

    /// Whether tier escalation is allowed.
    pub allow_escalation: bool,
}

/// Per-component proof tier configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentProofTier {
    /// Component type.
    pub component: ComponentType,

    /// Required proof tier.
    pub tier: ProofTier,
}

/// Rollback hook configuration.
#[derive(Debug, Clone)]
pub struct RollbackHookConfig {
    /// Hook name.
    pub name: String,

    /// Triggering event.
    pub trigger: RollbackTrigger,

    /// Function to call.
    pub function: String,

    /// Accessible regions during rollback.
    pub accessible_regions: Vec<u32>,

    /// Timeout in microseconds.
    pub timeout_us: u64,
}

/// Rollback trigger events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RollbackTrigger {
    /// Coherence score dropped below threshold.
    CoherenceDrop = 0,

    /// Proof verification failed.
    ProofFailure = 1,

    /// Checkpoint restore requested.
    CheckpointRestore = 2,

    /// Manual rollback requested.
    Manual = 3,
}

impl RollbackTrigger {
    /// Returns the trigger name.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::CoherenceDrop => "on_coherence_drop",
            Self::ProofFailure => "on_proof_failure",
            Self::CheckpointRestore => "on_checkpoint_restore",
            Self::Manual => "on_manual_rollback",
        }
    }
}

/// Witness log configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessLogConfig {
    /// Maximum entries.
    pub max_entries: u64,

    /// Maximum size in bytes.
    pub max_size_bytes: u64,

    /// Retention period in seconds (0 = forever).
    pub retention_seconds: u64,

    /// Enable hash chaining.
    pub hash_chain: bool,

    /// Export policy.
    pub export_on_rotation: bool,
}

impl Default for WitnessLogConfig {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            max_size_bytes: 64 * 1024,
            retention_seconds: 0,
            hash_chain: true,
            export_on_rotation: true,
        }
    }
}

impl DemoManifest {
    /// Creates the default cognitive demo manifest.
    #[must_use]
    pub fn cognitive_demo() -> Self {
        Self {
            version: ManifestVersion::CURRENT,
            package_id: String::from("cognitive_demo"),
            content_hash: [0u8; 32], // Computed during build
            components: Self::create_components(),
            regions: Self::create_regions(),
            proof_policy: Self::create_proof_policy(),
            rollback_hooks: Self::create_rollback_hooks(),
            witness_log: WitnessLogConfig::default(),
        }
    }

    fn create_components() -> Vec<DemoComponent> {
        vec![
            // Component 0: SensorAdapter
            DemoComponent {
                index: 0,
                name: String::from("SensorAdapter"),
                component_type: ComponentType::SensorAdapter,
                entry_point: String::from("run_sensor_adapter"),
                syscalls: vec![
                    SyscallUsage::new(11, "sensor_subscribe", 1),
                    SyscallUsage::new(3, "queue_send", config::FULL_PIPELINE_EVENTS as u32),
                ],
                dependencies: vec![],
                required_caps: vec![
                    CapabilityRequirement {
                        cap_type: CapType::Sensor,
                        rights: 0x01, // READ
                        target: 0,
                    },
                    CapabilityRequirement {
                        cap_type: CapType::Queue,
                        rights: 0x02, // WRITE
                        target: 0,
                    },
                ],
                wit_type: WitTypeId::new(1),
            },
            // Component 1: FeatureExtractor
            DemoComponent {
                index: 1,
                name: String::from("FeatureExtractor"),
                component_type: ComponentType::FeatureExtractor,
                entry_point: String::from("run_feature_extractor"),
                syscalls: vec![
                    SyscallUsage::new(4, "queue_recv", config::FULL_PIPELINE_EVENTS as u32),
                    SyscallUsage::new(9, "vector_put_proved", config::FULL_PIPELINE_EVENTS as u32),
                    SyscallUsage::new(3, "queue_send", config::FULL_PIPELINE_EVENTS as u32),
                ],
                dependencies: vec![0],
                required_caps: vec![
                    CapabilityRequirement {
                        cap_type: CapType::Queue,
                        rights: 0x01, // READ
                        target: 0,
                    },
                    CapabilityRequirement {
                        cap_type: CapType::VectorStore,
                        rights: 0x02, // WRITE
                        target: 0,
                    },
                    CapabilityRequirement {
                        cap_type: CapType::Queue,
                        rights: 0x02, // WRITE
                        target: 1,
                    },
                ],
                wit_type: WitTypeId::new(2),
            },
            // Component 2: ReasoningEngine
            DemoComponent {
                index: 2,
                name: String::from("ReasoningEngine"),
                component_type: ComponentType::ReasoningEngine,
                entry_point: String::from("run_reasoning_engine"),
                syscalls: vec![
                    SyscallUsage::new(4, "queue_recv", config::FULL_PIPELINE_EVENTS as u32),
                    SyscallUsage::new(8, "vector_get", config::FULL_PIPELINE_EVENTS as u32),
                    SyscallUsage::new(10, "graph_apply_proved", config::GRAPH_MUTATIONS as u32),
                ],
                dependencies: vec![1],
                required_caps: vec![
                    CapabilityRequirement {
                        cap_type: CapType::Queue,
                        rights: 0x01, // READ
                        target: 1,
                    },
                    CapabilityRequirement {
                        cap_type: CapType::VectorStore,
                        rights: 0x01, // READ
                        target: 0,
                    },
                    CapabilityRequirement {
                        cap_type: CapType::GraphStore,
                        rights: 0x03, // READ | WRITE
                        target: 0,
                    },
                ],
                wit_type: WitTypeId::new(3),
            },
            // Component 3: Attestor
            DemoComponent {
                index: 3,
                name: String::from("Attestor"),
                component_type: ComponentType::Attestor,
                entry_point: String::from("run_attestor"),
                syscalls: vec![SyscallUsage::new(
                    7,
                    "attest_emit",
                    config::FULL_PIPELINE_EVENTS as u32,
                )],
                dependencies: vec![0, 1, 2],
                required_caps: vec![CapabilityRequirement {
                    cap_type: CapType::Region,
                    rights: 0x02, // WRITE (append)
                    target: 1,    // witness_log region
                }],
                wit_type: WitTypeId::new(4),
            },
            // Component 4: Coordinator
            DemoComponent {
                index: 4,
                name: String::from("Coordinator"),
                component_type: ComponentType::Coordinator,
                entry_point: String::from("run_coordinator"),
                syscalls: vec![
                    SyscallUsage::new(0, "task_spawn", config::TASK_SPAWNS as u32),
                    SyscallUsage::new(1, "cap_grant", config::CAP_GRANTS as u32),
                    SyscallUsage::new(5, "timer_wait", config::TIMER_WAITS as u32),
                ],
                dependencies: vec![],
                required_caps: vec![CapabilityRequirement {
                    cap_type: CapType::Timer,
                    rights: 0x01, // READ
                    target: 0,
                }],
                wit_type: WitTypeId::new(5),
            },
        ]
    }

    fn create_regions() -> Vec<DemoRegion> {
        vec![
            // Region 0: Immutable - model_weights (1 MiB)
            DemoRegion {
                index: 0,
                name: String::from("model_weights"),
                region_type: DemoRegionType::Immutable {
                    size: config::MODEL_WEIGHTS_SIZE,
                },
                policy: RegionPolicy::Immutable,
                owner: ComponentType::FeatureExtractor,
                read_access: vec![
                    ComponentType::FeatureExtractor,
                    ComponentType::ReasoningEngine,
                ],
                write_access: vec![],
            },
            // Region 1: AppendOnly - witness_log (64 KiB)
            DemoRegion {
                index: 1,
                name: String::from("witness_log"),
                region_type: DemoRegionType::AppendOnly {
                    max_size: config::WITNESS_LOG_MAX_SIZE,
                },
                policy: RegionPolicy::AppendOnly {
                    max_size: config::WITNESS_LOG_MAX_SIZE,
                },
                owner: ComponentType::Attestor,
                read_access: vec![ComponentType::Coordinator, ComponentType::Attestor],
                write_access: vec![ComponentType::Attestor],
            },
            // Region 2: Slab - vector_store (3 KiB * 1024 = 3 MiB)
            DemoRegion {
                index: 2,
                name: String::from("vector_store"),
                region_type: DemoRegionType::Slab {
                    slot_size: config::VECTOR_SLOT_SIZE,
                    slots: config::VECTOR_SLOT_COUNT,
                },
                policy: RegionPolicy::Slab {
                    slot_size: config::VECTOR_SLOT_SIZE,
                    slot_count: config::VECTOR_SLOT_COUNT,
                },
                owner: ComponentType::FeatureExtractor,
                read_access: vec![
                    ComponentType::FeatureExtractor,
                    ComponentType::ReasoningEngine,
                ],
                write_access: vec![ComponentType::FeatureExtractor],
            },
        ]
    }

    fn create_proof_policy() -> ProofPolicyConfig {
        ProofPolicyConfig {
            default_tier: ProofTier::Standard,
            vector_mutations: ProofTier::Reflex,
            graph_mutations: ProofTier::Standard,
            structural_changes: ProofTier::Deep,
            component_tiers: vec![
                ComponentProofTier {
                    component: ComponentType::FeatureExtractor,
                    tier: ProofTier::Reflex,
                },
                ComponentProofTier {
                    component: ComponentType::ReasoningEngine,
                    tier: ProofTier::Standard,
                },
                ComponentProofTier {
                    component: ComponentType::Coordinator,
                    tier: ProofTier::Deep,
                },
            ],
            allow_escalation: true,
        }
    }

    fn create_rollback_hooks() -> Vec<RollbackHookConfig> {
        vec![
            RollbackHookConfig {
                name: String::from("on_coherence_drop"),
                trigger: RollbackTrigger::CoherenceDrop,
                function: String::from("handle_coherence_drop"),
                accessible_regions: vec![0, 2], // model_weights, vector_store
                timeout_us: 10_000,             // 10ms
            },
            RollbackHookConfig {
                name: String::from("on_proof_failure"),
                trigger: RollbackTrigger::ProofFailure,
                function: String::from("handle_proof_failure"),
                accessible_regions: vec![1, 2], // witness_log, vector_store
                timeout_us: 5_000,              // 5ms
            },
        ]
    }

    /// Returns the total memory required by all regions.
    #[must_use]
    pub fn total_memory_bytes(&self) -> usize {
        self.regions
            .iter()
            .map(|r| r.region_type.size_bytes())
            .sum()
    }

    /// Returns the expected syscall counts for the full pipeline.
    #[must_use]
    pub fn expected_syscall_counts(&self) -> [(u8, &'static str, u32); 12] {
        [
            (0, "task_spawn", config::TASK_SPAWNS as u32),
            (1, "cap_grant", config::CAP_GRANTS as u32),
            (2, "region_map", 3), // 3 regions
            (3, "queue_send", config::FULL_PIPELINE_EVENTS as u32 * 2), // sensor + extractor
            (4, "queue_recv", config::FULL_PIPELINE_EVENTS as u32 * 2), // extractor + engine
            (5, "timer_wait", config::TIMER_WAITS as u32),
            (6, "rvf_mount", 1),
            (7, "attest_emit", config::FULL_PIPELINE_EVENTS as u32),
            (8, "vector_get", config::FULL_PIPELINE_EVENTS as u32),
            (9, "vector_put_proved", config::FULL_PIPELINE_EVENTS as u32),
            (10, "graph_apply_proved", config::GRAPH_MUTATIONS as u32),
            (11, "sensor_subscribe", 1),
        ]
    }

    /// Validates the manifest structure.
    #[must_use]
    pub fn validate(&self) -> bool {
        // Check component indices are sequential
        for (i, comp) in self.components.iter().enumerate() {
            if comp.index as usize != i {
                return false;
            }
        }

        // Check region indices are sequential
        for (i, region) in self.regions.iter().enumerate() {
            if region.index as usize != i {
                return false;
            }
        }

        // Check all dependencies reference valid components
        let comp_count = self.components.len() as u32;
        for comp in &self.components {
            for &dep in &comp.dependencies {
                if dep >= comp_count || dep >= comp.index {
                    return false; // Dependencies must be lower indices (DAG)
                }
            }
        }

        // Check rollback hooks reference valid regions
        let region_count = self.regions.len() as u32;
        for hook in &self.rollback_hooks {
            for &region_idx in &hook.accessible_regions {
                if region_idx >= region_count {
                    return false;
                }
            }
        }

        true
    }

    /// Serializes the manifest to bytes (RVF format).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(4096);

        // Magic number "RVF1"
        bytes.extend_from_slice(b"RVF1");

        // Version
        bytes.extend_from_slice(&self.version.major.to_le_bytes());
        bytes.extend_from_slice(&self.version.minor.to_le_bytes());
        bytes.extend_from_slice(&self.version.patch.to_le_bytes());

        // Content hash
        bytes.extend_from_slice(&self.content_hash);

        // Component count
        bytes.extend_from_slice(&(self.components.len() as u32).to_le_bytes());

        // Region count
        bytes.extend_from_slice(&(self.regions.len() as u32).to_le_bytes());

        // Rollback hook count
        bytes.extend_from_slice(&(self.rollback_hooks.len() as u32).to_le_bytes());

        // Pad to minimum header size (96 bytes for compatibility)
        while bytes.len() < 96 {
            bytes.push(0);
        }

        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_demo_manifest() {
        let manifest = DemoManifest::cognitive_demo();

        // Verify component count
        assert_eq!(manifest.components.len(), 5);

        // Verify region count
        assert_eq!(manifest.regions.len(), 3);

        // Verify rollback hook count
        assert_eq!(manifest.rollback_hooks.len(), 2);

        // Verify manifest validates
        assert!(manifest.validate());
    }

    #[test]
    fn test_component_types() {
        assert_eq!(ComponentType::SensorAdapter.name(), "SensorAdapter");
        assert_eq!(ComponentType::FeatureExtractor.name(), "FeatureExtractor");
        assert_eq!(ComponentType::ReasoningEngine.name(), "ReasoningEngine");
        assert_eq!(ComponentType::Attestor.name(), "Attestor");
        assert_eq!(ComponentType::Coordinator.name(), "Coordinator");
    }

    #[test]
    fn test_region_types() {
        let immutable = DemoRegionType::Immutable { size: 1024 };
        assert_eq!(immutable.size_bytes(), 1024);
        assert_eq!(immutable.to_policy(), RegionPolicy::Immutable);

        let append = DemoRegionType::AppendOnly { max_size: 2048 };
        assert_eq!(append.size_bytes(), 2048);
        assert_eq!(
            append.to_policy(),
            RegionPolicy::AppendOnly { max_size: 2048 }
        );

        let slab = DemoRegionType::Slab {
            slot_size: 64,
            slots: 100,
        };
        assert_eq!(slab.size_bytes(), 6400);
        assert_eq!(
            slab.to_policy(),
            RegionPolicy::Slab {
                slot_size: 64,
                slot_count: 100,
            }
        );
    }

    #[test]
    fn test_proof_policy() {
        let manifest = DemoManifest::cognitive_demo();
        let policy = &manifest.proof_policy;

        assert_eq!(policy.vector_mutations, ProofTier::Reflex);
        assert_eq!(policy.graph_mutations, ProofTier::Standard);
        assert_eq!(policy.structural_changes, ProofTier::Deep);
        assert!(policy.allow_escalation);
    }

    #[test]
    fn test_manifest_total_memory() {
        let manifest = DemoManifest::cognitive_demo();
        let total = manifest.total_memory_bytes();

        // 1 MiB + 64 KiB + 3 MiB
        let expected = config::MODEL_WEIGHTS_SIZE
            + config::WITNESS_LOG_MAX_SIZE
            + (config::VECTOR_SLOT_SIZE * config::VECTOR_SLOT_COUNT);

        assert_eq!(total, expected);
    }

    #[test]
    fn test_manifest_to_bytes() {
        let manifest = DemoManifest::cognitive_demo();
        let bytes = manifest.to_bytes();

        // Check magic number
        assert_eq!(&bytes[0..4], b"RVF1");

        // Check minimum size
        assert!(bytes.len() >= 96);
    }

    #[test]
    fn test_expected_syscall_counts() {
        let manifest = DemoManifest::cognitive_demo();
        let counts = manifest.expected_syscall_counts();

        // Verify all 12 syscalls are represented
        assert_eq!(counts.len(), 12);

        // Verify specific counts
        assert_eq!(counts[0], (0, "task_spawn", config::TASK_SPAWNS as u32));
        assert_eq!(
            counts[7],
            (7, "attest_emit", config::FULL_PIPELINE_EVENTS as u32)
        );
        assert_eq!(counts[11], (11, "sensor_subscribe", 1));
    }
}
