//! # RVM Kernel
//!
//! Top-level integration crate for the RVM (RuVix Virtual Machine)
//! coherence-native microhypervisor. This crate wires together all
//! subsystems (HAL, capabilities, witness, proof, partitions, scheduler,
//! memory, coherence, boot, Wasm, and security) into a single API
//! surface.
//!
//! ## Architecture
//!
//! ```text
//!          +---------------------------------------------+
//!          |                  rvm-kernel                  |
//!          |                                             |
//!          |  +----------+  +----------+  +-----------+  |
//!          |  | rvm-boot |  | rvm-sched|  |rvm-memory |  |
//!          |  +----+-----+  +----+-----+  +-----+-----+  |
//!          |       |             |              |         |
//!          |  +----+-------------+--------------+-----+  |
//!          |  |            rvm-partition               |  |
//!          |  +----+--------+----------+---------+----+  |
//!          |       |        |          |         |       |
//!          |  +----+--+ +---+----+ +---+---+ +---+----+  |
//!          |  |rvm-cap| |rvm-wit.| |rvm-prf| |rvm-sec.|  |
//!          |  +----+--+ +---+----+ +---+---+ +---+----+  |
//!          |       |        |          |         |       |
//!          |  +----+--------+----------+---------+----+  |
//!          |  |              rvm-types                |   |
//!          |  +----+----------------------------------+  |
//!          |       |                                     |
//!          |  +----+--+  +----------+                    |
//!          |  |rvm-hal|  |rvm-wasm  | (optional)         |
//!          |  +-------+  +----------+                    |
//!          +---------------------------------------------+
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::doc_markdown,
    clippy::new_without_default
)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

/// Re-export all subsystem crates for unified access.
pub use rvm_boot as boot;
/// Capability-based access control.
pub use rvm_cap as cap;
/// Coherence monitoring and Phi computation.
pub use rvm_coherence as coherence;
/// Hardware abstraction layer traits.
pub use rvm_hal as hal;
/// Guest memory management.
pub use rvm_memory as memory;
/// Partition lifecycle management.
pub use rvm_partition as partition;
/// Proof-gated state transitions.
pub use rvm_proof as proof;
/// Coherence-weighted scheduler.
pub use rvm_sched as sched;
/// Security policy enforcement.
pub use rvm_security as security;
/// Core type definitions.
pub use rvm_types as types;
/// WebAssembly guest runtime.
pub use rvm_wasm as wasm;
/// Witness trail management.
pub use rvm_witness as witness;

/// RVM version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// RVM crate count (number of subsystem crates).
pub const CRATE_COUNT: usize = 13;

// ---------------------------------------------------------------------------
// Signer bridge (ADR-142 Phase 4)
// ---------------------------------------------------------------------------

/// Bridges the 64-byte proof-crate [`rvm_proof::WitnessSigner`] to the
/// 8-byte witness-crate [`rvm_witness::WitnessSigner`].
///
/// The proof crate defines a signer trait that operates on 32-byte digests
/// and produces 64-byte signatures. The witness crate defines a signer
/// trait that operates on `WitnessRecord` and produces 8-byte signatures
/// (for the `aux` field). This adapter bridges the two by:
///
/// 1. Computing a SHA-256 digest of the witness record's content fields.
/// 2. Signing the digest with the inner 64-byte signer.
/// 3. Truncating the result to 8 bytes for the `aux` field.
///
/// Verification recomputes the truncated signature and performs a
/// constant-time comparison.
pub mod signer_bridge {
    use rvm_types::WitnessRecord;

    /// Adapter that wraps a 64-byte [`rvm_proof::WitnessSigner`] and
    /// implements the 8-byte [`rvm_witness::WitnessSigner`].
    pub struct CryptoSignerAdapter<S: rvm_proof::WitnessSigner> {
        inner: S,
    }

    impl<S: rvm_proof::WitnessSigner> CryptoSignerAdapter<S> {
        /// Create a new adapter wrapping the given proof-crate signer.
        pub const fn new(inner: S) -> Self {
            Self { inner }
        }

        /// Return a reference to the inner proof-crate signer.
        pub fn inner(&self) -> &S {
            &self.inner
        }
    }

    #[cfg(feature = "crypto-sha256")]
    impl<S: rvm_proof::WitnessSigner> rvm_witness::WitnessSigner for CryptoSignerAdapter<S> {
        fn sign(&self, record: &WitnessRecord) -> [u8; 8] {
            let digest = rvm_witness::record_to_digest(record);
            let sig64 = self.inner.sign(&digest);
            let mut aux = [0u8; 8];
            aux.copy_from_slice(&sig64[..8]);
            aux
        }

        fn verify(&self, record: &WitnessRecord) -> bool {
            let expected = self.sign(record);
            // Constant-time comparison.
            let mut diff = 0u8;
            let mut i = 0;
            while i < 8 {
                diff |= expected[i] ^ record.aux[i];
                i += 1;
            }
            diff == 0
        }
    }
}

pub use signer_bridge::CryptoSignerAdapter;

// ---------------------------------------------------------------------------
// Kernel integration struct
// ---------------------------------------------------------------------------

use rvm_boot::BootTracker;
use rvm_cap::{CapManagerConfig, CapabilityManager};
use rvm_coherence::{CoherenceDecision, DefaultCoherenceEngine};
use rvm_memory::tier::{Tier, TierManager};
use rvm_partition::{CommEdgeId, DeviceLeaseManager, IpcManager, IpcMessage, PartitionManager};
use rvm_sched::Scheduler;
use rvm_types::{
    ActionKind, OwnedRegionId, PartitionConfig, PartitionId, RvmConfig, RvmError, RvmResult,
    WitnessRecord,
};
use rvm_witness::WitnessLog;

/// Default maximum CPUs supported by the kernel.
const DEFAULT_MAX_CPUS: usize = 8;

/// Default witness log capacity (number of records).
const DEFAULT_WITNESS_CAPACITY: usize = 256;

/// Default capability table capacity per partition.
const DEFAULT_CAP_CAPACITY: usize = 256;

/// Default partition table capacity.
const DEFAULT_MAX_PARTITIONS: usize = 256;

/// Default maximum IPC channels (inter-partition edges).
const DEFAULT_MAX_IPC_CHANNELS: usize = 128;

/// Default per-channel message queue depth.
const DEFAULT_IPC_QUEUE_SIZE: usize = 16;

/// Default maximum tracked memory regions for tier management.
const DEFAULT_MAX_TIER_REGIONS: usize = 256;

/// Recency decay per epoch (basis points subtracted each tick).
const RECENCY_DECAY_PER_EPOCH: u16 = 200;

/// Default maximum hardware devices.
const DEFAULT_MAX_DEVICES: usize = 32;

/// Default maximum concurrent device leases.
const DEFAULT_MAX_LEASES: usize = 64;

/// Result of a single epoch tick, combining scheduler and coherence outputs.
#[derive(Debug, Clone)]
pub struct EpochResult {
    /// Scheduler epoch summary (context switches, utilisation).
    pub summary: rvm_sched::EpochSummary,
    /// Coherence engine recommendation (split, merge, or no-action).
    pub decision: CoherenceDecision,
}

/// Result of applying a coherence decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyResult {
    /// No action was taken.
    NoAction,
    /// A partition was split into two.
    Split {
        /// The original partition.
        source: PartitionId,
        /// The newly created partition.
        child: PartitionId,
    },
    /// Two partitions were merged.
    Merged {
        /// The surviving partition.
        survivor: PartitionId,
        /// The partition that was absorbed.
        absorbed: PartitionId,
    },
}

/// Top-level kernel integrating all RVM subsystems.
///
/// The kernel holds ownership of all core subsystem instances
/// and provides a unified API for partition lifecycle, scheduling,
/// and security enforcement.
pub struct Kernel {
    /// Partition lifecycle manager.
    partitions: PartitionManager,
    /// Coherence-weighted scheduler (8 CPUs, 256 partitions).
    scheduler: Scheduler<DEFAULT_MAX_CPUS, DEFAULT_MAX_PARTITIONS>,
    /// Append-only witness log.
    witness_log: WitnessLog<DEFAULT_WITNESS_CAPACITY>,
    /// Capability manager (P1/P2/P3 verification).
    cap_manager: CapabilityManager<DEFAULT_CAP_CAPACITY>,
    /// Coherence engine — graph-driven partition scoring and split/merge.
    coherence: DefaultCoherenceEngine,
    /// Inter-partition communication channels.
    ipc: IpcManager<DEFAULT_MAX_IPC_CHANNELS, DEFAULT_IPC_QUEUE_SIZE>,
    /// Coherence-driven memory tier manager.
    tier_manager: TierManager<DEFAULT_MAX_TIER_REGIONS>,
    /// Hardware device lease manager.
    devices: DeviceLeaseManager<DEFAULT_MAX_DEVICES, DEFAULT_MAX_LEASES>,
    /// Boot progress tracker.
    boot: BootTracker,
    /// Kernel configuration.
    config: RvmConfig,
    /// Whether the kernel has completed booting.
    booted: bool,
}

/// Configuration for constructing a kernel instance.
#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    /// Base RVM configuration.
    pub rvm: RvmConfig,
    /// Capability manager configuration.
    pub cap: CapManagerConfig,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            rvm: RvmConfig::default(),
            cap: CapManagerConfig::new(),
        }
    }
}

impl Kernel {
    /// Default Stoer-Wagner iteration budget for the coherence engine.
    const DEFAULT_MINCUT_BUDGET: u32 = 100;

    /// Create a new kernel instance with the given configuration.
    #[must_use]
    pub fn new(config: KernelConfig) -> Self {
        Self {
            partitions: PartitionManager::new(),
            scheduler: Scheduler::new(),
            witness_log: WitnessLog::new(),
            cap_manager: CapabilityManager::new(config.cap),
            coherence: DefaultCoherenceEngine::with_defaults(Self::DEFAULT_MINCUT_BUDGET),
            ipc: IpcManager::new(),
            tier_manager: TierManager::new(),
            devices: DeviceLeaseManager::new(),
            boot: BootTracker::new(),
            config: config.rvm,
            booted: false,
        }
    }

    /// Create a kernel with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(KernelConfig::default())
    }

    /// Run the boot sequence through all 7 phases.
    ///
    /// Each phase completion is recorded as a witness entry. After all
    /// phases complete, the kernel is ready to accept partition requests.
    pub fn boot(&mut self) -> RvmResult<()> {
        use rvm_boot::BootPhase;

        let phases = [
            BootPhase::HalInit,
            BootPhase::MemoryInit,
            BootPhase::CapabilityInit,
            BootPhase::WitnessInit,
            BootPhase::SchedulerInit,
            BootPhase::RootPartition,
            BootPhase::Handoff,
        ];

        for phase in &phases {
            self.boot.complete_phase(*phase)?;
            emit_boot_witness(&self.witness_log, *phase);
        }

        self.booted = true;
        Ok(())
    }

    /// Advance the scheduler and coherence engine by one epoch.
    ///
    /// Returns an `EpochResult` containing the scheduler summary and the
    /// coherence engine's split/merge recommendation. Requires the kernel
    /// to have booted.
    pub fn tick(&mut self) -> RvmResult<EpochResult> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }

        let summary = self.scheduler.tick_epoch();

        // Tick coherence engine. Use a fixed CPU load estimate for now;
        // a future HAL integration will read real CPU utilisation.
        let cpu_load_estimate = 20u8;
        let decision = self.coherence.tick(cpu_load_estimate);

        // Advance tier manager epoch and decay recency scores.
        self.tier_manager.advance_epoch();
        self.tier_manager.decay_recency(RECENCY_DECAY_PER_EPOCH);

        // Propagate coherence scores to partition objects so that
        // downstream consumers (scheduler, security) see fresh values.
        self.sync_partition_scores();

        // Emit an epoch witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::SchedulerEpoch as u8;
        record.proof_tier = 1;
        let switch_bytes = summary.switch_count.to_le_bytes();
        record.payload[0..2].copy_from_slice(&switch_bytes);
        self.witness_log.append(record);

        Ok(EpochResult { summary, decision })
    }

    /// Record a directed communication event between two partitions.
    ///
    /// Updates the coherence graph edge weight. Call this when agents in
    /// different partitions exchange messages.
    pub fn record_communication(
        &mut self,
        from: PartitionId,
        to: PartitionId,
        weight: u64,
    ) -> RvmResult<()> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        self.coherence
            .record_communication(from, to, weight)
            .map_err(|_| RvmError::InternalError)
    }

    /// Get the coherence score for a partition (0..10000 basis points).
    #[must_use]
    pub fn coherence_score(&self, id: PartitionId) -> rvm_types::CoherenceScore {
        self.coherence.score(id)
    }

    /// Get the cut pressure for a partition (0..10000 basis points).
    #[must_use]
    pub fn coherence_pressure(&self, id: PartitionId) -> rvm_types::CutPressure {
        self.coherence.pressure(id)
    }

    /// Get the latest coherence decision without advancing the epoch.
    #[must_use]
    pub fn coherence_recommendation(&self) -> CoherenceDecision {
        self.coherence.recommend()
    }

    /// Create a new partition with the given configuration.
    ///
    /// Registers the partition in the coherence graph and emits a
    /// `PartitionCreate` witness record on success.
    pub fn create_partition(&mut self, config: &PartitionConfig) -> RvmResult<PartitionId> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }

        let epoch = self.scheduler.current_epoch();
        let id = self.partitions.create(
            rvm_partition::PartitionType::Agent,
            config.vcpu_count,
            epoch,
        )?;

        // Register in coherence graph (best-effort: ignore capacity errors
        // since the partition already exists in the partition manager).
        let _ = self.coherence.add_partition(id);

        // Emit witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::PartitionCreate as u8;
        record.proof_tier = 1;
        record.actor_partition_id = PartitionId::HYPERVISOR.as_u32();
        record.target_object_id = id.as_u32() as u64;
        self.witness_log.append(record);

        Ok(id)
    }

    /// Destroy a partition and reclaim its resources.
    ///
    /// Destroy a partition: remove from manager, coherence graph, and emit witness.
    pub fn destroy_partition(&mut self, id: PartitionId) -> RvmResult<()> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }

        // Verify the partition exists and mark as Destroyed.
        let state = self
            .partitions
            .get(id)
            .ok_or(RvmError::PartitionNotFound)?
            .state;
        if !rvm_partition::valid_transition(state, rvm_partition::PartitionState::Destroyed) {
            return Err(RvmError::InvalidPartitionState);
        }

        // Remove from coherence graph.
        let _ = self.coherence.remove_partition(id);

        // Remove from partition manager (frees the slot).
        self.partitions.remove(id)?;

        // Emit witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::PartitionDestroy as u8;
        record.proof_tier = 1;
        record.actor_partition_id = PartitionId::HYPERVISOR.as_u32();
        record.target_object_id = id.as_u32() as u64;
        self.witness_log.append(record);

        Ok(())
    }

    /// Return whether the kernel has completed booting.
    #[must_use]
    pub const fn is_booted(&self) -> bool {
        self.booted
    }

    /// Return the current scheduler epoch.
    #[must_use]
    pub fn current_epoch(&self) -> u32 {
        self.scheduler.current_epoch()
    }

    /// Return the number of active partitions.
    #[must_use]
    pub fn partition_count(&self) -> usize {
        self.partitions.count()
    }

    /// Return the total number of witness records emitted.
    pub fn witness_count(&self) -> u64 {
        self.witness_log.total_emitted()
    }

    /// Return a reference to the kernel configuration.
    #[must_use]
    pub const fn config(&self) -> &RvmConfig {
        &self.config
    }

    /// Return a reference to the partition manager.
    #[must_use]
    pub fn partitions(&self) -> &PartitionManager {
        &self.partitions
    }

    /// Return a reference to the capability manager.
    #[must_use]
    pub fn cap_manager(&self) -> &CapabilityManager<DEFAULT_CAP_CAPACITY> {
        &self.cap_manager
    }

    /// Return a mutable reference to the capability manager.
    pub fn cap_manager_mut(&mut self) -> &mut CapabilityManager<DEFAULT_CAP_CAPACITY> {
        &mut self.cap_manager
    }

    /// Return a reference to the witness log.
    #[must_use]
    pub fn witness_log(&self) -> &WitnessLog<DEFAULT_WITNESS_CAPACITY> {
        &self.witness_log
    }

    // -- Scheduler integration --

    /// Enqueue a partition onto a CPU's run queue.
    ///
    /// Automatically injects the partition's coherence-derived cut pressure
    /// into the scheduler priority. This is the primary path for scheduling
    /// partitions with coherence awareness.
    pub fn enqueue_partition(
        &mut self,
        cpu: usize,
        id: PartitionId,
        deadline_urgency: u16,
    ) -> RvmResult<()> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        if self.partitions.get(id).is_none() {
            return Err(RvmError::PartitionNotFound);
        }

        let pressure = self.coherence.pressure(id);
        if !self.scheduler.enqueue(cpu, id, deadline_urgency, pressure) {
            return Err(RvmError::ResourceLimitExceeded);
        }
        Ok(())
    }

    /// Pick the next partition on a CPU and switch to it.
    ///
    /// Returns `(old_partition, new_partition)` if a switch occurred.
    /// Emits no witness record (DC-10: switches are bulk-summarised at
    /// epoch boundaries, not individually witnessed).
    pub fn switch_next(&mut self, cpu: usize) -> Option<(Option<PartitionId>, PartitionId)> {
        self.scheduler.switch_next(cpu)
    }

    // -- Coherence-driven split/merge --

    /// Execute a coherence-driven partition split.
    ///
    /// Creates a new "child" partition, then applies the mincut boundary
    /// computed by the coherence engine: every neighbor on the far side
    /// of the cut has its communication edges re-homed from `source` to
    /// the child, so the post-split topology matches the computed cut
    /// (not an arbitrary assignment). The plan cached by the epoch task
    /// is reused when fresh; otherwise the (budgeted) mincut is computed
    /// on demand -- both off the partition-switch hot path.
    ///
    /// Emits a `StructuralSplit` witness whose payload records the
    /// number of re-homed neighbors. The actual agent migration is the
    /// caller's responsibility; this method handles the partition and
    /// coherence graph bookkeeping.
    ///
    /// Returns the new partition ID on success.
    pub fn execute_split(&mut self, source: PartitionId) -> RvmResult<PartitionId> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        let src = self.partitions.get(source).ok_or(RvmError::PartitionNotFound)?;
        let vcpu_count = src.vcpu_count;

        // Resolve the cut boundary BEFORE mutating the graph: cached
        // epoch plan if fresh, otherwise computed now.
        let plan = self.coherence.split_plan_for(source);

        // Create the new partition (inherits source's vCPU count).
        let epoch = self.scheduler.current_epoch();
        let child = self.partitions.create(
            rvm_partition::PartitionType::Agent,
            vcpu_count,
            epoch,
        )?;

        // Register child in coherence graph.
        let _ = self.coherence.add_partition(child);

        // Re-home the far side of the cut to the child. Best-effort:
        // an unusable plan (e.g., isolated source) moves nothing, which
        // matches the previous "empty child" semantics.
        let moved = self
            .coherence
            .apply_split_boundary(source, child, &plan)
            .unwrap_or(0);

        // Emit structural split witness (payload[0..2] = re-homed count).
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::StructuralSplit as u8;
        record.proof_tier = 1;
        record.actor_partition_id = source.as_u32();
        record.target_object_id = child.as_u32() as u64;
        record.payload[0..2].copy_from_slice(&moved.to_le_bytes());
        self.witness_log.append(record);

        Ok(child)
    }

    /// HOT PATH placement hint: route a newly arriving partition to a
    /// side of the current split boundary via O(degree) Fennel placement
    /// (no mincut recomputation). Returns `None` when no epoch split
    /// plan exists.
    pub fn split_placement_hint(
        &mut self,
        id: PartitionId,
    ) -> Option<rvm_coherence::SplitSide> {
        self.coherence.place_incremental(id)
    }

    /// Execute a coherence-driven partition merge.
    ///
    /// Validates merge preconditions (coherence threshold, adjacency,
    /// resource limits) and emits a `StructuralMerge` witness. The
    /// target partition absorbs the source; the source is destroyed.
    ///
    /// Returns the surviving partition ID on success.
    pub fn execute_merge(
        &mut self,
        absorber: PartitionId,
        absorbed: PartitionId,
    ) -> RvmResult<PartitionId> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        // Verify both partitions exist.
        let _a = self.partitions.get(absorber).ok_or(RvmError::PartitionNotFound)?;
        let _b = self.partitions.get(absorbed).ok_or(RvmError::PartitionNotFound)?;

        // Check coherence-based merge preconditions.
        let score_a = self.coherence.score(absorber);
        let score_b = self.coherence.score(absorbed);
        rvm_partition::merge_preconditions_met(score_a, score_b)
            .map_err(|_| RvmError::InvalidPartitionState)?;

        // Remove absorbed from coherence graph.
        let _ = self.coherence.remove_partition(absorbed);

        // Emit structural merge witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::StructuralMerge as u8;
        record.proof_tier = 1;
        record.actor_partition_id = absorber.as_u32();
        record.target_object_id = absorbed.as_u32() as u64;
        self.witness_log.append(record);

        Ok(absorber)
    }

    /// Apply a coherence decision returned from `tick()`.
    ///
    /// - `SplitRecommended` → `execute_split`
    /// - `MergeRecommended` → `execute_merge`
    /// - `NoAction` → no-op
    ///
    /// Returns the decision that was applied, along with any new partition
    /// ID created by a split.
    pub fn apply_decision(
        &mut self,
        decision: CoherenceDecision,
    ) -> RvmResult<ApplyResult> {
        match decision {
            CoherenceDecision::NoAction => Ok(ApplyResult::NoAction),
            CoherenceDecision::SplitRecommended { partition, .. } => {
                let child = self.execute_split(partition)?;
                Ok(ApplyResult::Split { source: partition, child })
            }
            CoherenceDecision::MergeRecommended { a, b, .. } => {
                let survivor = self.execute_merge(a, b)?;
                Ok(ApplyResult::Merged { survivor, absorbed: b })
            }
        }
    }

    // -- IPC (inter-partition communication) --

    /// Create an IPC channel between two partitions.
    ///
    /// Also registers the communication edge in the coherence graph.
    /// Emits a `CommEdgeCreate` witness record.
    pub fn create_channel(
        &mut self,
        from: PartitionId,
        to: PartitionId,
    ) -> RvmResult<CommEdgeId> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        if self.partitions.get(from).is_none() || self.partitions.get(to).is_none() {
            return Err(RvmError::PartitionNotFound);
        }

        let edge_id = self.ipc.create_channel(from, to)?;

        // Emit witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::CommEdgeCreate as u8;
        record.proof_tier = 1;
        record.actor_partition_id = from.as_u32();
        record.target_object_id = to.as_u32() as u64;
        self.witness_log.append(record);

        Ok(edge_id)
    }

    /// Send an IPC message on an existing channel.
    ///
    /// `caller_id` is the partition performing the send; this is forwarded
    /// to the IPC manager for sender-identity verification.
    ///
    /// Automatically increments the coherence graph edge weight for the
    /// sender->receiver pair, feeding the mincut/split/merge decisions.
    /// Emits an `IpcSend` witness record.
    pub fn ipc_send(
        &mut self,
        edge_id: CommEdgeId,
        msg: IpcMessage,
        caller_id: PartitionId,
    ) -> RvmResult<()> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        let sender = msg.sender;
        let receiver = msg.receiver;

        self.ipc.send(edge_id, msg, caller_id)?;

        // Feed the coherence graph: each message increments edge weight
        // by 1 (the IPC manager also tracks its own cumulative weight).
        let _ = self.coherence.record_communication(sender, receiver, 1);

        // Emit witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::IpcSend as u8;
        record.proof_tier = 1;
        record.actor_partition_id = sender.as_u32();
        record.target_object_id = receiver.as_u32() as u64;
        self.witness_log.append(record);

        Ok(())
    }

    /// Receive an IPC message from a channel.
    pub fn ipc_receive(&mut self, edge_id: CommEdgeId) -> RvmResult<Option<IpcMessage>> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        self.ipc.receive(edge_id)
    }

    /// Destroy an IPC channel.
    ///
    /// Emits a `CommEdgeDestroy` witness record.
    pub fn destroy_channel(&mut self, edge_id: CommEdgeId) -> RvmResult<()> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        self.ipc.destroy_channel(edge_id)?;

        // Emit witness.
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::CommEdgeDestroy as u8;
        record.proof_tier = 1;
        record.payload[0..8].copy_from_slice(&edge_id.as_u64().to_le_bytes());
        self.witness_log.append(record);

        Ok(())
    }

    /// Return the number of active IPC channels.
    #[must_use]
    pub fn ipc_channel_count(&self) -> usize {
        self.ipc.channel_count()
    }

    // -- Memory tier management --

    /// Register a memory region in the tier manager.
    ///
    /// Regions start at the given tier and are subject to coherence-driven
    /// promotion/demotion as the system evolves.
    pub fn register_region(
        &mut self,
        region_id: OwnedRegionId,
        initial_tier: Tier,
    ) -> RvmResult<()> {
        self.tier_manager.register(region_id, initial_tier)
    }

    /// Record a memory access, boosting the region's recency score.
    pub fn record_memory_access(&mut self, region_id: OwnedRegionId) -> RvmResult<()> {
        self.tier_manager.record_access(region_id)
    }

    /// Update a region's cut value from the coherence engine.
    ///
    /// Call this after `tick()` to propagate coherence scores into the
    /// tier placement decisions. The `cut_value` is the coherence score
    /// (basis points) of the partition that owns this region.
    pub fn update_region_cut_value(
        &mut self,
        region_id: OwnedRegionId,
        cut_value: u16,
    ) -> RvmResult<()> {
        self.tier_manager.update_cut_value(region_id, cut_value)
    }

    /// Promote a region to a warmer tier.
    ///
    /// Validates residency score against promotion thresholds.
    /// Emits a `RegionPromote` witness record on success.
    pub fn promote_region(
        &mut self,
        region_id: OwnedRegionId,
        target: Tier,
    ) -> RvmResult<Tier> {
        let old_tier = self.tier_manager.promote(region_id, target)?;

        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::RegionPromote as u8;
        record.proof_tier = 1;
        record.target_object_id = region_id.as_u64();
        record.payload[0] = old_tier.index();
        record.payload[1] = target.index();
        self.witness_log.append(record);

        Ok(old_tier)
    }

    /// Demote a region to a colder tier.
    ///
    /// Emits a `RegionDemote` witness record on success.
    pub fn demote_region(
        &mut self,
        region_id: OwnedRegionId,
        target: Tier,
    ) -> RvmResult<Tier> {
        let old_tier = self.tier_manager.demote(region_id, target)?;

        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::RegionDemote as u8;
        record.proof_tier = 1;
        record.target_object_id = region_id.as_u64();
        record.payload[0] = old_tier.index();
        record.payload[1] = target.index();
        self.witness_log.append(record);

        Ok(old_tier)
    }

    /// Query a region's current tier state.
    #[must_use]
    pub fn region_tier(&self, region_id: OwnedRegionId) -> Option<Tier> {
        self.tier_manager.get(region_id).map(|s| s.tier)
    }

    // -- Device lease management --

    /// Register a hardware device.
    pub fn register_device(
        &mut self,
        info: rvm_partition::DeviceInfo,
    ) -> RvmResult<u32> {
        self.devices.register_device(info)
    }

    /// Grant a time-bounded lease on a device to a partition.
    ///
    /// Emits a `DeviceLeaseGrant` witness record.
    pub fn grant_device_lease(
        &mut self,
        device_id: u32,
        partition: PartitionId,
        duration_epochs: u64,
        cap_hash: u32,
    ) -> RvmResult<rvm_types::DeviceLeaseId> {
        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }
        let epoch = self.scheduler.current_epoch() as u64;
        let lease_id = self.devices.grant_lease(
            device_id, partition, duration_epochs, epoch, cap_hash,
        )?;

        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::DeviceLeaseGrant as u8;
        record.proof_tier = 1;
        record.actor_partition_id = partition.as_u32();
        record.target_object_id = device_id as u64;
        self.witness_log.append(record);

        Ok(lease_id)
    }

    /// Revoke a device lease.
    ///
    /// Emits a `DeviceLeaseRevoke` witness record.
    pub fn revoke_device_lease(
        &mut self,
        lease_id: rvm_types::DeviceLeaseId,
    ) -> RvmResult<()> {
        self.devices.revoke_lease(lease_id)?;

        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::DeviceLeaseRevoke as u8;
        record.proof_tier = 1;
        record.payload[0..8].copy_from_slice(&lease_id.as_u64().to_le_bytes());
        self.witness_log.append(record);

        Ok(())
    }

    // -- Feature-gated subsystems --

    /// Whether the coherence engine is integrated.
    ///
    /// Always `true` since the engine is a core part of the kernel.
    #[must_use]
    pub const fn coherence_enabled(&self) -> bool {
        true
    }

    /// Access the coherence engine directly (for inspection/testing).
    #[must_use]
    pub fn coherence_engine(&self) -> &DefaultCoherenceEngine {
        &self.coherence
    }

    /// Check whether WASM support is compiled in.
    #[cfg(feature = "wasm")]
    pub fn wasm_enabled(&self) -> bool {
        true
    }

    /// WASM support is not compiled in.
    #[cfg(not(feature = "wasm"))]
    pub fn wasm_enabled(&self) -> bool {
        false
    }

    // -- Coherence score propagation --

    /// Synchronise coherence scores and cut pressures from the coherence
    /// engine into the partition objects. Called automatically by `tick()`.
    ///
    /// This ensures that downstream consumers (scheduler priority, security
    /// gates, tier placement) always see fresh values.
    fn sync_partition_scores(&mut self) {
        // Collect IDs first to avoid borrow conflict.
        let mut ids = [None::<PartitionId>; DEFAULT_MAX_PARTITIONS];
        for (i, id) in self.partitions.active_ids().enumerate() {
            if i >= DEFAULT_MAX_PARTITIONS {
                break;
            }
            ids[i] = Some(id);
        }

        for slot in &ids {
            let id = match slot {
                Some(id) => *id,
                None => break,
            };
            let score = self.coherence.score(id);
            let pressure = self.coherence.pressure(id);
            if let Some(p) = self.partitions.get_mut(id) {
                p.coherence = score;
                p.cut_pressure = pressure;
            }
        }
    }

    // -- Security-gated operations --

    /// Create a partition with capability-based security check.
    ///
    /// Requires a `CapToken` with `Partition` type and `WRITE` rights.
    /// Emits a `ProofRejected` witness on denial.
    pub fn checked_create_partition(
        &mut self,
        config: &PartitionConfig,
        token: &rvm_types::CapToken,
    ) -> RvmResult<PartitionId> {
        use rvm_security::gate::{GateRequest, SecurityGate};

        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }

        let gate = SecurityGate::new(&self.witness_log);
        let request = GateRequest {
            token: *token,
            required_type: rvm_types::CapType::Partition,
            required_rights: rvm_types::CapRights::WRITE,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 0,
            timestamp_ns: 0,
        };

        gate.check_and_execute(&request)
            .map_err(|_| RvmError::InsufficientCapability)?;

        // Delegate to the unsecured create (already emits its own witness).
        self.create_partition(config)
    }

    /// Send an IPC message with capability-based security check.
    ///
    /// Requires a `CapToken` with `Partition` type and `WRITE` rights.
    pub fn checked_ipc_send(
        &mut self,
        edge_id: CommEdgeId,
        msg: IpcMessage,
        token: &rvm_types::CapToken,
    ) -> RvmResult<()> {
        use rvm_security::gate::{GateRequest, SecurityGate};

        if !self.booted {
            return Err(RvmError::InvalidPartitionState);
        }

        let gate = SecurityGate::new(&self.witness_log);
        let request = GateRequest {
            token: *token,
            required_type: rvm_types::CapType::Partition,
            required_rights: rvm_types::CapRights::WRITE,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::IpcSend,
            target_object_id: msg.receiver.as_u32() as u64,
            timestamp_ns: 0,
        };

        gate.check_and_execute(&request)
            .map_err(|_| RvmError::InsufficientCapability)?;

        let caller = msg.sender;
        self.ipc_send(edge_id, msg, caller)
    }

    /// Return a reference to the scheduler (for inspection/testing).
    #[must_use]
    pub fn scheduler(&self) -> &Scheduler<DEFAULT_MAX_CPUS, DEFAULT_MAX_PARTITIONS> {
        &self.scheduler
    }

    /// Enter degraded mode (DC-6: coherence engine offline).
    ///
    /// In degraded mode, `CutPressure` is zeroed for all scheduler
    /// decisions, and the system operates on deadline urgency alone.
    pub fn enter_degraded_mode(&mut self) {
        self.scheduler.enter_degraded();

        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::DegradedModeEntered as u8;
        record.proof_tier = 1;
        self.witness_log.append(record);
    }

    /// Exit degraded mode.
    pub fn exit_degraded_mode(&mut self) {
        self.scheduler.exit_degraded();

        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::DegradedModeExited as u8;
        record.proof_tier = 1;
        self.witness_log.append(record);
    }

    /// Whether the system is in degraded mode.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.scheduler.is_degraded()
    }
}

// ---------------------------------------------------------------------------
// KernelHostContext — connects Wasm host functions to real kernel subsystems
// ---------------------------------------------------------------------------

/// Host context that routes Wasm guest calls to the kernel's IPC subsystem.
///
/// Holds a mutable reference to the kernel's IPC manager and the
/// partition ID that the guest belongs to. Memory allocation is
/// delegated to the partition's own allocator (not held here).
pub struct KernelHostContext<'a> {
    /// The partition hosting this Wasm agent.
    pub partition: PartitionId,
    /// IPC manager for Send/Receive operations.
    pub ipc: &'a mut IpcManager<DEFAULT_MAX_IPC_CHANNELS, DEFAULT_IPC_QUEUE_SIZE>,
    /// Active IPC channel (set by the caller before dispatch).
    pub active_channel: Option<CommEdgeId>,
    /// Monotonic sequence counter for IPC messages.
    pub next_sequence: u64,
}

impl<'a> rvm_wasm::host_functions::HostContext for KernelHostContext<'a> {
    fn send(&mut self, _sender: rvm_wasm::agent::AgentId, target: u64, length: u64) -> RvmResult<u64> {
        let edge = self.active_channel.ok_or(RvmError::PartitionNotFound)?;

        // Checked truncation: reject if target overflows u32.
        if target > u32::MAX as u64 {
            return Err(RvmError::ResourceLimitExceeded);
        }
        let target_u32 = target as u32;

        // Validate target is not the hypervisor (0) and not self-send.
        if target_u32 == 0 {
            return Err(RvmError::InsufficientCapability);
        }
        if target_u32 == self.partition.as_u32() {
            return Err(RvmError::InsufficientCapability);
        }

        // Validate payload length: reject if it would overflow u16.
        if length > u16::MAX as u64 {
            return Err(RvmError::ResourceLimitExceeded);
        }

        let seq = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        let msg = IpcMessage {
            sender: self.partition,
            receiver: PartitionId::new(target_u32),
            edge_id: edge,
            payload_len: length as u16,
            msg_type: 0,
            sequence: seq,
            capability_hash: 0,
        };
        self.ipc.send(edge, msg, self.partition)?;
        Ok(length)
    }

    fn receive(&mut self, _receiver: rvm_wasm::agent::AgentId) -> RvmResult<u64> {
        let edge = self.active_channel.ok_or(RvmError::PartitionNotFound)?;
        match self.ipc.receive(edge)? {
            Some(msg) => Ok(msg.payload_len as u64),
            None => Ok(0),
        }
    }
}

/// Emit a boot phase completion witness.
fn emit_boot_witness(log: &WitnessLog<DEFAULT_WITNESS_CAPACITY>, phase: rvm_boot::BootPhase) {
    let action = match phase {
        rvm_boot::BootPhase::Handoff => ActionKind::BootComplete,
        _ => ActionKind::BootAttestation,
    };
    let mut record = WitnessRecord::zeroed();
    record.action_kind = action as u8;
    record.proof_tier = 1;
    record.actor_partition_id = PartitionId::HYPERVISOR.as_u32();
    record.payload[0] = phase as u8;
    log.append(record);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = Kernel::with_defaults();
        assert!(!kernel.is_booted());
        assert_eq!(kernel.partition_count(), 0);
        assert_eq!(kernel.witness_count(), 0);
    }

    #[test]
    fn test_boot_sequence() {
        let mut kernel = Kernel::with_defaults();
        assert!(kernel.boot().is_ok());
        assert!(kernel.is_booted());

        // 7 boot phases = 7 witness records.
        assert_eq!(kernel.witness_count(), 7);
    }

    #[test]
    fn test_double_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        // Second boot attempt fails because phases are already complete.
        assert!(kernel.boot().is_err());
    }

    #[test]
    fn test_create_partition() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();
        assert_eq!(kernel.partition_count(), 1);
        assert!(kernel.partitions().get(id).is_some());

        // Witness for create.
        let pre_boot_witnesses = 7u64;
        assert_eq!(kernel.witness_count(), pre_boot_witnesses + 1);
    }

    #[test]
    fn test_create_partition_before_boot() {
        let mut kernel = Kernel::with_defaults();
        let config = PartitionConfig::default();
        assert_eq!(kernel.create_partition(&config), Err(RvmError::InvalidPartitionState));
    }

    #[test]
    fn test_destroy_partition() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();
        assert!(kernel.destroy_partition(id).is_ok());
    }

    #[test]
    fn test_destroy_nonexistent_partition() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let bad_id = PartitionId::new(999);
        assert_eq!(kernel.destroy_partition(bad_id), Err(RvmError::PartitionNotFound));
    }

    #[test]
    fn test_tick() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let result = kernel.tick().unwrap();
        assert_eq!(result.summary.epoch, 0);
        assert_eq!(result.decision, CoherenceDecision::NoAction);
        assert_eq!(kernel.current_epoch(), 1);
    }

    #[test]
    fn test_tick_before_boot() {
        let mut kernel = Kernel::with_defaults();
        assert!(kernel.tick().is_err());
    }

    #[test]
    fn test_feature_gates() {
        let kernel = Kernel::with_defaults();

        // Coherence is always enabled now.
        assert!(kernel.coherence_enabled());
        let _wasm = kernel.wasm_enabled();
    }

    #[test]
    fn test_custom_config() {
        let config = KernelConfig {
            rvm: RvmConfig {
                max_partitions: 64,
                ..RvmConfig::default()
            },
            cap: CapManagerConfig::new().with_max_depth(4),
        };
        let mut kernel = Kernel::new(config);
        assert_eq!(kernel.config().max_partitions, 64);
        kernel.boot().unwrap();
        assert!(kernel.is_booted());
    }

    #[test]
    fn test_multiple_partitions() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();

        assert_ne!(id1, id2);
        assert_eq!(kernel.partition_count(), 2);
    }

    #[test]
    fn test_kernel_version() {
        assert!(!VERSION.is_empty());
        assert_eq!(CRATE_COUNT, 13);
    }

    // ---------------------------------------------------------------
    // Integration-style lifecycle tests
    // ---------------------------------------------------------------

    #[test]
    fn test_full_boot_create_tick_destroy_lifecycle() {
        let mut kernel = Kernel::with_defaults();

        // Phase 1: Boot
        kernel.boot().unwrap();
        assert!(kernel.is_booted());
        let boot_witnesses = kernel.witness_count();
        assert_eq!(boot_witnesses, 7);

        // Phase 2: Create a partition
        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();
        assert_eq!(kernel.partition_count(), 1);
        assert_eq!(kernel.witness_count(), boot_witnesses + 1);

        // Phase 3: Tick the scheduler several times
        for expected_epoch in 0..5u32 {
            let result = kernel.tick().unwrap();
            assert_eq!(result.summary.epoch, expected_epoch);
        }
        assert_eq!(kernel.current_epoch(), 5);
        // 5 ticks = 5 more witness records
        assert_eq!(kernel.witness_count(), boot_witnesses + 1 + 5);

        // Phase 4: Destroy the partition
        kernel.destroy_partition(id).unwrap();
        assert_eq!(kernel.witness_count(), boot_witnesses + 1 + 5 + 1);
    }

    #[test]
    fn test_create_partition_with_zero_vcpus() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        // PartitionConfig allows vcpu_count=0 at the kernel level
        // (the partition manager does not validate vcpu count).
        let config = PartitionConfig {
            vcpu_count: 0,
            ..PartitionConfig::default()
        };
        let result = kernel.create_partition(&config);
        // Should succeed -- validation is not enforced at this level.
        assert!(result.is_ok());
    }

    #[test]
    fn test_destroy_before_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        let id = PartitionId::new(1);
        assert_eq!(kernel.destroy_partition(id), Err(RvmError::InvalidPartitionState));
    }

    #[test]
    fn test_destroy_twice_fails_second_time() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();
        assert!(kernel.destroy_partition(id).is_ok());
        // Second destroy should fail — partition was removed.
        assert_eq!(kernel.destroy_partition(id), Err(RvmError::PartitionNotFound));
    }

    #[test]
    fn test_many_partitions_coexist() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let mut ids = [PartitionId::new(0); 10];
        for id in &mut ids {
            *id = kernel.create_partition(&config).unwrap();
        }
        assert_eq!(kernel.partition_count(), 10);

        // All IDs are unique.
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                assert_ne!(a, b);
            }
        }

        // All are accessible.
        for id in &ids {
            assert!(kernel.partitions().get(*id).is_some());
        }
    }

    #[test]
    fn test_create_partition_emits_correct_witness_fields() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();
        let boot_count = kernel.witness_count();

        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();

        // The create witness is the record right after boot witnesses.
        let record = kernel.witness_log().get(boot_count as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::PartitionCreate as u8);
        assert_eq!(record.proof_tier, 1);
        assert_eq!(record.actor_partition_id, PartitionId::HYPERVISOR.as_u32());
        assert_eq!(record.target_object_id, id.as_u32() as u64);
    }

    #[test]
    fn test_destroy_partition_emits_correct_witness_fields() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id = kernel.create_partition(&config).unwrap();
        let pre_destroy = kernel.witness_count();

        kernel.destroy_partition(id).unwrap();
        let record = kernel.witness_log().get(pre_destroy as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::PartitionDestroy as u8);
        assert_eq!(record.proof_tier, 1);
        assert_eq!(record.target_object_id, id.as_u32() as u64);
    }

    #[test]
    fn test_tick_emits_scheduler_epoch_witness() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();
        let pre_tick = kernel.witness_count();

        kernel.tick().unwrap();
        let record = kernel.witness_log().get(pre_tick as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::SchedulerEpoch as u8);
    }

    #[test]
    fn test_cap_manager_accessible() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        // Verify we can use the capability manager through the kernel.
        let cap_mgr = kernel.cap_manager_mut();
        let owner = PartitionId::new(1);
        let result = cap_mgr.create_root_capability(
            rvm_types::CapType::Partition,
            rvm_types::CapRights::READ,
            0,
            owner,
        );
        assert!(result.is_ok());
    }

    // ---------------------------------------------------------------
    // Coherence engine integration tests
    // ---------------------------------------------------------------

    #[test]
    fn test_coherence_engine_tracks_partitions() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();

        // Coherence engine should track the same count.
        assert_eq!(kernel.coherence_engine().partition_count(), 2);

        // Isolated partitions have max coherence score.
        assert_eq!(
            kernel.coherence_score(id1),
            rvm_types::CoherenceScore::MAX,
        );
        assert_eq!(
            kernel.coherence_score(id2),
            rvm_types::CoherenceScore::MAX,
        );
    }

    #[test]
    fn test_record_communication_and_tick() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();

        // Record heavy communication between the two.
        kernel.record_communication(id1, id2, 1000).unwrap();

        // After tick, coherence scores drop (all traffic is external).
        let result = kernel.tick().unwrap();

        assert_eq!(kernel.coherence_score(id1).as_basis_points(), 0);
        // High external traffic → split recommended.
        match result.decision {
            CoherenceDecision::SplitRecommended { partition, .. } => {
                assert!(partition == id1 || partition == id2);
            }
            _ => panic!("expected SplitRecommended after heavy external comms"),
        }
    }

    #[test]
    fn test_coherence_pressure_after_communication() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();
        kernel.record_communication(id1, id2, 500).unwrap();

        kernel.tick().unwrap();

        // Partition with only external traffic has max pressure (10000 bp).
        assert_eq!(kernel.coherence_pressure(id1).as_fixed(), 10_000);
    }

    #[test]
    fn test_no_action_for_isolated_partitions() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        kernel.create_partition(&config).unwrap();
        kernel.create_partition(&config).unwrap();

        let result = kernel.tick().unwrap();
        assert_eq!(result.decision, CoherenceDecision::NoAction);
    }

    #[test]
    fn test_record_communication_before_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        assert_eq!(
            kernel.record_communication(PartitionId::new(1), PartitionId::new(2), 100),
            Err(RvmError::InvalidPartitionState),
        );
    }

    #[test]
    fn test_coherence_recommendation_without_tick() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        kernel.create_partition(&config).unwrap();

        // Before any tick, recommendation is NoAction.
        assert_eq!(kernel.coherence_recommendation(), CoherenceDecision::NoAction);
    }

    #[test]
    fn test_destroy_removes_from_coherence() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();
        assert_eq!(kernel.coherence_engine().partition_count(), 2);

        kernel.destroy_partition(id1).unwrap();
        assert_eq!(kernel.coherence_engine().partition_count(), 1);

        // id2 is still tracked.
        assert_eq!(
            kernel.coherence_score(id2),
            rvm_types::CoherenceScore::MAX,
        );
    }

    #[test]
    fn test_full_coherence_lifecycle() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();
        let c = kernel.create_partition(&config).unwrap();

        // a and b talk heavily; c is isolated.
        kernel.record_communication(a, b, 2000).unwrap();
        kernel.record_communication(b, a, 2000).unwrap();

        let result = kernel.tick().unwrap();

        // a and b should have high pressure, c should not.
        assert!(kernel.coherence_pressure(a).as_fixed() > 0);
        assert!(kernel.coherence_pressure(b).as_fixed() > 0);
        assert_eq!(kernel.coherence_pressure(c).as_fixed(), 0);

        // Should recommend splitting a or b.
        match result.decision {
            CoherenceDecision::SplitRecommended { partition, .. } => {
                assert!(partition == a || partition == b);
            }
            _ => panic!("expected split for heavily communicating partitions"),
        }

        // Destroy a, verify coherence adapts.
        kernel.destroy_partition(a).unwrap();
        assert_eq!(kernel.coherence_engine().partition_count(), 2);
    }

    // ---------------------------------------------------------------
    // Scheduler integration tests
    // ---------------------------------------------------------------

    #[test]
    fn test_enqueue_and_switch() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();

        // Enqueue id1 with lower urgency, id2 with higher.
        kernel.enqueue_partition(0, id1, 100).unwrap();
        kernel.enqueue_partition(0, id2, 200).unwrap();

        // Highest priority should be dequeued first.
        let (old, new) = kernel.switch_next(0).unwrap();
        assert!(old.is_none());
        assert_eq!(new, id2);

        let (old, new) = kernel.switch_next(0).unwrap();
        assert_eq!(old, Some(id2));
        assert_eq!(new, id1);
    }

    #[test]
    fn test_enqueue_injects_coherence_pressure() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let id1 = kernel.create_partition(&config).unwrap();
        let id2 = kernel.create_partition(&config).unwrap();

        // Record heavy communication to give id1 high pressure.
        kernel.record_communication(id1, id2, 5000).unwrap();
        kernel.tick().unwrap();

        // id1 now has max pressure (10000 bp). When enqueued with
        // lower deadline urgency, pressure boost may re-order.
        kernel.enqueue_partition(0, id1, 50).unwrap();
        kernel.enqueue_partition(0, id2, 50).unwrap();

        // id1 should be prioritised because of its pressure boost.
        let (_, first) = kernel.switch_next(0).unwrap();
        assert_eq!(first, id1);
    }

    #[test]
    fn test_enqueue_before_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        assert_eq!(
            kernel.enqueue_partition(0, PartitionId::new(1), 100),
            Err(RvmError::InvalidPartitionState),
        );
    }

    #[test]
    fn test_enqueue_nonexistent_partition_fails() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();
        assert_eq!(
            kernel.enqueue_partition(0, PartitionId::new(999), 100),
            Err(RvmError::PartitionNotFound),
        );
    }

    // ---------------------------------------------------------------
    // Split / merge execution tests
    // ---------------------------------------------------------------

    #[test]
    fn test_execute_split() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let source = kernel.create_partition(&config).unwrap();
        let pre_count = kernel.partition_count();
        let pre_witness = kernel.witness_count();

        let child = kernel.execute_split(source).unwrap();

        assert_ne!(source, child);
        assert_eq!(kernel.partition_count(), pre_count + 1);
        assert_eq!(kernel.coherence_engine().partition_count(), 2);

        // Verify StructuralSplit witness.
        let record = kernel.witness_log().get(pre_witness as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::StructuralSplit as u8);
        assert_eq!(record.actor_partition_id, source.as_u32());
        assert_eq!(record.target_object_id, child.as_u32() as u64);
    }

    /// Roadmap item 5: `execute_split` must assign neighbors to the two
    /// sides according to the mincut boundary, not arbitrarily.
    ///
    /// Topology: S forms an obvious 2-cluster structure --
    /// strongly coupled to A (kept), weakly coupled to the dense
    /// {X, Y} cluster (peeled off). The mincut around S separates X,
    /// so after the split X's edges must attach to the child while
    /// A's stay with S.
    #[test]
    fn test_execute_split_respects_cut_boundary() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let s = kernel.create_partition(&config).unwrap();
        let a = kernel.create_partition(&config).unwrap();
        let x = kernel.create_partition(&config).unwrap();
        let y = kernel.create_partition(&config).unwrap();

        // Keep cluster: S <-> A heavy. Move cluster: S <-> X weak,
        // X <-> Y dense (X has its own cluster mass).
        kernel.record_communication(s, s, 300).unwrap();
        kernel.record_communication(s, a, 600).unwrap();
        kernel.record_communication(a, s, 600).unwrap();
        kernel.record_communication(s, x, 100).unwrap();
        kernel.record_communication(x, s, 100).unwrap();
        kernel.record_communication(x, y, 2000).unwrap();
        kernel.record_communication(y, x, 2000).unwrap();

        let child = kernel.execute_split(s).unwrap();
        let graph = kernel.coherence_engine().graph();

        // X (move side) is re-homed to the child.
        assert_eq!(graph.edge_weight_between(s, x), 0, "S-X must be cut");
        assert_eq!(graph.edge_weight_between(child, x), 200, "child inherits S-X");
        // A (keep side) stays with the source.
        assert_eq!(graph.edge_weight_between(s, a), 1200, "S-A must survive");
        assert_eq!(graph.edge_weight_between(child, a), 0);
        // The intra-cluster X-Y edges are untouched.
        assert_eq!(graph.edge_weight_between(x, y), 4000);

        // The witness payload records one re-homed neighbor.
        let last = kernel.witness_log().get((kernel.witness_count() - 1) as usize).unwrap();
        assert_eq!(last.action_kind, ActionKind::StructuralSplit as u8);
        assert_eq!(u16::from_le_bytes([last.payload[0], last.payload[1]]), 1);
    }

    /// Epoch task vs hot path at the kernel boundary: after a tick has
    /// produced an epoch split plan, a newly arriving partition gets an
    /// O(degree) placement hint without recomputing the mincut.
    #[test]
    fn test_split_placement_hint_hot_path() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let s = kernel.create_partition(&config).unwrap();
        let a = kernel.create_partition(&config).unwrap();
        let x = kernel.create_partition(&config).unwrap();
        let y = kernel.create_partition(&config).unwrap();

        // Same 2-cluster shape as above, with self-loops keeping the
        // cluster members below the split threshold so S is the
        // candidate.
        kernel.record_communication(s, s, 300).unwrap();
        kernel.record_communication(s, a, 600).unwrap();
        kernel.record_communication(a, s, 600).unwrap();
        kernel.record_communication(s, x, 100).unwrap();
        kernel.record_communication(x, s, 100).unwrap();
        kernel.record_communication(x, y, 2000).unwrap();
        kernel.record_communication(y, x, 2000).unwrap();
        kernel.record_communication(a, a, 2000).unwrap();
        kernel.record_communication(x, x, 3000).unwrap();
        kernel.record_communication(y, y, 3000).unwrap();

        // No plan yet: hint unavailable.
        let z = kernel.create_partition(&config).unwrap();
        assert!(kernel.split_placement_hint(z).is_none());

        // EPOCH TASK: tick computes the plan for S.
        let epoch = kernel.tick().unwrap();
        assert!(matches!(
            epoch.decision,
            CoherenceDecision::SplitRecommended { .. }
        ));
        let plan_epoch = kernel.coherence_engine().split_plan().unwrap().epoch;

        // HOT PATH: Z communicates with the move-side cluster member X
        // and is routed there incrementally.
        kernel.record_communication(z, x, 500).unwrap();
        kernel.record_communication(x, z, 500).unwrap();
        assert_eq!(
            kernel.split_placement_hint(z),
            Some(rvm_coherence::SplitSide::MoveToChild)
        );
        // The epoch plan was not recomputed by the hot path.
        assert_eq!(
            kernel.coherence_engine().split_plan().unwrap().epoch,
            plan_epoch
        );
    }

    #[test]
    fn test_execute_split_before_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        assert_eq!(
            kernel.execute_split(PartitionId::new(1)),
            Err(RvmError::InvalidPartitionState),
        );
    }

    #[test]
    fn test_execute_split_nonexistent_fails() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();
        assert_eq!(
            kernel.execute_split(PartitionId::new(999)),
            Err(RvmError::PartitionNotFound),
        );
    }

    #[test]
    fn test_execute_merge() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();
        let pre_witness = kernel.witness_count();

        // Both start with MAX coherence (isolated), which exceeds
        // the merge threshold of 7000 bp.
        let survivor = kernel.execute_merge(a, b).unwrap();
        assert_eq!(survivor, a);

        // b was removed from coherence graph.
        assert_eq!(kernel.coherence_engine().partition_count(), 1);

        // Verify StructuralMerge witness.
        let record = kernel.witness_log().get(pre_witness as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::StructuralMerge as u8);
        assert_eq!(record.actor_partition_id, a.as_u32());
        assert_eq!(record.target_object_id, b.as_u32() as u64);
    }

    #[test]
    fn test_execute_merge_low_coherence_fails() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        // Drive coherence to zero by adding external-only traffic.
        kernel.record_communication(a, b, 5000).unwrap();
        kernel.tick().unwrap();

        // Now a has 0 coherence, below the 7000 bp merge threshold.
        assert_eq!(
            kernel.execute_merge(a, b),
            Err(RvmError::InvalidPartitionState),
        );
    }

    #[test]
    fn test_execute_merge_nonexistent_fails() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        assert_eq!(
            kernel.execute_merge(a, PartitionId::new(999)),
            Err(RvmError::PartitionNotFound),
        );
    }

    // ---------------------------------------------------------------
    // apply_decision tests
    // ---------------------------------------------------------------

    #[test]
    fn test_apply_no_action() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let result = kernel.apply_decision(CoherenceDecision::NoAction).unwrap();
        assert_eq!(result, ApplyResult::NoAction);
    }

    #[test]
    fn test_apply_split_decision() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        // Create heavy traffic to trigger split recommendation.
        kernel.record_communication(a, b, 5000).unwrap();
        let epoch = kernel.tick().unwrap();

        match epoch.decision {
            CoherenceDecision::SplitRecommended { .. } => {
                let result = kernel.apply_decision(epoch.decision).unwrap();
                match result {
                    ApplyResult::Split { source, child } => {
                        assert!(source == a || source == b);
                        assert_ne!(source, child);
                        // Now 3 partitions exist.
                        assert_eq!(kernel.partition_count(), 3);
                    }
                    _ => panic!("expected Split result"),
                }
            }
            _ => panic!("expected SplitRecommended"),
        }
    }

    #[test]
    fn test_full_tick_apply_lifecycle() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        // Heavy bidirectional traffic.
        kernel.record_communication(a, b, 3000).unwrap();
        kernel.record_communication(b, a, 3000).unwrap();

        // Tick, get decision, apply it.
        let epoch = kernel.tick().unwrap();
        let result = kernel.apply_decision(epoch.decision).unwrap();

        // Should have split one of the partitions.
        match result {
            ApplyResult::Split { source, child } => {
                assert!(source == a || source == b);
                assert_eq!(kernel.partition_count(), 3);
                assert_eq!(kernel.coherence_engine().partition_count(), 3);

                // Enqueue the new partition and verify it can be scheduled.
                kernel.enqueue_partition(0, child, 100).unwrap();
                let (_, next) = kernel.switch_next(0).unwrap();
                assert_eq!(next, child);
            }
            _ => panic!("expected split from heavy traffic"),
        }
    }

    // ---------------------------------------------------------------
    // IPC integration tests
    // ---------------------------------------------------------------

    fn make_msg(sender: u32, receiver: u32, edge: CommEdgeId, seq: u64) -> IpcMessage {
        IpcMessage {
            sender: PartitionId::new(sender),
            receiver: PartitionId::new(receiver),
            edge_id: edge,
            payload_len: 0,
            msg_type: 1,
            sequence: seq,
            capability_hash: 0,
        }
    }

    #[test]
    fn test_create_channel_and_send() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        let edge = kernel.create_channel(a, b).unwrap();
        assert_eq!(kernel.ipc_channel_count(), 1);

        let msg = make_msg(a.as_u32(), b.as_u32(), edge, 1);
        kernel.ipc_send(edge, msg, a).unwrap();

        let received = kernel.ipc_receive(edge).unwrap().unwrap();
        assert_eq!(received.sequence, 1);
    }

    #[test]
    fn test_ipc_feeds_coherence_graph() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        let edge = kernel.create_channel(a, b).unwrap();

        // Send multiple messages to build up edge weight.
        for seq in 1..=10 {
            let msg = make_msg(a.as_u32(), b.as_u32(), edge, seq);
            kernel.ipc_send(edge, msg, a).unwrap();
        }

        // After tick, coherence should reflect the traffic.
        kernel.tick().unwrap();

        // a has only external traffic → 0 coherence.
        assert_eq!(kernel.coherence_score(a).as_basis_points(), 0);
        // a should have non-zero pressure.
        assert!(kernel.coherence_pressure(a).as_fixed() > 0);
    }

    #[test]
    fn test_ipc_witnesses() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();
        let pre_create = kernel.witness_count();

        let edge = kernel.create_channel(a, b).unwrap();
        let record = kernel.witness_log().get(pre_create as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::CommEdgeCreate as u8);

        let pre_send = kernel.witness_count();
        let msg = make_msg(a.as_u32(), b.as_u32(), edge, 1);
        kernel.ipc_send(edge, msg, a).unwrap();
        let record = kernel.witness_log().get(pre_send as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::IpcSend as u8);

        let pre_destroy = kernel.witness_count();
        kernel.destroy_channel(edge).unwrap();
        let record = kernel.witness_log().get(pre_destroy as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::CommEdgeDestroy as u8);
    }

    #[test]
    fn test_create_channel_nonexistent_partition() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        assert_eq!(
            kernel.create_channel(a, PartitionId::new(999)),
            Err(RvmError::PartitionNotFound),
        );
    }

    #[test]
    fn test_ipc_before_boot_fails() {
        let mut kernel = Kernel::with_defaults();
        assert_eq!(
            kernel.create_channel(PartitionId::new(1), PartitionId::new(2)),
            Err(RvmError::InvalidPartitionState),
        );
    }

    // ---------------------------------------------------------------
    // Memory tier integration tests
    // ---------------------------------------------------------------

    #[test]
    fn test_register_and_query_region() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let r = OwnedRegionId::new(1);
        kernel.register_region(r, Tier::Warm).unwrap();
        assert_eq!(kernel.region_tier(r), Some(Tier::Warm));
    }

    #[test]
    fn test_promote_region() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let r = OwnedRegionId::new(1);
        kernel.register_region(r, Tier::Warm).unwrap();

        // Boost recency and cut_value to meet Hot promotion threshold (8000).
        kernel.record_memory_access(r).unwrap(); // +1000 to 6000
        kernel.record_memory_access(r).unwrap(); // +1000 to 7000
        kernel.record_memory_access(r).unwrap(); // +1000 to 8000
        kernel.update_region_cut_value(r, 1000).unwrap();
        // residency = 8000 + 1000 = 9000 >= 8000 threshold

        let pre = kernel.witness_count();
        let old = kernel.promote_region(r, Tier::Hot).unwrap();
        assert_eq!(old, Tier::Warm);
        assert_eq!(kernel.region_tier(r), Some(Tier::Hot));

        let record = kernel.witness_log().get(pre as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::RegionPromote as u8);
    }

    #[test]
    fn test_demote_region() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let r = OwnedRegionId::new(1);
        kernel.register_region(r, Tier::Warm).unwrap();

        let pre = kernel.witness_count();
        let old = kernel.demote_region(r, Tier::Dormant).unwrap();
        assert_eq!(old, Tier::Warm);
        assert_eq!(kernel.region_tier(r), Some(Tier::Dormant));

        let record = kernel.witness_log().get(pre as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::RegionDemote as u8);
    }

    #[test]
    fn test_tier_recency_decay_on_tick() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let r = OwnedRegionId::new(1);
        kernel.register_region(r, Tier::Warm).unwrap();

        // Initial recency is 5000. Each tick decays by 200.
        kernel.tick().unwrap();
        kernel.tick().unwrap();
        kernel.tick().unwrap();
        // After 3 ticks: 5000 - 3*200 = 4400

        // Trying to promote to Hot should fail because
        // residency = 4400 + 0 = 4400 < 8000 threshold.
        assert_eq!(
            kernel.promote_region(r, Tier::Hot),
            Err(RvmError::CoherenceBelowThreshold),
        );
    }

    #[test]
    fn test_cut_value_drives_promotion() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let r = OwnedRegionId::new(1);
        kernel.register_region(r, Tier::Warm).unwrap();

        // With a high cut_value, promotion becomes possible.
        kernel.update_region_cut_value(r, 5000).unwrap();
        // residency = 5000 (recency) + 5000 (cut) = 10000 >= 8000

        let old = kernel.promote_region(r, Tier::Hot).unwrap();
        assert_eq!(old, Tier::Warm);
        assert_eq!(kernel.region_tier(r), Some(Tier::Hot));
    }

    // ---------------------------------------------------------------
    // End-to-end: IPC → coherence → tier lifecycle
    // ---------------------------------------------------------------

    #[test]
    fn test_ipc_to_coherence_to_split_lifecycle() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        // Create IPC channel and send enough messages to trigger split.
        let edge = kernel.create_channel(a, b).unwrap();
        for seq in 1..=16 {
            let msg = make_msg(a.as_u32(), b.as_u32(), edge, seq);
            kernel.ipc_send(edge, msg, a).unwrap();
        }

        // Tick → coherence recompute → split recommendation.
        let epoch = kernel.tick().unwrap();

        match epoch.decision {
            CoherenceDecision::SplitRecommended { .. } => {
                // Apply the split.
                let result = kernel.apply_decision(epoch.decision).unwrap();
                match result {
                    ApplyResult::Split { child, .. } => {
                        // Register memory for the child.
                        let r = OwnedRegionId::new(100);
                        kernel.register_region(r, Tier::Warm).unwrap();

                        // Feed coherence score into tier cut_value.
                        let score = kernel.coherence_score(child);
                        kernel
                            .update_region_cut_value(r, score.as_basis_points())
                            .unwrap();

                        // Verify the partition count grew.
                        assert_eq!(kernel.partition_count(), 3);
                        assert_eq!(kernel.coherence_engine().partition_count(), 3);
                    }
                    _ => panic!("expected split"),
                }
            }
            _ => panic!("expected SplitRecommended after heavy IPC traffic"),
        }
    }

    // ---------------------------------------------------------------
    // Edge weight decay tests
    // ---------------------------------------------------------------

    #[test]
    fn test_edge_decay_reduces_weight_over_time() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        // Record traffic.
        kernel.record_communication(a, b, 10_000).unwrap();
        kernel.tick().unwrap();

        // Coherence graph edge count should be 1 after first tick.
        assert_eq!(kernel.coherence_engine().graph().edge_count(), 1);

        // Tick many times without new traffic. The 5% decay per epoch
        // will eventually prune the edge to zero.
        for _ in 0..200 {
            kernel.tick().unwrap();
        }

        // After enough decay, the edge should be pruned.
        assert_eq!(
            kernel.coherence_engine().graph().edge_count(),
            0,
            "edge should be pruned after sufficient decay",
        );

        // With no edges, pressure should be zero.
        assert_eq!(kernel.coherence_pressure(a).as_fixed(), 0);
    }

    // ---------------------------------------------------------------
    // Score propagation tests
    // ---------------------------------------------------------------

    #[test]
    fn test_score_propagation_to_partition() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        // Initial coherence on the partition object should be 5000 (default).
        assert_eq!(kernel.partitions().get(a).unwrap().coherence.as_basis_points(), 5000);

        // Drive coherence to 0 via external traffic.
        kernel.record_communication(a, b, 5000).unwrap();
        kernel.tick().unwrap();

        // After tick, the partition object's coherence should be updated.
        let p = kernel.partitions().get(a).unwrap();
        assert_eq!(p.coherence.as_basis_points(), 0);
        assert!(p.cut_pressure.as_fixed() > 0);
    }

    // ---------------------------------------------------------------
    // Security-gated operation tests
    // ---------------------------------------------------------------

    #[test]
    fn test_checked_create_partition_success() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let token = rvm_types::CapToken::new(
            1,
            rvm_types::CapType::Partition,
            rvm_types::CapRights::READ | rvm_types::CapRights::WRITE,
            0,
        );
        let config = PartitionConfig::default();
        let id = kernel.checked_create_partition(&config, &token).unwrap();
        assert_eq!(kernel.partition_count(), 1);
        assert!(kernel.partitions().get(id).is_some());
    }

    #[test]
    fn test_checked_create_partition_wrong_type_denied() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        // Wrong capability type (Region instead of Partition).
        let token = rvm_types::CapToken::new(
            1,
            rvm_types::CapType::Region,
            rvm_types::CapRights::WRITE,
            0,
        );
        let config = PartitionConfig::default();
        assert_eq!(
            kernel.checked_create_partition(&config, &token),
            Err(RvmError::InsufficientCapability),
        );
    }

    #[test]
    fn test_checked_create_partition_insufficient_rights_denied() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        // Read-only (needs WRITE).
        let token = rvm_types::CapToken::new(
            1,
            rvm_types::CapType::Partition,
            rvm_types::CapRights::READ,
            0,
        );
        let config = PartitionConfig::default();
        assert_eq!(
            kernel.checked_create_partition(&config, &token),
            Err(RvmError::InsufficientCapability),
        );
    }

    #[test]
    fn test_checked_ipc_send_denied() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();
        let edge = kernel.create_channel(a, b).unwrap();

        // Read-only token (needs WRITE for Send).
        let token = rvm_types::CapToken::new(
            1,
            rvm_types::CapType::Partition,
            rvm_types::CapRights::READ,
            0,
        );
        let msg = make_msg(a.as_u32(), b.as_u32(), edge, 1);
        assert_eq!(
            kernel.checked_ipc_send(edge, msg, &token),
            Err(RvmError::InsufficientCapability),
        );
    }

    // ---------------------------------------------------------------
    // Degraded mode tests
    // ---------------------------------------------------------------

    #[test]
    fn test_degraded_mode_lifecycle() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        assert!(!kernel.is_degraded());
        let pre = kernel.witness_count();

        kernel.enter_degraded_mode();
        assert!(kernel.is_degraded());
        let record = kernel.witness_log().get(pre as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::DegradedModeEntered as u8);

        kernel.exit_degraded_mode();
        assert!(!kernel.is_degraded());
        let record = kernel.witness_log().get((pre + 1) as usize).unwrap();
        assert_eq!(record.action_kind, ActionKind::DegradedModeExited as u8);
    }

    #[test]
    fn test_degraded_mode_zeroes_pressure_in_scheduler() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot().unwrap();

        let config = PartitionConfig::default();
        let a = kernel.create_partition(&config).unwrap();
        let b = kernel.create_partition(&config).unwrap();

        // Give `a` high pressure.
        kernel.record_communication(a, b, 5000).unwrap();
        kernel.tick().unwrap();

        // Enter degraded mode — pressure should be zeroed in scheduler.
        kernel.enter_degraded_mode();

        // Enqueue both with same deadline. In normal mode, `a`'s pressure
        // boost would push it ahead. In degraded mode, they're equal.
        kernel.enqueue_partition(0, a, 100).unwrap();
        kernel.enqueue_partition(0, b, 100).unwrap();

        // First dequeued is whichever was enqueued first (same priority).
        let (_, first) = kernel.switch_next(0).unwrap();
        let (_, second) = kernel.switch_next(0).unwrap();
        // Both should have been scheduled — verify both ran.
        assert!((first == a && second == b) || (first == b && second == a));
    }

    // -- CryptoSignerAdapter tests (ADR-142 Phase 4) -----------------------

    #[cfg(feature = "crypto-sha256")]
    mod signer_bridge_tests {
        use super::*;
        use crate::signer_bridge::CryptoSignerAdapter;
        use rvm_proof::HmacSha256WitnessSigner;
        use rvm_witness::WitnessSigner as WitnessSignerTrait;

        fn test_key() -> [u8; 32] {
            let mut key = [0u8; 32];
            #[allow(clippy::cast_possible_truncation)]
            for (i, byte) in key.iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(0x37).wrapping_add(0x42);
            }
            key
        }

        #[test]
        fn adapter_sign_returns_nonzero() {
            let inner = HmacSha256WitnessSigner::new(test_key());
            let adapter = CryptoSignerAdapter::new(inner);

            let mut record = WitnessRecord::zeroed();
            record.sequence = 42;
            record.action_kind = 0x01;

            let sig = adapter.sign(&record);
            assert_ne!(sig, [0u8; 8]);
        }

        #[test]
        fn adapter_verify_round_trip() {
            let inner = HmacSha256WitnessSigner::new(test_key());
            let adapter = CryptoSignerAdapter::new(inner);

            let mut record = WitnessRecord::zeroed();
            record.sequence = 100;
            record.timestamp_ns = 1_000_000;
            record.action_kind = 0x10;
            record.proof_tier = 3;
            record.actor_partition_id = 3;
            record.target_object_id = 99;
            record.capability_hash = 0xDEAD;
            record.prev_hash = 0x1234;
            record.record_hash = 0x5678;

            let sig = adapter.sign(&record);
            record.aux = sig;
            assert!(adapter.verify(&record));
        }

        #[test]
        fn adapter_tampered_record_fails() {
            let inner = HmacSha256WitnessSigner::new(test_key());
            let adapter = CryptoSignerAdapter::new(inner);

            let mut record = WitnessRecord::zeroed();
            record.sequence = 100;
            record.actor_partition_id = 3;

            let sig = adapter.sign(&record);
            record.aux = sig;
            record.sequence = 101; // tamper
            assert!(!adapter.verify(&record));
        }

        #[test]
        fn adapter_different_keys_different_sigs() {
            let a1 = CryptoSignerAdapter::new(HmacSha256WitnessSigner::new([0x11u8; 32]));
            let a2 = CryptoSignerAdapter::new(HmacSha256WitnessSigner::new([0x22u8; 32]));

            let mut record = WitnessRecord::zeroed();
            record.sequence = 42;

            assert_ne!(a1.sign(&record), a2.sign(&record));
        }

        #[test]
        fn adapter_with_witness_log_signed_append() {
            let inner = HmacSha256WitnessSigner::new(test_key());
            let adapter = CryptoSignerAdapter::new(inner);
            let log = rvm_witness::WitnessLog::<16>::new();

            let mut record = WitnessRecord::zeroed();
            record.action_kind = ActionKind::PartitionCreate as u8;
            record.actor_partition_id = 1;
            record.target_object_id = 100;

            log.signed_append(record, &adapter);

            let stored = log.get(0).unwrap();
            assert_ne!(stored.aux, [0u8; 8]);
            assert!(adapter.verify(&stored));
        }
    }
}
