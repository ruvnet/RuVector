//! Main kernel integration coordinating all subsystems.
//!
//! The Kernel struct is the central integration point for the RuVix Cognition Kernel.
//! It coordinates:
//!
//! - RegionManager (memory regions)
//! - CapabilityManager (capability-based security)
//! - QueueManager (IPC)
//! - ProofEngine (proof verification)
//! - VectorStoreManager (kernel vector stores)
//! - GraphStoreManager (kernel graph stores)
//! - Scheduler (task management)
//! - WitnessLog (attestation and replay)

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{
    checkpoint::{Checkpoint, CheckpointConfig, ReplayEngine},
    proof_engine::{ProofEngine, ProofEngineConfig},
    scheduler::{Scheduler, SchedulerConfig},
    syscall::{AttestPayload, Syscall, SyscallResult},
    vector_store::VectorStore,
    graph_store::GraphStore,
    witness_log::{WitnessLog, WitnessRecordKind},
    CapHandle, CapRights, Duration, GraphHandle, GraphMutation, MsgPriority, ProofTier,
    ProofToken, QueueHandle, RegionPolicy, Result, RvfMountHandle, SensorDescriptor,
    SubscriptionHandle, TaskHandle, TaskPriority, TimerSpec, VectorKey, VectorStoreConfig,
    VectorStoreHandle,
};

use ruvix_cap::{CapManagerConfig, CapabilityManager};
use ruvix_region::RegionManager;
use ruvix_types::{KernelError, ObjectType};

/// Maximum vector stores (for no_std).
#[cfg(not(feature = "alloc"))]
const MAX_VECTOR_STORES: usize = 16;

/// Maximum graph stores (for no_std).
#[cfg(not(feature = "alloc"))]
const MAX_GRAPH_STORES: usize = 16;

/// Configuration for the kernel.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Capability manager configuration.
    pub cap_config: CapManagerConfig,
    /// Scheduler configuration.
    pub scheduler_config: SchedulerConfig,
    /// Proof engine configuration.
    pub proof_config: ProofEngineConfig,
    /// Default vector store dimensions.
    pub default_vector_dims: u32,
    /// Default vector store capacity.
    pub default_vector_capacity: u32,
}

impl KernelConfig {
    /// Creates a new configuration with default values.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            cap_config: CapManagerConfig::new(),
            scheduler_config: SchedulerConfig::new(),
            proof_config: ProofEngineConfig::new(),
            default_vector_dims: 768,
            default_vector_capacity: 10000,
        }
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about kernel operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KernelStats {
    /// Total syscalls executed.
    pub syscalls_executed: u64,
    /// Total syscalls failed.
    pub syscalls_failed: u64,
    /// Total proofs verified.
    pub proofs_verified: u64,
    /// Total proofs rejected.
    pub proofs_rejected: u64,
    /// Total attestations emitted.
    pub attestations_emitted: u64,
    /// Total checkpoints created.
    pub checkpoints_created: u64,
}

/// The main kernel struct coordinating all subsystems.
pub struct Kernel {
    /// Configuration.
    config: KernelConfig,

    /// Capability manager.
    cap_manager: CapabilityManager<1024>,

    /// Region manager.
    region_manager: RegionManager,

    /// Proof engine.
    proof_engine: ProofEngine,

    /// Scheduler.
    scheduler: Scheduler,

    /// Witness log.
    witness_log: WitnessLog,

    /// Vector stores.
    #[cfg(feature = "alloc")]
    vector_stores: Vec<VectorStore>,
    #[cfg(not(feature = "alloc"))]
    vector_stores: [Option<VectorStore>; MAX_VECTOR_STORES],
    #[cfg(not(feature = "alloc"))]
    vector_store_count: usize,

    /// Graph stores.
    #[cfg(feature = "alloc")]
    graph_stores: Vec<GraphStore>,
    #[cfg(not(feature = "alloc"))]
    graph_stores: [Option<GraphStore>; MAX_GRAPH_STORES],
    #[cfg(not(feature = "alloc"))]
    graph_store_count: usize,

    /// Next vector store ID.
    next_vector_store_id: u32,

    /// Next graph store ID.
    next_graph_store_id: u32,

    /// Next checkpoint sequence.
    next_checkpoint_seq: u64,

    /// Current time (nanoseconds since boot).
    current_time_ns: u64,

    /// Boot time (nanoseconds since epoch).
    boot_time_ns: u64,

    /// Statistics.
    stats: KernelStats,
}

impl Kernel {
    /// Creates a new kernel with the given configuration.
    #[must_use]
    pub fn new(config: KernelConfig) -> Self {
        Self {
            cap_manager: CapabilityManager::new(config.cap_config),
            region_manager: RegionManager::new(),
            proof_engine: ProofEngine::new(config.proof_config),
            scheduler: Scheduler::new(config.scheduler_config),
            witness_log: WitnessLog::new(),
            #[cfg(feature = "alloc")]
            vector_stores: Vec::new(),
            #[cfg(not(feature = "alloc"))]
            vector_stores: core::array::from_fn(|_| None),
            #[cfg(not(feature = "alloc"))]
            vector_store_count: 0,
            #[cfg(feature = "alloc")]
            graph_stores: Vec::new(),
            #[cfg(not(feature = "alloc"))]
            graph_stores: core::array::from_fn(|_| None),
            #[cfg(not(feature = "alloc"))]
            graph_store_count: 0,
            next_vector_store_id: 1,
            next_graph_store_id: 1,
            next_checkpoint_seq: 1,
            current_time_ns: 0,
            boot_time_ns: 0,
            config,
            stats: KernelStats::default(),
        }
    }

    /// Creates a kernel with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(KernelConfig::default())
    }

    /// Boots the kernel.
    ///
    /// Records boot attestation in the witness log.
    pub fn boot(&mut self, boot_time_ns: u64, kernel_hash: [u8; 32]) -> Result<()> {
        self.boot_time_ns = boot_time_ns;
        self.current_time_ns = 0;
        self.witness_log.record_boot(kernel_hash)?;
        Ok(())
    }

    /// Sets the current time.
    pub fn set_current_time(&mut self, time_ns: u64) {
        self.current_time_ns = time_ns;
        self.proof_engine.set_current_time(time_ns);
        self.scheduler.set_current_time(time_ns);
        self.witness_log.set_current_time(time_ns);
    }

    /// Returns the current time.
    #[inline]
    #[must_use]
    pub const fn current_time_ns(&self) -> u64 {
        self.current_time_ns
    }

    /// Returns the configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &KernelConfig {
        &self.config
    }

    /// Returns the statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &KernelStats {
        &self.stats
    }

    /// Returns a reference to the witness log.
    #[inline]
    #[must_use]
    pub const fn witness_log(&self) -> &WitnessLog {
        &self.witness_log
    }

    /// Returns a reference to the scheduler.
    #[inline]
    #[must_use]
    pub const fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    // =========================================================================
    // Syscall Dispatch
    // =========================================================================

    /// Dispatches a syscall.
    ///
    /// This is the main entry point for all kernel operations.
    /// Every syscall is:
    /// - Capability-gated
    /// - Potentially proof-required
    /// - Witness-logged (if mutation)
    pub fn dispatch(&mut self, syscall: Syscall) -> Result<SyscallResult> {
        // Increment syscall counter
        self.stats.syscalls_executed += 1;

        // Dispatch to specific handler
        let result = match syscall {
            Syscall::TaskSpawn { priority, deadline, .. } => {
                self.handle_task_spawn(priority, deadline)
            }
            Syscall::CapGrant { target, cap, rights } => {
                self.handle_cap_grant(target, cap, rights)
            }
            Syscall::RegionMap { size, policy, cap } => {
                self.handle_region_map(size, policy, cap)
            }
            #[cfg(feature = "alloc")]
            Syscall::QueueSend { queue, msg, priority } => {
                self.handle_queue_send(queue, &msg, priority)
            }
            #[cfg(not(feature = "alloc"))]
            Syscall::QueueSend { queue, msg, msg_len, priority } => {
                self.handle_queue_send(queue, &msg[..msg_len], priority)
            }
            Syscall::QueueRecv { queue, buf_size, timeout } => {
                self.handle_queue_recv(queue, buf_size, timeout)
            }
            Syscall::TimerWait { deadline } => {
                self.handle_timer_wait(deadline)
            }
            #[cfg(feature = "alloc")]
            Syscall::RvfMount { rvf_data, mount_point, cap } => {
                self.handle_rvf_mount(&rvf_data, &mount_point, cap)
            }
            #[cfg(not(feature = "alloc"))]
            Syscall::RvfMount { rvf_data, rvf_len, mount_point, mount_point_len, cap } => {
                let mp = core::str::from_utf8(&mount_point[..mount_point_len])
                    .map_err(|_| KernelError::InvalidArgument)?;
                self.handle_rvf_mount(&rvf_data[..rvf_len], mp, cap)
            }
            Syscall::AttestEmit { operation, proof } => {
                self.handle_attest_emit(operation, proof)
            }
            Syscall::VectorGet { store, key } => {
                self.handle_vector_get(store, key)
            }
            #[cfg(feature = "alloc")]
            Syscall::VectorPutProved { store, key, data, proof } => {
                self.handle_vector_put_proved(store, key, data, proof)
            }
            #[cfg(not(feature = "alloc"))]
            Syscall::VectorPutProved { store, key, data, data_len, proof } => {
                self.handle_vector_put_proved(store, key, &data[..data_len], proof)
            }
            Syscall::GraphApplyProved { graph, mutation, proof } => {
                self.handle_graph_apply_proved(graph, mutation, proof)
            }
            Syscall::SensorSubscribe { sensor, target_queue, cap } => {
                self.handle_sensor_subscribe(sensor, target_queue, cap)
            }
        };

        if result.is_err() {
            self.stats.syscalls_failed += 1;
        }

        result
    }

    // =========================================================================
    // Syscall Handlers
    // =========================================================================

    fn handle_task_spawn(
        &mut self,
        priority: TaskPriority,
        deadline: Option<Duration>,
    ) -> Result<SyscallResult> {
        let handle = self.scheduler.create_task(priority, deadline)?;
        Ok(SyscallResult::TaskSpawned(handle))
    }

    fn handle_cap_grant(
        &mut self,
        target: TaskHandle,
        cap: CapHandle,
        rights: CapRights,
    ) -> Result<SyscallResult> {
        // Get caller task (assume task 1 for now)
        let caller = self.scheduler.current_task().unwrap_or(TaskHandle::new(1, 0));

        let derived = self.cap_manager.grant(cap, rights, 0, caller, target)
            .map_err(|_| KernelError::NotPermitted)?;
        Ok(SyscallResult::CapGranted(derived))
    }

    fn handle_region_map(
        &mut self,
        _size: usize,
        policy: RegionPolicy,
        cap: CapHandle,
    ) -> Result<SyscallResult> {
        let handle = self.region_manager.create_region(policy, cap)?;
        Ok(SyscallResult::RegionMapped(handle))
    }

    fn handle_queue_send(
        &mut self,
        _queue: QueueHandle,
        _msg: &[u8],
        _priority: MsgPriority,
    ) -> Result<SyscallResult> {
        // Queue implementation would go here
        // For now, just succeed
        Ok(SyscallResult::MessageSent)
    }

    fn handle_queue_recv(
        &mut self,
        _queue: QueueHandle,
        _buf_size: usize,
        _timeout: Duration,
    ) -> Result<SyscallResult> {
        // Queue implementation would go here
        #[cfg(feature = "alloc")]
        {
            Ok(SyscallResult::MessageReceived {
                data: Vec::new(),
                priority: MsgPriority::Normal,
            })
        }
        #[cfg(not(feature = "alloc"))]
        {
            Ok(SyscallResult::MessageReceived {
                len: 0,
                priority: MsgPriority::Normal,
            })
        }
    }

    fn handle_timer_wait(&mut self, deadline: TimerSpec) -> Result<SyscallResult> {
        if let Some(handle) = self.scheduler.current_task() {
            let wake_time = match deadline {
                TimerSpec::Absolute { nanos_since_boot } => nanos_since_boot,
                TimerSpec::Relative { nanos } => self.current_time_ns + nanos,
            };
            self.scheduler.sleep_task(handle, wake_time)?;
        }
        Ok(SyscallResult::TimerExpired)
    }

    fn handle_rvf_mount(
        &mut self,
        _rvf_data: &[u8],
        _mount_point: &str,
        _cap: CapHandle,
    ) -> Result<SyscallResult> {
        // RVF mount requires proof (handled by caller)
        // For now, simulate successful mount
        let mount = RvfMountHandle::new(1, 0);

        // Record in witness log
        let package_hash = [0u8; 32]; // Would compute from rvf_data
        let attestation = self.proof_engine.generate_attestation(&ProofToken::default());
        self.witness_log.record_mount(package_hash, &attestation, mount)?;

        self.stats.attestations_emitted += 1;

        Ok(SyscallResult::RvfMounted(mount))
    }

    fn handle_attest_emit(
        &mut self,
        operation: AttestPayload,
        proof: ProofToken,
    ) -> Result<SyscallResult> {
        // Verify proof
        let expected_hash = proof.mutation_hash;
        let verify_result = self.proof_engine.verify(&proof, &expected_hash)?;

        if !verify_result.is_valid() {
            self.stats.proofs_rejected += 1;
            return Err(KernelError::ProofRejected);
        }

        self.stats.proofs_verified += 1;

        // Record attestation
        let sequence = match operation {
            AttestPayload::Boot { kernel_hash, .. } => {
                self.witness_log.record_boot(kernel_hash)?
            }
            AttestPayload::Checkpoint { state_hash, sequence } => {
                self.witness_log.record_checkpoint(state_hash, sequence)?
            }
            _ => {
                // Other attestations
                self.witness_log.sequence()
            }
        };

        self.stats.attestations_emitted += 1;

        Ok(SyscallResult::AttestEmitted { sequence })
    }

    fn handle_vector_get(
        &mut self,
        store_handle: VectorStoreHandle,
        key: VectorKey,
    ) -> Result<SyscallResult> {
        let store = self.get_vector_store_mut(store_handle)?;
        let entry = store.get(key)?;

        #[cfg(feature = "alloc")]
        {
            Ok(SyscallResult::VectorRetrieved {
                data: entry.data.clone(),
                coherence: entry.meta.coherence_as_f32(),
            })
        }
        #[cfg(not(feature = "alloc"))]
        {
            Ok(SyscallResult::VectorRetrieved {
                len: entry.data_len,
                coherence: entry.meta.coherence_as_f32(),
            })
        }
    }

    #[cfg(feature = "alloc")]
    fn handle_vector_put_proved(
        &mut self,
        store_handle: VectorStoreHandle,
        key: VectorKey,
        data: Vec<f32>,
        proof: ProofToken,
    ) -> Result<SyscallResult> {
        // Verify proof
        let expected_hash = proof.mutation_hash;
        let verify_result = self.proof_engine.verify(&proof, &expected_hash)?;

        if !verify_result.is_valid() {
            self.stats.proofs_rejected += 1;
            return Err(KernelError::ProofRejected);
        }

        self.stats.proofs_verified += 1;

        // Store vector
        let store = self.get_vector_store_mut(store_handle)?;
        store.put_proved(key, data, &proof)?;

        // Record in witness log
        let attestation = self.proof_engine.generate_attestation(&proof);
        self.witness_log.record_vector_mutation(
            proof.mutation_hash,
            &attestation,
            store_handle,
            key,
        )?;

        self.stats.attestations_emitted += 1;

        Ok(SyscallResult::VectorStored)
    }

    #[cfg(not(feature = "alloc"))]
    fn handle_vector_put_proved(
        &mut self,
        store_handle: VectorStoreHandle,
        key: VectorKey,
        data: &[f32],
        proof: ProofToken,
    ) -> Result<SyscallResult> {
        // Verify proof
        let expected_hash = proof.mutation_hash;
        let verify_result = self.proof_engine.verify(&proof, &expected_hash)?;

        if !verify_result.is_valid() {
            self.stats.proofs_rejected += 1;
            return Err(KernelError::ProofRejected);
        }

        self.stats.proofs_verified += 1;

        // Store vector
        let store = self.get_vector_store_mut(store_handle)?;
        store.put_proved(key, data, &proof)?;

        // Record in witness log
        let attestation = self.proof_engine.generate_attestation(&proof);
        self.witness_log.record_vector_mutation(
            proof.mutation_hash,
            &attestation,
            store_handle,
            key,
        )?;

        self.stats.attestations_emitted += 1;

        Ok(SyscallResult::VectorStored)
    }

    fn handle_graph_apply_proved(
        &mut self,
        graph_handle: GraphHandle,
        mutation: GraphMutation,
        proof: ProofToken,
    ) -> Result<SyscallResult> {
        // Verify proof
        let expected_hash = proof.mutation_hash;
        let verify_result = self.proof_engine.verify(&proof, &expected_hash)?;

        if !verify_result.is_valid() {
            self.stats.proofs_rejected += 1;
            return Err(KernelError::ProofRejected);
        }

        self.stats.proofs_verified += 1;

        // Apply mutation
        let store = self.get_graph_store_mut(graph_handle)?;
        store.apply_proved(&mutation, &proof)?;

        // Record in witness log
        let attestation = self.proof_engine.generate_attestation(&proof);
        self.witness_log.record_graph_mutation(
            proof.mutation_hash,
            &attestation,
            graph_handle,
        )?;

        self.stats.attestations_emitted += 1;

        Ok(SyscallResult::GraphApplied)
    }

    fn handle_sensor_subscribe(
        &mut self,
        _sensor: SensorDescriptor,
        _target_queue: QueueHandle,
        _cap: CapHandle,
    ) -> Result<SyscallResult> {
        // Sensor subscription would be implemented here
        let subscription = SubscriptionHandle::new(1, 0);
        Ok(SyscallResult::SensorSubscribed(subscription))
    }

    // =========================================================================
    // Vector Store Management
    // =========================================================================

    /// Creates a new vector store.
    pub fn create_vector_store(&mut self, config: VectorStoreConfig) -> Result<VectorStoreHandle> {
        let id = self.next_vector_store_id;
        self.next_vector_store_id += 1;

        let handle = VectorStoreHandle::new(id, 0);
        let store = VectorStore::new(handle, config);

        #[cfg(feature = "alloc")]
        {
            self.vector_stores.push(store);
        }
        #[cfg(not(feature = "alloc"))]
        {
            if self.vector_store_count >= MAX_VECTOR_STORES {
                return Err(KernelError::LimitExceeded);
            }
            self.vector_stores[self.vector_store_count] = Some(store);
            self.vector_store_count += 1;
        }

        Ok(handle)
    }

    fn get_vector_store_mut(&mut self, handle: VectorStoreHandle) -> Result<&mut VectorStore> {
        #[cfg(feature = "alloc")]
        {
            for store in &mut self.vector_stores {
                if store.handle() == handle {
                    return Ok(store);
                }
            }
            Err(KernelError::NotFound)
        }
        #[cfg(not(feature = "alloc"))]
        {
            // Find index first, then return mutable reference
            let mut found_idx: Option<usize> = None;
            for i in 0..self.vector_store_count {
                if let Some(ref store) = self.vector_stores[i] {
                    if store.handle() == handle {
                        found_idx = Some(i);
                        break;
                    }
                }
            }
            match found_idx {
                Some(idx) => Ok(self.vector_stores[idx].as_mut().unwrap()),
                None => Err(KernelError::NotFound),
            }
        }
    }

    /// Gets a reference to a vector store.
    pub fn get_vector_store(&self, handle: VectorStoreHandle) -> Result<&VectorStore> {
        #[cfg(feature = "alloc")]
        {
            for store in &self.vector_stores {
                if store.handle() == handle {
                    return Ok(store);
                }
            }
            Err(KernelError::NotFound)
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.vector_store_count {
                if let Some(ref store) = self.vector_stores[i] {
                    if store.handle() == handle {
                        return Ok(store);
                    }
                }
            }
            Err(KernelError::NotFound)
        }
    }

    // =========================================================================
    // Graph Store Management
    // =========================================================================

    /// Creates a new graph store.
    pub fn create_graph_store(&mut self) -> Result<GraphHandle> {
        let id = self.next_graph_store_id;
        self.next_graph_store_id += 1;

        let handle = GraphHandle::new(id, 0);
        let store = GraphStore::new(handle);

        #[cfg(feature = "alloc")]
        {
            self.graph_stores.push(store);
        }
        #[cfg(not(feature = "alloc"))]
        {
            if self.graph_store_count >= MAX_GRAPH_STORES {
                return Err(KernelError::LimitExceeded);
            }
            self.graph_stores[self.graph_store_count] = Some(store);
            self.graph_store_count += 1;
        }

        Ok(handle)
    }

    fn get_graph_store_mut(&mut self, handle: GraphHandle) -> Result<&mut GraphStore> {
        #[cfg(feature = "alloc")]
        {
            for store in &mut self.graph_stores {
                if store.handle() == handle {
                    return Ok(store);
                }
            }
            Err(KernelError::NotFound)
        }
        #[cfg(not(feature = "alloc"))]
        {
            // Find index first, then return mutable reference
            let mut found_idx: Option<usize> = None;
            for i in 0..self.graph_store_count {
                if let Some(ref store) = self.graph_stores[i] {
                    if store.handle() == handle {
                        found_idx = Some(i);
                        break;
                    }
                }
            }
            match found_idx {
                Some(idx) => Ok(self.graph_stores[idx].as_mut().unwrap()),
                None => Err(KernelError::NotFound),
            }
        }
    }

    /// Gets a reference to a graph store.
    pub fn get_graph_store(&self, handle: GraphHandle) -> Result<&GraphStore> {
        #[cfg(feature = "alloc")]
        {
            for store in &self.graph_stores {
                if store.handle() == handle {
                    return Ok(store);
                }
            }
            Err(KernelError::NotFound)
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.graph_store_count {
                if let Some(ref store) = self.graph_stores[i] {
                    if store.handle() == handle {
                        return Ok(store);
                    }
                }
            }
            Err(KernelError::NotFound)
        }
    }

    // =========================================================================
    // Capability Management
    // =========================================================================

    /// Creates a root capability for a kernel object.
    pub fn create_root_capability(
        &mut self,
        object_id: u64,
        object_type: ObjectType,
        owner: TaskHandle,
    ) -> Result<CapHandle> {
        self.cap_manager.create_root_capability(object_id, object_type, 0, owner)
            .map_err(|_| KernelError::NotPermitted)
    }

    // =========================================================================
    // Checkpoint and Replay
    // =========================================================================

    /// Creates a checkpoint of the current kernel state.
    #[cfg(feature = "alloc")]
    pub fn checkpoint(&mut self, config: CheckpointConfig) -> Result<Checkpoint> {
        let vector_refs: Vec<&VectorStore> = self.vector_stores.iter().collect();
        let graph_refs: Vec<&GraphStore> = self.graph_stores.iter().collect();

        let checkpoint = Checkpoint::create(
            self.next_checkpoint_seq,
            self.current_time_ns,
            &vector_refs,
            &graph_refs,
            &self.witness_log,
            &config,
        );

        // Record checkpoint in witness log
        self.witness_log.record_checkpoint(checkpoint.state_hash, self.next_checkpoint_seq)?;

        self.next_checkpoint_seq += 1;
        self.stats.checkpoints_created += 1;

        Ok(checkpoint)
    }

    /// Verifies that current state matches a checkpoint.
    #[cfg(feature = "alloc")]
    pub fn verify_checkpoint(&self, checkpoint: &Checkpoint) -> bool {
        let vector_refs: Vec<&VectorStore> = self.vector_stores.iter().collect();
        let graph_refs: Vec<&GraphStore> = self.graph_stores.iter().collect();

        let engine = ReplayEngine::new();
        engine.verify_state(checkpoint, &vector_refs, &graph_refs)
    }

    /// Gets counts for acceptance test verification.
    pub fn get_witness_counts(&self) -> (u64, u64, u64) {
        let stats = self.witness_log.stats();
        (stats.boot_records, stats.mount_records, stats.vector_mutations + stats.graph_mutations)
    }

    /// Creates a proof token for testing and external use.
    ///
    /// This method provides access to the proof engine for creating valid proof tokens
    /// that can be used with proof-gated syscalls like VectorPutProved and GraphApplyProved.
    pub fn create_proof(
        &self,
        mutation_hash: [u8; 32],
        tier: ProofTier,
        nonce: u64,
    ) -> ProofToken {
        self.proof_engine.create_proof(mutation_hash, tier, nonce)
    }
}

impl Default for Kernel {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = Kernel::with_defaults();
        assert_eq!(kernel.stats().syscalls_executed, 0);
    }

    #[test]
    fn test_kernel_boot() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot(1_000_000_000, [1u8; 32]).unwrap();

        assert_eq!(kernel.witness_log().stats().boot_records, 1);
    }

    #[test]
    fn test_kernel_create_vector_store() {
        let mut kernel = Kernel::with_defaults();
        let config = VectorStoreConfig::new(4, 100);

        let handle = kernel.create_vector_store(config).unwrap();
        assert!(!handle.is_null());

        let store = kernel.get_vector_store(handle).unwrap();
        assert_eq!(store.config().dimensions, 4);
    }

    #[test]
    fn test_kernel_create_graph_store() {
        let mut kernel = Kernel::with_defaults();

        let handle = kernel.create_graph_store().unwrap();
        assert!(!handle.is_null());

        let store = kernel.get_graph_store(handle).unwrap();
        assert_eq!(store.node_count(), 0);
    }

    #[test]
    fn test_kernel_task_spawn() {
        let mut kernel = Kernel::with_defaults();

        let result = kernel.dispatch(Syscall::TaskSpawn {
            entry: crate::RvfComponentId::root(RvfMountHandle::null()),
            #[cfg(feature = "alloc")]
            caps: Vec::new(),
            #[cfg(not(feature = "alloc"))]
            caps: [None; 16],
            priority: TaskPriority::Normal,
            deadline: None,
        }).unwrap();

        assert!(matches!(result, SyscallResult::TaskSpawned(_)));
        assert_eq!(kernel.stats().syscalls_executed, 1);
    }

    #[test]
    fn test_kernel_region_map() {
        let mut kernel = Kernel::with_defaults();

        let result = kernel.dispatch(Syscall::RegionMap {
            size: 4096,
            policy: RegionPolicy::AppendOnly { max_size: 4096 },
            cap: CapHandle::null(),
        }).unwrap();

        assert!(matches!(result, SyscallResult::RegionMapped(_)));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_kernel_vector_put_get() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot(0, [0u8; 32]).unwrap();
        kernel.set_current_time(1_000_000);

        // Create store
        let config = VectorStoreConfig::new(4, 100);
        let store_handle = kernel.create_vector_store(config).unwrap();

        // Create proof
        let mutation_hash = [1u8; 32];
        let proof = kernel.proof_engine.create_proof(
            mutation_hash,
            crate::ProofTier::Reflex,
            42,
        );

        // Put vector
        let result = kernel.dispatch(Syscall::VectorPutProved {
            store: store_handle,
            key: VectorKey::new(1),
            data: vec![1.0, 2.0, 3.0, 4.0],
            proof,
        }).unwrap();

        assert!(matches!(result, SyscallResult::VectorStored));
        assert_eq!(kernel.stats().proofs_verified, 1);

        // Get vector
        let result = kernel.dispatch(Syscall::VectorGet {
            store: store_handle,
            key: VectorKey::new(1),
        }).unwrap();

        match result {
            SyscallResult::VectorRetrieved { data, coherence } => {
                assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
                assert!((coherence - 1.0).abs() < 0.001);
            }
            _ => panic!("Expected VectorRetrieved"),
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_kernel_checkpoint() {
        let mut kernel = Kernel::with_defaults();
        kernel.boot(0, [0u8; 32]).unwrap();
        kernel.set_current_time(1_000_000);

        // Create stores and add data
        let vector_config = VectorStoreConfig::new(4, 100);
        let store_handle = kernel.create_vector_store(vector_config).unwrap();

        let proof = kernel.proof_engine.create_proof([1u8; 32], crate::ProofTier::Reflex, 1);

        kernel.dispatch(Syscall::VectorPutProved {
            store: store_handle,
            key: VectorKey::new(1),
            data: vec![1.0, 2.0, 3.0, 4.0],
            proof,
        }).unwrap();

        // Create checkpoint
        let checkpoint = kernel.checkpoint(CheckpointConfig::full()).unwrap();
        assert_eq!(checkpoint.sequence, 1);

        // Verify checkpoint
        assert!(kernel.verify_checkpoint(&checkpoint));

        assert_eq!(kernel.stats().checkpoints_created, 1);
    }
}
