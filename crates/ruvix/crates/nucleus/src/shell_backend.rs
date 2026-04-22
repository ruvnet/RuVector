//! Shell backend implementation for the RuVix Kernel.
//!
//! This module provides the `ShellBackend` trait implementation for the `Kernel`,
//! enabling the debug shell to inspect real kernel state.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use ruvix_shell::{
    CapEntry, CpuInfo, KernelInfo, MemoryStats, PerfCounters, ProofStats, QueueStats, ShellBackend,
    TaskInfo, TaskState as ShellTaskState, VectorStats, WitnessEntry,
};

use crate::{scheduler::TaskState, Kernel};

/// Kernel version string for shell display.
const KERNEL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build timestamp (compile time).
const BUILD_TIME: &str = env!("CARGO_PKG_VERSION");

impl ShellBackend for Kernel {
    fn kernel_info(&self) -> KernelInfo {
        KernelInfo {
            version: KERNEL_VERSION,
            build_time: BUILD_TIME,
            boot_time_ns: self.boot_time_ns(),
            current_time_ns: self.current_time_ns(),
            cpu_count: 1, // Single CPU for now, SMP in future
        }
    }

    fn memory_stats(&self) -> MemoryStats {
        // Get region stats from region manager
        // For now, provide simulated stats based on kernel state
        let vector_memory = self.vector_store_memory_bytes();
        let graph_memory = self.graph_store_memory_bytes();
        let used_bytes = vector_memory + graph_memory;

        MemoryStats {
            total_bytes: 1024 * 1024 * 1024, // 1 GiB (would come from DTB)
            used_bytes,
            free_bytes: 1024 * 1024 * 1024 - used_bytes,
            region_count: self.region_count() as u32,
            slab_count: 0,          // Slab allocator not implemented yet
            peak_bytes: used_bytes, // Track peak in future
        }
    }

    fn task_list(&self) -> Vec<TaskInfo> {
        self.scheduler()
            .iter_tasks()
            .map(|tcb| {
                let mut name = [0u8; 16];
                // For now, just use task ID as name
                let id_str = format_task_id(tcb.handle.raw().id);
                let len = id_str.len().min(16);
                name[..len].copy_from_slice(&id_str.as_bytes()[..len]);

                TaskInfo {
                    id: tcb.handle.raw().id,
                    name,
                    state: convert_task_state(tcb.state),
                    priority: tcb.priority as u8,
                    partition: 0,       // Partitioning not implemented yet
                    cpu_affinity: 0xFF, // All CPUs
                    cap_count: 0,       // Would query cap manager
                }
            })
            .collect()
    }

    fn cpu_info(&self) -> Vec<CpuInfo> {
        // Single CPU for now
        // In SMP mode, this would query each CPU core
        alloc::vec![CpuInfo {
            id: 0,
            online: true,
            is_primary: true,
            freq_mhz: 1800, // Would come from DTB or cpuid
            load_percent: self.estimate_cpu_load(),
        }]
    }

    fn queue_stats(&self) -> QueueStats {
        // Queue manager stats would be integrated here
        // For now, return placeholder stats based on kernel activity
        let stats = self.stats();
        QueueStats {
            queue_count: 0, // Queue manager not fully integrated
            pending_messages: 0,
            messages_sent: stats.syscalls_executed / 10, // Estimate
            messages_received: stats.syscalls_executed / 10,
            zero_copy_count: 0,
        }
    }

    fn vector_stats(&self) -> VectorStats {
        let (store_count, vector_count, total_dims, memory_bytes, reads, writes) =
            self.aggregate_vector_stats();

        VectorStats {
            store_count: store_count as u32,
            vector_count,
            total_dimensions: total_dims,
            memory_bytes,
            reads,
            writes,
        }
    }

    fn proof_stats(&self) -> ProofStats {
        let stats = self.stats();
        let proof_stats = self.proof_engine_stats();

        ProofStats {
            generated: stats.proofs_verified + stats.proofs_rejected,
            verified: stats.proofs_verified,
            rejected: stats.proofs_rejected,
            cache_entries: proof_stats.cache_entries as u32,
            cache_hits: proof_stats.cache_hits,
            cache_misses: proof_stats.cache_misses,
            tier0_count: proof_stats.tier0_count,
            tier1_count: proof_stats.tier1_count,
            tier2_count: proof_stats.tier2_count,
        }
    }

    fn capability_entries(&self, task_id: Option<u32>) -> Vec<CapEntry> {
        // Query capability manager for entries
        self.list_capabilities(task_id)
            .into_iter()
            .map(|cap| CapEntry {
                handle: cap.handle,
                object_id: cap.object_id,
                object_type: cap.object_type,
                rights: cap.rights,
                badge: cap.badge,
                depth: cap.depth,
            })
            .collect()
    }

    fn witness_entries(&self, count: usize) -> Vec<WitnessEntry> {
        self.witness_log()
            .iter()
            .rev()
            .take(count)
            .map(|record| WitnessEntry {
                seq: record.sequence,
                timestamp_ns: record.timestamp_ns,
                operation: record.kind as u8,
                object_id: record.resource_id,
                hash_prefix: {
                    let mut prefix = [0u8; 8];
                    prefix.copy_from_slice(&record.mutation_hash[..8]);
                    prefix
                },
            })
            .collect()
    }

    fn perf_counters(&self) -> PerfCounters {
        let stats = self.stats();
        let sched_stats = self.scheduler().stats();

        PerfCounters {
            syscalls: stats.syscalls_executed,
            context_switches: sched_stats.context_switches,
            interrupts: 0,                           // Not tracked at kernel level yet
            page_faults: 0,                          // MMU not integrated yet
            ipi_sent: 0,                             // SMP not implemented yet
            cpu_cycles: self.current_time_ns() / 10, // Rough estimate
        }
    }

    fn trace_enabled(&self) -> bool {
        self.syscall_tracing_enabled()
    }

    fn set_trace(&mut self, enabled: bool) {
        self.set_syscall_tracing(enabled);
    }

    fn reboot(&mut self) {
        // In real hardware, this would trigger a system reset
        // For now, log the request and reset kernel state
        self.trigger_reboot();
    }
}

// ============================================================================
// Kernel Extension Methods for Shell Support
// ============================================================================

impl Kernel {
    /// Returns the boot time in nanoseconds.
    #[inline]
    pub(crate) fn boot_time_ns(&self) -> u64 {
        self.boot_time_ns
    }

    /// Returns the number of regions.
    pub(crate) fn region_count(&self) -> usize {
        // Would query region manager
        0
    }

    /// Returns total memory used by vector stores.
    pub(crate) fn vector_store_memory_bytes(&self) -> u64 {
        #[cfg(feature = "alloc")]
        {
            self.vector_stores.iter().map(|s| s.memory_bytes()).sum()
        }
        #[cfg(not(feature = "alloc"))]
        {
            let mut total = 0u64;
            for i in 0..self.vector_store_count {
                if let Some(ref store) = self.vector_stores[i] {
                    total += store.memory_bytes();
                }
            }
            total
        }
    }

    /// Returns total memory used by graph stores.
    pub(crate) fn graph_store_memory_bytes(&self) -> u64 {
        #[cfg(feature = "alloc")]
        {
            self.graph_stores.iter().map(|s| s.memory_bytes()).sum()
        }
        #[cfg(not(feature = "alloc"))]
        {
            let mut total = 0u64;
            for i in 0..self.graph_store_count {
                if let Some(ref store) = self.graph_stores[i] {
                    total += store.memory_bytes();
                }
            }
            total
        }
    }

    /// Aggregates vector store statistics.
    pub(crate) fn aggregate_vector_stats(&self) -> (usize, u64, u32, u64, u64, u64) {
        let mut store_count = 0usize;
        let mut vector_count = 0u64;
        let mut total_dims = 0u32;
        let mut memory_bytes = 0u64;
        let mut reads = 0u64;
        let mut writes = 0u64;

        #[cfg(feature = "alloc")]
        {
            for store in &self.vector_stores {
                store_count += 1;
                let stats = store.stats();
                vector_count += stats.entry_count as u64;
                total_dims += store.config().dimensions;
                memory_bytes += store.memory_bytes();
                reads += stats.reads;
                writes += stats.writes;
            }
        }
        #[cfg(not(feature = "alloc"))]
        {
            for i in 0..self.vector_store_count {
                if let Some(ref store) = self.vector_stores[i] {
                    store_count += 1;
                    let stats = store.stats();
                    vector_count += stats.entry_count as u64;
                    total_dims += store.config().dimensions;
                    memory_bytes += store.memory_bytes();
                    reads += stats.reads;
                    writes += stats.writes;
                }
            }
        }

        (
            store_count,
            vector_count,
            total_dims,
            memory_bytes,
            reads,
            writes,
        )
    }

    /// Returns proof engine statistics for shell display.
    pub(crate) fn proof_engine_stats(&self) -> ProofEngineShellStats {
        let stats = self.proof_engine.stats();
        ProofEngineShellStats {
            cache_entries: 0, // Cache not implemented yet
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
            tier0_count: stats.proofs_verified, // Approximate breakdown
            tier1_count: 0,
            tier2_count: 0,
        }
    }

    /// Lists capabilities, optionally filtered by task.
    pub(crate) fn list_capabilities(&self, _task_id: Option<u32>) -> Vec<CapabilityShellEntry> {
        // Would query cap_manager for real entries
        // For now, return empty list
        Vec::new()
    }

    /// Estimates CPU load based on scheduler activity.
    pub(crate) fn estimate_cpu_load(&self) -> u8 {
        let stats = self.scheduler().stats();
        // Simple estimate: more context switches = higher load
        let switches_per_sec =
            stats.context_switches.saturating_mul(1_000_000_000) / self.current_time_ns().max(1);

        // Cap at 100%
        switches_per_sec.min(100) as u8
    }

    /// Returns whether syscall tracing is enabled.
    pub(crate) fn syscall_tracing_enabled(&self) -> bool {
        self.trace_enabled
    }

    /// Sets syscall tracing state.
    pub(crate) fn set_syscall_tracing(&mut self, enabled: bool) {
        self.trace_enabled = enabled;
    }

    /// Triggers a system reboot.
    pub(crate) fn trigger_reboot(&mut self) {
        // Record reboot in witness log
        let _ = self.witness_log.record_checkpoint([0xFFu8; 32], u64::MAX);

        // In bare metal, this would:
        // 1. Save any persistent state
        // 2. Trigger watchdog or reset register
        // For hosted mode, we just reset kernel state
        self.reset_state();
    }

    /// Resets kernel state (for reboot).
    fn reset_state(&mut self) {
        // Clear stores
        #[cfg(feature = "alloc")]
        {
            self.vector_stores.clear();
            self.graph_stores.clear();
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.vector_store_count = 0;
            self.graph_store_count = 0;
        }

        // Reset IDs
        self.next_vector_store_id = 1;
        self.next_graph_store_id = 1;
        self.next_checkpoint_seq = 1;

        // Reset time
        self.current_time_ns = 0;

        // Reset stats
        self.stats = crate::KernelStats::default();
    }
}

// ============================================================================
// Helper Types
// ============================================================================

/// Proof engine statistics for shell display.
pub(crate) struct ProofEngineShellStats {
    pub cache_entries: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub tier0_count: u64,
    pub tier1_count: u64,
    pub tier2_count: u64,
}

/// Capability entry for shell display.
#[derive(Debug, Clone)]
pub(crate) struct CapabilityShellEntry {
    pub handle: u32,
    pub object_id: u64,
    pub object_type: u8,
    pub rights: u32,
    pub badge: u64,
    pub depth: u8,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Converts kernel TaskState to shell TaskState.
fn convert_task_state(state: TaskState) -> ShellTaskState {
    match state {
        TaskState::Ready => ShellTaskState::Ready,
        TaskState::Running => ShellTaskState::Running,
        TaskState::Blocked => ShellTaskState::Blocked,
        TaskState::Sleeping => ShellTaskState::Sleeping,
        TaskState::Terminated => ShellTaskState::Exited,
    }
}

/// Formats a task ID for display.
fn format_task_id(id: u32) -> alloc::string::String {
    alloc::format!("task-{}", id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelConfig;

    #[test]
    fn test_shell_backend_kernel_info() {
        let kernel = Kernel::new(KernelConfig::default());
        let info = kernel.kernel_info();

        assert!(!info.version.is_empty());
        assert_eq!(info.cpu_count, 1);
    }

    #[test]
    fn test_shell_backend_memory_stats() {
        let kernel = Kernel::new(KernelConfig::default());
        let stats = kernel.memory_stats();

        assert!(stats.total_bytes > 0);
        assert!(stats.free_bytes <= stats.total_bytes);
    }

    #[test]
    fn test_shell_backend_task_list() {
        let mut kernel = Kernel::new(KernelConfig::default());

        // Create some tasks
        kernel
            .scheduler_mut()
            .create_task(crate::TaskPriority::Normal, None)
            .unwrap();
        kernel
            .scheduler_mut()
            .create_task(crate::TaskPriority::High, None)
            .unwrap();

        let tasks = kernel.task_list();
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn test_shell_backend_proof_stats() {
        let kernel = Kernel::new(KernelConfig::default());
        let stats = kernel.proof_stats();

        // Initial stats should be zero
        assert_eq!(stats.verified, 0);
        assert_eq!(stats.rejected, 0);
    }

    #[test]
    fn test_shell_backend_trace_toggle() {
        let mut kernel = Kernel::new(KernelConfig::default());

        assert!(!kernel.trace_enabled());
        kernel.set_trace(true);
        assert!(kernel.trace_enabled());
        kernel.set_trace(false);
        assert!(!kernel.trace_enabled());
    }

    #[test]
    fn test_shell_backend_witness_entries() {
        let mut kernel = Kernel::new(KernelConfig::default());
        kernel.boot(0, [0u8; 32]).unwrap();

        let entries = kernel.witness_entries(10);
        assert!(!entries.is_empty());
    }
}
