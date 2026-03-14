//! Comparison framework for RuVix vs Linux benchmarks.
//!
//! This module provides structured comparisons between RuVix syscalls
//! and their Linux equivalents.

use crate::{BenchmarkResult, Comparison, MemoryComparison};
use crate::ruvix::BenchConfig;
use crate::linux::LinuxBenchConfig;

/// Comparison mapping between RuVix and Linux operations.
#[derive(Debug, Clone)]
pub struct ComparisonMapping {
    /// Human-readable operation name.
    pub operation: &'static str,
    /// RuVix syscall name.
    pub ruvix_syscall: &'static str,
    /// Linux equivalent operation.
    pub linux_equivalent: &'static str,
    /// Expected speedup (RuVix advantage).
    pub expected_speedup: f64,
    /// Notes about the comparison.
    pub notes: &'static str,
}

/// All comparison mappings.
pub const COMPARISON_MAPPINGS: &[ComparisonMapping] = &[
    ComparisonMapping {
        operation: "Capability Grant",
        ruvix_syscall: "cap_grant",
        linux_equivalent: "linux_setuid_simulation",
        expected_speedup: 16.0,
        notes: "O(1) capability lookup vs O(n) DAC/MAC check",
    },
    ComparisonMapping {
        operation: "IPC Send",
        ruvix_syscall: "queue_send",
        linux_equivalent: "linux_pipe_write",
        expected_speedup: 6.0,
        notes: "Zero-copy vs kernel buffer copy",
    },
    ComparisonMapping {
        operation: "IPC Receive",
        ruvix_syscall: "queue_recv",
        linux_equivalent: "linux_pipe_read",
        expected_speedup: 6.0,
        notes: "Zero-copy vs kernel buffer copy",
    },
    ComparisonMapping {
        operation: "Memory Map",
        ruvix_syscall: "region_map",
        linux_equivalent: "linux_mmap",
        expected_speedup: 3.0,
        notes: "Region-based vs page table walk",
    },
    ComparisonMapping {
        operation: "Task Spawn",
        ruvix_syscall: "task_spawn",
        linux_equivalent: "linux_clone_simulation",
        expected_speedup: 4.0,
        notes: "Bounded preemption vs full clone",
    },
    ComparisonMapping {
        operation: "Timer Wait",
        ruvix_syscall: "timer_wait",
        linux_equivalent: "linux_clock_gettime",
        expected_speedup: 1.0,
        notes: "Similar overhead for time operations",
    },
    ComparisonMapping {
        operation: "Security Check",
        ruvix_syscall: "cap_grant",
        linux_equivalent: "linux_selinux_simulation",
        expected_speedup: 16.0,
        notes: "O(1) capability vs O(n) policy evaluation",
    },
    ComparisonMapping {
        operation: "Durable Write",
        ruvix_syscall: "vector_put_proved",
        linux_equivalent: "linux_write_fsync",
        expected_speedup: 10.0,
        notes: "Proof + memory vs disk fsync",
    },
    ComparisonMapping {
        operation: "Socket IPC",
        ruvix_syscall: "queue_send",
        linux_equivalent: "linux_socket_send",
        expected_speedup: 6.0,
        notes: "Zero-copy vs socket buffer",
    },
];

/// Memory overhead comparison specifications.
pub const MEMORY_COMPARISONS: &[(&str, &str, &str)] = &[
    ("IPC Buffer", "No kernel/user copy (zero-copy)", "Two copies: user->kernel, kernel->user"),
    ("Page Tables", "Region-based (no page tables)", "Per-process page tables (~4KB min)"),
    ("Capability Table", "Fixed-size slab (64B per cap)", "Variable-size inode/dentry cache"),
    ("Proof Cache", "LRU cache (configurable)", "No equivalent (SELinux policy in kernel)"),
    ("Task State", "Minimal TCB (~256B)", "Full task_struct (~2KB)"),
];

/// Runs all comparisons and returns results.
#[cfg(unix)]
pub fn run_all_comparisons(
    ruvix_config: &BenchConfig,
    linux_config: &LinuxBenchConfig,
) -> Vec<Comparison> {
    use crate::ruvix;
    use crate::linux;

    let mut comparisons = Vec::new();

    // cap_grant vs setuid
    let ruvix_cap = ruvix::bench_cap_grant(ruvix_config);
    let linux_cap = linux::bench_linux_setuid_simulation(linux_config);
    comparisons.push(Comparison::new(
        "Capability Grant",
        "cap_grant",
        "setuid simulation",
        ruvix_cap,
        linux_cap,
        "O(1) capability lookup vs O(n) credential check",
    ));

    // queue_send vs pipe_write
    let ruvix_send = ruvix::bench_queue_send(ruvix_config);
    let linux_pipe = linux::bench_linux_pipe_write(linux_config);
    comparisons.push(Comparison::new(
        "IPC Send",
        "queue_send",
        "pipe write",
        ruvix_send,
        linux_pipe,
        "Zero-copy vs kernel buffer copy",
    ));

    // queue_recv vs pipe_read
    let ruvix_recv = ruvix::bench_queue_recv(ruvix_config);
    let linux_read = linux::bench_linux_pipe_read(linux_config);
    comparisons.push(Comparison::new(
        "IPC Receive",
        "queue_recv",
        "pipe read",
        ruvix_recv,
        linux_read,
        "Zero-copy vs kernel buffer copy",
    ));

    // region_map vs mmap
    let ruvix_region = ruvix::bench_region_map(ruvix_config);
    let linux_mmap = linux::bench_linux_mmap(linux_config);
    comparisons.push(Comparison::new(
        "Memory Map",
        "region_map",
        "mmap",
        ruvix_region,
        linux_mmap,
        "Region-based vs page table setup",
    ));

    // task_spawn vs clone
    let ruvix_spawn = ruvix::bench_task_spawn(ruvix_config);
    let linux_clone = linux::bench_linux_clone_simulation(linux_config);
    comparisons.push(Comparison::new(
        "Task Spawn",
        "task_spawn",
        "clone simulation",
        ruvix_spawn,
        linux_clone,
        "Bounded preemption vs full clone",
    ));

    // timer_wait vs clock_gettime
    let ruvix_timer = ruvix::bench_timer_wait(ruvix_config);
    let linux_clock = linux::bench_linux_clock_gettime(linux_config);
    comparisons.push(Comparison::new(
        "Timer Wait",
        "timer_wait",
        "clock_gettime",
        ruvix_timer,
        linux_clock,
        "Scheduler integration vs VDSO call",
    ));

    // cap_grant vs SELinux
    let ruvix_cap2 = ruvix::bench_cap_grant(ruvix_config);
    let linux_selinux = linux::bench_linux_selinux_simulation(linux_config);
    comparisons.push(Comparison::new(
        "Security Check",
        "cap_grant",
        "SELinux simulation",
        ruvix_cap2,
        linux_selinux,
        "O(1) capability vs O(n) policy evaluation",
    ));

    // vector_put_proved vs write+fsync
    let ruvix_vector = ruvix::bench_vector_put_proved(ruvix_config);
    let linux_fsync = linux::bench_linux_write_fsync(linux_config);
    comparisons.push(Comparison::new(
        "Durable Write",
        "vector_put_proved",
        "write+fsync",
        ruvix_vector,
        linux_fsync,
        "Proof + memory vs disk I/O",
    ));

    // queue_send vs socket_send
    let ruvix_queue = ruvix::bench_queue_send(ruvix_config);
    let linux_socket = linux::bench_linux_socket_send(linux_config);
    comparisons.push(Comparison::new(
        "Socket vs Queue",
        "queue_send",
        "unix socket send",
        ruvix_queue,
        linux_socket,
        "Zero-copy vs socket buffer",
    ));

    comparisons
}

#[cfg(not(unix))]
pub fn run_all_comparisons(
    _ruvix_config: &BenchConfig,
    _linux_config: &LinuxBenchConfig,
) -> Vec<Comparison> {
    Vec::new()
}

/// Generates memory overhead comparisons.
pub fn generate_memory_comparisons() -> Vec<MemoryComparison> {
    vec![
        MemoryComparison::new(
            "IPC Buffer (8B message)",
            8,      // RuVix: zero-copy, just pointer
            16384,  // Linux: pipe buffer minimum
            "Zero-copy eliminates buffer allocation",
        ),
        MemoryComparison::new(
            "Capability Entry",
            64,     // RuVix: fixed slab
            512,    // Linux: inode + dentry cache entry
            "Fixed-size slab vs variable allocation",
        ),
        MemoryComparison::new(
            "Task Control Block",
            256,    // RuVix: minimal TCB
            2048,   // Linux: task_struct
            "Minimal state vs full process context",
        ),
        MemoryComparison::new(
            "Memory Region Descriptor",
            32,     // RuVix: region descriptor
            128,    // Linux: vm_area_struct
            "Region-based vs VMA linked list",
        ),
        MemoryComparison::new(
            "Proof Token",
            82,     // RuVix: 82-byte attestation
            0,      // Linux: no equivalent
            "No Linux equivalent for proof",
        ),
        MemoryComparison::new(
            "Page Table (4KB region)",
            0,      // RuVix: no page tables
            4096,   // Linux: minimum page table
            "Region-based eliminates page tables",
        ),
    ]
}

/// Summary statistics for all comparisons.
#[derive(Debug, Clone)]
pub struct ComparisonSummary {
    /// Number of comparisons where RuVix is faster.
    pub ruvix_wins: usize,
    /// Number of comparisons where Linux is faster.
    pub linux_wins: usize,
    /// Number of comparisons within 10% (tie).
    pub ties: usize,
    /// Average speedup (RuVix over Linux).
    pub avg_speedup: f64,
    /// Maximum speedup observed.
    pub max_speedup: f64,
    /// Minimum speedup (or slowdown if <1).
    pub min_speedup: f64,
    /// Total memory reduction percentage.
    pub total_memory_reduction: f64,
}

impl ComparisonSummary {
    /// Creates a summary from comparison results.
    pub fn from_comparisons(comparisons: &[Comparison], memory: &[MemoryComparison]) -> Self {
        let mut ruvix_wins = 0;
        let mut linux_wins = 0;
        let mut ties = 0;
        let mut speedups = Vec::new();

        for comp in comparisons {
            speedups.push(comp.speedup);
            if comp.speedup > 1.1 {
                ruvix_wins += 1;
            } else if comp.speedup < 0.9 {
                linux_wins += 1;
            } else {
                ties += 1;
            }
        }

        let avg_speedup = if !speedups.is_empty() {
            speedups.iter().sum::<f64>() / speedups.len() as f64
        } else {
            1.0
        };

        let max_speedup = speedups.iter().cloned().fold(0.0, f64::max);
        let min_speedup = speedups.iter().cloned().fold(f64::INFINITY, f64::min);

        let total_memory_reduction = if !memory.is_empty() {
            memory.iter().map(|m| m.reduction).sum::<f64>() / memory.len() as f64
        } else {
            0.0
        };

        Self {
            ruvix_wins,
            linux_wins,
            ties,
            avg_speedup,
            max_speedup,
            min_speedup: if min_speedup.is_infinite() { 1.0 } else { min_speedup },
            total_memory_reduction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_mappings() {
        assert!(!COMPARISON_MAPPINGS.is_empty());

        for mapping in COMPARISON_MAPPINGS {
            assert!(!mapping.operation.is_empty());
            assert!(!mapping.ruvix_syscall.is_empty());
            assert!(!mapping.linux_equivalent.is_empty());
            assert!(mapping.expected_speedup > 0.0);
        }
    }

    #[test]
    fn test_memory_comparisons() {
        let comparisons = generate_memory_comparisons();
        assert!(!comparisons.is_empty());

        for comp in &comparisons {
            assert!(!comp.operation.is_empty());
            // RuVix should use less memory or equal in all cases
            assert!(comp.ruvix_bytes <= comp.linux_bytes || comp.linux_bytes == 0);
        }
    }

    #[test]
    fn test_comparison_summary() {
        let comparisons = vec![
            Comparison::new(
                "test1",
                "ruvix_op",
                "linux_op",
                BenchmarkResult::from_measurements("r", &[100.0], None),
                BenchmarkResult::from_measurements("l", &[600.0], None),
                "test",
            ),
        ];
        let memory = generate_memory_comparisons();
        let summary = ComparisonSummary::from_comparisons(&comparisons, &memory);

        assert_eq!(summary.ruvix_wins, 1);
        assert_eq!(summary.linux_wins, 0);
        assert!(summary.avg_speedup > 1.0);
    }
}
