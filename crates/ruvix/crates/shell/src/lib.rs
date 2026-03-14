//! # RuVix In-Kernel Debug Shell
//!
//! This crate provides an in-kernel debug shell for the RuVix Cognition Kernel
//! as specified in ADR-087. It enables runtime inspection of kernel state,
//! memory statistics, task information, and proof subsystem status.
//!
//! ## Design Principles
//!
//! - **`#![no_std]` compatible**: Uses only `alloc` for dynamic allocation
//! - **Line-based parsing**: Simple command parser suitable for serial consoles
//! - **Trait-based backend**: `ShellBackend` trait for kernel integration
//! - **Modular commands**: Each command in its own module for maintainability
//!
//! ## Available Commands
//!
//! | Command | Description |
//! |---------|-------------|
//! | `help` | Show available commands |
//! | `info` | Kernel version, boot time, uptime |
//! | `mem` | Memory statistics |
//! | `tasks` | Task listing |
//! | `caps` | Capability table dump |
//! | `queues` | Queue statistics |
//! | `vectors` | Vector store info |
//! | `proofs` | Proof statistics |
//! | `cpu` | CPU info for SMP |
//! | `witness` | Witness log viewer |
//! | `perf` | Performance counters |
//! | `trace` | Syscall tracing toggle |
//! | `reboot` | Trigger reboot |
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_shell::{Shell, ShellBackend, ShellConfig};
//!
//! struct MyKernel { /* kernel state */ }
//!
//! impl ShellBackend for MyKernel {
//!     // Implement trait methods...
//! }
//!
//! let config = ShellConfig::default();
//! let mut shell = Shell::new(config);
//! let output = shell.execute_line("help", &kernel);
//! ```

#![no_std]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

pub mod commands;
mod parser;

pub use parser::{Command, ParseError, Parser};

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;

/// Shell configuration options.
#[derive(Debug, Clone)]
pub struct ShellConfig {
    /// Maximum command history size.
    pub max_history: usize,
    /// Whether to echo commands.
    pub echo: bool,
    /// Shell prompt string.
    pub prompt: &'static str,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            max_history: 32,
            echo: true,
            prompt: "ruvix> ",
        }
    }
}

/// Memory statistics returned by the shell backend.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total physical memory in bytes.
    pub total_bytes: u64,
    /// Used memory in bytes.
    pub used_bytes: u64,
    /// Free memory in bytes.
    pub free_bytes: u64,
    /// Number of allocated regions.
    pub region_count: u32,
    /// Number of slab allocations.
    pub slab_count: u32,
    /// Peak memory usage in bytes.
    pub peak_bytes: u64,
}

/// Task information returned by the shell backend.
#[derive(Debug, Clone)]
pub struct TaskInfo {
    /// Task handle/ID.
    pub id: u32,
    /// Task name (if available).
    pub name: [u8; 16],
    /// Task state (running, blocked, etc.).
    pub state: TaskState,
    /// Priority level.
    pub priority: u8,
    /// Partition ID.
    pub partition: u8,
    /// CPU affinity mask.
    pub cpu_affinity: u8,
    /// Capability count.
    pub cap_count: u16,
}

/// Task state enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskState {
    /// Task is runnable.
    Ready = 0,
    /// Task is currently running.
    Running = 1,
    /// Task is blocked on a queue.
    Blocked = 2,
    /// Task is waiting on a timer.
    Sleeping = 3,
    /// Task has exited.
    Exited = 4,
}

/// CPU information for SMP.
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// CPU ID (0-255).
    pub id: u8,
    /// Whether CPU is online.
    pub online: bool,
    /// Whether CPU is the boot CPU.
    pub is_primary: bool,
    /// CPU frequency in MHz (if available).
    pub freq_mhz: u32,
    /// Current CPU load percentage (0-100).
    pub load_percent: u8,
}

/// Queue statistics.
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Total number of queues.
    pub queue_count: u32,
    /// Total messages pending across all queues.
    pub pending_messages: u64,
    /// Total messages sent.
    pub messages_sent: u64,
    /// Total messages received.
    pub messages_received: u64,
    /// Total zero-copy transfers.
    pub zero_copy_count: u64,
}

/// Vector store statistics.
#[derive(Debug, Clone, Default)]
pub struct VectorStats {
    /// Number of vector stores.
    pub store_count: u32,
    /// Total vectors stored.
    pub vector_count: u64,
    /// Total dimensions across all stores.
    pub total_dimensions: u32,
    /// Total memory used by vectors in bytes.
    pub memory_bytes: u64,
    /// Total vector reads.
    pub reads: u64,
    /// Total vector writes.
    pub writes: u64,
}

/// Proof subsystem statistics.
#[derive(Debug, Clone, Default)]
pub struct ProofStats {
    /// Total proofs generated.
    pub generated: u64,
    /// Total proofs verified.
    pub verified: u64,
    /// Total proofs rejected.
    pub rejected: u64,
    /// Current cache entries.
    pub cache_entries: u32,
    /// Cache hit count.
    pub cache_hits: u64,
    /// Cache miss count.
    pub cache_misses: u64,
    /// Reflex tier proofs.
    pub tier0_count: u64,
    /// Standard tier proofs.
    pub tier1_count: u64,
    /// Deep tier proofs.
    pub tier2_count: u64,
}

/// Capability table entry information.
#[derive(Debug, Clone)]
pub struct CapEntry {
    /// Capability handle.
    pub handle: u32,
    /// Object ID.
    pub object_id: u64,
    /// Object type.
    pub object_type: u8,
    /// Rights bitmap.
    pub rights: u32,
    /// Badge value.
    pub badge: u64,
    /// Delegation depth.
    pub depth: u8,
}

/// Witness log entry.
#[derive(Debug, Clone)]
pub struct WitnessEntry {
    /// Sequence number.
    pub seq: u64,
    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Operation type.
    pub operation: u8,
    /// Associated object ID.
    pub object_id: u64,
    /// Witness hash (first 8 bytes).
    pub hash_prefix: [u8; 8],
}

/// Performance counter data.
#[derive(Debug, Clone, Default)]
pub struct PerfCounters {
    /// Syscall count.
    pub syscalls: u64,
    /// Context switches.
    pub context_switches: u64,
    /// Interrupts handled.
    pub interrupts: u64,
    /// Page faults.
    pub page_faults: u64,
    /// IPI messages sent.
    pub ipi_sent: u64,
    /// Total CPU cycles (if available).
    pub cpu_cycles: u64,
}

/// Kernel information.
#[derive(Debug, Clone)]
pub struct KernelInfo {
    /// Kernel version string.
    pub version: &'static str,
    /// Build timestamp.
    pub build_time: &'static str,
    /// Boot time in nanoseconds since epoch.
    pub boot_time_ns: u64,
    /// Current time in nanoseconds since epoch.
    pub current_time_ns: u64,
    /// Number of online CPUs.
    pub cpu_count: u8,
}

/// Trait for kernel integration with the debug shell.
///
/// Implementors provide access to kernel state for each command.
pub trait ShellBackend {
    /// Get kernel information.
    fn kernel_info(&self) -> KernelInfo;

    /// Get memory statistics.
    fn memory_stats(&self) -> MemoryStats;

    /// Get task list.
    fn task_list(&self) -> Vec<TaskInfo>;

    /// Get CPU information for all CPUs.
    fn cpu_info(&self) -> Vec<CpuInfo>;

    /// Get queue statistics.
    fn queue_stats(&self) -> QueueStats;

    /// Get vector store statistics.
    fn vector_stats(&self) -> VectorStats;

    /// Get proof subsystem statistics.
    fn proof_stats(&self) -> ProofStats;

    /// Get capability table entries for a task.
    fn capability_entries(&self, task_id: Option<u32>) -> Vec<CapEntry>;

    /// Get recent witness log entries.
    fn witness_entries(&self, count: usize) -> Vec<WitnessEntry>;

    /// Get performance counters.
    fn perf_counters(&self) -> PerfCounters;

    /// Check if syscall tracing is enabled.
    fn trace_enabled(&self) -> bool;

    /// Toggle syscall tracing.
    fn set_trace(&mut self, enabled: bool);

    /// Trigger system reboot.
    fn reboot(&mut self);
}

/// The debug shell processor.
#[derive(Debug)]
pub struct Shell {
    config: ShellConfig,
    parser: Parser,
    history: Vec<String>,
}

impl Shell {
    /// Create a new shell with the given configuration.
    #[must_use]
    pub fn new(config: ShellConfig) -> Self {
        Self {
            config,
            parser: Parser::new(),
            history: Vec::new(),
        }
    }

    /// Create a shell with default configuration.
    #[must_use]
    pub fn default_shell() -> Self {
        Self::new(ShellConfig::default())
    }

    /// Get the shell prompt.
    #[must_use]
    pub fn prompt(&self) -> &'static str {
        self.config.prompt
    }

    /// Execute a command line and return the output.
    pub fn execute_line<B: ShellBackend>(&mut self, line: &str, backend: &mut B) -> String {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            return String::new();
        }

        // Add to history
        if self.history.len() >= self.config.max_history {
            self.history.remove(0);
        }
        self.history.push(String::from(trimmed));

        // Parse the command
        match self.parser.parse(trimmed) {
            Ok(cmd) => self.dispatch_command(cmd, backend),
            Err(e) => format!("Error: {}\nType 'help' for available commands.", e),
        }
    }

    /// Dispatch a parsed command to the appropriate handler.
    fn dispatch_command<B: ShellBackend>(&self, cmd: Command, backend: &mut B) -> String {
        match cmd {
            Command::Help => commands::help::execute(),
            Command::Info => commands::info::execute(backend),
            Command::Mem => commands::mem::execute(backend),
            Command::Tasks => commands::tasks::execute(backend),
            Command::Caps { task_id } => commands::caps::execute(backend, task_id),
            Command::Queues => commands::queues::execute(backend),
            Command::Vectors => commands::vectors::execute(backend),
            Command::Proofs => commands::proofs::execute(backend),
            Command::Cpu => commands::cpu::execute(backend),
            Command::Witness { count } => commands::witness::execute(backend, count),
            Command::Perf => commands::perf::execute(backend),
            Command::Trace { enable } => {
                if let Some(on) = enable {
                    backend.set_trace(on);
                    if on {
                        String::from("Syscall tracing enabled.")
                    } else {
                        String::from("Syscall tracing disabled.")
                    }
                } else {
                    if backend.trace_enabled() {
                        String::from("Syscall tracing: ENABLED")
                    } else {
                        String::from("Syscall tracing: DISABLED")
                    }
                }
            }
            Command::Reboot => {
                backend.reboot();
                String::from("Rebooting...")
            }
        }
    }

    /// Get command history.
    #[must_use]
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Clear command history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Mock backend for testing.
    struct MockBackend {
        trace_on: bool,
        rebooted: bool,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                trace_on: false,
                rebooted: false,
            }
        }
    }

    impl ShellBackend for MockBackend {
        fn kernel_info(&self) -> KernelInfo {
            KernelInfo {
                version: "0.1.0-test",
                build_time: "2024-01-01T00:00:00Z",
                boot_time_ns: 1_000_000_000,
                current_time_ns: 2_000_000_000,
                cpu_count: 4,
            }
        }

        fn memory_stats(&self) -> MemoryStats {
            MemoryStats {
                total_bytes: 1024 * 1024 * 1024, // 1 GiB
                used_bytes: 256 * 1024 * 1024,   // 256 MiB
                free_bytes: 768 * 1024 * 1024,   // 768 MiB
                region_count: 42,
                slab_count: 128,
                peak_bytes: 512 * 1024 * 1024,
            }
        }

        fn task_list(&self) -> Vec<TaskInfo> {
            vec![
                TaskInfo {
                    id: 0,
                    name: *b"idle\0\0\0\0\0\0\0\0\0\0\0\0",
                    state: TaskState::Running,
                    priority: 0,
                    partition: 0,
                    cpu_affinity: 0xFF,
                    cap_count: 4,
                },
                TaskInfo {
                    id: 1,
                    name: *b"init\0\0\0\0\0\0\0\0\0\0\0\0",
                    state: TaskState::Ready,
                    priority: 100,
                    partition: 0,
                    cpu_affinity: 0xFF,
                    cap_count: 16,
                },
            ]
        }

        fn cpu_info(&self) -> Vec<CpuInfo> {
            vec![
                CpuInfo {
                    id: 0,
                    online: true,
                    is_primary: true,
                    freq_mhz: 1800,
                    load_percent: 25,
                },
            ]
        }

        fn queue_stats(&self) -> QueueStats {
            QueueStats {
                queue_count: 8,
                pending_messages: 12,
                messages_sent: 1000,
                messages_received: 988,
                zero_copy_count: 500,
            }
        }

        fn vector_stats(&self) -> VectorStats {
            VectorStats {
                store_count: 2,
                vector_count: 10000,
                total_dimensions: 768,
                memory_bytes: 30 * 1024 * 1024,
                reads: 50000,
                writes: 10000,
            }
        }

        fn proof_stats(&self) -> ProofStats {
            ProofStats {
                generated: 15000,
                verified: 14950,
                rejected: 50,
                cache_entries: 64,
                cache_hits: 12000,
                cache_misses: 3000,
                tier0_count: 10000,
                tier1_count: 4500,
                tier2_count: 500,
            }
        }

        fn capability_entries(&self, _task_id: Option<u32>) -> Vec<CapEntry> {
            vec![
                CapEntry {
                    handle: 0,
                    object_id: 0x1000,
                    object_type: 1,
                    rights: 0x07,
                    badge: 0,
                    depth: 0,
                },
            ]
        }

        fn witness_entries(&self, count: usize) -> Vec<WitnessEntry> {
            (0..count.min(5))
                .map(|i| WitnessEntry {
                    seq: i as u64,
                    timestamp_ns: 1_000_000_000 + i as u64 * 1000,
                    operation: 1,
                    object_id: 0x1000 + i as u64,
                    hash_prefix: [0xAB; 8],
                })
                .collect()
        }

        fn perf_counters(&self) -> PerfCounters {
            PerfCounters {
                syscalls: 100000,
                context_switches: 5000,
                interrupts: 250000,
                page_faults: 100,
                ipi_sent: 500,
                cpu_cycles: 1_000_000_000_000,
            }
        }

        fn trace_enabled(&self) -> bool {
            self.trace_on
        }

        fn set_trace(&mut self, enabled: bool) {
            self.trace_on = enabled;
        }

        fn reboot(&mut self) {
            self.rebooted = true;
        }
    }

    #[test]
    fn test_shell_creation() {
        let shell = Shell::default_shell();
        assert_eq!(shell.prompt(), "ruvix> ");
    }

    #[test]
    fn test_help_command() {
        let mut shell = Shell::default_shell();
        let mut backend = MockBackend::new();
        let output = shell.execute_line("help", &mut backend);
        assert!(output.contains("help"));
        assert!(output.contains("info"));
        assert!(output.contains("mem"));
    }

    #[test]
    fn test_info_command() {
        let mut shell = Shell::default_shell();
        let mut backend = MockBackend::new();
        let output = shell.execute_line("info", &mut backend);
        assert!(output.contains("0.1.0-test"));
        assert!(output.contains("CPU"));
    }

    #[test]
    fn test_trace_toggle() {
        let mut shell = Shell::default_shell();
        let mut backend = MockBackend::new();

        // Check initial state
        let output = shell.execute_line("trace", &mut backend);
        assert!(output.contains("DISABLED"));

        // Enable tracing
        let output = shell.execute_line("trace on", &mut backend);
        assert!(output.contains("enabled"));
        assert!(backend.trace_enabled());

        // Disable tracing
        let output = shell.execute_line("trace off", &mut backend);
        assert!(output.contains("disabled"));
        assert!(!backend.trace_enabled());
    }

    #[test]
    fn test_unknown_command() {
        let mut shell = Shell::default_shell();
        let mut backend = MockBackend::new();
        let output = shell.execute_line("unknown_cmd", &mut backend);
        assert!(output.contains("Error"));
    }

    #[test]
    fn test_history() {
        let mut shell = Shell::default_shell();
        let mut backend = MockBackend::new();

        shell.execute_line("help", &mut backend);
        shell.execute_line("info", &mut backend);

        let history = shell.history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0], "help");
        assert_eq!(history[1], "info");
    }

    #[test]
    fn test_empty_line() {
        let mut shell = Shell::default_shell();
        let mut backend = MockBackend::new();
        let output = shell.execute_line("", &mut backend);
        assert!(output.is_empty());

        let output = shell.execute_line("   ", &mut backend);
        assert!(output.is_empty());
    }
}
