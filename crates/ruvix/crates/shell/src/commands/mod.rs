//! Command implementations for the RuVix debug shell.
//!
//! Each command is implemented in its own submodule for maintainability.

pub mod caps;
pub mod cpu;
pub mod info;
pub mod mem;
pub mod perf;
pub mod proofs;
pub mod queues;
pub mod tasks;
pub mod vectors;
pub mod witness;

/// Help command implementation.
pub mod help {
    use alloc::string::String;

    /// Execute the help command.
    #[must_use]
    pub fn execute() -> String {
        String::from(
            r"RuVix Debug Shell Commands:

  help, h, ?         Show this help message
  info, version      Kernel version, boot time, uptime
  mem, memory        Memory statistics
  tasks, ps          Task listing
  caps [task_id]     Capability table dump (optional: filter by task)
  queues, q          Queue statistics
  vectors, vec, v    Vector store info
  proofs, proof, p   Proof statistics
  cpu, smp           CPU info for SMP
  witness [n]        Witness log viewer (default: 10 entries)
  perf, counters     Performance counters
  trace [on|off]     Syscall tracing toggle
  reboot, restart    Trigger system reboot

Type a command name for more information.",
        )
    }
}
