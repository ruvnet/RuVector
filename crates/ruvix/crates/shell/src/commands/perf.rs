//! Performance counters command implementation.

use crate::ShellBackend;
use alloc::format;
use alloc::string::String;

/// Format large numbers with K/M/B suffixes.
fn format_count(count: u64) -> String {
    const K: u64 = 1_000;
    const M: u64 = 1_000_000;
    const B: u64 = 1_000_000_000;

    if count >= B {
        format!("{:.2}B", count as f64 / B as f64)
    } else if count >= M {
        format!("{:.2}M", count as f64 / M as f64)
    } else if count >= K {
        format!("{:.2}K", count as f64 / K as f64)
    } else {
        format!("{}", count)
    }
}

/// Execute the perf command.
#[must_use]
pub fn execute<B: ShellBackend>(backend: &B) -> String {
    let counters = backend.perf_counters();

    format!(
        r"Performance Counters
====================
Syscalls:          {}
Context Switches:  {}
Interrupts:        {}
Page Faults:       {}
IPI Messages:      {}
CPU Cycles:        {}",
        format_count(counters.syscalls),
        format_count(counters.context_switches),
        format_count(counters.interrupts),
        format_count(counters.page_faults),
        format_count(counters.ipi_sent),
        format_count(counters.cpu_cycles)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1.00K");
        assert_eq!(format_count(1500), "1.50K");
        assert_eq!(format_count(1_000_000), "1.00M");
        assert_eq!(format_count(1_500_000), "1.50M");
        assert_eq!(format_count(1_000_000_000), "1.00B");
        assert_eq!(format_count(2_500_000_000), "2.50B");
    }
}
