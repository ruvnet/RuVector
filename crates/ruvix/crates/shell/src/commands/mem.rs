//! Memory statistics command implementation.

use crate::ShellBackend;
use alloc::format;
use alloc::string::String;

/// Format bytes in a human-readable way.
fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;

    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Execute the mem command.
#[must_use]
pub fn execute<B: ShellBackend>(backend: &B) -> String {
    let stats = backend.memory_stats();
    let used_percent = if stats.total_bytes > 0 {
        (stats.used_bytes as f64 / stats.total_bytes as f64 * 100.0) as u32
    } else {
        0
    };

    format!(
        r"Memory Statistics
=================
Total:     {}
Used:      {} ({}%)
Free:      {}
Peak:      {}

Allocations:
  Regions: {}
  Slabs:   {}",
        format_bytes(stats.total_bytes),
        format_bytes(stats.used_bytes),
        used_percent,
        format_bytes(stats.free_bytes),
        format_bytes(stats.peak_bytes),
        stats.region_count,
        stats.slab_count
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1536), "1.50 KiB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MiB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GiB");
        assert_eq!(
            format_bytes(1024 * 1024 * 1024 + 512 * 1024 * 1024),
            "1.50 GiB"
        );
    }
}
