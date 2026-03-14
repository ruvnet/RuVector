//! Vector store statistics command implementation.

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

/// Execute the vectors command.
#[must_use]
pub fn execute<B: ShellBackend>(backend: &B) -> String {
    let stats = backend.vector_stats();

    if stats.store_count == 0 {
        return String::from("No vector stores configured.");
    }

    let avg_dims = if stats.store_count > 0 {
        stats.total_dimensions / stats.store_count
    } else {
        0
    };

    let bytes_per_vector = if stats.vector_count > 0 {
        stats.memory_bytes / stats.vector_count
    } else {
        0
    };

    let read_write_ratio = if stats.writes > 0 {
        stats.reads as f64 / stats.writes as f64
    } else {
        0.0
    };

    format!(
        r"Vector Store Statistics
=======================
Stores:           {}
Total Vectors:    {}
Avg Dimensions:   {}

Memory Usage:
  Total:          {}
  Per Vector:     {} bytes

Operations:
  Reads:          {}
  Writes:         {}
  R/W Ratio:      {:.2}:1",
        stats.store_count,
        stats.vector_count,
        avg_dims,
        format_bytes(stats.memory_bytes),
        bytes_per_vector,
        stats.reads,
        stats.writes,
        read_write_ratio
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MiB");
    }
}
