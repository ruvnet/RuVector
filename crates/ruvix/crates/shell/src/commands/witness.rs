//! Witness log viewer command implementation.

use crate::ShellBackend;
use alloc::format;
use alloc::string::String;

/// Convert operation type to display string.
fn operation_str(op: u8) -> &'static str {
    match op {
        0 => "VGET",      // vector_get
        1 => "VPUT",      // vector_put_proved
        2 => "GAPPLY",    // graph_apply_proved
        3 => "QSEND",     // queue_send
        4 => "QRECV",     // queue_recv
        5 => "RGRANT",    // region_grant
        6 => "CGRANT",    // cap_grant
        7 => "CREVOKE",   // cap_revoke
        8 => "TSPAWN",    // task_spawn
        9 => "TEXIT",     // task_exit
        _ => "UNKNOWN",
    }
}

/// Format hash prefix as hex string.
fn format_hash(hash: &[u8; 8]) -> String {
    let mut s = String::with_capacity(16);
    for byte in hash {
        s.push_str(&format!("{:02x}", byte));
    }
    s
}

/// Format timestamp as relative time.
fn format_timestamp(ns: u64) -> String {
    let secs = ns / 1_000_000_000;
    let millis = (ns % 1_000_000_000) / 1_000_000;
    format!("{}.{:03}s", secs, millis)
}

/// Execute the witness command.
#[must_use]
pub fn execute<B: ShellBackend>(backend: &B, count: usize) -> String {
    let entries = backend.witness_entries(count);

    if entries.is_empty() {
        return String::from("No witness log entries.");
    }

    let mut output = String::from("Witness Log\n");
    output.push_str("===========\n");
    output.push_str(&format!("Showing {} most recent entries\n\n", entries.len()));
    output.push_str("  SEQ       TIMESTAMP   OP       OBJECT_ID         HASH_PREFIX\n");
    output.push_str("  --------  ----------  -------  ----------------  ----------------\n");

    for entry in &entries {
        let line = format!(
            "  {:>8}  {:>10}  {:<7}  0x{:016X}  {}\n",
            entry.seq,
            format_timestamp(entry.timestamp_ns),
            operation_str(entry.operation),
            entry.object_id,
            format_hash(&entry.hash_prefix)
        );
        output.push_str(&line);
    }

    output.push_str("\nOperations: VGET=vector_get, VPUT=vector_put_proved, GAPPLY=graph_apply");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_str() {
        assert_eq!(operation_str(0), "VGET");
        assert_eq!(operation_str(1), "VPUT");
        assert_eq!(operation_str(2), "GAPPLY");
        assert_eq!(operation_str(255), "UNKNOWN");
    }

    #[test]
    fn test_format_hash() {
        let hash = [0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A];
        assert_eq!(format_hash(&hash), "abcdef123456789a");
    }

    #[test]
    fn test_format_timestamp() {
        assert_eq!(format_timestamp(1_500_000_000), "1.500s");
        assert_eq!(format_timestamp(0), "0.000s");
        assert_eq!(format_timestamp(999_999_999), "0.999s");
    }
}
