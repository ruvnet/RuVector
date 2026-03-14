//! Capability table command implementation.

use crate::ShellBackend;
use alloc::format;
use alloc::string::String;

/// Convert object type to display string.
fn object_type_str(obj_type: u8) -> &'static str {
    match obj_type {
        0 => "Task",
        1 => "Region",
        2 => "Queue",
        3 => "Timer",
        4 => "VecStore",
        5 => "GraphStore",
        6 => "RvfMount",
        7 => "Sensor",
        _ => "Unknown",
    }
}

/// Format rights bitmap as string.
fn format_rights(rights: u32) -> String {
    let mut s = String::with_capacity(8);

    // Standard rights
    if rights & 0x01 != 0 { s.push('R'); } else { s.push('-'); }
    if rights & 0x02 != 0 { s.push('W'); } else { s.push('-'); }
    if rights & 0x04 != 0 { s.push('X'); } else { s.push('-'); }
    if rights & 0x08 != 0 { s.push('G'); } else { s.push('-'); }
    if rights & 0x10 != 0 { s.push('P'); } else { s.push('-'); }
    if rights & 0x20 != 0 { s.push('D'); } else { s.push('-'); }

    s
}

/// Execute the caps command.
#[must_use]
pub fn execute<B: ShellBackend>(backend: &B, task_id: Option<u32>) -> String {
    let entries = backend.capability_entries(task_id);

    if entries.is_empty() {
        return match task_id {
            Some(id) => format!("No capabilities for task {}.", id),
            None => String::from("No capabilities."),
        };
    }

    let header = match task_id {
        Some(id) => format!("Capability Table (Task {})\n", id),
        None => String::from("Capability Table (All Tasks)\n"),
    };

    let mut output = header;
    output.push_str("===========================\n");
    output.push_str("  HANDLE  OBJECT_ID         TYPE       RIGHTS  BADGE    DEPTH\n");
    output.push_str("  ------  ----------------  ---------  ------  -------  -----\n");

    for entry in &entries {
        let line = format!(
            "  {:>6}  0x{:016X}  {:<9}  {}  {:>7}  {:>5}\n",
            entry.handle,
            entry.object_id,
            object_type_str(entry.object_type),
            format_rights(entry.rights),
            entry.badge,
            entry.depth
        );
        output.push_str(&line);
    }

    output.push_str(&format!("\nTotal: {} capability(ies)", entries.len()));

    // Add rights legend
    output.push_str("\n\nRights: R=Read, W=Write, X=Execute, G=Grant, P=Prove, D=Delete");

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_type_str() {
        assert_eq!(object_type_str(0), "Task");
        assert_eq!(object_type_str(1), "Region");
        assert_eq!(object_type_str(4), "VecStore");
        assert_eq!(object_type_str(255), "Unknown");
    }

    #[test]
    fn test_format_rights() {
        assert_eq!(format_rights(0x00), "------");
        assert_eq!(format_rights(0x01), "R-----");
        assert_eq!(format_rights(0x07), "RWX---");
        assert_eq!(format_rights(0x3F), "RWXGPD");
        assert_eq!(format_rights(0x15), "R-X-P-"); // 0x15 = 0b00010101 = R, X, P bits
        assert_eq!(format_rights(0x29), "R--G-D"); // 0x29 = 0b00101001 = R, G, D bits
    }
}
