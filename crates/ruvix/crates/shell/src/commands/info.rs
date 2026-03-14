//! Kernel information command implementation.

use crate::ShellBackend;
use alloc::format;
use alloc::string::String;

/// Format uptime in a human-readable way.
fn format_uptime(nanos: u64) -> String {
    let total_secs = nanos / 1_000_000_000;
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let minutes = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    if days > 0 {
        format!("{}d {}h {}m {}s", days, hours, minutes, secs)
    } else if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

/// Execute the info command.
#[must_use]
pub fn execute<B: ShellBackend>(backend: &B) -> String {
    let info = backend.kernel_info();
    let uptime_ns = info.current_time_ns.saturating_sub(info.boot_time_ns);

    format!(
        r"RuVix Cognition Kernel
======================
Version:    {}
Build:      {}
CPUs:       {} online
Uptime:     {}",
        info.version,
        info.build_time,
        info.cpu_count,
        format_uptime(uptime_ns)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_uptime_seconds() {
        assert_eq!(format_uptime(5_000_000_000), "5s");
        assert_eq!(format_uptime(59_000_000_000), "59s");
    }

    #[test]
    fn test_format_uptime_minutes() {
        assert_eq!(format_uptime(60_000_000_000), "1m 0s");
        assert_eq!(format_uptime(125_000_000_000), "2m 5s");
    }

    #[test]
    fn test_format_uptime_hours() {
        assert_eq!(format_uptime(3600_000_000_000), "1h 0m 0s");
        assert_eq!(format_uptime(3725_000_000_000), "1h 2m 5s");
    }

    #[test]
    fn test_format_uptime_days() {
        assert_eq!(format_uptime(86400_000_000_000), "1d 0h 0m 0s");
        assert_eq!(format_uptime(90125_000_000_000), "1d 1h 2m 5s");
    }
}
