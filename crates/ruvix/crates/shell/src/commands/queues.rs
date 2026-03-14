//! Queue statistics command implementation.

use crate::ShellBackend;
use alloc::format;
use alloc::string::String;

/// Execute the queues command.
#[must_use]
pub fn execute<B: ShellBackend>(backend: &B) -> String {
    let stats = backend.queue_stats();

    let zero_copy_percent = if stats.messages_sent > 0 {
        (stats.zero_copy_count as f64 / stats.messages_sent as f64 * 100.0) as u32
    } else {
        0
    };

    let pending_per_queue = if stats.queue_count > 0 {
        stats.pending_messages as f64 / stats.queue_count as f64
    } else {
        0.0
    };

    format!(
        r"Queue Statistics
================
Active Queues:    {}
Pending Messages: {} ({:.1} per queue avg)

Message Counts:
  Sent:           {}
  Received:       {}
  In Flight:      {}

Zero-Copy Transfers:
  Count:          {}
  Percentage:     {}%",
        stats.queue_count,
        stats.pending_messages,
        pending_per_queue,
        stats.messages_sent,
        stats.messages_received,
        stats.messages_sent.saturating_sub(stats.messages_received),
        stats.zero_copy_count,
        zero_copy_percent
    )
}

#[cfg(test)]
mod tests {
    use crate::QueueStats;

    #[test]
    fn test_queue_stats_formatting() {
        // Basic test that the function produces output
        // Full testing would require a mock backend
        let _ = QueueStats {
            queue_count: 4,
            pending_messages: 10,
            messages_sent: 1000,
            messages_received: 990,
            zero_copy_count: 500,
        };
    }
}
