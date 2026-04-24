//! Task listing command implementation.

use crate::{ShellBackend, TaskState};
use alloc::format;
use alloc::string::String;

/// Convert task state to display string.
fn state_str(state: TaskState) -> &'static str {
    match state {
        TaskState::Ready => "READY",
        TaskState::Running => "RUN",
        TaskState::Blocked => "BLOCK",
        TaskState::Sleeping => "SLEEP",
        TaskState::Exited => "EXIT",
    }
}

/// Extract task name from fixed-size buffer.
fn task_name(name: &[u8; 16]) -> String {
    let end = name.iter().position(|&b| b == 0).unwrap_or(16);
    String::from_utf8_lossy(&name[..end]).into_owned()
}

/// Execute the tasks command.
#[must_use]
pub fn execute<B: ShellBackend>(backend: &B) -> String {
    let tasks = backend.task_list();

    if tasks.is_empty() {
        return String::from("No tasks.");
    }

    let mut output = String::from("Task List\n=========\n");
    output.push_str("  ID   NAME             STATE   PRI  PART  CPU   CAPS\n");
    output.push_str("  ---  ---------------  ------  ---  ----  ----  ----\n");

    for task in &tasks {
        let name = task_name(&task.name);
        let line = format!(
            "  {:<3}  {:<15}  {:<6}  {:>3}  {:>4}  0x{:02X}  {:>4}\n",
            task.id,
            if name.len() > 15 { &name[..15] } else { &name },
            state_str(task.state),
            task.priority,
            task.partition,
            task.cpu_affinity,
            task.cap_count
        );
        output.push_str(&line);
    }

    output.push_str(&format!("\nTotal: {} task(s)", tasks.len()));
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_str() {
        assert_eq!(state_str(TaskState::Ready), "READY");
        assert_eq!(state_str(TaskState::Running), "RUN");
        assert_eq!(state_str(TaskState::Blocked), "BLOCK");
        assert_eq!(state_str(TaskState::Sleeping), "SLEEP");
        assert_eq!(state_str(TaskState::Exited), "EXIT");
    }

    #[test]
    fn test_task_name() {
        let name1 = *b"test\0\0\0\0\0\0\0\0\0\0\0\0";
        assert_eq!(task_name(&name1), "test");

        let name2 = *b"very_long_name!!";
        assert_eq!(task_name(&name2), "very_long_name!!");

        let name3 = [0u8; 16];
        assert_eq!(task_name(&name3), "");
    }
}
