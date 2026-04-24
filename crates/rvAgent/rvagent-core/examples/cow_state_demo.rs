//! Demo of COW-backed StateBackend for efficient subagent forking.

use rvagent_core::cow_state::CowStateBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== COW StateBackend Demo ===\n");

    // Create parent backend and add some state.
    let parent = CowStateBackend::new();
    parent.set("config", b"production".to_vec())?;
    parent.set("api_key", b"secret123".to_vec())?;

    println!("Parent state:");
    println!(
        "  - config: {:?}",
        String::from_utf8_lossy(&parent.get("config").unwrap())
    );
    println!(
        "  - api_key: {:?}",
        String::from_utf8_lossy(&parent.get("api_key").unwrap())
    );
    println!("  - Branch ID: {}", parent.branch_id());
    println!("  - Version: {}\n", parent.version());

    // Fork for a subagent (O(1) operation via Arc clone).
    let child = parent.fork_for_subagent();

    println!("Child (forked from parent):");
    println!(
        "  - Inherits config: {:?}",
        String::from_utf8_lossy(&child.get("config").unwrap())
    );
    println!(
        "  - Inherits api_key: {:?}",
        String::from_utf8_lossy(&child.get("api_key").unwrap())
    );
    println!("  - Branch ID: {}", child.branch_id());
    println!("  - Local key count: {}\n", child.local_key_count());

    // Child makes modifications (copy-on-write).
    child.set("config", b"staging".to_vec())?;
    child.set("temp_data", b"child_only".to_vec())?;

    println!("After child modifications:");
    println!(
        "  - Child config: {:?}",
        String::from_utf8_lossy(&child.get("config").unwrap())
    );
    println!(
        "  - Child temp_data: {:?}",
        String::from_utf8_lossy(&child.get("temp_data").unwrap())
    );
    println!(
        "  - Parent config: {:?}",
        String::from_utf8_lossy(&parent.get("config").unwrap())
    );
    println!("  - Parent temp_data: {:?}\n", parent.get("temp_data"));

    // Take a snapshot (O(1) via Arc clone).
    let snapshot = child.snapshot();

    println!("Snapshot taken:");
    println!(
        "  - Snapshot config: {:?}",
        String::from_utf8_lossy(&snapshot.get("config").unwrap())
    );
    println!("  - Version: {}\n", snapshot.version());

    // Modify child again (triggers COW from snapshot).
    child.set("config", b"development".to_vec())?;

    println!("After further child modification:");
    println!(
        "  - Child config: {:?}",
        String::from_utf8_lossy(&child.get("config").unwrap())
    );
    println!(
        "  - Snapshot config: {:?}",
        String::from_utf8_lossy(&snapshot.get("config").unwrap())
    );
    println!();

    // Merge child changes back to parent.
    parent.merge_from(&child)?;

    println!("After merge:");
    println!(
        "  - Parent config: {:?}",
        String::from_utf8_lossy(&parent.get("config").unwrap())
    );
    println!(
        "  - Parent temp_data: {:?}",
        String::from_utf8_lossy(&parent.get("temp_data").unwrap())
    );
    println!("  - Parent version: {}", parent.version());
    println!("  - Modified keys: {:?}", parent.modified_keys());
    println!();

    // Demonstrate deletion with COW.
    let child2 = parent.fork_for_subagent();
    child2.delete("api_key");

    println!("Child2 deleted api_key:");
    println!("  - Child2 api_key: {:?}", child2.get("api_key"));
    println!(
        "  - Parent api_key: {:?}",
        String::from_utf8_lossy(&parent.get("api_key").unwrap())
    );
    println!();

    parent.merge_from(&child2)?;

    println!("After merging deletion:");
    println!("  - Parent api_key: {:?}", parent.get("api_key"));
    println!();

    // Demonstrate multiple forks (siblings).
    let sibling_a = parent.fork_for_subagent();
    let sibling_b = parent.fork_for_subagent();

    sibling_a.set("task", b"analyze".to_vec())?;
    sibling_b.set("task", b"synthesize".to_vec())?;

    println!("Sibling forks:");
    println!(
        "  - Sibling A task: {:?}",
        String::from_utf8_lossy(&sibling_a.get("task").unwrap())
    );
    println!(
        "  - Sibling B task: {:?}",
        String::from_utf8_lossy(&sibling_b.get("task").unwrap())
    );
    println!("  - Siblings don't see each other's changes");
    println!();

    println!("=== Demo Complete ===");

    Ok(())
}
