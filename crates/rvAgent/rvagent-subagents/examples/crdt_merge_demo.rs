//! Demonstration of CRDT-based state merging for parallel subagents.
//!
//! This example shows how to use the CRDT merge functionality to
//! deterministically merge state updates from multiple parallel subagents.
//!
//! Run with:
//! ```bash
//! cargo run --example crdt_merge_demo
//! ```

use rvagent_subagents::crdt_merge::{merge_subagent_results, CrdtState};

fn main() {
    println!("CRDT State Merging Demo for Parallel Subagents");
    println!("===============================================\n");

    // Scenario: A parent agent spawns 3 subagents in parallel to analyze
    // different aspects of a codebase. Each subagent produces findings
    // that need to be merged back into the parent's state.

    // Parent agent starts with initial state
    let mut parent = CrdtState::new(0);
    parent.set("analysis_status", b"in_progress".to_vec());
    parent.set("total_files", b"100".to_vec());
    println!("Parent initial state:");
    println!("  - analysis_status: in_progress");
    println!("  - total_files: 100\n");

    // Subagent 1: Security scanner
    let mut security_scanner = CrdtState::new(1);
    security_scanner.set("security_issues", b"3".to_vec());
    security_scanner.set("critical_issues", b"1".to_vec());
    security_scanner.set("analysis_status", b"security_complete".to_vec());
    println!("Security scanner (agent 1) findings:");
    println!("  - security_issues: 3");
    println!("  - critical_issues: 1");
    println!("  - analysis_status: security_complete\n");

    // Subagent 2: Performance analyzer
    let mut performance_analyzer = CrdtState::new(2);
    performance_analyzer.set("slow_functions", b"12".to_vec());
    performance_analyzer.set("memory_leaks", b"2".to_vec());
    performance_analyzer.set("analysis_status", b"performance_complete".to_vec());
    println!("Performance analyzer (agent 2) findings:");
    println!("  - slow_functions: 12");
    println!("  - memory_leaks: 2");
    println!("  - analysis_status: performance_complete\n");

    // Subagent 3: Code quality checker
    let mut quality_checker = CrdtState::new(3);
    quality_checker.set("code_smells", b"45".to_vec());
    quality_checker.set("duplicates", b"8".to_vec());
    quality_checker.set("analysis_status", b"quality_complete".to_vec());
    println!("Quality checker (agent 3) findings:");
    println!("  - code_smells: 45");
    println!("  - duplicates: 8");
    println!("  - analysis_status: quality_complete\n");

    // Merge all subagent results into parent
    println!("Merging all subagent results into parent...\n");

    merge_subagent_results(
        &mut parent,
        vec![security_scanner, performance_analyzer, quality_checker],
    )
    .expect("Merge should succeed");

    println!("Parent state after merge:");
    for key in parent.keys() {
        if let Some(value) = parent.get(key) {
            let value_str = String::from_utf8_lossy(value);
            println!("  - {}: {}", key, value_str);
        }
    }

    println!("\n✓ All findings from 3 parallel subagents merged successfully!");
    println!("✓ Conflicts resolved deterministically (highest node_id wins)");
    println!("✓ analysis_status = quality_complete (agent 3 had highest node_id)\n");

    // Demonstrate conflict resolution
    println!("\nDemonstrating Conflict Resolution");
    println!("==================================\n");

    let mut parent2 = CrdtState::new(0);
    parent2.set("shared_key", b"parent_value".to_vec());

    let mut child1 = CrdtState::new(1);
    child1.set("shared_key", b"child1_value".to_vec());

    let mut child2 = CrdtState::new(2);
    child2.set("shared_key", b"child2_value".to_vec());

    println!("Before merge:");
    println!("  - parent (node 0): shared_key = parent_value");
    println!("  - child1 (node 1): shared_key = child1_value");
    println!("  - child2 (node 2): shared_key = child2_value\n");

    merge_subagent_results(&mut parent2, vec![child1, child2]).expect("Merge should succeed");

    let final_value = String::from_utf8_lossy(parent2.get("shared_key").unwrap());
    println!("After merge:");
    println!(
        "  - shared_key = {} (winner: node 2, highest node_id)",
        final_value
    );
    println!("\n✓ Deterministic conflict resolution ensures consistency!\n");

    // Demonstrate causal ordering
    println!("\nDemonstrating Causal Ordering");
    println!("==============================\n");

    let mut state0 = CrdtState::new(0);
    state0.set("counter", b"0".to_vec());
    println!("State 0: counter = 0");

    // State 1 observes state 0 (by merging), then increments
    let mut state1 = CrdtState::new(1);
    state1.merge(&state0); // Observe state0's clock
    state1.set("counter", b"1".to_vec());
    println!("State 1 (after observing state 0): counter = 1");

    // State 2 observes state 1 (by merging), then increments
    let mut state2 = CrdtState::new(2);
    state2.merge(&state1); // Observe state1's clock
    state2.set("counter", b"2".to_vec());
    println!("State 2 (after observing state 1): counter = 2\n");

    state0.merge(&state1);
    state0.merge(&state2);

    let final_counter = String::from_utf8_lossy(state0.get("counter").unwrap());
    println!(
        "After merge: counter = {} (latest in causal chain)",
        final_counter
    );
    println!("\n✓ Causal ordering preserved through vector clocks!\n");
}
