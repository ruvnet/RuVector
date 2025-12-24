//! Basic usage example for leviathan-swarm

use leviathan_swarm::{
    AgentSpec, ExecutionStrategy, SwarmBuilder, Task, TopologyType,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Create a swarm with mesh topology and DAG execution
    let swarm = SwarmBuilder::new()
        .name("demo-swarm")
        .topology(TopologyType::Mesh)
        .max_agents(8)
        .strategy(ExecutionStrategy::DAG)
        .audit_enabled(true)
        .metrics_enabled(true)
        .build();

    println!("Created swarm: {}", swarm.config().name);
    println!("Topology: {:?}", swarm.config().topology);
    println!("Max agents: {}", swarm.config().max_agents);

    // Register agents with different capabilities
    let rust_agent = AgentSpec::new("rust-developer", vec!["rust".to_string(), "cargo".to_string()])
        .with_tool("cargo")
        .with_tool("rustfmt")
        .with_tool("clippy")
        .with_metadata("version", "1.0.0")
        .with_metadata("specialization", "systems-programming");

    let test_agent = AgentSpec::new("test-engineer", vec!["testing".to_string()])
        .with_tool("cargo-test")
        .with_tool("criterion")
        .with_metadata("coverage", "90%");

    let docs_agent = AgentSpec::new("documentation-writer", vec!["docs".to_string()])
        .with_tool("rustdoc")
        .with_tool("mdbook");

    swarm.register_agent(rust_agent)?;
    swarm.register_agent(test_agent)?;
    swarm.register_agent(docs_agent)?;

    println!("\nRegistered {} agents", swarm.list_agents().len());

    // Create tasks with dependencies
    let task1 = Task::new("Check dependencies", "cargo check");
    let task1_id = task1.id;

    let task2 = Task::new("Build project", "cargo build")
        .with_dependency(task1_id)
        .with_priority(leviathan_swarm::task::Priority::High);
    let task2_id = task2.id;

    let task3 = Task::new("Run tests", "cargo test")
        .with_dependency(task2_id)
        .with_timeout(300);

    let task4 = Task::new("Generate docs", "cargo doc")
        .with_dependency(task2_id);

    let task5 = Task::new("Run benchmarks", "cargo bench")
        .with_dependency(task2_id)
        .with_dependency(task3.id);

    println!("\nExecuting task DAG with {} tasks...", 5);

    // Execute with automatic dependency resolution
    let results = swarm.execute(vec![task1, task2, task3, task4, task5]).await?;

    // Display results
    println!("\n=== Execution Results ===");
    for (i, result) in results.iter().enumerate() {
        println!(
            "Task {}: {} - {} in {}ms",
            i + 1,
            result.description,
            if result.success { "SUCCESS" } else { "FAILED" },
            result.duration_ms
        );
        if !result.output.is_empty() {
            println!("  Output: {}", result.output);
        }
        if let Some(error) = &result.error {
            println!("  Error: {}", error);
        }
    }

    // Get and display metrics
    let metrics = swarm.get_metrics();
    println!("\n=== Performance Metrics ===");
    println!("{}", serde_json::to_string_pretty(&metrics)?);

    Ok(())
}
