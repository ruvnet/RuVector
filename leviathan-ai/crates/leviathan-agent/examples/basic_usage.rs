//! Basic usage example of the leviathan-agent crate
//!
//! Run with: cargo run --example basic_usage

use leviathan_agent::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Leviathan Agent - Basic Usage Example ===\n");

    // 1. Create a Junior AI Engineer agent from template
    println!("1. Creating Junior AI Engineer agent...");
    let agent_spec = junior_ai_engineer_spec();
    println!("   Agent: {}", agent_spec.name);
    println!("   Role: {:?}", agent_spec.role);
    println!("   Capabilities: {}", agent_spec.capabilities.len());
    println!("   Tools: {}", agent_spec.tools.len());
    println!("   Knowledge items: {}\n", agent_spec.knowledge_base.len());

    // 2. Validate the agent specification
    println!("2. Validating agent specification...");
    match agent_spec.validate() {
        Ok(()) => println!("   ✓ Agent specification is valid\n"),
        Err(e) => {
            eprintln!("   ✗ Validation failed: {}", e);
            return Ok(());
        }
    }

    // 3. Display capabilities
    println!("3. Agent Capabilities:");
    for cap in &agent_spec.capabilities {
        println!("   - {}: {}", cap.name, cap.description);
        println!("     Required tools: {}", cap.required_tools.join(", "));
    }
    println!();

    // 4. Display available tools
    println!("4. Available Tools:");
    for tool in &agent_spec.tools {
        println!("   - {}: {}", tool.name, tool.command);
    }
    println!();

    // 5. Display knowledge base
    println!("5. Knowledge Base (first 5 items):");
    for (i, item) in agent_spec.knowledge_base.iter().take(5).enumerate() {
        match item {
            KnowledgeItem::Framework(name) => println!("   {}. Framework: {}", i + 1, name),
            KnowledgeItem::Concept(name) => println!("   {}. Concept: {}", i + 1, name),
            KnowledgeItem::BestPractice(name) => println!("   {}. Best Practice: {}", i + 1, name),
            KnowledgeItem::Reference { title, url } => {
                println!("   {}. Reference: {} ({})", i + 1, title, url)
            }
            KnowledgeItem::Custom { category, content } => {
                println!("   {}. {}: {}", i + 1, category, content)
            }
        }
    }
    println!("   ... and {} more items\n", agent_spec.knowledge_base.len() - 5);

    // 6. Create an executor
    println!("6. Creating executor for agent...");
    let executor = AgentExecutor::new(agent_spec.clone());
    println!("   ✓ Executor created\n");

    // 7. Calculate agent spec hash for lineage tracking
    let agent_hash = agent_spec.hash();
    println!("7. Agent specification hash:");
    println!("   {}\n", hex::encode(agent_hash));

    println!("=== Example completed successfully! ===");

    Ok(())
}
