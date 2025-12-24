//! Agent replication and mutation example
//!
//! Run with: cargo run --example replication

use leviathan_agent::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== Leviathan Agent - Replication Example ===\n");

    // 1. Create a parent agent
    println!("1. Creating parent agent...");
    let parent = AgentBuilder::new("Research Agent Alpha")
        .role(AgentRole::Researcher)
        .capability(leviathan_agent::builder::capability(
            "data_analysis",
            "Analyze and interpret data",
            vec![],
        ))
        .instruction("Conduct thorough research and analysis")
        .knowledge(KnowledgeItem::Concept("Statistical Analysis".into()))
        .build()?;

    println!("   Parent ID: {}", parent.id);
    println!("   Parent hash: {}\n", hex::encode(parent.hash()));

    // 2. Create a replicator
    println!("2. Creating replicator...");
    let mut replicator = AgentReplicator::new();
    println!("   ✓ Replicator initialized\n");

    // 3. Replicate exact copy
    println!("3. Creating exact replica...");
    let child1 = replicator.replicate(&parent);
    println!("   Child 1 ID: {}", child1.id);
    println!("   Child 1 hash: {}", hex::encode(child1.hash()));
    println!("   Parent hash: {:?}\n", child1.parent_spec_hash.map(hex::encode));

    // 4. Replicate with mutations
    println!("4. Creating mutated replica...");
    let mutations = vec![
        MutationOperator::AppendInstructions(
            "Focus on quantitative analysis using statistical methods.".into(),
        ),
        MutationOperator::AddKnowledge(KnowledgeItem::Framework("pandas".into())),
        MutationOperator::AddKnowledge(KnowledgeItem::Framework("numpy".into())),
    ];

    let child2 = replicator.replicate_with_mutation(&parent, mutations)?;
    println!("   Child 2 ID: {}", child2.id);
    println!("   Child 2 has {} knowledge items (parent had {})",
        child2.knowledge_base.len(),
        parent.knowledge_base.len()
    );
    println!("   Instructions updated: {}\n",
        child2.instructions != parent.instructions
    );

    // 5. Create a swarm
    println!("5. Spawning a swarm of 5 agents...");
    let template = AgentBuilder::new("Swarm Worker")
        .role(AgentRole::Tester)
        .instruction("Execute test suites")
        .build()?;

    let swarm = replicator.spawn_swarm(&template, 5, true);
    println!("   Swarm size: {}", swarm.len());
    for (i, agent) in swarm.iter().enumerate() {
        println!("   Agent {}: ID={}", i + 1, agent.id);
    }
    println!();

    // 6. Multi-generation replication
    println!("6. Creating multi-generation lineage...");
    let grandchild = replicator.replicate(&child2);
    println!("   Grandchild ID: {}", grandchild.id);

    // 7. Examine lineage
    println!("\n7. Lineage Analysis:");
    let lineage = replicator.lineage();
    let parent_hash = parent.hash();

    if let Some(children) = lineage.children(&parent_hash) {
        println!("   Parent has {} direct children", children.len());
    }

    let child2_generation = lineage.generation(&child2.hash());
    println!("   Child 2 is generation: {}", child2_generation);

    let child2_ancestry = lineage.ancestry(&child2.hash());
    println!("   Child 2 ancestry path length: {}", child2_ancestry.len());

    println!("   Grandchild generation: {}", lineage.generation(&grandchild.hash()));
    println!();

    println!("=== Replication example completed successfully! ===");

    Ok(())
}
