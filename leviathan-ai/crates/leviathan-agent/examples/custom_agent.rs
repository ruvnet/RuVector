//! Custom agent builder example
//!
//! Run with: cargo run --example custom_agent

use leviathan_agent::prelude::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== Leviathan Agent - Custom Agent Example ===\n");

    // 1. Build a custom Data Engineer agent
    println!("1. Building custom Data Engineer agent...");

    let agent = AgentBuilder::new("Senior Data Engineer")
        .role(AgentRole::DataEngineer)

        // Add capabilities
        .capability(leviathan_agent::builder::capability(
            "etl_development",
            "Design and implement ETL pipelines for data processing",
            vec!["python".into(), "sql".into()],
        ))
        .capability(leviathan_agent::builder::capability(
            "data_modeling",
            "Create efficient data models and schemas",
            vec!["sql".into()],
        ))
        .capability(leviathan_agent::builder::capability(
            "pipeline_orchestration",
            "Orchestrate complex data workflows",
            vec!["airflow".into()],
        ))

        // Add tools
        .tool(create_python_tool())
        .tool(create_sql_tool())
        .tool(create_airflow_tool())

        // Add knowledge
        .knowledge(KnowledgeItem::Framework("Apache Spark".into()))
        .knowledge(KnowledgeItem::Framework("dbt".into()))
        .knowledge(KnowledgeItem::Concept("Data Warehousing".into()))
        .knowledge(KnowledgeItem::Concept("Dimensional Modeling".into()))
        .knowledge(KnowledgeItem::BestPractice("Incremental Loading".into()))

        // Set instructions
        .instruction(r#"
You are a Senior Data Engineer responsible for:

1. Designing scalable ETL pipelines
2. Implementing data quality checks
3. Optimizing query performance
4. Maintaining data documentation
5. Ensuring data security and compliance

Follow these principles:
- Write modular, testable code
- Document all transformations
- Implement proper error handling
- Monitor pipeline performance
- Version control all code
        "#)

        .build()?;

    println!("   ✓ Agent created successfully");
    println!("   Name: {}", agent.name);
    println!("   Role: {:?}", agent.role);
    println!("   Capabilities: {}", agent.capabilities.len());
    println!();

    // 2. Validate the agent
    println!("2. Validating agent specification...");
    agent.validate()?;
    println!("   ✓ Agent is valid\n");

    // 3. Display capabilities and their tools
    println!("3. Capability-Tool Mapping:");
    for cap in &agent.capabilities {
        println!("   Capability: {}", cap.name);
        println!("   Description: {}", cap.description);
        println!("   Required tools:");
        for tool_name in &cap.required_tools {
            if let Some(tool) = agent.get_tool(tool_name) {
                println!("     - {} -> {}", tool_name, tool.command);
            }
        }
        println!();
    }

    // 4. Create specialized variants
    println!("4. Creating specialized variants...");

    let mut replicator = AgentReplicator::new();

    // Create a streaming-focused variant
    let streaming_variant = replicator.replicate_with_mutation(
        &agent,
        vec![
            MutationOperator::AddKnowledge(KnowledgeItem::Framework("Apache Kafka".into())),
            MutationOperator::AddKnowledge(KnowledgeItem::Framework("Apache Flink".into())),
            MutationOperator::AppendInstructions(
                "\nSpecialization: Real-time streaming data processing".into()
            ),
        ],
    )?;

    println!("   ✓ Created streaming variant");
    println!("     Knowledge items: {}", streaming_variant.knowledge_base.len());

    // Create a batch-focused variant
    let batch_variant = replicator.replicate_with_mutation(
        &agent,
        vec![
            MutationOperator::AddKnowledge(KnowledgeItem::Framework("Apache Airflow".into())),
            MutationOperator::AddKnowledge(KnowledgeItem::Concept("Batch Processing".into())),
            MutationOperator::AppendInstructions(
                "\nSpecialization: Large-scale batch data processing".into()
            ),
        ],
    )?;

    println!("   ✓ Created batch variant");
    println!("     Knowledge items: {}\n", batch_variant.knowledge_base.len());

    // 5. Compare variants
    println!("5. Variant Comparison:");
    println!("   Original: {} knowledge items", agent.knowledge_base.len());
    println!("   Streaming: {} knowledge items", streaming_variant.knowledge_base.len());
    println!("   Batch: {} knowledge items\n", batch_variant.knowledge_base.len());

    println!("=== Custom agent example completed successfully! ===");

    Ok(())
}

// Helper functions to create tools

fn create_python_tool() -> leviathan_agent::spec::ToolSpec {
    leviathan_agent::spec::ToolSpec {
        name: "python".into(),
        command: "python3".into(),
        args_template: "-c '{{code}}'".into(),
        output_parser: leviathan_agent::spec::OutputParser::Raw,
        working_dir: Some("{{workspace}}".into()),
        env_vars: HashMap::new(),
    }
}

fn create_sql_tool() -> leviathan_agent::spec::ToolSpec {
    leviathan_agent::spec::ToolSpec {
        name: "sql".into(),
        command: "psql".into(),
        args_template: "-d {{database}} -c '{{query}}'".into(),
        output_parser: leviathan_agent::spec::OutputParser::Lines,
        working_dir: None,
        env_vars: HashMap::new(),
    }
}

fn create_airflow_tool() -> leviathan_agent::spec::ToolSpec {
    leviathan_agent::spec::ToolSpec {
        name: "airflow".into(),
        command: "airflow".into(),
        args_template: "{{command}}".into(),
        output_parser: leviathan_agent::spec::OutputParser::Raw,
        working_dir: None,
        env_vars: HashMap::new(),
    }
}
