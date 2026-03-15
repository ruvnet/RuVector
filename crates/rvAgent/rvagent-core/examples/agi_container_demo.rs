//! Demonstration of AGI Container building and parsing.
//!
//! Run with: cargo run --example agi_container_demo

use rvagent_core::agi_container::{
    AgentNode, AgentPrompt, AgiContainerBuilder, OrchestratorConfig, SkillDefinition,
    ToolDefinition,
};
use serde_json::json;

fn main() {
    println!("=== AGI Container Demo ===\n");

    // Create tool definitions
    let tools = vec![
        ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web for information".to_string(),
            parameters: json!({
                "query": "string",
                "max_results": "number"
            }),
            returns: Some("SearchResults".to_string()),
        },
        ToolDefinition {
            name: "execute_code".to_string(),
            description: "Execute code in a safe sandbox".to_string(),
            parameters: json!({
                "code": "string",
                "language": "string"
            }),
            returns: Some("ExecutionResult".to_string()),
        },
    ];

    // Create agent prompts
    let prompts = vec![
        AgentPrompt {
            name: "researcher".to_string(),
            system_prompt: "You are a research assistant specialized in gathering and analyzing information.".to_string(),
            version: "1.0.0".to_string(),
        },
        AgentPrompt {
            name: "coder".to_string(),
            system_prompt: "You are an expert programmer focused on clean, efficient code.".to_string(),
            version: "2.0.0".to_string(),
        },
    ];

    // Create skill definitions
    let skills = vec![
        SkillDefinition {
            name: "code-review".to_string(),
            description: "Review code for quality and best practices".to_string(),
            trigger: "/review".to_string(),
            content: "Analyze code for readability, performance, and security.".to_string(),
        },
        SkillDefinition {
            name: "refactor".to_string(),
            description: "Refactor code for better structure".to_string(),
            trigger: "/refactor".to_string(),
            content: "Improve code organization and eliminate duplication.".to_string(),
        },
    ];

    // Create orchestrator configuration
    let orchestrator = OrchestratorConfig {
        topology: "hierarchical".to_string(),
        agents: vec![
            AgentNode {
                id: "researcher-1".to_string(),
                agent_type: "researcher".to_string(),
                prompt_ref: "researcher".to_string(),
            },
            AgentNode {
                id: "coder-1".to_string(),
                agent_type: "coder".to_string(),
                prompt_ref: "coder".to_string(),
            },
        ],
        connections: vec![("researcher-1".to_string(), "coder-1".to_string())],
    };

    // Build the container
    println!("Building AGI container...");
    let container = AgiContainerBuilder::new()
        .with_tools(&tools)
        .with_prompts(&prompts)
        .with_skills(&skills)
        .with_orchestrator(&orchestrator)
        .build();

    println!("Container size: {} bytes", container.len());
    println!("Magic bytes: {:?}", &container[0..4]);
    println!(
        "Checksum: {}",
        hex::encode(&container[container.len() - 32..])
    );

    // Parse the container back
    println!("\nParsing container...");
    let parsed = AgiContainerBuilder::parse(&container).expect("Failed to parse container");

    println!("\nParsed contents:");
    println!("  Tools: {}", parsed.tools.len());
    for tool in &parsed.tools {
        println!("    - {}: {}", tool.name, tool.description);
    }

    println!("  Prompts: {}", parsed.prompts.len());
    for prompt in &parsed.prompts {
        println!("    - {} (v{})", prompt.name, prompt.version);
    }

    println!("  Skills: {}", parsed.skills.len());
    for skill in &parsed.skills {
        println!("    - {}: {}", skill.name, skill.trigger);
    }

    if let Some(orch) = &parsed.orchestrator {
        println!("  Orchestrator:");
        println!("    Topology: {}", orch.topology);
        println!("    Agents: {}", orch.agents.len());
        for agent in &orch.agents {
            println!("      - {}: {} ({})", agent.id, agent.agent_type, agent.prompt_ref);
        }
        println!("    Connections: {}", orch.connections.len());
    }

    // Verify round-trip integrity
    println!("\nVerifying round-trip integrity...");
    assert_eq!(parsed.tools, tools, "Tools mismatch");
    assert_eq!(parsed.prompts, prompts, "Prompts mismatch");
    assert_eq!(parsed.skills, skills, "Skills mismatch");
    assert_eq!(
        parsed.orchestrator.as_ref(),
        Some(&orchestrator),
        "Orchestrator mismatch"
    );

    println!("✓ All data verified successfully!");
    println!("\n=== Demo complete ===");
}
