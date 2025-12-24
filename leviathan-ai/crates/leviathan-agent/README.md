# Leviathan Agent - Self-Replicating AI Agent System

A powerful framework for creating, managing, and replicating AI agents with specific roles, capabilities, and tools. Agents can spawn new agents, evolve their specifications, and maintain complete lineage tracking.

## Features

- 🤖 **Self-Replicating Agents**: Spawn new agents from existing specifications
- 🧬 **Mutation System**: Evolve agents with controlled mutations
- 📊 **Lineage Tracking**: Full ancestry trees with hash-based versioning
- 🛠️ **Tool Integration**: Execute external tools with output parsing
- 📝 **Audit Logging**: Complete history of all agent actions
- 🎯 **Role-Based**: Pre-defined roles (Data Engineer, ML Engineer, etc.)
- 🔧 **Builder Pattern**: Fluent API for agent construction
- ✅ **Validation**: Automatic capability-tool validation

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
leviathan-agent = "0.1.0"
tokio = { version = "1.35", features = ["full"] }
```

### Basic Usage

```rust
use leviathan_agent::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a Junior AI Engineer agent from template
    let spec = junior_ai_engineer_spec();

    // Execute a task
    let mut executor = AgentExecutor::new(spec);
    let result = executor.execute_task("Implement a RAG system").await?;

    println!("Success: {}", result.success);
    Ok(())
}
```

### Building Custom Agents

```rust
use leviathan_agent::prelude::*;

let agent = AgentBuilder::new("Data Pipeline Engineer")
    .role(AgentRole::DataEngineer)
    .capability(capability(
        "etl_development",
        "Build ETL pipelines",
        vec!["python".into()],
    ))
    .tool(tool("python", "python3", "{{script}}"))
    .instruction("Design and implement data pipelines")
    .knowledge(KnowledgeItem::Framework("Apache Spark".into()))
    .build()?;
```

### Agent Replication

```rust
use leviathan_agent::prelude::*;

let mut replicator = AgentReplicator::new();

// Exact replica
let child = replicator.replicate(&parent);

// Mutated replica
let mutations = vec![
    MutationOperator::AddKnowledge(KnowledgeItem::Framework("LangChain".into())),
    MutationOperator::AppendInstructions("Focus on RAG systems".into()),
];
let mutated = replicator.replicate_with_mutation(&parent, mutations)?;

// Spawn a swarm
let swarm = replicator.spawn_swarm(&template, 10, true);
```

## Core Concepts

### Agent Specification

An `AgentSpec` is the complete blueprint of an agent:

- **ID**: Unique identifier (UUID)
- **Name**: Human-readable name
- **Role**: The agent's primary function
- **Capabilities**: What the agent can do
- **Tools**: External commands the agent can execute
- **Instructions**: Detailed guidance for the agent
- **Knowledge Base**: Frameworks, concepts, and references
- **Parent Hash**: For lineage tracking (if spawned)

### Roles

Pre-defined roles with specific purposes:

- `DataEngineer`: Data pipelines and infrastructure
- `MLEngineer`: Machine learning development
- `Researcher`: Research and analysis
- `Tester`: Quality assurance
- `Reviewer`: Code and process review
- `Orchestrator`: High-level coordination
- `Custom(String)`: User-defined roles

### Capabilities

Capabilities define what an agent can do:

```rust
Capability {
    name: "python_development",
    description: "Expert Python development",
    required_tools: vec!["python", "pytest", "git"],
}
```

### Tools

Tools are external commands agents can execute:

```rust
ToolSpec {
    name: "python",
    command: "python3",
    args_template: "-c '{{code}}'",
    output_parser: OutputParser::Raw,
    working_dir: Some("/workspace"),
    env_vars: HashMap::new(),
}
```

### Mutations

Evolve agents with mutation operators:

- `AddCapability(Capability)`: Add new capability
- `RemoveCapability(String)`: Remove capability by name
- `AddTool(ToolSpec)`: Add new tool
- `RemoveTool(String)`: Remove tool
- `ModifyInstructions(String)`: Replace instructions
- `AppendInstructions(String)`: Append to instructions
- `AddKnowledge(KnowledgeItem)`: Add knowledge
- `ChangeRole(AgentRole)`: Change agent role

## Built-in Templates

### Junior AI Engineer

A comprehensive agent for AI development:

```rust
let agent = junior_ai_engineer_spec();
```

**Capabilities:**
- Python development
- Azure deployment
- LLM integration (LangChain, AutoGen, LangGraph)
- RAG implementation
- MLOps practices
- Agile/Scrum workflows

**Tools:**
- `python`, `pytest`, `git`, `docker`, `az`, `pip`

**Knowledge:**
- Frameworks: LangChain, AutoGen, LangGraph, PyTorch, TensorFlow
- Concepts: RAG, Vector Embeddings, Prompt Engineering, MLOps
- Best Practices: TDD, CI/CD, Code Review, Monitoring

## Examples

Run the examples to see the system in action:

```bash
# Basic usage
cargo run --example basic_usage

# Agent replication
cargo run --example replication

# Custom agent builder
cargo run --example custom_agent
```

## Architecture

```
leviathan-agent/
├── src/
│   ├── lib.rs           # Main module
│   ├── spec.rs          # Agent specifications
│   ├── builder.rs       # Builder pattern
│   ├── replication.rs   # Replication system
│   ├── executor.rs      # Task execution
│   └── templates/       # Built-in templates
│       ├── mod.rs
│       └── junior_ai_engineer.rs
├── examples/            # Usage examples
└── tests/              # Integration tests
```

## Advanced Features

### Lineage Tracking

Track complete agent ancestry:

```rust
let lineage = replicator.lineage();

// Get children
let children = lineage.children(&parent_hash);

// Get parent
let parent = lineage.parent(&child_hash);

// Get full ancestry path
let path = lineage.ancestry(&agent_hash);

// Calculate generation depth
let generation = lineage.generation(&agent_hash);
```

### Audit Logging

Every action is logged:

```rust
let executor = AgentExecutor::new(spec);
executor.execute_task("Task description").await?;

// Get audit log
let log = executor.audit_log();

// Export to JSON
let json = executor.export_audit_log()?;
```

### State Management

Maintain state across tool invocations:

```rust
let mut state = ExecutionState::new();

// Add outputs
state.add_output("tool1".into(), "result".into());

// Set variables
state.set_variable("key".into(), "value".into());

// Retrieve
let output = state.get_output("tool1");
let var = state.get_variable("key");
```

## Testing

```bash
# Run all tests
cargo test

# Run with logging
RUST_LOG=debug cargo test

# Run specific test
cargo test test_agent_spec_hash
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Roadmap

- [ ] LLM integration for intelligent task parsing
- [ ] Parallel agent execution
- [ ] Agent communication protocols
- [ ] Cloud storage for specifications
- [ ] Web UI for agent management
- [ ] Multi-agent collaboration patterns
- [ ] Advanced mutation strategies
- [ ] Performance optimization

## Support

- Issues: GitHub Issues
- Documentation: docs.rs
- Examples: `examples/` directory

---

Built with ❤️ by the Leviathan AI Team
