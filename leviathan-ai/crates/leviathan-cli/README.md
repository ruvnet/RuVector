# Leviathan CLI

Command-line interface for Leviathan AI with interactive TUI and declarative action sequences.

## Features

- **Easy Clicking Action Sequences**: Define complex workflows in YAML/JSON
- **Interactive TUI**: Real-time monitoring with ratatui
- **Full Audit Trail**: All actions logged with cryptographic verification
- **Comprehensive Commands**: Train models, spawn agents, manage swarms, and more
- **Configuration Management**: TOML-based config with env var overrides

## Installation

```bash
cargo install --path .
```

## Quick Start

### Initialize a Project

```bash
leviathan init my-project --template default
```

### Train a φ-Lattice Model

```bash
leviathan train corpus.txt --output model.bin --epochs 100
```

### Generate Completions

```bash
leviathan generate "Once upon a time" --max-tokens 100 --temperature 0.7
```

### Execute Swarm Tasks

```bash
leviathan swarm "Analyze data pipeline" --topology mesh --agents 5
```

### Launch Interactive TUI

```bash
leviathan ui
```

The TUI provides:
- **Dashboard**: Swarm status, metrics, and recent actions
- **Agents**: List of active agents with status indicators
- **DAG**: ASCII visualization of task dependencies
- **Actions**: Complete action history with replay capability
- **Help**: Keyboard shortcuts and command reference

#### TUI Keyboard Shortcuts

- `q`, `Ctrl+C` - Quit
- `Tab` - Next tab
- `Shift+Tab` - Previous tab
- `↑/↓` - Navigate items
- `h`, `?` - Toggle help
- `r` - Refresh data

### Agent Management

```bash
# Spawn an agent
leviathan agent spawn "general-purpose" --name worker-1

# List all agents
leviathan agent list

# Show agent details
leviathan agent show agent-001

# Stop an agent
leviathan agent stop agent-001
```

### Audit Trail

```bash
# Verify audit chain integrity
leviathan audit verify --from 2025-01-01T00:00:00Z

# Export audit log
leviathan audit export audit.json --format json
```

### DAG Visualization

```bash
# Show DAG in terminal
leviathan dag show

# Export DAG to file
leviathan dag export dag.dot --format dot
```

## Action Sequences

Define complex workflows in YAML or JSON and execute them with dependency management and audit logging.

### Example Sequence (YAML)

```yaml
name: "Training Pipeline"
description: "Complete training and evaluation workflow"

variables:
  model_path: "models/phi-lattice.bin"

actions:
  - type: log
    message: "Starting pipeline"
    level: "info"
    depends_on: []
    continue_on_error: false

  - type: init
    path: "output"
    template: "default"
    depends_on: [0]

  - type: train
    corpus: "data/corpus.txt"
    output: "output/model.bin"
    epochs: 10
    depends_on: [1]

  - type: generate
    prompt: "Test generation"
    max_tokens: 50
    depends_on: [2]

  - type: verify_audit
    depends_on: [3]

  - type: export_audit
    output: "output/audit.json"
    format: "json"
    depends_on: [4]
```

Execute with:

```bash
leviathan sequence pipeline.yaml
```

### Available Actions

- `init` - Initialize project
- `train` - Train φ-lattice model
- `generate` - Generate text completion
- `swarm_task` - Execute swarm task
- `spawn_agent` - Spawn an agent
- `shell` - Execute shell command
- `verify_audit` - Verify audit chain
- `export_audit` - Export audit log
- `wait` - Wait for duration
- `log` - Log message

Each action supports:
- `depends_on`: List of action indices that must complete first
- `continue_on_error`: Whether to continue if action fails

## Configuration

Create a `leviathan.toml` in your project directory or `~/.config/leviathan/config.toml`:

```toml
# Data directory for models and caches
data_dir = "/var/lib/leviathan"

[audit]
enabled = true
log_path = "/var/log/leviathan/audit.log"
max_size_mb = 100
crypto_verify = true

[swarm]
default_topology = "mesh"
max_agents = 10
task_timeout_secs = 300
auto_heal = true

[agent]
default_type = "general"
memory_limit_mb = 512
persist_state = true

[training]
batch_size = 32
learning_rate = 0.001
max_epochs = 100
use_gpu = false

[ui]
refresh_rate_ms = 250
colors = true
show_help = true
max_history = 100
```

### Environment Variables

Override configuration with environment variables:

- `LEVIATHAN_DATA_DIR` - Data directory
- `LEVIATHAN_AUDIT_ENABLED` - Enable/disable audit logging
- `LEVIATHAN_SWARM_TOPOLOGY` - Default swarm topology

## Command Reference

### Global Options

- `-c, --config <CONFIG>` - Path to configuration file
- `-v, --verbose` - Enable verbose output
- `-h, --help` - Print help
- `-V, --version` - Print version

### Commands

#### `init`

Initialize a new Leviathan project.

```bash
leviathan init [PATH] [--template <TEMPLATE>]
```

#### `train`

Train a φ-lattice model on a corpus.

```bash
leviathan train <CORPUS> [--output <OUTPUT>] [--epochs <EPOCHS>]
```

#### `generate`

Generate text completion from a prompt.

```bash
leviathan generate <PROMPT> [--max-tokens <MAX>] [--temperature <TEMP>]
```

#### `swarm`

Execute a swarm task.

```bash
leviathan swarm <TASK> [--topology <TOPOLOGY>] [--agents <AGENTS>]
```

#### `audit verify`

Verify audit chain integrity.

```bash
leviathan audit verify [--from <DATE>] [--to <DATE>]
```

#### `audit export`

Export audit log to file.

```bash
leviathan audit export <OUTPUT> [--format <FORMAT>]
```

#### `agent spawn`

Spawn a new agent.

```bash
leviathan agent spawn <SPEC> [--name <NAME>]
```

#### `agent list`

List all agents.

```bash
leviathan agent list
```

#### `agent show`

Show agent details.

```bash
leviathan agent show <AGENT_ID>
```

#### `agent stop`

Stop an agent.

```bash
leviathan agent stop <AGENT_ID>
```

#### `dag show`

Show DAG visualization.

```bash
leviathan dag show
```

#### `dag export`

Export DAG to file.

```bash
leviathan dag export <OUTPUT> [--format <FORMAT>]
```

#### `sequence`

Execute action sequence from file.

```bash
leviathan sequence <FILE>
```

#### `ui`

Launch interactive TUI.

```bash
leviathan ui
```

## Architecture

### Modules

- **`src/lib.rs`** - CLI framework types and utilities
- **`src/config.rs`** - Configuration management with TOML support
- **`src/action.rs`** - Action sequence system with audit logging
- **`src/tui.rs`** - Terminal UI with ratatui
- **`src/main.rs`** - Main CLI with clap subcommands

### Dependencies

- `clap` - CLI argument parsing with derive macros
- `crossterm` - Cross-platform terminal manipulation
- `ratatui` - Terminal UI framework
- `serde` - Serialization/deserialization
- `tokio` - Async runtime
- `anyhow` / `thiserror` - Error handling
- `chrono` - Date/time handling
- `uuid` - Unique identifiers
- `colored` - Terminal colors

## Development

### Build

```bash
cargo build --release
```

### Test

```bash
cargo test --all-features
```

### Run

```bash
cargo run -- --help
```

## License

MIT OR Apache-2.0
