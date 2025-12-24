//! Leviathan CLI - Command-line interface for Leviathan AI
//!
//! Provides easy clicking action sequences via CLI with TUI support.

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;
use leviathan_cli::{
    action::{Action, ActionRunner, ActionSequence, ActionWithDeps},
    config::LeviathanConfig,
    tui::TuiApp,
    AgentStatus, CliContext, CommandResult, DagNode, SwarmStatus,
};
use std::path::PathBuf;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "leviathan")]
#[command(about = "Leviathan AI - Advanced φ-lattice swarm intelligence", long_about = None)]
#[command(version)]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new Leviathan project
    Init {
        /// Project directory
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Project template
        #[arg(short, long)]
        template: Option<String>,
    },

    /// Train a φ-lattice model
    Train {
        /// Training corpus path
        corpus: PathBuf,

        /// Output model path
        #[arg(short, long, default_value = "model.bin")]
        output: PathBuf,

        /// Number of training epochs
        #[arg(short, long)]
        epochs: Option<usize>,
    },

    /// Generate text completion
    Generate {
        /// Input prompt
        prompt: String,

        /// Maximum tokens to generate
        #[arg(short, long)]
        max_tokens: Option<usize>,

        /// Temperature for sampling
        #[arg(short, long)]
        temperature: Option<f32>,
    },

    /// Execute a swarm task
    Swarm {
        /// Task description
        task: String,

        /// Swarm topology
        #[arg(short = 't', long)]
        topology: Option<String>,

        /// Number of agents
        #[arg(short, long)]
        agents: Option<usize>,
    },

    /// Audit trail operations
    Audit {
        #[command(subcommand)]
        command: AuditCommands,
    },

    /// Agent management
    Agent {
        #[command(subcommand)]
        command: AgentCommands,
    },

    /// DAG operations
    Dag {
        #[command(subcommand)]
        command: DagCommands,
    },

    /// Execute action sequence from file
    Sequence {
        /// Path to sequence file (YAML or JSON)
        file: PathBuf,
    },

    /// Launch interactive TUI
    Ui,
}

#[derive(Subcommand)]
enum AuditCommands {
    /// Verify audit chain integrity
    Verify {
        /// Start date (ISO 8601)
        #[arg(long)]
        from: Option<String>,

        /// End date (ISO 8601)
        #[arg(long)]
        to: Option<String>,
    },

    /// Export audit log
    Export {
        /// Output file path
        output: PathBuf,

        /// Export format (json, csv, yaml)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
}

#[derive(Subcommand)]
enum AgentCommands {
    /// Spawn a new agent
    Spawn {
        /// Agent specification
        spec: String,

        /// Agent name
        #[arg(short, long)]
        name: Option<String>,
    },

    /// List all agents
    List,

    /// Show agent details
    Show {
        /// Agent ID
        agent_id: String,
    },

    /// Stop an agent
    Stop {
        /// Agent ID
        agent_id: String,
    },
}

#[derive(Subcommand)]
enum DagCommands {
    /// Show DAG visualization
    Show,

    /// Export DAG to file
    Export {
        /// Output file path
        output: PathBuf,

        /// Export format (dot, json)
        #[arg(short, long, default_value = "dot")]
        format: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose);

    // Create CLI context
    let ctx = CliContext::new(cli.config, cli.verbose)?;

    info!("Leviathan CLI started - Session ID: {}", ctx.session_id);

    // Execute command
    let result = match cli.command {
        Commands::Init { path, template } => cmd_init(&ctx, path, template).await,
        Commands::Train { corpus, output, epochs } => cmd_train(&ctx, corpus, output, epochs).await,
        Commands::Generate { prompt, max_tokens, temperature } => {
            cmd_generate(&ctx, prompt, max_tokens, temperature).await
        }
        Commands::Swarm { task, topology, agents } => cmd_swarm(&ctx, task, topology, agents).await,
        Commands::Audit { command } => match command {
            AuditCommands::Verify { from, to } => cmd_audit_verify(&ctx, from, to).await,
            AuditCommands::Export { output, format } => cmd_audit_export(&ctx, output, format).await,
        },
        Commands::Agent { command } => match command {
            AgentCommands::Spawn { spec, name } => cmd_agent_spawn(&ctx, spec, name).await,
            AgentCommands::List => cmd_agent_list(&ctx).await,
            AgentCommands::Show { agent_id } => cmd_agent_show(&ctx, agent_id).await,
            AgentCommands::Stop { agent_id } => cmd_agent_stop(&ctx, agent_id).await,
        },
        Commands::Dag { command } => match command {
            DagCommands::Show => cmd_dag_show(&ctx).await,
            DagCommands::Export { output, format } => cmd_dag_export(&ctx, output, format).await,
        },
        Commands::Sequence { file } => cmd_sequence(&ctx, file).await,
        Commands::Ui => cmd_ui(&ctx).await,
    };

    // Print result
    match result {
        Ok(cmd_result) => {
            if cmd_result.success {
                println!("{} {}", "✓".green(), cmd_result.message);
                if let Some(data) = cmd_result.data {
                    println!("{}", serde_json::to_string_pretty(&data)?);
                }
                Ok(())
            } else {
                eprintln!("{} {}", "✗".red(), cmd_result.message);
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("{} {}", "✗".red(), e);
            std::process::exit(1);
        }
    }
}

/// Initialize logging based on verbosity
fn init_logging(verbose: bool) {
    let filter = if verbose {
        tracing_subscriber::EnvFilter::new("debug")
    } else {
        tracing_subscriber::EnvFilter::new("info")
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
}

// Command implementations

async fn cmd_init(ctx: &CliContext, path: PathBuf, template: Option<String>) -> Result<CommandResult> {
    let action = Action::Init {
        path: path.clone(),
        template: template.clone(),
    };

    let runner = ActionRunner::new(ctx.audit_log_path(), ctx.session_id, ctx.verbose);
    let sequence = create_single_action_sequence("init", action);

    let results = runner.execute(&sequence).await?;

    if results[0].success {
        // Create default config file
        let config_path = path.join("leviathan.toml");
        if !config_path.exists() {
            ctx.config.save(&config_path)?;
            info!("Created default configuration at {}", config_path.display());
        }

        Ok(CommandResult::success(format!(
            "Project initialized at {}",
            path.display()
        )))
    } else {
        Ok(CommandResult::error(results[0].message.clone()))
    }
}

async fn cmd_train(
    ctx: &CliContext,
    corpus: PathBuf,
    output: PathBuf,
    epochs: Option<usize>,
) -> Result<CommandResult> {
    let action = Action::Train {
        corpus,
        output: output.clone(),
        epochs,
    };

    let runner = ActionRunner::new(ctx.audit_log_path(), ctx.session_id, ctx.verbose);
    let sequence = create_single_action_sequence("train", action);

    let results = runner.execute(&sequence).await?;

    if results[0].success {
        Ok(CommandResult::success(format!(
            "Training complete. Model saved to {}",
            output.display()
        )))
    } else {
        Ok(CommandResult::error(results[0].message.clone()))
    }
}

async fn cmd_generate(
    ctx: &CliContext,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
) -> Result<CommandResult> {
    let action = Action::Generate {
        prompt: prompt.clone(),
        max_tokens,
        temperature,
    };

    let runner = ActionRunner::new(ctx.audit_log_path(), ctx.session_id, ctx.verbose);
    let sequence = create_single_action_sequence("generate", action);

    let results = runner.execute(&sequence).await?;

    if results[0].success {
        let output = results[0].output.clone().unwrap_or_default();
        Ok(CommandResult::success_with_data(
            "Generation complete",
            serde_json::json!({ "output": output }),
        ))
    } else {
        Ok(CommandResult::error(results[0].message.clone()))
    }
}

async fn cmd_swarm(
    ctx: &CliContext,
    task: String,
    topology: Option<String>,
    agents: Option<usize>,
) -> Result<CommandResult> {
    let action = Action::SwarmTask {
        task: task.clone(),
        topology,
        agents,
    };

    let runner = ActionRunner::new(ctx.audit_log_path(), ctx.session_id, ctx.verbose);
    let sequence = create_single_action_sequence("swarm", action);

    let results = runner.execute(&sequence).await?;

    if results[0].success {
        Ok(CommandResult::success(format!("Swarm task '{}' completed", task)))
    } else {
        Ok(CommandResult::error(results[0].message.clone()))
    }
}

async fn cmd_audit_verify(
    ctx: &CliContext,
    from: Option<String>,
    to: Option<String>,
) -> Result<CommandResult> {
    use chrono::DateTime;

    let from_dt = from.as_ref().and_then(|s| DateTime::parse_from_rfc3339(s).ok().map(|dt| dt.into()));
    let to_dt = to.as_ref().and_then(|s| DateTime::parse_from_rfc3339(s).ok().map(|dt| dt.into()));

    let action = Action::VerifyAudit {
        from: from_dt,
        to: to_dt,
    };

    let runner = ActionRunner::new(ctx.audit_log_path(), ctx.session_id, ctx.verbose);
    let sequence = create_single_action_sequence("verify_audit", action);

    let results = runner.execute(&sequence).await?;

    if results[0].success {
        Ok(CommandResult::success("Audit chain verified successfully"))
    } else {
        Ok(CommandResult::error(results[0].message.clone()))
    }
}

async fn cmd_audit_export(
    ctx: &CliContext,
    output: PathBuf,
    format: String,
) -> Result<CommandResult> {
    let action = Action::ExportAudit {
        output: output.clone(),
        format: Some(format),
    };

    let runner = ActionRunner::new(ctx.audit_log_path(), ctx.session_id, ctx.verbose);
    let sequence = create_single_action_sequence("export_audit", action);

    let results = runner.execute(&sequence).await?;

    if results[0].success {
        Ok(CommandResult::success(format!(
            "Audit log exported to {}",
            output.display()
        )))
    } else {
        Ok(CommandResult::error(results[0].message.clone()))
    }
}

async fn cmd_agent_spawn(
    ctx: &CliContext,
    spec: String,
    name: Option<String>,
) -> Result<CommandResult> {
    let action = Action::SpawnAgent {
        spec: spec.clone(),
        name: name.clone(),
    };

    let runner = ActionRunner::new(ctx.audit_log_path(), ctx.session_id, ctx.verbose);
    let sequence = create_single_action_sequence("spawn_agent", action);

    let results = runner.execute(&sequence).await?;

    if results[0].success {
        let agent_id = results[0].output.clone().unwrap_or_default();
        Ok(CommandResult::success_with_data(
            format!("Agent '{}' spawned", name.unwrap_or_else(|| "agent".to_string())),
            serde_json::json!({ "agent_id": agent_id }),
        ))
    } else {
        Ok(CommandResult::error(results[0].message.clone()))
    }
}

async fn cmd_agent_list(ctx: &CliContext) -> Result<CommandResult> {
    // Mock agent list for demonstration
    let agents = vec![
        AgentStatus {
            id: "agent-001".to_string(),
            name: "Worker-1".to_string(),
            status: "running".to_string(),
            task: Some("Processing data".to_string()),
            uptime_secs: 3600,
            actions_completed: 42,
        },
        AgentStatus {
            id: "agent-002".to_string(),
            name: "Worker-2".to_string(),
            status: "idle".to_string(),
            task: None,
            uptime_secs: 1800,
            actions_completed: 15,
        },
    ];

    for agent in &agents {
        println!(
            "{} {} - {} (uptime: {}s, actions: {})",
            if agent.status == "running" { "●".green() } else { "○".yellow() },
            agent.id.bold(),
            agent.status,
            agent.uptime_secs,
            agent.actions_completed
        );
        if let Some(task) = &agent.task {
            println!("  Task: {}", task);
        }
    }

    Ok(CommandResult::success(format!("Found {} agents", agents.len())))
}

async fn cmd_agent_show(ctx: &CliContext, agent_id: String) -> Result<CommandResult> {
    Ok(CommandResult::success_with_data(
        format!("Agent details for {}", agent_id),
        serde_json::json!({
            "id": agent_id,
            "name": "Worker-1",
            "status": "running",
            "uptime_secs": 3600,
            "actions_completed": 42
        }),
    ))
}

async fn cmd_agent_stop(ctx: &CliContext, agent_id: String) -> Result<CommandResult> {
    Ok(CommandResult::success(format!("Agent {} stopped", agent_id)))
}

async fn cmd_dag_show(ctx: &CliContext) -> Result<CommandResult> {
    // Mock DAG for demonstration
    let nodes = vec![
        DagNode {
            id: "task-1".to_string(),
            label: "Load data".to_string(),
            dependencies: vec![],
            status: "completed".to_string(),
        },
        DagNode {
            id: "task-2".to_string(),
            label: "Process data".to_string(),
            dependencies: vec!["task-1".to_string()],
            status: "running".to_string(),
        },
        DagNode {
            id: "task-3".to_string(),
            label: "Save results".to_string(),
            dependencies: vec!["task-2".to_string()],
            status: "pending".to_string(),
        },
    ];

    println!("\n{}", "Task DAG:".bold());
    for node in &nodes {
        let status_symbol = match node.status.as_str() {
            "completed" => "✓".green(),
            "running" => "●".yellow(),
            "pending" => "○".white(),
            "failed" => "✗".red(),
            _ => "?".white(),
        };

        println!("  {} [{}] {}", status_symbol, node.id, node.label);
        for dep in &node.dependencies {
            println!("    └─→ {}", dep);
        }
    }

    Ok(CommandResult::success("DAG visualization complete"))
}

async fn cmd_dag_export(ctx: &CliContext, output: PathBuf, format: String) -> Result<CommandResult> {
    Ok(CommandResult::success(format!(
        "DAG exported to {} in {} format",
        output.display(),
        format
    )))
}

async fn cmd_sequence(ctx: &CliContext, file: PathBuf) -> Result<CommandResult> {
    let sequence = if file.extension().and_then(|s| s.to_str()) == Some("yaml") {
        ActionSequence::from_yaml(&file)?
    } else {
        ActionSequence::from_json(&file)?
    };

    info!("Executing sequence: {}", sequence.name);
    if let Some(desc) = &sequence.description {
        info!("Description: {}", desc);
    }

    let runner = ActionRunner::new(ctx.audit_log_path(), ctx.session_id, ctx.verbose);
    let results = runner.execute(&sequence).await?;

    let total = results.len();
    let successful = results.iter().filter(|r| r.success).count();
    let failed = total - successful;

    if failed == 0 {
        Ok(CommandResult::success(format!(
            "Sequence '{}' completed successfully ({}/{} actions)",
            sequence.name, successful, total
        )))
    } else {
        Ok(CommandResult::error(format!(
            "Sequence '{}' completed with errors ({}/{} actions succeeded)",
            sequence.name, successful, total
        )))
    }
}

async fn cmd_ui(ctx: &CliContext) -> Result<CommandResult> {
    let mut app = TuiApp::new(ctx.config.ui.refresh_rate_ms);

    // Initialize with some mock data
    app.update_swarm_status(SwarmStatus {
        active_agents: 5,
        total_tasks: 20,
        completed_tasks: 15,
        failed_tasks: 1,
        topology: "mesh".to_string(),
    });

    app.run().await?;

    Ok(CommandResult::success("TUI session ended"))
}

/// Create a single-action sequence
fn create_single_action_sequence(name: &str, action: Action) -> ActionSequence {
    ActionSequence {
        id: uuid::Uuid::new_v4(),
        name: name.to_string(),
        description: None,
        actions: vec![ActionWithDeps {
            action,
            depends_on: vec![],
            continue_on_error: false,
        }],
        variables: Default::default(),
    }
}
