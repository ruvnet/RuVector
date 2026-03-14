//! QEMU Swarm CLI - Launch and manage RuVix QEMU clusters.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use ruvix_qemu_swarm::{
    ClusterConfig, QemuCluster, SwarmConfig, Topology,
    orchestrator::scenarios,
};

#[derive(Parser)]
#[command(name = "qemu-swarm")]
#[command(author = "RuVector Contributors")]
#[command(version = "0.1.0")]
#[command(about = "QEMU Swarm Simulation for RuVix cluster testing")]
struct Cli {
    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch a QEMU swarm cluster
    Launch {
        /// Configuration file path
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Number of nodes (overrides config)
        #[arg(short, long)]
        nodes: Option<usize>,

        /// Network topology (mesh, ring, star, tree)
        #[arg(short, long)]
        topology: Option<String>,

        /// Path to kernel binary
        #[arg(short, long)]
        kernel: Option<PathBuf>,

        /// Memory per node in MB
        #[arg(short, long, default_value = "512")]
        memory: u32,

        /// CPUs per node
        #[arg(long, default_value = "2")]
        cpus: u32,

        /// Enable GDB servers
        #[arg(long)]
        gdb: bool,

        /// Wait for ready pattern
        #[arg(long)]
        wait_pattern: Option<String>,

        /// Timeout in seconds
        #[arg(long, default_value = "60")]
        timeout: u64,
    },

    /// Run a predefined test scenario
    Test {
        /// Scenario name
        #[arg(value_parser = ["leader-crash", "network-partition", "cascading-failures", "slow-network", "byzantine"])]
        scenario: String,

        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Number of nodes
        #[arg(short, long, default_value = "3")]
        nodes: usize,

        /// Output file for results
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Deploy an RVF package to the cluster
    Deploy {
        /// RVF package path
        rvf: PathBuf,

        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Target nodes (comma-separated)
        #[arg(long)]
        nodes: Option<String>,
    },

    /// Show cluster status
    Status {
        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Generate a sample configuration file
    Init {
        /// Output file
        #[arg(short, long, default_value = "swarm.toml")]
        output: PathBuf,

        /// Number of nodes
        #[arg(short, long, default_value = "3")]
        nodes: usize,

        /// Topology
        #[arg(short, long, default_value = "mesh")]
        topology: String,
    },

    /// Validate a configuration file
    Validate {
        /// Configuration file to validate
        config: PathBuf,
    },

    /// Show network topology
    Topology {
        /// Topology type
        #[arg(value_parser = ["mesh", "ring", "star", "tree"])]
        topology_type: String,

        /// Number of nodes
        #[arg(short, long, default_value = "8")]
        nodes: usize,
    },
}

fn setup_logging(verbosity: u8) {
    let level = match verbosity {
        0 => Level::WARN,
        1 => Level::INFO,
        2 => Level::DEBUG,
        _ => Level::TRACE,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    match cli.command {
        Commands::Launch {
            config,
            nodes,
            topology,
            kernel,
            memory,
            cpus,
            gdb,
            wait_pattern,
            timeout,
        } => {
            cmd_launch(config, nodes, topology, kernel, memory, cpus, gdb, wait_pattern, timeout).await
        }
        Commands::Test {
            scenario,
            config,
            nodes,
            output,
        } => cmd_test(scenario, config, nodes, output).await,
        Commands::Deploy { rvf, config, nodes } => cmd_deploy(rvf, config, nodes).await,
        Commands::Status { config } => cmd_status(config).await,
        Commands::Init {
            output,
            nodes,
            topology,
        } => cmd_init(output, nodes, topology),
        Commands::Validate { config } => cmd_validate(config),
        Commands::Topology {
            topology_type,
            nodes,
        } => cmd_topology(topology_type, nodes),
    }
}

async fn cmd_launch(
    config_path: Option<PathBuf>,
    nodes: Option<usize>,
    topology: Option<String>,
    kernel: Option<PathBuf>,
    memory: u32,
    cpus: u32,
    gdb: bool,
    wait_pattern: Option<String>,
    timeout: u64,
) -> Result<()> {
    // Build configuration
    let mut config = if let Some(path) = config_path {
        info!(path = %path.display(), "Loading configuration");
        ClusterConfig::from_file(&path)?
    } else {
        ClusterConfig::default()
    };

    // Apply overrides
    if let Some(n) = nodes {
        config.node_count = n;
    }
    if let Some(t) = topology {
        config.topology = parse_topology(&t)?;
    }
    if let Some(k) = kernel {
        config.node_defaults.kernel = Some(k);
    }
    config.node_defaults.memory_mb = memory;
    config.node_defaults.cpu_count = cpus;
    config.node_defaults.enable_gdb = gdb;

    info!(
        name = %config.name,
        nodes = config.node_count,
        topology = %config.topology,
        "Launching QEMU swarm"
    );

    // Create and start cluster
    let mut cluster = QemuCluster::new(config).await?;
    cluster.start_all().await?;

    // Wait for ready
    let timeout_duration = Duration::from_secs(timeout);
    if let Some(pattern) = wait_pattern {
        info!(pattern = %pattern, "Waiting for boot pattern");
        cluster.wait_for_boot_message(&pattern, timeout_duration).await?;
    } else {
        cluster.wait_for_ready(timeout_duration).await?;
    }

    // Print status
    cluster.print_status();

    info!("Cluster is running. Press Ctrl+C to stop.");

    // Wait for Ctrl+C
    tokio::signal::ctrl_c().await?;

    info!("Shutting down cluster...");
    cluster.stop_all().await?;

    Ok(())
}

async fn cmd_test(
    scenario: String,
    config_path: Option<PathBuf>,
    nodes: usize,
    output: Option<PathBuf>,
) -> Result<()> {
    let config = if let Some(path) = config_path {
        ClusterConfig::from_file(&path)?
    } else {
        ClusterConfig::builder()
            .name("chaos-test")
            .node_count(nodes)
            .topology(Topology::Mesh)
            .build()?
    };

    info!(
        scenario = %scenario,
        nodes = config.node_count,
        "Running chaos test scenario"
    );

    let mut cluster = QemuCluster::new(config).await?;
    cluster.start_all().await?;
    cluster.wait_for_ready(Duration::from_secs(30)).await?;

    let chaos_scenario = match scenario.as_str() {
        "leader-crash" => scenarios::leader_crash(),
        "network-partition" => scenarios::network_partition(nodes),
        "cascading-failures" => scenarios::cascading_failures(nodes),
        "slow-network" => scenarios::slow_network(nodes),
        "byzantine" => scenarios::byzantine(0),
        _ => anyhow::bail!("Unknown scenario: {}", scenario),
    };

    let result = cluster.orchestrator().run_chaos_scenario(chaos_scenario).await?;

    println!("\nChaos Test Results:");
    println!("  Scenario: {}", result.scenario_name);
    println!("  Faults injected: {}", result.faults_injected);
    println!("  Duration: {:?}", result.end_time - result.start_time);
    println!("  Healthy nodes during: {}/{}",
        result.metrics_during.cluster.healthy_nodes,
        result.metrics_during.cluster.total_nodes
    );

    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(&output_path, json)?;
        info!(path = %output_path.display(), "Results written");
    }

    cluster.stop_all().await?;
    Ok(())
}

async fn cmd_deploy(rvf: PathBuf, config_path: Option<PathBuf>, nodes: Option<String>) -> Result<()> {
    let config = if let Some(path) = config_path {
        ClusterConfig::from_file(&path)?
    } else {
        ClusterConfig::default()
    };

    let target_nodes = nodes.map(|s| {
        s.split(',')
            .filter_map(|n| n.trim().parse::<usize>().ok())
            .collect::<Vec<_>>()
    });

    info!(
        rvf = %rvf.display(),
        nodes = ?target_nodes,
        "Deploying RVF package"
    );

    let mut cluster = QemuCluster::new(config).await?;
    cluster.start_all().await?;
    cluster.wait_for_ready(Duration::from_secs(30)).await?;

    let result = cluster
        .orchestrator()
        .deploy_rvf_to_nodes(&rvf, target_nodes)
        .await?;

    println!("\nDeployment Results:");
    println!("  RVF: {}", rvf.display());
    println!("  Success rate: {:.1}%", result.success_rate());
    println!("  Successful nodes: {:?}", result.successful_nodes);
    if !result.failed_nodes.is_empty() {
        println!("  Failed nodes: {:?}", result.failed_nodes);
    }

    cluster.stop_all().await?;
    Ok(())
}

async fn cmd_status(config_path: Option<PathBuf>) -> Result<()> {
    let config = if let Some(path) = config_path {
        ClusterConfig::from_file(&path)?
    } else {
        ClusterConfig::default()
    };

    let cluster = QemuCluster::new(config).await?;
    cluster.print_status();
    Ok(())
}

fn cmd_init(output: PathBuf, nodes: usize, topology: String) -> Result<()> {
    let topology_enum = parse_topology(&topology)?;

    let config = format!(
        r#"# RuVix QEMU Swarm Configuration
# Generated by qemu-swarm init

[cluster]
name = "ruvix-swarm"
node_count = {}
topology = "{}"
startup_delay_ms = 500

[node.defaults]
cpu_count = 2
memory_mb = 512
machine = "virt"
cpu_model = "cortex-a72"
# kernel = "/path/to/ruvix-kernel"
# dtb = "/path/to/ruvix.dtb"
enable_gdb = false
enable_monitor = true

[network]
base_mac = "52:54:00:12:34:00"
multicast_group = "239.0.0.1:5000"
mtu = 1500

# Per-node overrides (optional)
# [[node.overrides]]
# index = 0
# cpu_count = 4
# memory_mb = 1024
# extra_args = ["-d", "guest_errors"]

# Test scenarios (optional)
# [[scenarios]]
# name = "leader-failover"
# description = "Test leader node failure and recovery"
# [[scenarios.steps]]
# action = "wait"
# duration_ms = 5000
# [[scenarios.steps]]
# action = "fault"
# fault_type = "crash"
# node = 0
"#,
        nodes, topology_enum
    );

    std::fs::write(&output, config)?;
    println!("Configuration written to: {}", output.display());
    Ok(())
}

fn cmd_validate(config_path: PathBuf) -> Result<()> {
    info!(path = %config_path.display(), "Validating configuration");

    let config = SwarmConfig::from_file(&config_path)?;

    println!("Configuration valid!");
    println!("  Cluster: {}", config.cluster.name);
    println!("  Nodes: {}", config.cluster.node_count);
    println!("  Topology: {:?}", config.cluster.topology);
    println!("  Node defaults:");
    println!("    CPUs: {}", config.node.defaults.cpu_count);
    println!("    Memory: {}MB", config.node.defaults.memory_mb);
    println!("    Machine: {}", config.node.defaults.machine);
    println!("  Network:");
    println!("    Multicast: {}", config.network.multicast_group);
    println!("    Base MAC: {}", config.network.base_mac);

    Ok(())
}

fn cmd_topology(topology_type: String, nodes: usize) -> Result<()> {
    use ruvix_qemu_swarm::network::NetworkTopology;

    let topology = parse_topology(&topology_type)?;
    let network_topology = NetworkTopology::new(topology, nodes);

    println!("{}", network_topology.ascii_diagram());

    // Show sample node info
    println!("\nSample paths:");
    for i in 0..nodes.min(3) {
        let distances = network_topology.shortest_paths(i);
        let avg_dist: f64 = distances.iter()
            .filter(|&&d| d != usize::MAX && d > 0)
            .map(|&d| d as f64)
            .sum::<f64>() / (nodes - 1) as f64;
        println!("  Node {} average distance: {:.2}", i, avg_dist);
    }

    Ok(())
}

fn parse_topology(s: &str) -> Result<Topology> {
    match s.to_lowercase().as_str() {
        "mesh" => Ok(Topology::Mesh),
        "ring" => Ok(Topology::Ring),
        "star" => Ok(Topology::Star),
        "tree" => Ok(Topology::Tree),
        _ => anyhow::bail!("Unknown topology: {}. Valid: mesh, ring, star, tree", s),
    }
}
