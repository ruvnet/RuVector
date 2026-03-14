# RuVix QEMU Swarm Simulation System

A comprehensive simulation environment for testing distributed RuVix clusters using multiple QEMU instances. Enables swarm intelligence testing, multi-node consensus verification, and chaos engineering for the RuVix Cognition Kernel.

## Architecture

```
                           ┌─────────────────────────────────────┐
                           │         SwarmOrchestrator          │
                           │   - RVF deployment                  │
                           │   - Fault injection                 │
                           │   - Metrics collection              │
                           │   - Chaos scenarios                 │
                           └─────────────────┬───────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
    ┌─────────▼─────────┐        ┌──────────▼──────────┐        ┌─────────▼─────────┐
    │    QemuCluster    │        │   VirtualNetwork    │        │     ConsoleIO     │
    │  - Node lifecycle │        │  - Topology mgmt    │        │  - I/O mux        │
    │  - Health checks  │        │  - Partitioning     │        │  - Log aggregation│
    │  - Status         │        │  - Latency sim      │        │  - Filtering      │
    └─────────┬─────────┘        └──────────┬──────────┘        └─────────┬─────────┘
              │                              │                              │
    ┌─────────┼─────────┬──────────┬────────┼────────┬──────────┬─────────┼─────────┐
    │         │         │          │        │        │          │         │         │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐    │   ┌───▼───┐ ┌───▼───┐ ┌───▼───┐     │
│ Node0 │ │ Node1 │ │ Node2 │ │ Node3 │    │   │ Node4 │ │ Node5 │ │ Node6 │ ... │
│ QEMU  │ │ QEMU  │ │ QEMU  │ │ QEMU  │    │   │ QEMU  │ │ QEMU  │ │ QEMU  │     │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘    │   └───┬───┘ └───┬───┘ └───┬───┘     │
    │         │         │          │        │        │         │         │         │
    └─────────┴─────────┴──────────┴────────┴────────┴─────────┴─────────┴─────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   VirtIO Network    │
                         │  (Multicast UDP)    │
                         └─────────────────────┘
```

## Quick Start

### Prerequisites

- QEMU 8.0+ with AArch64 support (`qemu-system-aarch64`)
- Rust 1.77+ (for building)
- socat (for shell scripts)

### Installation

```bash
# From the ruvix workspace
cd crates/ruvix/qemu-swarm
cargo build --release
```

### Launch a 3-Node Cluster

```bash
# Using the CLI
cargo run --release --bin qemu-swarm -- launch \
    --nodes 3 \
    --topology mesh \
    --memory 512 \
    --cpus 2

# Using the shell script
./scripts/launch-swarm.sh -n 3 -t mesh -m 512
```

### Launch from Configuration

```bash
# Using a pre-defined config
cargo run --release --bin qemu-swarm -- launch --config configs/3-node-cluster.toml

# Generate a new config
cargo run --release --bin qemu-swarm -- init --output my-cluster.toml --nodes 5 --topology ring
```

## Network Topologies

| Topology | Description | Use Case |
|----------|-------------|----------|
| **Mesh** | Fully connected, all nodes can communicate directly | General consensus testing |
| **Ring** | Each node connects to two neighbors | Gossip protocol testing |
| **Star** | Central hub (node 0) with spokes | Leader-follower architectures |
| **Tree** | Binary tree structure | Hierarchical systems |

```bash
# Visualize a topology
cargo run --release --bin qemu-swarm -- topology mesh --nodes 8
```

## Configuration Reference

### Cluster Configuration (TOML)

```toml
[cluster]
name = "my-cluster"
node_count = 8
topology = "mesh"           # mesh, ring, star, tree
startup_delay_ms = 500      # Delay between node starts

[node.defaults]
cpu_count = 2
memory_mb = 512
machine = "virt"
cpu_model = "cortex-a72"
kernel = "/path/to/ruvix-kernel"    # Optional
dtb = "/path/to/ruvix.dtb"          # Optional
enable_gdb = false
enable_monitor = true

[network]
base_mac = "52:54:00:12:34:00"
multicast_group = "239.0.0.1:5000"
mtu = 1500
latency_us = 0              # Simulated latency
packet_loss_pct = 0         # Simulated packet loss

# Per-node overrides
[[node.overrides]]
index = 0
cpu_count = 4               # Leader gets more resources
memory_mb = 1024
```

### Test Scenarios

Define automated test scenarios in configuration:

```toml
[[scenarios]]
name = "leader-failover"
description = "Test leader crash and re-election"

[[scenarios.steps]]
action = "wait"
duration_ms = 5000

[[scenarios.steps]]
action = "fault"
fault_type = "crash"
node = 0

[[scenarios.steps]]
action = "wait"
duration_ms = 15000

[[scenarios.steps]]
action = "metrics"
output = "./results.json"

[[scenarios.assertions]]
type = "healthy_nodes"
expected = "2"
```

## CLI Commands

### Launch

```bash
qemu-swarm launch [OPTIONS]
    -c, --config <FILE>     Load configuration from file
    -n, --nodes <NUM>       Number of nodes (default: 3)
    -t, --topology <TYPE>   Topology type (mesh, ring, star, tree)
    -k, --kernel <PATH>     Path to kernel binary
    -m, --memory <MB>       Memory per node (default: 512)
    --cpus <NUM>            CPUs per node (default: 2)
    -g, --gdb               Enable GDB servers
    -w, --wait <PATTERN>    Wait for boot pattern
    --timeout <SECS>        Timeout in seconds (default: 60)
```

### Test Scenarios

```bash
# Run a predefined chaos scenario
qemu-swarm test leader-crash --nodes 3 --output results.json

# Available scenarios:
#   - leader-crash       Crash node 0 (leader)
#   - network-partition  Split cluster in half
#   - cascading-failures Progressive node failures
#   - slow-network       Add latency to all nodes
#   - byzantine          Data corruption on one node
```

### Deploy RVF

```bash
qemu-swarm deploy ./my-package.rvf --nodes 0,1,2
```

### Monitor

```bash
# Using the TUI monitor
cargo run --bin swarm-monitor -- --tui --nodes 8

# Using the shell script
./scripts/monitor.sh -n 8 -f error -t
```

## Chaos Engineering

### Fault Types

| Fault | Description | QEMU Implementation |
|-------|-------------|---------------------|
| `Crash` | Immediate node termination | `kill -9` |
| `Pause` | Freeze execution | QEMU monitor `stop` |
| `NetworkPartition` | Isolate from cluster | Separate multicast group |
| `NetworkSlow` | Add latency | netem via QEMU |
| `PacketLoss` | Drop packets | netem via QEMU |
| `CpuOverload` | Busy loop | Inject via console |
| `MemoryPressure` | Allocate memory | Inject via console |
| `ClockSkew` | Time offset | QEMU `-rtc` |
| `DataCorruption` | Random bit flips | Custom driver |

### Example: Byzantine Fault Tolerance Test

```rust
use ruvix_qemu_swarm::prelude::*;
use ruvix_qemu_swarm::orchestrator::scenarios;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create 8-node cluster (can tolerate f=2 Byzantine faults)
    let config = ClusterConfig::builder()
        .name("bft-test")
        .node_count(8)
        .topology(Topology::Mesh)
        .build()?;

    let mut cluster = QemuCluster::new(config).await?;
    cluster.start_all().await?;
    cluster.wait_for_ready(Duration::from_secs(60)).await?;

    // Inject Byzantine faults
    let orchestrator = cluster.orchestrator();
    orchestrator.inject_fault(6, FaultType::DataCorruption { probability_pct: 10 }).await?;
    orchestrator.inject_fault(7, FaultType::DataCorruption { probability_pct: 10 }).await?;

    // Wait and collect metrics
    tokio::time::sleep(Duration::from_secs(60)).await;
    let metrics = orchestrator.collect_metrics().await?;

    // Verify cluster still achieved consensus
    assert!(metrics.cluster.healthy_nodes >= 6);

    cluster.stop_all().await?;
    Ok(())
}
```

## Console Monitoring

The monitor aggregates console output from all nodes with:

- **Color-coded output** per node
- **Severity detection** (debug, info, warn, error, panic)
- **Pattern filtering** for specific messages
- **Statistics** (total lines, errors, panics)
- **TUI mode** for real-time dashboard

```bash
# Filter errors only
./scripts/monitor.sh -f error

# Search for specific pattern
./scripts/monitor.sh -p "consensus"

# TUI mode with timestamps
cargo run --bin swarm-monitor -- --tui -t
```

## Performance Considerations

| Cluster Size | RAM Required | Recommended Host |
|--------------|--------------|------------------|
| 3 nodes | ~2 GB | 8 GB RAM |
| 8 nodes | ~5 GB | 16 GB RAM |
| 16 nodes | ~10 GB | 32 GB RAM |
| 64 nodes (max) | ~40 GB | 64+ GB RAM |

Tips for large clusters:
- Use `startup_delay_ms` to prevent resource spikes
- Enable monitor only on subset of nodes
- Use ring topology to reduce network overhead
- Consider host CPU pinning for determinism

## Integration with RuVix

This swarm system is designed to test:

1. **Consensus Mechanisms**: Multi-node agreement on state changes
2. **Proof Attestation**: Cross-node proof verification
3. **RVF Distribution**: Deploying cognitive containers across cluster
4. **Fault Tolerance**: Recovery from node failures
5. **Network Partitions**: Split-brain handling
6. **Byzantine Behavior**: Detecting malicious nodes

## API Reference

See the generated documentation:

```bash
cargo doc --open -p ruvix-qemu-swarm
```

## License

MIT OR Apache-2.0
