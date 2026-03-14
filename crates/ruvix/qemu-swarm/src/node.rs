//! Single QEMU node management.
//!
//! This module handles the lifecycle of individual QEMU instances within the swarm.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::error::{SwarmError, SwarmResult};
use crate::network::MacAddress;
use crate::{CONSOLE_BASE_PORT, DEFAULT_QEMU_BINARY, GDB_BASE_PORT, MONITOR_BASE_PORT};

/// Unique identifier for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(Uuid);

impl NodeId {
    /// Create a new random node ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create a node ID from an index.
    pub fn from_index(index: usize) -> Self {
        // Use a deterministic UUID based on index for reproducibility
        let bytes = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
            (index >> 56) as u8,
            (index >> 48) as u8,
            (index >> 40) as u8,
            (index >> 32) as u8,
            (index >> 24) as u8,
            (index >> 16) as u8,
            (index >> 8) as u8,
            index as u8,
        ];
        Self(Uuid::from_bytes(bytes))
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node-{}", &self.0.to_string()[..8])
    }
}

/// Configuration for a single QEMU node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Node identifier.
    pub id: NodeId,

    /// Node index within the cluster.
    pub index: usize,

    /// Number of CPUs.
    pub cpu_count: u32,

    /// Memory in MB.
    pub memory_mb: u32,

    /// Path to kernel binary.
    pub kernel_path: Option<PathBuf>,

    /// Path to device tree blob.
    pub dtb_path: Option<PathBuf>,

    /// QEMU machine type.
    pub machine: String,

    /// CPU model.
    pub cpu_model: String,

    /// Path to QEMU binary.
    pub qemu_binary: PathBuf,

    /// Working directory for this node.
    pub work_dir: PathBuf,

    /// Additional QEMU arguments.
    pub extra_args: Vec<String>,

    /// MAC address for network interface.
    pub mac_address: MacAddress,

    /// Multicast group for networking.
    pub multicast_group: String,

    /// Enable GDB server.
    pub enable_gdb: bool,

    /// GDB port.
    pub gdb_port: u16,

    /// Enable QEMU monitor.
    pub enable_monitor: bool,

    /// Monitor port.
    pub monitor_port: u16,

    /// Console socket path.
    pub console_socket: PathBuf,
}

impl NodeConfig {
    /// Create a new node configuration with defaults.
    pub fn new(index: usize) -> Self {
        let id = NodeId::from_index(index);
        let work_dir = std::env::temp_dir().join(format!("ruvix-qemu-{}", id));

        Self {
            id,
            index,
            cpu_count: crate::DEFAULT_CPU_COUNT,
            memory_mb: crate::DEFAULT_MEMORY_MB,
            kernel_path: None,
            dtb_path: None,
            machine: "virt".to_string(),
            cpu_model: "cortex-a72".to_string(),
            qemu_binary: PathBuf::from(DEFAULT_QEMU_BINARY),
            work_dir: work_dir.clone(),
            extra_args: Vec::new(),
            mac_address: MacAddress::from_index(index),
            multicast_group: "239.0.0.1:5000".to_string(),
            enable_gdb: false,
            gdb_port: GDB_BASE_PORT + index as u16,
            enable_monitor: true,
            monitor_port: MONITOR_BASE_PORT + index as u16,
            console_socket: work_dir.join("console.sock"),
        }
    }

    /// Builder pattern: set CPU count.
    pub fn with_cpus(mut self, count: u32) -> Self {
        self.cpu_count = count;
        self
    }

    /// Builder pattern: set memory.
    pub fn with_memory(mut self, mb: u32) -> Self {
        self.memory_mb = mb;
        self
    }

    /// Builder pattern: set kernel path.
    pub fn with_kernel(mut self, path: PathBuf) -> Self {
        self.kernel_path = Some(path);
        self
    }

    /// Builder pattern: set DTB path.
    pub fn with_dtb(mut self, path: PathBuf) -> Self {
        self.dtb_path = Some(path);
        self
    }

    /// Builder pattern: enable GDB.
    pub fn with_gdb(mut self, enable: bool) -> Self {
        self.enable_gdb = enable;
        self
    }

    /// Builder pattern: add extra args.
    pub fn with_extra_args(mut self, args: Vec<String>) -> Self {
        self.extra_args = args;
        self
    }

    /// Generate QEMU command line arguments.
    pub fn to_qemu_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // Machine and CPU
        args.push("-machine".to_string());
        args.push(format!("{},virtualization=true", self.machine));
        args.push("-cpu".to_string());
        args.push(self.cpu_model.clone());
        args.push("-smp".to_string());
        args.push(self.cpu_count.to_string());

        // Memory
        args.push("-m".to_string());
        args.push(format!("{}M", self.memory_mb));

        // No graphics
        args.push("-nographic".to_string());

        // Kernel
        if let Some(ref kernel) = self.kernel_path {
            args.push("-kernel".to_string());
            args.push(kernel.to_string_lossy().to_string());
        }

        // DTB
        if let Some(ref dtb) = self.dtb_path {
            args.push("-dtb".to_string());
            args.push(dtb.to_string_lossy().to_string());
        }

        // Network: VirtIO with multicast socket for inter-VM communication
        args.push("-netdev".to_string());
        args.push(format!(
            "socket,id=net0,mcast={},localaddr=127.0.0.1",
            self.multicast_group
        ));
        args.push("-device".to_string());
        args.push(format!(
            "virtio-net-pci,netdev=net0,mac={}",
            self.mac_address
        ));

        // Serial console via Unix socket
        args.push("-serial".to_string());
        args.push(format!("unix:{},server,nowait", self.console_socket.display()));

        // GDB server
        if self.enable_gdb {
            args.push("-gdb".to_string());
            args.push(format!("tcp::{}", self.gdb_port));
            args.push("-S".to_string()); // Start paused
        }

        // QEMU monitor
        if self.enable_monitor {
            args.push("-monitor".to_string());
            args.push(format!("tcp:127.0.0.1:{},server,nowait", self.monitor_port));
        }

        // Deterministic execution
        args.push("-icount".to_string());
        args.push("shift=1,align=off,sleep=on".to_string());

        // Extra args
        args.extend(self.extra_args.clone());

        args
    }
}

/// Status of a QEMU node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is created but not started.
    Created,
    /// Node is starting up.
    Starting,
    /// Node is running and ready.
    Running,
    /// Node is paused (GDB breakpoint or user request).
    Paused,
    /// Node is stopping.
    Stopping,
    /// Node has stopped.
    Stopped,
    /// Node has failed.
    Failed,
}

impl Default for NodeStatus {
    fn default() -> Self {
        Self::Created
    }
}

/// Internal state of a QEMU node.
struct NodeState {
    status: NodeStatus,
    pid: Option<u32>,
    start_time: Option<std::time::Instant>,
    console_connected: bool,
    error_message: Option<String>,
}

impl Default for NodeState {
    fn default() -> Self {
        Self {
            status: NodeStatus::Created,
            pid: None,
            start_time: None,
            console_connected: false,
            error_message: None,
        }
    }
}

/// A single QEMU node instance.
pub struct QemuNode {
    config: NodeConfig,
    state: Arc<RwLock<NodeState>>,
    child: Arc<RwLock<Option<Child>>>,
    console_tx: Option<mpsc::Sender<String>>,
}

impl QemuNode {
    /// Create a new QEMU node.
    pub fn new(config: NodeConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(NodeState::default())),
            child: Arc::new(RwLock::new(None)),
            console_tx: None,
        }
    }

    /// Spawn a new QEMU node from configuration.
    pub async fn spawn(config: NodeConfig) -> SwarmResult<Self> {
        let mut node = Self::new(config);
        node.start().await?;
        Ok(node)
    }

    /// Get the node ID.
    pub fn id(&self) -> NodeId {
        self.config.id
    }

    /// Get the node index.
    pub fn index(&self) -> usize {
        self.config.index
    }

    /// Get the node configuration.
    pub fn config(&self) -> &NodeConfig {
        &self.config
    }

    /// Get the current node status.
    pub fn status(&self) -> NodeStatus {
        self.state.read().status
    }

    /// Get the process ID if running.
    pub fn pid(&self) -> Option<u32> {
        self.state.read().pid
    }

    /// Get the GDB port.
    pub fn gdb_port(&self) -> Option<u16> {
        if self.config.enable_gdb {
            Some(self.config.gdb_port)
        } else {
            None
        }
    }

    /// Get the monitor port.
    pub fn monitor_port(&self) -> Option<u16> {
        if self.config.enable_monitor {
            Some(self.config.monitor_port)
        } else {
            None
        }
    }

    /// Start the QEMU process.
    pub async fn start(&mut self) -> SwarmResult<()> {
        {
            let state = self.state.read();
            if state.status != NodeStatus::Created && state.status != NodeStatus::Stopped {
                return Err(SwarmError::InvalidNodeConfig(format!(
                    "Cannot start node in state {:?}",
                    state.status
                )));
            }
        }

        // Update status to starting
        {
            let mut state = self.state.write();
            state.status = NodeStatus::Starting;
        }

        // Create work directory
        tokio::fs::create_dir_all(&self.config.work_dir)
            .await
            .map_err(SwarmError::Io)?;

        // Build command
        let args = self.config.to_qemu_args();
        debug!(
            node_id = %self.config.id,
            "Starting QEMU with args: {:?}",
            args
        );

        let mut cmd = Command::new(&self.config.qemu_binary);
        cmd.args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // Spawn the process
        let child = cmd.spawn().map_err(|e| {
            SwarmError::qemu_spawn(format!(
                "Failed to spawn {}: {}",
                self.config.qemu_binary.display(),
                e
            ))
        })?;

        let pid = child.id();
        info!(
            node_id = %self.config.id,
            pid = ?pid,
            "QEMU process started"
        );

        // Store child process
        {
            let mut child_lock = self.child.write();
            *child_lock = Some(child);
        }

        // Update state
        {
            let mut state = self.state.write();
            state.status = NodeStatus::Running;
            state.pid = pid;
            state.start_time = Some(std::time::Instant::now());
        }

        // Start console reader task
        self.start_console_reader().await;

        Ok(())
    }

    /// Start the console reader task.
    async fn start_console_reader(&mut self) {
        let socket_path = self.config.console_socket.clone();
        let node_id = self.config.id;
        let state = Arc::clone(&self.state);

        let (tx, mut rx) = mpsc::channel::<String>(1024);
        self.console_tx = Some(tx);

        tokio::spawn(async move {
            // Wait for socket to be available
            for _ in 0..30 {
                if socket_path.exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }

            match UnixStream::connect(&socket_path).await {
                Ok(stream) => {
                    debug!(node_id = %node_id, "Console connected");
                    {
                        let mut s = state.write();
                        s.console_connected = true;
                    }

                    let reader = BufReader::new(stream);
                    let mut lines = reader.lines();

                    while let Ok(Some(line)) = lines.next_line().await {
                        debug!(node_id = %node_id, console = %line);
                    }
                }
                Err(e) => {
                    warn!(node_id = %node_id, "Failed to connect console: {}", e);
                }
            }
        });
    }

    /// Stop the QEMU process gracefully.
    pub async fn stop(&self) -> SwarmResult<()> {
        let status = self.status();
        if status == NodeStatus::Stopped || status == NodeStatus::Created {
            return Ok(());
        }

        {
            let mut state = self.state.write();
            state.status = NodeStatus::Stopping;
        }

        info!(node_id = %self.config.id, "Stopping QEMU node");

        // Try graceful shutdown via monitor if available
        if self.config.enable_monitor {
            if let Err(e) = self.send_monitor_command("quit").await {
                warn!(
                    node_id = %self.config.id,
                    "Failed to send quit command: {}",
                    e
                );
            }
        }

        // Wait for process to exit
        let mut child_guard = self.child.write();
        if let Some(ref mut child) = *child_guard {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(5)) => {
                    warn!(node_id = %self.config.id, "Timeout waiting for graceful shutdown, killing");
                    let _ = child.kill().await;
                }
                result = child.wait() => {
                    debug!(node_id = %self.config.id, "Process exited: {:?}", result);
                }
            }
        }

        *child_guard = None;

        {
            let mut state = self.state.write();
            state.status = NodeStatus::Stopped;
            state.pid = None;
        }

        // Cleanup socket
        let _ = tokio::fs::remove_file(&self.config.console_socket).await;

        Ok(())
    }

    /// Kill the QEMU process immediately.
    pub async fn kill(&self) -> SwarmResult<()> {
        let mut child_guard = self.child.write();
        if let Some(ref mut child) = *child_guard {
            child.kill().await.map_err(SwarmError::Io)?;
        }
        *child_guard = None;

        {
            let mut state = self.state.write();
            state.status = NodeStatus::Stopped;
            state.pid = None;
        }

        Ok(())
    }

    /// Pause execution (requires GDB or QEMU monitor).
    pub async fn pause(&self) -> SwarmResult<()> {
        if !self.config.enable_monitor {
            return Err(SwarmError::InvalidNodeConfig(
                "Monitor not enabled for pause".to_string(),
            ));
        }

        self.send_monitor_command("stop").await?;

        {
            let mut state = self.state.write();
            state.status = NodeStatus::Paused;
        }

        Ok(())
    }

    /// Resume execution.
    pub async fn resume(&self) -> SwarmResult<()> {
        if !self.config.enable_monitor {
            return Err(SwarmError::InvalidNodeConfig(
                "Monitor not enabled for resume".to_string(),
            ));
        }

        self.send_monitor_command("cont").await?;

        {
            let mut state = self.state.write();
            state.status = NodeStatus::Running;
        }

        Ok(())
    }

    /// Send a command to the QEMU monitor.
    pub async fn send_monitor_command(&self, command: &str) -> SwarmResult<String> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpStream;

        let addr = format!("127.0.0.1:{}", self.config.monitor_port);
        let mut stream = TcpStream::connect(&addr)
            .await
            .map_err(|e| SwarmError::network(format!("Failed to connect to monitor: {}", e)))?;

        // Read initial prompt
        let mut buf = [0u8; 1024];
        let _ = tokio::time::timeout(Duration::from_millis(500), stream.read(&mut buf)).await;

        // Send command
        stream
            .write_all(format!("{}\n", command).as_bytes())
            .await
            .map_err(SwarmError::Io)?;

        // Read response
        let mut response = Vec::new();
        let _ = tokio::time::timeout(Duration::from_millis(500), async {
            loop {
                let n = stream.read(&mut buf).await?;
                if n == 0 {
                    break;
                }
                response.extend_from_slice(&buf[..n]);
            }
            Ok::<_, std::io::Error>(())
        })
        .await;

        Ok(String::from_utf8_lossy(&response).to_string())
    }

    /// Write data to the console.
    pub async fn write_console(&self, data: &[u8]) -> SwarmResult<()> {
        use tokio::io::AsyncWriteExt;

        let mut stream = UnixStream::connect(&self.config.console_socket)
            .await
            .map_err(|e| {
                SwarmError::SocketConnection {
                    path: self.config.console_socket.clone(),
                }
            })?;

        stream.write_all(data).await.map_err(SwarmError::Io)?;
        Ok(())
    }

    /// Check if the node is healthy.
    pub async fn health_check(&self) -> bool {
        let status = self.status();
        if status != NodeStatus::Running {
            return false;
        }

        // Check if process is still alive
        let child_guard = self.child.read();
        if child_guard.is_none() {
            return false;
        }

        // Try to ping monitor
        if self.config.enable_monitor {
            if let Ok(response) = self.send_monitor_command("info status").await {
                return response.contains("running") || response.contains("paused");
            }
        }

        true
    }
}

impl Drop for QemuNode {
    fn drop(&mut self) {
        // Cleanup will happen when child is dropped (kill_on_drop=true)
        let _ = std::fs::remove_file(&self.config.console_socket);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_from_index() {
        let id1 = NodeId::from_index(0);
        let id2 = NodeId::from_index(0);
        let id3 = NodeId::from_index(1);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_node_config_qemu_args() {
        let config = NodeConfig::new(0)
            .with_cpus(4)
            .with_memory(1024);

        let args = config.to_qemu_args();
        assert!(args.contains(&"-smp".to_string()));
        assert!(args.contains(&"4".to_string()));
        assert!(args.contains(&"1024M".to_string()));
    }

    #[test]
    fn test_mac_address_generation() {
        let config0 = NodeConfig::new(0);
        let config1 = NodeConfig::new(1);

        assert_ne!(config0.mac_address, config1.mac_address);
    }
}
