//! Real-time swarm monitor - aggregate and display console output from all nodes.

use std::collections::HashMap;
use std::io::{stdout, Write};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{self, ClearType},
};
use tokio::io::AsyncBufReadExt;

#[derive(Parser)]
#[command(name = "swarm-monitor")]
#[command(about = "Real-time console monitor for QEMU swarm")]
struct Cli {
    /// Socket directory (contains console.sock files)
    #[arg(short, long)]
    socket_dir: Option<PathBuf>,

    /// Number of nodes to monitor
    #[arg(short, long, default_value = "3")]
    nodes: usize,

    /// Filter by severity (debug, info, warn, error, panic)
    #[arg(short, long)]
    filter: Option<String>,

    /// Search for pattern
    #[arg(short = 'p', long)]
    pattern: Option<String>,

    /// Show only specific nodes (comma-separated)
    #[arg(long)]
    only_nodes: Option<String>,

    /// Output file for logging
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Enable TUI mode
    #[arg(long)]
    tui: bool,

    /// Disable colors
    #[arg(long)]
    no_color: bool,
}

struct MonitorState {
    node_buffers: HashMap<usize, Vec<String>>,
    total_lines: u64,
    error_count: u64,
    panic_count: u64,
    filter_pattern: Option<String>,
    active_nodes: Vec<usize>,
    paused: bool,
}

impl MonitorState {
    fn new(nodes: usize) -> Self {
        Self {
            node_buffers: (0..nodes).map(|i| (i, Vec::new())).collect(),
            total_lines: 0,
            error_count: 0,
            panic_count: 0,
            filter_pattern: None,
            active_nodes: (0..nodes).collect(),
            paused: false,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.tui {
        run_tui(cli).await
    } else {
        run_simple(cli).await
    }
}

async fn run_simple(cli: Cli) -> Result<()> {
    let socket_dir = cli.socket_dir.unwrap_or_else(|| {
        std::env::temp_dir().join("ruvix-swarm")
    });

    let filter_nodes: Option<Vec<usize>> = cli.only_nodes.map(|s| {
        s.split(',')
            .filter_map(|n| n.trim().parse().ok())
            .collect()
    });

    let mut output_file = cli.output.map(|p| {
        std::fs::File::create(p).expect("Failed to create output file")
    });

    println!("RuVix Swarm Monitor");
    println!("==================");
    println!("Watching {} nodes in {}", cli.nodes, socket_dir.display());
    if let Some(ref pattern) = cli.pattern {
        println!("Filter pattern: {}", pattern);
    }
    println!("Press Ctrl+C to exit\n");

    // Colors for different nodes
    let colors = [
        Color::Cyan,
        Color::Green,
        Color::Yellow,
        Color::Magenta,
        Color::Blue,
        Color::Red,
        Color::White,
        Color::DarkCyan,
    ];

    // Simulate reading from console sockets
    // In a real implementation, this would use Unix sockets
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(usize, String)>(1024);

    // Spawn readers for each node
    for i in 0..cli.nodes {
        let socket_path = socket_dir.join(format!("node-{}/console.sock", i));
        let tx = tx.clone();
        let filter = cli.filter.clone();
        let pattern = cli.pattern.clone();

        tokio::spawn(async move {
            // Wait for socket
            for _ in 0..30 {
                if socket_path.exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }

            if let Ok(stream) = tokio::net::UnixStream::connect(&socket_path).await {
                let reader = tokio::io::BufReader::new(stream);
                let mut lines = reader.lines();

                while let Ok(Some(line)) = lines.next_line().await {
                    // Apply filters
                    if let Some(ref f) = filter {
                        let lower = line.to_lowercase();
                        let passes = match f.as_str() {
                            "error" => lower.contains("error") || lower.contains("panic"),
                            "warn" => lower.contains("warn") || lower.contains("error") || lower.contains("panic"),
                            "info" => !lower.contains("debug") && !lower.contains("trace"),
                            "debug" => !lower.contains("trace"),
                            _ => true,
                        };
                        if !passes {
                            continue;
                        }
                    }

                    if let Some(ref p) = pattern {
                        if !line.contains(p) {
                            continue;
                        }
                    }

                    let _ = tx.send((i, line)).await;
                }
            }
        });
    }

    drop(tx);

    // Print incoming lines
    while let Some((node, line)) = rx.recv().await {
        // Check node filter
        if let Some(ref nodes) = filter_nodes {
            if !nodes.contains(&node) {
                continue;
            }
        }

        let color = colors[node % colors.len()];
        let timestamp = chrono::Local::now().format("%H:%M:%S%.3f");

        if cli.no_color {
            println!("[{}] [N{}] {}", timestamp, node, line);
        } else {
            execute!(
                stdout(),
                SetForegroundColor(color),
                Print(format!("[{}] [N{}] ", timestamp, node)),
                ResetColor,
                Print(format!("{}\n", line))
            )?;
        }

        // Write to output file
        if let Some(ref mut f) = output_file {
            writeln!(f, "[{}] [N{}] {}", timestamp, node, line)?;
        }
    }

    Ok(())
}

async fn run_tui(cli: Cli) -> Result<()> {
    // Enable raw mode for TUI
    terminal::enable_raw_mode()?;

    let mut stdout = stdout();
    execute!(
        stdout,
        terminal::EnterAlternateScreen,
        cursor::Hide
    )?;

    let mut state = MonitorState::new(cli.nodes);
    state.filter_pattern = cli.pattern.clone();

    // Main TUI loop
    let result = tui_loop(&mut stdout, &cli, &mut state).await;

    // Cleanup
    execute!(
        stdout,
        cursor::Show,
        terminal::LeaveAlternateScreen
    )?;
    terminal::disable_raw_mode()?;

    result
}

async fn tui_loop(
    stdout: &mut std::io::Stdout,
    cli: &Cli,
    state: &mut MonitorState,
) -> Result<()> {
    let socket_dir = cli.socket_dir.clone().unwrap_or_else(|| {
        std::env::temp_dir().join("ruvix-swarm")
    });

    let (tx, mut rx) = tokio::sync::mpsc::channel::<(usize, String)>(1024);

    // Spawn readers
    for i in 0..cli.nodes {
        let socket_path = socket_dir.join(format!("node-{}/console.sock", i));
        let tx = tx.clone();

        tokio::spawn(async move {
            if let Ok(stream) = tokio::net::UnixStream::connect(&socket_path).await {
                let reader = tokio::io::BufReader::new(stream);
                let mut lines = reader.lines();

                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = tx.send((i, line)).await;
                }
            }
        });
    }

    drop(tx);

    let mut interval = tokio::time::interval(Duration::from_millis(100));

    loop {
        // Handle input
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Char(' ') => state.paused = !state.paused,
                    KeyCode::Char('c') => {
                        for buf in state.node_buffers.values_mut() {
                            buf.clear();
                        }
                    }
                    _ => {}
                }
            }
        }

        // Receive messages
        while let Ok((node, line)) = rx.try_recv() {
            if !state.paused {
                state.total_lines += 1;

                let lower = line.to_lowercase();
                if lower.contains("error") {
                    state.error_count += 1;
                }
                if lower.contains("panic") {
                    state.panic_count += 1;
                }

                if let Some(buf) = state.node_buffers.get_mut(&node) {
                    buf.push(line);
                    if buf.len() > 100 {
                        buf.remove(0);
                    }
                }
            }
        }

        // Render
        render_tui(stdout, state, cli.nodes)?;

        interval.tick().await;
    }

    Ok(())
}

fn render_tui(stdout: &mut std::io::Stdout, state: &MonitorState, nodes: usize) -> Result<()> {
    let (cols, rows) = terminal::size()?;

    execute!(
        stdout,
        cursor::MoveTo(0, 0),
        terminal::Clear(ClearType::All)
    )?;

    // Header
    let status = if state.paused { "PAUSED" } else { "RUNNING" };
    execute!(
        stdout,
        SetForegroundColor(Color::Cyan),
        Print(format!("RuVix Swarm Monitor - {} nodes - {} ", nodes, status)),
        ResetColor,
        Print(format!("| Lines: {} | Errors: {} | Panics: {}\n",
            state.total_lines, state.error_count, state.panic_count)),
        Print(format!("{}\n", "=".repeat(cols as usize - 1)))
    )?;

    // Calculate columns per node
    let cols_per_node = cols as usize / nodes.min(4);
    let visible_rows = (rows - 4) as usize;

    // Node columns
    for row in 0..visible_rows {
        let mut line_parts = Vec::new();

        for node in 0..nodes.min(4) {
            if let Some(buf) = state.node_buffers.get(&node) {
                let start = buf.len().saturating_sub(visible_rows);
                if let Some(line) = buf.get(start + row) {
                    let truncated: String = line.chars().take(cols_per_node - 2).collect();
                    line_parts.push(format!("{:width$}", truncated, width = cols_per_node - 1));
                } else {
                    line_parts.push(" ".repeat(cols_per_node - 1));
                }
            }
        }

        execute!(stdout, Print(format!("{}\n", line_parts.join("|"))))?;
    }

    // Footer
    execute!(
        stdout,
        cursor::MoveTo(0, rows - 1),
        SetForegroundColor(Color::DarkGrey),
        Print("[q] Quit  [Space] Pause  [c] Clear"),
        ResetColor
    )?;

    stdout.flush()?;
    Ok(())
}
