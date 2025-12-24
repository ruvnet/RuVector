//! Terminal User Interface with ratatui
//!
//! Provides an interactive TUI for monitoring swarms, agents, and actions.

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, List, ListItem, Paragraph, Row, Table, Tabs, Wrap,
    },
    Frame, Terminal,
};
use std::io;
use std::time::{Duration, Instant};

use crate::{action::ActionResult, AgentStatus, DagNode, SwarmStatus};

/// TUI application state
pub struct TuiApp {
    /// Current tab index
    current_tab: usize,

    /// Tab names
    tabs: Vec<&'static str>,

    /// Swarm status
    swarm_status: SwarmStatus,

    /// List of agents
    agents: Vec<AgentStatus>,

    /// Recent actions
    action_history: Vec<ActionHistoryEntry>,

    /// DAG nodes for visualization
    dag_nodes: Vec<DagNode>,

    /// Metrics data
    metrics: MetricsData,

    /// Selected item in current view
    selected: usize,

    /// Whether to show help
    show_help: bool,

    /// Last update time
    last_update: Instant,

    /// Refresh rate in milliseconds
    refresh_rate_ms: u64,
}

/// Action history entry for display
#[derive(Debug, Clone)]
pub struct ActionHistoryEntry {
    pub timestamp: String,
    pub description: String,
    pub status: String,
    pub duration_ms: i64,
}

impl From<ActionResult> for ActionHistoryEntry {
    fn from(result: ActionResult) -> Self {
        let duration = (result.completed_at - result.started_at).num_milliseconds();
        Self {
            timestamp: result.started_at.format("%H:%M:%S").to_string(),
            description: result.message.clone(),
            status: if result.success { "✓".to_string() } else { "✗".to_string() },
            duration_ms: duration,
        }
    }
}

/// Metrics data
#[derive(Debug, Clone)]
pub struct MetricsData {
    pub cpu_usage: f32,
    pub memory_usage_mb: usize,
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub success_rate: f32,
}

impl Default for MetricsData {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage_mb: 0,
            active_tasks: 0,
            completed_tasks: 0,
            success_rate: 100.0,
        }
    }
}

impl TuiApp {
    /// Create a new TUI application
    pub fn new(refresh_rate_ms: u64) -> Self {
        Self {
            current_tab: 0,
            tabs: vec!["Dashboard", "Agents", "DAG", "Actions", "Help"],
            swarm_status: SwarmStatus {
                active_agents: 0,
                total_tasks: 0,
                completed_tasks: 0,
                failed_tasks: 0,
                topology: "mesh".to_string(),
            },
            agents: Vec::new(),
            action_history: Vec::new(),
            dag_nodes: Vec::new(),
            metrics: MetricsData::default(),
            selected: 0,
            show_help: false,
            last_update: Instant::now(),
            refresh_rate_ms,
        }
    }

    /// Run the TUI
    pub async fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Run event loop
        let result = self.event_loop(&mut terminal).await;

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        result
    }

    /// Main event loop
    async fn event_loop<B: ratatui::backend::Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        loop {
            terminal.draw(|f| self.render(f))?;

            // Handle events with timeout
            if event::poll(Duration::from_millis(self.refresh_rate_ms))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(()),
                        KeyCode::Tab => self.next_tab(),
                        KeyCode::BackTab => self.prev_tab(),
                        KeyCode::Up => self.select_prev(),
                        KeyCode::Down => self.select_next(),
                        KeyCode::Char('h') | KeyCode::Char('?') => self.toggle_help(),
                        KeyCode::Char('r') => self.refresh_data().await,
                        _ => {}
                    }
                }
            }

            // Auto-refresh data
            if self.last_update.elapsed() > Duration::from_secs(5) {
                self.refresh_data().await;
            }
        }
    }

    /// Render the TUI
    fn render(&mut self, f: &mut Frame) {
        let size = f.size();

        // Create main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Tabs
                Constraint::Min(0),     // Content
                Constraint::Length(1),  // Status bar
            ])
            .split(size);

        // Render tabs
        self.render_tabs(f, chunks[0]);

        // Render content based on current tab
        match self.current_tab {
            0 => self.render_dashboard(f, chunks[1]),
            1 => self.render_agents(f, chunks[1]),
            2 => self.render_dag(f, chunks[1]),
            3 => self.render_actions(f, chunks[1]),
            4 => self.render_help(f, chunks[1]),
            _ => {}
        }

        // Render status bar
        self.render_status_bar(f, chunks[2]);
    }

    /// Render tabs
    fn render_tabs(&self, f: &mut Frame, area: Rect) {
        let titles: Vec<Line> = self.tabs.iter().map(|t| Line::from(*t)).collect();
        let tabs = Tabs::new(titles)
            .block(Block::default().borders(Borders::ALL).title("Leviathan TUI"))
            .select(self.current_tab)
            .style(Style::default().fg(Color::Cyan))
            .highlight_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .bg(Color::DarkGray),
            );
        f.render_widget(tabs, area);
    }

    /// Render dashboard view
    fn render_dashboard(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),  // Swarm status
                Constraint::Length(7),  // Metrics
                Constraint::Min(0),     // Recent actions
            ])
            .split(area);

        // Swarm status
        let swarm_text = vec![
            Line::from(vec![
                Span::styled("Topology: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(&self.swarm_status.topology),
            ]),
            Line::from(vec![
                Span::styled("Active Agents: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::styled(
                    self.swarm_status.active_agents.to_string(),
                    Style::default().fg(Color::Green),
                ),
            ]),
            Line::from(vec![
                Span::styled("Tasks: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(format!(
                    "{} total, {} completed, {} failed",
                    self.swarm_status.total_tasks,
                    self.swarm_status.completed_tasks,
                    self.swarm_status.failed_tasks
                )),
            ]),
        ];

        let swarm_widget = Paragraph::new(swarm_text)
            .block(Block::default().borders(Borders::ALL).title("Swarm Status"))
            .wrap(Wrap { trim: true });
        f.render_widget(swarm_widget, chunks[0]);

        // Metrics
        let metrics_text = vec![
            Line::from(vec![
                Span::styled("CPU Usage: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::styled(
                    format!("{:.1}%", self.metrics.cpu_usage),
                    Style::default().fg(if self.metrics.cpu_usage > 80.0 { Color::Red } else { Color::Green }),
                ),
            ]),
            Line::from(vec![
                Span::styled("Memory: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(format!("{} MB", self.metrics.memory_usage_mb)),
            ]),
            Line::from(vec![
                Span::styled("Success Rate: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::styled(
                    format!("{:.1}%", self.metrics.success_rate),
                    Style::default().fg(if self.metrics.success_rate > 90.0 { Color::Green } else { Color::Yellow }),
                ),
            ]),
        ];

        let metrics_widget = Paragraph::new(metrics_text)
            .block(Block::default().borders(Borders::ALL).title("Metrics"))
            .wrap(Wrap { trim: true });
        f.render_widget(metrics_widget, chunks[1]);

        // Recent actions
        let actions: Vec<ListItem> = self
            .action_history
            .iter()
            .take(10)
            .map(|entry| {
                let status_color = if entry.status == "✓" { Color::Green } else { Color::Red };
                ListItem::new(Line::from(vec![
                    Span::styled(&entry.timestamp, Style::default().fg(Color::DarkGray)),
                    Span::raw(" "),
                    Span::styled(&entry.status, Style::default().fg(status_color)),
                    Span::raw(" "),
                    Span::raw(&entry.description),
                    Span::styled(
                        format!(" ({}ms)", entry.duration_ms),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]))
            })
            .collect();

        let actions_widget = List::new(actions)
            .block(Block::default().borders(Borders::ALL).title("Recent Actions"));
        f.render_widget(actions_widget, chunks[2]);
    }

    /// Render agents view
    fn render_agents(&self, f: &mut Frame, area: Rect) {
        let header = Row::new(vec!["ID", "Name", "Status", "Task", "Uptime", "Actions"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1);

        let rows = self.agents.iter().map(|agent| {
            let status_color = match agent.status.as_str() {
                "running" => Color::Green,
                "idle" => Color::Yellow,
                "error" => Color::Red,
                _ => Color::White,
            };

            Row::new(vec![
                agent.id.clone(),
                agent.name.clone(),
                agent.status.clone(),
                agent.task.clone().unwrap_or_else(|| "-".to_string()),
                format!("{}s", agent.uptime_secs),
                agent.actions_completed.to_string(),
            ])
            .style(Style::default().fg(status_color))
        });

        let table = Table::new(
            rows,
            [
                Constraint::Length(8),
                Constraint::Length(20),
                Constraint::Length(10),
                Constraint::Min(20),
                Constraint::Length(10),
                Constraint::Length(10),
            ],
        )
        .header(header)
        .block(Block::default().borders(Borders::ALL).title("Agents"))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

        f.render_widget(table, area);
    }

    /// Render DAG visualization
    fn render_dag(&self, f: &mut Frame, area: Rect) {
        let mut lines = vec![Line::from("DAG Visualization (ASCII Art)")];
        lines.push(Line::from(""));

        // Simple ASCII DAG rendering
        for node in &self.dag_nodes {
            let status_symbol = match node.status.as_str() {
                "completed" => "✓",
                "running" => "●",
                "pending" => "○",
                "failed" => "✗",
                _ => "?",
            };

            lines.push(Line::from(format!(
                "  {} [{}] {}",
                status_symbol, node.id, node.label
            )));

            for dep in &node.dependencies {
                lines.push(Line::from(format!("    └─→ {}", dep)));
            }
        }

        if self.dag_nodes.is_empty() {
            lines.push(Line::from("  No DAG nodes available"));
        }

        let dag_widget = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Task DAG"))
            .wrap(Wrap { trim: true });
        f.render_widget(dag_widget, area);
    }

    /// Render actions history view
    fn render_actions(&self, f: &mut Frame, area: Rect) {
        let header = Row::new(vec!["Time", "Status", "Description", "Duration"])
            .style(Style::default().add_modifier(Modifier::BOLD))
            .bottom_margin(1);

        let rows = self.action_history.iter().map(|entry| {
            let status_color = if entry.status == "✓" { Color::Green } else { Color::Red };
            Row::new(vec![
                entry.timestamp.clone(),
                entry.status.clone(),
                entry.description.clone(),
                format!("{}ms", entry.duration_ms),
            ])
            .style(Style::default().fg(status_color))
        });

        let table = Table::new(
            rows,
            [
                Constraint::Length(10),
                Constraint::Length(8),
                Constraint::Min(40),
                Constraint::Length(12),
            ],
        )
        .header(header)
        .block(Block::default().borders(Borders::ALL).title("Action History"))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

        f.render_widget(table, area);
    }

    /// Render help view
    fn render_help(&self, f: &mut Frame, area: Rect) {
        let help_text = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("Keyboard Shortcuts", Style::default().add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from("  q, Ctrl+C    - Quit"),
            Line::from("  Tab          - Next tab"),
            Line::from("  Shift+Tab    - Previous tab"),
            Line::from("  ↑/↓          - Navigate items"),
            Line::from("  h, ?         - Toggle help"),
            Line::from("  r            - Refresh data"),
            Line::from(""),
            Line::from(vec![
                Span::styled("Tabs", Style::default().add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from("  Dashboard    - Overview of swarm status and metrics"),
            Line::from("  Agents       - List of active agents"),
            Line::from("  DAG          - Task dependency visualization"),
            Line::from("  Actions      - Action execution history"),
            Line::from("  Help         - This help screen"),
        ];

        let help_widget = Paragraph::new(help_text)
            .block(Block::default().borders(Borders::ALL).title("Help"))
            .alignment(Alignment::Left);
        f.render_widget(help_widget, area);
    }

    /// Render status bar
    fn render_status_bar(&self, f: &mut Frame, area: Rect) {
        let status_text = format!(
            " Tab: {} | Items: {} | Press '?' for help",
            self.tabs[self.current_tab],
            match self.current_tab {
                1 => self.agents.len(),
                2 => self.dag_nodes.len(),
                3 => self.action_history.len(),
                _ => 0,
            }
        );

        let status = Paragraph::new(status_text)
            .style(Style::default().bg(Color::DarkGray).fg(Color::White));
        f.render_widget(status, area);
    }

    // Navigation methods

    fn next_tab(&mut self) {
        self.current_tab = (self.current_tab + 1) % self.tabs.len();
        self.selected = 0;
    }

    fn prev_tab(&mut self) {
        if self.current_tab > 0 {
            self.current_tab -= 1;
        } else {
            self.current_tab = self.tabs.len() - 1;
        }
        self.selected = 0;
    }

    fn select_next(&mut self) {
        let max_items = match self.current_tab {
            1 => self.agents.len(),
            2 => self.dag_nodes.len(),
            3 => self.action_history.len(),
            _ => 0,
        };

        if max_items > 0 {
            self.selected = (self.selected + 1) % max_items;
        }
    }

    fn select_prev(&mut self) {
        let max_items = match self.current_tab {
            1 => self.agents.len(),
            2 => self.dag_nodes.len(),
            3 => self.action_history.len(),
            _ => 0,
        };

        if max_items > 0 && self.selected > 0 {
            self.selected -= 1;
        } else if max_items > 0 {
            self.selected = max_items - 1;
        }
    }

    fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
        if self.show_help {
            self.current_tab = 4; // Help tab
        }
    }

    /// Refresh data from backend (stub)
    async fn refresh_data(&mut self) {
        self.last_update = Instant::now();

        // In a real implementation, this would fetch data from the backend
        // For now, we'll just update mock data
        self.metrics.cpu_usage = (self.metrics.cpu_usage + 5.0) % 100.0;
        self.metrics.memory_usage_mb = (self.metrics.memory_usage_mb + 10) % 1024;
    }

    /// Update swarm status
    pub fn update_swarm_status(&mut self, status: SwarmStatus) {
        self.swarm_status = status;
    }

    /// Update agents list
    pub fn update_agents(&mut self, agents: Vec<AgentStatus>) {
        self.agents = agents;
    }

    /// Add action to history
    pub fn add_action(&mut self, action: ActionHistoryEntry) {
        self.action_history.insert(0, action);
        if self.action_history.len() > 100 {
            self.action_history.truncate(100);
        }
    }

    /// Update DAG nodes
    pub fn update_dag(&mut self, nodes: Vec<DagNode>) {
        self.dag_nodes = nodes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tui_app_creation() {
        let app = TuiApp::new(250);
        assert_eq!(app.current_tab, 0);
        assert_eq!(app.tabs.len(), 5);
    }

    #[test]
    fn test_navigation() {
        let mut app = TuiApp::new(250);
        app.next_tab();
        assert_eq!(app.current_tab, 1);
        app.prev_tab();
        assert_eq!(app.current_tab, 0);
    }
}
