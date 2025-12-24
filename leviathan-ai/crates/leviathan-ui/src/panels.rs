use egui::{Color32, Ui, Widget};
use crate::components::{Win95Button, Win95ProgressBar, Win95TextInput, TerminalOutput, Win95ListView};
use crate::theme::LeviathanTheme;

/// Dashboard panel with system overview and metrics
pub struct DashboardPanel {
    theme: LeviathanTheme,
    metrics: DashboardMetrics,
}

#[derive(Default)]
pub struct DashboardMetrics {
    pub active_agents: usize,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub uptime: String,
}

impl DashboardPanel {
    pub fn new(theme: LeviathanTheme) -> Self {
        Self {
            theme,
            metrics: DashboardMetrics::default(),
        }
    }

    pub fn update_metrics(&mut self, metrics: DashboardMetrics) {
        self.metrics = metrics;
    }

    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading(
            egui::RichText::new("LEVIATHAN CONTROL DASHBOARD")
                .color(self.theme.accent_cyan)
                .monospace()
        );

        ui.separator();
        ui.add_space(8.0);

        // System metrics grid
        egui::Grid::new("metrics_grid")
            .num_columns(2)
            .spacing([40.0, 12.0])
            .show(ui, |ui| {
                // Row 1: Agents and Tasks
                self.metric_card(ui, "ACTIVE AGENTS", &self.metrics.active_agents.to_string(), self.theme.accent_cyan);
                self.metric_card(ui, "TOTAL TASKS", &self.metrics.total_tasks.to_string(), self.theme.text_primary);
                ui.end_row();

                // Row 2: Task completion
                self.metric_card(ui, "COMPLETED", &self.metrics.completed_tasks.to_string(), self.theme.success);
                self.metric_card(ui, "UPTIME", &self.metrics.uptime, self.theme.text_secondary);
                ui.end_row();
            });

        ui.add_space(16.0);

        // Resource usage
        ui.label(
            egui::RichText::new("SYSTEM RESOURCES")
                .color(self.theme.text_secondary)
                .monospace()
        );
        ui.add_space(4.0);

        ui.label(egui::RichText::new(format!("CPU: {:.1}%", self.metrics.cpu_usage * 100.0)).monospace());
        Win95ProgressBar::new(self.metrics.cpu_usage).show(ui);

        ui.add_space(8.0);

        ui.label(egui::RichText::new(format!("MEMORY: {:.1}%", self.metrics.memory_usage * 100.0)).monospace());
        Win95ProgressBar::new(self.metrics.memory_usage).show(ui);
    }

    fn metric_card(&self, ui: &mut Ui, label: &str, value: &str, color: Color32) {
        egui::Frame::none()
            .fill(self.theme.bg_tertiary)
            .stroke(egui::Stroke::new(1.0, self.theme.border_dark))
            .inner_margin(12.0)
            .show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new(label)
                            .color(self.theme.text_muted)
                            .small()
                            .monospace()
                    );
                    ui.label(
                        egui::RichText::new(value)
                            .color(color)
                            .heading()
                            .monospace()
                    );
                });
            });
    }
}

/// Swarm panel for agent visualization
pub struct SwarmPanel {
    theme: LeviathanTheme,
    agents: Vec<AgentInfo>,
    selected_agent: Option<usize>,
}

#[derive(Clone)]
pub struct AgentInfo {
    pub id: String,
    pub agent_type: String,
    pub status: AgentStatus,
    pub current_task: Option<String>,
}

#[derive(Clone, PartialEq, Debug)]
pub enum AgentStatus {
    Idle,
    Active,
    Busy,
    Error,
}

impl SwarmPanel {
    pub fn new(theme: LeviathanTheme) -> Self {
        Self {
            theme,
            agents: Vec::new(),
            selected_agent: None,
        }
    }

    pub fn update_agents(&mut self, agents: Vec<AgentInfo>) {
        self.agents = agents;
    }

    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading(
            egui::RichText::new("AGENT SWARM")
                .color(self.theme.accent_magenta)
                .monospace()
        );

        ui.separator();
        ui.add_space(8.0);

        if self.agents.is_empty() {
            ui.label(
                egui::RichText::new("No active agents")
                    .color(self.theme.text_muted)
                    .monospace()
            );
            return;
        }

        // Split view: list on left, details on right
        ui.columns(2, |columns| {
            // Left column: agent list
            columns[0].group(|ui| {
                ui.label(egui::RichText::new("ACTIVE AGENTS").monospace().strong());
                ui.separator();

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (idx, agent) in self.agents.iter().enumerate() {
                            let is_selected = Some(idx) == self.selected_agent;

                            let status_color = match agent.status {
                                AgentStatus::Idle => self.theme.text_muted,
                                AgentStatus::Active => self.theme.success,
                                AgentStatus::Busy => self.theme.warning,
                                AgentStatus::Error => self.theme.error,
                            };

                            let response = ui.selectable_label(
                                is_selected,
                                egui::RichText::new(format!("● {} [{}]", agent.id, agent.agent_type))
                                    .color(status_color)
                                    .monospace()
                            );

                            if response.clicked() {
                                self.selected_agent = Some(idx);
                            }
                        }
                    });
            });

            // Right column: agent details
            columns[1].group(|ui| {
                ui.label(egui::RichText::new("AGENT DETAILS").monospace().strong());
                ui.separator();

                if let Some(idx) = self.selected_agent {
                    if let Some(agent) = self.agents.get(idx) {
                        ui.add_space(8.0);

                        ui.label(egui::RichText::new("ID:").color(self.theme.text_muted).monospace());
                        ui.label(egui::RichText::new(&agent.id).color(self.theme.accent_cyan).monospace());
                        ui.add_space(4.0);

                        ui.label(egui::RichText::new("Type:").color(self.theme.text_muted).monospace());
                        ui.label(egui::RichText::new(&agent.agent_type).color(self.theme.text_primary).monospace());
                        ui.add_space(4.0);

                        ui.label(egui::RichText::new("Status:").color(self.theme.text_muted).monospace());
                        let status_text = format!("{:?}", agent.status);
                        let status_color = match agent.status {
                            AgentStatus::Idle => self.theme.text_muted,
                            AgentStatus::Active => self.theme.success,
                            AgentStatus::Busy => self.theme.warning,
                            AgentStatus::Error => self.theme.error,
                        };
                        ui.label(egui::RichText::new(status_text).color(status_color).monospace());
                        ui.add_space(4.0);

                        if let Some(task) = &agent.current_task {
                            ui.label(egui::RichText::new("Current Task:").color(self.theme.text_muted).monospace());
                            ui.label(egui::RichText::new(task).color(self.theme.text_primary).monospace());
                        }
                    }
                }
            });
        });
    }
}

/// Audit panel for DAG exploration
pub struct AuditPanel {
    theme: LeviathanTheme,
    dag_entries: Vec<String>,
    selected_entry: Option<usize>,
}

impl AuditPanel {
    pub fn new(theme: LeviathanTheme) -> Self {
        Self {
            theme,
            dag_entries: Vec::new(),
            selected_entry: None,
        }
    }

    pub fn update_dag(&mut self, entries: Vec<String>) {
        self.dag_entries = entries;
    }

    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading(
            egui::RichText::new("AUDIT DAG EXPLORER")
                .color(self.theme.accent_purple)
                .monospace()
        );

        ui.separator();
        ui.add_space(8.0);

        Win95ListView::new(&self.dag_entries, &mut self.selected_entry, &self.theme).show(ui);

        if let Some(idx) = self.selected_entry {
            if let Some(entry) = self.dag_entries.get(idx) {
                ui.add_space(8.0);
                ui.separator();
                ui.label(
                    egui::RichText::new("ENTRY DETAILS")
                        .color(self.theme.text_secondary)
                        .monospace()
                );
                ui.label(egui::RichText::new(entry).monospace());
            }
        }
    }
}

/// Terminal panel for command input/output
pub struct TerminalPanel {
    theme: LeviathanTheme,
    output_lines: Vec<String>,
    input_buffer: String,
}

impl TerminalPanel {
    pub fn new(theme: LeviathanTheme) -> Self {
        Self {
            theme,
            output_lines: vec![
                "[SYSTEM] Leviathan Terminal v1.0".to_string(),
                "[SYSTEM] Type 'help' for available commands".to_string(),
                ">".to_string(),
            ],
            input_buffer: String::new(),
        }
    }

    pub fn add_output(&mut self, line: String) {
        self.output_lines.push(line);
    }

    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading(
            egui::RichText::new("TERMINAL")
                .color(self.theme.success)
                .monospace()
        );

        ui.separator();
        ui.add_space(8.0);

        // Output area
        egui::Frame::none()
            .fill(Color32::from_rgb(0, 0, 0))
            .stroke(egui::Stroke::new(2.0, self.theme.border_dark))
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.set_min_height(300.0);
                TerminalOutput::new(&self.output_lines, &self.theme).show(ui);
            });

        ui.add_space(8.0);

        // Input area
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("$")
                    .color(self.theme.accent_cyan)
                    .monospace()
            );

            let response = Win95TextInput::new(&mut self.input_buffer, &self.theme)
                .hint("Enter command...")
                .show(ui);

            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                if !self.input_buffer.is_empty() {
                    self.output_lines.push(format!("> {}", self.input_buffer));
                    // Process command here
                    self.process_command(&self.input_buffer.clone());
                    self.input_buffer.clear();
                    response.request_focus();
                }
            }

            if Win95Button::new("EXECUTE", &self.theme).ui(ui).clicked() {
                if !self.input_buffer.is_empty() {
                    self.output_lines.push(format!("> {}", self.input_buffer));
                    self.process_command(&self.input_buffer.clone());
                    self.input_buffer.clear();
                }
            }
        });
    }

    fn process_command(&mut self, cmd: &str) {
        match cmd.trim() {
            "help" => {
                self.output_lines.push("[HELP] Available commands:".to_string());
                self.output_lines.push("  status - Show system status".to_string());
                self.output_lines.push("  agents - List active agents".to_string());
                self.output_lines.push("  clear  - Clear terminal".to_string());
                self.output_lines.push("  help   - Show this help".to_string());
            }
            "clear" => {
                self.output_lines.clear();
                self.output_lines.push("[SYSTEM] Terminal cleared".to_string());
            }
            "status" => {
                self.output_lines.push("[INFO] System operational".to_string());
            }
            "agents" => {
                self.output_lines.push("[INFO] Agent list not implemented yet".to_string());
            }
            _ => {
                self.output_lines.push(format!("[ERROR] Unknown command: {}", cmd));
            }
        }
    }
}

/// Config panel with Win95 property sheets
pub struct ConfigPanel {
    theme: LeviathanTheme,
    config: SystemConfig,
}

#[derive(Clone)]
pub struct SystemConfig {
    pub max_agents: String,
    pub log_level: String,
    pub auto_scale: bool,
    pub enable_metrics: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            max_agents: "10".to_string(),
            log_level: "INFO".to_string(),
            auto_scale: true,
            enable_metrics: true,
        }
    }
}

impl ConfigPanel {
    pub fn new(theme: LeviathanTheme) -> Self {
        Self {
            theme,
            config: SystemConfig::default(),
        }
    }

    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading(
            egui::RichText::new("SYSTEM CONFIGURATION")
                .color(self.theme.text_primary)
                .monospace()
        );

        ui.separator();
        ui.add_space(8.0);

        egui::Grid::new("config_grid")
            .num_columns(2)
            .spacing([20.0, 12.0])
            .show(ui, |ui| {
                // Max agents
                ui.label(egui::RichText::new("Max Agents:").monospace());
                Win95TextInput::new(&mut self.config.max_agents, &self.theme).show(ui);
                ui.end_row();

                // Log level
                ui.label(egui::RichText::new("Log Level:").monospace());
                Win95TextInput::new(&mut self.config.log_level, &self.theme).show(ui);
                ui.end_row();

                // Auto scale
                ui.label(egui::RichText::new("Auto Scale:").monospace());
                ui.checkbox(&mut self.config.auto_scale, "");
                ui.end_row();

                // Enable metrics
                ui.label(egui::RichText::new("Enable Metrics:").monospace());
                ui.checkbox(&mut self.config.enable_metrics, "");
                ui.end_row();
            });

        ui.add_space(16.0);

        ui.horizontal(|ui| {
            if Win95Button::new("APPLY", &self.theme).ui(ui).clicked() {
                // Apply config
            }

            if Win95Button::new("RESET", &self.theme).ui(ui).clicked() {
                self.config = SystemConfig::default();
            }
        });
    }
}
