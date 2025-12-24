use egui::{Context, CentralPanel, TopBottomPanel};
use crate::components::StatusBar;
use crate::panels::*;
use crate::theme::LeviathanTheme;

/// Main Leviathan UI application
pub struct LeviathanApp {
    theme: LeviathanTheme,

    // Panels
    dashboard: DashboardPanel,
    swarm: SwarmPanel,
    audit: AuditPanel,
    terminal: TerminalPanel,
    config: ConfigPanel,

    // Window state
    show_dashboard: bool,
    show_swarm: bool,
    show_audit: bool,
    show_terminal: bool,
    show_config: bool,

    // Active view
    active_panel: ActivePanel,
}

#[derive(Debug, PartialEq)]
enum ActivePanel {
    Dashboard,
    Swarm,
    Audit,
    Terminal,
    Config,
}

impl Default for LeviathanApp {
    fn default() -> Self {
        let theme = LeviathanTheme::default();

        Self {
            theme: theme.clone(),
            dashboard: DashboardPanel::new(theme.clone()),
            swarm: SwarmPanel::new(theme.clone()),
            audit: AuditPanel::new(theme.clone()),
            terminal: TerminalPanel::new(theme.clone()),
            config: ConfigPanel::new(theme.clone()),
            show_dashboard: true,
            show_swarm: false,
            show_audit: false,
            show_terminal: false,
            show_config: false,
            active_panel: ActivePanel::Dashboard,
        }
    }
}

impl LeviathanApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }

    fn update_theme(&self, ctx: &Context) {
        let mut style = (*ctx.style()).clone();
        self.theme.apply_to_style(&mut style);
        ctx.set_style(style);
    }

    fn draw_menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("FILE", |ui| {
                if ui.button("New Session").clicked() {
                    // New session
                }
                if ui.button("Save State").clicked() {
                    // Save state
                }
                ui.separator();
                if ui.button("Exit").clicked() {
                    std::process::exit(0);
                }
            });

            ui.menu_button("VIEW", |ui| {
                if ui.button("Dashboard").clicked() {
                    self.active_panel = ActivePanel::Dashboard;
                }
                if ui.button("Swarm").clicked() {
                    self.active_panel = ActivePanel::Swarm;
                }
                if ui.button("Audit").clicked() {
                    self.active_panel = ActivePanel::Audit;
                }
                if ui.button("Terminal").clicked() {
                    self.active_panel = ActivePanel::Terminal;
                }
                if ui.button("Config").clicked() {
                    self.active_panel = ActivePanel::Config;
                }
            });

            ui.menu_button("TOOLS", |ui| {
                if ui.button("Spawn Agent").clicked() {
                    // Spawn agent dialog
                }
                if ui.button("Run Task").clicked() {
                    // Run task dialog
                }
                ui.separator();
                if ui.button("Benchmark").clicked() {
                    // Benchmark tool
                }
            });

            ui.menu_button("HELP", |ui| {
                if ui.button("Documentation").clicked() {
                    // Open docs
                }
                if ui.button("About").clicked() {
                    // About dialog
                }
            });
        });
    }

    fn draw_taskbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Start button
            if ui.button(
                egui::RichText::new("⚡ START")
                    .color(self.theme.accent_cyan)
                    .monospace()
            ).clicked() {
                // Show start menu
            }

            ui.separator();

            // Panel buttons
            if ui.selectable_label(
                self.active_panel == ActivePanel::Dashboard,
                "📊 DASHBOARD"
            ).clicked() {
                self.active_panel = ActivePanel::Dashboard;
            }

            if ui.selectable_label(
                self.active_panel == ActivePanel::Swarm,
                "🤖 SWARM"
            ).clicked() {
                self.active_panel = ActivePanel::Swarm;
            }

            if ui.selectable_label(
                self.active_panel == ActivePanel::Audit,
                "🔍 AUDIT"
            ).clicked() {
                self.active_panel = ActivePanel::Audit;
            }

            if ui.selectable_label(
                self.active_panel == ActivePanel::Terminal,
                "💻 TERMINAL"
            ).clicked() {
                self.active_panel = ActivePanel::Terminal;
            }

            if ui.selectable_label(
                self.active_panel == ActivePanel::Config,
                "⚙️ CONFIG"
            ).clicked() {
                self.active_panel = ActivePanel::Config;
            }

            // Right-aligned system tray
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Clock (only on native, WASM uses browser time)
                #[cfg(not(target_arch = "wasm32"))]
                ui.label(
                    egui::RichText::new(format!("⏰ {}", chrono::Local::now().format("%H:%M")))
                        .color(self.theme.text_primary)
                        .monospace()
                );

                #[cfg(target_arch = "wasm32")]
                ui.label(
                    egui::RichText::new("⏰ --:--")
                        .color(self.theme.text_primary)
                        .monospace()
                );

                ui.separator();

                ui.label(
                    egui::RichText::new("🟢 ONLINE")
                        .color(self.theme.success)
                        .monospace()
                );
            });
        });
    }
}

impl eframe::App for LeviathanApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // Apply custom theme
        self.update_theme(ctx);

        // Top menu bar
        TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            self.draw_menu_bar(ui);
        });

        // Bottom taskbar
        TopBottomPanel::bottom("taskbar")
            .min_height(32.0)
            .show(ctx, |ui| {
                self.draw_taskbar(ui);
            });

        // Bottom status bar
        TopBottomPanel::bottom("status_bar")
            .min_height(24.0)
            .show(ctx, |ui| {
                let sections = [
                    ("STATUS", "READY"),
                    ("AGENTS", "3/10"),
                    ("TASKS", "12"),
                    ("MODE", "STANDARD"),
                ];
                StatusBar::new(&sections, &self.theme).show(ui);
            });

        // Main content area
        CentralPanel::default().show(ctx, |ui| {
            // Add some padding
            ui.add_space(8.0);

            // Show active panel
            match self.active_panel {
                ActivePanel::Dashboard => self.dashboard.show(ui),
                ActivePanel::Swarm => self.swarm.show(ui),
                ActivePanel::Audit => self.audit.show(ui),
                ActivePanel::Terminal => self.terminal.show(ui),
                ActivePanel::Config => self.config.show(ui),
            }
        });

        // Request repaint for animations
        ctx.request_repaint();
    }
}
