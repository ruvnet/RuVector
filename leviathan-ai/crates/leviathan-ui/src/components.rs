use egui::{Color32, Response, Sense, Stroke, Ui, Vec2, Widget};
use crate::theme::LeviathanTheme;

// Win95-style window is just a type alias since egui::Window handles everything
// The theme styling is applied globally via Style

/// Win95-style 3D button with beveled edges
pub struct Win95Button<'a> {
    text: &'a str,
    theme: &'a LeviathanTheme,
    enabled: bool,
}

impl<'a> Win95Button<'a> {
    pub fn new(text: &'a str, theme: &'a LeviathanTheme) -> Self {
        Self {
            text,
            theme,
            enabled: true,
        }
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

impl<'a> Widget for Win95Button<'a> {
    fn ui(self, ui: &mut Ui) -> Response {
        let desired_size = Vec2::new(
            self.text.len() as f32 * 8.0 + 16.0,
            24.0,
        );

        let (rect, response) = ui.allocate_exact_size(desired_size, Sense::click());

        if ui.is_rect_visible(rect) {
            // Determine button state
            let is_pressed = response.is_pointer_button_down_on();
            let is_hovered = response.hovered();

            // Background
            let bg_color = if !self.enabled {
                self.theme.bg_secondary
            } else if is_pressed {
                Color32::from_rgb(15, 25, 50)
            } else if is_hovered {
                Color32::from_rgb(30, 40, 70)
            } else {
                self.theme.bg_secondary
            };

            let painter = ui.painter();
            painter.rect_filled(rect, 0.0, bg_color);

            // 3D border effect (inline to avoid borrow issues)
            let (light, dark) = if !is_pressed {
                self.theme.raised_border()
            } else {
                self.theme.sunken_border()
            };

            painter.line_segment(
                [rect.left_top(), rect.right_top()],
                Stroke::new(2.0, light),
            );
            painter.line_segment(
                [rect.left_top(), rect.left_bottom()],
                Stroke::new(2.0, light),
            );
            painter.line_segment(
                [rect.right_top(), rect.right_bottom()],
                Stroke::new(2.0, dark),
            );
            painter.line_segment(
                [rect.left_bottom(), rect.right_bottom()],
                Stroke::new(2.0, dark),
            );

            // Neon glow on hover
            if is_hovered && self.enabled {
                painter.rect_stroke(
                    rect.shrink(1.0),
                    0.0,
                    Stroke::new(1.0, self.theme.glow_color(self.theme.accent_cyan, 0.6)),
                );
            }

            // Text
            let text_color = if !self.enabled {
                self.theme.text_muted
            } else if is_hovered {
                self.theme.accent_cyan
            } else {
                self.theme.text_primary
            };

            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                self.text,
                egui::FontId::monospace(14.0),
                text_color,
            );
        }

        response
    }
}

/// Win95-style sunken text input
pub struct Win95TextInput<'a> {
    text: &'a mut String,
    theme: &'a LeviathanTheme,
    hint: &'a str,
}

impl<'a> Win95TextInput<'a> {
    pub fn new(text: &'a mut String, theme: &'a LeviathanTheme) -> Self {
        Self {
            text,
            theme,
            hint: "",
        }
    }

    pub fn hint(mut self, hint: &'a str) -> Self {
        self.hint = hint;
        self
    }

    pub fn show(self, ui: &mut Ui) -> Response {
        let response = ui.add(
            egui::TextEdit::singleline(self.text)
                .hint_text(self.hint)
                .desired_width(f32::INFINITY)
        );

        // Draw sunken border around the text input (inline to avoid borrow issues)
        if ui.is_rect_visible(response.rect) {
            let painter = ui.painter();
            let rect = response.rect.shrink(1.0);
            let (dark, light) = self.theme.sunken_border();

            painter.line_segment(
                [rect.left_top(), rect.right_top()],
                Stroke::new(2.0, dark),
            );
            painter.line_segment(
                [rect.left_top(), rect.left_bottom()],
                Stroke::new(2.0, dark),
            );
            painter.line_segment(
                [rect.right_top(), rect.right_bottom()],
                Stroke::new(2.0, light),
            );
            painter.line_segment(
                [rect.left_bottom(), rect.right_bottom()],
                Stroke::new(2.0, light),
            );
        }

        response
    }
}

/// Win95-style list view
pub struct Win95ListView<'a> {
    items: &'a [String],
    selected: &'a mut Option<usize>,
    theme: &'a LeviathanTheme,
}

impl<'a> Win95ListView<'a> {
    pub fn new(
        items: &'a [String],
        selected: &'a mut Option<usize>,
        theme: &'a LeviathanTheme,
    ) -> Self {
        Self {
            items,
            selected,
            theme,
        }
    }

    pub fn show(self, ui: &mut Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (idx, item) in self.items.iter().enumerate() {
                    let is_selected = Some(idx) == *self.selected;

                    let response = ui.selectable_label(is_selected, item);

                    if response.clicked() {
                        *self.selected = Some(idx);
                    }

                    if response.hovered() {
                        response.on_hover_cursor(egui::CursorIcon::PointingHand);
                    }
                }
            });
    }
}

/// Win95-style progress bar with chunky segments
pub struct Win95ProgressBar {
    progress: f32, // 0.0 to 1.0
    theme: LeviathanTheme,
}

impl Win95ProgressBar {
    pub fn new(progress: f32) -> Self {
        Self {
            progress: progress.clamp(0.0, 1.0),
            theme: LeviathanTheme::default(),
        }
    }

    pub fn show(self, ui: &mut Ui) {
        let height = 20.0;
        let desired_size = Vec2::new(ui.available_width(), height);
        let (rect, _) = ui.allocate_exact_size(desired_size, Sense::hover());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();

            // Background
            painter.rect_filled(rect, 0.0, self.theme.bg_tertiary);

            // Draw sunken border inline
            let (dark, light) = self.theme.sunken_border();
            painter.line_segment(
                [rect.left_top(), rect.right_top()],
                Stroke::new(2.0, dark),
            );
            painter.line_segment(
                [rect.left_top(), rect.left_bottom()],
                Stroke::new(2.0, dark),
            );
            painter.line_segment(
                [rect.right_top(), rect.right_bottom()],
                Stroke::new(2.0, light),
            );
            painter.line_segment(
                [rect.left_bottom(), rect.right_bottom()],
                Stroke::new(2.0, light),
            );

            // Progress chunks (Win95 style)
            let chunk_width = 10.0;
            let chunk_spacing = 2.0;
            let total_chunk_width = chunk_width + chunk_spacing;
            let num_chunks = ((rect.width() - 4.0) / total_chunk_width) as usize;
            let filled_chunks = (num_chunks as f32 * self.progress) as usize;

            for i in 0..filled_chunks {
                let chunk_x = rect.left() + 2.0 + i as f32 * total_chunk_width;
                let chunk_rect = egui::Rect::from_min_size(
                    [chunk_x, rect.top() + 2.0].into(),
                    [chunk_width, height - 4.0].into(),
                );

                // Gradient from cyan to magenta
                let t = i as f32 / num_chunks.max(1) as f32;
                let color = Color32::from_rgb(
                    (t * 255.0) as u8,
                    0,
                    ((1.0 - t) * 255.0 + t * 245.0) as u8,
                );

                painter.rect_filled(chunk_rect, 0.0, color);

                // Glow effect
                painter.rect_stroke(
                    chunk_rect,
                    0.0,
                    Stroke::new(1.0, self.theme.glow_color(color, 0.5)),
                );
            }

            // Percentage text
            let text = format!("{}%", (self.progress * 100.0) as u32);
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                &text,
                egui::FontId::monospace(12.0),
                self.theme.text_primary,
            );
        }
    }
}

/// Status bar at the bottom with sections
pub struct StatusBar<'a> {
    sections: &'a [(&'a str, &'a str)], // (label, value) pairs
    theme: &'a LeviathanTheme,
}

impl<'a> StatusBar<'a> {
    pub fn new(sections: &'a [(&'a str, &'a str)], theme: &'a LeviathanTheme) -> Self {
        Self { sections, theme }
    }

    pub fn show(self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;

            for (i, (label, value)) in self.sections.iter().enumerate() {
                if i > 0 {
                    ui.separator();
                }

                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!("{}: ", label))
                        .color(self.theme.text_muted)
                        .monospace(),
                );
                ui.label(
                    egui::RichText::new(*value)
                        .color(self.theme.text_primary)
                        .monospace(),
                );
                ui.add_space(4.0);
            }
        });
    }
}

/// Terminal-style scrolling output
pub struct TerminalOutput<'a> {
    lines: &'a [String],
    theme: &'a LeviathanTheme,
    max_lines: usize,
}

impl<'a> TerminalOutput<'a> {
    pub fn new(lines: &'a [String], theme: &'a LeviathanTheme) -> Self {
        Self {
            lines,
            theme,
            max_lines: 1000,
        }
    }

    pub fn max_lines(mut self, max: usize) -> Self {
        self.max_lines = max;
        self
    }

    pub fn show(self, ui: &mut Ui) {
        let text_style = egui::TextStyle::Monospace;
        let row_height = ui.text_style_height(&text_style);
        let num_rows = self.lines.len().min(self.max_lines);

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show_rows(ui, row_height, num_rows, |ui, row_range| {
                for idx in row_range {
                    if idx < self.lines.len() {
                        let line = &self.lines[idx];

                        // Color code lines based on content
                        let color = if line.starts_with("[ERROR]") || line.contains("error") {
                            self.theme.error
                        } else if line.starts_with("[WARN]") || line.contains("warning") {
                            self.theme.warning
                        } else if line.starts_with("[SUCCESS]") || line.contains("success") {
                            self.theme.success
                        } else if line.starts_with(">") || line.starts_with("$") {
                            self.theme.accent_cyan
                        } else {
                            self.theme.text_primary
                        };

                        ui.label(egui::RichText::new(line).color(color).monospace());
                    }
                }
            });
    }
}
