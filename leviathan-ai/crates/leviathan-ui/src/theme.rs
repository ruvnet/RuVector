use egui::{Color32, FontId, Rounding, Shadow, Stroke, Style, Visuals, epaint::Margin};

/// Windows 95 + Cyberpunk Corporate theme
#[derive(Clone)]
pub struct LeviathanTheme {
    // Win95 base colors with cyberpunk accents
    pub bg_primary: Color32,
    pub bg_secondary: Color32,
    pub bg_tertiary: Color32,
    pub accent_cyan: Color32,
    pub accent_magenta: Color32,
    pub accent_purple: Color32,
    pub text_primary: Color32,
    pub text_secondary: Color32,
    pub text_muted: Color32,
    pub border_light: Color32,
    pub border_dark: Color32,
    pub success: Color32,
    pub warning: Color32,
    pub error: Color32,
}

impl Default for LeviathanTheme {
    fn default() -> Self {
        Self {
            // Dark cyberpunk background palette
            bg_primary: Color32::from_rgb(26, 26, 46),      // #1a1a2e - Main dark
            bg_secondary: Color32::from_rgb(22, 33, 62),    // #16213e - Panels
            bg_tertiary: Color32::from_rgb(15, 15, 30),     // #0f0f1e - Deeper dark

            // Neon cyberpunk accents
            accent_cyan: Color32::from_rgb(0, 255, 245),    // #00fff5 - Primary accent
            accent_magenta: Color32::from_rgb(255, 0, 255), // #ff00ff - Secondary accent
            accent_purple: Color32::from_rgb(138, 43, 226), // #8a2be2 - Tertiary accent

            // Text colors
            text_primary: Color32::from_rgb(240, 240, 255),   // Bright white
            text_secondary: Color32::from_rgb(192, 192, 220), // Muted white
            text_muted: Color32::from_rgb(120, 120, 150),     // Dark gray

            // Win95-style 3D borders
            border_light: Color32::from_rgb(223, 223, 223),  // Light bevel
            border_dark: Color32::from_rgb(64, 64, 64),      // Dark bevel

            // Status colors
            success: Color32::from_rgb(0, 255, 65),    // Matrix green
            warning: Color32::from_rgb(255, 180, 0),   // Amber
            error: Color32::from_rgb(255, 40, 80),     // Neon red
        }
    }
}

impl LeviathanTheme {
    /// Apply theme to egui style
    pub fn apply_to_style(&self, style: &mut Style) {
        let visuals = &mut style.visuals;

        // Dark mode base
        visuals.dark_mode = true;

        // Window styling
        visuals.window_fill = self.bg_primary;
        visuals.window_stroke = Stroke::new(2.0, self.accent_cyan);
        visuals.window_shadow = Shadow {
            offset: [4.0, 4.0].into(),
            blur: 16.0,
            spread: 0.0,
            color: Color32::from_rgba_premultiplied(0, 255, 245, 40),
        };
        visuals.window_rounding = Rounding::ZERO; // Win95 sharp corners

        // Panel styling
        visuals.panel_fill = self.bg_secondary;

        // Widget defaults
        visuals.widgets.noninteractive.bg_fill = self.bg_secondary;
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, self.text_secondary);
        visuals.widgets.noninteractive.rounding = Rounding::ZERO;

        // Inactive widgets
        visuals.widgets.inactive.bg_fill = self.bg_secondary;
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, self.text_primary);
        visuals.widgets.inactive.rounding = Rounding::ZERO;

        // Hovered widgets - neon glow effect
        visuals.widgets.hovered.bg_fill = Color32::from_rgb(30, 40, 70);
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.5, self.accent_cyan);
        visuals.widgets.hovered.rounding = Rounding::ZERO;
        visuals.widgets.hovered.expansion = 1.0;

        // Active/clicked widgets
        visuals.widgets.active.bg_fill = Color32::from_rgb(15, 25, 50);
        visuals.widgets.active.fg_stroke = Stroke::new(2.0, self.accent_magenta);
        visuals.widgets.active.rounding = Rounding::ZERO;

        // Open menus
        visuals.widgets.open.bg_fill = self.bg_tertiary;
        visuals.widgets.open.fg_stroke = Stroke::new(1.5, self.accent_cyan);
        visuals.widgets.open.rounding = Rounding::ZERO;

        // Selection
        visuals.selection.bg_fill = Color32::from_rgba_premultiplied(0, 255, 245, 60);
        visuals.selection.stroke = Stroke::new(1.0, self.accent_cyan);

        // Hyperlinks
        visuals.hyperlink_color = self.accent_cyan;

        // Text cursor
        visuals.text_cursor.stroke = Stroke::new(2.0, self.accent_cyan);

        // Extreme background colors
        visuals.extreme_bg_color = self.bg_tertiary;
        visuals.faint_bg_color = self.bg_secondary;

        // Code background
        visuals.code_bg_color = Color32::from_rgb(10, 10, 20);

        // Window resizing
        visuals.resize_corner_size = 12.0;

        // Spacing
        style.spacing.item_spacing = [8.0, 6.0].into();
        style.spacing.button_padding = [8.0, 4.0].into();
        style.spacing.window_margin = Margin::same(8.0);

        // Text styles - monospace for corporate clean look
        style.text_styles.insert(
            egui::TextStyle::Body,
            FontId::monospace(14.0),
        );
        style.text_styles.insert(
            egui::TextStyle::Button,
            FontId::monospace(14.0),
        );
        style.text_styles.insert(
            egui::TextStyle::Heading,
            FontId::monospace(18.0),
        );
        style.text_styles.insert(
            egui::TextStyle::Monospace,
            FontId::monospace(12.0),
        );
    }

    /// Create a glow effect color
    pub fn glow_color(&self, base: Color32, intensity: f32) -> Color32 {
        Color32::from_rgba_premultiplied(
            base.r(),
            base.g(),
            base.b(),
            (intensity * 255.0) as u8,
        )
    }

    /// Get Win95-style 3D raised border colors
    pub fn raised_border(&self) -> (Color32, Color32) {
        (self.border_light, self.border_dark)
    }

    /// Get Win95-style 3D sunken border colors
    pub fn sunken_border(&self) -> (Color32, Color32) {
        (self.border_dark, self.border_light)
    }
}

/// Helper to draw Win95-style 3D borders
pub fn draw_win95_border(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    raised: bool,
    theme: &LeviathanTheme,
) {
    let painter = ui.painter();
    let (light, dark) = if raised {
        theme.raised_border()
    } else {
        theme.sunken_border()
    };

    // Top-left light border
    painter.line_segment(
        [rect.left_top(), rect.right_top()],
        Stroke::new(2.0, light),
    );
    painter.line_segment(
        [rect.left_top(), rect.left_bottom()],
        Stroke::new(2.0, light),
    );

    // Bottom-right dark border
    painter.line_segment(
        [rect.right_top(), rect.right_bottom()],
        Stroke::new(2.0, dark),
    );
    painter.line_segment(
        [rect.left_bottom(), rect.right_bottom()],
        Stroke::new(2.0, dark),
    );
}

/// Draw scanline overlay effect (subtle cyberpunk touch)
pub fn draw_scanlines(ui: &mut egui::Ui, rect: egui::Rect, intensity: f32) {
    let painter = ui.painter();
    let step = 4.0;
    let mut y = rect.top();

    while y < rect.bottom() {
        painter.line_segment(
            [[rect.left(), y].into(), [rect.right(), y].into()],
            Stroke::new(1.0, Color32::from_rgba_premultiplied(0, 255, 245, (intensity * 10.0) as u8)),
        );
        y += step;
    }
}
