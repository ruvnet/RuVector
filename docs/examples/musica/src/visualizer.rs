//! Terminal-based audio oscilloscope and spectrum analyzer.
//!
//! Zero-dependency TUI visualization using ANSI escape codes and Unicode
//! block characters. Renders waveforms, frequency spectra, and separation
//! masks directly in the terminal.
//!
//! Inspired by terminal-oscilloscope (Nim) but implemented in pure Rust
//! with no external dependencies.

use crate::stft::{self, StftResult};

/// Display configuration for terminal visualization.
#[derive(Debug, Clone)]
pub struct DisplayConfig {
    /// Terminal width in characters.
    pub width: usize,
    /// Terminal height in lines for each pane.
    pub height: usize,
    /// Whether to use color (ANSI codes).
    pub color: bool,
    /// Whether to use Unicode block characters for higher resolution.
    pub unicode_blocks: bool,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 16,
            color: true,
            unicode_blocks: true,
        }
    }
}

// ── ANSI Colors ────────────────────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const MAGENTA: &str = "\x1b[35m";
const BLUE: &str = "\x1b[34m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";

/// Unicode block characters for sub-character vertical resolution.
const BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Render a waveform to the terminal.
pub fn render_waveform(
    signal: &[f64],
    label: &str,
    config: &DisplayConfig,
) -> String {
    let mut output = String::new();

    // Header
    if config.color {
        output.push_str(&format!("  {BOLD}{CYAN}┌─ {} ─{}{RESET}", label, "─".repeat(config.width.saturating_sub(label.len() + 6))));
    } else {
        output.push_str(&format!("  ┌─ {} ─{}", label, "─".repeat(config.width.saturating_sub(label.len() + 6))));
    }
    output.push('\n');

    if signal.is_empty() {
        output.push_str("  │ (empty signal)\n");
        output.push_str(&format!("  └{}┘\n", "─".repeat(config.width - 4)));
        return output;
    }

    // Downsample signal to fit width
    let display_width = config.width - 4; // margins
    let samples_per_col = (signal.len() as f64 / display_width as f64).max(1.0);

    // Find peak for normalization
    let peak = signal.iter().map(|x| x.abs()).fold(0.0f64, f64::max).max(1e-6);

    // Render each row (top to bottom = +1.0 to -1.0)
    let half_height = config.height / 2;
    for row in 0..config.height {
        output.push_str("  │");

        let y_top = 1.0 - (row as f64 / config.height as f64) * 2.0;
        let y_bot = 1.0 - ((row + 1) as f64 / config.height as f64) * 2.0;

        for col in 0..display_width {
            let start = (col as f64 * samples_per_col) as usize;
            let end = ((col + 1) as f64 * samples_per_col) as usize;
            let end = end.min(signal.len());

            if start >= signal.len() {
                output.push(' ');
                continue;
            }

            // Find min/max in this column
            let mut min_val = f64::MAX;
            let mut max_val = f64::MIN;
            for i in start..end {
                let v = signal[i] / peak;
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }

            // Does the waveform pass through this row?
            if max_val >= y_bot && min_val <= y_top {
                if config.color {
                    // Color by amplitude
                    let amp = max_val.abs().max(min_val.abs());
                    let color = if amp > 0.8 { RED } else if amp > 0.5 { YELLOW } else { GREEN };
                    if config.unicode_blocks {
                        // Calculate fill level within this cell
                        let fill = ((max_val - y_bot) / (y_top - y_bot)).clamp(0.0, 1.0);
                        let block_idx = (fill * 8.0) as usize;
                        let block_idx = block_idx.min(8);
                        output.push_str(&format!("{}{}{}", color, BLOCKS[block_idx], RESET));
                    } else {
                        output.push_str(&format!("{}█{}", color, RESET));
                    }
                } else {
                    output.push('█');
                }
            } else if row == half_height {
                // Zero line
                if config.color {
                    output.push_str(&format!("{DIM}─{RESET}"));
                } else {
                    output.push('─');
                }
            } else {
                output.push(' ');
            }
        }
        output.push_str("│\n");
    }

    // Footer with stats
    let rms: f64 = (signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64).sqrt();
    let footer = format!("peak={:.3} rms={:.3} samples={}", peak, rms, signal.len());
    if config.color {
        output.push_str(&format!("  {CYAN}└─ {} ─{}{RESET}\n", footer,
            "─".repeat(config.width.saturating_sub(footer.len() + 6))));
    } else {
        output.push_str(&format!("  └─ {} ─{}\n", footer,
            "─".repeat(config.width.saturating_sub(footer.len() + 6))));
    }

    output
}

/// Render a frequency spectrum (magnitude vs frequency bins).
pub fn render_spectrum(
    stft_result: &StftResult,
    frame: usize,
    label: &str,
    config: &DisplayConfig,
) -> String {
    let mut output = String::new();

    if config.color {
        output.push_str(&format!("  {BOLD}{MAGENTA}┌─ {} ─{}{RESET}\n", label,
            "─".repeat(config.width.saturating_sub(label.len() + 6))));
    } else {
        output.push_str(&format!("  ┌─ {} ─{}\n", label,
            "─".repeat(config.width.saturating_sub(label.len() + 6))));
    }

    let frame = frame.min(stft_result.num_frames.saturating_sub(1));
    let base = frame * stft_result.num_freq_bins;
    let num_bins = stft_result.num_freq_bins;

    // Get magnitudes for this frame
    let mags: Vec<f64> = (0..num_bins)
        .map(|f| {
            if base + f < stft_result.bins.len() {
                stft_result.bins[base + f].magnitude
            } else {
                0.0
            }
        })
        .collect();

    let peak_mag = mags.iter().cloned().fold(0.0f64, f64::max).max(1e-6);

    // Render spectrum as vertical bars
    let display_width = config.width - 4;
    let bins_per_col = (num_bins as f64 / display_width as f64).max(1.0);

    for row in 0..config.height {
        output.push_str("  │");
        let threshold = 1.0 - (row as f64 + 0.5) / config.height as f64;

        for col in 0..display_width {
            let start_bin = (col as f64 * bins_per_col) as usize;
            let end_bin = ((col + 1) as f64 * bins_per_col) as usize;
            let end_bin = end_bin.min(num_bins);

            let max_mag = (start_bin..end_bin)
                .map(|b| mags[b] / peak_mag)
                .fold(0.0f64, f64::max);

            if max_mag >= threshold {
                if config.color {
                    // Color by frequency region
                    let freq_ratio = col as f64 / display_width as f64;
                    let color = if freq_ratio < 0.15 { RED }
                        else if freq_ratio < 0.3 { YELLOW }
                        else if freq_ratio < 0.5 { GREEN }
                        else if freq_ratio < 0.7 { CYAN }
                        else { BLUE };

                    if config.unicode_blocks {
                        let fill = ((max_mag - threshold) / (1.0 / config.height as f64)).clamp(0.0, 1.0);
                        let idx = (fill * 8.0) as usize;
                        output.push_str(&format!("{}{}{}", color, BLOCKS[idx.min(8)], RESET));
                    } else {
                        output.push_str(&format!("{}█{}", color, RESET));
                    }
                } else {
                    output.push('█');
                }
            } else {
                output.push(' ');
            }
        }
        output.push_str("│\n");
    }

    // Frequency axis labels
    let nyquist = stft_result.sample_rate / 2.0;
    let footer = format!("0 Hz {:>width$} {:.0} Hz | frame {}/{}",
        "", nyquist, frame, stft_result.num_frames,
        width = display_width.saturating_sub(30));
    if config.color {
        output.push_str(&format!("  {MAGENTA}└─ {} ─{}{RESET}\n", footer,
            "─".repeat(config.width.saturating_sub(footer.len() + 6).min(40))));
    } else {
        output.push_str(&format!("  └─ {} ─{}\n", footer,
            "─".repeat(config.width.saturating_sub(footer.len() + 6).min(40))));
    }

    output
}

/// Render separation masks as a heatmap.
pub fn render_masks(
    masks: &[Vec<f64>],
    num_frames: usize,
    num_freq_bins: usize,
    label: &str,
    config: &DisplayConfig,
) -> String {
    let mut output = String::new();

    if config.color {
        output.push_str(&format!("  {BOLD}{YELLOW}┌─ {} ─{}{RESET}\n", label,
            "─".repeat(config.width.saturating_sub(label.len() + 6))));
    } else {
        output.push_str(&format!("  ┌─ {} ─{}\n", label,
            "─".repeat(config.width.saturating_sub(label.len() + 6))));
    }

    let display_width = config.width - 4;
    let display_height = config.height;
    let frames_per_col = (num_frames as f64 / display_width as f64).max(1.0);
    let bins_per_row = (num_freq_bins as f64 / display_height as f64).max(1.0);

    // Grayscale blocks: from empty to full
    let shading = [' ', '░', '▒', '▓', '█'];

    for row in 0..display_height {
        output.push_str("  │");
        // Map row to frequency bins (high freq at top)
        let freq_start = ((display_height - 1 - row) as f64 * bins_per_row) as usize;
        let freq_end = (((display_height - row) as f64) * bins_per_row) as usize;
        let freq_end = freq_end.min(num_freq_bins);

        for col in 0..display_width {
            let frame_start = (col as f64 * frames_per_col) as usize;
            let frame_end = ((col + 1) as f64 * frames_per_col) as usize;
            let frame_end = frame_end.min(num_frames);

            // Average mask value in this cell for source 0
            let mut sum = 0.0;
            let mut count = 0;
            for f in frame_start..frame_end {
                for k in freq_start..freq_end {
                    let idx = f * num_freq_bins + k;
                    if idx < masks[0].len() {
                        sum += masks[0][idx];
                        count += 1;
                    }
                }
            }
            let val = if count > 0 { sum / count as f64 } else { 0.5 };
            let shade_idx = (val * 4.0).clamp(0.0, 4.0) as usize;

            if config.color {
                let color = if val > 0.7 { GREEN } else if val > 0.3 { YELLOW } else { RED };
                output.push_str(&format!("{}{}{}", color, shading[shade_idx], RESET));
            } else {
                output.push(shading[shade_idx]);
            }
        }
        output.push_str("│\n");
    }

    let footer = format!("{} sources | {}×{} TF bins", masks.len(), num_frames, num_freq_bins);
    if config.color {
        output.push_str(&format!("  {YELLOW}└─ {} ─{}{RESET}\n", footer,
            "─".repeat(config.width.saturating_sub(footer.len() + 6))));
    } else {
        output.push_str(&format!("  └─ {} ─{}\n", footer,
            "─".repeat(config.width.saturating_sub(footer.len() + 6))));
    }

    output
}

/// Render a compact comparison: original mix vs separated sources.
pub fn render_separation_comparison(
    mixed: &[f64],
    sources: &[Vec<f64>],
    sample_rate: f64,
    config: &DisplayConfig,
) -> String {
    let mut output = String::new();
    let compact = DisplayConfig {
        height: config.height / 2,
        ..config.clone()
    };

    output.push_str(&render_waveform(mixed, "Mixed Signal", &compact));
    for (i, source) in sources.iter().enumerate() {
        let label = format!("Source {} (separated)", i);
        output.push_str(&render_waveform(source, &label, &compact));
    }

    // Add STFT spectrum of the mix at the middle frame
    let stft_result = stft::stft(mixed, 256, 128, sample_rate);
    let mid_frame = stft_result.num_frames / 2;
    output.push_str(&render_spectrum(&stft_result, mid_frame, "Spectrum (mid-frame)", &compact));

    output
}

/// Render a Lissajous (X-Y) display from stereo channels.
pub fn render_lissajous(
    left: &[f64],
    right: &[f64],
    label: &str,
    config: &DisplayConfig,
) -> String {
    let mut output = String::new();
    let size = config.height;

    if config.color {
        output.push_str(&format!("  {BOLD}{BLUE}┌─ {} ─{}{RESET}\n", label,
            "─".repeat(size * 2 + 2 - label.len().min(size * 2))));
    } else {
        output.push_str(&format!("  ┌─ {} ─{}\n", label,
            "─".repeat(size * 2 + 2 - label.len().min(size * 2))));
    }

    // 2D grid for Lissajous pattern
    let grid_size = size;
    let mut grid = vec![vec![0u32; grid_size * 2]; grid_size];

    let n = left.len().min(right.len());
    let peak_l = left.iter().map(|x| x.abs()).fold(0.0f64, f64::max).max(1e-6);
    let peak_r = right.iter().map(|x| x.abs()).fold(0.0f64, f64::max).max(1e-6);

    for i in 0..n {
        let x = ((left[i] / peak_l + 1.0) / 2.0 * (grid_size * 2 - 1) as f64) as usize;
        let y = ((right[i] / peak_r + 1.0) / 2.0 * (grid_size - 1) as f64) as usize;
        let x = x.min(grid_size * 2 - 1);
        let y = y.min(grid_size - 1);
        grid[grid_size - 1 - y][x] += 1;
    }

    let max_hits = grid.iter().flat_map(|r| r.iter()).cloned().max().unwrap_or(1).max(1);

    for row in &grid {
        output.push_str("  │");
        for &hits in row {
            if hits == 0 {
                output.push(' ');
            } else {
                let intensity = (hits as f64 / max_hits as f64 * 4.0) as usize;
                let ch = match intensity {
                    0 => '·',
                    1 => '░',
                    2 => '▒',
                    3 => '▓',
                    _ => '█',
                };
                if config.color {
                    let color = match intensity {
                        0..=1 => BLUE,
                        2 => CYAN,
                        3 => GREEN,
                        _ => YELLOW,
                    };
                    output.push_str(&format!("{}{}{}", color, ch, RESET));
                } else {
                    output.push(ch);
                }
            }
        }
        output.push_str("│\n");
    }

    let correlation: f64 = if n > 0 {
        let dot: f64 = left[..n].iter().zip(right[..n].iter()).map(|(l, r)| l * r).sum();
        dot / (peak_l * peak_r * n as f64)
    } else { 0.0 };
    let footer = format!("L/R correlation: {:.3}", correlation);
    output.push_str(&format!("  └─ {} ─{}\n", footer,
        "─".repeat((grid_size * 2).saturating_sub(footer.len() + 2))));

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_waveform() {
        let signal: Vec<f64> = (0..1000)
            .map(|i| (i as f64 * 0.02 * std::f64::consts::PI).sin())
            .collect();

        let config = DisplayConfig { width: 60, height: 8, color: false, unicode_blocks: false };
        let output = render_waveform(&signal, "Test Sine", &config);

        assert!(output.contains("Test Sine"));
        assert!(output.contains("peak="));
        assert!(output.contains("rms="));
        assert!(output.lines().count() > 5);
    }

    #[test]
    fn test_render_spectrum() {
        let signal: Vec<f64> = (0..2000)
            .map(|i| (i as f64 * 0.05 * std::f64::consts::PI).sin())
            .collect();
        let stft_result = stft::stft(&signal, 256, 128, 8000.0);

        let config = DisplayConfig { width: 60, height: 8, color: false, unicode_blocks: false };
        let output = render_spectrum(&stft_result, 0, "Test Spectrum", &config);

        assert!(output.contains("Test Spectrum"));
        assert!(output.contains("Hz"));
    }

    #[test]
    fn test_render_masks() {
        let mask0 = vec![0.8; 100];
        let mask1 = vec![0.2; 100];
        let masks = vec![mask0, mask1];

        let config = DisplayConfig { width: 40, height: 6, color: false, unicode_blocks: false };
        let output = render_masks(&masks, 10, 10, "Test Mask", &config);

        assert!(output.contains("Test Mask"));
        assert!(output.contains("2 sources"));
    }

    #[test]
    fn test_render_lissajous() {
        let left: Vec<f64> = (0..1000)
            .map(|i| (i as f64 * 0.02 * std::f64::consts::PI).sin())
            .collect();
        let right: Vec<f64> = (0..1000)
            .map(|i| (i as f64 * 0.03 * std::f64::consts::PI).sin())
            .collect();

        let config = DisplayConfig { width: 40, height: 12, color: false, unicode_blocks: false };
        let output = render_lissajous(&left, &right, "Lissajous", &config);

        assert!(output.contains("Lissajous"));
        assert!(output.contains("correlation"));
    }

    #[test]
    fn test_render_separation_comparison() {
        let mixed: Vec<f64> = (0..2000).map(|i| {
            let t = i as f64 / 8000.0;
            (t * 200.0 * std::f64::consts::PI * 2.0).sin()
                + 0.5 * (t * 1500.0 * std::f64::consts::PI * 2.0).sin()
        }).collect();
        let src1: Vec<f64> = (0..2000).map(|i| {
            (i as f64 / 8000.0 * 200.0 * std::f64::consts::PI * 2.0).sin()
        }).collect();
        let src2: Vec<f64> = (0..2000).map(|i| {
            0.5 * (i as f64 / 8000.0 * 1500.0 * std::f64::consts::PI * 2.0).sin()
        }).collect();

        let config = DisplayConfig { width: 60, height: 10, color: false, unicode_blocks: false };
        let output = render_separation_comparison(&mixed, &[src1, src2], 8000.0, &config);

        assert!(output.contains("Mixed Signal"));
        assert!(output.contains("Source 0"));
        assert!(output.contains("Source 1"));
        assert!(output.contains("Spectrum"));
    }

    #[test]
    fn test_empty_signal() {
        let config = DisplayConfig::default();
        let output = render_waveform(&[], "Empty", &config);
        assert!(output.contains("empty signal"));
    }
}
