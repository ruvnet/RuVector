//! Deterministic synthetic dataset of simple shape classes.
//!
//! Public demos avoid MNIST (ADR-260 §20.2); for fast, dependency-free,
//! reproducible benchmarks we render a handful of oriented/structured
//! patterns. Each class is a distinct geometric primitive so a tiny decoder
//! can separate them — letting us isolate the *optical* contribution.

use photonlayer_core::field::InputImage;
use photonlayer_core::rng::DeterministicRng;

/// A labeled image sample.
#[derive(Clone, Debug)]
pub struct Sample {
    pub image: InputImage,
    pub label: usize,
}

/// Number of built-in classes.
pub const NUM_CLASSES: usize = 4;

/// Human-readable class names.
pub fn class_names() -> [&'static str; NUM_CLASSES] {
    ["vbar", "hbar", "diag", "ring"]
}

fn render(class: usize, n: usize, rng: &mut DeterministicRng) -> Vec<f32> {
    let mut px = vec![0.0f32; n * n];
    let cx = n as f32 / 2.0;
    let cy = n as f32 / 2.0;
    // Small random offset so samples within a class differ.
    let jx = (rng.next_f32() - 0.5) * (n as f32 * 0.15);
    let jy = (rng.next_f32() - 0.5) * (n as f32 * 0.15);
    let thick = n as f32 * 0.18;
    for y in 0..n {
        for x in 0..n {
            let fx = x as f32 - cx - jx;
            let fy = y as f32 - cy - jy;
            let v = match class {
                0 => (fx.abs() < thick) as i32 as f32,        // vertical bar
                1 => (fy.abs() < thick) as i32 as f32,        // horizontal bar
                2 => ((fx - fy).abs() < thick) as i32 as f32, // diagonal
                _ => {
                    // ring
                    let r = (fx * fx + fy * fy).sqrt();
                    let target = n as f32 * 0.32;
                    ((r - target).abs() < thick * 0.6) as i32 as f32
                }
            };
            // Light additive texture keeps it from being trivially separable.
            px[y * n + x] = (v * 0.9 + rng.next_f32() * 0.1).clamp(0.0, 1.0);
        }
    }
    px
}

/// Generate `per_class` samples for each class on an `n x n` grid.
pub fn make_dataset(n: usize, per_class: usize, seed: u64) -> Vec<Sample> {
    let mut rng = DeterministicRng::new(seed);
    let mut out = Vec::with_capacity(NUM_CLASSES * per_class);
    for label in 0..NUM_CLASSES {
        for _ in 0..per_class {
            let px = render(label, n, &mut rng);
            let image = InputImage::from_norm_f32(n, n, px).unwrap();
            out.push(Sample { image, label });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_is_deterministic_and_balanced() {
        let a = make_dataset(16, 5, 1);
        let b = make_dataset(16, 5, 1);
        assert_eq!(a.len(), NUM_CLASSES * 5);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.label, y.label);
            assert_eq!(x.image.pixels, y.image.pixels);
        }
    }
}
