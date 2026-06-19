//! Real-data MNIST loader for the optical-compression benchmark (ADR-260).
//!
//! ADR-260 §20.2 deliberately kept the *public demo* on a synthetic 4-class set
//! and flagged the synthetic accuracy numbers as a scientific-integrity risk.
//! This module supplies the honest counterpart: standard MNIST handwritten
//! digits (10 classes) so the learned-optical-frontend claim is measured on
//! recognized real data.
//!
//! The IDX files are **not** downloaded here. They are fetched + decompressed
//! once into a gitignored cache dir (see `tests/mnist_differential_bench.rs`
//! for the exact command) and this module only parses the raw, uncompressed
//! IDX bytes from disk. Keeping network/decompression out of the crate means
//! the loader has zero new dependencies and stays fully deterministic.
//!
//! Each 28x28 digit is box-averaged down to `cell x cell` then centered on a
//! power-of-two `grid x grid` field so it feeds `OpticalField::from_image`
//! unchanged. Default: 28x28 -> 20x20 detail centered in a 32x32 grid.

use crate::synthetic::Sample;
use photonlayer_core::field::InputImage;
use std::path::{Path, PathBuf};

/// MNIST has ten digit classes, 0-9.
pub const MNIST_CLASSES: usize = 10;

/// Standard IDX image magic (2 zero bytes, ndim marker 0x08, 3 dims).
const IDX_IMAGE_MAGIC: u32 = 0x0000_0803;
/// Standard IDX label magic (2 zero bytes, ndim marker 0x08, 1 dim).
const IDX_LABEL_MAGIC: u32 = 0x0000_0801;

/// Native MNIST side length.
const SRC_DIM: usize = 28;

/// Errors that can arise while loading MNIST from the cache dir.
#[derive(Debug)]
pub enum MnistError {
    /// A required IDX file was not present in the cache dir.
    Missing(PathBuf),
    /// An IDX file was present but malformed (bad magic, truncated, etc.).
    Parse(String),
}

impl std::fmt::Display for MnistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MnistError::Missing(p) => write!(
                f,
                "MNIST file not found: {} (fetch the IDX files into the cache dir first)",
                p.display()
            ),
            MnistError::Parse(m) => write!(f, "MNIST parse error: {m}"),
        }
    }
}

impl std::error::Error for MnistError {}

/// One MNIST split parsed from disk: row-major u8 pixels + labels.
pub struct RawMnist {
    pub images: Vec<u8>, // count * 28 * 28
    pub labels: Vec<u8>, // count
    pub count: usize,
}

fn read_u32_be(buf: &[u8], off: usize) -> Result<u32, MnistError> {
    buf.get(off..off + 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| MnistError::Parse(format!("truncated header at byte {off}")))
}

/// Parse a raw IDX image file (magic 0x00000803, 28x28 expected).
fn parse_idx_images(bytes: &[u8]) -> Result<(Vec<u8>, usize), MnistError> {
    let magic = read_u32_be(bytes, 0)?;
    if magic != IDX_IMAGE_MAGIC {
        return Err(MnistError::Parse(format!(
            "bad image magic {magic:#010x}, expected {IDX_IMAGE_MAGIC:#010x}"
        )));
    }
    let count = read_u32_be(bytes, 4)? as usize;
    let rows = read_u32_be(bytes, 8)? as usize;
    let cols = read_u32_be(bytes, 12)? as usize;
    if rows != SRC_DIM || cols != SRC_DIM {
        return Err(MnistError::Parse(format!(
            "unexpected image dims {rows}x{cols}, expected {SRC_DIM}x{SRC_DIM}"
        )));
    }
    let want = 16 + count * rows * cols;
    if bytes.len() < want {
        return Err(MnistError::Parse(format!(
            "image file truncated: have {} bytes, need {want}",
            bytes.len()
        )));
    }
    Ok((bytes[16..want].to_vec(), count))
}

/// Parse a raw IDX label file (magic 0x00000801).
fn parse_idx_labels(bytes: &[u8]) -> Result<(Vec<u8>, usize), MnistError> {
    let magic = read_u32_be(bytes, 0)?;
    if magic != IDX_LABEL_MAGIC {
        return Err(MnistError::Parse(format!(
            "bad label magic {magic:#010x}, expected {IDX_LABEL_MAGIC:#010x}"
        )));
    }
    let count = read_u32_be(bytes, 4)? as usize;
    let want = 8 + count;
    if bytes.len() < want {
        return Err(MnistError::Parse(format!(
            "label file truncated: have {} bytes, need {want}",
            bytes.len()
        )));
    }
    Ok((bytes[8..want].to_vec(), count))
}

fn load_split(images_path: &Path, labels_path: &Path) -> Result<RawMnist, MnistError> {
    if !images_path.exists() {
        return Err(MnistError::Missing(images_path.to_path_buf()));
    }
    if !labels_path.exists() {
        return Err(MnistError::Missing(labels_path.to_path_buf()));
    }
    let img_bytes = std::fs::read(images_path)
        .map_err(|e| MnistError::Parse(format!("read {}: {e}", images_path.display())))?;
    let lab_bytes = std::fs::read(labels_path)
        .map_err(|e| MnistError::Parse(format!("read {}: {e}", labels_path.display())))?;
    let (images, ic) = parse_idx_images(&img_bytes)?;
    let (labels, lc) = parse_idx_labels(&lab_bytes)?;
    if ic != lc {
        return Err(MnistError::Parse(format!(
            "image/label count mismatch: {ic} images vs {lc} labels"
        )));
    }
    Ok(RawMnist {
        images,
        labels,
        count: ic,
    })
}

/// Load the raw training split (`train-images/labels-idx*-ubyte`) from `dir`.
pub fn load_train(dir: &Path) -> Result<RawMnist, MnistError> {
    load_split(
        &dir.join("train-images-idx3-ubyte"),
        &dir.join("train-labels-idx1-ubyte"),
    )
}

/// Load the raw test split (`t10k-images/labels-idx*-ubyte`) from `dir`.
pub fn load_test(dir: &Path) -> Result<RawMnist, MnistError> {
    load_split(
        &dir.join("t10k-images-idx3-ubyte"),
        &dir.join("t10k-labels-idx1-ubyte"),
    )
}

/// Box-average a 28x28 u8 digit down to `cell x cell` normalized f32, then
/// center it on a `grid x grid` zero-padded field. `cell <= grid` and both are
/// independent of the source 28 so callers can pick any optical grid.
fn digit_to_image(src: &[u8], cell: usize, grid: usize) -> InputImage {
    debug_assert_eq!(src.len(), SRC_DIM * SRC_DIM);
    // 1. Downsample 28x28 -> cell x cell by area averaging.
    let mut small = vec![0.0f32; cell * cell];
    for oy in 0..cell {
        for ox in 0..cell {
            let x0 = ox * SRC_DIM / cell;
            let x1 = ((ox + 1) * SRC_DIM / cell).max(x0 + 1).min(SRC_DIM);
            let y0 = oy * SRC_DIM / cell;
            let y1 = ((oy + 1) * SRC_DIM / cell).max(y0 + 1).min(SRC_DIM);
            let mut acc = 0.0f32;
            let mut cnt = 0.0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    acc += src[y * SRC_DIM + x] as f32 / 255.0;
                    cnt += 1.0;
                }
            }
            small[oy * cell + ox] = if cnt > 0.0 { acc / cnt } else { 0.0 };
        }
    }
    // 2. Center on the power-of-two grid.
    let mut px = vec![0.0f32; grid * grid];
    let off = (grid - cell) / 2;
    for y in 0..cell {
        for x in 0..cell {
            px[(y + off) * grid + (x + off)] = small[y * cell + x];
        }
    }
    InputImage::from_norm_f32(grid, grid, px).expect("grid-sized image is well formed")
}

/// Take the first `per_class` samples of each digit class from a raw split,
/// converting each to a centered optical image. The scan order is the file's
/// natural order, so the result is deterministic for a fixed file + counts.
///
/// `cell` is the downsampled digit side; `grid` is the (power-of-two) optical
/// field side it is centered in. Caps total at `MNIST_CLASSES * per_class`.
pub fn subset(raw: &RawMnist, per_class: usize, cell: usize, grid: usize) -> Vec<Sample> {
    assert!(cell <= grid, "cell {cell} must be <= grid {grid}");
    let mut taken = [0usize; MNIST_CLASSES];
    let mut out = Vec::with_capacity(MNIST_CLASSES * per_class);
    for i in 0..raw.count {
        let label = raw.labels[i] as usize;
        if label >= MNIST_CLASSES || taken[label] >= per_class {
            continue;
        }
        let src = &raw.images[i * SRC_DIM * SRC_DIM..(i + 1) * SRC_DIM * SRC_DIM];
        out.push(Sample {
            image: digit_to_image(src, cell, grid),
            label,
        });
        taken[label] += 1;
        if taken.iter().all(|&t| t >= per_class) {
            break;
        }
    }
    out
}

/// Convenience: cache dir resolved relative to the bench crate
/// (`CARGO_MANIFEST_DIR/data/mnist`). Tests use this so the path is stable
/// regardless of the process working directory.
pub fn default_cache_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("mnist")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_magic() {
        let bytes = [0u8; 32];
        assert!(parse_idx_images(&bytes).is_err());
        assert!(parse_idx_labels(&bytes).is_err());
    }

    #[test]
    fn parses_synthetic_idx() {
        // One 28x28 image of all-127 + label 3.
        let mut img = Vec::new();
        img.extend_from_slice(&IDX_IMAGE_MAGIC.to_be_bytes());
        img.extend_from_slice(&1u32.to_be_bytes());
        img.extend_from_slice(&(SRC_DIM as u32).to_be_bytes());
        img.extend_from_slice(&(SRC_DIM as u32).to_be_bytes());
        img.extend(std::iter::repeat(127u8).take(SRC_DIM * SRC_DIM));
        let (pix, c) = parse_idx_images(&img).unwrap();
        assert_eq!(c, 1);
        assert_eq!(pix.len(), SRC_DIM * SRC_DIM);

        let mut lab = Vec::new();
        lab.extend_from_slice(&IDX_LABEL_MAGIC.to_be_bytes());
        lab.extend_from_slice(&1u32.to_be_bytes());
        lab.push(3);
        let (labels, lc) = parse_idx_labels(&lab).unwrap();
        assert_eq!(lc, 1);
        assert_eq!(labels[0], 3);
    }

    #[test]
    fn downsample_and_center_is_grid_sized() {
        let src = vec![255u8; SRC_DIM * SRC_DIM];
        let img = digit_to_image(&src, 20, 32);
        assert_eq!(img.width, 32);
        assert_eq!(img.height, 32);
        // Centered 20x20 of 1.0 inside a 32x32 zero field.
        let off = (32 - 20) / 2;
        assert!((img.pixels[off * 32 + off] - 1.0).abs() < 1e-6);
        assert_eq!(img.pixels[0], 0.0); // corner is padding
    }
}
