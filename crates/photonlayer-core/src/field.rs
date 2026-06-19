//! Optical field representation and image → field conversion (ADR-260 §9.1).

use crate::complex::Complex;
use crate::error::{PhotonError, Result};
use crate::fft::is_pow2;

/// A grayscale input image with intensities normalized to `[0, 1]`.
#[derive(Clone, Debug)]
pub struct InputImage {
    pub width: usize,
    pub height: usize,
    /// Row-major normalized intensities in `[0, 1]`.
    pub pixels: Vec<f32>,
}

impl InputImage {
    /// Build from raw `u8` grayscale pixels, normalizing by 255.
    pub fn from_gray_u8(width: usize, height: usize, data: &[u8]) -> Result<Self> {
        if data.len() != width * height {
            return Err(PhotonError::DimensionMismatch {
                expected: width * height,
                got: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels: data.iter().map(|&p| p as f32 / 255.0).collect(),
        })
    }

    /// Build from already-normalized `f32` pixels in `[0, 1]`.
    pub fn from_norm_f32(width: usize, height: usize, pixels: Vec<f32>) -> Result<Self> {
        if pixels.len() != width * height {
            return Err(PhotonError::DimensionMismatch {
                expected: width * height,
                got: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }
}

/// A complex scalar optical field on a power-of-two grid.
#[derive(Clone, Debug)]
pub struct OpticalField {
    pub width: usize,
    pub height: usize,
    pub data: Vec<Complex>,
}

impl OpticalField {
    /// Convert an input image into a field amplitude.
    ///
    /// Following ADR-260 §9.1: `amplitude = sqrt(intensity)`, `phase = 0`.
    /// The image is centered onto a (possibly larger) power-of-two grid so
    /// that diffraction has room to spread without wrap-around artifacts.
    pub fn from_image(img: &InputImage, grid_w: usize, grid_h: usize) -> Result<Self> {
        if !is_pow2(grid_w) {
            return Err(PhotonError::NotPowerOfTwo(grid_w));
        }
        if !is_pow2(grid_h) {
            return Err(PhotonError::NotPowerOfTwo(grid_h));
        }
        // Cap grid size before allocating, to block DoS / usize overflow from
        // an untrusted config (ADR-260 boundary hardening).
        if grid_w > crate::config::MAX_GRID_DIM || grid_h > crate::config::MAX_GRID_DIM {
            return Err(PhotonError::InvalidConfig(format!(
                "grid {grid_w}x{grid_h} exceeds MAX_GRID_DIM={}",
                crate::config::MAX_GRID_DIM
            )));
        }
        if img.width > grid_w || img.height > grid_h {
            return Err(PhotonError::InvalidConfig(format!(
                "image {}x{} larger than grid {}x{}",
                img.width, img.height, grid_w, grid_h
            )));
        }

        let mut data = vec![Complex::ZERO; grid_w * grid_h];
        let off_x = (grid_w - img.width) / 2;
        let off_y = (grid_h - img.height) / 2;
        for y in 0..img.height {
            for x in 0..img.width {
                let intensity = img.pixels[y * img.width + x].clamp(0.0, 1.0);
                let amp = intensity.sqrt();
                data[(y + off_y) * grid_w + (x + off_x)] = Complex::new(amp, 0.0);
            }
        }
        Ok(Self {
            width: grid_w,
            height: grid_h,
            data,
        })
    }

    /// Total optical power `sum |field|^2`. Conserved by lossless propagation.
    pub fn power(&self) -> f64 {
        self.data.iter().map(|c| c.norm_sqr() as f64).sum()
    }
}
