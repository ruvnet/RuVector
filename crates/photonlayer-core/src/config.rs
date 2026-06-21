//! Optical configuration types (ADR-260 §10).
//!
//! These are the physically meaningful knobs of a simulated optical frontend.
//! All lengths are stored in the unit named by their field suffix so that
//! receipts and serialized configs are self-describing.

use serde::{Deserialize, Serialize};

/// Scalar-diffraction propagation model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropagationMode {
    /// Near-field Fresnel transfer function.
    Fresnel,
    /// Far-field Fraunhofer (single FFT, intensity is the power spectrum).
    Fraunhofer,
    /// Angular-spectrum method — highest fidelity, valid across regimes.
    AngularSpectrum,
}

/// Sensor / detector post-processing applied after intensity capture.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// Photon shot-noise strength (mean photons at unit intensity). `0` disables.
    pub shot_noise_photons: f32,
    /// Additive Gaussian read-noise standard deviation (in intensity units).
    pub read_noise_std: f32,
    /// Quantization levels (e.g. 256 for 8-bit). `0` disables quantization.
    pub quantization_levels: u32,
    /// Spatial binning factor `b`: each `b*b` block is averaged into one pixel.
    /// `1` disables binning. Used to model lower-resolution sensors.
    pub binning: usize,
    /// Saturation clip in intensity units (after noise). `0` disables clipping.
    pub saturation: f32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            shot_noise_photons: 0.0,
            read_noise_std: 0.0,
            quantization_levels: 0,
            binning: 1,
            saturation: 0.0,
        }
    }
}

/// Full optical system configuration. Every field participates in the
/// determinism invariant (ADR-260 §21): identical configs + inputs + seed
/// must yield identical output hashes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpticalConfig {
    pub width: usize,
    pub height: usize,
    pub wavelength_nm: f32,
    pub propagation_mm: f32,
    pub pixel_pitch_um: f32,
    pub propagation: PropagationMode,
    pub detector: DetectorConfig,
    pub seed: u64,
}

impl OpticalConfig {
    /// A small green-light far-field default suitable for the barcode demo.
    pub fn demo(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            wavelength_nm: 532.0,
            propagation_mm: 10.0,
            pixel_pitch_um: 8.0,
            propagation: PropagationMode::AngularSpectrum,
            detector: DetectorConfig::default(),
            seed: 0xC0FFEE,
        }
    }

    /// Wavelength in metres.
    #[inline]
    pub fn wavelength_m(&self) -> f32 {
        self.wavelength_nm * 1e-9
    }

    /// Pixel pitch in metres.
    #[inline]
    pub fn pixel_pitch_m(&self) -> f32 {
        self.pixel_pitch_um * 1e-6
    }

    /// Propagation distance in metres.
    #[inline]
    pub fn distance_m(&self) -> f32 {
        self.propagation_mm * 1e-3
    }

    /// Validate an (untrusted) config before it drives any allocation or FFT.
    ///
    /// This is the security choke point (ADR-260 boundary hardening): a
    /// malicious `config_json` from the browser/WASM path could otherwise
    /// request absurd grid dimensions — causing an allocation DoS or, on
    /// 32-bit `wasm32` (`usize == u32`), a `width * height` overflow. We cap
    /// dimensions, require power-of-two grids, and reject non-finite or
    /// non-physical optical parameters.
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::PhotonError;
        use crate::fft::is_pow2;

        if !is_pow2(self.width) || self.width > MAX_GRID_DIM {
            return Err(PhotonError::InvalidConfig(format!(
                "width {} must be a power of two in 1..={MAX_GRID_DIM}",
                self.width
            )));
        }
        if !is_pow2(self.height) || self.height > MAX_GRID_DIM {
            return Err(PhotonError::InvalidConfig(format!(
                "height {} must be a power of two in 1..={MAX_GRID_DIM}",
                self.height
            )));
        }
        if !(self.wavelength_nm.is_finite() && self.wavelength_nm > 0.0) {
            return Err(PhotonError::InvalidConfig(
                "wavelength_nm must be finite and > 0".into(),
            ));
        }
        if !(self.pixel_pitch_um.is_finite() && self.pixel_pitch_um > 0.0) {
            return Err(PhotonError::InvalidConfig(
                "pixel_pitch_um must be finite and > 0".into(),
            ));
        }
        if !self.propagation_mm.is_finite() {
            return Err(PhotonError::InvalidConfig(
                "propagation_mm must be finite".into(),
            ));
        }
        let d = &self.detector;
        if !(d.shot_noise_photons.is_finite()
            && d.read_noise_std.is_finite()
            && d.saturation.is_finite())
        {
            return Err(PhotonError::InvalidConfig(
                "detector noise/saturation must be finite".into(),
            ));
        }
        if d.binning == 0 {
            return Err(PhotonError::InvalidConfig(
                "detector.binning must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

/// Maximum grid side length accepted from an untrusted config. 4096*4096 fits
/// comfortably in a 32-bit `usize`, so no dimension product can overflow.
pub const MAX_GRID_DIM: usize = 4096;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_config_is_valid() {
        assert!(OpticalConfig::demo(32, 32).validate().is_ok());
    }

    #[test]
    fn rejects_oversized_and_non_pow2_and_nonfinite() {
        let mut c = OpticalConfig::demo(32, 32);
        c.width = 8192; // > MAX_GRID_DIM
        assert!(c.validate().is_err());

        let mut c = OpticalConfig::demo(32, 32);
        c.height = 30; // not power of two
        assert!(c.validate().is_err());

        let mut c = OpticalConfig::demo(32, 32);
        c.wavelength_nm = f32::NAN;
        assert!(c.validate().is_err());

        let mut c = OpticalConfig::demo(32, 32);
        c.detector.binning = 0;
        assert!(c.validate().is_err());
    }
}
