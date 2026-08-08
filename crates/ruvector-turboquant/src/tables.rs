//! Precomputed Lloyd-Max tables for the 4-bit (16-level) quantizer.
//!
//! After the randomized rotation and per-vector standardization by
//! `α = ‖v‖₂/√D`, each coordinate is approximately `N(0, 1)`, so a single
//! analytic table serves every dimension — no training pass, online ingest.
//! Levels are the 16 optimal N(0,1) reconstruction values (Max, *Quantizing
//! for Minimum Distortion*, IRE 1960), identical to ADR-254's `BitWidth::Four`
//! so `ruvector-turbovec` remains a cross-crate determinism oracle.

/// The 16 Lloyd-Max reconstruction levels for N(0,1), ascending.
pub const LEVELS_F32: [f32; 16] = [
    -2.732_6, -2.069_0, -1.618_0, -1.256_2, -0.942_4, -0.656_8, -0.388_1, -0.128_4, 0.128_4,
    0.388_1, 0.656_8, 0.942_4, 1.256_2, 1.618_0, 2.069_0, 2.732_6,
];

/// Largest |level| — the int8 grid scales against this.
pub const MAX_LEVEL: f32 = 2.732_6;

/// Int8 level grid: `LEVELS_I8[j] = round(LEVELS_F32[j] * 127 / MAX_LEVEL)`.
/// Exactly 16 entries — fits a single `pshufb` / `vqtbl1q_s8` shuffle register
/// for the SIMD kernels. Symmetric by construction.
pub const LEVELS_I8: [i8; 16] = [
    -127, -96, -75, -58, -44, -31, -18, -6, 6, 18, 31, 44, 58, 75, 96, 127,
];

/// f32 value of one int8 level unit: `MAX_LEVEL / 127`.
pub const I8_UNIT: f32 = MAX_LEVEL / 127.0;

/// Decision boundaries: midpoints between adjacent levels (15 values).
/// `code(z) = number of boundaries below z`, which is exactly the Lloyd-Max
/// nearest-level rule.
pub const BOUNDARIES: [f32; 15] = [
    -2.400_8, -1.843_5, -1.437_1, -1.099_3, -0.799_6, -0.522_45, -0.258_25, 0.0, 0.258_25,
    0.522_45, 0.799_6, 1.099_3, 1.437_1, 1.843_5, 2.400_8,
];

/// Quantize a standardized coordinate to its 4-bit code (0..=15) by branchless
/// boundary counting — identical results to nearest-level scan.
#[inline]
pub fn quantize_coord(z: f32) -> u8 {
    // Binary search unrolled over the 15 boundaries: 4 comparisons.
    let mut code = 0usize;
    code += 8 * usize::from(z >= BOUNDARIES[7]);
    code += 4 * usize::from(z >= BOUNDARIES[code + 3]);
    code += 2 * usize::from(z >= BOUNDARIES[code + 1]);
    code += usize::from(z >= BOUNDARIES[code]);
    code as u8
}

/// Reconstruction level for a code.
#[inline]
pub fn level(code: u8) -> f32 {
    LEVELS_F32[(code & 0x0F) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries_are_midpoints() {
        for i in 0..15 {
            let mid = (LEVELS_F32[i] + LEVELS_F32[i + 1]) / 2.0;
            assert!(
                (BOUNDARIES[i] - mid).abs() < 5e-4,
                "boundary {i}: {} vs midpoint {mid}",
                BOUNDARIES[i]
            );
        }
    }

    #[test]
    fn quantize_matches_nearest_level_scan() {
        let mut z = -4.0f32;
        while z <= 4.0 {
            let fast = quantize_coord(z);
            let mut best = 0u8;
            let mut best_d = f32::INFINITY;
            for (j, &l) in LEVELS_F32.iter().enumerate() {
                let d = (z - l).abs();
                if d < best_d {
                    best_d = d;
                    best = j as u8;
                }
            }
            assert_eq!(fast, best, "z = {z}");
            z += 0.003;
        }
    }

    #[test]
    fn i8_grid_matches_rounding() {
        for (j, &l) in LEVELS_F32.iter().enumerate() {
            let expect = (l * 127.0 / MAX_LEVEL).round() as i8;
            assert_eq!(LEVELS_I8[j], expect, "level {j}");
        }
    }

    #[test]
    fn codes_cover_extremes() {
        assert_eq!(quantize_coord(-10.0), 0);
        assert_eq!(quantize_coord(10.0), 15);
        assert_eq!(quantize_coord(0.0), 8); // 0.0 >= boundary[7]=0.0 ⇒ first positive level
    }
}
