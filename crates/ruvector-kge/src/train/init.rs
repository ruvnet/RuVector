//! Optional re-initialisation of the tables at the start of
//! [`Trainer::fit`](super::Trainer::fit) (`TrainConfig::init`). Lives here,
//! not in `tables.rs`, so table construction is unchanged: the default
//! [`Init::Keep`] trains whatever rows the caller built (continual / snapshot
//! training never re-inits).

use crate::data::Rng;
use crate::Tables;
use serde::{Deserialize, Serialize};

/// How `fit` initialises the tables before the first epoch.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Init {
    /// Train the rows as given (backwards-compatible default).
    #[default]
    Keep,
    /// Re-draw every row uniform in `±sqrt(6 / (2·dims))` (the
    /// [`Tables::new`] bound) from the run seed.
    Xavier,
    /// Re-draw every value as `N(0, 1) · scale` from the run seed — the
    /// ComplEx-N3 recipe uses `scale = 1e-3` (kbc `init_size`).
    Normal {
        #[serde(default = "normal_scale_default")]
        scale: f32,
    },
}

fn normal_scale_default() -> f32 {
    1e-3
}

impl Init {
    pub(crate) fn validate(&self) -> crate::Result<()> {
        if let Init::Normal { scale } = self {
            if !(scale.is_finite() && *scale > 0.0) {
                return Err(crate::KgeError::Invalid(
                    "init.scale must be finite and > 0".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Apply `init` to `tables`, deterministically in `seed`.
pub fn apply_init(tables: &mut Tables, init: Init, seed: u64) {
    let dims = tables.dims() as f32;
    match init {
        Init::Keep => {}
        Init::Xavier => {
            let bound = (6.0 / (dims + dims)).sqrt();
            let mut rng = Rng::seeded(seed ^ 0x5851_F42D_4C95_7F2D);
            let mut draw = || (2.0 * unit_open(&mut rng) as f32 - 1.0) * bound;
            fill(tables, &mut draw);
        }
        Init::Normal { scale } => {
            let mut rng = Rng::seeded(seed ^ 0x2545_F491_4F6C_DD1D);
            let mut draw = || standard_normal(&mut rng) as f32 * scale;
            fill(tables, &mut draw);
        }
    }
}

fn fill(tables: &mut Tables, draw: &mut impl FnMut() -> f32) {
    for x in tables.entities_raw_mut() {
        *x = draw();
    }
    for x in tables.relations_raw_mut() {
        *x = draw();
    }
}

/// Uniform in `(0, 1]` (never 0, so `ln` is finite).
fn unit_open(rng: &mut Rng) -> f64 {
    ((rng.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
}

/// One standard normal draw (Box–Muller, cosine branch only: one draw per
/// call keeps the stream position independent of table size parity).
fn standard_normal(rng: &mut Rng) -> f64 {
    let u1 = unit_open(rng);
    let u2 = unit_open(rng);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_init_moments_and_determinism() {
        let mut a = Tables::new(400, 10, 32, 1);
        let mut b = Tables::new(400, 10, 32, 2); // different prior contents
        apply_init(&mut a, Init::Normal { scale: 1e-3 }, 9);
        apply_init(&mut b, Init::Normal { scale: 1e-3 }, 9);
        assert_eq!(a, b, "init overwrites every value from the seed alone");
        let v = a.entities_raw();
        let n = v.len() as f64;
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 1e-4, "mean {mean}");
        let sd = var.sqrt();
        assert!((sd - 1e-3).abs() < 5e-5, "sd {sd}");
    }

    #[test]
    fn keep_is_identity_and_xavier_bounded() {
        let orig = Tables::new(20, 4, 8, 3);
        let mut t = orig.clone();
        apply_init(&mut t, Init::Keep, 5);
        assert_eq!(t, orig);
        apply_init(&mut t, Init::Xavier, 5);
        assert_ne!(t, orig);
        let bound = (6.0f32 / 16.0).sqrt();
        assert!(t.entities_raw().iter().all(|x| x.abs() <= bound));
        assert!(Init::Normal { scale: 0.0 }.validate().is_err());
        assert!(Init::Normal { scale: f32::NAN }.validate().is_err());
    }
}
