//! SGP4 satellite layer for the dashboard: TLE → TEME → geodetic → observer
//! az/el/range, feeding the same all-sky projection as aircraft.
//!
//! TEME → pseudo-ECEF uses the IAU-1982 GMST rotation (polar motion and
//! equation-of-equinoxes ignored); ECEF → geodetic uses Bowring's closed-form
//! method. Display-grade accuracy (dots on a dome), not ephemeris-grade.

// Lint suggests `as_chunks`, stable only since Rust 1.88 — past the declared
// `rust-version = "1.77"`. `unknown_lints` keeps older toolchains quiet.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use sky_monitor::coords::{geodetic_to_ecef, observer_frame};
use wasm_bindgen::prelude::*;

const WGS84_A_KM: f64 = 6378.137;
const WGS84_F: f64 = 1.0 / 298.257_223_563;

/// IAU-1982 Greenwich mean sidereal time for a Unix timestamp, radians.
fn gmst_rad(unix_s: f64) -> f64 {
    let d = unix_s / 86_400.0 - 10_957.5; // days since J2000.0 (2000-01-01 12:00 UTC)
    let deg = (280.460_618_37 + 360.985_647_366_29 * d) % 360.0;
    (if deg < 0.0 { deg + 360.0 } else { deg }).to_radians()
}

/// TEME position (km) at `unix_s` → geodetic `(lat_deg, lon_deg, alt_m)`.
fn teme_to_geodetic(pos_km: &[f64; 3], unix_s: f64) -> (f64, f64, f64) {
    // TEME → pseudo-ECEF: rotate by GMST around Z.
    let (s, c) = gmst_rad(unix_s).sin_cos();
    let x = c * pos_km[0] + s * pos_km[1];
    let y = -s * pos_km[0] + c * pos_km[1];
    let z = pos_km[2];
    // ECEF → geodetic (Bowring).
    let a = WGS84_A_KM;
    let b = a * (1.0 - WGS84_F);
    let e2 = WGS84_F * (2.0 - WGS84_F);
    let ep2 = (a * a - b * b) / (b * b);
    let p = x.hypot(y);
    let theta = (z * a).atan2(p * b);
    let lat = (z + ep2 * b * theta.sin().powi(3)).atan2(p - e2 * a * theta.cos().powi(3));
    let n = a / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    let alt_km = p / lat.cos() - n;
    (lat.to_degrees(), y.atan2(x).to_degrees(), alt_km * 1000.0)
}

const EARTH_R_KM: f64 = 6371.0;
/// Sun elevation below which the observer's sky counts as dark (civil
/// twilight) for naked-eye satellite visibility — matches astro.js.
const DARK_SUN_EL_DEG: f64 = -6.0;

/// Low-precision solar ECEF unit direction via the subsolar point
/// (truncated Meeus, mirrors ui/dashboard/astro.js `sunPosition`).
/// Good to a fraction of a degree — display / pass-flagging grade.
fn sun_dir_ecef(unix_s: f64) -> [f64; 3] {
    let d = unix_s / 86_400.0 - 10_957.5; // days since J2000.0
    let l = (280.460 + 0.985_647_4 * d).rem_euclid(360.0);
    let g = ((357.528 + 0.985_600_3 * d).rem_euclid(360.0)).to_radians();
    let lambda = (l + 1.915 * g.sin() + 0.020 * (2.0 * g).sin()).to_radians();
    let eps = (23.439 - 0.000_000_4 * d).to_radians();
    let ra = (eps.cos() * lambda.sin())
        .atan2(lambda.cos())
        .to_degrees()
        .rem_euclid(360.0);
    let dec = (eps.sin() * lambda.sin()).asin().to_degrees();
    let mut sub_lon = (ra - gmst_rad(unix_s).to_degrees()).rem_euclid(360.0);
    if sub_lon > 180.0 {
        sub_lon -= 360.0;
    }
    let e = geodetic_to_ecef(dec, sub_lon, 0.0);
    let n = (e.x * e.x + e.y * e.y + e.z * e.z).sqrt();
    [e.x / n, e.y / n, e.z / n]
}

/// Sun elevation (degrees) above the local geodetic horizon at a point.
fn sun_elevation_deg(lat_deg: f64, lon_deg: f64, dir: &[f64; 3]) -> f64 {
    let (sl, cl) = lat_deg.to_radians().sin_cos();
    let (so, co) = lon_deg.to_radians().sin_cos();
    (cl * co * dir[0] + cl * so * dir[1] + sl * dir[2])
        .asin()
        .to_degrees()
}

/// Cylinder-shadow illumination test (mirrors astro.js `satSunlit`).
fn sat_sunlit(lat_deg: f64, lon_deg: f64, alt_m: f64, dir: &[f64; 3]) -> bool {
    let e = geodetic_to_ecef(lat_deg, lon_deg, alt_m);
    let p = [e.x / 1000.0, e.y / 1000.0, e.z / 1000.0]; // km
    let dot = p[0] * dir[0] + p[1] * dir[1] + p[2] * dir[2];
    if dot > 0.0 {
        return true; // on the sun side of Earth
    }
    let o = [
        p[0] - dot * dir[0],
        p[1] - dot * dir[1],
        p[2] - dot * dir[2],
    ];
    (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() > EARTH_R_KM
}

struct Sat {
    name: String,
    epoch_unix: f64,
    constants: sgp4::Constants,
}

/// SGP4 propagator over a set of TLEs, projecting each satellite into a fixed
/// observer's sky (same §10 observer frame as aircraft).
#[wasm_bindgen]
pub struct SatPropagator {
    lat: f64,
    lon: f64,
    alt_m: f64,
    sats: Vec<Sat>,
}

#[wasm_bindgen]
impl SatPropagator {
    /// New propagator for the observer's geodetic position.
    #[wasm_bindgen(constructor)]
    pub fn new(lat: f64, lon: f64, alt_m: f64) -> SatPropagator {
        SatPropagator {
            lat,
            lon,
            alt_m,
            sats: Vec::new(),
        }
    }

    /// Add one TLE; returns `false` (and skips it) on parse/init failure.
    pub fn add_tle(&mut self, name: &str, line1: &str, line2: &str) -> bool {
        let Ok(elements) = sgp4::Elements::from_tle(
            Some(name.trim().to_string()),
            line1.as_bytes(),
            line2.as_bytes(),
        ) else {
            return false;
        };
        let Ok(constants) = sgp4::Constants::from_elements(&elements) else {
            return false;
        };
        let epoch_unix = elements.datetime.and_utc().timestamp_millis() as f64 / 1000.0;
        self.sats.push(Sat {
            name: name.trim().to_string(),
            epoch_unix,
            constants,
        });
        true
    }

    /// Number of loaded satellites.
    pub fn count(&self) -> usize {
        self.sats.len()
    }

    /// Name of satellite `i` (insertion order, as passed to `add_tle`).
    pub fn name(&self, i: usize) -> String {
        self.sats.get(i).map(|s| s.name.clone()).unwrap_or_default()
    }

    /// Propagate every satellite to Unix time `unix_s` and project it into
    /// the observer's sky. Returns a `Float64Array` of
    /// `[lat_deg, lon_deg, alt_m, azimuth_deg, elevation_deg, range_m]` per
    /// satellite (insertion order); a satellite that fails to propagate
    /// (e.g. decayed) yields six `NaN`s.
    pub fn positions(&self, unix_s: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.sats.len() * 6);
        for sat in &self.sats {
            let minutes = (unix_s - sat.epoch_unix) / 60.0;
            match sat.constants.propagate(sgp4::MinutesSinceEpoch(minutes)) {
                Ok(pred) => {
                    let (lat, lon, alt_m) = teme_to_geodetic(&pred.position, unix_s);
                    let f = observer_frame(self.lat, self.lon, self.alt_m, lat, lon, alt_m);
                    out.extend_from_slice(&[
                        lat,
                        lon,
                        alt_m,
                        f.azimuth_deg,
                        f.elevation_deg,
                        f.range_m,
                    ]);
                }
                Err(_) => out.extend_from_slice(&[f64::NAN; 6]),
            }
        }
        out
    }

    /// Predict horizon-to-horizon passes for every loaded satellite,
    /// stepping SGP4 from `start_unix` over `hours` in `step_s`-second
    /// samples (use ~30 s; rise/set instants are linearly interpolated
    /// between samples).
    ///
    /// Returns a `Float64Array` of 7-tuples, one per pass:
    /// `[sat_index, t_rise, t_culminate, t_set, max_elevation_deg,
    ///   culmination_azimuth_deg, visible]`. `visible` is `1.0` when the
    /// satellite is sunlit against a dark observer sky (sun below −6°) at
    /// any sampled point of the pass — the same naked-eye criterion the
    /// dashboard's astro.js applies to the live layer. A pass still in
    /// progress at the window end is truncated there; satellites that fail
    /// to propagate (e.g. decayed) simply yield no passes.
    pub fn predict_passes(&self, start_unix: f64, hours: f64, step_s: f64) -> Vec<f64> {
        struct Open {
            rise: f64,
            max_el: f64,
            az_culm: f64,
            t_culm: f64,
            visible: bool,
        }
        let step_s = if step_s > 0.0 { step_s } else { 30.0 };
        let steps = ((hours.max(0.0) * 3_600.0 / step_s).ceil() as usize).max(1);
        // The sun moves identically for every satellite: precompute its
        // direction and the observer's "dark sky" flag once per sample.
        let sun: Vec<([f64; 3], bool)> = (0..=steps)
            .map(|k| {
                let dir = sun_dir_ecef(start_unix + k as f64 * step_s);
                let dark = sun_elevation_deg(self.lat, self.lon, &dir) < DARK_SUN_EL_DEG;
                (dir, dark)
            })
            .collect();
        let mut out = Vec::new();
        for (si, sat) in self.sats.iter().enumerate() {
            let mut open: Option<Open> = None;
            let mut prev_el = f64::NEG_INFINITY;
            let mut prev_t = start_unix;
            // `sun` has exactly `steps + 1` samples, one per time step.
            for (k, (dir, dark)) in sun.iter().enumerate() {
                let t = start_unix + k as f64 * step_s;
                let minutes = (t - sat.epoch_unix) / 60.0;
                let geo = sat
                    .constants
                    .propagate(sgp4::MinutesSinceEpoch(minutes))
                    .ok()
                    .map(|p| teme_to_geodetic(&p.position, t));
                let (lat, lon, alt_m, az, el) = match geo {
                    Some((lat, lon, alt_m)) => {
                        let f = observer_frame(self.lat, self.lon, self.alt_m, lat, lon, alt_m);
                        (lat, lon, alt_m, f.azimuth_deg, f.elevation_deg)
                    }
                    None => (0.0, 0.0, 0.0, 0.0, f64::NEG_INFINITY),
                };
                if el > 0.0 {
                    let p = open.get_or_insert_with(|| Open {
                        // Interpolate the horizon crossing when the previous
                        // sample was a real below-horizon elevation.
                        rise: if prev_el.is_finite() && prev_el <= 0.0 {
                            prev_t + step_s * (-prev_el) / (el - prev_el)
                        } else {
                            t
                        },
                        max_el: f64::NEG_INFINITY,
                        az_culm: az,
                        t_culm: t,
                        visible: false,
                    });
                    if el > p.max_el {
                        p.max_el = el;
                        p.az_culm = az;
                        p.t_culm = t;
                    }
                    if *dark && sat_sunlit(lat, lon, alt_m, dir) {
                        p.visible = true;
                    }
                    if k == steps {
                        if let Some(p) = open.take() {
                            out.extend_from_slice(&[
                                si as f64,
                                p.rise,
                                p.t_culm,
                                t,
                                p.max_el,
                                p.az_culm,
                                if p.visible { 1.0 } else { 0.0 },
                            ]);
                        }
                    }
                } else if let Some(p) = open.take() {
                    let set = if el.is_finite() {
                        prev_t + step_s * prev_el / (prev_el - el)
                    } else {
                        prev_t
                    };
                    out.extend_from_slice(&[
                        si as f64,
                        p.rise,
                        p.t_culm,
                        set,
                        p.max_el,
                        p.az_culm,
                        if p.visible { 1.0 } else { 0.0 },
                    ]);
                }
                prev_el = el;
                prev_t = t;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real TLE from the CelesTrak `visual` group (fetched 2026-06-10).
    // Perigee ≈ 460 km, apogee ≈ 1250 km, inclination 30.36°.
    const L1: &str = "1 00694U 63047A   26161.07228530  .00001358  00000+0  15205-3 0  9994";
    const L2: &str = "2 00694  30.3551  39.1817 0545986 174.7629 185.8982 14.12503204144697";

    #[test]
    fn propagates_atlas_centaur_to_plausible_geodetic() {
        let mut sp = SatPropagator::new(43.4675, -79.6877, 100.0);
        assert!(sp.add_tle("ATLAS CENTAUR 2", L1, L2));
        // At the TLE's own epoch the orbit must respect its inclination and
        // LEO altitude band.
        let epoch_unix = sp.sats[0].epoch_unix;
        let out = sp.positions(epoch_unix);
        assert_eq!(out.len(), 6);
        let (lat, lon, alt_m) = (out[0], out[1], out[2]);
        assert!(lat.abs() <= 30.5, "lat {lat}");
        assert!((-180.0..=180.0).contains(&lon), "lon {lon}");
        assert!((300_000.0..=1_400_000.0).contains(&alt_m), "alt {alt_m}");
        assert!((-90.0..=90.0).contains(&out[4]), "el {}", out[4]);
        assert!(out[5] > 300_000.0, "range {}", out[5]);
    }

    #[test]
    fn rejects_garbage_tle() {
        let mut sp = SatPropagator::new(0.0, 0.0, 0.0);
        assert!(!sp.add_tle("junk", "not a tle", "also not"));
        assert_eq!(sp.count(), 0);
    }

    #[test]
    fn gmst_reference_value_at_j2000() {
        // 2000-01-01 12:00 UTC: GMST ≈ 280.4606°.
        let g = gmst_rad(946_728_000.0).to_degrees();
        assert!((g - 280.4606).abs() < 0.01, "gmst {g}");
    }
    #[test]
    fn sun_elevation_sane_at_toronto_noon_and_night() {
        // 2026-06-10 17:00 UTC ≈ solar noon in Toronto: high sun.
        let dir = sun_dir_ecef(1_781_110_800.0);
        let el = sun_elevation_deg(43.4675, -79.6877, &dir);
        assert!((60.0..80.0).contains(&el), "noon el {el}");
        // 2026-06-10 05:00 UTC ≈ 1 a.m. local: deep night.
        let dir = sun_dir_ecef(1_781_067_600.0);
        let el = sun_elevation_deg(43.4675, -79.6877, &dir);
        assert!(el < -10.0, "night el {el}");
    }

    #[test]
    fn predicts_ordered_passes_over_24h() {
        let mut sp = SatPropagator::new(43.4675, -79.6877, 100.0);
        assert!(sp.add_tle("ATLAS CENTAUR 2", L1, L2));
        let start = sp.sats[0].epoch_unix;
        let out = sp.predict_passes(start, 24.0, 30.0);
        assert_eq!(out.len() % 7, 0, "7 numbers per pass");
        assert!(!out.is_empty(), "LEO sat must pass at least once in 24 h");
        for p in out.chunks_exact(7) {
            assert_eq!(p[0], 0.0, "single sat index");
            assert!(
                p[1] <= p[2] && p[2] <= p[3],
                "rise {} culm {} set {}",
                p[1],
                p[2],
                p[3]
            );
            assert!(p[3] - p[1] < 3_600.0, "LEO pass under an hour");
            assert!(p[4] > 0.0 && p[4] <= 90.0, "max el {}", p[4]);
            assert!((0.0..360.0).contains(&p[5]), "az {}", p[5]);
            assert!(p[6] == 0.0 || p[6] == 1.0, "visible flag {}", p[6]);
        }
    }
}
