//! # sky-monitor-wasm — browser projection engine for the SkyGraph dashboard
//!
//! ADR-199 presentation plane ("dashboard first"): the heavy stores
//! (RuVector `VectorDB`, SkyGraph `GraphDB`) stay native behind the
//! `appliance` feature of `sky-monitor`; this crate wraps the **pure** subset
//! (`coords`, `anomaly`, `adsb`) with `wasm-bindgen` so the Canvas dashboard
//! can do exact §10 projection math and §15 anomaly scoring in the browser.
//!
//! Exposed API:
//!
//! * [`SkyProjector`] — WGS-84 → observer-relative az/el/range/bearing, single
//!   and batched (`Float64Array` in/out, for fast trail rendering), plus the
//!   polar "fisheye" all-sky screen mapping ([`screen::polar_screen_xy`]).
//! * [`AnomalyScorer`] — baseline + scoring over JSON track summaries, reusing
//!   the core `anomaly` module (`BaselineStats::from_summaries` /
//!   `score_summary`) so browser scores match the native pipeline exactly.
//! * [`embed::embed_track`] / [`embed::novelty`] — §13 32-dim track
//!   embeddings from live points and the §15 vector-novelty score (mirrors
//!   the native indexer calibration), for the browser novelty store.
//! * [`SatPropagator`] — SGP4 satellite propagation from TLEs
//!   ([`sat`]: TEME → GMST-rotated ECEF → geodetic → observer az/el/range).
//! * [`parse_dump1090_json`] — the core dump1090 `aircraft.json` parser, for
//!   live feeds proxied into the browser.

// Lint suggests `as_chunks`, stable only since Rust 1.88 — past the declared
// `rust-version = "1.77"`. `unknown_lints` keeps older toolchains quiet.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use sky_monitor::anomaly::{score_summary, BaselineStats, Interpretation, TrackSummary};
use sky_monitor::config::AnomalyConfig;
use sky_monitor::coords::observer_frame;
use wasm_bindgen::prelude::*;

pub mod embed;
pub mod sat;
pub mod screen;

pub use sat::SatPropagator;

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// `{x, y, visible}` result of the all-sky screen mapping.
#[derive(serde::Serialize)]
struct ScreenPos {
    x: f64,
    y: f64,
    visible: bool,
}

/// `{score, band, reasons}` result of [`AnomalyScorer::score`].
#[derive(serde::Serialize)]
struct ScoreResult {
    score: f64,
    band: String,
    reasons: Vec<String>,
}

/// Fixed observer projecting WGS-84 targets into its local sky
/// (ADR-199 §10: geodetic → ECEF → ENU → azimuth/elevation/range/bearing).
#[wasm_bindgen]
pub struct SkyProjector {
    lat: f64,
    lon: f64,
    alt_m: f64,
}

#[wasm_bindgen]
impl SkyProjector {
    /// New projector at the observer's geodetic position.
    #[wasm_bindgen(constructor)]
    pub fn new(lat: f64, lon: f64, alt_m: f64) -> SkyProjector {
        SkyProjector { lat, lon, alt_m }
    }

    /// Project one target; returns `{range_m, azimuth_deg, elevation_deg,
    /// bearing_deg}`.
    pub fn project(&self, lat: f64, lon: f64, alt_m: f64) -> Result<JsValue, JsValue> {
        let frame = observer_frame(self.lat, self.lon, self.alt_m, lat, lon, alt_m);
        serde_wasm_bindgen::to_value(&frame).map_err(js_err)
    }

    /// Batched projection for trail rendering: input is a `Float64Array` of
    /// `[lat, lon, alt_m]` triplets; output is a `Float64Array` of
    /// `[azimuth_deg, elevation_deg, range_m, bearing_deg]` quadruplets, one
    /// per input triplet (a trailing partial triplet is ignored).
    pub fn project_batch(&self, coords: &[f64]) -> Vec<f64> {
        let n = coords.len() / 3;
        let mut out = Vec::with_capacity(n * 4);
        for c in coords.chunks_exact(3) {
            let f = observer_frame(self.lat, self.lon, self.alt_m, c[0], c[1], c[2]);
            out.extend_from_slice(&[f.azimuth_deg, f.elevation_deg, f.range_m, f.bearing_deg]);
        }
        out
    }

    /// Map an az/el direction onto a `width`×`height` canvas using the polar
    /// "fisheye" all-sky projection: zenith (el = 90°) at the canvas centre,
    /// horizon (el = 0°) on the inscribed-circle edge, azimuth 0° = North =
    /// straight up. Returns `{x, y, visible}` (`visible` = above horizon).
    pub fn screen_position(
        &self,
        azimuth_deg: f64,
        elevation_deg: f64,
        width: f64,
        height: f64,
    ) -> Result<JsValue, JsValue> {
        let (x, y, visible) = screen::polar_screen_xy(azimuth_deg, elevation_deg, width, height);
        serde_wasm_bindgen::to_value(&ScreenPos { x, y, visible }).map_err(js_err)
    }
}

/// §15 anomaly scorer over track summaries, sharing the exact native scoring
/// path (`BaselineStats::from_summaries` + `score_summary`).
///
/// `Default` gives the ADR-199 §15 weights (`AnomalyConfig::default()`) and
/// an empty baseline.
#[derive(Default)]
#[wasm_bindgen]
pub struct AnomalyScorer {
    cfg: AnomalyConfig,
    baseline: BaselineStats,
}

#[wasm_bindgen]
impl AnomalyScorer {
    /// New scorer with the ADR-199 §15 default weights and an empty baseline.
    #[wasm_bindgen(constructor)]
    pub fn new() -> AnomalyScorer {
        AnomalyScorer::default()
    }

    /// Build the baseline from an array of track summaries:
    /// `[{icao24, callsign, mean_alt_m, dominant_heading_deg, start_hour,
    ///    mean_signal_dbfs, min_range_m, max_elevation_deg}, ...]`.
    /// Returns the number of baseline tracks ingested.
    pub fn baseline_from(&mut self, tracks_json: JsValue) -> Result<usize, JsValue> {
        let summaries: Vec<TrackSummary> =
            serde_wasm_bindgen::from_value(tracks_json).map_err(js_err)?;
        self.baseline = BaselineStats::from_summaries(&summaries);
        Ok(summaries.len())
    }

    /// Score one track summary against the baseline; `novelty` in `[0, 1]`
    /// (vector novelty from the native indexer, or 0 when unavailable).
    /// Returns `{score, band, reasons}`.
    pub fn score(&self, track_json: JsValue, novelty: f64) -> Result<JsValue, JsValue> {
        let summary: TrackSummary = serde_wasm_bindgen::from_value(track_json).map_err(js_err)?;
        // No second sensor modality in the browser: cross_sensor = 0.
        let report = score_summary(&self.cfg, &summary, &self.baseline, novelty, 0.0);
        serde_wasm_bindgen::to_value(&ScoreResult {
            score: report.score,
            band: report.band.to_string(),
            reasons: report.reasons,
        })
        .map_err(js_err)
    }

    /// Number of tracks in the current baseline.
    pub fn baseline_len(&self) -> usize {
        self.baseline.n_tracks
    }
}

/// Parse a dump1090-style `aircraft.json` payload (live RTL-SDR feed) into an
/// array of aircraft state objects, using the same core parser as the native
/// pipeline. Entries without a position fix are skipped.
#[wasm_bindgen]
pub fn parse_dump1090_json(json: &str) -> Result<JsValue, JsValue> {
    let states = sky_monitor::parse_dump1090(json).map_err(js_err)?;
    serde_wasm_bindgen::to_value(&states).map_err(js_err)
}

/// ADR-199 §15 interpretation band for a composite score:
/// `normal | mildly unusual | interesting | strong anomaly | rare`.
#[wasm_bindgen]
pub fn band_for(score: f64) -> String {
    Interpretation::band(score).to_string()
}

/// Crate version (for the dashboard footer / cache busting).
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
