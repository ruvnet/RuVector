//! Native unit tests for the wasm bindings' running state. Error paths that
//! construct a `JsError` cannot run off-wasm; those live in `tests/smoke.mjs`.

use super::*;

fn delta(progress: f64) -> StateDelta {
    StateDelta::new(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, progress)
}

#[test]
fn shrinking_window_drops_old_samples() {
    let mut c = AgenticClock::new();
    c.set_noise_floor(0.0);
    // Fill the default 8-tick window with regressing progress.
    for _ in 0..8 {
        c.tick(&delta(-1.0));
    }
    c.set_window(2);
    c.tick(&delta(1.0));
    c.tick(&delta(1.0));
    assert_eq!(c.recent_delta.len(), 2);
    assert_eq!(c.recent_progress.len(), 2);
    // Only the last two (positive-progress) ticks are in the window.
    assert!((c.ati() - 2.0 / 0.2).abs() < 1e-9, "ati = {}", c.ati());
}

#[test]
fn windowed_detector_history_is_bounded() {
    let mut d = WindowedDeltaClock::new(10, 4.0, 1e-6);
    for i in 0..1000 {
        d.push((i % 3) as f64);
    }
    assert!(d.history.len() <= 10, "history grew to {}", d.history.len());
}

#[test]
fn windowed_detector_z_unchanged_by_trimming() {
    // Same stream through a detector whose window covers everything must agree
    // with the trimmed one on the trailing-window z-score.
    let mut d = WindowedDeltaClock::new(5, 100.0, 1e-6);
    let mut last = 0.0;
    for i in 0..50 {
        last = d.push((i * 7 % 11) as f64);
    }
    let win: Vec<f64> = (44..49).map(|i| (i * 7 % 11) as f64).collect();
    let mean = win.iter().sum::<f64>() / 5.0;
    let std = (win.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 5.0).sqrt();
    let want = ((49 * 7 % 11) as f64 - mean).abs() / std;
    assert!((last - want).abs() < 1e-12, "{last} vs {want}");
}

#[test]
fn non_finite_deltas_do_not_poison_the_clock() {
    let mut c = AgenticClock::new();
    c.tick(&StateDelta::new(
        f64::NAN,
        f64::INFINITY,
        0.0,
        f64::NAN,
        f64::NEG_INFINITY,
        0.0,
        f64::NAN,
        f64::NAN,
    ));
    c.tick(&delta(0.5));
    assert!(c.cumulative_time().is_finite());
    assert!(c.cumulative_progress().is_finite());
    assert!(c.ati().is_finite());
}
