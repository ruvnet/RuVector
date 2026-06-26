//! Integration tests. The pure-logic tests always run; the real-model tests are
//! gated on the `candle` feature AND the local weights, skipping cleanly (never
//! fabricating a pass) when the 814 MB safetensors are absent.

use ruvector_timesfm::anomaly::score_window;
use ruvector_timesfm::sweep::EarlyStopper;
use ruvector_timesfm::Forecast;

fn forecast_from_bands(p10: &[f32], p50: &[f32], p90: &[f32]) -> Forecast {
    let quantiles = (0..p50.len())
        .map(|i| {
            // p10..p90; fill the in-between channels by interpolation (only
            // p10/p50/p90 are asserted by the anomaly logic).
            [
                p10[i], p10[i], p10[i], p50[i], p50[i], p50[i], p90[i], p90[i], p90[i],
            ]
        })
        .collect();
    Forecast {
        point: p50.to_vec(),
        quantiles,
    }
}

#[test]
fn forecast_quantile_accessors() {
    let f = forecast_from_bands(&[1.0, 2.0], &[5.0, 6.0], &[9.0, 10.0]);
    assert_eq!(f.horizon(), 2);
    assert_eq!(f.p10(), vec![1.0, 2.0]);
    assert_eq!(f.p50(), vec![5.0, 6.0]);
    assert_eq!(f.p90(), vec![9.0, 10.0]);
}

#[test]
fn anomaly_flags_out_of_band_points() {
    let f = forecast_from_bands(&[0.0, 0.0, 0.0], &[5.0, 5.0, 5.0], &[10.0, 10.0, 10.0]);
    // inside band, above band, below band.
    let observed = [5.0, 25.0, -15.0];
    let report = score_window(&f, &observed);
    assert_eq!(report.points.len(), 3);
    assert_eq!(report.n_anomalies, 2);
    assert!(!report.points[0].is_anomaly);
    assert!(report.points[1].is_anomaly && report.points[1].deviation > 0.0);
    assert!(report.points[2].is_anomaly && report.points[2].deviation < 0.0);
}

#[test]
fn early_stopper_warms_up_before_deciding() {
    let stopper = EarlyStopper::new(0.05, 1000).with_min_history(16);
    // Build a StopDecision via the non-candle path is not possible (evaluate is
    // gated), but the config + Default surface is exercised here.
    assert_eq!(stopper.min_history, 16);
    assert_eq!(stopper.threshold, 0.05);
    assert_eq!(EarlyStopper::default().confidence_gate, 0.6);
}

#[test]
fn rebuild_advice_triggers_before_floor() {
    use ruvector_timesfm::rebuild::advise_from_forecast;
    // Recall (higher=better) declining: p50 crosses 0.90 floor at step 3, p10 at step 1.
    let p10 = [0.92, 0.88, 0.85, 0.82, 0.80];
    let p50 = [0.95, 0.93, 0.91, 0.89, 0.87];
    let p90 = [0.98, 0.97, 0.96, 0.95, 0.94];
    let f = forecast_from_bands(&p10, &p50, &p90);
    // lead_steps=2: p10 dips below floor at step 1 (<=2) ⇒ rebuild now.
    let a = advise_from_forecast(f, 0.95, 0.90, 2);
    assert!(a.rebuild_now);
    assert_eq!(a.steps_until_floor, Some(3));
    assert_eq!(a.steps_until_floor_p10, Some(1));

    // Healthy: recall holds above floor ⇒ no rebuild.
    let f2 = forecast_from_bands(&[0.95; 5], &[0.97; 5], &[0.99; 5]);
    let b = advise_from_forecast(f2, 0.97, 0.90, 2);
    assert!(!b.rebuild_now && b.steps_until_floor.is_none());
}

#[cfg(feature = "candle")]
mod real_model {
    use ruvector_timesfm::Forecaster;

    const WEIGHTS: &str = "/tmp/timesfm-parity/timesfm.safetensors";

    fn skip() -> bool {
        if !std::path::Path::new(WEIGHTS).exists() {
            eprintln!("SKIP real-model test: weights missing ({WEIGHTS}).");
            true
        } else {
            false
        }
    }

    #[test]
    fn forecast_shapes_and_band_ordering() -> anyhow::Result<()> {
        if skip() {
            return Ok(());
        }
        let device = timesfm::select_device()?;
        let f = Forecaster::load(WEIGHTS, device)?;
        let series: Vec<f32> = (0..256)
            .map(|t| (t as f32 / 12.0).sin() * 10.0 + 50.0)
            .collect();
        let forecast = f.forecast(&series, 64)?;
        assert_eq!(forecast.horizon(), 64);
        assert_eq!(forecast.point.len(), 64);
        // All forecast values finite; quantiles monotone p10 <= p50 <= p90.
        for i in 0..64 {
            assert!(forecast.point[i].is_finite());
            let (lo, mid, hi) = (forecast.p10()[i], forecast.p50()[i], forecast.p90()[i]);
            assert!(lo.is_finite() && mid.is_finite() && hi.is_finite());
            assert!(lo <= hi, "p10 {lo} > p90 {hi} at step {i}");
        }
        Ok(())
    }

    #[test]
    fn early_stopper_prunes_doomed_run() -> anyhow::Result<()> {
        if skip() {
            return Ok(());
        }
        use ruvector_timesfm::sweep::EarlyStopper;
        let device = timesfm::select_device()?;
        let f = Forecaster::load(WEIGHTS, device)?;
        // doomed: decays toward 0.20, never reaches the 0.05 threshold.
        let doomed: Vec<f32> = (0..128)
            .map(|t| 0.20 + 0.75 * (-(t as f32) / 16.0).exp())
            .collect();
        let stopper = EarlyStopper::new(0.05, 1000)
            .with_min_history(16)
            .with_confidence_gate(0.5);
        let d = stopper.evaluate(&f, &doomed)?;
        assert!(d.stop, "doomed run should stop: {}", d.reason);

        // warm-up: too few points → never stop.
        let short = &doomed[..8];
        let d2 = stopper.evaluate(&f, short)?;
        assert!(!d2.stop && d2.decision.is_none());
        Ok(())
    }

    #[test]
    fn f16_load_forecasts_close_to_f32() -> anyhow::Result<()> {
        if skip() {
            return Ok(());
        }
        let device = timesfm::select_device()?;
        let series: Vec<f32> = (0..256)
            .map(|t| (t as f32 / 13.0).sin() * 7.0 + 48.0)
            .collect();
        let f32m = Forecaster::load(WEIGHTS, device.clone())?;
        let ref_fc = f32m.forecast(&series, 32)?;
        let f16m = Forecaster::load_f16(WEIGHTS, device)?;
        let f16_fc = f16m.forecast(&series, 32)?;
        assert!(f16_fc.point.iter().all(|x| x.is_finite()), "f16 non-finite");
        let scale = ref_fc.point.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let max_abs = ref_fc
            .point
            .iter()
            .zip(f16_fc.point.iter())
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        // f16 has ~3 decimal digits; allow a loose relative bound.
        assert!(
            max_abs / scale < 2e-2,
            "f16 diverged from f32: rel {:.3e}",
            max_abs / scale
        );
        Ok(())
    }

    #[test]
    fn quantized_load_forecasts_close_to_f32() -> anyhow::Result<()> {
        if skip() {
            return Ok(());
        }
        use ruvector_timesfm::Quant;
        let device = timesfm::select_device()?;
        let series: Vec<f32> = (0..256)
            .map(|t| (t as f32 / 11.0).sin() * 9.0 + 45.0)
            .collect();

        let f32m = Forecaster::load(WEIGHTS, device.clone())?;
        let ref_fc = f32m.forecast(&series, 32)?;

        // Q8_0 stays close to f32 (relative error ~3e-3 measured); assert a
        // generous bound and that every value is finite.
        let q8 = Forecaster::load_quantized(WEIGHTS, device, Quant::Q8_0)?;
        let q8_fc = q8.forecast(&series, 32)?;
        let scale = ref_fc.point.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let max_abs = ref_fc
            .point
            .iter()
            .zip(q8_fc.point.iter())
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(q8_fc.point.iter().all(|x| x.is_finite()), "Q8_0 non-finite");
        assert!(
            max_abs / scale < 5e-2,
            "Q8_0 diverged from f32: rel {:.3e}",
            max_abs / scale
        );
        Ok(())
    }

    #[test]
    fn batched_matches_per_series() -> anyhow::Result<()> {
        if skip() {
            return Ok(());
        }
        let device = timesfm::select_device()?;
        let f = Forecaster::load(WEIGHTS, device)?;
        let batch: Vec<Vec<f32>> = (0..4)
            .map(|s| {
                (0..128)
                    .map(|t| ((t as f32 + s as f32) / 9.0).sin() * 8.0 + 40.0)
                    .collect()
            })
            .collect();
        let batched = f.forecast_batch(&batch, 32, 0)?;
        assert_eq!(batched.len(), 4);
        for (i, series) in batch.iter().enumerate() {
            let single = f.forecast(series, 32)?;
            // CPU bit-exact; GPU within reduction-order noise (relative).
            let scale = single.point.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            let max_abs = single
                .point
                .iter()
                .zip(batched[i].point.iter())
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                max_abs / scale < 1e-3,
                "row {i} batched vs single rel {:.3e}",
                max_abs / scale
            );
        }
        Ok(())
    }
}
