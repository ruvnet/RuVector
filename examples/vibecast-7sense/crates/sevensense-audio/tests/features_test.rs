//! Tests for acoustic feature extraction (ADR-011).
//!
//! Every test drives the extractor with a synthetic signal whose descriptors are
//! known analytically, so failures point at the maths rather than at a fixture.

use sevensense_audio::features::{
    dominant_modulation, FeatureConfig, FeatureExtractor, MIN_FRAMES_FOR_MODULATION,
};

const SR: u32 = 32_000;

/// Generates a sine wave of `freq` Hz and `n` samples at amplitude `amp`.
fn tone(freq: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (std::f32::consts::TAU * freq * i as f32 / SR as f32).sin())
        .collect()
}

/// Deterministic uniform noise in [-amp, amp] via a linear congruential generator.
///
/// A fixed seed keeps assertions on statistical descriptors reproducible.
fn noise(n: usize, amp: f32, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / f32::from(u16::MAX) / 256.0;
            amp * (unit * 2.0 - 1.0)
        })
        .collect()
}

fn extractor() -> FeatureExtractor {
    FeatureExtractor::new(FeatureConfig::default()).expect("default config must be valid")
}

#[test]
fn pure_tone_centroid_matches_its_frequency() {
    let features = extractor().extract(&tone(4000.0, SR as usize, 0.8)).unwrap();

    let centroid = features.summary.centroid_hz.mean;
    assert!(
        (centroid - 4000.0).abs() < 60.0,
        "expected centroid near 4000 Hz, got {centroid}"
    );

    // A steady tone should not wander.
    assert!(
        features.summary.centroid_hz.std < 20.0,
        "centroid std was {}",
        features.summary.centroid_hz.std
    );
}

#[test]
fn dominant_frequency_beats_bin_resolution() {
    // 5010 Hz sits between bins (bin width is 31.25 Hz at the default config), so a
    // non-interpolated peak could only report 5000.0 or 5031.25.
    let features = extractor().extract(&tone(5010.0, SR as usize, 0.8)).unwrap();

    let dominant = features.summary.dominant_hz.mean;
    let bin_width = features.config.bin_width_hz();
    assert!(
        (dominant - 5010.0).abs() < bin_width / 2.0,
        "interpolated peak {dominant} should be within half a bin ({}) of 5010",
        bin_width / 2.0
    );
}

#[test]
fn tone_is_tonal_and_noise_is_flat() {
    let tonal = extractor().extract(&tone(4000.0, SR as usize, 0.8)).unwrap();
    let noisy = extractor().extract(&noise(SR as usize, 0.8, 42)).unwrap();

    assert!(
        tonal.summary.tonality.mean > 0.9,
        "tone tonality was {}",
        tonal.summary.tonality.mean
    );
    assert!(
        noisy.summary.tonality.mean < tonal.summary.tonality.mean - 0.3,
        "noise tonality {} should fall well below tone tonality {}",
        noisy.summary.tonality.mean,
        tonal.summary.tonality.mean
    );
    assert!(
        noisy.summary.entropy.mean > tonal.summary.entropy.mean + 0.3,
        "noise entropy {} should clearly exceed tone entropy {}",
        noisy.summary.entropy.mean,
        tonal.summary.entropy.mean
    );
}

#[test]
fn spread_is_narrow_for_a_tone_and_wide_for_noise() {
    let tonal = extractor().extract(&tone(4000.0, SR as usize, 0.8)).unwrap();
    let noisy = extractor().extract(&noise(SR as usize, 0.8, 7)).unwrap();

    assert!(
        tonal.summary.spread_hz.mean < 500.0,
        "tone spread was {}",
        tonal.summary.spread_hz.mean
    );
    assert!(
        noisy.summary.spread_hz.mean > 2000.0,
        "noise spread was {}",
        noisy.summary.spread_hz.mean
    );
}

#[test]
fn amplitude_modulation_recovers_a_known_rate() {
    // 4 kHz carrier, fully modulated at 20 Hz.
    let modulation_hz = 20.0;
    let samples: Vec<f32> = (0..SR as usize)
        .map(|i| {
            let t = i as f32 / SR as f32;
            let envelope = 0.5 * (1.0 - (std::f32::consts::TAU * modulation_hz * t).cos());
            envelope * (std::f32::consts::TAU * 4000.0 * t).sin()
        })
        .collect();

    let features = extractor().extract(&samples).unwrap();
    let modulation = features
        .summary
        .modulation
        .expect("one second of audio is long enough for modulation analysis");

    assert!(
        (modulation.am_rate_hz - modulation_hz).abs() < 2.0,
        "expected AM rate near {modulation_hz} Hz, got {}",
        modulation.am_rate_hz
    );
    assert!(
        modulation.am_depth > 0.1,
        "fully modulated signal should show depth, got {}",
        modulation.am_depth
    );
}

#[test]
fn frequency_sweep_shows_wide_fm_extent() {
    // Linear sweep from 2 kHz to 8 kHz over one second.
    let samples: Vec<f32> = (0..SR as usize)
        .map(|i| {
            let t = i as f32 / SR as f32;
            let phase = std::f32::consts::TAU * (2000.0 * t + 3000.0 * t * t);
            0.8 * phase.sin()
        })
        .collect();

    let features = extractor().extract(&samples).unwrap();
    let modulation = features.summary.modulation.expect("long enough for modulation");

    assert!(
        modulation.fm_extent_hz > 1000.0,
        "a 2-8 kHz sweep should show wide FM extent, got {}",
        modulation.fm_extent_hz
    );

    // The centroid should actually traverse the swept range.
    let track = features.centroid_track();
    let first = track.first().copied().unwrap();
    let last = track.last().copied().unwrap();
    assert!(
        last > first + 2000.0,
        "centroid should rise across the sweep: {first} -> {last}"
    );
}

#[test]
fn steady_tone_shows_little_amplitude_modulation() {
    let features = extractor().extract(&tone(4000.0, SR as usize, 0.8)).unwrap();
    let modulation = features.summary.modulation.expect("long enough");

    assert!(
        modulation.am_depth < 0.1,
        "unmodulated tone should have low AM depth, got {}",
        modulation.am_depth
    );
}

#[test]
fn silence_is_gated_out_of_summary_statistics() {
    let features = extractor().extract(&vec![0.0f32; SR as usize]).unwrap();

    assert_eq!(features.summary.voiced_fraction, 0.0);
    assert!(features.frames.iter().all(|f| !f.voiced));
    assert!(features.is_likely_noise());
}

#[test]
fn quiet_frames_do_not_drag_the_centroid() {
    // Half a second of a 6 kHz tone followed by half a second of silence. The
    // centroid summary should reflect the tone, not the average of tone and silence.
    let mut samples = tone(6000.0, SR as usize / 2, 0.8);
    samples.extend(std::iter::repeat(0.0).take(SR as usize / 2));

    let features = extractor().extract(&samples).unwrap();

    assert!(
        (features.summary.centroid_hz.mean - 6000.0).abs() < 100.0,
        "gated centroid was {}",
        features.summary.centroid_hz.mean
    );
    assert!(
        (features.summary.voiced_fraction - 0.5).abs() < 0.05,
        "voiced fraction was {}",
        features.summary.voiced_fraction
    );
}

#[test]
fn zero_crossing_rate_tracks_frequency() {
    let low = extractor().extract(&tone(1000.0, SR as usize, 0.8)).unwrap();
    let high = extractor().extract(&tone(8000.0, SR as usize, 0.8)).unwrap();

    assert!(
        high.summary.zcr.mean > low.summary.zcr.mean * 4.0,
        "8 kHz zcr {} should be far above 1 kHz zcr {}",
        high.summary.zcr.mean,
        low.summary.zcr.mean
    );
}

#[test]
fn spectral_slope_is_negative_for_low_frequency_energy() {
    let features = extractor().extract(&tone(800.0, SR as usize, 0.8)).unwrap();

    assert!(
        features.summary.slope.mean < 0.0,
        "energy concentrated low should give negative slope, got {}",
        features.summary.slope.mean
    );
}

#[test]
fn rolloff_sits_above_the_centroid_for_a_tone() {
    let features = extractor().extract(&tone(4000.0, SR as usize, 0.8)).unwrap();

    assert!(
        features.summary.rolloff_hz.mean >= features.summary.centroid_hz.mean - 50.0,
        "rolloff {} should not fall below centroid {}",
        features.summary.rolloff_hz.mean,
        features.summary.centroid_hz.mean
    );
}

#[test]
fn short_input_yields_no_modulation_estimate() {
    let config = FeatureConfig::default();
    // Just under the frame count required for modulation analysis.
    let n_samples = config.n_fft + (MIN_FRAMES_FOR_MODULATION - 2) * config.hop_length;
    let features = extractor().extract(&tone(4000.0, n_samples, 0.8)).unwrap();

    assert!(features.frames.len() < MIN_FRAMES_FOR_MODULATION);
    assert!(
        features.summary.modulation.is_none(),
        "modulation must not be fabricated for short input"
    );
}

#[test]
fn input_shorter_than_one_window_is_an_error() {
    let config = FeatureConfig::default();
    let result = extractor().extract(&tone(4000.0, config.n_fft - 1, 0.8));

    assert!(result.is_err(), "expected an error for sub-window input");
}

#[test]
fn frame_count_matches_frames_produced() {
    let extractor = extractor();
    for n_samples in [1024, 2048, 5000, 32_000, 44_100] {
        let expected = extractor.frame_count(n_samples);
        let actual = extractor.extract(&tone(4000.0, n_samples, 0.8)).unwrap();
        assert_eq!(
            expected,
            actual.frames.len(),
            "frame_count disagreed for {n_samples} samples"
        );
    }
}

#[test]
fn frame_timestamps_advance_by_the_hop_duration() {
    let features = extractor().extract(&tone(4000.0, SR as usize, 0.8)).unwrap();
    let config = &features.config;
    let expected_step = f64::from(config.hop_length as u32) * 1000.0 / f64::from(config.sample_rate);

    assert!((features.frames[0].time_ms - 0.0).abs() < 1e-6);
    let step = features.frames[1].time_ms - features.frames[0].time_ms;
    assert!(
        (step - expected_step).abs() < 1e-6,
        "expected {expected_step} ms per frame, got {step}"
    );
}

#[test]
fn normalized_descriptors_stay_within_their_declared_range() {
    let signals = [
        tone(4000.0, SR as usize, 0.8),
        noise(SR as usize, 0.8, 99),
        tone(200.0, SR as usize, 0.01),
        vec![0.0; SR as usize],
    ];

    for (idx, signal) in signals.iter().enumerate() {
        let features = extractor().extract(signal).unwrap();
        for frame in &features.frames {
            assert!(
                (0.0..=1.0).contains(&frame.tonality),
                "signal {idx}: tonality {} out of range",
                frame.tonality
            );
            assert!(
                (0.0..=1.0).contains(&frame.flatness),
                "signal {idx}: flatness {} out of range",
                frame.flatness
            );
            assert!(
                (0.0..=1.0).contains(&frame.crest),
                "signal {idx}: crest {} out of range",
                frame.crest
            );
            assert!(
                (0.0..=1.0).contains(&frame.entropy),
                "signal {idx}: entropy {} out of range",
                frame.entropy
            );
            assert!(
                (0.0..=1.0).contains(&frame.zcr),
                "signal {idx}: zcr {} out of range",
                frame.zcr
            );
        }
    }
}

#[test]
fn descriptors_are_finite_for_pathological_input() {
    // Constant DC, alternating full-scale, and a single impulse each break naive
    // implementations of flatness, crest, or slope.
    let dc = vec![1.0f32; SR as usize / 4];
    let alternating: Vec<f32> = (0..SR as usize / 4)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut impulse = vec![0.0f32; SR as usize / 4];
    impulse[5000] = 1.0;

    for (name, signal) in [("dc", dc), ("alternating", alternating), ("impulse", impulse)] {
        let features = extractor().extract(&signal).unwrap();
        for frame in &features.frames {
            assert!(frame.centroid_hz.is_finite(), "{name}: centroid not finite");
            assert!(frame.spread_hz.is_finite(), "{name}: spread not finite");
            assert!(frame.flatness.is_finite(), "{name}: flatness not finite");
            assert!(frame.entropy.is_finite(), "{name}: entropy not finite");
            assert!(frame.slope.is_finite(), "{name}: slope not finite");
            assert!(frame.crest.is_finite(), "{name}: crest not finite");
            assert!(frame.skewness.is_finite(), "{name}: skewness not finite");
        }
    }
}

#[test]
fn invalid_configurations_are_rejected() {
    let cases = [
        ("zero hop", FeatureConfig { hop_length: 0, ..Default::default() }),
        ("tiny fft", FeatureConfig { n_fft: 1, ..Default::default() }),
        ("zero sample rate", FeatureConfig { sample_rate: 0, ..Default::default() }),
        ("inverted band", FeatureConfig { f_min: 9000.0, f_max: 400.0, ..Default::default() }),
        ("zero rolloff", FeatureConfig { rolloff_percent: 0.0, ..Default::default() }),
    ];

    for (name, config) in cases {
        assert!(
            FeatureExtractor::new(config).is_err(),
            "{name} should have been rejected"
        );
    }
}

#[test]
fn a_band_too_narrow_to_analyze_is_rejected() {
    // A 64-point FFT at 32 kHz gives 500 Hz bins.

    // No bins at all: 1100..1400 falls entirely between bin 2 (1000 Hz) and bin 3.
    let empty = FeatureConfig {
        n_fft: 64,
        f_min: 1100.0,
        f_max: 1400.0,
        ..Default::default()
    };
    assert!(
        FeatureExtractor::new(empty).is_err(),
        "a band containing no bins must be rejected"
    );

    // Exactly one bin: 1000..1010 captures bin 2 and nothing else. Spread, slope,
    // and entropy are undefined over a single bin, so this is rejected too.
    let single = FeatureConfig {
        n_fft: 64,
        f_min: 1000.0,
        f_max: 1010.0,
        ..Default::default()
    };
    assert!(
        FeatureExtractor::new(single).is_err(),
        "a single-bin band must be rejected rather than returning zeros"
    );
}

#[test]
fn features_survive_a_serde_round_trip() {
    let features = extractor().extract(&tone(4000.0, SR as usize / 4, 0.8)).unwrap();

    let json = serde_json::to_string(&features).expect("serialization must succeed");
    let restored: sevensense_audio::features::AcousticFeatures =
        serde_json::from_str(&json).expect("deserialization must succeed");

    assert_eq!(features, restored);
}

#[test]
fn modulation_search_is_bounded_by_envelope_nyquist() {
    // A 30 Hz modulation sampled at a 100 Hz frame rate is measurable; the reported
    // rate must never exceed half the frame rate regardless of input.
    let frame_rate = 100.0;
    let track: Vec<f32> = (0..128)
        .map(|i| (std::f32::consts::TAU * 30.0 * i as f32 / frame_rate).sin())
        .collect();

    let (rate, magnitude) = dominant_modulation(&track, frame_rate);

    assert!((rate - 30.0).abs() < 2.0, "expected ~30 Hz, got {rate}");
    assert!(rate <= frame_rate / 2.0, "rate {rate} exceeded Nyquist");
    assert!(magnitude > 0.0);
}

#[test]
fn extractor_is_reusable_across_calls() {
    // The FFT plan and window are built once; repeated use must not accumulate state.
    let extractor = extractor();
    let first = extractor.extract(&tone(4000.0, SR as usize / 2, 0.8)).unwrap();
    let _ = extractor.extract(&noise(SR as usize / 2, 0.5, 3)).unwrap();
    let third = extractor.extract(&tone(4000.0, SR as usize / 2, 0.8)).unwrap();

    assert_eq!(
        first.summary.centroid_hz.mean, third.summary.centroid_hz.mean,
        "identical input must produce identical output"
    );
}
