//! Real audio evaluation using downloaded public domain WAV files.
//!
//! Downloads ESC-50 environmental sounds, Signalogic speech samples,
//! and SampleLib music. Mixes them into realistic scenarios, separates
//! with Musica's graph mincut, and measures SDR/SIR/SAR.

use crate::audio_graph::{build_audio_graph, GraphParams};
use crate::evaluation::{compute_sdr, compute_sir, compute_sar};
use crate::separator::{separate, SeparatorConfig};
use crate::stft;
use crate::wav;

/// Result of evaluating separation on a real audio mix.
#[derive(Debug, Clone)]
pub struct RealAudioResult {
    /// Scenario name.
    pub name: String,
    /// Per-source SDR (dB).
    pub source_sdr: Vec<(String, f64)>,
    /// Average SDR.
    pub avg_sdr: f64,
    /// Processing time (ms).
    pub processing_ms: f64,
    /// Number of samples processed.
    pub num_samples: usize,
    /// Sample rate.
    pub sample_rate: f64,
    /// Graph nodes.
    pub graph_nodes: usize,
}

/// Load a WAV file and return mono f64 samples at the native sample rate.
fn load_mono(path: &str) -> Option<(Vec<f64>, u32)> {
    match wav::read_wav(path) {
        Ok(data) => {
            let mono = if data.channels == 1 {
                data.channel_data[0].clone()
            } else {
                // Mix to mono
                let n = data.channel_data[0].len();
                (0..n)
                    .map(|i| {
                        data.channel_data
                            .iter()
                            .map(|ch| ch[i])
                            .sum::<f64>()
                            / data.channels as f64
                    })
                    .collect()
            };
            Some((mono, data.sample_rate))
        }
        Err(e) => {
            println!("    [WARN] Could not load {}: {}", path, e);
            None
        }
    }
}

/// Resample to target rate using linear interpolation.
fn resample(samples: &[f64], from_rate: u32, to_rate: u32) -> Vec<f64> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (samples.len() as f64 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = pos - idx as f64;
            let s0 = samples[idx.min(samples.len() - 1)];
            let s1 = samples[(idx + 1).min(samples.len() - 1)];
            s0 + frac * (s1 - s0)
        })
        .collect()
}

/// Trim or pad a signal to exactly `n` samples.
fn fit_length(signal: &[f64], n: usize) -> Vec<f64> {
    if signal.len() >= n {
        signal[..n].to_vec()
    } else {
        let mut out = signal.to_vec();
        out.resize(n, 0.0);
        out
    }
}

/// Mix two signals at a given SNR (dB). Returns (mixed, [signal, noise]).
fn mix_at_snr(signal: &[f64], noise: &[f64], snr_db: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = signal.len().min(noise.len());
    let sig_power: f64 = signal[..n].iter().map(|x| x * x).sum::<f64>() / n as f64;
    let noise_power: f64 = noise[..n].iter().map(|x| x * x).sum::<f64>() / n as f64;

    // Scale noise to achieve target SNR
    let target_noise_power = sig_power / 10.0f64.powf(snr_db / 10.0);
    let scale = if noise_power > 1e-12 {
        (target_noise_power / noise_power).sqrt()
    } else {
        1.0
    };

    let scaled_noise: Vec<f64> = noise[..n].iter().map(|x| x * scale).collect();
    let mixed: Vec<f64> = signal[..n]
        .iter()
        .zip(scaled_noise.iter())
        .map(|(s, n)| s + n)
        .collect();

    (mixed, vec![signal[..n].to_vec(), scaled_noise])
}

/// Run separation on a mix and compute SDR for each source.
fn evaluate_mix(
    mixed: &[f64],
    sources: &[Vec<f64>],
    labels: &[&str],
    sample_rate: f64,
    window_size: usize,
    hop_size: usize,
) -> RealAudioResult {
    let start = std::time::Instant::now();

    let stft_result = stft::stft(mixed, window_size, hop_size, sample_rate);
    let graph = build_audio_graph(&stft_result, &GraphParams::default());
    let graph_nodes = graph.num_nodes;

    let sep_config = SeparatorConfig {
        num_sources: sources.len(),
        ..SeparatorConfig::default()
    };
    let separation = separate(&graph, &sep_config);

    let processing_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut source_sdr = Vec::new();
    let mut total_sdr = 0.0;
    let num = separation.masks.len().min(sources.len());

    for s in 0..num {
        let recovered = stft::istft(&stft_result, &separation.masks[s], mixed.len());
        let sdr = compute_sdr(&sources[s], &recovered);
        let label = if s < labels.len() { labels[s] } else { "unknown" };
        source_sdr.push((label.to_string(), sdr));
        total_sdr += sdr;
    }

    let avg_sdr = if num > 0 { total_sdr / num as f64 } else { f64::NEG_INFINITY };

    RealAudioResult {
        name: labels.join(" + "),
        source_sdr,
        avg_sdr,
        processing_ms,
        num_samples: mixed.len(),
        sample_rate,
        graph_nodes,
    }
}

/// Run all real audio evaluation scenarios.
///
/// Expects WAV files in `test_audio/` directory. If files are missing,
/// those scenarios are skipped with a warning.
pub fn run_real_audio_benchmarks(audio_dir: &str) -> Vec<RealAudioResult> {
    let target_sr = 8000u32; // Use 8kHz for faster processing
    let target_duration = 2.0; // 2 seconds
    let target_samples = (target_sr as f64 * target_duration) as usize;
    let mut results = Vec::new();

    println!("  Loading real audio from {}/", audio_dir);

    // Load all available files
    let files = [
        ("rain", format!("{}/rain.wav", audio_dir)),
        ("birds", format!("{}/birds.wav", audio_dir)),
        ("clapping", format!("{}/clapping.wav", audio_dir)),
        ("laughing", format!("{}/laughing.wav", audio_dir)),
        ("dog", format!("{}/dog.wav", audio_dir)),
        ("church_bells", format!("{}/church_bells.wav", audio_dir)),
        ("speech_male", format!("{}/speech_male.wav", audio_dir)),
        ("speech_female", format!("{}/speech_female.wav", audio_dir)),
        ("music", format!("{}/music_6s.wav", audio_dir)),
        ("noise_tone", format!("{}/noise_tone.wav", audio_dir)),
    ];

    let mut loaded: std::collections::HashMap<&str, Vec<f64>> = std::collections::HashMap::new();

    for (name, path) in &files {
        if let Some((samples, sr)) = load_mono(path) {
            let resampled = resample(&samples, sr, target_sr);
            let fitted = fit_length(&resampled, target_samples);
            loaded.insert(name, fitted);
            println!("    Loaded {}: {} samples at {}Hz → resampled to {}Hz", name, samples.len(), sr, target_sr);
        }
    }

    if loaded.is_empty() {
        println!("    [ERROR] No audio files found. Download with scripts/download_test_audio.sh");
        return results;
    }

    let ws = 256;
    let hs = 128;
    let sr = target_sr as f64;

    // Scenario 1: Speech + Rain (SNR = 5 dB)
    if let (Some(speech), Some(rain)) = (loaded.get("speech_male"), loaded.get("rain")) {
        println!("\n  ── Scenario 1: Speech + Rain Noise (5dB SNR) ──");
        let (mixed, sources) = mix_at_snr(speech, rain, 5.0);
        let result = evaluate_mix(&mixed, &sources, &["speech", "rain"], sr, ws, hs);
        print_result(&result);
        results.push(result);
    }

    // Scenario 2: Male + Female speech (equal energy)
    if let (Some(male), Some(female)) = (loaded.get("speech_male"), loaded.get("speech_female")) {
        println!("\n  ── Scenario 2: Male + Female Speech (0dB) ──");
        let (mixed, sources) = mix_at_snr(male, female, 0.0);
        let result = evaluate_mix(&mixed, &sources, &["male", "female"], sr, ws, hs);
        print_result(&result);
        results.push(result);
    }

    // Scenario 3: Music + Crowd noise (clapping)
    if let (Some(music), Some(crowd)) = (loaded.get("music"), loaded.get("clapping")) {
        println!("\n  ── Scenario 3: Music + Crowd Noise (3dB) ──");
        let (mixed, sources) = mix_at_snr(music, crowd, 3.0);
        let result = evaluate_mix(&mixed, &sources, &["music", "crowd"], sr, ws, hs);
        print_result(&result);
        results.push(result);
    }

    // Scenario 4: Birds + Church bells (environmental separation)
    if let (Some(birds), Some(bells)) = (loaded.get("birds"), loaded.get("church_bells")) {
        println!("\n  ── Scenario 4: Birds + Church Bells (0dB) ──");
        let (mixed, sources) = mix_at_snr(birds, bells, 0.0);
        let result = evaluate_mix(&mixed, &sources, &["birds", "bells"], sr, ws, hs);
        print_result(&result);
        results.push(result);
    }

    // Scenario 5: Speech + Dog barking (hearing aid scenario)
    if let (Some(speech), Some(dog)) = (loaded.get("speech_female"), loaded.get("dog")) {
        println!("\n  ── Scenario 5: Speech + Dog Barking (10dB SNR) ──");
        let (mixed, sources) = mix_at_snr(speech, dog, 10.0);
        let result = evaluate_mix(&mixed, &sources, &["speech", "dog"], sr, ws, hs);
        print_result(&result);
        results.push(result);
    }

    // Scenario 6: Speech + Music background
    if let (Some(speech), Some(music)) = (loaded.get("speech_male"), loaded.get("music")) {
        println!("\n  ── Scenario 6: Speech over Music (-3dB) ──");
        let (mixed, sources) = mix_at_snr(speech, music, -3.0);
        let result = evaluate_mix(&mixed, &sources, &["speech", "music"], sr, ws, hs);
        print_result(&result);
        results.push(result);
    }

    // Summary
    if !results.is_empty() {
        println!("\n  ── Summary: Real Audio Separation Quality ──");
        println!("  {:<35} {:>8} {:>10}", "Scenario", "Avg SDR", "Time(ms)");
        println!("  {}", "-".repeat(55));
        for r in &results {
            println!("  {:<35} {:>+7.2}dB {:>9.1}", r.name, r.avg_sdr, r.processing_ms);
        }
        let overall_avg: f64 = results.iter().map(|r| r.avg_sdr).sum::<f64>() / results.len() as f64;
        println!("  {}", "-".repeat(55));
        println!("  {:<35} {:>+7.2}dB", "OVERALL AVERAGE", overall_avg);
    }

    results
}

fn print_result(result: &RealAudioResult) {
    for (label, sdr) in &result.source_sdr {
        println!("    {:<12} SDR: {:+.2} dB", label, sdr);
    }
    println!("    Average:     {:+.2} dB | {:.1}ms | {} nodes", result.avg_sdr, result.processing_ms, result.graph_nodes);
}

/// Download script content for test audio files.
pub fn download_script() -> &'static str {
    r#"#!/bin/bash
# Download public domain audio files for Musica evaluation
set -e
AUDIO_DIR="$(dirname "$0")/../test_audio"
mkdir -p "$AUDIO_DIR" && cd "$AUDIO_DIR"

echo "Downloading ESC-50 environmental sounds..."
curl -s -o rain.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-17367-A-10.wav"
curl -s -o birds.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-100038-A-14.wav"
curl -s -o clapping.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-104089-A-22.wav"
curl -s -o laughing.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-1791-A-26.wav"
curl -s -o dog.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-100032-A-0.wav"
curl -s -o church_bells.wav "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-13571-A-46.wav"

echo "Downloading speech samples..."
curl -s -o speech_male.wav "https://www.signalogic.com/melp/EngSamples/Orig/male.wav"
curl -s -o speech_female.wav "https://www.signalogic.com/melp/EngSamples/Orig/female.wav"

echo "Downloading music..."
curl -s -o music_6s.wav "https://samplelib.com/wav/sample-6s.wav"

echo "Downloading test tone..."
curl -s -o noise_tone.wav "https://raw.githubusercontent.com/exaile/exaile-test-files/master/noise_tone.wav"

echo "Downloaded $(ls *.wav | wc -l) WAV files:"
ls -lh *.wav
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_identity() {
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = resample(&signal, 16000, 16000);
        assert_eq!(result.len(), signal.len());
    }

    #[test]
    fn test_resample_downsample() {
        let signal: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
        let result = resample(&signal, 44100, 8000);
        let expected = (1000.0 * 8000.0 / 44100.0) as usize;
        assert!((result.len() as i64 - expected as i64).abs() <= 1);
    }

    #[test]
    fn test_fit_length_pad() {
        let signal = vec![1.0, 2.0, 3.0];
        let result = fit_length(&signal, 5);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_fit_length_trim() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = fit_length(&signal, 3);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mix_at_snr() {
        let signal = vec![1.0; 100];
        let noise = vec![1.0; 100];
        let (mixed, sources) = mix_at_snr(&signal, &noise, 10.0);
        assert_eq!(mixed.len(), 100);
        assert_eq!(sources.len(), 2);

        // At 10dB SNR, noise should be ~0.316x the signal
        let noise_rms: f64 = (sources[1].iter().map(|x| x * x).sum::<f64>() / 100.0).sqrt();
        assert!(noise_rms < 0.5, "Noise at 10dB SNR should be attenuated: {}", noise_rms);
    }

    #[test]
    fn test_download_script_content() {
        let script = download_script();
        assert!(script.contains("curl"));
        assert!(script.contains("ESC-50"));
    }
}
