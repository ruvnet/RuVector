//! Minimal WAV file reader/writer — no external dependencies.
//!
//! Supports 16-bit PCM mono and stereo WAV files.
//! Sufficient for testing with real audio data.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Audio data loaded from a WAV file.
#[derive(Debug, Clone)]
pub struct WavData {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Bits per sample.
    pub bits_per_sample: u16,
    /// Interleaved samples normalized to [-1.0, 1.0].
    pub samples: Vec<f64>,
    /// Per-channel de-interleaved samples.
    pub channel_data: Vec<Vec<f64>>,
}

/// Read a WAV file and return normalized f64 samples.
pub fn read_wav<P: AsRef<Path>>(path: P) -> io::Result<WavData> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // RIFF header
    let mut riff = [0u8; 4];
    reader.read_exact(&mut riff)?;
    if &riff != b"RIFF" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a RIFF file"));
    }

    let mut size_buf = [0u8; 4];
    reader.read_exact(&mut size_buf)?; // file size - 8

    let mut wave = [0u8; 4];
    reader.read_exact(&mut wave)?;
    if &wave != b"WAVE" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a WAVE file"));
    }

    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut data_bytes = Vec::new();

    // Read chunks
    loop {
        let mut chunk_id = [0u8; 4];
        if reader.read_exact(&mut chunk_id).is_err() {
            break;
        }

        let mut chunk_size_buf = [0u8; 4];
        reader.read_exact(&mut chunk_size_buf)?;
        let chunk_size = u32::from_le_bytes(chunk_size_buf) as usize;

        match &chunk_id {
            b"fmt " => {
                let mut fmt = vec![0u8; chunk_size];
                reader.read_exact(&mut fmt)?;

                let audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
                if audio_format != 1 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Unsupported audio format: {audio_format} (only PCM=1 supported)"),
                    ));
                }

                channels = u16::from_le_bytes([fmt[2], fmt[3]]);
                sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
                bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
            }
            b"data" => {
                data_bytes = vec![0u8; chunk_size];
                reader.read_exact(&mut data_bytes)?;
            }
            _ => {
                // Skip unknown chunks
                let mut skip = vec![0u8; chunk_size];
                reader.read_exact(&mut skip)?;
            }
        }
    }

    if data_bytes.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "No data chunk found"));
    }

    // Parse samples
    let samples: Vec<f64> = match bits_per_sample {
        8 => data_bytes
            .iter()
            .map(|&b| {
                // 8-bit WAV is unsigned: 0-255, center at 128
                (b as f64 - 128.0) / 128.0
            })
            .collect(),
        16 => data_bytes
            .chunks_exact(2)
            .map(|b| {
                let s = i16::from_le_bytes([b[0], b[1]]);
                s as f64 / 32768.0
            })
            .collect(),
        24 => data_bytes
            .chunks_exact(3)
            .map(|b| {
                let s = ((b[0] as i32) | ((b[1] as i32) << 8) | ((b[2] as i32) << 16))
                    << 8 >> 8; // Sign extend
                s as f64 / 8388608.0
            })
            .collect(),
        32 => data_bytes
            .chunks_exact(4)
            .map(|b| {
                let s = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                s as f64 / 2147483648.0
            })
            .collect(),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported bits per sample: {bits_per_sample}"),
            ));
        }
    };

    // De-interleave
    let ch = channels as usize;
    let mut channel_data = vec![Vec::new(); ch];
    for (i, &s) in samples.iter().enumerate() {
        channel_data[i % ch].push(s);
    }

    Ok(WavData {
        sample_rate,
        channels,
        bits_per_sample,
        samples,
        channel_data,
    })
}

/// Write normalized f64 samples to a 16-bit PCM WAV file.
pub fn write_wav<P: AsRef<Path>>(
    path: P,
    samples: &[f64],
    sample_rate: u32,
    channels: u16,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    // RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // chunk size
    writer.write_all(&1u16.to_le_bytes())?; // PCM
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let quantized = (clamped * 32767.0) as i16;
        writer.write_all(&quantized.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

/// Generate a synthetic test WAV for benchmarking.
pub fn generate_test_wav<P: AsRef<Path>>(
    path: P,
    sample_rate: u32,
    duration_secs: f64,
    frequencies: &[f64],
    amplitudes: &[f64],
) -> io::Result<()> {
    use std::f64::consts::PI;
    let n = (sample_rate as f64 * duration_secs) as usize;
    let mut samples = vec![0.0f64; n];

    for (&freq, &amp) in frequencies.iter().zip(amplitudes.iter()) {
        for (i, s) in samples.iter_mut().enumerate() {
            let t = i as f64 / sample_rate as f64;
            *s += amp * (2.0 * PI * freq * t).sin();
        }
    }

    // Normalize to prevent clipping
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
    if peak > 0.95 {
        let scale = 0.9 / peak;
        for s in &mut samples {
            *s *= scale;
        }
    }

    write_wav(path, &samples, sample_rate, 1)
}

/// Generate a binaural (stereo) test WAV with spatial cues.
pub fn generate_binaural_test_wav<P: AsRef<Path>>(
    path: P,
    sample_rate: u32,
    duration_secs: f64,
    speech_freq: f64,
    noise_freqs: &[f64],
    speech_angle_deg: f64,
) -> io::Result<()> {
    use std::f64::consts::PI;
    let n = (sample_rate as f64 * duration_secs) as usize;
    let mut left = vec![0.0f64; n];
    let mut right = vec![0.0f64; n];

    // Interaural time difference (ITD) model: ~0.6ms max at 90 degrees
    let max_itd_samples = (0.0006 * sample_rate as f64) as usize;
    let angle_rad = speech_angle_deg * PI / 180.0;
    let itd = (angle_rad.sin() * max_itd_samples as f64) as isize;

    // Speech signal with harmonics
    for i in 0..n {
        let t = i as f64 / sample_rate as f64;
        let speech = 0.5 * (2.0 * PI * speech_freq * t).sin()
            + 0.15 * (2.0 * PI * speech_freq * 2.0 * t).sin()
            + 0.08 * (2.0 * PI * speech_freq * 3.0 * t).sin();

        // Apply ITD for spatial cue
        let li = i;
        let ri = (i as isize + itd).clamp(0, n as isize - 1) as usize;

        left[li] += speech;
        right[ri] += speech * 0.9; // Slight ILD
    }

    // Diffuse noise (different at each ear)
    let mut rng = 42u64;
    for i in 0..n {
        let t = i as f64 / sample_rate as f64;
        let mut noise_l = 0.0;
        let mut noise_r = 0.0;

        for &nf in noise_freqs {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let phase_l = (rng >> 32) as f64 / u32::MAX as f64 * 2.0 * PI;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let phase_r = (rng >> 32) as f64 / u32::MAX as f64 * 2.0 * PI;

            noise_l += 0.2 * (2.0 * PI * nf * t + phase_l).sin();
            noise_r += 0.2 * (2.0 * PI * nf * t + phase_r).sin();
        }

        left[i] += noise_l;
        right[i] += noise_r;
    }

    // Normalize
    let peak = left
        .iter()
        .chain(right.iter())
        .map(|s| s.abs())
        .fold(0.0f64, f64::max);
    if peak > 0.95 {
        let scale = 0.9 / peak;
        for s in &mut left {
            *s *= scale;
        }
        for s in &mut right {
            *s *= scale;
        }
    }

    // Interleave
    let mut stereo = Vec::with_capacity(n * 2);
    for i in 0..n {
        stereo.push(left[i]);
        stereo.push(right[i]);
    }

    write_wav(path, &stereo, sample_rate, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_wav_roundtrip() {
        use std::f64::consts::PI;
        let sr = 16000u32;
        let n = 1600; // 100ms
        let samples: Vec<f64> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 440.0 * i as f64 / sr as f64).sin())
            .collect();

        let path = "/tmp/musica_test_roundtrip.wav";
        write_wav(path, &samples, sr, 1).unwrap();
        let loaded = read_wav(path).unwrap();

        assert_eq!(loaded.sample_rate, sr);
        assert_eq!(loaded.channels, 1);
        assert_eq!(loaded.channel_data.len(), 1);
        assert_eq!(loaded.channel_data[0].len(), n);

        // 16-bit quantization error should be small
        for (i, (&orig, &loaded_s)) in samples.iter().zip(loaded.channel_data[0].iter()).enumerate() {
            assert!(
                (orig - loaded_s).abs() < 0.001,
                "Sample {i}: orig={orig:.4}, loaded={loaded_s:.4}"
            );
        }

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_8bit_wav_read() {
        use std::fs;

        // Write a raw 8-bit WAV manually
        let path = "/tmp/musica_test_8bit.wav";
        let sr = 8000u32;
        let n = 800u32; // 100ms
        let bits: u16 = 8;
        let channels: u16 = 1;
        let byte_rate = sr * channels as u32 * bits as u32 / 8;
        let block_align = channels * bits / 8;
        let data_size = n;
        let file_size = 36 + data_size;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sr.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());

        // 8-bit unsigned samples: silence=128, max=255, min=0
        for i in 0..n {
            let t = i as f64 / sr as f64;
            let s = (std::f64::consts::PI * 2.0 * 440.0 * t).sin();
            let byte = ((s * 127.0) + 128.0).clamp(0.0, 255.0) as u8;
            buf.push(byte);
        }

        fs::write(path, &buf).unwrap();
        let loaded = read_wav(path).unwrap();

        assert_eq!(loaded.sample_rate, sr);
        assert_eq!(loaded.bits_per_sample, 8);
        assert_eq!(loaded.channel_data[0].len(), n as usize);

        // Verify samples are in [-1, 1] range
        for &s in &loaded.channel_data[0] {
            assert!(s >= -1.01 && s <= 1.01, "8-bit sample out of range: {s}");
        }

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_stereo_wav_roundtrip() {
        let path = "/tmp/musica_test_stereo.wav";
        generate_binaural_test_wav(
            path, 16000, 0.1, 300.0, &[800.0, 1200.0], 30.0,
        )
        .unwrap();

        let loaded = read_wav(path).unwrap();
        assert_eq!(loaded.channels, 2);
        assert_eq!(loaded.channel_data.len(), 2);
        assert!(loaded.channel_data[0].len() > 0);

        fs::remove_file(path).ok();
    }
}
