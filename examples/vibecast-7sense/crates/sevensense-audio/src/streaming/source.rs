//! Audio source abstraction for streaming ingestion.
//!
//! Device access sits behind a trait so the core crate stays testable without
//! hardware, and so targets with no audio backend -- notably
//! `wasm32-unknown-unknown` -- keep building. ADR-010 records the decision to
//! keep `cpal` behind a non-default feature for exactly this reason.

use crate::AudioError;

/// A source of mono or interleaved audio samples.
///
/// Implementations range from a live capture device to an in-memory buffer. The
/// streaming pipeline is written against this trait alone, so file replay, live
/// capture, and tests all exercise the same code path.
pub trait AudioSource: Send {
    /// Sample rate of the samples this source yields, in Hz.
    fn sample_rate(&self) -> u32;

    /// Number of interleaved channels.
    fn channels(&self) -> u16;

    /// Fills `out` with the next available samples, returning how many were
    /// written.
    ///
    /// Returns `Ok(0)` when a finite source is exhausted or a live source has
    /// nothing available yet; callers distinguish the two with
    /// [`AudioSource::is_exhausted`].
    ///
    /// # Errors
    /// Returns [`AudioError`] when the underlying source fails.
    fn read(&mut self, out: &mut [f32]) -> Result<usize, AudioError>;

    /// Whether this source will never produce another sample.
    ///
    /// Live devices return `false` for their lifetime; finite sources return
    /// `true` once drained.
    fn is_exhausted(&self) -> bool;
}

/// An in-memory audio source, for tests, benchmarks, and file replay.
///
/// `chunk_size` bounds how much a single [`AudioSource::read`] returns, which
/// lets a test reproduce the arrival pattern of a live device without hardware.
#[derive(Debug, Clone)]
pub struct MemorySource {
    samples: Vec<f32>,
    position: usize,
    sample_rate: u32,
    channels: u16,
    chunk_size: usize,
}

impl MemorySource {
    /// Creates a source that yields `samples` in chunks of at most `chunk_size`.
    ///
    /// # Panics
    /// Panics if `chunk_size` is zero.
    #[must_use]
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be non-zero");
        Self {
            samples,
            position: 0,
            sample_rate,
            channels,
            chunk_size,
        }
    }

    /// Creates a mono source yielding 512-sample chunks, matching a typical
    /// device callback size.
    #[must_use]
    pub fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self::new(samples, sample_rate, 1, 512)
    }

    /// Samples not yet read.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.samples.len() - self.position
    }

    /// Restarts playback from the beginning.
    pub fn rewind(&mut self) {
        self.position = 0;
    }
}

impl AudioSource for MemorySource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn read(&mut self, out: &mut [f32]) -> Result<usize, AudioError> {
        let count = self.remaining().min(out.len()).min(self.chunk_size);
        out[..count].copy_from_slice(&self.samples[self.position..self.position + count]);
        self.position += count;
        Ok(count)
    }

    fn is_exhausted(&self) -> bool {
        self.position >= self.samples.len()
    }
}

/// Downmixes interleaved multi-channel audio to mono in place.
///
/// Returns the number of mono samples written to the front of `buffer`.
/// Averaging rather than taking channel zero preserves energy from sources where
/// the signal is panned or where one channel carries more of the vocalization.
pub fn downmix_to_mono(buffer: &mut [f32], channels: u16) -> usize {
    let channels = channels.max(1) as usize;
    if channels == 1 {
        return buffer.len();
    }

    let frames = buffer.len() / channels;
    let scale = 1.0 / channels as f32;
    for frame in 0..frames {
        let sum: f32 = buffer[frame * channels..frame * channels + channels]
            .iter()
            .sum();
        buffer[frame] = sum * scale;
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_source_yields_everything_in_order() {
        let mut source = MemorySource::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], 32_000, 1, 2);
        let mut out = [0.0f32; 8];
        let mut collected = Vec::new();

        while !source.is_exhausted() {
            let count = source.read(&mut out).unwrap();
            collected.extend_from_slice(&out[..count]);
        }

        assert_eq!(collected, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn chunk_size_bounds_each_read() {
        let mut source = MemorySource::new(vec![0.0; 100], 32_000, 1, 7);
        let mut out = [0.0f32; 64];

        assert_eq!(source.read(&mut out).unwrap(), 7);
    }

    #[test]
    fn output_slice_also_bounds_a_read() {
        let mut source = MemorySource::new(vec![0.0; 100], 32_000, 1, 64);
        let mut out = [0.0f32; 5];

        assert_eq!(source.read(&mut out).unwrap(), 5);
    }

    #[test]
    fn an_exhausted_source_reads_zero() {
        let mut source = MemorySource::mono(vec![1.0, 2.0], 32_000);
        let mut out = [0.0f32; 8];

        assert_eq!(source.read(&mut out).unwrap(), 2);
        assert!(source.is_exhausted());
        assert_eq!(source.read(&mut out).unwrap(), 0);
    }

    #[test]
    fn rewind_replays_from_the_start() {
        let mut source = MemorySource::mono(vec![1.0, 2.0], 32_000);
        let mut out = [0.0f32; 8];

        source.read(&mut out).unwrap();
        assert!(source.is_exhausted());

        source.rewind();
        assert!(!source.is_exhausted());
        assert_eq!(source.read(&mut out).unwrap(), 2);
    }

    #[test]
    fn mono_downmix_is_a_no_op() {
        let mut buffer = [1.0, 2.0, 3.0];
        assert_eq!(downmix_to_mono(&mut buffer, 1), 3);
        assert_eq!(buffer, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn stereo_downmix_averages_channel_pairs() {
        let mut buffer = [1.0, 3.0, 2.0, 4.0, 0.0, 10.0];
        let frames = downmix_to_mono(&mut buffer, 2);

        assert_eq!(frames, 3);
        assert_eq!(&buffer[..3], &[2.0, 3.0, 5.0]);
    }

    #[test]
    fn downmix_preserves_a_centred_signal() {
        // The same signal in both channels must survive downmix unchanged.
        let mut buffer = [0.5, 0.5, -0.25, -0.25];
        let frames = downmix_to_mono(&mut buffer, 2);

        assert_eq!(frames, 2);
        assert_eq!(&buffer[..2], &[0.5, -0.25]);
    }
}
