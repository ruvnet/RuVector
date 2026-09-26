//! Real-time streaming audio ingestion.
//!
//! The offline path decodes a whole file into a `Vec<f32>` before anything runs,
//! which cannot work for live audio. This module implements the pipeline ADR-010
//! specifies: a source abstraction that keeps device access optional, a
//! lock-free ring buffer with an explicit drop policy, incremental segmentation,
//! and fixed analysis windows sized for the embedding model.
//!
//! ```text
//! AudioSource --> RingBuffer --> StreamSegmenter --> AnalysisWindow --> features
//!  (rt thread)    (lock-free)    (EMA + hysteresis)   (5 s, 50% overlap)
//! ```
//!
//! # Example
//!
//! ```
//! use sevensense_audio::streaming::{MemorySource, StreamPipeline, StreamConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Half a second of silence, then a 4 kHz call, then silence.
//! let mut samples = vec![0.0f32; 16_000];
//! samples.extend((0..16_000).map(|i| {
//!     0.5 * (std::f32::consts::TAU * 4000.0 * i as f32 / 32_000.0).sin()
//! }));
//! samples.extend(vec![0.0f32; 32_000]);
//!
//! let source = MemorySource::mono(samples, 32_000);
//! let mut pipeline = StreamPipeline::new(source, StreamConfig::default())?;
//!
//! let windows = pipeline.run_to_completion()?;
//! assert_eq!(windows.len(), 1, "one call should yield one analysis window");
//! assert_eq!(windows[0].samples.len(), 160_000, "5 s at 32 kHz");
//! # Ok(())
//! # }
//! ```

mod ring;
mod segmenter;
mod source;

pub use ring::RingBuffer;
pub use segmenter::{SegmentEvent, StreamSegment, StreamSegmenter, StreamSegmenterConfig};
pub use source::{downmix_to_mono, AudioSource, MemorySource};

use crate::AudioError;

/// Samples in one analysis window: 5 seconds at 32 kHz, matching Perch 2.0.
pub const WINDOW_SAMPLES: usize = 160_000;

/// Configuration for [`StreamPipeline`].
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Seconds of audio the ring buffer holds before overwriting.
    pub ring_seconds: f32,
    /// Samples pulled from the source per iteration.
    pub read_chunk: usize,
    /// Length of an analysis window in samples.
    pub window_samples: usize,
    /// Hop between successive windows. Half the window length gives 50% overlap,
    /// so a call straddling one boundary lands near the centre of another.
    pub window_hop: usize,
    /// Segmentation parameters.
    pub segmenter: StreamSegmenterConfig,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            ring_seconds: 30.0,
            read_chunk: 4096,
            window_samples: WINDOW_SAMPLES,
            window_hop: WINDOW_SAMPLES / 2,
            segmenter: StreamSegmenterConfig::default(),
        }
    }
}

impl StreamConfig {
    fn validate(&self) -> Result<(), AudioError> {
        if self.read_chunk == 0 {
            return Err(AudioError::invalid_data("read_chunk must be non-zero"));
        }
        if self.window_samples == 0 {
            return Err(AudioError::invalid_data("window_samples must be non-zero"));
        }
        if self.window_hop == 0 {
            return Err(AudioError::invalid_data("window_hop must be non-zero"));
        }
        if self.ring_seconds <= 0.0 {
            return Err(AudioError::invalid_data("ring_seconds must be positive"));
        }
        Ok(())
    }
}

/// A fixed-length window of audio ready for embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisWindow {
    /// Stream-relative sample index where this window starts.
    pub start_sample: u64,
    /// Window audio, zero-padded to exactly `window_samples`.
    pub samples: Vec<f32>,
    /// Index of the segment this window came from, counted from stream start.
    pub segment_index: u64,
    /// Position of this window within its segment.
    pub window_index: usize,
    /// Root-mean-square energy of the source segment.
    pub rms_energy: f32,
    /// Signal-to-noise ratio of the source segment, in dB.
    pub snr_db: f32,
}

/// Throughput and loss counters for a running stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamStats {
    /// Samples read from the source.
    pub samples_read: u64,
    /// Samples lost to ring-buffer overwrite. Non-zero means the consumer fell
    /// behind and the audio has a gap.
    pub samples_dropped: u64,
    /// Segments emitted.
    pub segments: u64,
    /// Segments discarded for being too short.
    pub discarded: u64,
    /// Analysis windows produced.
    pub windows: u64,
}

/// Drives audio from a source through segmentation to analysis windows.
///
/// The pipeline is synchronous and pull-based: each [`StreamPipeline::poll`]
/// moves whatever audio is available and returns any windows that completed.
/// Callers that want it on its own thread can drive it from a loop; the ring
/// buffer between the source and the segmenter is what makes that safe.
pub struct StreamPipeline<S: AudioSource> {
    source: S,
    ring: RingBuffer,
    segmenter: StreamSegmenter,
    config: StreamConfig,
    scratch: Vec<f32>,
    stats: StreamStats,
    segment_counter: u64,
}

impl<S: AudioSource> StreamPipeline<S> {
    /// Creates a pipeline reading from `source`.
    ///
    /// # Errors
    /// Returns [`AudioError`] when the configuration is invalid.
    pub fn new(source: S, mut config: StreamConfig) -> Result<Self, AudioError> {
        config.validate()?;
        config.segmenter.sample_rate = source.sample_rate();

        let ring = RingBuffer::with_duration(config.ring_seconds, source.sample_rate());
        let segmenter = StreamSegmenter::new(config.segmenter.clone())?;
        let scratch = vec![0.0f32; config.read_chunk];

        Ok(Self {
            source,
            ring,
            segmenter,
            config,
            scratch,
            stats: StreamStats::default(),
            segment_counter: 0,
        })
    }

    /// Counters for the stream so far.
    #[must_use]
    pub fn stats(&self) -> StreamStats {
        self.stats
    }

    /// Current noise-floor estimate.
    #[must_use]
    pub fn noise_floor(&self) -> f32 {
        self.segmenter.noise_floor()
    }

    /// Whether the source is drained and no segment remains open.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.source.is_exhausted() && self.ring.available() == 0 && !self.segmenter.in_segment()
    }

    /// Moves one chunk of audio through the pipeline.
    ///
    /// Returns any analysis windows completed by this step, which is often none.
    ///
    /// # Errors
    /// Returns [`AudioError`] when the source fails.
    pub fn poll(&mut self) -> Result<Vec<AnalysisWindow>, AudioError> {
        // Source -> ring. A live device would do this from its own callback; the
        // ring is what decouples the two.
        let read = self.source.read(&mut self.scratch)?;
        if read > 0 {
            let mono = if self.source.channels() > 1 {
                downmix_to_mono(&mut self.scratch[..read], self.source.channels())
            } else {
                read
            };
            self.ring.write(&self.scratch[..mono]);
            self.stats.samples_read += mono as u64;
        }

        // Ring -> segmenter.
        let mut buffer = vec![0.0f32; self.config.read_chunk];
        let (count, dropped) = self.ring.read(&mut buffer);
        self.stats.samples_dropped += dropped;

        if count == 0 {
            return Ok(Vec::new());
        }

        let events = self.segmenter.push(&buffer[..count]);
        Ok(self.windows_from(events))
    }

    /// Closes any open segment and returns its windows.
    #[must_use]
    pub fn finish(&mut self) -> Vec<AnalysisWindow> {
        let events = self.segmenter.flush();
        self.windows_from(events)
    }

    /// Runs until the source is exhausted, collecting every window.
    ///
    /// Convenient for file replay and tests. A live stream never terminates, so
    /// it should be driven with [`StreamPipeline::poll`] instead.
    ///
    /// # Errors
    /// Returns [`AudioError`] when the source fails.
    pub fn run_to_completion(&mut self) -> Result<Vec<AnalysisWindow>, AudioError> {
        let mut windows = Vec::new();
        while !self.source.is_exhausted() || self.ring.available() > 0 {
            windows.extend(self.poll()?);
        }
        windows.extend(self.finish());
        Ok(windows)
    }

    /// Converts segmenter events into analysis windows, updating counters.
    fn windows_from(&mut self, events: Vec<SegmentEvent>) -> Vec<AnalysisWindow> {
        let mut windows = Vec::new();

        for event in events {
            match event {
                SegmentEvent::Closed(segment) => {
                    let index = self.segment_counter;
                    self.segment_counter += 1;
                    self.stats.segments += 1;
                    windows.extend(self.split_into_windows(&segment, index));
                }
                SegmentEvent::Discarded { .. } => self.stats.discarded += 1,
                SegmentEvent::Opened { .. } => {}
            }
        }

        self.stats.windows += windows.len() as u64;
        windows
    }

    /// Splits a segment into overlapping fixed-length windows.
    ///
    /// A segment shorter than one window yields a single zero-padded window
    /// rather than nothing, since a short call is still worth embedding.
    fn split_into_windows(&self, segment: &StreamSegment, index: u64) -> Vec<AnalysisWindow> {
        let window_len = self.config.window_samples;
        let hop = self.config.window_hop;
        let snr_db = segment.snr_db();

        let count = if segment.samples.len() <= window_len {
            1
        } else {
            (segment.samples.len() - window_len).div_ceil(hop) + 1
        };

        (0..count)
            .map(|i| {
                let offset = i * hop;
                let mut samples = vec![0.0f32; window_len];
                let available = segment.samples.len().saturating_sub(offset).min(window_len);
                if available > 0 {
                    samples[..available]
                        .copy_from_slice(&segment.samples[offset..offset + available]);
                }

                AnalysisWindow {
                    start_sample: segment.start_sample + offset as u64,
                    samples,
                    segment_index: index,
                    window_index: i,
                    rms_energy: segment.rms_energy,
                    snr_db,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 32_000;

    fn tone(freq: f32, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (std::f32::consts::TAU * freq * i as f32 / SR as f32).sin())
            .collect()
    }

    fn call_in_silence() -> Vec<f32> {
        let mut samples = vec![0.0f32; SR as usize / 2];
        samples.extend(tone(4000.0, SR as usize / 2, 0.5));
        samples.extend(vec![0.0f32; SR as usize]);
        samples
    }

    #[test]
    fn a_single_call_yields_one_padded_window() {
        let source = MemorySource::mono(call_in_silence(), SR);
        let mut pipeline = StreamPipeline::new(source, StreamConfig::default()).unwrap();

        let windows = pipeline.run_to_completion().unwrap();

        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].samples.len(), WINDOW_SAMPLES);
        assert_eq!(windows[0].window_index, 0);
        assert_eq!(windows[0].segment_index, 0);
    }

    #[test]
    fn silence_yields_no_windows() {
        let source = MemorySource::mono(vec![0.0; SR as usize * 2], SR);
        let mut pipeline = StreamPipeline::new(source, StreamConfig::default()).unwrap();

        assert!(pipeline.run_to_completion().unwrap().is_empty());
        assert_eq!(pipeline.stats().segments, 0);
    }

    #[test]
    fn a_long_call_splits_into_overlapping_windows() {
        // 12 s of tone against a 5 s window with a 2.5 s hop.
        let mut samples = vec![0.0f32; SR as usize / 2];
        samples.extend(tone(4000.0, SR as usize * 12, 0.5));
        samples.extend(vec![0.0f32; SR as usize]);

        let config = StreamConfig {
            segmenter: StreamSegmenterConfig {
                max_segment_ms: 20_000,
                ..Default::default()
            },
            ..Default::default()
        };
        let source = MemorySource::mono(samples, SR);
        let mut pipeline = StreamPipeline::new(source, config).unwrap();

        let windows = pipeline.run_to_completion().unwrap();

        assert!(windows.len() >= 4, "got {} windows", windows.len());
        assert!(windows.iter().all(|w| w.samples.len() == WINDOW_SAMPLES));

        // 50% overlap means consecutive starts differ by exactly the hop.
        let step = windows[1].start_sample - windows[0].start_sample;
        assert_eq!(step, (WINDOW_SAMPLES / 2) as u64);
    }

    #[test]
    fn window_contents_match_the_source_audio() {
        let source = MemorySource::mono(call_in_silence(), SR);
        let mut pipeline = StreamPipeline::new(source, StreamConfig::default()).unwrap();

        let windows = pipeline.run_to_completion().unwrap();
        let window = &windows[0];

        // The call occupies the first half of the window; the tail is padding.
        let front_energy: f32 = window.samples[..SR as usize / 2].iter().map(|s| s * s).sum();
        let tail_energy: f32 = window.samples[SR as usize..].iter().map(|s| s * s).sum();

        assert!(front_energy > 100.0, "call audio should be present");
        assert!(tail_energy < 1e-3, "tail should be zero padding");
    }

    #[test]
    fn stats_account_for_every_segment() {
        let mut samples = vec![0.0f32; SR as usize / 2];
        samples.extend(tone(4000.0, SR as usize / 2, 0.5));
        samples.extend(vec![0.0f32; SR as usize]);
        samples.extend(tone(6000.0, SR as usize / 2, 0.5));
        samples.extend(vec![0.0f32; SR as usize]);

        let source = MemorySource::mono(samples, SR);
        let mut pipeline = StreamPipeline::new(source, StreamConfig::default()).unwrap();
        let windows = pipeline.run_to_completion().unwrap();

        let stats = pipeline.stats();
        assert_eq!(stats.segments, 2);
        assert_eq!(stats.windows, windows.len() as u64);
        assert_eq!(stats.samples_dropped, 0, "consumer keeps pace here");
        assert!(stats.samples_read > 0);
    }

    #[test]
    fn segment_indices_increase_monotonically() {
        let mut samples = Vec::new();
        for freq in [3000.0, 5000.0, 7000.0] {
            samples.extend(vec![0.0f32; SR as usize / 2]);
            samples.extend(tone(freq, SR as usize / 2, 0.5));
        }
        samples.extend(vec![0.0f32; SR as usize]);

        let source = MemorySource::mono(samples, SR);
        let mut pipeline = StreamPipeline::new(source, StreamConfig::default()).unwrap();
        let windows = pipeline.run_to_completion().unwrap();

        let indices: Vec<u64> = windows.iter().map(|w| w.segment_index).collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn results_are_independent_of_source_chunk_size() {
        // A device delivers whatever chunk size it likes; output must not vary.
        let reference = {
            let source = MemorySource::new(call_in_silence(), SR, 1, 4096);
            let mut pipeline = StreamPipeline::new(source, StreamConfig::default()).unwrap();
            pipeline.run_to_completion().unwrap()
        };

        for chunk in [64, 333, 1024, 8192] {
            let source = MemorySource::new(call_in_silence(), SR, 1, chunk);
            let mut pipeline = StreamPipeline::new(source, StreamConfig::default()).unwrap();
            let windows = pipeline.run_to_completion().unwrap();

            assert_eq!(windows.len(), reference.len(), "chunk {chunk} changed count");
            assert_eq!(
                windows[0].start_sample, reference[0].start_sample,
                "chunk {chunk} shifted the segment start"
            );
        }
    }

    #[test]
    fn stereo_input_is_downmixed_to_mono() {
        // Identical content in both channels must survive as the same mono call.
        let mono = call_in_silence();
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();

        let source = MemorySource::new(stereo, SR, 2, 4096);
        let mut pipeline = StreamPipeline::new(source, StreamConfig::default()).unwrap();
        let windows = pipeline.run_to_completion().unwrap();

        assert_eq!(windows.len(), 1);
    }

    #[test]
    fn polling_reports_completion() {
        let source = MemorySource::mono(call_in_silence(), SR);
        let mut pipeline = StreamPipeline::new(source, StreamConfig::default()).unwrap();

        let mut guard = 0;
        while !pipeline.is_finished() && guard < 10_000 {
            pipeline.poll().unwrap();
            guard += 1;
        }

        assert!(pipeline.is_finished());
        assert!(guard < 10_000, "pipeline failed to terminate");
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        for config in [
            StreamConfig { read_chunk: 0, ..Default::default() },
            StreamConfig { window_samples: 0, ..Default::default() },
            StreamConfig { window_hop: 0, ..Default::default() },
            StreamConfig { ring_seconds: 0.0, ..Default::default() },
        ] {
            let source = MemorySource::mono(vec![0.0; 100], SR);
            assert!(StreamPipeline::new(source, config).is_err());
        }
    }

    #[test]
    fn the_pipeline_adopts_the_source_sample_rate() {
        let source = MemorySource::mono(vec![0.0; 1000], 48_000);
        let pipeline = StreamPipeline::new(source, StreamConfig::default()).unwrap();

        assert_eq!(pipeline.segmenter.config().sample_rate, 48_000);
    }
}
