//! Incremental segmentation for continuous audio.
//!
//! [`crate::infrastructure::EnergySegmenter`] estimates its noise floor from
//! global statistics over a complete recording, so it cannot emit a segment until
//! the recording ends. This segmenter replaces that with an exponential moving
//! average and open/close hysteresis, so a vocalization can be closed and handed
//! downstream while audio keeps arriving.
//!
//! Both are kept. Global statistics are genuinely more accurate when the whole
//! signal is available, and ADR-010 records the decision not to pretend one
//! supersedes the other.

use std::collections::VecDeque;

use crate::AudioError;

/// Configuration for [`StreamSegmenter`].
#[derive(Debug, Clone)]
pub struct StreamSegmenterConfig {
    /// Samples per analysis frame.
    pub frame_size: usize,
    /// Sample rate of the incoming audio, in Hz.
    pub sample_rate: u32,
    /// Time constant of the noise-floor average, in milliseconds.
    pub noise_time_constant_ms: f32,
    /// A segment opens when frame energy exceeds `noise_floor * open_ratio`.
    pub open_ratio: f32,
    /// A segment closes when frame energy falls below `noise_floor * close_ratio`.
    /// Keeping this below `open_ratio` gives hysteresis, which stops a segment
    /// flickering when energy sits near the threshold.
    pub close_ratio: f32,
    /// Consecutive frames above the open threshold required to open a segment.
    pub min_open_frames: usize,
    /// Milliseconds below the close threshold before a segment ends.
    pub hangover_ms: u64,
    /// Segments shorter than this are discarded as transients.
    pub min_segment_ms: u64,
    /// Segments are force-closed at this length, bounding latency and memory.
    pub max_segment_ms: u64,
    /// Frames of audio retained before the opening frame, so a segment is not
    /// clipped by the frames it took to cross the threshold.
    pub preroll_frames: usize,
    /// Noise floor assumed before any audio has been observed.
    pub initial_noise_floor: f32,
}

impl Default for StreamSegmenterConfig {
    fn default() -> Self {
        Self {
            frame_size: 512, // 16 ms at 32 kHz
            sample_rate: 32_000,
            noise_time_constant_ms: 2000.0,
            open_ratio: 3.0,
            close_ratio: 1.8,
            min_open_frames: 2,
            hangover_ms: 200,
            min_segment_ms: 120,
            max_segment_ms: 10_000,
            preroll_frames: 4,
            initial_noise_floor: 1e-4,
        }
    }
}

impl StreamSegmenterConfig {
    /// Duration of one analysis frame in milliseconds.
    #[must_use]
    pub fn frame_ms(&self) -> f32 {
        self.frame_size as f32 * 1000.0 / self.sample_rate as f32
    }

    fn validate(&self) -> Result<(), AudioError> {
        if self.frame_size == 0 {
            return Err(AudioError::invalid_data("frame_size must be non-zero"));
        }
        if self.sample_rate == 0 {
            return Err(AudioError::invalid_data("sample_rate must be non-zero"));
        }
        if self.close_ratio > self.open_ratio {
            return Err(AudioError::invalid_data(
                "close_ratio must not exceed open_ratio; hysteresis requires open >= close",
            ));
        }
        if self.min_segment_ms > self.max_segment_ms {
            return Err(AudioError::invalid_data(
                "min_segment_ms must not exceed max_segment_ms",
            ));
        }
        Ok(())
    }
}

/// A vocalization detected in a continuous stream.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamSegment {
    /// Sample index where the segment starts, counted from stream start.
    pub start_sample: u64,
    /// Sample index one past the end of the segment.
    pub end_sample: u64,
    /// The segment's audio, including pre-roll.
    pub samples: Vec<f32>,
    /// Largest absolute sample value in the segment.
    pub peak_amplitude: f32,
    /// Root-mean-square energy across the segment.
    pub rms_energy: f32,
    /// Noise floor at the moment the segment opened, for signal-to-noise context.
    pub noise_floor: f32,
    /// Whether the segment ended because it hit `max_segment_ms` rather than
    /// falling quiet. A truncated segment may continue in the next one.
    pub truncated: bool,
}

impl StreamSegment {
    /// Duration of the segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self, sample_rate: u32) -> u64 {
        (self.end_sample - self.start_sample) * 1000 / u64::from(sample_rate)
    }

    /// Signal-to-noise ratio in dB, relative to the noise floor when the segment
    /// opened.
    #[must_use]
    pub fn snr_db(&self) -> f32 {
        if self.noise_floor <= 0.0 || self.rms_energy <= 0.0 {
            return 0.0;
        }
        20.0 * (self.rms_energy / self.noise_floor).log10()
    }
}

/// What happened as a result of pushing samples.
#[derive(Debug, Clone, PartialEq)]
pub enum SegmentEvent {
    /// A segment began at the given stream-relative sample index.
    Opened {
        /// Sample index where the segment starts.
        start_sample: u64,
    },
    /// A segment completed and is ready for downstream analysis.
    Closed(Box<StreamSegment>),
    /// A candidate was discarded for being shorter than `min_segment_ms`.
    Discarded {
        /// How long the discarded candidate was.
        duration_ms: u64,
    },
}

/// Tracks whether a segment is currently open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    InSegment,
}

/// Segments a continuous audio stream incrementally.
///
/// Feed samples with [`StreamSegmenter::push`]; it returns any events the new
/// audio completed. Call [`StreamSegmenter::flush`] at end of stream to close a
/// segment still open.
#[derive(Debug)]
pub struct StreamSegmenter {
    config: StreamSegmenterConfig,
    state: State,
    noise_floor: f32,
    alpha: f32,
    /// Partial frame carried between pushes.
    pending: Vec<f32>,
    /// Recent frames kept as pre-roll while idle.
    preroll: VecDeque<Vec<f32>>,
    /// Audio accumulated for the segment currently open.
    current: Vec<f32>,
    /// Consecutive frames above the open threshold.
    frames_above: usize,
    /// Consecutive frames below the close threshold.
    frames_below: usize,
    /// Total samples consumed since stream start.
    samples_seen: u64,
    /// Stream-relative index where the open segment began.
    segment_start: u64,
    /// Noise floor captured when the segment opened.
    segment_noise_floor: f32,
    /// Whether any frame has been observed, for noise-floor initialization.
    warmed_up: bool,
}

impl StreamSegmenter {
    /// Creates a segmenter.
    ///
    /// # Errors
    /// Returns [`AudioError`] when the configuration is inconsistent.
    pub fn new(config: StreamSegmenterConfig) -> Result<Self, AudioError> {
        config.validate()?;

        // Convert the time constant to a per-frame smoothing coefficient.
        let frames_per_tc = (config.noise_time_constant_ms / config.frame_ms()).max(1.0);
        let alpha = (-1.0 / frames_per_tc).exp();

        Ok(Self {
            noise_floor: config.initial_noise_floor,
            alpha,
            pending: Vec::with_capacity(config.frame_size),
            preroll: VecDeque::with_capacity(config.preroll_frames + 1),
            current: Vec::new(),
            state: State::Idle,
            frames_above: 0,
            frames_below: 0,
            samples_seen: 0,
            segment_start: 0,
            segment_noise_floor: config.initial_noise_floor,
            warmed_up: false,
            config,
        })
    }

    /// Creates a segmenter with default configuration for the given sample rate.
    ///
    /// # Errors
    /// Returns [`AudioError`] when `sample_rate` is zero.
    pub fn for_sample_rate(sample_rate: u32) -> Result<Self, AudioError> {
        Self::new(StreamSegmenterConfig {
            sample_rate,
            ..Default::default()
        })
    }

    /// The configuration in use.
    #[must_use]
    pub fn config(&self) -> &StreamSegmenterConfig {
        &self.config
    }

    /// Current noise-floor estimate, in RMS amplitude.
    #[must_use]
    pub fn noise_floor(&self) -> f32 {
        self.noise_floor
    }

    /// Whether a segment is currently open.
    #[must_use]
    pub fn in_segment(&self) -> bool {
        self.state == State::InSegment
    }

    /// Total samples consumed since stream start.
    #[must_use]
    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Feeds samples into the segmenter, returning any completed events.
    pub fn push(&mut self, samples: &[f32]) -> Vec<SegmentEvent> {
        let mut events = Vec::new();
        let frame_size = self.config.frame_size;

        for &sample in samples {
            self.pending.push(sample);
            if self.pending.len() == frame_size {
                let frame = std::mem::replace(&mut self.pending, Vec::with_capacity(frame_size));
                self.process_frame(frame, &mut events);
            }
        }

        events
    }

    /// Closes any open segment at end of stream.
    ///
    /// Discards a partial frame: fewer than `frame_size` samples cannot be
    /// assessed against the threshold.
    pub fn flush(&mut self) -> Vec<SegmentEvent> {
        let mut events = Vec::new();
        if self.state == State::InSegment {
            self.close_segment(false, &mut events);
        }
        self.pending.clear();
        events
    }

    /// Applies one complete frame to the state machine.
    fn process_frame(&mut self, frame: Vec<f32>, events: &mut Vec<SegmentEvent>) {
        let rms = rms(&frame);
        self.samples_seen += frame.len() as u64;

        // Seed from the first frame rather than drifting up from a guess.
        if !self.warmed_up {
            self.noise_floor = rms.max(self.config.initial_noise_floor);
            self.warmed_up = true;
        }

        match self.state {
            State::Idle => self.process_idle(frame, rms, events),
            State::InSegment => self.process_in_segment(frame, rms, events),
        }
    }

    fn process_idle(&mut self, frame: Vec<f32>, rms: f32, events: &mut Vec<SegmentEvent>) {
        let open_threshold = self.noise_floor * self.config.open_ratio;

        if rms > open_threshold {
            self.frames_above += 1;
            self.preroll.push_back(frame);
            while self.preroll.len() > self.config.preroll_frames + self.frames_above {
                self.preroll.pop_front();
            }

            if self.frames_above >= self.config.min_open_frames {
                // Pre-roll rewinds the start so the opening transient survives.
                let preroll_samples: usize = self.preroll.iter().map(Vec::len).sum();
                self.segment_start = self.samples_seen.saturating_sub(preroll_samples as u64);
                self.segment_noise_floor = self.noise_floor;
                self.current = self.preroll.drain(..).flatten().collect();
                self.state = State::InSegment;
                self.frames_below = 0;
                events.push(SegmentEvent::Opened {
                    start_sample: self.segment_start,
                });
            }
        } else {
            self.frames_above = 0;
            // Only quiet frames update the floor; otherwise a long call would
            // drag the threshold up behind itself and close its own segment.
            self.noise_floor = self.alpha * self.noise_floor + (1.0 - self.alpha) * rms;
            self.noise_floor = self.noise_floor.max(self.config.initial_noise_floor);

            self.preroll.push_back(frame);
            while self.preroll.len() > self.config.preroll_frames {
                self.preroll.pop_front();
            }
        }
    }

    fn process_in_segment(&mut self, frame: Vec<f32>, rms: f32, events: &mut Vec<SegmentEvent>) {
        self.current.extend_from_slice(&frame);

        let close_threshold = self.noise_floor * self.config.close_ratio;
        let hangover_frames =
            (self.config.hangover_ms as f32 / self.config.frame_ms()).ceil() as usize;

        if rms < close_threshold {
            self.frames_below += 1;
        } else {
            self.frames_below = 0;
        }

        let duration_ms = self.current.len() as u64 * 1000 / u64::from(self.config.sample_rate);
        if duration_ms >= self.config.max_segment_ms {
            self.close_segment(true, events);
        } else if self.frames_below >= hangover_frames.max(1) {
            self.close_segment(false, events);
        }
    }

    /// Ends the open segment, emitting it if it is long enough to be useful.
    fn close_segment(&mut self, truncated: bool, events: &mut Vec<SegmentEvent>) {
        let mut samples = std::mem::take(&mut self.current);

        // Drop the hangover we waited out. Those frames are below the close
        // threshold by definition, so keeping them would pad every segment with
        // silence and let a brief transient masquerade as a long call. One frame
        // is retained so a natural decay is not clipped.
        if !truncated {
            let trim = self
                .frames_below
                .saturating_sub(1)
                .saturating_mul(self.config.frame_size);
            samples.truncate(samples.len().saturating_sub(trim));
        }

        self.state = State::Idle;
        self.frames_above = 0;
        self.frames_below = 0;
        self.preroll.clear();

        let duration_ms = samples.len() as u64 * 1000 / u64::from(self.config.sample_rate);
        if duration_ms < self.config.min_segment_ms {
            events.push(SegmentEvent::Discarded { duration_ms });
            return;
        }

        let peak_amplitude = samples.iter().fold(0.0f32, |a, s| a.max(s.abs()));
        let rms_energy = rms(&samples);
        let start_sample = self.segment_start;

        events.push(SegmentEvent::Closed(Box::new(StreamSegment {
            start_sample,
            end_sample: start_sample + samples.len() as u64,
            samples,
            peak_amplitude,
            rms_energy,
            noise_floor: self.segment_noise_floor,
            truncated,
        })));
    }
}

/// Root-mean-square amplitude of a block.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
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

    fn quiet(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    fn closed(events: &[SegmentEvent]) -> Vec<&StreamSegment> {
        events
            .iter()
            .filter_map(|e| match e {
                SegmentEvent::Closed(s) => Some(s.as_ref()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn silence_produces_no_segments() {
        let mut segmenter = StreamSegmenter::for_sample_rate(SR).unwrap();
        let events = segmenter.push(&quiet(SR as usize));

        assert!(events.is_empty());
        assert!(!segmenter.in_segment());
    }

    #[test]
    fn a_burst_between_silences_yields_one_segment() {
        let mut segmenter = StreamSegmenter::for_sample_rate(SR).unwrap();

        let mut events = segmenter.push(&quiet(SR as usize / 2));
        events.extend(segmenter.push(&tone(4000.0, SR as usize / 2, 0.5)));
        events.extend(segmenter.push(&quiet(SR as usize)));

        let segments = closed(&events);
        assert_eq!(segments.len(), 1, "expected exactly one segment");
        assert!(!segments[0].truncated);
        assert!(segments[0].peak_amplitude > 0.4);
    }

    #[test]
    fn two_bursts_separated_by_silence_yield_two_segments() {
        let mut segmenter = StreamSegmenter::for_sample_rate(SR).unwrap();

        let mut events = segmenter.push(&quiet(SR as usize / 2));
        events.extend(segmenter.push(&tone(4000.0, SR as usize / 2, 0.5)));
        events.extend(segmenter.push(&quiet(SR as usize)));
        events.extend(segmenter.push(&tone(6000.0, SR as usize / 2, 0.5)));
        events.extend(segmenter.push(&quiet(SR as usize)));

        assert_eq!(closed(&events).len(), 2);
    }

    #[test]
    fn a_segment_closes_before_the_stream_ends() {
        // The point of incremental segmentation: no flush required.
        let mut segmenter = StreamSegmenter::for_sample_rate(SR).unwrap();

        let mut events = segmenter.push(&quiet(SR as usize / 2));
        events.extend(segmenter.push(&tone(4000.0, SR as usize / 2, 0.5)));
        events.extend(segmenter.push(&quiet(SR as usize)));

        assert_eq!(
            closed(&events).len(),
            1,
            "segment must be emitted from push, not deferred to flush"
        );
    }

    #[test]
    fn flush_closes_a_segment_still_open() {
        let mut segmenter = StreamSegmenter::for_sample_rate(SR).unwrap();

        let mut events = segmenter.push(&quiet(SR as usize / 4));
        events.extend(segmenter.push(&tone(4000.0, SR as usize, 0.5)));
        assert!(segmenter.in_segment());

        events.extend(segmenter.flush());
        assert_eq!(closed(&events).len(), 1);
        assert!(!segmenter.in_segment());
    }

    #[test]
    fn segments_are_truncated_at_the_configured_maximum() {
        let config = StreamSegmenterConfig {
            sample_rate: SR,
            max_segment_ms: 500,
            ..Default::default()
        };
        let mut segmenter = StreamSegmenter::new(config).unwrap();

        let mut events = segmenter.push(&quiet(SR as usize / 4));
        events.extend(segmenter.push(&tone(4000.0, SR as usize * 2, 0.5)));

        let segments = closed(&events);
        assert!(segments.len() >= 3, "2 s of tone at 500 ms should split");
        assert!(
            segments[0].truncated,
            "a segment cut at the maximum must be marked truncated"
        );
        assert!(segments[0].duration_ms(SR) <= 520);
    }

    #[test]
    fn brief_transients_are_discarded() {
        let config = StreamSegmenterConfig {
            sample_rate: SR,
            min_segment_ms: 300,
            min_open_frames: 1,
            ..Default::default()
        };
        let mut segmenter = StreamSegmenter::new(config).unwrap();

        let mut events = segmenter.push(&quiet(SR as usize / 4));
        // 30 ms of tone, well under the 300 ms minimum.
        events.extend(segmenter.push(&tone(4000.0, SR as usize / 33, 0.5)));
        events.extend(segmenter.push(&quiet(SR as usize)));

        assert!(closed(&events).is_empty(), "transient should not survive");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, SegmentEvent::Discarded { .. })),
            "a discarded candidate should be reported, not dropped silently"
        );
    }

    #[test]
    fn the_hangover_silence_is_trimmed_from_the_segment() {
        let mut segmenter = StreamSegmenter::for_sample_rate(SR).unwrap();
        let call_samples = SR as usize / 2;

        let mut events = segmenter.push(&quiet(SR as usize / 2));
        events.extend(segmenter.push(&tone(4000.0, call_samples, 0.5)));
        events.extend(segmenter.push(&quiet(SR as usize)));

        let segment = closed(&events)[0];
        let hangover_samples =
            segmenter.config().hangover_ms as usize * SR as usize / 1000;

        // The segment should be roughly the call plus pre-roll, not the call plus
        // the whole hangover we waited out before closing.
        assert!(
            segment.samples.len() < call_samples + hangover_samples,
            "segment of {} samples still carries the {hangover_samples}-sample hangover",
            segment.samples.len()
        );
        assert!(
            segment.samples.len() >= call_samples,
            "segment of {} samples lost part of the {call_samples}-sample call",
            segment.samples.len()
        );

        // Trailing silence is bounded by frame quantization plus the one decay
        // frame kept by design -- roughly two frames, never the full hangover.
        let frame = segmenter.config().frame_size;
        let last_audible = segment
            .samples
            .iter()
            .rposition(|s| s.abs() > 0.01)
            .expect("segment must contain audio");
        let trailing_silence = segment.samples.len() - last_audible - 1;
        assert!(
            trailing_silence <= 2 * frame,
            "{trailing_silence} samples of trailing silence exceeds the 2-frame bound"
        );
    }

    #[test]
    fn preroll_keeps_the_opening_transient() {
        let mut segmenter = StreamSegmenter::for_sample_rate(SR).unwrap();

        let mut events = segmenter.push(&quiet(SR as usize / 2));
        let onset = segmenter.samples_seen();
        events.extend(segmenter.push(&tone(4000.0, SR as usize / 2, 0.5)));
        events.extend(segmenter.push(&quiet(SR as usize)));

        let segments = closed(&events);
        assert!(
            segments[0].start_sample < onset,
            "segment should start before the onset frame at {onset}, got {}",
            segments[0].start_sample
        );
    }

    #[test]
    fn noise_floor_adapts_to_a_noisy_background() {
        let mut segmenter = StreamSegmenter::for_sample_rate(SR).unwrap();
        let initial = segmenter.noise_floor();

        // Steady low-level hiss should raise the floor.
        let hiss: Vec<f32> = (0..SR as usize)
            .map(|i| 0.01 * ((i * 7919) % 17) as f32 / 17.0)
            .collect();
        segmenter.push(&hiss);

        assert!(
            segmenter.noise_floor() > initial,
            "floor {} should have risen above {initial}",
            segmenter.noise_floor()
        );
    }

    #[test]
    fn a_loud_background_suppresses_a_quiet_signal() {
        // The same tone that segments cleanly against silence must not segment
        // when it sits below the adapted floor.
        let mut segmenter = StreamSegmenter::for_sample_rate(SR).unwrap();
        let loud_noise: Vec<f32> = (0..SR as usize * 2)
            .map(|i| 0.3 * (((i * 7919) % 1021) as f32 / 1021.0 - 0.5))
            .collect();

        segmenter.push(&loud_noise);
        let events = segmenter.push(&tone(4000.0, SR as usize / 2, 0.02));

        assert!(
            closed(&events).is_empty(),
            "a signal below the adapted floor should not open a segment"
        );
    }

    #[test]
    fn results_do_not_depend_on_push_chunking() {
        // A live device delivers arbitrary chunk sizes; segmentation must not care.
        let mut signal = quiet(SR as usize / 2);
        signal.extend(tone(4000.0, SR as usize / 2, 0.5));
        signal.extend(quiet(SR as usize));

        let mut one_shot = StreamSegmenter::for_sample_rate(SR).unwrap();
        let bulk = one_shot.push(&signal);

        let mut chunked = StreamSegmenter::for_sample_rate(SR).unwrap();
        let mut incremental = Vec::new();
        for chunk in signal.chunks(333) {
            incremental.extend(chunked.push(chunk));
        }

        let bulk_segments = closed(&bulk);
        let chunk_segments = closed(&incremental);
        assert_eq!(bulk_segments.len(), chunk_segments.len());
        assert_eq!(bulk_segments[0].start_sample, chunk_segments[0].start_sample);
        assert_eq!(bulk_segments[0].end_sample, chunk_segments[0].end_sample);
    }

    #[test]
    fn snr_is_positive_for_a_clear_signal() {
        let mut segmenter = StreamSegmenter::for_sample_rate(SR).unwrap();

        let mut events = segmenter.push(&quiet(SR as usize / 2));
        events.extend(segmenter.push(&tone(4000.0, SR as usize / 2, 0.5)));
        events.extend(segmenter.push(&quiet(SR as usize)));

        assert!(closed(&events)[0].snr_db() > 20.0);
    }

    #[test]
    fn inverted_hysteresis_is_rejected() {
        let config = StreamSegmenterConfig {
            open_ratio: 1.5,
            close_ratio: 3.0,
            ..Default::default()
        };
        assert!(StreamSegmenter::new(config).is_err());
    }

    #[test]
    fn degenerate_configuration_is_rejected() {
        for config in [
            StreamSegmenterConfig { frame_size: 0, ..Default::default() },
            StreamSegmenterConfig { sample_rate: 0, ..Default::default() },
            StreamSegmenterConfig { min_segment_ms: 5000, max_segment_ms: 1000, ..Default::default() },
        ] {
            assert!(StreamSegmenter::new(config).is_err());
        }
    }
}
