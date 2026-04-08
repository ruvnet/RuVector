//! Pre-built pipeline configurations for common hearing aid use-cases.

use super::*;
use crate::hearing_aid::Audiogram;

/// Compute an approximate insertion gain from an audiogram using the half-gain rule.
fn mid_gain_db(audiogram: &Audiogram) -> f32 {
    let loss_1k = audiogram.gain_at(1000.0) as f32;
    loss_1k * 0.5
}

/// Standard hearing aid: high-pass prefilter -> WDRC -> gain -> limiter.
pub fn standard_hearing_aid(
    audiogram: &Audiogram,
    sample_rate: f32,
    block_size: usize,
) -> Pipeline {
    let mut pipeline = Pipeline::new(sample_rate, block_size);
    pipeline.add(Box::new(BiquadFilter::new(FilterType::HighPass, 100.0, 0.707)));
    pipeline.add(Box::new(WDRCompressor::new(-30.0, 2.0)));
    pipeline.add(Box::new(GainProcessor::new(mid_gain_db(audiogram))));
    pipeline.add(Box::new(Limiter::new(-1.0)));
    pipeline.prepare();
    pipeline
}

/// Speech-in-noise: prefilter -> feedback cancel -> graph separator -> WDRC -> gain -> limiter.
pub fn speech_in_noise(
    audiogram: &Audiogram,
    sample_rate: f32,
    block_size: usize,
) -> Pipeline {
    let mut pipeline = Pipeline::new(sample_rate, block_size);
    pipeline.add(Box::new(BiquadFilter::new(FilterType::HighPass, 100.0, 0.707)));
    pipeline.add(Box::new(FeedbackCanceller::new(128, 0.01)));
    pipeline.add(Box::new(GraphSeparatorBlock::new()));
    pipeline.add(Box::new(WDRCompressor::new(-30.0, 2.0)));
    pipeline.add(Box::new(GainProcessor::new(mid_gain_db(audiogram))));
    pipeline.add(Box::new(Limiter::new(-1.0)));
    pipeline.prepare();
    pipeline
}

/// Music mode: gentle wideband compression -> gain -> limiter.
pub fn music_mode(
    audiogram: &Audiogram,
    sample_rate: f32,
    block_size: usize,
) -> Pipeline {
    let mut pipeline = Pipeline::new(sample_rate, block_size);
    pipeline.add(Box::new(WDRCompressor::new(-15.0, 1.5)));
    let gain = mid_gain_db(audiogram) * 0.75;
    pipeline.add(Box::new(GainProcessor::new(gain)));
    pipeline.add(Box::new(Limiter::new(-0.5)));
    pipeline.prepare();
    pipeline
}

/// Maximum clarity: all blocks including feedback cancel and graph separation.
pub fn maximum_clarity(
    audiogram: &Audiogram,
    sample_rate: f32,
    block_size: usize,
) -> Pipeline {
    let mut pipeline = Pipeline::new(sample_rate, block_size);
    pipeline.add(Box::new(BiquadFilter::new(FilterType::HighPass, 80.0, 0.707)));
    pipeline.add(Box::new(FeedbackCanceller::new(256, 0.005)));
    pipeline.add(Box::new(GraphSeparatorBlock::new()));
    pipeline.add(Box::new(DelayLine::new(2.0)));
    pipeline.add(Box::new(WDRCompressor::new(-35.0, 4.0)));
    pipeline.add(Box::new(GainProcessor::new(mid_gain_db(audiogram) * 1.2)));
    pipeline.add(Box::new(Mixer::unity()));
    pipeline.add(Box::new(Limiter::new(-1.0)));
    pipeline.prepare();
    pipeline
}
