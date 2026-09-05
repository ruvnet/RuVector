// SPDX-License-Identifier: MIT
//
// 26-second pulse detector. Uses a small naive DFT scan over the
// ~0.035-0.042 Hz band (~24-28s) to find the dominant spectral peak,
// then derives amplitude, phase-coherence, glide and confidence.
// Pure & deterministic: no Date.now, no Math.random.

import type { PulseEvent, SeismicWindow, PeriodBand } from './types.js';

const DEFAULT_LOW_PERIOD_S = 24;
const DEFAULT_HIGH_PERIOD_S = 28;
const DEFAULT_MIN_COHERENCE = 0.15;

/** Number of frequency bins scanned across the band. */
const SCAN_BINS = 48;
const TWO_PI = Math.PI * 2;

interface DftBin {
  freqHz: number;
  re: number;
  im: number;
  magnitude: number;
}

/**
 * Remove the DC component (mean) so the peak reflects oscillatory energy.
 */
function removeMean(samples: number[]): number[] {
  if (samples.length === 0) return [];
  let sum = 0;
  for (const s of samples) sum += s;
  const mean = sum / samples.length;
  return samples.map((s) => s - mean);
}

/**
 * Evaluate a single DFT bin at `freqHz` over `samples` sampled at `rateHz`.
 */
function dftBin(samples: number[], rateHz: number, freqHz: number): DftBin {
  let re = 0;
  let im = 0;
  const n = samples.length;
  for (let i = 0; i < n; i++) {
    const t = i / rateHz;
    const angle = TWO_PI * freqHz * t;
    re += samples[i] * Math.cos(angle);
    im -= samples[i] * Math.sin(angle);
  }
  // Normalize by N for a stable, length-independent magnitude.
  re /= n;
  im /= n;
  const magnitude = Math.sqrt(re * re + im * im);
  return { freqHz, re, im, magnitude };
}

/**
 * Scan the band and return all evaluated bins (ascending frequency).
 */
function scanBand(samples: number[], rateHz: number, band: PeriodBand): DftBin[] {
  const lowHz = 1 / band.highPeriodS; // longest period => lowest frequency
  const highHz = 1 / band.lowPeriodS;
  const bins: DftBin[] = [];
  const step = (highHz - lowHz) / (SCAN_BINS - 1);
  for (let b = 0; b < SCAN_BINS; b++) {
    const f = lowHz + step * b;
    bins.push(dftBin(samples, rateHz, f));
  }
  return bins;
}

function peakBin(bins: DftBin[]): DftBin {
  let best = bins[0];
  for (const bin of bins) {
    if (bin.magnitude > best.magnitude) best = bin;
  }
  return best;
}

/**
 * Noise-floor reference: the minimum bin magnitude across the scanned band.
 *
 * The naive DFT bins inside this narrow band are non-orthogonal, so a strong
 * in-band peak leaks energy into every nearby bin (inflating a median/mean
 * floor). The off-peak *minimum* better reflects the residual non-peak energy,
 * giving a meaningful peak-to-floor SNR.
 */
function noiseFloor(bins: DftBin[]): number {
  if (bins.length === 0) return 0;
  let min = bins[0].magnitude;
  for (const bin of bins) {
    if (bin.magnitude < min) min = bin.magnitude;
  }
  return min;
}

/**
 * Find the dominant period (in seconds) within `band` for the given samples.
 * Returns 0 if there is no signal.
 */
export function dominantPeriod(
  samples: number[],
  sampleRateHz: number,
  band: PeriodBand,
): number {
  if (samples.length < 4 || sampleRateHz <= 0) return 0;
  const centered = removeMean(samples);
  const bins = scanBand(centered, sampleRateHz, band);
  const peak = peakBin(bins);
  if (peak.magnitude <= 0 || peak.freqHz <= 0) return 0;
  return 1 / peak.freqHz;
}

/**
 * Phase-coherence proxy: compare the complex phase of the dominant bin in
 * the first vs second half of the window. Coherent (stationary) signals
 * keep a near-constant phase advance; noise scrambles it. Returns [0,1].
 */
function phaseCoherenceProxy(
  samples: number[],
  rateHz: number,
  freqHz: number,
): number {
  const half = Math.floor(samples.length / 2);
  if (half < 2) return 0;
  const a = dftBin(samples.slice(0, half), rateHz, freqHz);
  const b = dftBin(samples.slice(half), rateHz, freqHz);
  if (a.magnitude === 0 || b.magnitude === 0) return 0;
  const phaseA = Math.atan2(a.im, a.re);
  const phaseB = Math.atan2(b.im, b.re);
  // Expected phase advance over `half` samples at this frequency.
  const expected = -TWO_PI * freqHz * (half / rateHz);
  let delta = phaseB - phaseA - expected;
  // Wrap to (-pi, pi].
  while (delta > Math.PI) delta -= TWO_PI;
  while (delta <= -Math.PI) delta += TWO_PI;
  // Map |delta| in [0,pi] to coherence in [0,1].
  const phaseScore = 1 - Math.abs(delta) / Math.PI;
  // Amplitude stability also indicates coherence.
  const ampRatio =
    Math.min(a.magnitude, b.magnitude) / Math.max(a.magnitude, b.magnitude);
  return Math.max(0, Math.min(1, 0.6 * phaseScore + 0.4 * ampRatio));
}

/**
 * Glide heuristic: detect a frequency drift between the two halves of the
 * window. Real gliding tremors shift their dominant frequency over time.
 * Returns the per-half dominant frequencies and a glide flag.
 */
function detectGlide(
  samples: number[],
  rateHz: number,
  band: PeriodBand,
): { glideDetected: boolean; driftHz: number } {
  const half = Math.floor(samples.length / 2);
  if (half < 4) return { glideDetected: false, driftHz: 0 };
  const first = removeMean(samples.slice(0, half));
  const second = removeMean(samples.slice(half));
  const f1 = peakBin(scanBand(first, rateHz, band)).freqHz;
  const f2 = peakBin(scanBand(second, rateHz, band)).freqHz;
  const driftHz = f2 - f1;
  // A meaningful glide is a drift larger than one scan bin width.
  const lowHz = 1 / band.highPeriodS;
  const highHz = 1 / band.lowPeriodS;
  const binWidth = (highHz - lowHz) / (SCAN_BINS - 1);
  const glideDetected = Math.abs(driftHz) > binWidth * 1.5;
  return { glideDetected, driftHz };
}

/**
 * Detect a 26-second pulse in a seismic window.
 * Returns null when no in-band peak rises sufficiently above the noise floor
 * or when phase coherence is below the configured minimum.
 */
export function detectPulse(
  window: SeismicWindow,
  opts?: { lowPeriodS?: number; highPeriodS?: number; minCoherence?: number },
): PulseEvent | null {
  const band: PeriodBand = {
    lowPeriodS: opts?.lowPeriodS ?? DEFAULT_LOW_PERIOD_S,
    highPeriodS: opts?.highPeriodS ?? DEFAULT_HIGH_PERIOD_S,
  };
  const minCoherence = opts?.minCoherence ?? DEFAULT_MIN_COHERENCE;
  const { samples, sampleRateHz } = window;
  if (samples.length < 8 || sampleRateHz <= 0) return null;

  const centered = removeMean(samples);
  const bins = scanBand(centered, sampleRateHz, band);
  const peak = peakBin(bins);
  const floor = noiseFloor(bins);
  if (peak.magnitude <= 0) return null;

  // Signal-to-noise: peak relative to the median bin magnitude.
  const snr = floor > 0 ? peak.magnitude / floor : peak.magnitude;
  // Require the peak to clearly exceed the noise floor.
  if (snr < 1.5) return null;

  const frequencyHz = peak.freqHz;
  const dominantPeriodS = 1 / frequencyHz;
  const amplitude = peak.magnitude * 2; // single-sided amplitude estimate
  const phaseCoherence = phaseCoherenceProxy(centered, sampleRateHz, frequencyHz);
  if (phaseCoherence < minCoherence) return null;

  const { glideDetected } = detectGlide(samples, sampleRateHz, band);

  // Confidence blends SNR (saturating), coherence, and band-centeredness.
  const snrScore = Math.min(1, (snr - 1.5) / 6.5); // snr 1.5->0, 8+->1
  const bandCenterS = (band.lowPeriodS + band.highPeriodS) / 2;
  const bandHalfWidth = (band.highPeriodS - band.lowPeriodS) / 2;
  const centerScore =
    bandHalfWidth > 0
      ? Math.max(0, 1 - Math.abs(dominantPeriodS - bandCenterS) / bandHalfWidth)
      : 1;
  const confidence = Math.max(
    0,
    Math.min(1, 0.45 * snrScore + 0.4 * phaseCoherence + 0.15 * centerScore),
  );

  return {
    eventId: makeEventId(window, frequencyHz),
    timestampIso: window.startIso,
    dominantPeriodS,
    frequencyHz,
    amplitude,
    phaseCoherence,
    glideDetected,
    sourceRegion: 'unknown',
    confidence,
  };
}

/**
 * Deterministic event id from station, start time and frequency.
 * No randomness, no clock — same inputs always yield the same id.
 */
function makeEventId(window: SeismicWindow, freqHz: number): string {
  const freqTag = Math.round(freqHz * 1e5);
  const stamp = window.startIso.replace(/[^0-9]/g, '');
  return `evt-${window.stationId}-${stamp}-${freqTag}`;
}
