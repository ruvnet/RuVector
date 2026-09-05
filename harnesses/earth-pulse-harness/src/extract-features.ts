// SPDX-License-Identifier: MIT
//
// Feature extraction: turn a PulseEvent + its window + env into a numeric
// FeatureVector. Deterministic; depends only on inputs.

import type {
  EnvironmentContext,
  FeatureVector,
  PulseEvent,
  SeismicWindow,
} from './types.js';

const TWO_PI = Math.PI * 2;

/** Sub-band centers (seconds) around the 26s pulse for the spectral vector. */
const SPECTRAL_PERIODS_S = [22, 24, 26, 28, 30];

/** Length of the downsampled amplitude-envelope vector. */
const ENVELOPE_BINS = 8;

function removeMean(samples: number[]): number[] {
  if (samples.length === 0) return [];
  let sum = 0;
  for (const s of samples) sum += s;
  const mean = sum / samples.length;
  return samples.map((s) => s - mean);
}

/**
 * Goertzel-style single-frequency power estimate at `freqHz`.
 */
function bandPower(samples: number[], rateHz: number, freqHz: number): number {
  let re = 0;
  let im = 0;
  const n = samples.length;
  for (let i = 0; i < n; i++) {
    const angle = (TWO_PI * freqHz * i) / rateHz;
    re += samples[i] * Math.cos(angle);
    im -= samples[i] * Math.sin(angle);
  }
  re /= n;
  im /= n;
  return re * re + im * im; // power
}

/**
 * Spectral feature vector: power in sub-bands around 26s, L1-normalized so it
 * describes spectral *shape* independent of absolute amplitude.
 */
function spectralFeatures(samples: number[], rateHz: number): number[] {
  const centered = removeMean(samples);
  const powers = SPECTRAL_PERIODS_S.map((p) => bandPower(centered, rateHz, 1 / p));
  const total = powers.reduce((a, b) => a + b, 0);
  if (total <= 0) return powers.map(() => 0);
  return powers.map((p) => p / total);
}

/**
 * Downsampled rectified amplitude envelope. The window is divided into
 * `ENVELOPE_BINS` segments; each value is the mean absolute deviation in that
 * segment. Captures amplitude modulation over time.
 */
function amplitudeEnvelope(samples: number[], rateHz: number): number[] {
  void rateHz;
  const centered = removeMean(samples);
  const n = centered.length;
  const out: number[] = [];
  const segLen = Math.max(1, Math.floor(n / ENVELOPE_BINS));
  for (let b = 0; b < ENVELOPE_BINS; b++) {
    const start = b * segLen;
    const end = b === ENVELOPE_BINS - 1 ? n : Math.min(n, start + segLen);
    let acc = 0;
    let count = 0;
    for (let i = start; i < end; i++) {
      acc += Math.abs(centered[i]);
      count++;
    }
    out.push(count > 0 ? acc / count : 0);
  }
  return out;
}

/**
 * Glide slope: dominant-frequency drift per second across the window halves.
 * Positive = upward glide. Hz/s.
 */
function glideSlope(samples: number[], rateHz: number, event: PulseEvent): number {
  const half = Math.floor(samples.length / 2);
  if (half < 4) return 0;
  const f = event.frequencyHz;
  // Probe a small neighborhood around the detected frequency in each half.
  const probe = (seg: number[]): number => {
    const c = removeMean(seg);
    let bestF = f;
    let bestP = -1;
    for (let k = -3; k <= 3; k++) {
      const cand = f * (1 + k * 0.02);
      if (cand <= 0) continue;
      const p = bandPower(c, rateHz, cand);
      if (p > bestP) {
        bestP = p;
        bestF = cand;
      }
    }
    return bestF;
  };
  const f1 = probe(samples.slice(0, half));
  const f2 = probe(samples.slice(half));
  const dtSeconds = half / rateHz;
  return dtSeconds > 0 ? (f2 - f1) / dtSeconds : 0;
}

/**
 * Station-geometry placeholder vector derived deterministically from the
 * station id and source region. A real harness would compute back-azimuth /
 * array response here; this keeps the contract stable and deterministic.
 */
function stationGeometry(
  stationId: string,
  env: EnvironmentContext,
): number[] {
  // Cheap deterministic hash of the station id into 3 components.
  let h = 2166136261 >>> 0;
  for (let i = 0; i < stationId.length; i++) {
    h ^= stationId.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  const a = ((h & 0xff) / 255) * 2 - 1;
  const b = (((h >>> 8) & 0xff) / 255) * 2 - 1;
  const dirRad = (env.swellDirectionDeg / 360) * TWO_PI;
  return [a, b, Math.cos(dirRad), Math.sin(dirRad)];
}

/**
 * Extract a deterministic FeatureVector for a detected event.
 */
export function extractFeatures(
  event: PulseEvent,
  window: SeismicWindow,
  env: EnvironmentContext,
): FeatureVector {
  const { samples, sampleRateHz } = window;
  return {
    eventId: event.eventId,
    spectral: spectralFeatures(samples, sampleRateHz),
    amplitudeEnvelope: amplitudeEnvelope(samples, sampleRateHz),
    glideSlope: glideSlope(samples, sampleRateHz, event),
    phaseCoherence: event.phaseCoherence,
    stationGeometry: stationGeometry(window.stationId, env),
    environment: env,
  };
}
