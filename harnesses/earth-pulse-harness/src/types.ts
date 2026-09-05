// SPDX-License-Identifier: MIT
//
// Shared domain types for the Earth "26-second pulse" research harness.
// FREEZE THE PHYSICS, EVOLVE THE HARNESS: these types describe immutable
// observation/derivation contracts; the harness logic may evolve around them.

/**
 * A contiguous slice of seismic samples from one station.
 * `startIso` is an ISO-8601 timestamp marking the first sample.
 */
export interface SeismicWindow {
  stationId: string;
  startIso: string;
  sampleRateHz: number;
  samples: number[];
}

/**
 * Oceanographic / geophysical context co-located with a window.
 * `tidePhase` in [0,1), `swellDirectionDeg` in [0,360).
 */
export interface EnvironmentContext {
  swellHeightM: number;
  swellPeriodS: number;
  swellDirectionDeg: number;
  tidePhase: number;
  barometricGradient: number;
  volcanicProxyScore: number;
  sourceRegion: string;
}

/**
 * A detected microseism / pulse event.
 */
export interface PulseEvent {
  eventId: string;
  timestampIso: string;
  dominantPeriodS: number;
  frequencyHz: number;
  amplitude: number;
  phaseCoherence: number;
  glideDetected: boolean;
  sourceRegion: string;
  confidence: number;
}

/**
 * Numeric feature representation derived from an event + its window + env.
 */
export interface FeatureVector {
  eventId: string;
  spectral: number[];
  amplitudeEnvelope: number[];
  glideSlope: number;
  phaseCoherence: number;
  stationGeometry: number[];
  environment: EnvironmentContext;
}

/**
 * Separated embeddings (ADR-002: don't force everything into one vector).
 * Each sub-embedding is independently L2-normalized; `combined` is their
 * concatenation (also L2-normalized as a whole).
 */
export interface EventEmbedding {
  eventId: string;
  waveform: number[];
  environment: number[];
  source: number[];
  combined: number[];
}

/**
 * A scientific hypothesis with a scoring genome (weight overrides).
 */
export interface Hypothesis {
  id: string;
  name: string;
  weights: Record<string, number>;
}

/**
 * Result of scoring a hypothesis against gathered evidence.
 */
export interface HypothesisScore {
  hypothesisId: string;
  score: number;
  components: Record<string, number>;
  contradictions: string[];
}

/**
 * Output of the promotion gate.
 */
export interface ValidationReport {
  passed: boolean;
  leakageDetected: boolean;
  heldOutError: number;
  baselineError: number;
  improvementPct: number;
  notes: string[];
}

/** Period band expressed in seconds (inclusive low/high). */
export interface PeriodBand {
  lowPeriodS: number;
  highPeriodS: number;
}

/** Evidence gathered for hypothesis scoring; all components in [0,1]. */
export interface HypothesisEvidence {
  sourceStability: number;
  environmentalCorrelation: number;
  outOfSamplePrediction: number;
  spectralShapeMatch: number;
  parsimony: number;
  contradictionSurvival: number;
}
