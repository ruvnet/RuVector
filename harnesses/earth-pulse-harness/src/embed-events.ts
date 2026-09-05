// SPDX-License-Identifier: MIT
//
// Event embedding. Per ADR-002 ("don't force everything into one vector") we
// build SEPARATE normalized embeddings for waveform, environment, and source
// geometry, plus a combined concatenation. Deterministic; pure math.

import type { EventEmbedding, FeatureVector } from './types.js';

/** Target dimensionality of each sub-embedding. */
const WAVEFORM_DIM = 13;
const ENVIRONMENT_DIM = 7;
const SOURCE_DIM = 4;

/**
 * L2-normalize a vector. A zero vector is returned unchanged (norm 0).
 */
function l2Normalize(v: number[]): number[] {
  let sumSq = 0;
  for (const x of v) sumSq += x * x;
  const norm = Math.sqrt(sumSq);
  if (norm === 0) return v.slice();
  return v.map((x) => x / norm);
}

/**
 * Pad or truncate a vector to exactly `dim` entries.
 */
function fit(v: number[], dim: number): number[] {
  if (v.length === dim) return v.slice();
  if (v.length > dim) return v.slice(0, dim);
  const out = v.slice();
  while (out.length < dim) out.push(0);
  return out;
}

/**
 * Waveform embedding: spectral shape + amplitude envelope + scalar dynamics.
 */
function waveformEmbedding(f: FeatureVector): number[] {
  const raw = [
    ...f.spectral,
    ...f.amplitudeEnvelope,
    f.glideSlope,
    f.phaseCoherence,
  ];
  return l2Normalize(fit(raw, WAVEFORM_DIM));
}

/**
 * Environment embedding from the EnvironmentContext numeric fields.
 */
function environmentEmbedding(f: FeatureVector): number[] {
  const e = f.environment;
  const dirRad = (e.swellDirectionDeg / 360) * Math.PI * 2;
  const raw = [
    e.swellHeightM,
    e.swellPeriodS,
    Math.cos(dirRad),
    Math.sin(dirRad),
    e.tidePhase,
    e.barometricGradient,
    e.volcanicProxyScore,
  ];
  return l2Normalize(fit(raw, ENVIRONMENT_DIM));
}

/**
 * Source-geometry embedding from the station geometry placeholder vector.
 */
function sourceEmbedding(f: FeatureVector): number[] {
  return l2Normalize(fit(f.stationGeometry, SOURCE_DIM));
}

/**
 * Build the separated embeddings for an event.
 */
export function embedEvent(features: FeatureVector): EventEmbedding {
  const waveform = waveformEmbedding(features);
  const environment = environmentEmbedding(features);
  const source = sourceEmbedding(features);
  const combined = l2Normalize([...waveform, ...environment, ...source]);
  return {
    eventId: features.eventId,
    waveform,
    environment,
    source,
    combined,
  };
}

/**
 * Cosine similarity of two equal-length vectors. Returns 0 for mismatched
 * lengths or zero vectors.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/**
 * k nearest neighbors of `query` within `corpus` by cosine similarity on the
 * chosen embedding `field` (default 'combined'). The query itself is excluded.
 * Ties are broken deterministically by eventId.
 */
export function nearestNeighbors(
  query: EventEmbedding,
  corpus: EventEmbedding[],
  k: number,
  field: 'waveform' | 'environment' | 'source' | 'combined' = 'combined',
): { eventId: string; score: number }[] {
  const q = query[field];
  const scored = corpus
    .filter((c) => c.eventId !== query.eventId)
    .map((c) => ({ eventId: c.eventId, score: cosineSimilarity(q, c[field]) }));
  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.eventId < b.eventId ? -1 : a.eventId > b.eventId ? 1 : 0;
  });
  return scored.slice(0, Math.max(0, k));
}
