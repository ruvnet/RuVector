// SPDX-License-Identifier: MIT
//
// End-to-end pipeline: detect -> extract -> embed -> score.
// Pure & deterministic. NO network, NO fabricated observations — it only
// computes over the window/env it is given.

import { detectPulse } from './detect-26s.js';
import { embedEvent } from './embed-events.js';
import { extractFeatures } from './extract-features.js';
import { rankHypotheses } from './score-hypotheses.js';
import type {
  EnvironmentContext,
  EventEmbedding,
  FeatureVector,
  HypothesisEvidence,
  HypothesisScore,
  PulseEvent,
  SeismicWindow,
} from './types.js';

export interface PipelineResult {
  event: PulseEvent | null;
  features?: FeatureVector;
  embedding?: EventEmbedding;
  ranking?: HypothesisScore[];
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/**
 * Derive hypothesis evidence deterministically from the detected event and
 * features. These are proxies computed purely from the supplied data — no
 * external observations are invented.
 */
function deriveEvidence(
  event: PulseEvent,
  features: FeatureVector,
): HypothesisEvidence {
  const env = features.environment;

  // Source stability: a coherent, confident, in-band detection implies a
  // stable persistent source.
  const sourceStability = clamp01(
    0.6 * event.phaseCoherence + 0.4 * event.confidence,
  );

  // Environmental correlation: how strongly swell energy (height/period in the
  // microseism range) co-occurs with the pulse. Bounded swell proxy.
  const swellEnergy = clamp01((env.swellHeightM / 5) * (env.swellPeriodS / 20));
  const environmentalCorrelation = clamp01(
    0.7 * swellEnergy + 0.3 * (1 - Math.abs(0.5 - env.tidePhase) * 2),
  );

  // Out-of-sample prediction proxy: concentrated spectral shape around 26s
  // predicts well; the central (26s) sub-band fraction is a good signal.
  const central = features.spectral[2] ?? 0; // 26s sub-band
  const outOfSamplePrediction = clamp01(central);

  // Spectral shape match (citation grounding): central-band dominance vs tails.
  const tails =
    (features.spectral[0] ?? 0) + (features.spectral[features.spectral.length - 1] ?? 0);
  const spectralShapeMatch = clamp01(central - 0.5 * tails + 0.5);

  // Parsimony: stronger when no exotic glide is required to explain the data.
  const parsimony = clamp01(event.glideDetected ? 0.45 : 0.7);

  // Contradiction survival: high coherence + low volcanic dependence survives
  // the cross-station and quiescence contradictions.
  const contradictionSurvival = clamp01(
    0.5 * event.phaseCoherence +
      0.3 * (1 - env.volcanicProxyScore) +
      0.2 * sourceStability,
  );

  return {
    sourceStability,
    environmentalCorrelation,
    outOfSamplePrediction,
    spectralShapeMatch,
    parsimony,
    contradictionSurvival,
  };
}

/**
 * Run the full harness pipeline over a single window + environment context.
 */
export function runPipeline(
  window: SeismicWindow,
  env: EnvironmentContext,
): PipelineResult {
  const event = detectPulse(window);
  if (!event) return { event: null };

  // Carry the environment's source region onto the event.
  const located: PulseEvent = { ...event, sourceRegion: env.sourceRegion };

  const features = extractFeatures(located, window, env);
  const embedding = embedEvent(features);
  const evidence = deriveEvidence(located, features);
  const ranking = rankHypotheses(evidence);

  return { event: located, features, embedding, ranking };
}
