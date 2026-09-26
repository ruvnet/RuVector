// SPDX-License-Identifier: MIT
//
// Hypothesis scoring for the 26-second pulse. Uses a discovery-score weighting
// (the default "genome"); each hypothesis can override weights. Records the
// known "killer contradiction" when the relevant evidence is weak.

import type {
  Hypothesis,
  HypothesisEvidence,
  HypothesisScore,
} from './types.js';

/**
 * DEFAULT discovery-score weighting (the genome). Hypotheses can override any
 * subset of these via their `weights` map. Keys map to evidence components:
 *   sourceStability           -> evidence.sourceStability
 *   environmentalCorrelation  -> evidence.environmentalCorrelation
 *   outOfSamplePrediction     -> evidence.outOfSamplePrediction
 *   contradictionSurvival     -> evidence.contradictionSurvival
 *   parsimony                 -> evidence.parsimony  (mechanistic plausibility)
 *   spectralShapeMatch        -> evidence.spectralShapeMatch (citation grounding)
 */
export const DEFAULT_WEIGHTS: Record<string, number> = {
  sourceStability: 0.25,
  environmentalCorrelation: 0.2,
  outOfSamplePrediction: 0.2,
  contradictionSurvival: 0.15,
  parsimony: 0.1,
  spectralShapeMatch: 0.1,
};

/** Each hypothesis's known "killer contradiction". */
const KILLER_CONTRADICTIONS: Record<string, string> = {
  'ocean-shelf-resonance':
    'If the 26s period drifts with shelf bathymetry tides yet the pulse stays fixed, pure shelf resonance is falsified.',
  'coupled-ocean-geology':
    'If removing the geologic (volcanic) proxy leaves environmental correlation unchanged, the coupling term is unnecessary.',
  'volcanic-source':
    'A purely volcanic source predicts strong correlation with the volcanic proxy and eruptions; persistent pulses during volcanic quiescence falsify it.',
  'instrument-artifact':
    'An instrument artifact predicts no source stability across independent stations; cross-station coherence falsifies it.',
};

/**
 * The default candidate hypotheses with starting weight overrides.
 */
export const DEFAULT_HYPOTHESES: Hypothesis[] = [
  {
    id: 'ocean-shelf-resonance',
    name: 'Ocean–continental shelf resonance (Gulf of Guinea)',
    // Leans on stable source + environmental (swell/tide) correlation.
    weights: {
      sourceStability: 0.28,
      environmentalCorrelation: 0.24,
      outOfSamplePrediction: 0.18,
      contradictionSurvival: 0.12,
      parsimony: 0.1,
      spectralShapeMatch: 0.08,
    },
  },
  {
    id: 'coupled-ocean-geology',
    name: 'Coupled ocean-wave / local geology mechanism',
    // Balanced; rewards out-of-sample prediction and contradiction survival.
    weights: {
      sourceStability: 0.24,
      environmentalCorrelation: 0.2,
      outOfSamplePrediction: 0.22,
      contradictionSurvival: 0.16,
      parsimony: 0.08,
      spectralShapeMatch: 0.1,
    },
  },
  {
    id: 'volcanic-source',
    name: 'Volcanic / magmatic source',
    // Heavily weights environmental correlation (volcanic proxy).
    weights: {
      sourceStability: 0.18,
      environmentalCorrelation: 0.3,
      outOfSamplePrediction: 0.18,
      contradictionSurvival: 0.14,
      parsimony: 0.1,
      spectralShapeMatch: 0.1,
    },
  },
  {
    id: 'instrument-artifact',
    name: 'Instrument / processing artifact',
    // Parsimony-heavy null hypothesis; punished by cross-station stability.
    weights: {
      sourceStability: 0.1,
      environmentalCorrelation: 0.1,
      outOfSamplePrediction: 0.15,
      contradictionSurvival: 0.15,
      parsimony: 0.4,
      spectralShapeMatch: 0.1,
    },
  },
];

/** Threshold below which a weak component triggers the killer contradiction. */
const CONTRADICTION_THRESHOLD = 0.35;

function clamp01(x: number): number {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

/**
 * Score a single hypothesis against evidence. Uses the hypothesis's weight
 * overrides falling back to DEFAULT_WEIGHTS, normalizing weights so the score
 * is a weighted average in [0,1].
 */
export function scoreHypothesis(
  h: Hypothesis,
  evidence: HypothesisEvidence,
): HypothesisScore {
  const weights = { ...DEFAULT_WEIGHTS, ...h.weights };
  const evidenceMap: Record<string, number> = {
    sourceStability: clamp01(evidence.sourceStability),
    environmentalCorrelation: clamp01(evidence.environmentalCorrelation),
    outOfSamplePrediction: clamp01(evidence.outOfSamplePrediction),
    contradictionSurvival: clamp01(evidence.contradictionSurvival),
    parsimony: clamp01(evidence.parsimony),
    spectralShapeMatch: clamp01(evidence.spectralShapeMatch),
  };

  let weightSum = 0;
  for (const key of Object.keys(weights)) weightSum += weights[key];

  const components: Record<string, number> = {};
  let score = 0;
  for (const key of Object.keys(weights)) {
    const ev = evidenceMap[key] ?? 0;
    const contribution = weights[key] * ev;
    components[key] = contribution;
    score += contribution;
  }
  if (weightSum > 0) score /= weightSum;

  // Record the killer contradiction when the hypothesis fails to survive
  // contradictions or its decisive evidence is weak.
  const contradictions: string[] = [];
  const killer = KILLER_CONTRADICTIONS[h.id];
  if (killer) {
    const decisive = decisiveEvidenceFor(h.id, evidenceMap);
    if (
      evidence.contradictionSurvival < CONTRADICTION_THRESHOLD ||
      decisive < CONTRADICTION_THRESHOLD
    ) {
      contradictions.push(killer);
    }
  }

  return {
    hypothesisId: h.id,
    score: clamp01(score),
    components,
    contradictions,
  };
}

/**
 * The evidence component most decisive for falsifying each hypothesis.
 */
function decisiveEvidenceFor(
  id: string,
  ev: Record<string, number>,
): number {
  switch (id) {
    case 'ocean-shelf-resonance':
      return ev.sourceStability;
    case 'coupled-ocean-geology':
      return ev.outOfSamplePrediction;
    case 'volcanic-source':
      return ev.environmentalCorrelation;
    case 'instrument-artifact':
      return 1 - ev.sourceStability; // artifact is contradicted by stability
    default:
      return 1;
  }
}

/**
 * Score and rank hypotheses descending by score. Ties broken by id for
 * determinism.
 */
export function rankHypotheses(
  evidence: HypothesisEvidence,
  hypotheses: Hypothesis[] = DEFAULT_HYPOTHESES,
): HypothesisScore[] {
  const scored = hypotheses.map((h) => scoreHypothesis(h, evidence));
  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.hypothesisId < b.hypothesisId
      ? -1
      : a.hypothesisId > b.hypothesisId
        ? 1
        : 0;
  });
  return scored;
}
