// SPDX-License-Identifier: MIT
// Gliding-tremor search + the contradiction test that rejects the false positive.
//   1. CODE — glideSearch flags an upward chirp and not a stationary tone; the
//      coherence verdict confirms a co-azimuth coherent band and rejects an
//      incoherent / off-azimuth one.
//   2. REAL — the committed search: a naive detector finds "glides", but the
//      back-azimuth contradiction test REJECTS them as secondary microseism
//      (the honest null — ADR-004).

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { glideSearch, sourceCoherenceVerdict } from '../src/spectrogram.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function noise(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
    return s / 0x7fffffff - 0.5;
  };
}

describe('gliding-tremor search — code validation', () => {
  it('flags an upward chirp and not a stationary tone', () => {
    const N = 120 * 3600;
    const rnd = noise(3);
    const chirp: number[] = [];
    const tone: number[] = [];
    for (let i = 0; i < N; i++) {
      const f = 0.05 + (0.02 * i) / N; // sweeps 0.05 → 0.07
      chirp.push(8 * Math.cos(2 * Math.PI * f * i) + rnd());
      tone.push(8 * Math.cos(2 * Math.PI * 0.06 * i) + rnd());
    }
    const gc = glideSearch(chirp, 1, [0.045, 0.075]);
    const gt = glideSearch(tone, 1, [0.045, 0.075]);
    expect(gc.candidates.length).toBeGreaterThan(gt.candidates.length);
  });

  it('the coherence verdict confirms co-azimuth coherent bands and rejects others', () => {
    const fund = { backAzimuth: 100, R: 0.76 };
    expect(sourceCoherenceVerdict(fund, { backAzimuth: 105, R: 0.7 }).confirmed).toBe(true);
    expect(sourceCoherenceVerdict(fund, { backAzimuth: 269, R: 0.17 }).confirmed).toBe(false); // incoherent
    expect(sourceCoherenceVerdict(fund, { backAzimuth: 200, R: 0.7 }).confirmed).toBe(false); // off-azimuth
  });
});

describe('real data — the gliding tremors are NOT isolable single-station (honest null)', () => {
  const g = JSON.parse(
    readFileSync(resolve(root, 'data/seismic/dbic-gliding-search.json'), 'utf8'),
  ) as {
    naive_detector: { candidate_glides: number; dominant_feature: { Q: number } };
    contradiction_test_ADR004: { confirmed: boolean };
  };

  it('a naive detector finds candidates but the band is broad (secondary microseism)', () => {
    expect(g.naive_detector.candidate_glides).toBeGreaterThan(0);
    expect(g.naive_detector.dominant_feature.Q).toBeLessThan(10); // broad, not a narrow tremor line
  });

  it('the contradiction test rejects the candidates (not the 26 s source)', () => {
    expect(g.contradiction_test_ADR004.confirmed).toBe(false);
  });
});
