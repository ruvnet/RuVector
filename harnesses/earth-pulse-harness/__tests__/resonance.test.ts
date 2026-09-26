// SPDX-License-Identifier: MIT
// Resonance sharpness (Q) and temporal gliding.
//   1. CODE — a stationary tone yields a sharp, well-resolved line and no glide;
//      a chirp (sweeping frequency) yields a long monotonic glide run.
//   2. REAL — the committed GT.DBIC measurement: a narrowband resonance
//      (Q ≳ 30, resolved) whose FUNDAMENTAL does not glide in time.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { welchPsd } from '../src/spectrum.js';
import { spectralLineQ, glideStats } from '../src/climatology.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function noise(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
    return s / 0x7fffffff - 0.5;
  };
}

describe('resonance — code validation', () => {
  it('a stationary tone gives a sharp, well-resolved spectral line', () => {
    const N = 1 << 16;
    const rnd = noise(7);
    const x: number[] = [];
    for (let i = 0; i < N; i++) x.push(10 * Math.cos(2 * Math.PI * 0.03607 * i) + rnd());
    const spec = welchPsd(x, { fs: 1, segment: 16384, overlap: 8192, average: 'median' });
    const q = spectralLineQ(spec, [0.034, 0.039], [[0.030, 0.034], [0.039, 0.043]]);
    expect(q.f0Hz).toBeGreaterThan(0.0358);
    expect(q.f0Hz).toBeLessThan(0.0363);
    expect(q.q).toBeGreaterThan(40); // a near-pure tone is a sharp line
  });

  it('detects a glide: a chirp has a much longer monotonic run than a stationary tone', () => {
    const N = 200 * 3600; // ~8 days
    const stationary: number[] = [];
    const chirp: number[] = [];
    const rnd = noise(11);
    for (let i = 0; i < N; i++) {
      stationary.push(10 * Math.cos(2 * Math.PI * 0.0365 * i) + rnd());
      // frequency sweeps 0.035 → 0.039 across the record
      const f = 0.035 + (0.004 * i) / N;
      chirp.push(10 * Math.cos(2 * Math.PI * f * i) + rnd());
    }
    const band: [number, number] = [0.034, 0.040];
    const gStat = glideStats(stationary, 1, band);
    const gChirp = glideStats(chirp, 1, band);
    expect(gChirp.longestMonotonicRun).toBeGreaterThan(gStat.longestMonotonicRun * 3);
  });
});

describe('real data — the 27.7 s resonance is narrowband and does not glide', () => {
  const q = JSON.parse(
    readFileSync(resolve(root, 'data/seismic/dbic-resonance-q.json'), 'utf8'),
  ) as {
    sharpness: { q: number; widthBins: number; periodS: number };
    temporal: { windows: number; longestMonotonicRunHours: number };
  };

  it('is a genuine narrowband resonance (Q ≳ 30, resolved)', () => {
    expect(q.sharpness.q).toBeGreaterThan(30);
    expect(q.sharpness.widthBins).toBeGreaterThanOrEqual(3); // resolved, not bin-limited
    expect(q.sharpness.periodS).toBeGreaterThan(27);
    expect(q.sharpness.periodS).toBeLessThan(28.5);
  });

  it('the fundamental does not sustain a glide (longest run ≪ record length)', () => {
    // A real sweep would monotonically march across most of the record.
    expect(q.temporal.longestMonotonicRunHours).toBeLessThan(q.temporal.windows / 10);
  });
});
