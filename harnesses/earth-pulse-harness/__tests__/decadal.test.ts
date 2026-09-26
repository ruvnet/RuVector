// SPDX-License-Identifier: MIT
// Decadal-scale frequency stability at the source station (GT.DBIC, 1995–2002),
// and the honest record that a >=20-year bridge is not achievable with open data.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const d = JSON.parse(
  readFileSync(resolve(root, 'data/seismic/dbic-decadal-stability.json'), 'utf8'),
) as {
  years: { year: number; periodS: number; freqHz: number }[];
  stability: { freqCvPct: number; meanPeriodS: number };
  decadal_bridge: { result: string; honest_conclusion: string };
};

describe('decadal — the source frequency is stable across DBIC full lifetime', () => {
  it('measures the line across many years spanning ~8 years', () => {
    expect(d.years.length).toBeGreaterThanOrEqual(5);
    const span = Math.max(...d.years.map((y) => y.year)) - Math.min(...d.years.map((y) => y.year));
    expect(span).toBeGreaterThanOrEqual(6);
  });

  it('the frequency is constant to well under 1% over the whole span', () => {
    for (const y of d.years) {
      expect(y.periodS).toBeGreaterThan(27);
      expect(y.periodS).toBeLessThan(28.2);
    }
    expect(d.stability.freqCvPct).toBeLessThan(0.5);
    expect(d.stability.meanPeriodS).toBeGreaterThan(27.5);
    expect(d.stability.meanPeriodS).toBeLessThan(28);
  });

  it('honestly records that the >=20-year bridge is not achievable with open data', () => {
    expect(d.decadal_bridge.result).toMatch(/NOT achievable/i);
    expect(d.decadal_bridge.honest_conclusion).toMatch(/8-year|cannot extend/i);
  });
});
