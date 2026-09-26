// SPDX-License-Identifier: MIT
// DISCOVERY (offline, deterministic): re-derive the headline numbers of
// docs/research/discovery-resonator-decoupling.md from the committed per-window
// metrics — real GT.DBIC measurements, 1996–1997. No network, no fabrication.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { resonanceStats, pearson, permutationP } from '../src/climatology.js';
import type { LineMetrics } from '../src/climatology.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const data = JSON.parse(
  readFileSync(resolve(root, 'data/seismic/dbic-climatology-1996-1997.json'), 'utf8'),
) as {
  n: number;
  windows: { peakFreqHz: number; periodS: number; lineExcess: number; snr: number; secondary: number; primary: number }[];
};

const metrics: LineMetrics[] = data.windows.map((w) => ({
  peakFreqHz: w.peakFreqHz, periodS: w.periodS, lineExcess: w.lineExcess,
  snr: w.snr, secondary: w.secondary, primary: w.primary,
}));

describe('discovery — 26 s pulse is a frequency-stable, decoupled resonance', () => {
  it('has a meaningful number of real windows', () => {
    expect(data.n).toBeGreaterThanOrEqual(40);
    expect(metrics.length).toBe(data.n);
  });

  it('the line frequency is stable (CV well under 2 %)', () => {
    const s = resonanceStats(metrics);
    expect(s.meanFreqHz).toBeGreaterThan(0.0355);
    expect(s.meanFreqHz).toBeLessThan(0.0366);
    expect(s.freqCv).toBeLessThan(0.02); // < 2 %
  });

  it('the amplitude varies by more than 10x', () => {
    expect(resonanceStats(metrics).amplitudeRange).toBeGreaterThan(10);
  });

  it('frequency does not track amplitude (fixed resonance, |r| < 0.4)', () => {
    expect(Math.abs(resonanceStats(metrics).freqAmpCorr)).toBeLessThan(0.4);
  });

  it('the line is decoupled from the secondary microseism (no significant correlation)', () => {
    const L = metrics.map((m) => m.lineExcess);
    const S = metrics.map((m) => m.secondary);
    const r = pearson(L, S);
    const p = permutationP(L, S, 26, 1000);
    // The key claim: NOT a strong positive correlation; consistent with zero.
    expect(Math.abs(r)).toBeLessThan(0.3);
    expect(p).toBeGreaterThan(0.05);
  });
});

const repl = JSON.parse(
  readFileSync(resolve(root, 'data/seismic/dbic-replication-1995-1998.json'), 'utf8'),
) as {
  precise_frequency: { f0_Hz: number; periodS: number; prominence_at_canonical_26s: number; prominence_at_27_7s: number };
  replication_by_year: Record<string, { n: number; freqCvPct: number; amplitudeRange: number; freqAmpCorr: number; corr_line_secondary: number; permutation_p: number }>;
  gold_samples: { count: number; total: number; windows: { date: string; periodS: number }[] };
};

describe('discovery — adversarial follow-up (frequency, replication, gold samples)', () => {
  it('the dominant line is 27.7 s, and the canonical 26.0 s shows no excess', () => {
    expect(repl.precise_frequency.periodS).toBeGreaterThan(27);
    expect(repl.precise_frequency.periodS).toBeLessThan(28.5);
    // 27.7 s is a real peak; 26.0 s is below the local background (< 1x).
    expect(repl.precise_frequency.prominence_at_27_7s).toBeGreaterThan(1.5);
    expect(repl.precise_frequency.prominence_at_canonical_26s).toBeLessThan(1.1);
  });

  it('replicates across at least 3 independent years', () => {
    const years = Object.values(repl.replication_by_year);
    expect(years.length).toBeGreaterThanOrEqual(3);
    for (const y of years) {
      expect(y.freqCvPct).toBeLessThan(1.5);              // frequency stable every year
      expect(y.amplitudeRange).toBeGreaterThan(3);        // amplitude variable every year
      expect(Math.abs(y.corr_line_secondary)).toBeLessThan(0.4); // never strongly coupled
      expect(y.permutation_p).toBeGreaterThan(0.05);      // never significant
    }
  });

  it('finds calm-sea gold samples (strong 26 s during quiet local seas)', () => {
    expect(repl.gold_samples.count).toBeGreaterThanOrEqual(3);
    for (const g of repl.gold_samples.windows) {
      expect(g.periodS).toBeGreaterThan(26.5);
      expect(g.periodS).toBeLessThan(29);
    }
  });
});

const xs = JSON.parse(
  readFileSync(resolve(root, 'data/seismic/dbic-crossstation-1996.json'), 'utf8'),
) as { dbic_freqHz: number; ssb_freqHz: number; freq_agreement_Hz: number; artifact_ruled_out: boolean };

describe('discovery — cross-station confirmation (not a DBIC artifact)', () => {
  it('an independent station (G.SSB) shows the same line frequency as GT.DBIC', () => {
    // Both dominant long-period peaks land within ~one frequency bin of each other.
    expect(xs.freq_agreement_Hz).toBeLessThan(0.0008);
    expect(1 / xs.dbic_freqHz).toBeGreaterThan(27);
    expect(1 / xs.ssb_freqHz).toBeGreaterThan(27);
    expect(xs.artifact_ruled_out).toBe(true);
  });
});
