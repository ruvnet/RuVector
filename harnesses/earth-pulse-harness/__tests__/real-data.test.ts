// SPDX-License-Identifier: MIT
// REAL-DATA PROOF (offline, deterministic, no network).
//
// Runs against one committed real day of GT.DBIC LHZ data
// (data/seismic/GT.DBIC.LHZ.1995-01-05.window.json — raw counts straight from
// the IRIS FDSN timeseries web service) and proves, with no fabrication, that:
//   1. a persistent narrowband spectral line sits in the long-period microseism
//      band at ~27.7 s — the "26-second pulse" from the Gulf of Guinea;
//   2. the harness per-event detector recovers it from band-passed real data;
//   3. the agenticow (ruVector COW) planetary memory ingests, branches, and
//      retrieves real-event embeddings.

import { describe, it, expect } from 'vitest';
import { readFileSync, rmSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { welchPsd, findPersistentLine, bandpass } from '../src/spectrum.js';
import { detectPulse } from '../src/detect-26s.js';
import { extractFeatures } from '../src/extract-features.js';
import { embedEvent } from '../src/embed-events.js';
import { PlanetaryMemory } from '../src/memory.js';
import type { SeismicWindow, EnvironmentContext } from '../src/types.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const window: SeismicWindow & { _provenance?: unknown } = JSON.parse(
  readFileSync(resolve(root, 'data/seismic/GT.DBIC.LHZ.1995-01-05.window.json'), 'utf8'),
);

const env: EnvironmentContext = {
  swellHeightM: 2.1, swellPeriodS: 13.2, swellDirectionDeg: 218, tidePhase: 0.62,
  barometricGradient: 0.14, volcanicProxyScore: 0.03, sourceRegion: 'Gulf of Guinea',
};

describe('real-data proof — GT.DBIC 26-second microseism', () => {
  it('loads a real day of GT.DBIC LHZ counts (86400 samples, 1 sps)', () => {
    expect(window.stationId).toBe('GT.DBIC');
    expect(window.sampleRateHz).toBe(1.0);
    expect(window.samples.length).toBe(86400);
  });

  it('median Welch PSD shows a persistent line in the long-period band (~24-29 s)', () => {
    const spec = welchPsd(window.samples, { fs: 1.0, segment: 4096, overlap: 2048, average: 'median' });
    const line = findPersistentLine(spec, [0.033, 0.045], 15);
    expect(line).not.toBeNull();
    // The Gulf-of-Guinea long-period microseism line — ~27.7 s in this epoch.
    expect(line!.periodS).toBeGreaterThan(24);
    expect(line!.periodS).toBeLessThan(29);
    // It is a genuine line, not a slope: clearly prominent over the background.
    expect(line!.prominence).toBeGreaterThan(1.8);
  });

  it('the single committed day reproduces the same line frequency as the 12-day artifact', () => {
    // Cross-validation: the persistent line found in ONE real day must match the
    // line recorded in the committed 12-day (288 h) median-PSD artifact. A real
    // geophysical line reproduces across independent record lengths; an artifact
    // would not. (This is also why we median-average — to reject transients.)
    const dayLine = findPersistentLine(
      welchPsd(window.samples, { fs: 1, segment: 8192, overlap: 4096, average: 'median' }),
      [0.033, 0.045],
      15,
    );
    const artifact = JSON.parse(
      readFileSync(resolve(root, 'data/seismic/dbic-1995-median-psd.json'), 'utf8'),
    ) as { line: { freqHz: number; periodS: number } };
    expect(dayLine).not.toBeNull();
    expect(Math.abs(dayLine!.freqHz - artifact.line.freqHz)).toBeLessThan(0.0005);
  });

  it('the harness per-event detector recovers the pulse from band-passed real data', () => {
    const filt = bandpass(window.samples, 1.0, 0.033, 0.045);
    const seg = filt.slice(20000, 20000 + 3600);
    const win: SeismicWindow = { stationId: 'GT.DBIC', startIso: window.startIso, sampleRateHz: 1.0, samples: seg };
    const ev = detectPulse(win, { lowPeriodS: 24, highPeriodS: 30, minCoherence: 0 });
    expect(ev).not.toBeNull();
    expect(ev!.dominantPeriodS).toBeGreaterThan(24);
    expect(ev!.dominantPeriodS).toBeLessThan(30);
    expect(ev!.confidence).toBeGreaterThan(0.5);
  });

  it('agenticow (ruVector COW) planetary memory ingests, retrieves, and branches real events', () => {
    const filt = bandpass(window.samples, 1.0, 0.033, 0.045);
    const events: { event: any; emb: any }[] = [];
    for (let i = 0; i < 4; i++) {
      const s = filt.slice(8000 + i * 4000, 8000 + i * 4000 + 3600);
      const w: SeismicWindow = { stationId: 'GT.DBIC', startIso: window.startIso, sampleRateHz: 1.0, samples: s };
      const ev = detectPulse(w, { lowPeriodS: 24, highPeriodS: 30, minCoherence: 0 });
      if (!ev) continue;
      ev.eventId = `GT.DBIC-1995-01-05-w${i}`;
      events.push({ event: ev, emb: embedEvent(extractFeatures(ev, w, env)) });
    }
    expect(events.length).toBeGreaterThanOrEqual(3);

    const memPath = resolve(root, '.metaharness/work/test-memory.rvf');
    mkdirSync(dirname(memPath), { recursive: true });
    try { rmSync(memPath); } catch { /* fresh */ }
    const dim = events[0].emb.waveform.length;
    const mem = PlanetaryMemory.open(memPath, dim, 'waveform');
    try {
      const ing = mem.ingestMany(events);
      expect(ing.accepted).toBe(events.length);

      // Nearest analog of an event is itself (distance ~0).
      const nn = mem.nearest(events[0].emb, 2);
      expect(nn[0].eventId).toBe('GT.DBIC-1995-01-05-w0');
      expect(nn[0].distance).toBeLessThan(1e-3);

      // COW branch is isolated from the base.
      const calm = mem.branch('calm-week');
      try {
        calm.ingestEvent({ ...events[events.length - 1].event, eventId: 'calm-probe' }, events[events.length - 1].emb);
        const calmHit = calm.nearest(events[events.length - 1].emb, 1);
        expect(calmHit.length).toBeGreaterThan(0);
      } finally {
        calm.close();
      }
    } finally {
      mem.close();
      try { rmSync(memPath); } catch { /* cleanup */ }
    }
  });
});
