// SPDX-License-Identifier: MIT
// Signal-class partitioning via ruVector dynamic MinCut (@ruvector/mincut-wasm),
// with the connected-components fallback. Two kinds of assertion:
//   1. REAL data: the 26 s events from one GT.DBIC window form ONE coherent
//      signal class (a single stable source — the honest geophysical result).
//   2. MECHANISM: given two deliberately orthogonal embedding groups the
//      partitioner separates them into TWO classes, and a similarity bridge
//      merges them back — verifying the cut actually works.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { bandpass } from '../src/spectrum.js';
import { detectPulse } from '../src/detect-26s.js';
import { extractFeatures } from '../src/extract-features.js';
import { embedEvent } from '../src/embed-events.js';
import { partitionEvents, streamPartition, loadMincut } from '../src/partition.js';
import type { EventEmbedding, SeismicWindow, EnvironmentContext } from '../src/types.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const win: SeismicWindow = JSON.parse(
  readFileSync(resolve(root, 'data/seismic/GT.DBIC.LHZ.1995-01-05.window.json'), 'utf8'),
);
const env: EnvironmentContext = {
  swellHeightM: 2.1, swellPeriodS: 13.2, swellDirectionDeg: 218, tidePhase: 0.62,
  barometricGradient: 0.14, volcanicProxyScore: 0.03, sourceRegion: 'Gulf of Guinea',
};

function realOceanEvents(n: number) {
  const filt = bandpass(win.samples, 1.0, 0.033, 0.045);
  const out: { eventId: string; emb: EventEmbedding }[] = [];
  for (let i = 0; i < n; i++) {
    const s = filt.slice(6000 + i * 3000, 6000 + i * 3000 + 3600);
    const w: SeismicWindow = { stationId: 'GT.DBIC', startIso: win.startIso, sampleRateHz: 1.0, samples: s };
    const ev = detectPulse(w, { lowPeriodS: 24, highPeriodS: 30, minCoherence: 0 });
    if (!ev) continue;
    ev.eventId = `ocean-${i}`;
    out.push({ eventId: ev.eventId, emb: embedEvent(extractFeatures(ev, w, env)) });
  }
  return out;
}

/** A crafted embedding whose `combined` vector points along axis `axis`. */
function crafted(eventId: string, axis: 0 | 1, jitter: number): { eventId: string; emb: EventEmbedding } {
  const v = axis === 0 ? [1, jitter, 0, 0] : [0, 0, 1, jitter];
  return { eventId, emb: { eventId, waveform: v, environment: v, source: v, combined: v } };
}

describe('signal-class partition — ruVector dynamic MinCut', () => {
  it('the mincut backend reports which engine it used', async () => {
    const available = !!(await loadMincut());
    // Either the real wasm or the deterministic fallback must be usable.
    const res = await partitionEvents(realOceanEvents(4), { field: 'combined', threshold: 0.5 });
    expect(['ruvector-mincut-wasm', 'fallback-connected-components']).toContain(res.backend);
    if (available) expect(res.backend).toBe('ruvector-mincut-wasm');
  });

  it('real GT.DBIC 26 s events form ONE coherent signal class', async () => {
    const events = realOceanEvents(8);
    expect(events.length).toBeGreaterThanOrEqual(6);
    const res = await partitionEvents(events, { field: 'combined', threshold: 0.5 });
    expect(res.classes).toBe(1);
    expect(res.vertices).toBe(events.length);
  });

  it('separates two orthogonal classes, and a bridge merges them', async () => {
    const groupA = [0, 1, 2].map((i) => crafted(`A${i}`, 0, i * 0.01));
    const groupB = [0, 1, 2].map((i) => crafted(`B${i}`, 1, i * 0.01));
    const two = await partitionEvents([...groupA, ...groupB], { field: 'combined', threshold: 0.5 });
    expect(two.classes).toBe(2);

    // Add an event similar to BOTH groups → one connected class.
    const bridge = { eventId: 'bridge', emb: { eventId: 'bridge', waveform: [1, 0, 1, 0], environment: [1, 0, 1, 0], source: [1, 0, 1, 0], combined: [1, 0, 1, 0] } as EventEmbedding };
    const merged = await partitionEvents([...groupA, ...groupB, bridge], { field: 'combined', threshold: 0.4 });
    expect(merged.classes).toBe(1);
  });

  it('dynamically detects a new class entering the stream', async () => {
    const groupA = [0, 1, 2, 3].map((i) => crafted(`A${i}`, 0, i * 0.01));
    const groupB = [0, 1, 2].map((i) => crafted(`B${i}`, 1, i * 0.01));
    // First two batches are the same class; the third introduces a new one.
    const counts = (await streamPartition([groupA.slice(0, 2), groupA.slice(2), groupB], { field: 'combined', threshold: 0.5 })).map((r) => r.classes);
    expect(counts[0]).toBe(1);
    expect(counts[1]).toBe(1);
    expect(counts[2]).toBe(2); // regime change: a second mechanism appears
  });
});
