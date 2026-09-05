// SPDX-License-Identifier: MIT
// Triangulation + bearing-stability.
//   1. CODE — recover a known source from exact bearings; degrade gracefully
//      with bearing noise.
//   2. REAL — GT.DBIC's back-azimuth is a FIXED Gulf-of-Guinea bearing across
//      seasons and years; and the honest finding that only the source-proximal
//      station cleanly polarizes the 27.7 s line (no tight triangulation yet).

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { triangulate, distanceKm } from '../src/triangulate.js';
import { greatCircleAzimuth } from '../src/polarization.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const STATIONS = [
  { name: 'DBIC', lat: 6.67016, lon: -4.85656 },
  { name: 'TAM', lat: 22.79149, lon: 5.52838 },
  { name: 'ASCN', lat: -7.9327, lon: -14.3601 },
  { name: 'LBTB', lat: -25.0151, lon: 25.5966 },
];

function angDiff(a: number, b: number): number {
  let d = Math.abs(a - b) % 360;
  if (d > 180) d = 360 - d;
  return d;
}

describe('triangulation — code validation', () => {
  it('recovers a known source from exact bearings (≈ 0 km)', () => {
    const src = { lat: 2.0, lon: 7.0 };
    const bearings = STATIONS.map((s) => ({ ...s, backAzimuth: greatCircleAzimuth(s.lat, s.lon, src.lat, src.lon) }));
    const r = triangulate(bearings);
    expect(distanceKm(r.lat, r.lon, src.lat, src.lon)).toBeLessThan(20);
    expect(r.rmsResidualDeg).toBeLessThan(1);
  });

  it('degrades gracefully under bearing noise (still regional)', () => {
    const src = { lat: 2.0, lon: 7.0 };
    let s = 7;
    const rnd = () => {
      s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
      return s / 0x7fffffff - 0.5;
    };
    const bearings = STATIONS.map((st) => ({ ...st, backAzimuth: greatCircleAzimuth(st.lat, st.lon, src.lat, src.lon) + 20 * rnd() }));
    const r = triangulate(bearings);
    expect(distanceKm(r.lat, r.lon, src.lat, src.lon)).toBeLessThan(1000);
  });
});

describe('real data — the source direction is a fixed Gulf-of-Guinea bearing', () => {
  const stab = JSON.parse(
    readFileSync(resolve(root, 'data/seismic/dbic-backazimuth-stability.json'), 'utf8'),
  ) as {
    windows: { window: string; backAzimuth: number; R: number }[];
    mean_backAzimuth: number; spread_deg: number;
    expected_gulf_of_guinea: { bight_of_bonny: number };
  };

  it('DBIC back-azimuth is stable across seasons and years', () => {
    expect(stab.windows.length).toBeGreaterThanOrEqual(3);
    for (const w of stab.windows) {
      expect(w.backAzimuth).toBeGreaterThan(80);
      expect(w.backAzimuth).toBeLessThan(120);
      expect(w.R).toBeGreaterThan(0.6);
    }
    expect(stab.spread_deg).toBeLessThan(25);
    // The fixed mean bearing points into the Gulf of Guinea.
    expect(angDiff(stab.mean_backAzimuth, stab.expected_gulf_of_guinea.bight_of_bonny)).toBeLessThan(25);
  });

  it('honestly records that only the source-proximal station tracks the 27.7 s line', () => {
    const tri = JSON.parse(
      readFileSync(resolve(root, 'data/seismic/triangulation-1996.json'), 'utf8'),
    ) as { bearings: { name: string; tracks_27_7s_source: boolean; R: number }[] };
    const tracking = tri.bearings.filter((b) => b.tracks_27_7s_source);
    expect(tracking.map((b) => b.name)).toContain('DBIC');
    // Distant stations are flagged as NOT tracking the source (low R / wrong line).
    expect(tracking.length).toBe(1);
  });
});
