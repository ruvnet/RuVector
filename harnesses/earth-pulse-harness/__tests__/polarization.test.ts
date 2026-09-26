// SPDX-License-Identifier: MIT
// Single-station source localization via 3-component Rayleigh polarization.
//   1. CODE validation — recover known back-azimuths from synthetic retrograde
//      Rayleigh waves (exact), so the convention is trustworthy.
//   2. REAL data — the 27.7 s wave at GT.DBIC is retrograde and arrives from an
//      offshore (Gulf-of-Guinea) bearing, not from the continental interior.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { rayleighBackAzimuth, greatCircleAzimuth } from '../src/polarization.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** Synthetic retrograde Rayleigh wave from a known back-azimuth (Z leads radial 90°). */
function synth(phi0: number, n = 4096, fHz = 0.03607, noise = 0) {
  const w = 2 * Math.PI * fHz;
  const pr = (phi0 * Math.PI) / 180;
  const z: number[] = [];
  const nn: number[] = [];
  const e: number[] = [];
  let s = 7;
  const rnd = () => {
    s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
    return s / 0x7fffffff - 0.5;
  };
  for (let i = 0; i < n; i++) {
    const R = Math.cos(w * i);
    z.push(-Math.sin(w * i) + noise * rnd());
    nn.push(R * Math.cos(pr) + noise * rnd());
    e.push(R * Math.sin(pr) + noise * rnd());
  }
  return { z, n: nn, e };
}

function angDiff(a: number, b: number): number {
  let d = Math.abs(a - b) % 360;
  if (d > 180) d = 360 - d;
  return d;
}

describe('polarization — code validation on synthetic Rayleigh waves', () => {
  it('recovers known back-azimuths exactly (clean)', () => {
    for (const phi0 of [0, 45, 90, 118, 200, 300]) {
      const { z, n, e } = synth(phi0);
      const r = rayleighBackAzimuth(z, n, e, 1.0, [0.034, 0.039]);
      expect(angDiff(r.backAzimuth, phi0)).toBeLessThanOrEqual(2);
      expect(r.retrograde).toBe(true);
      expect(r.quality).toBeGreaterThan(0.95);
    }
  });

  it('is robust to noise', () => {
    const { z, n, e } = synth(118, 4096, 0.03607, 0.6);
    const r = rayleighBackAzimuth(z, n, e, 1.0, [0.034, 0.039]);
    expect(angDiff(r.backAzimuth, 118)).toBeLessThanOrEqual(8);
  });

  it('great-circle azimuth DBIC→São Tomé is ~118°', () => {
    const az = greatCircleAzimuth(6.67016, -4.85656, 0.34, 6.73);
    expect(angDiff(az, 118)).toBeLessThan(3);
  });
});

describe('polarization — real GT.DBIC data localizes the 27.7 s source offshore', () => {
  const win = JSON.parse(
    readFileSync(resolve(root, 'data/seismic/dbic-3comp-1996-06-20.json'), 'utf8'),
  ) as { channels: { LHZ: number[]; LHN: number[]; LHE: number[] } };

  it('the 27.7 s wave is a retrograde Rayleigh wave from a Gulf-of-Guinea bearing', () => {
    const r = rayleighBackAzimuth(win.channels.LHZ, win.channels.LHN, win.channels.LHE, 1.0, [0.0350, 0.0372]);
    expect(r.retrograde).toBe(true);                 // fundamental-mode Rayleigh
    expect(r.quality).toBeGreaterThan(0.4);
    // Offshore SE/S half (toward the Gulf of Guinea), NOT the continental
    // interior (≈ 250–360°). The robust multi-day mean is ~101°.
    expect(r.backAzimuth).toBeGreaterThan(90);
    expect(r.backAzimuth).toBeLessThan(190);
  });

  it('records the multi-day back-azimuth result near the Bight of Bonny', () => {
    const baz = JSON.parse(
      readFileSync(resolve(root, 'data/seismic/dbic-backazimuth-1996.json'), 'utf8'),
    ) as { measured: { vector_mean_baz: number; concentration_R: number }; expected: { sao_tome: number; bight_of_bonny: number } };
    expect(baz.measured.concentration_R).toBeGreaterThan(0.6);   // tight across windows
    // Measured mean within ~25° of the expected Gulf-of-Guinea source bearing.
    expect(angDiff(baz.measured.vector_mean_baz, baz.expected.bight_of_bonny)).toBeLessThan(25);
  });
});
