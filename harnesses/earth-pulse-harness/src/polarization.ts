// SPDX-License-Identifier: MIT
// Single-station source localization for the 26 s pulse via three-component
// Rayleigh-wave polarization. Pure, deterministic.
//
// A fundamental-mode Rayleigh wave is RETROGRADE elliptical in the
// vertical–radial plane: the horizontal motion lies along the source–station
// line (radial), and the vertical leads the radial by 90°. So:
//   1. the horizontal particle-motion AXIS gives the propagation direction
//      (mod 180°), and
//   2. the vertical↔radial 90° phase (Hilbert) resolves which end of that axis
//      points at the source — the back-azimuth.
// This recovers a source bearing from ONE station. See docs (Level 2).

import { bandpass, fft } from './spectrum.js';

/** Hilbert transform (90° phase shift) via the analytic-signal FFT method. */
export function hilbert(x: number[] | Float64Array): number[] {
  const n0 = x.length;
  let n = 1;
  while (n < n0) n <<= 1;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n0; i++) re[i] = x[i];
  fft(re, im);
  // Analytic signal: double positive freqs, zero negatives, keep DC + Nyquist.
  for (let k = 0; k < n; k++) {
    if (k === 0 || k === n / 2) continue;
    if (k < n / 2) {
      re[k] *= 2;
      im[k] *= 2;
    } else {
      re[k] = 0;
      im[k] = 0;
    }
  }
  fft(re, im, true);
  // H[x] is the imaginary part of the analytic signal.
  const out = new Array<number>(n0);
  for (let i = 0; i < n0; i++) out[i] = im[i];
  return out;
}

function corr(a: number[] | Float64Array, b: number[] | Float64Array): number {
  const n = Math.min(a.length, b.length);
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let nu = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma;
    const y = b[i] - mb;
    nu += x * y;
    da += x * x;
    db += y * y;
  }
  return nu / Math.sqrt(da * db);
}

export interface BackAzimuthResult {
  /** Direction from station to source, degrees clockwise from North (0–360). */
  backAzimuth: number;
  /** Polarization axis (mod 180) — robust; the 180° sense is resolved separately. */
  axisDeg: number;
  /** |peak correlation| of Hilbert(Z) with the radial — polarization quality (0–1). */
  quality: number;
  /** true if the motion is retrograde (fundamental-mode Rayleigh signature). */
  retrograde: boolean;
}

/**
 * Rayleigh-wave back-azimuth from 3-component records (N up at azimuth 0, E at
 * 90, Z vertical). All three are band-passed to `band` first. Convention is
 * calibrated against a synthetic retrograde wave in the test suite.
 *
 * For a retrograde Rayleigh wave, Hilbert(Z) correlates with the radial
 * component pointing TOWARD the source. We sweep the trial azimuth and take the
 * angle of maximum positive correlation as the back-azimuth.
 */
export function rayleighBackAzimuth(
  z: number[] | Float64Array,
  n: number[] | Float64Array,
  e: number[] | Float64Array,
  fs: number,
  band: [number, number],
): BackAzimuthResult {
  const zb = bandpass(z, fs, band[0], band[1]);
  const nb = bandpass(n, fs, band[0], band[1]);
  const eb = bandpass(e, fs, band[0], band[1]);
  // Trim bandpass edge ringing (~8% each end).
  const full = Math.min(zb.length, nb.length, eb.length);
  const trim = Math.floor(full * 0.08);
  const lo = trim;
  const hi = full - trim;
  const zhAll = hilbert(zb);
  // Mean-remove Hilbert(Z) over the analysis span.
  let zm = 0;
  for (let i = lo; i < hi; i++) zm += zhAll[i];
  zm /= hi - lo;
  let zz = 0;
  for (let i = lo; i < hi; i++) zz += (zhAll[i] - zm) ** 2;
  const zstd = Math.sqrt(zz);

  // Amplitude-weighted projection: maximize the cross-covariance of Hilbert(Z)
  // with the radial. Pearson would normalize out the radial's |cos(θ−φ)|
  // amplitude (a flat ±1 step), so we keep the radial amplitude and only
  // normalize by Z — the projection then PEAKS (positive) at the back-azimuth.
  let bestProj = -Infinity;
  let bestPosAz = 0;
  for (let deg = 0; deg < 360; deg++) {
    const th = (deg * Math.PI) / 180;
    const c = Math.cos(th);
    const s = Math.sin(th);
    let rm = 0;
    for (let i = lo; i < hi; i++) rm += nb[i] * c + eb[i] * s;
    rm /= hi - lo;
    let proj = 0;
    for (let i = lo; i < hi; i++) proj += (zhAll[i] - zm) * (nb[i] * c + eb[i] * s - rm);
    const p = proj / zstd; // normalize by Z only → radial amplitude preserved
    if (p > bestProj) {
      bestProj = p;
      bestPosAz = deg;
    }
  }
  // Quality: full Pearson correlation at the chosen back-azimuth.
  const th = (bestPosAz * Math.PI) / 180;
  const radial = new Float64Array(hi - lo);
  for (let i = lo; i < hi; i++) radial[i - lo] = nb[i] * Math.cos(th) + eb[i] * Math.sin(th);
  const quality = Math.abs(corr(zhAll.slice(lo, hi), radial));
  return {
    backAzimuth: bestPosAz,
    axisDeg: bestPosAz % 180,
    quality,
    retrograde: bestProj > 0,
  };
}

/** Great-circle back-azimuth from (lat1,lon1) to (lat2,lon2), degrees 0–360. */
export function greatCircleAzimuth(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const d = Math.PI / 180;
  const dLon = (lon2 - lon1) * d;
  const y = Math.sin(dLon) * Math.cos(lat2 * d);
  const x = Math.cos(lat1 * d) * Math.sin(lat2 * d) - Math.sin(lat1 * d) * Math.cos(lat2 * d) * Math.cos(dLon);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}
