// SPDX-License-Identifier: MIT
// Spectral analysis for the long-period microseism — the engine behind the
// real-data proof. Pure, deterministic, dependency-free TypeScript.
//
// The "26-second pulse" is a narrowband, low-amplitude spectral line that is
// NOT the dominant feature of a raw seismic record (the secondary microseism at
// ~6-16 s dwarfs it). Isolating it the way the literature does requires:
//   1. Welch averaging over many overlapping windows (variance reduction), and
//   2. MEDIAN averaging across segments to reject earthquake transients, and
//   3. spectral whitening (divide by a smoothed background) to expose the line.
// That is exactly what this module implements.

import type { SeismicWindow } from './types.js';

/** In-place iterative radix-2 Cooley-Tukey FFT. `re`/`im` length must be a power of 2. */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  if ((n & (n - 1)) !== 0) throw new Error(`fft: length ${n} is not a power of 2`);
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = i + k + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/** Remove the best-fit line (least squares) from a segment — kills tides/drift. */
export function detrend(seg: number[] | Float64Array): Float64Array {
  const n = seg.length;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += i;
    sy += seg[i];
    sxx += i * i;
    sxy += i * seg[i];
  }
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const intercept = (sy - slope * sx) / n;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = seg[i] - (intercept + slope * i);
  return out;
}

function hann(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export interface Psd {
  freqs: Float64Array;
  psd: Float64Array;
  segments: number;
  fs: number;
}

export interface WelchOptions {
  fs?: number;
  segment?: number;
  overlap?: number;
  average?: 'median' | 'mean';
}

/**
 * Welch power spectral density. `average:'median'` (default) is what makes the
 * 26 s line survive: it rejects earthquake transients that would otherwise
 * inflate the mean of a handful of segments.
 */
export function welchPsd(samples: number[] | Float64Array, opts: WelchOptions = {}): Psd {
  const fs = opts.fs ?? 1.0;
  const N = opts.segment ?? 4096;
  if ((N & (N - 1)) !== 0) throw new Error(`welchPsd: segment ${N} must be a power of 2`);
  const hop = N - (opts.overlap ?? N / 2);
  const win = hann(N);
  let winNorm = 0;
  for (let i = 0; i < N; i++) winNorm += win[i] * win[i];
  const half = N / 2;
  const perBin: number[][] = Array.from({ length: half }, () => []);
  let segCount = 0;
  for (let start = 0; start + N <= samples.length; start += hop) {
    const seg = detrend(Array.prototype.slice.call(samples, start, start + N));
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    for (let i = 0; i < N; i++) re[i] = seg[i] * win[i];
    fft(re, im);
    for (let k = 0; k < half; k++) {
      perBin[k].push((re[k] * re[k] + im[k] * im[k]) / (fs * winNorm));
    }
    segCount++;
  }
  const psd = new Float64Array(half);
  const freqs = new Float64Array(half);
  const useMedian = (opts.average ?? 'median') === 'median';
  for (let k = 0; k < half; k++) {
    freqs[k] = (k * fs) / N;
    if (perBin[k].length === 0) continue;
    psd[k] = useMedian
      ? median(perBin[k])
      : perBin[k].reduce((a, b) => a + b, 0) / perBin[k].length;
  }
  return { freqs, psd, segments: segCount, fs };
}

/**
 * Spectral whitening: divide each bin by a running-median background so narrow
 * persistent lines stand out as values > 1, independent of the steep
 * microseism continuum. Returns the prominence curve (same length as psd).
 */
export function whiten(psd: Float64Array, halfWindow = 15): Float64Array {
  const out = new Float64Array(psd.length);
  for (let k = 0; k < psd.length; k++) {
    const lo = Math.max(0, k - halfWindow);
    const hi = Math.min(psd.length - 1, k + halfWindow);
    const bg: number[] = [];
    for (let j = lo; j <= hi; j++) if (psd[j] > 0) bg.push(psd[j]);
    const b = bg.length ? median(bg) : 0;
    out[k] = b > 0 ? psd[k] / b : 0;
  }
  return out;
}

export interface SpectralLine {
  periodS: number;
  freqHz: number;
  prominence: number;
  psd: number;
}

/**
 * Find the most prominent persistent spectral line inside a frequency band,
 * using whitened prominence (PSD / smoothed background). This is the detector
 * used to prove the 26 s pulse in real data.
 */
export function findPersistentLine(
  spec: Psd,
  bandHz: [number, number],
  halfWindow = 15,
): SpectralLine | null {
  const w = whiten(spec.psd, halfWindow);
  let bestK = -1;
  let bestProm = -Infinity;
  for (let k = 1; k < spec.freqs.length - 1; k++) {
    const f = spec.freqs[k];
    if (f < bandHz[0] || f > bandHz[1]) continue;
    // Require a local maximum so we report a line, not a slope.
    if (w[k] >= w[k - 1] && w[k] >= w[k + 1] && w[k] > bestProm) {
      bestProm = w[k];
      bestK = k;
    }
  }
  if (bestK < 0) return null;
  return {
    periodS: 1 / spec.freqs[bestK],
    freqHz: spec.freqs[bestK],
    prominence: bestProm,
    psd: spec.psd[bestK],
  };
}

/**
 * Zero-phase FFT band-pass filter. Pads to the next power of two, zeroes all
 * bins outside [loHz, hiHz] (and their conjugate mirror), inverse-transforms,
 * and returns the real part truncated to the original length. Used to isolate
 * the long-period band before running the per-event detector on real data.
 */
export function bandpass(
  samples: number[] | Float64Array,
  fs: number,
  loHz: number,
  hiHz: number,
): number[] {
  const n0 = samples.length;
  let n = 1;
  while (n < n0) n <<= 1;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const dt = detrend(Array.prototype.slice.call(samples, 0, n0));
  for (let i = 0; i < n0; i++) re[i] = dt[i];
  fft(re, im);
  for (let k = 0; k <= n / 2; k++) {
    const f = (k * fs) / n;
    if (f < loHz || f > hiHz) {
      re[k] = 0;
      im[k] = 0;
      const mk = (n - k) % n; // conjugate-symmetric mirror
      re[mk] = 0;
      im[mk] = 0;
    }
  }
  fft(re, im, true);
  const out = new Array<number>(n0);
  for (let i = 0; i < n0; i++) out[i] = re[i];
  return out;
}

/** Convenience: median Welch PSD straight from a SeismicWindow. */
export function windowPsd(win: SeismicWindow, opts: WelchOptions = {}): Psd {
  return welchPsd(win.samples, { fs: win.sampleRateHz, ...opts });
}
