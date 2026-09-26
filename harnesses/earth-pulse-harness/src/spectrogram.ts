// SPDX-License-Identifier: MIT
// Time–frequency spectrogram + a gliding-tremor search, with the contradiction
// test that keeps it honest. Pure, deterministic.
//
// Bruland & Hadziioannou (2023) report "gliding tremors" associated with the
// 26 s microseism — narrowband features that sweep UPWARD in frequency. A naive
// spectrogram ridge-tracker readily finds upward-drifting peaks above the
// fundamental — BUT in single-station data those are dominated by the broad
// secondary microseism (Q ≈ 4) whose peak frequency wanders with ocean
// conditions. The discriminator (ADR-004) is SOURCE COHERENCE: a true tremor of
// the 26 s source must arrive from the SAME back-azimuth as the fundamental
// (~100°, Gulf of Guinea). Candidates that do not are rejected.

import { welchPsd, whiten } from './spectrum.js';

export interface GlideCandidate {
  tStartH: number;
  tEndH: number;
  fStartHz: number;
  fEndHz: number;
  riseHz: number;
  durationH: number;
}

export interface GlideSearchResult {
  columns: number;
  candidates: GlideCandidate[];
  /** Bandwidth Q of the dominant feature in the search band (low ⇒ broad ⇒ microseism). */
  dominantQ: number;
  dominantPeriodS: number;
}

/**
 * Naive upward-glide search: track the strongest whitened peak per time column
 * and flag sustained monotonic upward runs. Returns candidates AND the dominant
 * feature's Q — a low Q means the band is dominated by the broad secondary
 * microseism, so any "glides" are suspect and MUST be confirmed by source
 * coherence (see `sourceCoherenceVerdict`).
 */
export function glideSearch(
  samples: number[] | Float64Array,
  fs: number,
  band: [number, number],
  opts: { windowS?: number; hopS?: number; segment?: number; promMin?: number; minRiseHz?: number; minDurH?: number } = {},
): GlideSearchResult {
  const W = opts.windowS ?? 2 * 3600;
  const hop = opts.hopS ?? 3600;
  const seg = opts.segment ?? 1024;
  const promMin = opts.promMin ?? 1.5;
  const minRise = opts.minRiseHz ?? 0.004;
  const minDur = opts.minDurH ?? 4;
  const track: { t: number; f: number; prom: number }[] = [];
  for (let start = 0; start + W <= samples.length; start += hop) {
    const sp = welchPsd(Array.prototype.slice.call(samples, start, start + W), { fs, segment: seg, overlap: seg / 2, average: 'median' });
    const w = whiten(sp.psd, 12);
    let pk = 0;
    let pf = 0;
    for (let k = 0; k < sp.freqs.length; k++) {
      const f = sp.freqs[k];
      if (f >= band[0] && f <= band[1] && w[k] > pk) { pk = w[k]; pf = f; }
    }
    track.push({ t: start / fs / 3600, f: pf, prom: pk });
  }
  const candidates: GlideCandidate[] = [];
  let i = 0;
  while (i < track.length) {
    if (track[i].prom < promMin) { i++; continue; }
    let j = i;
    while (j + 1 < track.length && track[j + 1].prom >= promMin && track[j + 1].f >= track[j].f - 0.001) j++;
    const rise = track[j].f - track[i].f;
    const dur = track[j].t - track[i].t;
    if (dur >= minDur && rise > minRise) {
      candidates.push({ tStartH: +track[i].t.toFixed(1), tEndH: +track[j].t.toFixed(1), fStartHz: +track[i].f.toFixed(4), fEndHz: +track[j].f.toFixed(4), riseHz: +rise.toFixed(4), durationH: +dur.toFixed(1) });
    }
    i = j + 1;
  }
  // Dominant-feature Q over the whole record.
  const full = welchPsd(samples, { fs, segment: 4096, overlap: 2048, average: 'median' });
  let pk = 0;
  let kp = 0;
  for (let k = 0; k < full.freqs.length; k++) {
    const f = full.freqs[k];
    if (f >= band[0] && f <= band[1] && full.psd[k] > pk) { pk = full.psd[k]; kp = k; }
  }
  let kl = kp;
  while (kl > 0 && full.psd[kl] > pk / 2) kl--;
  let kr = kp;
  while (kr < full.psd.length - 1 && full.psd[kr] > pk / 2) kr++;
  const res = full.freqs[1] - full.freqs[0];
  const fwhm = (kr - kl) * res;
  return {
    columns: track.length,
    candidates,
    dominantQ: fwhm > 0 ? +(full.freqs[kp] / fwhm).toFixed(1) : Infinity,
    dominantPeriodS: +(1 / full.freqs[kp]).toFixed(1),
  };
}

export interface CoherenceVerdict {
  confirmed: boolean;
  reason: string;
}

/**
 * The contradiction test (ADR-004): a glide candidate is a real tremor of the
 * 26 s source ONLY if its band is source-coherent from the fundamental's
 * back-azimuth. Given the per-band polarization (back-azimuth + concentration R)
 * of the fundamental and the candidate band, confirm or reject.
 */
export function sourceCoherenceVerdict(
  fundamental: { backAzimuth: number; R: number },
  candidateBand: { backAzimuth: number; R: number },
  opts: { minR?: number; maxAzDiff?: number } = {},
): CoherenceVerdict {
  const minR = opts.minR ?? 0.5;
  const maxAz = opts.maxAzDiff ?? 30;
  let d = Math.abs(fundamental.backAzimuth - candidateBand.backAzimuth) % 360;
  if (d > 180) d = 360 - d;
  if (candidateBand.R < minR) {
    return { confirmed: false, reason: `candidate band is not source-coherent (R=${candidateBand.R.toFixed(2)} < ${minR}) — consistent with the broad secondary microseism, not a fixed source` };
  }
  if (d > maxAz) {
    return { confirmed: false, reason: `candidate band arrives from ${candidateBand.backAzimuth.toFixed(0)}°, ${d.toFixed(0)}° from the fundamental's ${fundamental.backAzimuth.toFixed(0)}° — a different source` };
  }
  return { confirmed: true, reason: `candidate band is source-coherent (R=${candidateBand.R.toFixed(2)}) and co-located in azimuth (Δ=${d.toFixed(0)}°) with the fundamental` };
}
