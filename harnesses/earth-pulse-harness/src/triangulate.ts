// SPDX-License-Identifier: MIT
// Triangulate the 26 s source LOCATION by intersecting single-station
// back-azimuths from several stations. Pure, deterministic.
//
// Each station gives a bearing (great-circle azimuth) toward the source. The
// source is the point that best fits all bearings — found by minimizing the
// weighted sum of squared angular residuals over a lat/lon grid (refined). With
// ≥ 2 well-separated bearings this turns a set of directions into a coordinate.

import { greatCircleAzimuth } from './polarization.js';

export interface StationBearing {
  name: string;
  lat: number;
  lon: number;
  backAzimuth: number; // degrees, station→source
  weight?: number;     // e.g. polarization concentration R (default 1)
}

export interface TriangulationResult {
  lat: number;
  lon: number;
  /** RMS angular residual (degrees) at the solution. */
  rmsResidualDeg: number;
  /** Per-station residual (measured − predicted bearing). */
  residuals: { name: string; residualDeg: number }[];
  /** Approx 1-σ location radius (km) from the residual curvature. */
  uncertaintyKm: number;
}

function angDiff(a: number, b: number): number {
  let d = Math.abs(a - b) % 360;
  if (d > 180) d = 360 - d;
  return d;
}

function misfit(stations: StationBearing[], lat: number, lon: number): number {
  let s = 0;
  let w = 0;
  for (const st of stations) {
    const pred = greatCircleAzimuth(st.lat, st.lon, lat, lon);
    const wi = st.weight ?? 1;
    const r = angDiff(st.backAzimuth, pred);
    s += wi * r * r;
    w += wi;
  }
  return s / w; // weighted mean squared residual
}

const KM_PER_DEG = 111.195;

/**
 * Grid-search + refine the source location. `bounds` defaults to a wide
 * Atlantic/Africa box around the Gulf of Guinea.
 */
export function triangulate(
  stations: StationBearing[],
  bounds: { latMin: number; latMax: number; lonMin: number; lonMax: number } = {
    latMin: -30, latMax: 35, lonMin: -35, lonMax: 35,
  },
): TriangulationResult {
  if (stations.length < 2) throw new Error('triangulate: need at least 2 stations');
  let best = { lat: 0, lon: 0, m: Infinity };
  // Coarse grid, then two refinement passes.
  let { latMin, latMax, lonMin, lonMax } = bounds;
  let step = Math.max((latMax - latMin) / 100, (lonMax - lonMin) / 100);
  for (let pass = 0; pass < 3; pass++) {
    for (let la = latMin; la <= latMax; la += step) {
      for (let lo = lonMin; lo <= lonMax; lo += step) {
        const m = misfit(stations, la, lo);
        if (m < best.m) best = { lat: la, lon: lo, m };
      }
    }
    latMin = best.lat - step * 4; latMax = best.lat + step * 4;
    lonMin = best.lon - step * 4; lonMax = best.lon + step * 4;
    step /= 8;
  }
  const rms = Math.sqrt(best.m);
  const residuals = stations.map((st) => ({
    name: st.name,
    residualDeg: +(((st.backAzimuth - greatCircleAzimuth(st.lat, st.lon, best.lat, best.lon) + 540) % 360) - 180).toFixed(1),
  }));
  // Crude uncertainty: how far must we move (in km) to raise RMS by 1°?
  const d = 0.25;
  const dm = (misfit(stations, best.lat + d, best.lon) + misfit(stations, best.lat, best.lon + d)) / 2 - best.m;
  const curv = Math.sqrt(Math.max(dm, 1e-6)) / (d * KM_PER_DEG);
  const uncertaintyKm = curv > 0 ? Math.round(1 / curv) : 0;
  return {
    lat: +best.lat.toFixed(2),
    lon: +best.lon.toFixed(2),
    rmsResidualDeg: +rms.toFixed(1),
    residuals,
    uncertaintyKm,
  };
}

/** Great-circle distance (km) between two points. */
export function distanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const d = Math.PI / 180;
  const a =
    Math.sin(((lat2 - lat1) * d) / 2) ** 2 +
    Math.cos(lat1 * d) * Math.cos(lat2 * d) * Math.sin(((lon2 - lon1) * d) / 2) ** 2;
  return 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}
