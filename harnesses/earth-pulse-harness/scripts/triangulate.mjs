// SPDX-License-Identifier: MIT
// Attempt to triangulate the 27.7 s source from multiple stations' back-azimuths.
//
// Measures the Rayleigh-wave back-azimuth at each station, then triangulates
// using only bearings with polarization concentration R above a threshold (so
// stations that do not cleanly track the 27.7 s line are excluded rather than
// corrupting the solution). Writes data/seismic/triangulation-<year>.json.
//
//   node scripts/triangulate.mjs            # default June-1996 station set
// Requires `npm run build`. Real IRIS observations only.

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { rayleighBackAzimuth, greatCircleAzimuth } from '../dist/polarization.js';
import { triangulate, distanceKm } from '../dist/triangulate.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CACHE = resolve(root, 'data/seismic/3comp');
mkdirSync(CACHE, { recursive: true });
const BAND = [0.0350, 0.0372];
const R_MIN = 0.6; // only trust bearings this concentrated
const pad = (n) => String(n).padStart(2, '0');

const STATIONS = [
  { net: 'GT', sta: 'DBIC', loc: '--', lat: 6.67016, lon: -4.85656, start: [1996, 6, 15], days: 6 },
  { net: 'G', sta: 'TAM', loc: '--', lat: 22.79149, lon: 5.52838, start: [1996, 6, 12], days: 10 },
  { net: 'II', sta: 'ASCN', loc: '00', lat: -7.9327, lon: -14.3601, start: [1996, 6, 12], days: 10 },
  { net: 'GT', sta: 'LBTB', loc: '--', lat: -25.0151, lon: 25.5966, start: [1996, 6, 12], days: 10 },
];

function fetchCh(net, sta, loc, cha, y, m, d) {
  const f = resolve(CACHE, `${net}.${sta}.${cha}.${y}-${pad(m)}-${pad(d)}.ascii`);
  if (existsSync(f) && readFileSync(f, 'utf8').startsWith('TIMESERIES')) return readFileSync(f, 'utf8');
  const e = new Date(Date.UTC(y, m - 1, d + 1));
  const url =
    `https://service.iris.edu/irisws/timeseries/1/query?net=${net}&sta=${sta}&loc=${loc}&cha=${cha}` +
    `&starttime=${y}-${pad(m)}-${pad(d)}T00:00:00&endtime=${e.getUTCFullYear()}-${pad(e.getUTCMonth() + 1)}-${pad(e.getUTCDate())}T00:00:00&output=ascii&format=ascii`;
  try {
    const o = execFileSync('curl', ['-sS', '--max-time', '120', url], { maxBuffer: 1 << 28, encoding: 'utf8' });
    if (o.startsWith('TIMESERIES')) { writeFileSync(f, o); return o; }
  } catch { /* gap */ }
  return null;
}
const parse = (t) => { const x = []; for (const l of t.split('\n')) { if (!l || l[0] === 'T') continue; const v = parseFloat(l.trim().split(/\s+/)[1]); if (Number.isFinite(v)) x.push(v); } return x; };

function bearing(st) {
  const Z = []; const N = []; const E = [];
  for (let i = 0; i < st.days; i++) {
    const dt = new Date(Date.UTC(st.start[0], st.start[1] - 1, st.start[2] + i));
    const [y, m, d] = [dt.getUTCFullYear(), dt.getUTCMonth() + 1, dt.getUTCDate()];
    const z = fetchCh(st.net, st.sta, st.loc, 'LHZ', y, m, d);
    const n = fetchCh(st.net, st.sta, st.loc, 'LHN', y, m, d);
    const e = fetchCh(st.net, st.sta, st.loc, 'LHE', y, m, d);
    if (z && n && e) { const zp = parse(z); const np = parse(n); const ep = parse(e); const L = Math.min(zp.length, np.length, ep.length); if (L >= 80000) { Z.push(...zp.slice(0, L)); N.push(...np.slice(0, L)); E.push(...ep.slice(0, L)); } }
  }
  if (Z.length < 200000) return null;
  const W = 8192; const hop = 4096; const baz = [];
  for (let s = 0; s + W <= Z.length; s += hop) { const r = rayleighBackAzimuth(Z.slice(s, s + W), N.slice(s, s + W), E.slice(s, s + W), 1.0, BAND); if (r.quality > 0.35) baz.push({ az: r.backAzimuth, q: r.quality }); }
  let sx = 0; let sy = 0; let sw = 0;
  for (const b of baz) { const a = (b.az * Math.PI) / 180; sx += b.q * Math.cos(a); sy += b.q * Math.sin(a); sw += b.q; }
  return { name: st.sta, lat: st.lat, lon: st.lon, backAzimuth: +(((Math.atan2(sy, sx) * 180) / Math.PI + 360) % 360).toFixed(1), R: +(Math.hypot(sx, sy) / sw).toFixed(3) };
}

const bearings = [];
for (const st of STATIONS) {
  const b = bearing(st);
  if (!b) { console.log(`${st.sta}: insufficient data`); continue; }
  b.expected_to_source = +greatCircleAzimuth(st.lat, st.lon, 2, 7).toFixed(0);
  b.tracks_27_7s_source = b.R >= R_MIN;
  bearings.push(b);
  console.log(`${b.name}: baz ${b.backAzimuth}° R ${b.R} (expected ${b.expected_to_source}°) ${b.tracks_27_7s_source ? 'TRACKS' : 'unreliable'}`);
}

const reliable = bearings.filter((b) => b.tracks_27_7s_source);
let result;
if (reliable.length >= 2) {
  const r = triangulate(reliable.map((b) => ({ ...b, weight: b.R })));
  result = { ...r, distance_to_sao_tome_km: +distanceKm(r.lat, r.lon, 0.34, 6.73).toFixed(0) };
  console.log(`\nTRIANGULATED source: ${r.lat}, ${r.lon}  (±${r.uncertaintyKm} km)  ${result.distance_to_sao_tome_km} km from São Tomé`);
} else {
  result = null;
  console.log(`\nOnly ${reliable.length} station tracks the 27.7 s line (R≥${R_MIN}) — tight triangulation not possible. The source DIRECTION from ${reliable.map((b) => b.name).join(',') || 'DBIC'} is the robust constraint.`);
}
writeFileSync(resolve(root, 'data/seismic/triangulation-1996.json'), JSON.stringify({ band_Hz: BAND, R_MIN, bearings, triangulation: result }, null, 1));
