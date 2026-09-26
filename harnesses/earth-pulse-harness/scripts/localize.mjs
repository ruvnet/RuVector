// SPDX-License-Identifier: MIT
// Locate the 27.7 s source DIRECTION from a single station, from real data.
//
// Fetches GT.DBIC three-component (LHZ/LHN/LHE) data, band-passes to the line,
// and computes the Rayleigh-wave back-azimuth per window — the bearing from the
// station to the source. Writes data/seismic/dbic-backazimuth-<year>.json.
//
//   node scripts/localize.mjs [YYYY-MM-DD startDay] [numDays]
//   node scripts/localize.mjs 1996-06-15 5
//
// Requires `npm run build`. No fabrication — real IRIS observations only.

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { rayleighBackAzimuth, greatCircleAzimuth } from '../dist/polarization.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CACHE = resolve(root, 'data/seismic/3comp');
mkdirSync(CACHE, { recursive: true });
const start = (process.argv[2] || '1996-06-15').split('-').map(Number);
const numDays = Number(process.argv[3] || '5');
const BAND = [0.0350, 0.0372];
const pad = (n) => String(n).padStart(2, '0');

function fetchCh(cha, y, m, d) {
  const f = resolve(CACHE, `${cha}.${y}-${pad(m)}-${pad(d)}.ascii`);
  if (existsSync(f) && readFileSync(f, 'utf8').startsWith('TIMESERIES')) return readFileSync(f, 'utf8');
  const e = new Date(Date.UTC(y, m - 1, d + 1));
  const url =
    `https://service.iris.edu/irisws/timeseries/1/query?net=GT&sta=DBIC&loc=--&cha=${cha}` +
    `&starttime=${y}-${pad(m)}-${pad(d)}T00:00:00&endtime=${e.getUTCFullYear()}-${pad(e.getUTCMonth() + 1)}-${pad(e.getUTCDate())}T00:00:00&output=ascii&format=ascii`;
  try {
    const o = execFileSync('curl', ['-sS', '--max-time', '120', url], { maxBuffer: 1 << 28, encoding: 'utf8' });
    if (o.startsWith('TIMESERIES')) { writeFileSync(f, o); return o; }
  } catch { /* gap */ }
  return null;
}
const parse = (t) => {
  const x = [];
  for (const l of t.split('\n')) { if (!l || l[0] === 'T') continue; const v = parseFloat(l.trim().split(/\s+/)[1]); if (Number.isFinite(v)) x.push(v); }
  return x;
};

const Z = [];
const N = [];
const E = [];
for (let i = 0; i < numDays; i++) {
  const dt = new Date(Date.UTC(start[0], start[1] - 1, start[2] + i));
  const [y, m, d] = [dt.getUTCFullYear(), dt.getUTCMonth() + 1, dt.getUTCDate()];
  const z = fetchCh('LHZ', y, m, d);
  const n = fetchCh('LHN', y, m, d);
  const e = fetchCh('LHE', y, m, d);
  if (z && n && e) {
    const zp = parse(z); const np = parse(n); const ep = parse(e);
    const L = Math.min(zp.length, np.length, ep.length);
    if (L >= 80000) { Z.push(...zp.slice(0, L)); N.push(...np.slice(0, L)); E.push(...ep.slice(0, L)); }
  }
}
console.log(`3-component samples: ${Z.length} (${(Z.length / 86400).toFixed(1)} days)`);

const whole = rayleighBackAzimuth(Z, N, E, 1.0, BAND);
const W = 8192; const hop = 4096; const baz = [];
for (let s = 0; s + W <= Z.length; s += hop) {
  const r = rayleighBackAzimuth(Z.slice(s, s + W), N.slice(s, s + W), E.slice(s, s + W), 1.0, BAND);
  if (r.quality > 0.35) baz.push({ az: r.backAzimuth, q: r.quality });
}
let sx = 0; let sy = 0; let sw = 0;
for (const b of baz) { const a = (b.az * Math.PI) / 180; sx += b.q * Math.cos(a); sy += b.q * Math.sin(a); sw += b.q; }
const meanAz = ((Math.atan2(sy, sx) * 180) / Math.PI + 360) % 360;
const R = Math.hypot(sx, sy) / sw;

const out = {
  test: 'single-station Rayleigh-wave back-azimuth of the 27.7s line',
  station: 'GT.DBIC', window: `${process.argv[2] || '1996-06-15'} +${numDays}d`, band_Hz: BAND,
  measured: { whole_record_baz: whole.backAzimuth, vector_mean_baz: +meanAz.toFixed(1), concentration_R: +R.toFixed(3), windows: baz.length, quality: +whole.quality.toFixed(3), retrograde: whole.retrograde },
  expected: { sao_tome: +greatCircleAzimuth(6.67016, -4.85656, 0.34, 6.73).toFixed(0), bight_of_bonny: +greatCircleAzimuth(6.67016, -4.85656, 2.5, 7.0).toFixed(0) },
};
writeFileSync(resolve(root, `data/seismic/dbic-backazimuth-${start[0]}.json`), JSON.stringify(out, null, 1));
console.log(`back-azimuth: whole ${whole.backAzimuth}°, mean ${meanAz.toFixed(0)}° (R=${R.toFixed(2)}, ${baz.length} windows), retrograde=${whole.retrograde}`);
console.log(`expected Gulf of Guinea: Bight of Bonny ${out.expected.bight_of_bonny}°, São Tomé ${out.expected.sao_tome}°`);
