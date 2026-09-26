// SPDX-License-Identifier: MIT
// Reproduce the resonator/decoupling discovery from REAL data.
//
// For each month of each year, fetch a few short GT.DBIC LHZ windows from the
// IRIS FDSN service, measure the 26 s line (peak frequency + excess power) and
// the secondary/primary microseism bands, then compute the resonance and
// decoupling statistics. Writes data/seismic/dbic-climatology-<span>.json.
//
//   node scripts/climatology.mjs [year1,year2,...]
//   node scripts/climatology.mjs 1996,1997
//
// Requires `npm run build`. No fabrication — every metric is computed from real
// observations; the raw windows are cached under data/seismic/clim/.

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { lineMetrics, resonanceStats, pearson, permutationP, seasonalPhase } from '../dist/climatology.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CACHE = resolve(root, 'data/seismic/clim');
mkdirSync(CACHE, { recursive: true });
const years = (process.argv[2] || '1996,1997').split(',').map(Number);
const DAY_STARTS = [6, 14, 22];
const base = 'https://service.iris.edu/irisws/timeseries/1/query?net=GT&sta=DBIC&loc=--&cha=LHZ&output=ascii&format=ascii';
const pad = (n) => String(n).padStart(2, '0');

function fetchWin(y, m, d, days) {
  const end = new Date(Date.UTC(y, m - 1, d + days));
  const e = `${end.getUTCFullYear()}-${pad(end.getUTCMonth() + 1)}-${pad(end.getUTCDate())}`;
  const url = `${base}&starttime=${y}-${pad(m)}-${pad(d)}T00:00:00&endtime=${e}T00:00:00`;
  try {
    const out = execFileSync('curl', ['-sS', '--max-time', '120', url], { maxBuffer: 1 << 28, encoding: 'utf8' });
    return out.startsWith('TIMESERIES') ? out : null;
  } catch {
    return null;
  }
}
function parse(t) {
  const x = [];
  for (const l of t.split('\n')) {
    if (!l || l[0] === 'T') continue;
    const v = parseFloat(l.trim().split(/\s+/)[1]);
    if (Number.isFinite(v)) x.push(v);
  }
  return x;
}

const windows = [];
for (const y of years) {
  for (let m = 1; m <= 12; m++) {
    for (const d of DAY_STARTS) {
      const f = resolve(CACHE, `${y}-${pad(m)}-${pad(d)}.ascii`);
      let txt = existsSync(f) && readFileSync(f, 'utf8').startsWith('TIMESERIES') ? readFileSync(f, 'utf8') : null;
      if (!txt) {
        txt = fetchWin(y, m, d, 2);
        if (txt) writeFileSync(f, txt);
      }
      if (!txt) continue;
      const x = parse(txt);
      if (x.length < 120000) continue;
      const me = lineMetrics(x, 1.0);
      if (me.lineExcess <= 0) continue;
      windows.push({
        year: y, month: m, day: d, samples: x.length,
        peakFreqHz: +me.peakFreqHz.toFixed(6), periodS: +me.periodS.toFixed(3),
        lineExcess: +me.lineExcess.toPrecision(4), snr: +me.snr.toFixed(3),
        secondary: +me.secondary.toPrecision(4), primary: +me.primary.toPrecision(4),
      });
      console.log(`${y}-${pad(m)}-${pad(d)}: ${me.periodS.toFixed(2)}s excess=${me.lineExcess.toExponential(2)} sec=${me.secondary.toExponential(2)}`);
    }
  }
}

const stats = resonanceStats(windows.map((w) => ({ peakFreqHz: w.peakFreqHz, periodS: w.periodS, lineExcess: w.lineExcess, snr: w.snr, secondary: w.secondary, primary: w.primary })));
const L = windows.map((w) => w.lineExcess);
const S = windows.map((w) => w.secondary);
const monthlyLine = {};
const monthlySec = {};
for (let m = 1; m <= 12; m++) {
  const lm = windows.filter((w) => w.month === m).map((w) => w.lineExcess);
  const sm = windows.filter((w) => w.month === m).map((w) => w.secondary);
  if (lm.length) { monthlyLine[m] = lm.reduce((a, b) => a + b, 0) / lm.length; monthlySec[m] = sm.reduce((a, b) => a + b, 0) / sm.length; }
}
const out = {
  station: 'GT.DBIC', channel: 'LHZ', source: 'IRIS/EarthScope FDSN timeseries',
  span: `${Math.min(...years)}-${Math.max(...years)}`, windowDays: 2, n: windows.length,
  resonance: {
    meanFreqHz: +stats.meanFreqHz.toFixed(6), meanPeriodS: +(1 / stats.meanFreqHz).toFixed(2),
    stdFreqHz: +stats.stdFreqHz.toExponential(3), freqCvPct: +(100 * stats.freqCv).toFixed(2),
    amplitudeRange: +stats.amplitudeRange.toFixed(1), freqAmpCorr: +stats.freqAmpCorr.toFixed(3),
  },
  decoupling: { corr_line_secondary: +pearson(L, S).toFixed(3), permutation_p: +permutationP(L, S, 26, 2000).toFixed(3) },
  seasonal: { line: seasonalPhase(monthlyLine), secondary: seasonalPhase(monthlySec) },
  windows,
};
writeFileSync(resolve(root, `data/seismic/dbic-climatology-${out.span}.json`), JSON.stringify(out, null, 1));
console.log('\n=== DISCOVERY METRICS (real GT.DBIC data) ===');
console.log(`n=${out.n} windows  mean period ${out.resonance.meanPeriodS}s`);
console.log(`frequency stability: CV ${out.resonance.freqCvPct}%   amplitude range ${out.resonance.amplitudeRange}x   freq-amp corr ${out.resonance.freqAmpCorr}`);
console.log(`decoupling from secondary microseism: r=${out.decoupling.corr_line_secondary}  perm-p=${out.decoupling.permutation_p}`);
