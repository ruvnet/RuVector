// SPDX-License-Identifier: MIT
// Reproducibly fetch REAL seismic data for the 26-second pulse proof.
//
// Source: IRIS/EarthScope FDSN `timeseries` web service (public, no auth).
// Station: GT.DBIC (Dimbokro, Côte d'Ivoire) — a borehole broadband station
// ~on the Gulf of Guinea coast, i.e. the closest permanent station to the
// 26 s microseism source region. Channel LHZ (1 sample/s, vertical).
// Window: boreal winter 1995 (the 26 s microseism is strongest Nov–Feb), a
// continuous GT.DBIC data segment (1994-12-29 → 1995-01-26).
//
// The harness NEVER fabricates observations. This script only downloads real,
// citable samples and writes them under data/seismic/ with full provenance.
//
//   node scripts/fetch-dbic.mjs [outDir] [startDay] [numDays]
//   node scripts/fetch-dbic.mjs data/seismic 1995-01-02 12

import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');
const outDir = resolve(repoRoot, process.argv[2] || 'data/seismic/raw');
const startDay = process.argv[3] || '1995-01-02';
const numDays = Number(process.argv[4] || '12');

const NET = 'GT';
const STA = 'DBIC';
const LOC = '--';
const CHA = 'LHZ';
const BASE = 'https://service.iris.edu/irisws/timeseries/1/query';

function addDays(iso, n) {
  const [y, m, d] = iso.split('-').map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d + n));
  return dt.toISOString().slice(0, 10);
}

function fetchDay(day) {
  const end = addDays(day, 1);
  const url =
    `${BASE}?net=${NET}&sta=${STA}&loc=${LOC}&cha=${CHA}` +
    `&starttime=${day}T00:00:00&endtime=${end}T00:00:00&output=ascii&format=ascii`;
  // curl honors the environment proxy + CA bundle in this runtime.
  const out = execFileSync('curl', ['-sS', '--max-time', '180', url], {
    maxBuffer: 1 << 28,
    encoding: 'utf8',
  });
  if (!out.startsWith('TIMESERIES')) {
    throw new Error(`no data for ${day}: ${out.slice(0, 120)}`);
  }
  return { url, body: out };
}

mkdirSync(outDir, { recursive: true });
const provenance = {
  source: 'IRIS/EarthScope FDSN timeseries web service',
  network: NET,
  station: STA,
  location: LOC,
  channel: CHA,
  sampleRateHz: 1.0,
  units: 'counts (raw)',
  station_metadata:
    'https://service.iris.edu/fdsnws/station/1/query?net=GT&sta=DBIC&cha=LHZ&level=channel&format=text',
  note: 'GT.DBIC borehole, lat 6.67016, lon -4.85656 — Gulf of Guinea coast.',
  days: [],
};

let total = 0;
for (let i = 0; i < numDays; i++) {
  const day = addDays(startDay, i);
  const file = join(outDir, `${NET}.${STA}.${CHA}.${day}.ascii`);
  if (existsSync(file) && readFileSync(file, 'utf8').startsWith('TIMESERIES')) {
    console.log(`cached ${day}`);
    provenance.days.push({ day, file: file.replace(repoRoot + '/', '') });
    continue;
  }
  const { url, body } = fetchDay(day);
  writeFileSync(file, body);
  const samples = body.split('\n').filter((l) => l && l[0] !== 'T').length;
  total += samples;
  provenance.days.push({ day, file: file.replace(repoRoot + '/', ''), url, samples });
  console.log(`fetched ${day}: ${samples} samples`);
}

writeFileSync(join(outDir, 'PROVENANCE.json'), JSON.stringify(provenance, null, 2));
console.log(`\nWrote ${provenance.days.length} day(s), ~${total} new samples to ${outDir}`);
console.log(`Provenance: ${join(outDir, 'PROVENANCE.json')}`);
