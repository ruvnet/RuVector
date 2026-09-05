// Drive Darwin-Mode `evolve` for earth-pulse-harness with the OpenRouter LLM mutator.
//
// The OpenRouter mutator is library-only (not exposed by the `metaharness-darwin`
// CLI), so this small driver wires it into the evolve engine. The API key is read
// by the mutator from OPENROUTER_API_KEY (env) — `evolve-openrouter.sh` populates
// that from GCP Secret Manager at runtime; the key is never stored in the repo.
//
// Resolution: prefer the installed `@metaharness/darwin` devDependency; fall back
// to DARWIN_DIST=<path/to/darwin-mode/dist/index.js> for monorepo/local runs.
//
//   node scripts/evolve-openrouter.mjs [harness-dir]
//
// Env: GENERATIONS, CHILDREN, SANDBOX(real|mock|agent), DARWIN_MUTATOR_MODEL,
//      CONCURRENCY, SEED.

import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

let darwin;
try {
  darwin = await import('@metaharness/darwin');
} catch {
  const dist = process.env.DARWIN_DIST;
  if (!dist) {
    console.error(
      'evolve-openrouter: install @metaharness/darwin (npm i) or set DARWIN_DIST to its dist/index.js',
    );
    process.exit(1);
  }
  darwin = await import(dist);
}
const { evolve, OpenRouterMutator } = darwin;

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(process.argv[2] || resolve(here, '..'));

const generations = Number(process.env.GENERATIONS || '2');
const children = Number(process.env.CHILDREN || '3');
const sandboxMode = process.env.SANDBOX || 'real';
const model = process.env.DARWIN_MUTATOR_MODEL || 'google/gemini-2.5-flash';

if (!process.env.OPENROUTER_API_KEY) {
  console.error('evolve-openrouter: OPENROUTER_API_KEY not set (use evolve-openrouter.sh to source it from GCP).');
  process.exit(1);
}

const generator = new OpenRouterMutator({ model, maxTokens: 1800, temperature: 0.4 });

const t0 = process.hrtime.bigint();
const result = await evolve({
  repoRoot,
  workRoot: `${repoRoot}/.metaharness/work`,
  generations,
  childrenPerGeneration: children,
  concurrency: Number(process.env.CONCURRENCY || '2'),
  seed: Number(process.env.SEED || '26'),
  promotionDelta: 0.05,
  tasks: [
    'run repository test suite',
    'verify generated harness safety',
    'check detection F1 and held-out amplitude prediction',
    'verify no test-window leakage and every cited claim maps to a source',
  ],
  sandboxMode,
  generator,
  tieBreaker: 'insertion',
  selection: 'score',
});
const ms = Number(process.hrtime.bigint() - t0) / 1e6;

console.log(
  'EVOLVE_RESULT ' +
    JSON.stringify(
      {
        model,
        sandboxMode,
        generations,
        children,
        wallMs: Math.round(ms),
        baselineScore: result?.baseline?.score,
        winnerScore: result?.winner?.score,
        improved: (result?.winner?.score ?? -Infinity) > (result?.baseline?.score ?? Infinity),
        winnerLineage: result?.winnerLineage,
        mutatorTelemetry: generator.telemetry,
      },
      null,
      2,
    ),
);
