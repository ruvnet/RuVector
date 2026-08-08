#!/usr/bin/env node
/**
 * Fixture generator for `scripts/rvforge-parity-check.sh`.
 *
 * Writes the one input a publish needs that the CLI cannot produce on its own:
 * an `rvforge.json` that packs clean. The `.rvf` beside it is no longer written
 * here — `rvforge create` writes it, which is the point of the parity run.
 * Generating the artifact from literals would have tested the fixture rather
 * than the writer that ships.
 *
 * The scaffold comes from the CLI's own compiled `dist/project.js` rather than
 * from a hand-copied object, so it cannot drift from the validator that reads
 * it back.
 *
 * The shape mirrors `npm/packages/rvforge/tests/fixtures/project-fixture.ts`,
 * which is TypeScript and therefore not loadable from a plain node process.
 *
 * Usage: node scripts/rvforge-parity-fixture.cjs <cli-dist-dir> <out-dir>
 */

'use strict';

const { writeFileSync } = require('node:fs');
const { join, resolve } = require('node:path');

const [, , distDirArg, outDirArg] = process.argv;
if (!distDirArg || !outDirArg) {
  process.stderr.write('usage: rvforge-parity-fixture.cjs <cli-dist-dir> <out-dir>\n');
  process.exit(2);
}

const distDir = resolve(distDirArg);
const outDir = resolve(outDirArg);

const { defaultDenials, defaultProject } = require(join(distDir, 'project.js'));

/** Narrow, concrete scopes: nothing here trips a manual-review trigger. */
const requests = [
  { class: 'filesystem', scope: 'user-selected', rationale: 'reads only documents you choose' },
  { class: 'memory', scope: '512MiB', rationale: 'model working set' },
  { class: 'persistent-state', scope: 'encrypted-local', rationale: 'agent memory' },
];

const project = defaultProject({
  version: '1.0.0',
  license: 'Apache-2.0',
  listing: {
    name: 'analyst',
    displayName: 'Cognitum Analyst',
    description: 'Reads the documents you select and answers questions about them.',
    category: 'developer-tools',
    icon: null,
    screenshots: [],
    priceModel: 'free',
  },
  publisher: {
    displayName: 'RVForge Parity Check',
    identityEvidence: { method: 'manual-review', reference: 'scripts/rvforge-parity-check.sh' },
    contact: { support: 'support@example.invalid', privacyPolicy: 'https://example.invalid/privacy' },
    keyFile: null,
  },
  runtimeRequirements: {
    profiles: ['wasm'],
    systems: ['linux-x64', 'macos-universal', 'windows-x64'],
    memoryMiB: 512,
    rvmVersionMin: '0.1.0',
    stateSchemaVersion: 1,
    witnessSchemaVersion: 1,
  },
  capabilities: { defaultPolicy: 'deny', requests, denials: defaultDenials(requests) },
  modelManifest: { location: 'embedded', digests: [`sha256:${'a'.repeat(64)}`] },
  externalServices: [],
  state: { checkpointing: true, recovery: true, rollbackSafeUntilStateSchema: 2 },
});

const projectPath = join(outDir, 'rvforge.json');
writeFileSync(projectPath, `${JSON.stringify(project, null, 2)}\n`, 'utf8');

process.stdout.write(`${projectPath}\n`);
