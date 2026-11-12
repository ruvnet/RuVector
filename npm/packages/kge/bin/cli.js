#!/usr/bin/env node
'use strict';

/**
 * `kge` CLI entry (plain Node, no transpile). Every subcommand except `bench`
 * is handled by ../dist/cli/main.js. `bench` lives here because it must load
 * bench/run.mjs with a real dynamic import() — one that a CommonJS build step
 * cannot rewrite into require(). No child_process, no network.
 */

const path = require('path');
const fs = require('fs');
const { pathToFileURL } = require('url');

async function runBench(argv) {
  const benchPath = path.join(__dirname, '..', 'bench', 'run.mjs');
  if (!fs.existsSync(benchPath)) {
    process.stderr.write(
      'bench harness not found at bench/run.mjs.\n' +
        'It ships with the bench track; see docs/adr/ADR-006 and the README "Measured" section.\n',
    );
    process.exit(2);
    return;
  }
  const mod = await import(pathToFileURL(benchPath).href);
  const entry = typeof mod.run === 'function' ? mod.run : mod.main;
  if (typeof entry === 'function') {
    const code = await entry(argv, {});
    process.exit(typeof code === 'number' ? code : 0);
    return;
  }
  process.exit(0);
}

function loadMain() {
  try {
    return require('../dist/cli/main.js').main;
  } catch (e) {
    process.stderr.write('Failed to load @ruvector/kge CLI. Run: npm run build\n');
    process.stderr.write(String((e && e.message) || e) + '\n');
    process.exit(1);
    return null;
  }
}

async function cli() {
  const argv = process.argv.slice(2);
  if (argv[0] === 'bench') {
    return runBench(argv.slice(1));
  }
  const main = loadMain();
  const result = await main(argv);
  // For `serve`, the server holds the event loop open; do not exit.
  if (!result || !result.serve) {
    process.exit(result ? result.code : 0);
  }
}

if (require.main === module) {
  cli().catch((e) => {
    process.stderr.write('error: ' + String((e && e.message) || e) + '\n');
    process.exit(1);
  });
}

module.exports = { cli };
