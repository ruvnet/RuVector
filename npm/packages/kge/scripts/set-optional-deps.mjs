#!/usr/bin/env node
// Inject exact-pinned optionalDependencies for the five @ruvector/kge platform
// packages into a package.json. Used only by .github/workflows/build-kge.yml,
// on the manifest that is packed for publish, after the platform packages have
// been built (and, for a real publish, before the meta package goes out only
// once all five are verified on the registry).
//
// The committed package.json deliberately does not carry these yet:
// regression-guard.yml fails any optionalDependency that is not on npm, and
// ADR-007 §5 / ADR-001 §5 put the committed declaration in a follow-up PR.
//
//   node scripts/set-optional-deps.mjs <package.json> <version>

import { readFileSync, writeFileSync } from 'node:fs';

export const PLATFORMS = ['linux-x64-gnu', 'linux-arm64-gnu', 'darwin-x64', 'darwin-arm64', 'win32-x64-msvc'];

const SEMVER = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/;

const [file, version] = process.argv.slice(2);
if (!file || !version) {
  console.error('usage: set-optional-deps.mjs <package.json> <version>');
  process.exit(2);
}
if (!SEMVER.test(version)) {
  console.error(`not an exact semver version: ${version}`);
  process.exit(2);
}

const pkg = JSON.parse(readFileSync(file, 'utf8'));
if (pkg.name !== '@ruvector/kge') {
  console.error(`refusing to edit ${pkg.name}; expected @ruvector/kge`);
  process.exit(1);
}
if (pkg.version !== version) {
  console.error(`package.json is ${pkg.version} but platform packages are ${version}`);
  process.exit(1);
}

pkg.optionalDependencies = Object.fromEntries(PLATFORMS.map((p) => [`@ruvector/kge-${p}`, version]));
writeFileSync(file, `${JSON.stringify(pkg, null, 2)}\n`);
console.log(`optionalDependencies -> ${JSON.stringify(pkg.optionalDependencies)}`);
