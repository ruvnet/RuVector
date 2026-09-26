// SPDX-License-Identifier: MIT
// The harness's `earth-pulse-harness init` entry (TypeScript mirror of bin/cli.js init).

import { loadKernel } from '@metaharness/kernel';
import adapter from '@metaharness/host-claude-code';

const HARNESS_NAME = 'earth-pulse-harness';

async function main(): Promise<number> {
  const kernel = await loadKernel();
  const info = kernel.kernelInfo();
  console.log(`${HARNESS_NAME} — kernel ${info.version} (${kernel.backend})`);
  console.log(`Host adapter: ${adapter.name}`);
  console.log('Earth Pulse Observatory: freeze the physics, evolve the harness.');
  console.log(`Run \`${HARNESS_NAME} doctor\` to verify the install.`);
  return 0;
}

main().then(c => process.exit(c)).catch(err => {
  console.error(err);
  process.exit(1);
});
