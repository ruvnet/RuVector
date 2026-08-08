/**
 * A valid `rvforge.json` for the publisher-verb tests, plus the mutators the
 * individual cases need.
 *
 * The base project passes every `pack` check and every inspection-only `test`
 * category, so a failing assertion in a test is about the mutation that test
 * applied rather than about the fixture drifting.
 */

import { defaultDenials, defaultProject, type RvforgeProject } from '../../src/project';

const REQUESTS = [
  { class: 'filesystem' as const, scope: 'user-selected', rationale: 'reads only documents you choose' },
  { class: 'memory' as const, scope: '512MiB', rationale: 'model working set' },
  { class: 'persistent-state' as const, scope: 'encrypted-local', rationale: 'agent memory' },
];

/** A project that packs and tests clean. */
export function fixtureProject(overrides: Partial<RvforgeProject> = {}): RvforgeProject {
  return defaultProject({
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
      displayName: 'RuVector Tests',
      identityEvidence: { method: 'manual-review', reference: 'tests' },
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
    capabilities: { defaultPolicy: 'deny', requests: REQUESTS, denials: defaultDenials(REQUESTS) },
    modelManifest: { location: 'embedded', digests: [`sha256:${'a'.repeat(64)}`] },
    externalServices: [],
    state: { checkpointing: true, recovery: true, rollbackSafeUntilStateSchema: 2 },
    ...overrides,
  });
}

/** Replace the capability requests, recomputing the denial list to match. */
export function withRequests(
  requests: RvforgeProject['capabilities']['requests'],
  overrides: Partial<RvforgeProject> = {},
): RvforgeProject {
  return fixtureProject({
    ...overrides,
    capabilities: { defaultPolicy: 'deny', requests, denials: defaultDenials(requests) },
  });
}
