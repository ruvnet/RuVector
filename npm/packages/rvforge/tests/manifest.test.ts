import { ForgeError, ForgeErrorCode } from '../src/errors';
import { canonicalJson } from '../src/hash';
import {
  buildManifest,
  capabilityPolicyHash,
  defaultCapabilityPolicy,
  manifestHash,
  normalizeCapabilityPolicy,
  serializeManifest,
  type BuildManifestInput,
} from '../src/manifest';
import { CAPABILITY_CLASSES } from '../src/types';

const identity = {
  sha256: 'b'.repeat(64),
  size: 4608,
  fileId: '0123456789abcdef0123456789abcdef',
  formatVersion: 1,
  signatureAlgo: 1,
  signaturePresent: true,
};

const input = (overrides: Partial<BuildManifestInput> = {}): BuildManifestInput => ({
  app: {
    name: 'My Agent',
    version: '1.2.3',
    publisher: 'Example Publisher',
    identifier: 'com.example.myagent',
  },
  identity,
  targets: ['linux-x64', 'windows-x64'],
  packaging: { mode: 'embedded' },
  runtime: { profile: 'wasm', rvmVersion: '0.1.0', rvmCommit: 'abc1234' },
  generatorVersion: '0.1.0',
  ...overrides,
});

describe('canonical JSON', () => {
  it('sorts keys regardless of insertion order', () => {
    expect(canonicalJson({ b: 1, a: 2 })).toBe(canonicalJson({ a: 2, b: 1 }));
    expect(canonicalJson({ b: 1, a: 2 })).toBe('{"a":2,"b":1}');
  });

  it('drops undefined members but keeps null', () => {
    expect(canonicalJson({ a: undefined, b: null })).toBe('{"b":null}');
  });

  it('preserves array order', () => {
    expect(canonicalJson([3, 1, 2])).toBe('[3,1,2]');
  });
});

describe('capability policy', () => {
  it('defaults to deny with every class present and empty', () => {
    const policy = defaultCapabilityPolicy();
    expect(policy.defaultDeny).toBe(true);
    for (const cls of CAPABILITY_CLASSES) expect(policy.allow[cls]).toEqual([]);
  });

  it('forces defaultDeny even when the input tries to opt out', () => {
    const policy = normalizeCapabilityPolicy({
      defaultDeny: false as unknown as true,
      allow: { network: ['https://api.example'] },
    });
    expect(policy.defaultDeny).toBe(true);
  });

  it('de-duplicates and sorts grants', () => {
    const policy = normalizeCapabilityPolicy({ allow: { filesystem: ['/b', '/a', '/b'] } });
    expect(policy.allow.filesystem).toEqual(['/a', '/b']);
  });

  it('rejects an unknown capability class', () => {
    expect(() => normalizeCapabilityPolicy({ allow: { telepathy: [] } as never })).toThrow(ForgeError);
  });

  it('hashes equal policies identically regardless of grant order', () => {
    const a = normalizeCapabilityPolicy({ allow: { network: ['b.example', 'a.example'] } });
    const b = normalizeCapabilityPolicy({ allow: { network: ['a.example', 'b.example'] } });
    expect(capabilityPolicyHash(a)).toBe(capabilityPolicyHash(b));
  });
});

describe('buildManifest determinism', () => {
  it('produces byte-identical JSON for the same logical input', () => {
    const first = serializeManifest(buildManifest(input()));
    const second = serializeManifest(buildManifest(input()));
    expect(first).toBe(second);
    expect(manifestHash(buildManifest(input()))).toBe(manifestHash(buildManifest(input())));
  });

  it('is insensitive to target order and duplicates', () => {
    const a = buildManifest(input({ targets: ['windows-x64', 'linux-x64'] }));
    const b = buildManifest(input({ targets: ['linux-x64', 'windows-x64', 'linux'] }));
    expect(serializeManifest(a)).toBe(serializeManifest(b));
    expect(a.targets).toEqual(['linux-x64', 'windows-x64']);
  });

  it('is insensitive to capability grant order', () => {
    const a = buildManifest(input({ capabilityPolicy: { allow: { tools: ['fetch', 'shell'] } } }));
    const b = buildManifest(input({ capabilityPolicy: { allow: { tools: ['shell', 'fetch'] } } }));
    expect(manifestHash(a)).toBe(manifestHash(b));
  });

  it('contains no timestamp, path, or hostname field', () => {
    const json = serializeManifest(buildManifest(input()));
    for (const banned of ['createdAt', 'timestamp', 'builtAt', 'hostname', 'cwd']) {
      expect(json).not.toContain(banned);
    }
  });

  it('changes the hash when the capability policy changes', () => {
    const open = buildManifest(input({ capabilityPolicy: { allow: { network: ['*'] } } }));
    expect(manifestHash(open)).not.toBe(manifestHash(buildManifest(input())));
  });

  it('embeds the RVM integration contract', () => {
    const manifest = buildManifest(input());
    expect(manifest.runtime).toEqual({
      rvmVersion: '0.1.0',
      rvmCommit: 'abc1234',
      runtimeProfile: 'wasm',
      stateSchemaVersion: 1,
      witnessSchemaVersion: 1,
    });
    expect(manifest.capabilityPolicyHash).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe('buildManifest validation', () => {
  const expectCode = (fn: () => unknown, code: ForgeErrorCode) => {
    try {
      fn();
      throw new Error('expected a ForgeError');
    } catch (err) {
      expect(err).toBeInstanceOf(ForgeError);
      expect((err as ForgeError).code).toBe(code);
    }
  };

  it('rejects an unsupported target', () => {
    expectCode(() => buildManifest(input({ targets: ['solaris'] })), ForgeErrorCode.UNSUPPORTED_TARGET);
  });

  it('rejects an empty target list', () => {
    expectCode(() => buildManifest(input({ targets: [] })), ForgeErrorCode.MANIFEST);
  });

  it('rejects a non-semver app version', () => {
    expectCode(
      () => buildManifest(input({ app: { ...input().app, version: 'v1' } })),
      ForgeErrorCode.MANIFEST,
    );
  });

  it('rejects a non reverse-DNS identifier', () => {
    expectCode(
      () => buildManifest(input({ app: { ...input().app, identifier: 'myagent' } })),
      ForgeErrorCode.MANIFEST,
    );
  });

  it('rejects a runtime profile outside the RVM compatibility matrix', () => {
    expectCode(
      () => buildManifest(input({ runtime: { profile: 'native', rvmVersion: '0.1.0', rvmCommit: 'abc' } })),
      ForgeErrorCode.MANIFEST,
    );
  });

  it('rejects thin packaging without a distribution URL', () => {
    expectCode(() => buildManifest(input({ packaging: { mode: 'thin' } })), ForgeErrorCode.MANIFEST);
  });

  it('rejects an identity whose sha256 was never computed', () => {
    expectCode(
      () => buildManifest(input({ identity: { ...identity, sha256: '' } })),
      ForgeErrorCode.MANIFEST,
    );
  });

  it('keeps signing references and never accepts key material', () => {
    const manifest = buildManifest(
      input({ signing: { windows: { provider: 'azure-kv', keyRef: 'forge-signing', identity: 'Example Ltd' } } }),
    );
    expect(manifest.signing.windows).toEqual({
      provider: 'azure-kv',
      keyRef: 'forge-signing',
      identity: 'Example Ltd',
    });
    expect(serializeManifest(manifest)).not.toMatch(/BEGIN [A-Z ]*PRIVATE KEY/);
  });
});
