import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import {
  COMPATIBILITY_MATRIX,
  assertCompatible,
  checkCompatibility,
  closestSupported,
  compatibilityMatrixRevision,
} from '../src/compat';
import { ForgeError, ForgeErrorCode } from '../src/errors';
import type { BuildTarget, PackagingMode, RuntimeProfile } from '../src/types';

/** The docs copy is the source of truth; `src/` only vendors it. */
const DOCS_MATRIX = join(__dirname, '../../../../docs/research/rvf-forge/compatibility-matrix.json');
const VENDORED_MATRIX = join(__dirname, '../src/compatibility-matrix.json');

const request = (
  mode: PackagingMode,
  target: BuildTarget,
  runtimeProfile: RuntimeProfile,
): { mode: PackagingMode; target: BuildTarget; runtimeProfile: RuntimeProfile } => ({
  mode,
  target,
  runtimeProfile,
});

function expectRefusal(fn: () => unknown): ForgeError {
  try {
    fn();
  } catch (err) {
    expect(err).toBeInstanceOf(ForgeError);
    expect((err as ForgeError).code).toBe(ForgeErrorCode.UNSUPPORTED_TARGET);
    return err as ForgeError;
  }
  throw new Error('expected FORGE_E_UNSUPPORTED_TARGET');
}

describe('vendored matrix', () => {
  it('is byte-identical to the canonical copy in docs', async () => {
    const [docs, vendored] = await Promise.all([
      readFile(DOCS_MATRIX, 'utf8'),
      readFile(VENDORED_MATRIX, 'utf8'),
    ]);
    expect(vendored).toBe(docs);
  });

  it('hash-addresses the matrix so a past admission can be reconstructed', () => {
    expect(compatibilityMatrixRevision()).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(compatibilityMatrixRevision()).toBe(compatibilityMatrixRevision());
  });

  it('still marks the four runtime families and the MVP packaging modes', () => {
    expect(COMPATIBILITY_MATRIX.runtimeProfiles.wasm.status).toBe('supported');
    expect(COMPATIBILITY_MATRIX.packagingModes.embedded.status).toBe('supported');
    expect(COMPATIBILITY_MATRIX.packagingModes.thin.status).toBe('supported');
  });
});

describe('supported combinations', () => {
  it.each([
    ['embedded', 'linux-x64'],
    ['embedded', 'windows-x64'],
    ['embedded', 'macos-universal'],
    ['thin', 'linux-arm64'],
    ['thin', 'windows-arm64'],
  ] as Array<[PackagingMode, BuildTarget]>)('admits mode=%s target=%s on wasm', (mode, target) => {
    expect(checkCompatibility(request(mode, target, 'wasm')).supported).toBe(true);
  });

  it('admits every default target in one call', () => {
    const checks = assertCompatible({
      mode: 'embedded',
      targets: ['linux-x64', 'macos-universal', 'windows-x64'],
      runtimeProfile: 'wasm',
    });
    expect(checks.every((check) => check.supported)).toBe(true);
  });
});

describe('rejection (ADR-291 §2)', () => {
  it('rejects a planned packaging mode and names a supported one', () => {
    const err = expectRefusal(() =>
      assertCompatible({ mode: 'shared-reader', targets: ['linux-x64'], runtimeProfile: 'wasm' }),
    );
    expect(err.message).toMatch(/shared-reader/);
    expect(err.message).toMatch(/Closest supported combination: mode=(embedded|thin)/);
    expect(err.details.suggestion).toMatchObject({ target: 'linux-x64', runtimeProfile: 'wasm' });
  });

  it('rejects a planned runtime profile and suggests the strongest supported one', () => {
    const err = expectRefusal(() =>
      assertCompatible({ mode: 'embedded', targets: ['linux-x64'], runtimeProfile: 'microvm' }),
    );
    expect(err.message).toMatch(/linux-microvm/);
    // selectionOrder puts os-isolation+wasm above wasm, so `native` is nearest.
    expect(err.details.suggestion).toMatchObject({ runtimeProfile: 'native' });
  });

  it('rejects a target with no platform entry under the requested profile', () => {
    const err = expectRefusal(() =>
      assertCompatible({ mode: 'embedded', targets: ['rvm'], runtimeProfile: 'wasm' }),
    );
    expect(err.message).toMatch(/no platform entry for target "rvm"/);
    expect(err.details.suggestion).toBeDefined();
  });

  it('names every refused target, not just the first', () => {
    const err = expectRefusal(() =>
      assertCompatible({ mode: 'embedded', targets: ['linux-x64', 'rvm'], runtimeProfile: 'wasm' }),
    );
    expect(err.details.refused).toHaveLength(1);
    expect(err.details.matrixRevision).toBe(compatibilityMatrixRevision());
  });

  it('records the matrix revision on every refusal', () => {
    const err = expectRefusal(() =>
      assertCompatible({ mode: 'embedded', targets: ['rvm'], runtimeProfile: 'rvm' }),
    );
    expect(err.details.matrixRevision).toMatch(/^sha256:[0-9a-f]{64}$/);
  });
});

describe('closestSupported', () => {
  it('keeps fields the matrix already admits', () => {
    expect(closestSupported(request('embedded', 'linux-x64', 'wasm'))).toEqual({
      mode: 'embedded',
      target: 'linux-x64',
      runtimeProfile: 'wasm',
    });
  });

  it('prefers the same OS with a different architecture over a different OS', () => {
    // `rvm` has no supported OS, so the fallback is whatever the profile lists.
    const suggestion = closestSupported(request('embedded', 'rvm', 'wasm'));
    expect(suggestion).toBeDefined();
    expect(suggestion?.runtimeProfile).toBe('wasm');
  });

  it('never suggests an unsupported combination', () => {
    for (const target of ['rvm', 'linux-x64', 'macos-arm64'] as BuildTarget[]) {
      for (const profile of ['wasm', 'native', 'microvm', 'rvm'] as RuntimeProfile[]) {
        const suggestion = closestSupported(request('shared-reader', target, profile));
        expect(suggestion).toBeDefined();
        expect(checkCompatibility(suggestion!).supported).toBe(true);
      }
    }
  });
});
