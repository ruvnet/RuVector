/**
 * The transparency log's tree is a wire format shared with
 * `crates/rvforge-registry/src/merkle.rs`, so these tests pin the RFC 6962
 * construction itself — the domain-separation prefixes, the split rule, and the
 * empty tree — rather than restating what `src/merkle.ts` happens to compute.
 *
 * `scripts/rvforge-parity-check.sh` is what proves the two implementations
 * agree on real bytes; this is what catches a regression without a Rust
 * toolchain in the loop.
 */

import { createHash } from 'node:crypto';

import { inclusionPath, leafHash, nodeHash, root, verifyInclusion } from '../src/merkle';

const sha256 = (...parts: readonly Uint8Array[]): Buffer => {
  const hash = createHash('sha256');
  for (const part of parts) hash.update(part);
  return hash.digest();
};

const leaves = (n: number): Buffer[] =>
  Array.from({ length: n }, (_, i) => leafHash(Buffer.from([i])));

describe('RFC 6962 hashing', () => {
  it('hashes the empty tree as the hash of the empty string', () => {
    expect(root([]).toString('hex')).toBe(sha256(Buffer.alloc(0)).toString('hex'));
    expect(root([]).toString('hex')).toBe(
      'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    );
  });

  it('prefixes leaves with 0x00', () => {
    expect(leafHash(Buffer.alloc(0)).toString('hex')).toBe(
      '6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d',
    );
    expect(leafHash(Buffer.from('x')).equals(sha256(Buffer.from('x')))).toBe(false);
  });

  it('prefixes interior nodes with 0x01, and is order-sensitive', () => {
    const [a, b] = [leafHash(Buffer.from('a')), leafHash(Buffer.from('b'))];
    expect(nodeHash(a, b).equals(sha256(Buffer.from([0x01]), a, b))).toBe(true);
    expect(nodeHash(a, b).equals(nodeHash(b, a))).toBe(false);
  });

  it('is its own leaf for a single-leaf tree', () => {
    const one = leaves(1);
    expect(root(one).equals(one[0])).toBe(true);
    expect(inclusionPath(0, one)).toEqual([]);
  });

  it('splits at the largest power of two below the leaf count', () => {
    const five = leaves(5);
    expect(
      root(five).equals(nodeHash(root(five.slice(0, 4)), root(five.slice(4)))),
    ).toBe(true);
  });
});

describe('inclusion proofs', () => {
  it('proves every leaf for every tree size from one to sixteen', () => {
    for (let n = 1; n <= 16; n++) {
      const l = leaves(n);
      const r = root(l);
      for (let i = 0; i < n; i++) {
        expect(verifyInclusion(l[i], i, n, inclusionPath(i, l), r)).toBe(true);
      }
    }
  });

  it('refuses a tampered leaf, a wrong index, and an out-of-range index', () => {
    const l = leaves(7);
    const r = root(l);
    const path = inclusionPath(3, l);

    expect(verifyInclusion(leafHash(Buffer.from('forged')), 3, 7, path, r)).toBe(false);
    expect(verifyInclusion(l[3], 4, 7, path, r)).toBe(false);
    expect(verifyInclusion(l[0], 7, 7, [], r)).toBe(false);
  });

  it('changes the head when a leaf is appended', () => {
    expect(root(leaves(5)).equals(root(leaves(6)))).toBe(false);
  });
});
