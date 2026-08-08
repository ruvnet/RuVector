/**
 * RFC 6962 Merkle tree hashing, for the transparency log.
 *
 * The log needs a tree whose shape is fixed by the number of leaves alone, so
 * that two parties who agree on the entry list agree on the root without
 * exchanging anything else — and so that a Reader can be handed an *inclusion
 * proof* for one release rather than the whole log. RFC 6962's unbalanced tree
 * has both properties and is the format Certificate Transparency verifiers
 * already implement.
 *
 * Leaves carry a `0x00` prefix and interior nodes a `0x01`. Without that domain
 * separation a leaf could be replayed as an interior node, which is the
 * second-preimage attack an undecorated tree admits.
 *
 * This is the same construction as `crates/rvforge-registry/src/merkle.rs`;
 * `scripts/rvforge-parity-check.sh` is what keeps the two byte-identical.
 */

import { createHash } from 'node:crypto';

/** A 32-byte SHA-256 digest. */
export type Hash = Buffer;

const sha256 = (...parts: readonly Uint8Array[]): Hash => {
  const hash = createHash('sha256');
  for (const part of parts) hash.update(part);
  return hash.digest();
};

const LEAF_PREFIX = Buffer.from([0x00]);
const NODE_PREFIX = Buffer.from([0x01]);

/** `MTH({d}) = SHA-256(0x00 || d)`. */
export const leafHash = (payload: Uint8Array): Hash => sha256(LEAF_PREFIX, payload);

/** `SHA-256(0x01 || left || right)`. */
export const nodeHash = (left: Hash, right: Hash): Hash => sha256(NODE_PREFIX, left, right);

/** The largest power of two strictly less than `n`. Requires `n > 1`. */
function splitPoint(n: number): number {
  let k = 1;
  while (k * 2 < n) k *= 2;
  return k;
}

/** The tree head over `leaves`. The empty tree hashes the empty string. */
export function root(leaves: readonly Hash[]): Hash {
  if (leaves.length === 0) return sha256(Buffer.alloc(0));
  if (leaves.length === 1) return leaves[0];
  const k = splitPoint(leaves.length);
  return nodeHash(root(leaves.slice(0, k)), root(leaves.slice(k)));
}

/**
 * The audit path for leaf `index`.
 *
 * Empty for a single-leaf tree, and empty for an out-of-range index — which
 * {@link verifyInclusion} then rejects, rather than this returning a proof that
 * looks plausible.
 */
export function inclusionPath(index: number, leaves: readonly Hash[]): Hash[] {
  const n = leaves.length;
  if (index < 0 || index >= n || n <= 1) return [];
  const k = splitPoint(n);
  return index < k
    ? [...inclusionPath(index, leaves.slice(0, k)), root(leaves.slice(k))]
    : [...inclusionPath(index - k, leaves.slice(k)), root(leaves.slice(0, k))];
}

/**
 * Verify that `leaf` sits at `index` of a `treeSize`-leaf tree whose head is
 * `expectedRoot` (RFC 6962 §2.1.1).
 */
export function verifyInclusion(
  leaf: Hash,
  index: number,
  treeSize: number,
  path: readonly Hash[],
  expectedRoot: Hash,
): boolean {
  if (index < 0 || index >= treeSize) return false;
  let nodeIndex = index;
  let lastIndex = treeSize - 1;
  let acc = leaf;

  for (const sibling of path) {
    if (lastIndex === 0) return false;
    if (nodeIndex % 2 === 1 || nodeIndex === lastIndex) {
      acc = nodeHash(sibling, acc);
      while (nodeIndex % 2 === 0 && nodeIndex !== 0) {
        nodeIndex = Math.floor(nodeIndex / 2);
        lastIndex = Math.floor(lastIndex / 2);
      }
    } else {
      acc = nodeHash(acc, sibling);
    }
    nodeIndex = Math.floor(nodeIndex / 2);
    lastIndex = Math.floor(lastIndex / 2);
  }

  return lastIndex === 0 && acc.equals(expectedRoot);
}
