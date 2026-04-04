/**
 * witness.js - SHA-256 witness chain generation and verification.
 *
 * A witness chain is a Merkle-like structure that cryptographically proves
 * the decompiled output derives from a specific input bundle.
 *
 * Chain structure:
 *   root = H(source_hash || module_hashes[0] || ... || module_hashes[n])
 *
 * Each entry records:
 *   { hash, label, parent }
 * so the chain can be verified without re-running the decompiler.
 */

'use strict';

const crypto = require('crypto');

/**
 * Compute SHA-256 hash of a string or buffer.
 * @param {string|Buffer} data
 * @returns {string} hex-encoded hash
 */
function sha256(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

/**
 * Build a witness chain from source and decompiled modules.
 *
 * @param {string} source - original bundle source code
 * @param {Array<{name: string, content: string}>} modules - decompiled modules
 * @returns {{
 *   source_hash: string,
 *   module_hashes: Array<{name: string, hash: string}>,
 *   root: string,
 *   chain: Array<{hash: string, label: string, parent: string|null}>,
 *   created: string,
 *   algorithm: string
 * }}
 */
function buildWitnessChain(source, modules) {
  const sourceHash = sha256(source);
  const chain = [];
  const moduleHashes = [];

  // Root node: the source hash
  chain.push({
    hash: sourceHash,
    label: 'source',
    parent: null,
  });

  // One node per decompiled module
  for (const mod of modules) {
    const modHash = sha256(mod.content);
    moduleHashes.push({ name: mod.name, hash: modHash });

    chain.push({
      hash: modHash,
      label: `module:${mod.name}`,
      parent: sourceHash,
    });
  }

  // Compute Merkle root: H(source_hash || mod_hash_0 || ... || mod_hash_n)
  const allHashes = sourceHash + moduleHashes.map((m) => m.hash).join('');
  const root = sha256(allHashes);

  chain.push({
    hash: root,
    label: 'root',
    parent: sourceHash,
  });

  return {
    source_hash: sourceHash,
    module_hashes: moduleHashes,
    root,
    chain,
    created: new Date().toISOString(),
    algorithm: 'sha256',
  };
}

/**
 * Verify a witness chain against a source file.
 *
 * @param {object} witness - the witness object (from buildWitnessChain)
 * @param {string} [sourceContent] - original source to verify against (optional)
 * @returns {{valid: boolean, chain_length: number, root: string, errors: string[]}}
 */
function verifyWitnessChain(witness, sourceContent) {
  const errors = [];

  if (!witness || !witness.chain || !witness.root) {
    return { valid: false, chain_length: 0, root: '', errors: ['Missing witness data'] };
  }

  // Verify source hash if content provided
  if (sourceContent) {
    const actualSourceHash = sha256(sourceContent);
    if (actualSourceHash !== witness.source_hash) {
      errors.push(
        `Source hash mismatch: expected ${witness.source_hash}, got ${actualSourceHash}`,
      );
    }
  }

  // Verify chain integrity: each node's parent must exist in the chain
  const hashSet = new Set(witness.chain.map((n) => n.hash));
  for (const node of witness.chain) {
    if (node.parent && !hashSet.has(node.parent)) {
      errors.push(`Broken chain: node ${node.label} references missing parent ${node.parent}`);
    }
  }

  // Recompute root from module hashes
  if (witness.module_hashes && witness.source_hash) {
    const allHashes =
      witness.source_hash + witness.module_hashes.map((m) => m.hash).join('');
    const expectedRoot = sha256(allHashes);
    if (expectedRoot !== witness.root) {
      errors.push(`Root mismatch: expected ${expectedRoot}, got ${witness.root}`);
    }
  }

  return {
    valid: errors.length === 0,
    chain_length: witness.chain.length,
    root: witness.root,
    errors,
  };
}

module.exports = {
  sha256,
  buildWitnessChain,
  verifyWitnessChain,
};
