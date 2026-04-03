//! RVF witness chain for cryptographic provenance.
//!
//! Builds a Merkle-style chain proving that every byte of decompiler output
//! derives from the original minified bundle.

use sha3::{Digest, Sha3_256};

use crate::types::{InferredName, Module, ModuleWitnessData, WitnessChainData};

/// A witness chain linking the original bundle to extracted modules.
pub struct WitnessChain {
    /// SHA3-256 hash of the original bundle.
    pub source_hash: [u8; 32],
    /// Per-module witness entries.
    pub module_witnesses: Vec<ModuleWitness>,
    /// Merkle root of all module hashes.
    pub chain_root: [u8; 32],
}

/// Witness data for a single extracted module.
pub struct ModuleWitness {
    /// Module name.
    pub module_name: String,
    /// Byte range in the original bundle.
    pub byte_range: (usize, usize),
    /// Hash of the original bytes in the bundle for this module's range.
    pub content_hash: [u8; 32],
    /// Hash of the inferred names for this module.
    pub inferred_names_hash: [u8; 32],
}

/// Build a witness chain from the original bundle, extracted modules, and
/// inferred names.
pub fn build_witness_chain(
    source: &str,
    modules: &[Module],
    inferred_names: &[InferredName],
) -> WitnessChain {
    let source_hash = hash_bytes(source.as_bytes());

    let mut module_witnesses = Vec::new();

    for module in modules {
        let (start, end) = module.byte_range;
        let end = end.min(source.len());
        let content_bytes = if start < end {
            &source.as_bytes()[start..end]
        } else {
            &[]
        };

        let content_hash = hash_bytes(content_bytes);

        // Hash the inferred names for declarations in this module.
        let module_decl_names: Vec<&str> = module
            .declarations
            .iter()
            .map(|d| d.name.as_str())
            .collect();

        let names_str: String = inferred_names
            .iter()
            .filter(|n| module_decl_names.contains(&n.original.as_str()))
            .map(|n| format!("{}={}", n.original, n.inferred))
            .collect::<Vec<_>>()
            .join(";");

        let inferred_names_hash = hash_bytes(names_str.as_bytes());

        module_witnesses.push(ModuleWitness {
            module_name: module.name.clone(),
            byte_range: module.byte_range,
            content_hash,
            inferred_names_hash,
        });
    }

    let chain_root = compute_merkle_root(&module_witnesses);

    WitnessChain {
        source_hash,
        module_witnesses,
        chain_root,
    }
}

/// Compute a Merkle root from the module witness hashes.
fn compute_merkle_root(witnesses: &[ModuleWitness]) -> [u8; 32] {
    if witnesses.is_empty() {
        return [0u8; 32];
    }

    let mut leaves: Vec<[u8; 32]> = witnesses
        .iter()
        .map(|w| {
            let mut combined = Vec::with_capacity(64);
            combined.extend_from_slice(&w.content_hash);
            combined.extend_from_slice(&w.inferred_names_hash);
            hash_bytes(&combined)
        })
        .collect();

    // Standard binary Merkle tree.
    while leaves.len() > 1 {
        let mut next_level = Vec::new();
        let mut i = 0;
        while i < leaves.len() {
            if i + 1 < leaves.len() {
                let mut combined = Vec::with_capacity(64);
                combined.extend_from_slice(&leaves[i]);
                combined.extend_from_slice(&leaves[i + 1]);
                next_level.push(hash_bytes(&combined));
            } else {
                // Odd leaf: promote as-is.
                next_level.push(leaves[i]);
            }
            i += 2;
        }
        leaves = next_level;
    }

    leaves[0]
}

/// SHA3-256 hash of arbitrary bytes.
fn hash_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Convert a witness chain to the serializable data format.
pub fn witness_to_data(chain: &WitnessChain) -> WitnessChainData {
    WitnessChainData {
        source_hash: hex_encode(&chain.source_hash),
        module_witnesses: chain
            .module_witnesses
            .iter()
            .map(|w| ModuleWitnessData {
                module_name: w.module_name.clone(),
                byte_range: w.byte_range,
                content_hash: hex_encode(&w.content_hash),
                inferred_names_hash: hex_encode(&w.inferred_names_hash),
            })
            .collect(),
        chain_root: hex_encode(&chain.chain_root),
    }
}

/// Verify that a witness chain is internally consistent.
///
/// Checks that the Merkle root matches the recomputed root from module hashes.
pub fn verify_witness_chain(chain: &WitnessChain) -> bool {
    let recomputed = compute_merkle_root(&chain.module_witnesses);
    chain.chain_root == recomputed
}

/// Hex-encode a byte array.
fn hex_encode(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX_CHARS[(b >> 4) as usize] as char);
        s.push(HEX_CHARS[(b & 0x0f) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DeclKind, Declaration, Module};

    #[test]
    fn test_hash_deterministic() {
        let h1 = hash_bytes(b"hello");
        let h2 = hash_bytes(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different() {
        let h1 = hash_bytes(b"hello");
        let h2 = hash_bytes(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_witness_chain_verifies() {
        let module = Module {
            name: "test".to_string(),
            index: 0,
            declarations: vec![Declaration {
                name: "a".to_string(),
                kind: DeclKind::Var,
                byte_range: (0, 5),
                string_literals: vec![],
                property_accesses: vec![],
                references: vec![],
            }],
            source: String::new(),
            byte_range: (0, 5),
        };

        let chain = build_witness_chain("var a=1;", &[module], &[]);
        assert!(verify_witness_chain(&chain));
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[0, 255, 16]), "00ff10");
    }
}
