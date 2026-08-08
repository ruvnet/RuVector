//! RFC 6962 Merkle tree hashing, inclusion paths, and proof verification.
//!
//! The transparency log needs a tree whose shape is fully determined by the
//! number of leaves, so that two parties who agree on the entry list agree on
//! the root without exchanging anything else. RFC 6962's unbalanced tree has
//! that property and is the format Certificate Transparency verifiers already
//! implement, so a reader can reuse an existing proof checker.
//!
//! Leaves are domain-separated with a `0x00` prefix and interior nodes with
//! `0x01`, which is what prevents a leaf from being reinterpreted as an
//! interior node (the second-preimage attack that an undecorated tree admits).

use sha2::{Digest, Sha256};

/// A 32-byte SHA-256 digest.
pub type Hash = [u8; 32];

/// SHA-256 of a byte slice.
pub fn sha256(bytes: &[u8]) -> Hash {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

/// `MTH({d}) = SHA-256(0x00 || d)`.
pub fn leaf_hash(payload: &[u8]) -> Hash {
    let mut h = Sha256::new();
    h.update([0u8]);
    h.update(payload);
    h.finalize().into()
}

/// `SHA-256(0x01 || left || right)`.
pub fn node_hash(left: &Hash, right: &Hash) -> Hash {
    let mut h = Sha256::new();
    h.update([1u8]);
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// The Merkle tree head over `leaves`. The empty tree hashes the empty string,
/// per RFC 6962 §2.1.
pub fn root(leaves: &[Hash]) -> Hash {
    match leaves.len() {
        0 => sha256(&[]),
        1 => leaves[0],
        n => {
            let k = split_point(n);
            node_hash(&root(&leaves[..k]), &root(&leaves[k..]))
        }
    }
}

/// The audit path for leaf `index` within `leaves`.
///
/// Returns an empty path for a single-leaf tree, and an empty path for an
/// out-of-range index (which [`verify_inclusion`] then rejects).
pub fn inclusion_path(index: usize, leaves: &[Hash]) -> Vec<Hash> {
    let n = leaves.len();
    if index >= n || n <= 1 {
        return Vec::new();
    }
    let k = split_point(n);
    if index < k {
        let mut path = inclusion_path(index, &leaves[..k]);
        path.push(root(&leaves[k..]));
        path
    } else {
        let mut path = inclusion_path(index - k, &leaves[k..]);
        path.push(root(&leaves[..k]));
        path
    }
}

/// Verify that `leaf` sits at `index` of a tree of `tree_size` leaves whose
/// head is `expected_root`, using `path` (RFC 6962 §2.1.1).
pub fn verify_inclusion(
    leaf: Hash,
    index: usize,
    tree_size: usize,
    path: &[Hash],
    expected_root: Hash,
) -> bool {
    if index >= tree_size {
        return false;
    }
    let mut node_index = index;
    let mut last_index = tree_size - 1;
    let mut acc = leaf;
    for sibling in path {
        if last_index == 0 {
            return false;
        }
        if node_index % 2 == 1 || node_index == last_index {
            acc = node_hash(sibling, &acc);
            while node_index % 2 == 0 && node_index != 0 {
                node_index /= 2;
                last_index /= 2;
            }
        } else {
            acc = node_hash(&acc, sibling);
        }
        node_index /= 2;
        last_index /= 2;
    }
    last_index == 0 && acc == expected_root
}

/// The largest power of two strictly less than `n`. Requires `n > 1`.
fn split_point(n: usize) -> usize {
    debug_assert!(n > 1);
    let mut k = 1;
    while k * 2 < n {
        k *= 2;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves(n: usize) -> Vec<Hash> {
        (0..n).map(|i| leaf_hash(&[i as u8])).collect()
    }

    #[test]
    fn split_points_are_powers_of_two_below_n() {
        assert_eq!(split_point(2), 1);
        assert_eq!(split_point(3), 2);
        assert_eq!(split_point(4), 2);
        assert_eq!(split_point(5), 4);
        assert_eq!(split_point(9), 8);
    }

    #[test]
    fn empty_tree_hashes_the_empty_string() {
        assert_eq!(root(&[]), sha256(&[]));
    }

    #[test]
    fn single_leaf_tree_is_its_leaf() {
        let l = leaves(1);
        assert_eq!(root(&l), l[0]);
        assert!(inclusion_path(0, &l).is_empty());
        assert!(verify_inclusion(l[0], 0, 1, &[], root(&l)));
    }

    #[test]
    fn leaves_and_nodes_are_domain_separated() {
        // Without the prefixes these would collide.
        assert_ne!(leaf_hash(b"x"), sha256(b"x"));
        let a = leaf_hash(b"a");
        let b = leaf_hash(b"b");
        assert_ne!(node_hash(&a, &b), node_hash(&b, &a));
    }

    #[test]
    fn every_leaf_proves_inclusion_for_sizes_one_through_sixteen() {
        for n in 1..=16 {
            let l = leaves(n);
            let r = root(&l);
            for i in 0..n {
                let path = inclusion_path(i, &l);
                assert!(
                    verify_inclusion(l[i], i, n, &path, r),
                    "size {n} index {i} failed"
                );
            }
        }
    }

    #[test]
    fn a_tampered_leaf_fails_verification() {
        let l = leaves(7);
        let r = root(&l);
        let path = inclusion_path(3, &l);
        let forged = leaf_hash(b"not the third leaf");
        assert!(!verify_inclusion(forged, 3, 7, &path, r));
    }

    #[test]
    fn a_proof_for_the_wrong_index_fails() {
        let l = leaves(7);
        let r = root(&l);
        let path = inclusion_path(3, &l);
        assert!(!verify_inclusion(l[3], 4, 7, &path, r));
    }

    #[test]
    fn out_of_range_index_fails() {
        let l = leaves(4);
        assert!(!verify_inclusion(l[0], 4, 4, &[], root(&l)));
    }

    #[test]
    fn appending_a_leaf_changes_the_root() {
        let a = root(&leaves(5));
        let b = root(&leaves(6));
        assert_ne!(a, b);
    }
}
