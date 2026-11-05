//! COW-aware compaction engine.
//!
//! Two compaction modes:
//! - **Read optimize**: rewrite hot clusters contiguously for sequential I/O.
//! - **Space reclaim**: if a local cluster is byte-identical to the parent's,
//!   replace LocalOffset with ParentRef to reclaim local storage.
//!
//! Segment preservation: unknown segments are copied forward unless
//! `strip_unknown` is set.

use std::collections::HashMap;

use rvf_types::cow_map::CowMapEntry;
use rvf_types::RvfError;

use crate::cow_map::CowMap;

/// Result of a COW compaction operation.
pub struct CompactionResult {
    /// Number of clusters rewritten or reclaimed.
    pub clusters_affected: u32,
    /// Bytes reclaimed (for space_reclaim mode).
    pub bytes_reclaimed: u64,
    /// Number of clusters that matched parent and were converted to ParentRef.
    pub clusters_deduplicated: u32,
}

/// Refcount data for shared clusters.
pub struct RefcountData {
    /// Map from cluster_id to reference count.
    pub refcounts: HashMap<u32, u32>,
}

/// COW-aware compaction engine.
pub struct CowCompactor {
    /// Whether to strip unknown segment types during compaction.
    pub strip_unknown: bool,
}

impl Default for CowCompactor {
    fn default() -> Self {
        Self::new()
    }
}

impl CowCompactor {
    /// Create a new compactor with default settings.
    pub fn new() -> Self {
        Self {
            strip_unknown: false,
        }
    }

    /// Read-optimize compaction: reorder local clusters for sequential read.
    ///
    /// Scans the COW map, reads all LocalOffset clusters, and rewrites them
    /// contiguously in cluster_id order. Updates the map entries to point to
    /// the new contiguous offsets.
    pub fn compact_read_optimize(
        cow_map: &mut CowMap,
        local_data: &HashMap<u32, Vec<u8>>,
        cluster_size: u32,
    ) -> Result<CompactionResult, RvfError> {
        let mut clusters_affected = 0u32;
        let mut new_data: Vec<(u32, Vec<u8>)> = Vec::new();

        // Collect all local clusters in order
        for cluster_id in 0..cow_map.cluster_count() {
            if let CowMapEntry::LocalOffset(_) = cow_map.lookup(cluster_id) {
                if let Some(data) = local_data.get(&cluster_id) {
                    new_data.push((cluster_id, data.clone()));
                    clusters_affected += 1;
                }
            }
        }

        // Assign new sequential offsets (these would be written to file)
        let mut offset = 0u64;
        for (cluster_id, _data) in &new_data {
            cow_map.update(*cluster_id, CowMapEntry::LocalOffset(offset));
            offset += cluster_size as u64;
        }

        Ok(CompactionResult {
            clusters_affected,
            bytes_reclaimed: 0,
            clusters_deduplicated: 0,
        })
    }

    /// Space-reclaim compaction: if local cluster data matches parent data,
    /// replace LocalOffset with ParentRef to reclaim space.
    pub fn compact_space_reclaim(
        cow_map: &mut CowMap,
        local_data: &HashMap<u32, Vec<u8>>,
        parent_data: &HashMap<u32, Vec<u8>>,
        cluster_size: u32,
    ) -> Result<CompactionResult, RvfError> {
        let mut clusters_deduplicated = 0u32;
        let mut bytes_reclaimed = 0u64;

        for cluster_id in 0..cow_map.cluster_count() {
            if let CowMapEntry::LocalOffset(_) = cow_map.lookup(cluster_id) {
                let local = match local_data.get(&cluster_id) {
                    Some(d) => d,
                    None => continue,
                };
                let parent = match parent_data.get(&cluster_id) {
                    Some(d) => d,
                    None => continue,
                };

                // Compare the bytes, not a digest of them. Dropping the local
                // cluster is only safe when it is identical to the parent's,
                // and a memcmp is both cheaper than hashing both sides and
                // immune to any collision.
                if local == parent {
                    cow_map.update(cluster_id, CowMapEntry::ParentRef);
                    clusters_deduplicated += 1;
                    bytes_reclaimed += cluster_size as u64;
                }
            }
        }

        Ok(CompactionResult {
            clusters_affected: clusters_deduplicated,
            bytes_reclaimed,
            clusters_deduplicated,
        })
    }

    /// Rebuild reference counts from the COW map.
    ///
    /// Each LocalOffset cluster has refcount 1.
    /// ParentRef clusters increment the parent's refcount.
    pub fn rebuild_refcounts(cow_map: &CowMap) -> RefcountData {
        let mut refcounts = HashMap::new();

        for cluster_id in 0..cow_map.cluster_count() {
            match cow_map.lookup(cluster_id) {
                CowMapEntry::LocalOffset(_) => {
                    *refcounts.entry(cluster_id).or_insert(0) += 1;
                }
                CowMapEntry::ParentRef => {
                    // Parent cluster is referenced
                    *refcounts.entry(cluster_id).or_insert(0) += 1;
                }
                CowMapEntry::Unallocated => {}
            }
        }

        RefcountData { refcounts }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_optimize_reorders_clusters() {
        let mut map = CowMap::new_flat(4);
        map.update(0, CowMapEntry::LocalOffset(0x1000));
        map.update(2, CowMapEntry::LocalOffset(0x3000));
        // 1 and 3 are unallocated

        let mut local_data = HashMap::new();
        local_data.insert(0, vec![0xAA; 256]);
        local_data.insert(2, vec![0xBB; 256]);

        let result = CowCompactor::compact_read_optimize(&mut map, &local_data, 256).unwrap();

        assert_eq!(result.clusters_affected, 2);

        // Clusters should now have sequential offsets
        assert_eq!(map.lookup(0), CowMapEntry::LocalOffset(0));
        assert_eq!(map.lookup(2), CowMapEntry::LocalOffset(256));
    }

    #[test]
    fn space_reclaim_deduplicates() {
        let mut map = CowMap::new_flat(3);
        let shared_data = vec![0xAA; 128];
        let different_data = vec![0xBB; 128];

        map.update(0, CowMapEntry::LocalOffset(0x100));
        map.update(1, CowMapEntry::LocalOffset(0x200));
        map.update(2, CowMapEntry::ParentRef);

        let mut local_data = HashMap::new();
        local_data.insert(0, shared_data.clone()); // same as parent
        local_data.insert(1, different_data); // different from parent

        let mut parent_data = HashMap::new();
        parent_data.insert(0, shared_data); // matches local
        parent_data.insert(1, vec![0xCC; 128]); // does not match local

        let result =
            CowCompactor::compact_space_reclaim(&mut map, &local_data, &parent_data, 128).unwrap();

        assert_eq!(result.clusters_deduplicated, 1);
        assert_eq!(result.bytes_reclaimed, 128);

        // Cluster 0 should be ParentRef now (deduplicated)
        assert_eq!(map.lookup(0), CowMapEntry::ParentRef);
        // Cluster 1 should remain local (different data)
        assert_eq!(map.lookup(1), CowMapEntry::LocalOffset(0x200));
    }

    /// Two different 64-byte clusters with the same digest under the
    /// pre-SHAKE-256 add-and-rotate fold (found with an SMT solver).
    const FOLD_COLLISION_A: &str = "1107f7bea48fa08fa9b582d41863bd450b2c8c1e8ad51d394f518c693c12eac0ea2805667004221e08c0060c1c212183fc26d6e4d5226341a8a579e881ba0680";
    const FOLD_COLLISION_B: &str = "8fbfbb38056d8f098071fd642c101f91332faf2fc408b53fc3151b5d7ac4e0322840f240089e08080877c04bf880fa190820c45ccf38ed60b740e5c3f061e07e";

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn space_reclaim_keeps_a_local_cluster_that_collides_with_the_parent() {
        let local = unhex(FOLD_COLLISION_A);
        let parent = unhex(FOLD_COLLISION_B);
        assert_ne!(local, parent);
        // The premise: the old digest could not tell these apart.
        assert_eq!(
            crate::hashing::legacy_fold_256(&local),
            crate::hashing::legacy_fold_256(&parent)
        );

        let mut map = CowMap::new_flat(1);
        map.update(0, CowMapEntry::LocalOffset(0x100));
        let local_data = HashMap::from([(0, local)]);
        let parent_data = HashMap::from([(0, parent)]);

        let result =
            CowCompactor::compact_space_reclaim(&mut map, &local_data, &parent_data, 64).unwrap();

        assert_eq!(result.clusters_deduplicated, 0);
        assert_eq!(result.bytes_reclaimed, 0);
        assert_eq!(map.lookup(0), CowMapEntry::LocalOffset(0x100));
    }

    #[test]
    fn rebuild_refcounts() {
        let mut map = CowMap::new_flat(4);
        map.update(0, CowMapEntry::LocalOffset(0x100));
        map.update(1, CowMapEntry::ParentRef);
        map.update(2, CowMapEntry::LocalOffset(0x200));
        // 3 is unallocated

        let refcounts = CowCompactor::rebuild_refcounts(&map);

        assert_eq!(refcounts.refcounts.get(&0), Some(&1));
        assert_eq!(refcounts.refcounts.get(&1), Some(&1));
        assert_eq!(refcounts.refcounts.get(&2), Some(&1));
        assert_eq!(refcounts.refcounts.get(&3), None);
    }
}
