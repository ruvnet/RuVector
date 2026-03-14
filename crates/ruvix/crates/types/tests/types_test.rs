//! Comprehensive integration tests for ruvix-types crate.
//!
//! These tests verify type construction, serialization, and property-based
//! invariants for all kernel interface types defined in ADR-087.

use ruvix_types::*;

// =============================================================================
// Handle Tests
// =============================================================================

mod handle_tests {
    use super::*;

    #[test]
    fn handle_construction() {
        let h = Handle::new(42, 7);
        assert_eq!(h.id, 42);
        assert_eq!(h.generation, 7);
        assert!(!h.is_null());
    }

    #[test]
    fn handle_null() {
        let h = Handle::null();
        assert!(h.is_null());
        assert_eq!(h.id, u32::MAX);
    }

    #[test]
    fn handle_raw_roundtrip() {
        for id in [0, 1, 100, u32::MAX - 1, u32::MAX] {
            for gen in [0, 1, 100, u32::MAX - 1, u32::MAX] {
                let h = Handle::new(id, gen);
                let raw = h.to_raw();
                let h2 = Handle::from_raw(raw);
                assert_eq!(h, h2, "Roundtrip failed for id={}, gen={}", id, gen);
            }
        }
    }

    #[test]
    fn handle_default_is_null() {
        let h: Handle = Default::default();
        assert!(h.is_null());
    }

    #[test]
    fn handle_eq_and_hash() {
        use std::collections::HashSet;

        let h1 = Handle::new(1, 2);
        let h2 = Handle::new(1, 2);
        let h3 = Handle::new(1, 3);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);

        let mut set = HashSet::new();
        set.insert(h1);
        assert!(set.contains(&h2));
        assert!(!set.contains(&h3));
    }
}

// =============================================================================
// ObjectType Tests
// =============================================================================

mod object_type_tests {
    use super::*;

    #[test]
    fn object_type_all_variants() {
        let types = [
            ObjectType::Task,
            ObjectType::Region,
            ObjectType::Queue,
            ObjectType::Timer,
            ObjectType::VectorStore,
            ObjectType::GraphStore,
            ObjectType::RvfMount,
            ObjectType::Capability,
            ObjectType::WitnessLog,
            ObjectType::Subscription,
        ];

        for (i, ot) in types.iter().enumerate() {
            assert_eq!(*ot as u8, i as u8);
            assert_eq!(ObjectType::from_u8(i as u8), Some(*ot));
            assert!(!ot.as_str().is_empty());
        }
    }

    #[test]
    fn object_type_invalid_values() {
        for i in 10..=255 {
            assert!(ObjectType::from_u8(i).is_none());
        }
    }

    #[test]
    fn object_type_str_representations() {
        assert_eq!(ObjectType::Task.as_str(), "Task");
        assert_eq!(ObjectType::Region.as_str(), "Region");
        assert_eq!(ObjectType::Queue.as_str(), "Queue");
        assert_eq!(ObjectType::VectorStore.as_str(), "VectorStore");
    }
}

// =============================================================================
// Capability Tests
// =============================================================================

mod capability_tests {
    use super::*;

    #[test]
    fn cap_rights_basic_operations() {
        let empty = CapRights::NONE;
        assert!(empty.is_empty());

        let read = CapRights::READ;
        assert!(!read.is_empty());
        assert!(read.contains(CapRights::READ));
        assert!(!read.contains(CapRights::WRITE));
    }

    #[test]
    fn cap_rights_union() {
        let rw = CapRights::READ | CapRights::WRITE;
        assert!(rw.contains(CapRights::READ));
        assert!(rw.contains(CapRights::WRITE));
        assert!(!rw.contains(CapRights::GRANT));
    }

    #[test]
    fn cap_rights_intersection() {
        let rw = CapRights::READ | CapRights::WRITE;
        let wg = CapRights::WRITE | CapRights::GRANT;
        let result = rw & wg;

        assert!(result.contains(CapRights::WRITE));
        assert!(!result.contains(CapRights::READ));
        assert!(!result.contains(CapRights::GRANT));
    }

    #[test]
    fn cap_rights_difference() {
        let all = CapRights::ALL;
        let without_write = all.difference(CapRights::WRITE);

        assert!(without_write.contains(CapRights::READ));
        assert!(!without_write.contains(CapRights::WRITE));
        assert!(without_write.contains(CapRights::GRANT));
    }

    #[test]
    fn cap_rights_subset() {
        let read = CapRights::READ;
        let rw = CapRights::READ | CapRights::WRITE;
        let rwg = CapRights::READ | CapRights::WRITE | CapRights::GRANT;

        assert!(read.is_subset_of(rw));
        assert!(read.is_subset_of(rwg));
        assert!(rw.is_subset_of(rwg));
        assert!(!rw.is_subset_of(read));
    }

    #[test]
    fn capability_construction() {
        let cap = Capability::new(
            1,
            ObjectType::Region,
            CapRights::READ | CapRights::WRITE,
            100,
            1,
        );

        assert_eq!(cap.object_id, 1);
        assert_eq!(cap.object_type, ObjectType::Region);
        assert_eq!(cap.badge, 100);
        assert_eq!(cap.epoch, 1);
        assert!(cap.has_rights(CapRights::READ));
        assert!(cap.has_rights(CapRights::WRITE));
        assert!(!cap.has_rights(CapRights::GRANT));
    }

    #[test]
    fn capability_derive_success() {
        let cap = Capability::new(
            1,
            ObjectType::VectorStore,
            CapRights::READ | CapRights::WRITE | CapRights::GRANT | CapRights::PROVE,
            0,
            5,
        );

        let derived = cap.derive(CapRights::READ | CapRights::PROVE, 42).unwrap();

        assert_eq!(derived.object_id, cap.object_id);
        assert_eq!(derived.object_type, cap.object_type);
        assert_eq!(derived.epoch, cap.epoch);
        assert_eq!(derived.badge, 42);
        assert!(derived.has_rights(CapRights::READ));
        assert!(derived.has_rights(CapRights::PROVE));
        assert!(!derived.has_rights(CapRights::WRITE));
    }

    #[test]
    fn capability_derive_fails_without_grant() {
        let cap = Capability::new(
            1,
            ObjectType::Region,
            CapRights::READ | CapRights::WRITE, // No GRANT
            0,
            1,
        );

        assert!(cap.derive(CapRights::READ, 0).is_none());
    }

    #[test]
    fn capability_derive_fails_escalation() {
        let cap = Capability::new(
            1,
            ObjectType::Region,
            CapRights::READ | CapRights::GRANT, // No WRITE
            0,
            1,
        );

        // Trying to derive with WRITE (which we don't have)
        assert!(cap.derive(CapRights::READ | CapRights::WRITE, 0).is_none());
    }

    #[test]
    fn capability_grant_once() {
        let cap = Capability::new(
            1,
            ObjectType::Region,
            CapRights::READ | CapRights::GRANT | CapRights::GRANT_ONCE,
            0,
            1,
        );

        // Derived capability should lose GRANT and GRANT_ONCE
        let derived = cap.derive(CapRights::READ | CapRights::GRANT, 0).unwrap();
        assert!(!derived.has_rights(CapRights::GRANT));
        assert!(!derived.has_rights(CapRights::GRANT_ONCE));
        assert!(derived.has_rights(CapRights::READ));
    }

    #[test]
    fn cap_handle_operations() {
        let h = CapHandle::new(5, 3);
        assert!(!h.is_null());
        assert_eq!(h.raw().id, 5);
        assert_eq!(h.raw().generation, 3);

        let null = CapHandle::null();
        assert!(null.is_null());
    }
}

// =============================================================================
// Region Policy Tests
// =============================================================================

mod region_policy_tests {
    use super::*;

    #[test]
    fn immutable_policy() {
        let policy = RegionPolicy::immutable();

        assert_eq!(policy.as_str(), "Immutable");
        assert!(!policy.is_writable());
        assert!(!policy.allows_overwrite());
        assert!(policy.capacity().is_none());
    }

    #[test]
    fn append_only_policy() {
        let policy = RegionPolicy::append_only(4096);

        assert_eq!(policy.as_str(), "AppendOnly");
        assert!(policy.is_writable());
        assert!(!policy.allows_overwrite());
        assert_eq!(policy.capacity(), Some(4096));
    }

    #[test]
    fn slab_policy() {
        let policy = RegionPolicy::slab(64, 1000);

        assert_eq!(policy.as_str(), "Slab");
        assert!(policy.is_writable());
        assert!(policy.allows_overwrite());
        assert_eq!(policy.capacity(), Some(64_000));
    }

    #[test]
    fn region_handle_operations() {
        let h = RegionHandle::new(10, 20);
        assert!(!h.is_null());
        assert_eq!(h.raw().id, 10);

        let null = RegionHandle::null();
        assert!(null.is_null());
    }

    #[test]
    fn region_policy_default() {
        let policy: RegionPolicy = Default::default();
        assert_eq!(policy, RegionPolicy::Immutable);
    }
}

// =============================================================================
// Task Tests
// =============================================================================

mod task_tests {
    use super::*;

    #[test]
    fn task_priority_ordering() {
        assert!(TaskPriority::Idle < TaskPriority::Normal);
        assert!(TaskPriority::Normal < TaskPriority::High);
        assert!(TaskPriority::High < TaskPriority::RealTime);
        assert!(TaskPriority::RealTime < TaskPriority::Critical);
    }

    #[test]
    fn task_priority_weights() {
        assert_eq!(TaskPriority::Idle.weight(), 0);
        assert_eq!(TaskPriority::Normal.weight(), 25);
        assert_eq!(TaskPriority::High.weight(), 50);
        assert_eq!(TaskPriority::RealTime.weight(), 75);
        assert_eq!(TaskPriority::Critical.weight(), 100);
    }

    #[test]
    fn task_priority_roundtrip() {
        for i in 0..=4 {
            let p = TaskPriority::from_u8(i).unwrap();
            assert_eq!(p as u8, i);
        }
    }

    #[test]
    fn task_priority_invalid() {
        for i in 5..=255 {
            assert!(TaskPriority::from_u8(i).is_none());
        }
    }

    #[test]
    fn task_priority_default() {
        let p: TaskPriority = Default::default();
        assert_eq!(p, TaskPriority::Normal);
    }

    #[test]
    fn task_handle_operations() {
        let h = TaskHandle::new(1, 2);
        assert!(!h.is_null());
        assert_eq!(h.raw().id, 1);
        assert_eq!(h.raw().generation, 2);
    }
}

// =============================================================================
// Constants Tests
// =============================================================================

mod constants_tests {
    use super::*;

    #[test]
    fn attestation_size() {
        // ADR-047 specifies 82-byte attestations
        assert_eq!(ATTESTATION_SIZE, 82);
    }

    #[test]
    fn max_delegation_depth() {
        // ADR-087 Section 20.2 specifies max depth of 8
        assert_eq!(MAX_DELEGATION_DEPTH, 8);
    }

    #[test]
    fn reflex_cache_ttl() {
        // ADR-087 Section 20.4 specifies 100ms TTL
        assert_eq!(REFLEX_CACHE_TTL_MS, 100);
    }

    #[test]
    fn reflex_cache_size() {
        // ADR-087 Section 20.4 specifies 64 entry cache
        assert_eq!(REFLEX_CACHE_SIZE, 64);
    }
}

// =============================================================================
// Property-Based Tests (with proptest)
// =============================================================================

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn handle_raw_roundtrip_prop(id: u32, gen: u32) {
            let h = Handle::new(id, gen);
            let raw = h.to_raw();
            let h2 = Handle::from_raw(raw);
            prop_assert_eq!(h, h2);
        }

        #[test]
        fn cap_rights_union_commutative(a: u32, b: u32) {
            let ra = CapRights::from_bits(a & 0x7F);
            let rb = CapRights::from_bits(b & 0x7F);
            prop_assert_eq!(ra | rb, rb | ra);
        }

        #[test]
        fn cap_rights_intersection_commutative(a: u32, b: u32) {
            let ra = CapRights::from_bits(a & 0x7F);
            let rb = CapRights::from_bits(b & 0x7F);
            prop_assert_eq!(ra & rb, rb & ra);
        }

        #[test]
        fn cap_rights_subset_reflexive(bits: u32) {
            let r = CapRights::from_bits(bits & 0x7F);
            prop_assert!(r.is_subset_of(r));
        }

        #[test]
        fn cap_rights_subset_transitive(a: u32, b: u32, c: u32) {
            let ra = CapRights::from_bits(a & 0x7F);
            let rb = CapRights::from_bits(b & 0x7F);
            let rc = CapRights::from_bits(c & 0x7F);

            if ra.is_subset_of(rb) && rb.is_subset_of(rc) {
                prop_assert!(ra.is_subset_of(rc));
            }
        }

        #[test]
        fn capability_derive_preserves_object(
            object_id: u64,
            badge: u64,
            epoch: u64,
            new_badge: u64
        ) {
            let cap = Capability::new(
                object_id,
                ObjectType::Region,
                CapRights::ALL,
                badge,
                epoch,
            );

            let derived = cap.derive(CapRights::READ, new_badge).unwrap();
            prop_assert_eq!(derived.object_id, object_id);
            prop_assert_eq!(derived.object_type, ObjectType::Region);
            prop_assert_eq!(derived.epoch, epoch);
            prop_assert_eq!(derived.badge, new_badge);
        }

        #[test]
        fn region_policy_slab_capacity(slot_size in 1usize..1000, slot_count in 1usize..1000) {
            let policy = RegionPolicy::slab(slot_size, slot_count);
            prop_assert_eq!(policy.capacity(), Some(slot_size * slot_count));
        }

        #[test]
        fn task_priority_weight_monotonic(a: u8, b: u8) {
            if let (Some(pa), Some(pb)) = (TaskPriority::from_u8(a % 5), TaskPriority::from_u8(b % 5)) {
                if pa < pb {
                    prop_assert!(pa.weight() < pb.weight());
                } else if pa > pb {
                    prop_assert!(pa.weight() > pb.weight());
                } else {
                    prop_assert_eq!(pa.weight(), pb.weight());
                }
            }
        }
    }
}
