//! ADR-351 §4.3 / §4.4: DO names, collection uids, shard routing.

use proptest::prelude::*;
use ruvector_edge_tenancy::names::DO_NAME_LEN;
use ruvector_edge_tenancy::{
    derive_tenant_key, do_name, ledger_do_name, shard::shard_hash, shard_for, CollectionUid,
    EntropySource, Service, ShardCount, ShardIndex, TenancyError, UidAllocator, VectorId,
    MAX_SHARDS,
};
use std::cell::Cell;
use std::collections::HashSet;

const ISS: &str = "https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev";

// Cross-checked against coreutils (sha256sum | base32) independently of this crate.
// Changing any of these re-keys tenants or orphans Durable Objects.
const GOLDEN_UID_SALT0_SEQ0_NONCE0: &str = "577b5f4690041596a31b593fcd5d44a3";
const GOLDEN_DO_NAME: &str = "6d7f07c9d5bbd5ccea36e856c939904e7a20d836e330d36bbe07b6d62ad2282c";
const GOLDEN_LEDGER_NAME: &str = "0789b1f04b53838df2b2f48e2f89f81a145fff6855923b832c2f8f39e0c91d24";
const GOLDEN_SHARD_HASH_DOC1: u64 = 0xf9ea_4d80_c358_d15b;

fn tk(org: &str) -> ruvector_edge_tenancy::TenantKey {
    derive_tenant_key(ISS, org, "ws").unwrap()
}

fn is_hex64(s: &str) -> bool {
    s.len() == DO_NAME_LEN
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[test]
fn golden_values_are_stable() {
    let uid = CollectionUid::derive(&[0u8; 32], 0, &[0; 16]);
    assert_eq!(uid.to_hex(), GOLDEN_UID_SALT0_SEQ0_NONCE0);
    let n = do_name(&tk("org"), Service::Vector, &uid, ShardIndex::ZERO);
    assert_eq!(n.as_str(), GOLDEN_DO_NAME);
    assert_eq!(ledger_do_name(&tk("org")).as_str(), GOLDEN_LEDGER_NAME);
    assert_eq!(
        shard_hash(&VectorId::parse("doc-1").unwrap()),
        GOLDEN_SHARD_HASH_DOC1
    );
}

#[test]
fn do_names_are_hex64_and_separate_every_component() {
    let a = tk("org-a");
    let b = tk("org-b");
    let u1 = CollectionUid::derive(&[1; 32], 0, &[0; 16]);
    let u2 = CollectionUid::derive(&[1; 32], 1, &[0; 16]);
    let s0 = ShardIndex::ZERO;
    let s1 = ShardIndex::new(1, ShardCount::new(2).unwrap()).unwrap();
    let mut seen = HashSet::new();
    for t in [&a, &b] {
        for svc in Service::ALL {
            for u in [&u1, &u2] {
                for s in [s0, s1] {
                    let n = do_name(t, svc, u, s);
                    assert!(is_hex64(n.as_str()));
                    assert!(seen.insert(n), "collision");
                }
            }
        }
    }
    seen.insert(ledger_do_name(&a));
    seen.insert(ledger_do_name(&b));
    assert_eq!(seen.len(), 2 * 4 * 2 * 2 + 2);
}

#[test]
fn cross_tenant_same_uid_gets_different_do() {
    // Even if two tenants somehow held the same uid, the tenant key separates them.
    let u = CollectionUid::derive(&[9; 32], 42, &[0; 16]);
    assert_ne!(
        do_name(&tk("a"), Service::Vector, &u, ShardIndex::ZERO),
        do_name(&tk("b"), Service::Vector, &u, ShardIndex::ZERO)
    );
    assert_ne!(ledger_do_name(&tk("a")), ledger_do_name(&tk("b")));
}

/// Deterministic stand-in for the CSPRNG port: each fill writes a distinct
/// counter-derived pattern (distinct across instances via `base`).
struct CounterEntropy {
    base: u64,
    n: Cell<u64>,
}

impl CounterEntropy {
    fn new(base: u64) -> Self {
        CounterEntropy {
            base,
            n: Cell::new(0),
        }
    }
}

impl EntropySource for CounterEntropy {
    fn fill(&self, out: &mut [u8]) -> Result<(), TenancyError> {
        let v = self.base.wrapping_mul(1 << 32).wrapping_add(self.n.get());
        self.n.set(self.n.get() + 1);
        for (i, b) in out.iter_mut().enumerate() {
            *b = v.to_be_bytes()[i % 8] ^ (i as u8);
        }
        Ok(())
    }
}

#[test]
fn delete_then_recreate_yields_new_uid_and_new_do() {
    let tenant = tk("org");
    let rng = CounterEntropy::new(1);
    let mut alloc = UidAllocator::new([3; 32], 0);
    let first = alloc.allocate(&rng).unwrap();
    // Collection "docs" deleted; ledger persisted next_seq = 1.
    let mut restored = UidAllocator::new(*alloc.salt(), alloc.next_seq());
    let second = restored.allocate(&rng).unwrap();
    assert_eq!(restored.next_seq(), 2);
    assert_ne!(first, second);
    assert_ne!(
        do_name(&tenant, Service::Vector, &first, ShardIndex::ZERO),
        do_name(&tenant, Service::Vector, &second, ShardIndex::ZERO)
    );
}

/// Regression (uid finding): a ledger rolled back to an earlier
/// `(salt, next_seq)` (operator PITR, stale snapshot) must not reproduce a
/// deleted collection's uid, or the next create would land on the orphaned
/// VectorShard whose `meta` matches exactly and serve its old vectors.
#[test]
fn rolled_back_ledger_does_not_reproduce_deleted_uid() {
    let snapshot = UidAllocator::new([4; 32], 7);
    let mut live = snapshot.clone();
    let deleted = live.allocate(&CounterEntropy::new(10)).unwrap();
    // Roll the ledger back to the snapshot and create again.
    let mut rolled_back = snapshot.clone();
    let recreated = rolled_back.allocate(&CounterEntropy::new(20)).unwrap();
    assert_eq!(rolled_back.next_seq(), live.next_seq(), "same seq reused");
    assert_ne!(deleted, recreated);
    let tenant = tk("org");
    assert_ne!(
        do_name(&tenant, Service::Vector, &deleted, ShardIndex::ZERO),
        do_name(&tenant, Service::Vector, &recreated, ShardIndex::ZERO)
    );
}

#[test]
fn allocator_debug_never_prints_salt() {
    let a = UidAllocator::new([0x5A; 32], 1);
    let s = format!("{a:?}");
    assert!(s.contains("redacted"), "{s}");
    assert!(!s.contains("90"), "salt byte 0x5A leaked: {s}");
}

#[test]
fn allocator_uids_unique_over_many_seqs() {
    let rng = CounterEntropy::new(3);
    let mut alloc = UidAllocator::new([5; 32], 0);
    let uids: HashSet<_> = (0..10_000).map(|_| alloc.allocate(&rng).unwrap()).collect();
    assert_eq!(uids.len(), 10_000);
    assert_eq!(alloc.next_seq(), 10_000);
}

#[test]
fn salts_separate_tenants_allocators() {
    assert_ne!(
        CollectionUid::derive(&[1; 32], 0, &[0; 16]),
        CollectionUid::derive(&[2; 32], 0, &[0; 16])
    );
}

#[test]
fn uid_parse_round_trip_and_strict() {
    let u = CollectionUid::derive(&[7; 32], 7, &[0; 16]);
    assert_eq!(CollectionUid::parse(&u.to_hex()).unwrap(), u);
    assert_eq!(CollectionUid::from_bytes(*u.as_bytes()), u);
    assert_eq!(format!("{u}"), u.to_hex());
    let upper = u.to_hex().to_ascii_uppercase();
    for bad in [
        upper.as_str(),
        "",
        "00",
        &u.to_hex()[..31],
        "0123456789abcdef0123456789abcdeg",
        "+123456789abcdef0123456789abcdef",
        "0123456789abcdef0123456789abcdef0",
        "é123456789abcdef0123456789abcde",
    ] {
        assert_eq!(
            CollectionUid::parse(bad),
            Err(TenancyError::MalformedIdentifier("collection_uid")),
            "{bad:?}"
        );
    }
}

#[test]
fn service_parse_is_exact() {
    for s in Service::ALL {
        assert_eq!(Service::parse(s.as_str()).unwrap(), s);
    }
    for bad in ["Vector", "vector ", "", "ledger", "vector|x"] {
        assert!(Service::parse(bad).is_err());
    }
}

#[test]
fn shard_count_bounds() {
    assert!(ShardCount::new(0).is_err());
    assert!(ShardCount::new(MAX_SHARDS + 1).is_err());
    assert!(ShardCount::new(u32::MAX).is_err());
    for n in 1..=MAX_SHARDS {
        let c = ShardCount::new(n).unwrap();
        assert_eq!(c.indices().count() as u32, n);
        assert_eq!(
            c.indices().map(ShardIndex::get).collect::<Vec<_>>(),
            (0..n).collect::<Vec<_>>()
        );
    }
    assert_eq!(ShardCount::ONE.get(), 1);
}

#[test]
fn shard_index_bounds_and_parse() {
    let c = ShardCount::new(3).unwrap();
    assert!(ShardIndex::new(2, c).is_ok());
    assert!(ShardIndex::new(3, c).is_err());
    assert_eq!(ShardIndex::parse("0").unwrap(), ShardIndex::ZERO);
    assert_eq!(ShardIndex::parse("5").unwrap().get(), 5);
    for bad in [
        "",
        "6",
        "-1",
        "+1",
        "01",
        "00",
        " 1",
        "1 ",
        "x",
        "4294967296",
        "99999999999",
    ] {
        assert!(ShardIndex::parse(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn single_shard_routes_everything_to_zero() {
    for id in ["a", "b", "doc-1", "\u{1F600}"] {
        assert_eq!(
            shard_for(&VectorId::parse(id).unwrap(), ShardCount::ONE),
            ShardIndex::ZERO
        );
    }
}

#[test]
fn shard_routing_is_roughly_uniform() {
    let c = ShardCount::new(6).unwrap();
    let mut buckets = [0u32; 6];
    for i in 0..6000 {
        let id = VectorId::parse(&format!("id-{i}")).unwrap();
        buckets[shard_for(&id, c).get() as usize] += 1;
    }
    for b in buckets {
        assert!((850..=1150).contains(&b), "{buckets:?}");
    }
}

#[test]
fn shard_routing_distinguishes_nfc_and_nfd_ids() {
    let nfc = VectorId::parse("caf\u{00E9}").unwrap();
    let nfd = VectorId::parse("cafe\u{0301}").unwrap();
    assert_ne!(nfc, nfd);
    assert_ne!(shard_hash(&nfc), shard_hash(&nfd));
}

proptest! {
    #[test]
    fn prop_shard_in_range_and_stable(id in "[^\\p{Cc}]{1,64}", n in 1u32..=MAX_SHARDS) {
        if let Ok(v) = VectorId::parse(&id) {
            let c = ShardCount::new(n).unwrap();
            let s = shard_for(&v, c);
            prop_assert!(s.get() < n);
            prop_assert_eq!(s, shard_for(&v, c));
            prop_assert_eq!(u64::from(s.get()), shard_hash(&v) % u64::from(n));
        }
    }

    #[test]
    fn prop_uid_hex_round_trip(bytes in any::<[u8; 16]>()) {
        let u = CollectionUid::from_bytes(bytes);
        prop_assert_eq!(CollectionUid::parse(&u.to_hex()).unwrap(), u);
    }

    #[test]
    fn prop_distinct_seq_distinct_uid(salt in any::<[u8; 32]>(), a in any::<u64>(), b in any::<u64>()) {
        prop_assume!(a != b);
        prop_assert_ne!(CollectionUid::derive(&salt, a, &[0; 16]), CollectionUid::derive(&salt, b, &[0; 16]));
    }
}
