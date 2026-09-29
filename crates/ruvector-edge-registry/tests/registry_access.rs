//! Access regressions: explicit scope claims with look-alike refusal, yank
//! ownership, write checks on every session mutation, identity-free public
//! manifests, and bounded reads for resolution and listings.

mod common;

use common::*;
use ruvector_edge_auth::Capability;
use ruvector_edge_registry::mem::{FixedClock, MemKv};
use ruvector_edge_registry::ports::{KvStore, StoreError};
use ruvector_edge_registry::registry::PageRequest;
use ruvector_edge_registry::{
    BeginUpload, PartRecord, RegistryError, Scope, ScopeDirectory, Visibility,
};
use std::cell::Cell;

fn scope(s: &str) -> Scope {
    Scope::parse(s).unwrap()
}

#[test]
fn scope_claims_are_explicit_unique_by_skeleton_and_capped() {
    let dir = ScopeDirectory::new(MemKv::new(), FixedClock::at(T0), 2);
    let (a, carol) = (alice(), caller('b', "es1_carol", &ALL_CAPS));
    let c = dir.claim(&a, &scope("acme")).unwrap();
    assert_eq!((c.scope.as_str(), c.claimed_at), ("acme", T0));
    assert_eq!(dir.claim(&a, &scope("acme")).unwrap(), c, "idempotent");
    for squat in ["acrne", "ac-me", "accme"] {
        assert_eq!(
            dir.claim(&carol, &scope(squat)).unwrap_err(),
            RegistryError::ScopeTaken,
            "{squat}"
        );
        assert_eq!(
            dir.claim(&a, &scope(squat)).unwrap_err(),
            RegistryError::ScopeTaken,
            "one look-alike per skeleton, even for the owner"
        );
    }
    assert_eq!(
        dir.claim(&carol, &scope("acme")).unwrap_err(),
        RegistryError::ScopeTaken
    );
    let writer = caller('b', "es1_dan", &[Capability::Read, Capability::Write]);
    assert_eq!(
        dir.claim(&writer, &scope("dans")).unwrap_err(),
        RegistryError::Forbidden(Capability::Admin)
    );
    dir.claim(&a, &scope("second")).unwrap();
    let e = dir.claim(&a, &scope("third")).unwrap_err();
    assert_eq!(e, RegistryError::TooManyScopes);
    assert_eq!(dir.scopes_of(&a.tenant).unwrap(), vec!["acme", "second"]);
    // Pushes need the claim adopted by the scope's DO; adoption is owner-bound.
    let reg = registry();
    let bytes = small_store(1);
    let req = BeginUpload {
        name: name("@acme/pkg"),
        version: ver("1.0.0"),
        visibility: Visibility::Private,
        size: bytes.len() as u64,
        sha256: sha(&bytes),
    };
    assert_eq!(
        reg.begin_upload(&a, req.clone()).unwrap_err(),
        RegistryError::ScopeUnclaimed
    );
    reg.adopt_scope(&c).unwrap();
    assert_eq!(
        reg.adopt_scope(&claim_for(&carol, "acme")).unwrap_err(),
        RegistryError::ScopeTaken
    );
    reg.begin_upload(&a, req).unwrap();
}

#[test]
fn only_the_yanker_or_an_admin_can_unyank() {
    let (reg, r2) = (registry(), FakeR2::default());
    let uploader = caller('a', "es1_up", &[Capability::Read, Capability::Write]);
    let admin = caller('a', "es1_admin", &ALL_CAPS);
    push(
        &reg,
        &r2,
        &uploader,
        "@acme/pkg",
        "1.0.0",
        Visibility::Tenant,
        &small_store(1),
    )
    .unwrap();
    let (n, v) = (name("@acme/pkg"), ver("1.0.0"));
    reg.yank(&admin, &n, &v, "compromised").unwrap();
    assert_eq!(
        reg.unyank(&uploader, &n, &v).unwrap_err(),
        RegistryError::NotOwner
    );
    assert!(reg.get(&uploader, &n, &v).unwrap().yanked.is_some());
    reg.unyank(&admin, &n, &v).unwrap();
    reg.yank(&uploader, &n, &v, "oops").unwrap();
    assert!(reg.unyank(&uploader, &n, &v).unwrap().yanked.is_none());
}

#[test]
fn every_session_mutation_needs_write() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    let bytes = small_store(2);
    let id = begin(&reg, &a, "1.0.0", &bytes);
    let demoted = caller('a', "es1_alice", &[Capability::Read]);
    let part = PartRecord {
        number: 1,
        size: bytes.len() as u64,
        sha256: sha(&bytes),
    };
    let w = RegistryError::Forbidden(Capability::Write);
    assert_eq!(reg.record_part(&demoted, &id, part.clone()).unwrap_err(), w);
    assert_eq!(reg.abort_upload(&demoted, &id).unwrap_err(), w);
    reg.record_part(&a, &id, part).unwrap();
    assert_eq!(reg.finalize_plan(&demoted, &id).unwrap_err(), w);
    assert!(reg.upload(&demoted, &id).is_ok(), "status stays readable");
    let _ = r2;
}

fn keys(v: &serde_json::Value) -> Vec<String> {
    v.as_object().unwrap().keys().cloned().collect()
}

#[test]
fn public_manifests_carry_no_tenant_or_subject() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    push(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        Visibility::Tenant,
        &small_store(3),
    )
    .unwrap();
    publish(&reg, &r2, &a, "@acme/pkg", "1.0.0").unwrap();
    let (n, v) = (name("@acme/pkg"), ver("1.0.0"));
    reg.yank(&a, &n, &v, "superseded").unwrap();
    let carol = caller('b', "es1_carol", &[Capability::Read]);
    let foreign = [
        serde_json::to_value(reg.get(&carol, &n, &v).unwrap()).unwrap(),
        serde_json::to_value(reg.pull(&carol, &n, &v).unwrap().manifest).unwrap(),
    ];
    for j in &foreign {
        let k = keys(j);
        for secret in ["owner", "created_by", "yanked_by"] {
            assert!(!k.iter().any(|x| x == secret), "{secret} in {k:?}");
        }
        assert_eq!(
            j["yanked"],
            serde_json::json!({"at": T0, "reason": "superseded"})
        );
        let text = j.to_string();
        assert!(!text.contains(a.tenant.as_str()) && !text.contains("es1_alice"));
    }
    let own = serde_json::to_value(reg.get(&a, &n, &v).unwrap()).unwrap();
    assert_eq!(own["owner"], a.tenant.as_str());
    assert_eq!(own["created_by"], "es1_alice");
    assert_eq!(own["yanked_by"], "es1_alice");
}

/// Counts bytes returned by `list`, the call resolution and listings make.
struct Counting<'a> {
    inner: &'a MemKv,
    listed: Cell<usize>,
}

impl KvStore for Counting<'_> {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
        self.inner.get(key)
    }
    fn put(&self, key: &str, value: &[u8]) -> Result<(), StoreError> {
        self.inner.put(key, value)
    }
    fn delete(&self, key: &str) -> Result<(), StoreError> {
        self.inner.delete(key)
    }
    fn list(
        &self,
        prefix: &str,
        start_after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, Vec<u8>)>, StoreError> {
        let rows = self.inner.list(prefix, start_after, limit)?;
        let n: usize = rows.iter().map(|(_, v)| v.len()).sum();
        self.listed.set(self.listed.get() + n);
        Ok(rows)
    }
}

#[test]
fn resolution_and_listing_read_compact_rows_only() {
    let (reg, r2, a) = (registry(), FakeR2::default(), alice());
    // Many segments per version: a full manifest is far larger than its row.
    let wide = |seed: u64| {
        let mut rows = Vec::new();
        for i in 0..60u64 {
            rows.push((seed * 1000 + i, vec![i as f32; 4]));
        }
        super_store(&rows)
    };
    for i in 0..10u64 {
        push(
            &reg,
            &r2,
            &a,
            "@acme/pkg",
            &format!("1.0.{i}"),
            Visibility::Tenant,
            &wide(i),
        )
        .unwrap();
    }
    let full: usize = (0..10u64)
        .map(|i| {
            reg.store()
                .get(&format!("ver/acme/pkg/1.0.{i}"))
                .unwrap()
                .unwrap()
                .len()
        })
        .sum();
    let counting = Counting {
        inner: reg.store(),
        listed: Cell::new(0),
    };
    let view = ruvector_edge_registry::Registry::new(
        &counting,
        FixedClock::at(T0),
        ruvector_edge_registry::mem::CounterEntropy::default(),
        config(),
    );
    let (n, bob) = (
        name("@acme/pkg"),
        caller('a', "es1_bob", &[Capability::Read]),
    );
    assert_eq!(
        view.resolve_latest(&bob, &n, false).unwrap().version,
        ver("1.0.9")
    );
    assert_eq!(
        view.list_versions(&bob, &n, &PageRequest::default())
            .unwrap()
            .items
            .len(),
        10
    );
    assert!(
        counting.listed.get() * 4 < full,
        "listed {} bytes, full manifests {full}",
        counting.listed.get()
    );
    // Corrupt full records of the losing versions: listing and resolution
    // never touch them.
    for i in 0..9 {
        reg.store()
            .put(&format!("ver/acme/pkg/1.0.{i}"), b"garbage")
            .unwrap();
    }
    assert_eq!(
        reg.resolve_latest(&bob, &n, false).unwrap().version,
        ver("1.0.9")
    );
    assert_eq!(
        reg.list_versions(&bob, &n, &PageRequest::default())
            .unwrap()
            .items
            .len(),
        10
    );
}

/// A store with one VEC_SEG per row (many segments, each with an entry in
/// the manifest and in the package manifest).
fn super_store(rows: &[(u64, Vec<f32>)]) -> Vec<u8> {
    use rvf_types::SegmentType;
    let mut out = Vec::new();
    let mut dir = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let p = vec_payload(4, std::slice::from_ref(r));
        dir.push((
            i as u64 + 1,
            out.len() as u64,
            p.len() as u64,
            SegmentType::Vec as u8,
        ));
        out.extend(Seg::new(SegmentType::Vec, i as u64 + 1).build(&p, &[]));
    }
    let man = manifest_payload(4, rows.len() as u64, 0, &dir, &[]);
    out.extend(Seg::new(SegmentType::Manifest, 9999).build(&man, &[]));
    out
}
