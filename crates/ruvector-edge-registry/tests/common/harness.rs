//! Registry harness: in-memory index, fake R2, and the Worker-side push and
//! publish loops, following the documented contracts (freeze, re-measure
//! parts, copy only verified bytes, create-only public writes, evidence).

use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_registry::mem::{CounterEntropy, FixedClock, MemKv};
use ruvector_edge_registry::registry::{PublishOutcome, RegistryConfig};
use ruvector_edge_registry::upload::{UploadId, UploadLimits};
use ruvector_edge_registry::{
    BeginUpload, Caller, FinalizeReport, ObjectEvidence, PackageManifest, PackageName, PartRecord,
    Registry, RegistryError, ScopeClaim, StreamValidator, Version, Visibility,
};
use ruvector_edge_tenancy::TenantKey;
use std::cell::RefCell;
use std::collections::HashMap;

pub type Reg = Registry<MemKv, FixedClock, CounterEntropy>;

pub const T0: u64 = 1_700_000_000;

pub fn config() -> RegistryConfig {
    RegistryConfig {
        upload: UploadLimits {
            min_part_size: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn registry() -> Reg {
    Registry::new(
        MemKv::new(),
        FixedClock::at(T0),
        CounterEntropy::default(),
        config(),
    )
}

pub fn tenant(c: char) -> TenantKey {
    TenantKey::parse(&c.to_string().repeat(26)).unwrap()
}

pub const ALL_CAPS: [Capability; 4] = [
    Capability::Read,
    Capability::Write,
    Capability::Admin,
    Capability::PublishPublic,
];

pub fn caps(list: &[Capability]) -> CapabilitySet {
    let mut s = CapabilitySet::EMPTY;
    list.iter().for_each(|c| s.insert(*c));
    s
}

pub fn caller(t: char, sub: &str, list: &[Capability]) -> Caller {
    Caller {
        tenant: tenant(t),
        sub: sub.to_string(),
        caps: caps(list),
    }
}

pub fn alice() -> Caller {
    caller('a', "es1_alice", &ALL_CAPS)
}

pub fn name(s: &str) -> PackageName {
    PackageName::parse(s).unwrap()
}

pub fn ver(s: &str) -> Version {
    Version::parse(s).unwrap()
}

pub fn sha(b: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    sha2::Sha256::digest(b).into()
}

pub fn evidence(b: &[u8]) -> ObjectEvidence {
    ObjectEvidence {
        sha256: sha(b),
        size: b.len() as u64,
    }
}

/// Fake R2 bucket.
#[derive(Default)]
pub struct FakeR2 {
    pub objects: RefCell<HashMap<String, Vec<u8>>>,
}

impl FakeR2 {
    pub fn get(&self, k: &str) -> Option<Vec<u8>> {
        self.objects.borrow().get(k).cloned()
    }
    pub fn put(&self, k: &str, v: Vec<u8>) {
        self.objects.borrow_mut().insert(k.to_string(), v);
    }
    /// Conditional create: `false` (and no write) if the key exists.
    pub fn put_if_absent(&self, k: &str, v: Vec<u8>) -> bool {
        let mut o = self.objects.borrow_mut();
        if o.contains_key(k) {
            return false;
        }
        o.insert(k.to_string(), v);
        true
    }
    pub fn delete(&self, k: &str) {
        self.objects.borrow_mut().remove(k);
    }
}

/// A directory claim of `scope` for `who` (tests adopt it directly).
pub fn claim_for(who: &Caller, scope: &str) -> ScopeClaim {
    ScopeClaim {
        scope: scope.to_string(),
        owner: who.tenant.clone(),
        claimed_by: who.sub.clone(),
        claimed_at: T0,
    }
}

/// Freeze and finalize with a caller-supplied validation result (negative
/// tests); streamed parts are the frozen ones, copy evidence the declared.
pub fn finalize_raw(
    reg: &Reg,
    who: &Caller,
    id: &UploadId,
    validated: Result<
        ruvector_edge_registry::ValidatedRvf,
        ruvector_edge_registry::ValidationError,
    >,
) -> Result<PackageManifest, RegistryError> {
    let plan = reg.finalize_plan(who, id)?;
    let copied = plan.copy_required.then_some(ObjectEvidence {
        sha256: plan.sha256,
        size: plan.size,
    });
    reg.finalize(
        who,
        id,
        FinalizeReport {
            validated,
            streamed: plan.parts,
            copied,
            provenance: None,
        },
    )
}

/// The directory claim the gateway would obtain, adopted by the scope DO.
/// First pusher's tenant wins, as a directory claim would.
pub fn adopt(reg: &Reg, who: &Caller, pkg: &str) {
    let n = name(pkg);
    if reg.scope_record(n.scope()).unwrap().is_none() {
        reg.adopt_scope(&claim_for(who, n.scope().as_str()))
            .unwrap();
    }
}

/// Declared digest / size overrides for negative tests.
#[derive(Default, Clone, Copy)]
pub struct Declare {
    pub sha256: Option<[u8; 32]>,
    pub size: Option<u64>,
}

/// Upload parts to staging and record them.
pub fn stage(reg: &Reg, r2: &FakeR2, who: &Caller, id: &UploadId, bytes: &[u8], parts: usize) {
    let s = reg.upload(who, id).unwrap();
    let staging = ruvector_edge_registry::keys::staging_key(&s.tenant, id);
    let chunk = bytes.len().div_ceil(parts.max(1)).max(1);
    for (i, c) in bytes.chunks(chunk).enumerate() {
        let n = i as u32 + 1;
        r2.put(&format!("{staging}#{n}"), c.to_vec());
        reg.record_part(
            who,
            id,
            PartRecord {
                number: n,
                size: c.len() as u64,
                sha256: sha(c),
            },
        )
        .unwrap();
    }
}

/// The Worker's finish: plan (freeze), stream the frozen parts through the
/// validator re-measuring each, write the blob only when valid, digest-equal
/// and `copy_required`, then finalize with what it measured.
pub fn finish(
    reg: &Reg,
    r2: &FakeR2,
    who: &Caller,
    id: &UploadId,
) -> Result<PackageManifest, RegistryError> {
    let plan = reg.finalize_plan(who, id)?;
    let mut v = StreamValidator::new(plan.limits);
    let mut whole = Vec::new();
    let mut streamed = Vec::new();
    for p in &plan.parts {
        let part = r2.get(&format!("{}#{}", plan.staging, p.number)).unwrap();
        let _ = v.push(&part);
        streamed.push(PartRecord {
            number: p.number,
            size: part.len() as u64,
            sha256: sha(&part),
        });
        whole.extend_from_slice(&part);
    }
    let validated = v.finish();
    let mut copied = None;
    if plan.copy_required && validated.as_ref().is_ok_and(|r| r.sha256 == plan.sha256) {
        r2.put(plan.blob.as_str(), whole);
        copied = Some(evidence(&r2.get(plan.blob.as_str()).unwrap()));
    }
    reg.finalize(
        who,
        id,
        FinalizeReport {
            validated,
            streamed,
            copied,
            provenance: None,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn push_with(
    reg: &Reg,
    r2: &FakeR2,
    who: &Caller,
    pkg: &str,
    version: &str,
    visibility: Visibility,
    bytes: &[u8],
    parts: usize,
    declare: Declare,
) -> Result<PackageManifest, RegistryError> {
    adopt(reg, who, pkg);
    let req = BeginUpload {
        name: name(pkg),
        version: ver(version),
        visibility,
        size: declare.size.unwrap_or(bytes.len() as u64),
        sha256: declare.sha256.unwrap_or_else(|| sha(bytes)),
    };
    let s = reg.begin_upload(who, req)?;
    stage(reg, r2, who, &s.id, bytes, parts);
    finish(reg, r2, who, &s.id)
}

pub fn push(
    reg: &Reg,
    r2: &FakeR2,
    who: &Caller,
    pkg: &str,
    version: &str,
    vis: Visibility,
    bytes: &[u8],
) -> Result<PackageManifest, RegistryError> {
    push_with(
        reg,
        r2,
        who,
        pkg,
        version,
        vis,
        bytes,
        2,
        Declare::default(),
    )
}

/// The Worker's publish: create-only write of the public object (or read
/// back the existing one), measure it, commit with the evidence.
pub fn publish(
    reg: &Reg,
    r2: &FakeR2,
    who: &Caller,
    pkg: &str,
    version: &str,
) -> Result<PublishOutcome, RegistryError> {
    let (n, v) = (name(pkg), ver(version));
    let plan = reg.publish_plan(who, &n, &v)?;
    let ev = if plan.already_public {
        evidence(&r2.get(plan.to.as_str()).unwrap_or_default())
    } else {
        let src = r2.get(plan.from.as_str()).unwrap_or_default();
        r2.put_if_absent(plan.to.as_str(), src);
        evidence(&r2.get(plan.to.as_str()).unwrap())
    };
    reg.publish_commit(who, &n, &v, ev)
}

pub fn small_store(seed: u64) -> Vec<u8> {
    super::packed_store(4, &super::rows(3, 4, seed), &[], 2)
}

/// Open a tenant-visible upload session for `@acme/pkg`.
pub fn begin(reg: &Reg, who: &Caller, v: &str, bytes: &[u8]) -> UploadId {
    adopt(reg, who, "@acme/pkg");
    reg.begin_upload(
        who,
        BeginUpload {
            name: name("@acme/pkg"),
            version: ver(v),
            visibility: Visibility::Tenant,
            size: bytes.len() as u64,
            sha256: sha(bytes),
        },
    )
    .unwrap()
    .id
}
