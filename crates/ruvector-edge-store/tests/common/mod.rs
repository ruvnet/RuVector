//! Deterministic test ports and helpers (no `rand`, no clock).
#![allow(dead_code)]

use core::cell::Cell;
use ruvector_edge_auth::subject::edge_subject;
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_store::{
    CallerContext, Clock, Dispatcher, EntropySource, LocalCluster, SqlStore,
};
use ruvector_edge_tenancy::{
    derive_tenant_key, QuotaLimits, TenancyError, TenantKey, UPSTREAM_ISSUER,
};
use serde_json::{json, Value as Json};

pub const TARGET: &str =
    "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/ops";
pub const T0: u64 = 1_790_000_000;

/// splitmix64.
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.f32()).collect()
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

pub struct FixedClock(pub Cell<u64>);
impl FixedClock {
    pub fn at(t: u64) -> Self {
        FixedClock(Cell::new(t))
    }
    pub fn advance(&self, s: u64) {
        self.0.set(self.0.get() + s);
    }
}
impl Clock for FixedClock {
    fn now_unix(&self) -> u64 {
        self.0.get()
    }
}

/// Counter-seeded deterministic "CSPRNG" for tests.
pub struct CounterEntropy(pub Cell<u64>);
impl CounterEntropy {
    pub fn new(seed: u64) -> Self {
        CounterEntropy(Cell::new(seed))
    }
}
impl EntropySource for CounterEntropy {
    fn fill(&self, out: &mut [u8]) -> Result<(), TenancyError> {
        let mut r = Rng(self.0.get());
        for b in out.iter_mut() {
            *b = r.next_u64() as u8;
        }
        self.0.set(self.0.get().wrapping_add(1));
        Ok(())
    }
}

/// Always the same bytes (drives the uid-collision retry path).
pub struct ConstEntropy(pub u8);
impl EntropySource for ConstEntropy {
    fn fill(&self, out: &mut [u8]) -> Result<(), TenancyError> {
        out.fill(self.0);
        Ok(())
    }
}

pub fn limits() -> QuotaLimits {
    QuotaLimits {
        max_collections: 20,
        max_vectors: 250_000,
        max_float_budget: 100_000_000,
        max_bytes: 1 << 30,
        max_daily_ops: 1_000_000,
    }
}

pub fn tenant(org: &str) -> TenantKey {
    derive_tenant_key(UPSTREAM_ISSUER, org, "ws1").unwrap()
}

pub fn sub(user: &str) -> String {
    edge_subject(UPSTREAM_ISSUER, user)
}

pub fn caps(list: &[Capability]) -> CapabilitySet {
    let mut s = CapabilitySet::EMPTY;
    list.iter().for_each(|c| s.insert(*c));
    s
}

pub fn read_caps() -> CapabilitySet {
    caps(&[Capability::Read])
}

pub fn write_caps() -> CapabilitySet {
    caps(&[
        Capability::Read,
        Capability::Write,
        Capability::CreateCollection,
    ])
}

pub fn ctx(t: &TenantKey, user: &str, c: CapabilitySet) -> CallerContext {
    CallerContext::new(
        t.clone(),
        sub(user),
        "client-1",
        "AAAAAAAAAAAAAAAAAAAAAA",
        "BBBBBBBBBBBBBBBBBBBBBB",
        None,
        c,
    )
}

/// A 26-char op id derived from `n`.
pub fn op_id(n: u64) -> String {
    format!("{n:0>26}")
}

pub fn envelope(t: &TenantKey, op: &str, id: &str, args: Json, dry_run: bool) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "v": 1, "op_id": id, "target": TARGET, "tenant_key": t.as_str(),
        "op": op, "args": args, "dry_run": dry_run,
    }))
    .unwrap()
}

/// Test harness: one cluster, dispatcher, clock and entropy.
pub struct Harness<S> {
    pub cluster: LocalCluster<S>,
    pub disp: Dispatcher,
    pub clock: FixedClock,
    pub entropy: CounterEntropy,
    pub next_id: Cell<u64>,
}

impl<S: SqlStore + Default> Harness<S> {
    pub fn new() -> Self {
        Harness {
            cluster: LocalCluster::new(limits()),
            disp: Dispatcher::new(TARGET),
            clock: FixedClock::at(T0),
            entropy: CounterEntropy::new(42),
            next_id: Cell::new(1),
        }
    }

    /// Dispatch with a fresh op id; returns (status, parsed response).
    pub fn call(&mut self, c: &CallerContext, op: &str, args: Json) -> (u16, Json) {
        let id = op_id(self.next_id.get());
        self.next_id.set(self.next_id.get() + 1);
        self.call_with(c, op, &id, args, false)
    }

    pub fn call_with(
        &mut self,
        c: &CallerContext,
        op: &str,
        id: &str,
        args: Json,
        dry_run: bool,
    ) -> (u16, Json) {
        let body = envelope(c.tenant_key(), op, id, args, dry_run);
        let r = self.disp.dispatch(
            &mut self.cluster,
            c,
            &body,
            Some(id),
            &self.clock,
            &self.entropy,
        );
        (r.status, serde_json::from_str(&r.body).unwrap())
    }

    /// Claim the tenant for `owner` and add `members` with roles.
    pub fn claim(
        &mut self,
        t: &TenantKey,
        owner: &str,
        members: &[(&str, ruvector_edge_tenancy::Role)],
    ) {
        let lm = ruvector_edge_store::ledger_meta_for(t).unwrap();
        let now = self.clock.now_unix();
        let (store, ledger) = self.cluster.ledger(t).unwrap();
        ledger.claim(store, &lm, &sub(owner), now).unwrap();
        for (m, r) in members {
            ledger
                .put_member(store, &lm, &sub(owner), &sub(m), *r, now)
                .unwrap();
        }
    }
}

pub fn code(resp: &Json) -> &str {
    resp["error"]["code"].as_str().unwrap_or("")
}
