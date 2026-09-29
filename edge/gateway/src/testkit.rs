//! Native test helpers: a no-op executor, callers and deterministic data.

use crate::backend::mem::MemBackend;
use crate::service::Call;
use ruvector_edge_auth::subject::edge_subject;
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_store::CallerContext;
use ruvector_edge_tenancy::{derive_tenant_key, TenantKey, UPSTREAM_ISSUER};
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

pub const T0: u64 = 1_790_000_000;
pub const TARGET: &str = "https://gw.example/v1/ops";
pub const REST_MD: &str = "https://gw.example/.well-known/oauth-protected-resource/v1";
pub const MCP_MD: &str = "https://gw.example/.well-known/oauth-protected-resource/v1/mcp";

struct NoopWake;
impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

/// Poll to completion (every backend future here is ready immediately).
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let waker = Waker::from(Arc::new(NoopWake));
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
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

/// `ruvector:read ruvector:write`.
pub fn rw() -> CapabilitySet {
    caps(&[
        Capability::Read,
        Capability::Write,
        Capability::CreateCollection,
    ])
}

/// `ruvector:read` only.
pub fn ro() -> CapabilitySet {
    caps(&[Capability::Read])
}

pub fn ctx(t: &TenantKey, user: &str, scope: CapabilitySet) -> CallerContext {
    CallerContext::new(
        t.clone(),
        sub(user),
        "edc-test",
        "AAECAwQFBgcICQoLDA0ODw",
        "EBESExQVFhcYGRobHB0eHw",
        None,
        scope,
    )
}

pub fn call<'a>(b: &'a MemBackend, c: &'a CallerContext) -> Call<'a, MemBackend> {
    Call {
        b,
        ctx: c,
        dry_run: false,
        now: T0,
    }
}

/// splitmix64 vectors in [-1, 1).
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|_| ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
            .collect()
    }
}
