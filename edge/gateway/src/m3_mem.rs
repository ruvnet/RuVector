//! Native test doubles for the M3 ports: in-memory R2 (with multipart and
//! range reads), Queues, a deterministic Workers AI, the M3 side channel
//! over `MemBackend` (the real `m3_ledger` / `m3_shard` cores on the same
//! `MemSqlStore`s the M1 cores use), and a request harness.

use crate::backend::mem::{CounterEntropy, MemBackend};
use crate::ledger_core::FREE_PLAN;
use crate::m3_api::{dispatch, parse, Input, Out};
use crate::m3_ctx::M3;
use crate::m3_ports::{Blob, QueueName, Queues};
use crate::m3_wire::M3Backend;
use crate::rest::{self, Caller};
use crate::testkit::{block_on, caps, ctx, tenant, REST_MD, T0};
use crate::{m3_ledger, m3_shard};
use ruvector_edge_auth::Capability;
use ruvector_edge_snapshot::{EmbedError, EmbedOptions, EmbeddingPort, ObjectFacts};
use ruvector_edge_store::{ErrorCode, OpError};
use ruvector_edge_tenancy::DoName;
use serde_json::{json, Value as Json};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::future::Future;
use worker::Method;

thread_local! {
    static SEED: Cell<u64> = const { Cell::new(0x5eed) };
}

impl M3Backend for MemBackend {
    async fn m3_ledger(&self, name: &DoName, body: String) -> Result<String, OpError> {
        let seed = SEED.with(|s| {
            let v = s.get().wrapping_add(0x9E37_79B9);
            s.set(v);
            v
        });
        let entropy = CounterEntropy(Cell::new(seed));
        let mut all = self.ledgers.borrow_mut();
        let (store, slot) = all.entry(name.as_str().to_string()).or_default();
        Ok(m3_ledger::serve(
            slot,
            &*store,
            FREE_PLAN,
            body.as_bytes(),
            &entropy,
        ))
    }

    async fn m3_shard(&self, name: &DoName, body: String) -> Result<String, OpError> {
        if self.shard_down.get() {
            return Err(crate::wire::unavailable());
        }
        let mut all = self.shards.borrow_mut();
        let store = all.entry(name.as_str().to_string()).or_default();
        let key = name.as_str();
        Ok(m3_shard::serve(
            &mut self.host.borrow_mut(),
            key,
            Some(key),
            &*store,
            body.as_bytes(),
        ))
    }
}

/// A stored object: bytes plus the sha256 R2 was given.
type Stored = (Vec<u8>, Option<[u8; 32]>);
/// An open multipart upload: key plus parts.
type Multipart = (String, BTreeMap<u16, Vec<u8>>);

/// In-memory R2.
#[derive(Default)]
pub struct MemBlob {
    /// `key → (bytes, stored sha256)`.
    pub objects: RefCell<BTreeMap<String, Stored>>,
    uploads: RefCell<BTreeMap<String, Multipart>>,
    next: Cell<u64>,
    /// Largest range read served.
    pub max_range: Cell<u64>,
    /// Range reads served.
    pub range_reads: Cell<u64>,
    /// Fail every `put` (R2 outage).
    pub fail_puts: Cell<bool>,
}

impl MemBlob {
    /// Object bytes.
    pub fn bytes(&self, key: &str) -> Option<Vec<u8>> {
        self.objects.borrow().get(key).map(|(b, _)| b.clone())
    }
    /// Keys under `prefix`.
    pub fn keys(&self, prefix: &str) -> Vec<String> {
        let o = self.objects.borrow();
        o.keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect()
    }
}

impl Blob for MemBlob {
    async fn put(&self, key: &str, bytes: Vec<u8>, sha: Option<[u8; 32]>) -> Result<(), OpError> {
        if self.fail_puts.get() {
            return Err(crate::m3_ports::storage_err());
        }
        if let Some(s) = sha {
            use sha2::{Digest, Sha256};
            let d: [u8; 32] = Sha256::digest(&bytes).into();
            if d != s {
                return Err(OpError::invalid("checksum"));
            }
        }
        self.objects.borrow_mut().insert(key.into(), (bytes, sha));
        Ok(())
    }
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, OpError> {
        Ok(self.bytes(key))
    }
    async fn get_range(&self, key: &str, off: u64, len: u64) -> Result<Option<Vec<u8>>, OpError> {
        self.range_reads.set(self.range_reads.get() + 1);
        self.max_range.set(self.max_range.get().max(len));
        let o = self.objects.borrow();
        Ok(o.get(key).map(|(b, _)| {
            let s = (off as usize).min(b.len());
            let e = (s + len as usize).min(b.len());
            b[s..e].to_vec()
        }))
    }
    async fn head(&self, key: &str) -> Result<Option<ObjectFacts>, OpError> {
        let o = self.objects.borrow();
        Ok(o.get(key).map(|(b, s)| ObjectFacts {
            size: b.len() as u64,
            sha256: *s,
        }))
    }
    async fn sha256_stream(&self, key: &str) -> Result<Option<([u8; 32], u64)>, OpError> {
        use sha2::{Digest, Sha256};
        Ok(self
            .bytes(key)
            .map(|b| (Sha256::digest(&b).into(), b.len() as u64)))
    }
    async fn delete(&self, key: &str) -> Result<(), OpError> {
        self.objects.borrow_mut().remove(key);
        Ok(())
    }
    async fn mp_begin(&self, key: &str) -> Result<String, OpError> {
        let id = format!("mp{}", self.next.get());
        self.next.set(self.next.get() + 1);
        let mut u = self.uploads.borrow_mut();
        u.insert(id.clone(), (key.to_string(), BTreeMap::new()));
        Ok(id)
    }
    async fn mp_part(
        &self,
        key: &str,
        up: &str,
        n: u16,
        bytes: Vec<u8>,
    ) -> Result<String, OpError> {
        let mut u = self.uploads.borrow_mut();
        let (k, parts) = u.get_mut(up).ok_or(OpError::not_found())?;
        if k != key {
            return Err(OpError::not_found());
        }
        parts.insert(n, bytes);
        Ok(format!("etag-{n}"))
    }
    async fn mp_complete(
        &self,
        key: &str,
        up: &str,
        parts: &[(u16, String)],
    ) -> Result<u64, OpError> {
        let (k, stored) = self
            .uploads
            .borrow_mut()
            .remove(up)
            .ok_or(OpError::not_found())?;
        if k != key {
            return Err(OpError::not_found());
        }
        let mut out = Vec::new();
        let mut size0 = None;
        for (i, (n, etag)) in parts.iter().enumerate() {
            let p = stored.get(n).ok_or(OpError::invalid("part"))?;
            if *etag != format!("etag-{n}") {
                return Err(OpError::invalid("etag"));
            }
            // R2: every part but the last has the same size.
            if i + 1 < parts.len() && *size0.get_or_insert(p.len()) != p.len() {
                return Err(OpError::invalid("unequal parts"));
            }
            out.extend_from_slice(p);
        }
        let n = out.len() as u64;
        self.objects.borrow_mut().insert(key.into(), (out, None));
        Ok(n)
    }
    async fn mp_abort(&self, _key: &str, up: &str) -> Result<(), OpError> {
        self.uploads.borrow_mut().remove(up);
        Ok(())
    }
}

/// In-memory Queues producer.
#[derive(Default)]
pub struct MemQueues {
    /// Sent messages.
    pub sent: RefCell<Vec<(QueueName, Json)>>,
    /// Refuse `ruvector-edge-ingest` sends.
    pub fail_ingest: Cell<bool>,
}

impl MemQueues {
    /// Drain the messages of `q`.
    pub fn take(&self, q: QueueName) -> Vec<Json> {
        let mut all = self.sent.borrow_mut();
        let (hit, keep): (Vec<_>, Vec<_>) = all.drain(..).partition(|(n, _)| *n == q);
        *all = keep;
        hit.into_iter().map(|(_, j)| j).collect()
    }
}

impl Queues for MemQueues {
    async fn send(&self, q: QueueName, body: Json) -> Result<(), OpError> {
        if q == QueueName::Ingest && self.fail_ingest.get() {
            return Err(crate::m3_ports::storage_err());
        }
        self.sent.borrow_mut().push((q, body));
        Ok(())
    }
}

/// Deterministic 384-dim "bge" (splitmix of the text); `jitter` makes
/// every call differ (non-deterministic model), `down` fails the port.
#[derive(Default)]
pub struct MockAi {
    /// Model calls.
    pub calls: Cell<u64>,
    /// Texts embedded.
    pub texts: Cell<u64>,
    /// Perturb outputs per call.
    pub jitter: Cell<bool>,
    /// Fail every call.
    pub down: Cell<bool>,
}

/// The mock's embedding of `text` (call counter `k` when jittering).
pub fn mock_vec(text: &str, k: u64) -> Vec<f32> {
    let mut s = text.bytes().fold(0xcbf2_9ce4_8422_2325u64 ^ k, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    });
    let mut v: Vec<f32> = (0..384)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

impl EmbeddingPort for MockAi {
    fn embed(
        &self,
        model: &str,
        texts: &[&str],
        _o: EmbedOptions,
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbedError>> {
        assert_eq!(model, "@cf/baai/bge-small-en-v1.5");
        self.calls.set(self.calls.get() + 1);
        self.texts.set(self.texts.get() + texts.len() as u64);
        let k = if self.jitter.get() {
            self.calls.get()
        } else {
            0
        };
        let out = if self.down.get() {
            Err(EmbedError::Port("down"))
        } else {
            Ok(texts.iter().map(|t| mock_vec(t, k)).collect())
        };
        std::future::ready(out)
    }
}

/// Everything a test needs, with a settable clock.
pub struct World {
    /// DOs.
    pub b: MemBackend,
    /// R2.
    pub blob: MemBlob,
    /// Queues.
    pub q: MemQueues,
    /// Workers AI.
    pub ai: MockAi,
    /// Unix ms.
    pub now_ms: Cell<u64>,
}

impl Default for World {
    fn default() -> Self {
        World {
            b: MemBackend::new(),
            blob: MemBlob::default(),
            q: MemQueues::default(),
            ai: MockAi::default(),
            now_ms: Cell::new(T0 * 1000),
        }
    }
}

/// A caller of tenant `org` as `user` with `list` capabilities.
pub fn caller(org: &str, user: &str, list: &[Capability]) -> Caller {
    Caller {
        ctx: ctx(&tenant(org), user, caps(list)),
        org_id: org.into(),
        workspace_id: "ws1".into(),
        scopes: vec![],
    }
}

/// Every capability.
pub const ALL: &[Capability] = &[
    Capability::Read,
    Capability::Write,
    Capability::CreateCollection,
    Capability::Admin,
];

impl World {
    /// M3 context for `c`.
    pub fn m3<'a>(&'a self, c: &'a Caller) -> M3<'a, MemBackend, MemBlob, MemQueues> {
        M3 {
            b: &self.b,
            blob: &self.blob,
            queues: &self.q,
            ctx: &c.ctx,
            now_ms: self.now_ms.get(),
            note: crate::audit::Note::default(),
        }
    }

    /// Serve one request (M3 table first, then M1), JSON body.
    pub fn req(&self, c: &Caller, m: Method, path: &str, body: Json) -> (u16, Json) {
        let bytes = if body.is_null() {
            Vec::new()
        } else {
            body.to_string().into_bytes()
        };
        self.raw(c, m, path, bytes, false, None)
    }

    /// Serve one request with a raw body.
    pub fn raw(
        &self,
        c: &Caller,
        m: Method,
        path: &str,
        body: Vec<u8>,
        octet: bool,
        key: Option<&str>,
    ) -> (u16, Json) {
        let (path, query) = match path.split_once('?') {
            Some((p, q)) => (p, Some(q)),
            None => (path, None),
        };
        let rep = match parse(&m, path) {
            Some(route) => {
                let inp = Input {
                    route: &route,
                    body,
                    octet_stream: octet,
                    query,
                    key,
                };
                match block_on(dispatch(&self.m3(c), &self.ai, c, REST_MD, inp)) {
                    Out::Reply(r) => r,
                    Out::Download(d) => {
                        let v = json!({ "key": d.key, "size": d.size });
                        return (200, v);
                    }
                }
            }
            None => {
                let route = rest::parse(&m, path).expect(path);
                let now = self.now_ms.get() / 1000;
                block_on(rest::handle(&self.b, c, &route, &body, key, now, REST_MD))
            }
        };
        let v = serde_json::from_str(&rep.body).unwrap_or(Json::Null);
        (rep.status, v)
    }

    /// A claimed tenant `org` whose owner is `owner` (every capability).
    pub fn owner(&self, org: &str, owner: &str) -> Caller {
        let c = caller(org, owner, ALL);
        let (s, _) = self.req(&c, Method::Post, "/v1/claim", Json::Null);
        assert_eq!(s, 201);
        c
    }
}

/// `true` if `e` has code `code`.
pub fn is(e: &Json, code: ErrorCode) -> bool {
    e["code"] == code.as_str()
}

impl World {
    /// A world whose tenants have `limits`.
    pub fn with_limits(limits: ruvector_edge_tenancy::QuotaLimits) -> Self {
        let registry = ruvector_edge_store::ResidentRegistry::default();
        World {
            b: MemBackend::with(limits, registry),
            ..World::default()
        }
    }
}
