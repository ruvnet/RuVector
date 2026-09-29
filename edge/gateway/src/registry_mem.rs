//! Native test doubles for the registry ports: [`MemRegistry`] runs the
//! real DO cores over `MemSqlStore` (every call JSON-encoded across the DO
//! boundary as in production) and [`MemR2`] is an R2 bucket with
//! multipart uploads, SHA-256-checked puts and fault hooks.

use crate::registry_core::{serve_root, serve_scope};
use crate::registry_ports::{apply_sweep, BlobStore, RegistryRpc};
use crate::registry_sweep::SweepWork;
use crate::registry_wire::RvfError;
use crate::rvf_finalize_step::{serve_step, Jobs, Step, STEP_BYTES};
use crate::testkit::block_on;
use ruvector_edge_registry::keys::registry_do_name;
use ruvector_edge_registry::mem::{CounterEntropy, FixedClock};
use ruvector_edge_registry::registry::RegistryConfig;
use ruvector_edge_registry::upload::{ObjectEvidence, UploadLimits};
use ruvector_edge_registry::Scope;
use ruvector_edge_store::MemSqlStore;
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

/// Test configuration: parts of any size (R2's 5 MiB minimum off).
pub fn test_config() -> RegistryConfig {
    RegistryConfig {
        upload: UploadLimits {
            min_part_size: 1,
            max_part_size: 1 << 20,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Both registry DO classes, in process.
pub struct MemRegistry {
    /// `RegistryRoot` storage.
    pub root: MemSqlStore,
    /// `RegistryScope` storage by DO name.
    pub scopes: RefCell<BTreeMap<String, MemSqlStore>>,
    /// Shared clock.
    pub clock: FixedClock,
    entropy: CounterEntropy,
    cfg: RegistryConfig,
    /// Fail the scope call this many calls from now (0: the next one).
    pub fail_scope_call_in: Cell<Option<usize>>,
    /// The bucket the `RegistryScope` objects read in a stepped finalize.
    pub bucket: RefCell<Option<Rc<MemR2>>>,
    /// Each `RegistryScope`'s in-memory finalize jobs, by DO name.
    pub jobs: RefCell<BTreeMap<String, Rc<Jobs>>>,
    /// Byte budget of one finalize step.
    pub step_bytes: Cell<u64>,
    /// The sweep holds every object (a step must not touch the index).
    pub sweeping: Cell<bool>,
}

impl MemRegistry {
    /// At `now`.
    pub fn at(now: u64) -> Self {
        MemRegistry {
            root: MemSqlStore::default(),
            scopes: RefCell::new(BTreeMap::new()),
            clock: FixedClock::at(now),
            entropy: CounterEntropy::default(),
            cfg: test_config(),
            fail_scope_call_in: Cell::new(None),
            bucket: RefCell::new(None),
            jobs: RefCell::new(BTreeMap::new()),
            step_bytes: Cell::new(STEP_BYTES),
            sweeping: Cell::new(false),
        }
    }

    /// `do_name`'s finalize jobs.
    pub fn scope_jobs(&self, do_name: &str) -> Rc<Jobs> {
        self.jobs
            .borrow_mut()
            .entry(do_name.to_string())
            .or_default()
            .clone()
    }

    /// The configuration its DOs run with.
    pub fn cfg(&self) -> RegistryConfig {
        self.cfg
    }

    /// Run `scope`'s alarm against `r2`.
    pub fn alarm(&self, scope: &str, r2: &MemR2) -> SweepWork {
        self.alarm_while(scope, r2, &|| true)
    }

    /// Run `scope`'s alarm; it stops issuing R2 operations once `live`
    /// turns false (the gate deadline passed).
    pub fn alarm_while(&self, scope: &str, r2: &MemR2, live: &dyn Fn() -> bool) -> SweepWork {
        let name = registry_do_name(&Scope::parse(scope).unwrap());
        let mut all = self.scopes.borrow_mut();
        let store = all.entry(name).or_default();
        block_on(apply_sweep(
            &*store,
            &self.clock,
            &self.entropy,
            self.cfg,
            r2,
            live,
        ))
        .unwrap()
    }
}

impl RegistryRpc for MemRegistry {
    async fn call_root(&self, body: String) -> Result<String, RvfError> {
        Ok(serve_root(&self.root, &self.clock, body.as_bytes()))
    }

    async fn call_scope(&self, do_name: &str, body: String) -> Result<String, RvfError> {
        match self.fail_scope_call_in.get() {
            Some(0) => {
                self.fail_scope_call_in.set(None);
                return Err(RvfError::unavailable());
            }
            Some(n) => self.fail_scope_call_in.set(Some(n - 1)),
            None => {}
        }
        let mut all = self.scopes.borrow_mut();
        let store = all.entry(do_name.to_string()).or_default();
        let served = serve_scope(
            &*store,
            &self.clock,
            &self.entropy,
            self.cfg,
            body.as_bytes(),
        );
        Ok(served.body)
    }

    async fn call_scope_step(&self, do_name: &str, body: String) -> Result<String, RvfError> {
        let r2 = self
            .bucket
            .borrow()
            .clone()
            .ok_or_else(RvfError::unavailable)?;
        let jobs = self.scope_jobs(do_name);
        // The object's storage leaves the map while the step awaits R2 (no
        // borrow is held across an await); nothing else runs meanwhile.
        let store = self.scopes.borrow_mut().remove(do_name).unwrap_or_default();
        let open = || !self.sweeping.get();
        let st = Step {
            sql: &store,
            clock: &self.clock,
            entropy: &self.entropy,
            cfg: self.cfg,
            r2: &*r2,
            step_bytes: self.step_bytes.get(),
            open: &open,
        };
        let out = serve_step(&jobs, &st, body.as_bytes()).await;
        self.scopes.borrow_mut().insert(do_name.to_string(), store);
        Ok(out.body)
    }
}

/// One multipart upload.
#[derive(Default)]
struct Multipart {
    key: String,
    parts: BTreeMap<u16, (String, Vec<u8>)>,
}

/// In-memory R2.
#[derive(Default)]
pub struct MemR2 {
    /// Objects.
    pub objects: RefCell<HashMap<String, Vec<u8>>>,
    uploads: RefCell<HashMap<String, Multipart>>,
    next: Cell<u64>,
    /// Multipart uploads aborted, by upload id.
    pub aborted: RefCell<Vec<String>>,
    /// Flip one byte of any object completed under this key prefix
    /// (staging bytes changing after the freeze).
    pub tamper_prefix: RefCell<Option<String>>,
    /// Store different bytes than asked for under this key prefix (a
    /// broken write the Worker must detect by measuring).
    pub corrupt_prefix: RefCell<Option<String>>,
    /// Refuse every write of a blob (`rvf/…`) key: a transient R2 outage.
    pub fail_blob_writes: Cell<bool>,
    /// Refuse every delete and abort: an R2 outage during a sweep.
    pub fail_sweep_ops: Cell<bool>,
    /// Delete and abort calls received.
    pub sweep_ops: Cell<usize>,
    /// Every `mp_part` call: (key, R2 part number).
    pub part_calls: RefCell<Vec<(String, u16)>>,
    /// `get_range` and `measure` calls received.
    pub reads: Cell<(usize, usize)>,
}

fn fail() -> RvfError {
    RvfError::storage()
}

impl MemR2 {
    /// An object's bytes.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.objects.borrow().get(key).cloned()
    }

    /// Put an object directly (fixtures).
    pub fn put(&self, key: &str, bytes: Vec<u8>) {
        self.objects.borrow_mut().insert(key.to_string(), bytes);
    }

    /// Open multipart uploads.
    pub fn open_uploads(&self) -> usize {
        self.uploads.borrow().len()
    }

    fn store(&self, key: &str, mut bytes: Vec<u8>) {
        let hit = |p: &RefCell<Option<String>>| {
            p.borrow()
                .as_ref()
                .is_some_and(|p| key.starts_with(p.as_str()))
        };
        if (hit(&self.tamper_prefix) || hit(&self.corrupt_prefix)) && !bytes.is_empty() {
            let i = bytes.len() / 2;
            bytes[i] ^= 0x5a;
        }
        self.objects.borrow_mut().insert(key.to_string(), bytes);
    }
}

impl BlobStore for MemR2 {
    async fn mp_create(&self, key: &str) -> Result<String, RvfError> {
        let n = self.next.get() + 1;
        self.next.set(n);
        let id = format!("mp{n}");
        let m = Multipart {
            key: key.to_string(),
            ..Default::default()
        };
        self.uploads.borrow_mut().insert(id.clone(), m);
        Ok(id)
    }

    async fn mp_part(
        &self,
        key: &str,
        upload: &str,
        n: u16,
        bytes: Vec<u8>,
    ) -> Result<String, RvfError> {
        let mut ups = self.uploads.borrow_mut();
        let m = ups
            .get_mut(upload)
            .filter(|m| m.key == key)
            .ok_or_else(fail)?;
        self.part_calls.borrow_mut().push((key.to_string(), n));
        let etag = hex::encode(&Sha256::digest(&bytes)[..8]);
        m.parts.insert(n, (etag.clone(), bytes));
        Ok(etag)
    }

    async fn mp_complete(
        &self,
        key: &str,
        upload: &str,
        parts: &[(u16, String)],
    ) -> Result<(), RvfError> {
        if self.fail_blob_writes.get() && key.starts_with("rvf/") {
            return Err(fail());
        }
        let m = {
            let mut ups = self.uploads.borrow_mut();
            let ok = ups.get(upload).is_some_and(|m| {
                m.key == key
                    && parts
                        .iter()
                        .all(|(n, e)| m.parts.get(n).is_some_and(|(pe, _)| pe == e))
            });
            if !ok {
                return Err(fail());
            }
            ups.remove(upload).ok_or_else(fail)?
        };
        let mut bytes = Vec::new();
        for (n, _) in parts {
            bytes.extend_from_slice(&m.parts[n].1);
        }
        self.store(key, bytes);
        Ok(())
    }

    async fn mp_abort(&self, _key: &str, upload: &str) -> Result<(), RvfError> {
        self.sweep_ops.set(self.sweep_ops.get() + 1);
        if self.fail_sweep_ops.get() {
            return Err(fail());
        }
        if self.uploads.borrow_mut().remove(upload).is_some() {
            self.aborted.borrow_mut().push(upload.to_string());
        }
        Ok(())
    }

    async fn size(&self, key: &str) -> Result<Option<u64>, RvfError> {
        Ok(self.objects.borrow().get(key).map(|b| b.len() as u64))
    }

    async fn get_range(
        &self,
        key: &str,
        offset: u64,
        len: u64,
    ) -> Result<Option<Vec<u8>>, RvfError> {
        let (r, m) = self.reads.get();
        self.reads.set((r + 1, m));
        Ok(self.objects.borrow().get(key).map(|b| {
            let s = (offset as usize).min(b.len());
            let e = (offset.saturating_add(len) as usize).min(b.len());
            b[s..e].to_vec()
        }))
    }

    async fn put_checked(
        &self,
        key: &str,
        bytes: Vec<u8>,
        sha256: [u8; 32],
    ) -> Result<ObjectEvidence, RvfError> {
        if self.fail_blob_writes.get() && key.starts_with("rvf/") {
            return Err(fail());
        }
        if <[u8; 32]>::from(Sha256::digest(&bytes)) != sha256 {
            return Err(fail());
        }
        self.store(key, bytes);
        // What R2 reports is what it stored (a broken write shows here).
        let stored = self.get(key).unwrap_or_default();
        Ok(ObjectEvidence {
            sha256: Sha256::digest(&stored).into(),
            size: stored.len() as u64,
        })
    }

    async fn copy_create_only(
        &self,
        from: &str,
        to: &str,
        sha256: [u8; 32],
    ) -> Result<bool, RvfError> {
        if self.objects.borrow().contains_key(to) {
            return Ok(false);
        }
        let src = self.get(from).ok_or_else(fail)?;
        self.put_checked(to, src, sha256).await?;
        Ok(true)
    }

    async fn copy_checked(
        &self,
        from: &str,
        to: &str,
        sha256: [u8; 32],
    ) -> Result<ObjectEvidence, RvfError> {
        let src = self.get(from).ok_or_else(fail)?;
        self.put_checked(to, src, sha256).await
    }

    async fn measure(&self, key: &str) -> Result<Option<ObjectEvidence>, RvfError> {
        let (r, m) = self.reads.get();
        self.reads.set((r, m + 1));
        Ok(self.objects.borrow().get(key).map(|b| ObjectEvidence {
            sha256: Sha256::digest(b).into(),
            size: b.len() as u64,
        }))
    }

    async fn delete(&self, key: &str) -> Result<(), RvfError> {
        self.sweep_ops.set(self.sweep_ops.get() + 1);
        if self.fail_sweep_ops.get() {
            return Err(fail());
        }
        self.objects.borrow_mut().remove(key);
        Ok(())
    }
}
