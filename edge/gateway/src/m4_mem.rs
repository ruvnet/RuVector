//! In-process `QuantShard` / `GraphStore` / `AnalyticsJob` DOs for the
//! native tests (the M4 part of `backend::mem::MemBackend`): each DO's
//! storage is a `MemSqlStore`, bodies are JSON-encoded exactly as the
//! Worker sends them, and pending alarms are recorded so a test can run
//! them ([`M4Mem::run_alarms`]) the way the runtime would.

use crate::graph_store::{self, GraphHost};
use crate::mincut_job;
use crate::quant_shard::{self, QuantHost};
use ruvector_edge_store::MemSqlStore;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};

/// Which class an alarm belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// `QuantShard`.
    Quant,
    /// `AnalyticsJob`.
    Job,
}

/// M4 DO storage and resident state.
#[derive(Default)]
pub struct M4Mem {
    /// `QuantShard` stores by DO name.
    pub quant_stores: RefCell<BTreeMap<String, MemSqlStore>>,
    /// The simulated isolate's quant shards.
    pub quant_host: RefCell<QuantHost>,
    /// `GraphStore` stores by DO name.
    pub graph_stores: RefCell<BTreeMap<String, MemSqlStore>>,
    /// The simulated isolate's graphs.
    pub graph_host: RefCell<GraphHost>,
    /// `AnalyticsJob` stores by DO name.
    pub job_stores: RefCell<BTreeMap<String, MemSqlStore>>,
    /// Armed alarms (due now).
    pub alarms: RefCell<BTreeSet<(Kind, String)>>,
    /// Job alarms armed for later (retention), by DO name → due ms; moved
    /// into `alarms` by [`M4Mem::advance`].
    pub later: RefCell<BTreeMap<String, u64>>,
    /// Clock for job alarms (ms).
    pub now_ms: Cell<u64>,
}

impl M4Mem {
    /// Drop every resident state (isolate restart), keep storage.
    pub fn restart(&self) {
        let budget = self.quant_host.borrow().budget;
        *self.quant_host.borrow_mut() = QuantHost::default();
        self.quant_host.borrow_mut().budget = budget;
        *self.graph_host.borrow_mut() = GraphHost::default();
    }

    /// One `QuantShard` call (wipe included).
    pub fn quant(&self, name: &str, body: &[u8]) -> String {
        let mut stores = self.quant_stores.borrow_mut();
        let store = stores.entry(name.to_string()).or_default();
        let mut host = self.quant_host.borrow_mut();
        let (out, _wiped) = quant_shard::serve(&mut host, name, Some(name), &*store, body);
        if quant_shard::next_alarm(&host, name).is_some() {
            self.alarms
                .borrow_mut()
                .insert((Kind::Quant, name.to_string()));
        }
        out
    }

    /// One `GraphStore` call.
    pub fn graph(&self, name: &str, body: &[u8]) -> String {
        let mut stores = self.graph_stores.borrow_mut();
        let store = stores.entry(name.to_string()).or_default();
        graph_store::serve(
            &mut self.graph_host.borrow_mut(),
            name,
            Some(name),
            &*store,
            body,
        )
    }

    /// One `AnalyticsJob` call.
    pub fn job(&self, name: &str, body: &[u8]) -> String {
        let mut stores = self.job_stores.borrow_mut();
        let store = stores.entry(name.to_string()).or_default();
        let (out, alarm) = mincut_job::serve(Some(name), &*store, body);
        if let Some(ms) = alarm {
            self.arm_job(name, ms);
        }
        out
    }

    /// Arm a job alarm: due now when urgent, else at `now + ms`.
    fn arm_job(&self, name: &str, ms: u64) {
        if ms <= crate::shard_core::URGENT_ALARM_MS {
            self.alarms
                .borrow_mut()
                .insert((Kind::Job, name.to_string()));
        } else {
            let due = self.now().saturating_add(ms);
            self.later.borrow_mut().insert(name.to_string(), due);
        }
    }

    fn now(&self) -> u64 {
        self.now_ms.get().max(crate::testkit::T0 * 1000)
    }

    /// Move the clock `ms` forward; job alarms now due become armed.
    pub fn advance(&self, ms: u64) {
        let now = self.now().saturating_add(ms);
        self.now_ms.set(now);
        let mut later = self.later.borrow_mut();
        let due: Vec<String> = later
            .iter()
            .filter(|(_, at)| **at <= now)
            .map(|(n, _)| n.clone())
            .collect();
        for n in due {
            later.remove(&n);
            self.alarms.borrow_mut().insert((Kind::Job, n));
        }
    }

    /// Run every armed alarm once (re-arming those that ask); the number
    /// run.
    pub fn run_alarms(&self) -> usize {
        let due: Vec<(Kind, String)> = std::mem::take(&mut *self.alarms.borrow_mut())
            .into_iter()
            .collect();
        for (kind, name) in &due {
            let next = match kind {
                Kind::Quant => {
                    let stores = self.quant_stores.borrow();
                    let Some(store) = stores.get(name) else {
                        continue;
                    };
                    quant_shard::alarm(&mut self.quant_host.borrow_mut(), name, store)
                }
                Kind::Job => {
                    let stores = self.job_stores.borrow();
                    let Some(store) = stores.get(name) else {
                        continue;
                    };
                    self.now_ms.set(self.now() + 1);
                    let next = mincut_job::alarm(store, self.now_ms.get());
                    drop(stores);
                    if let Some(ms) = next {
                        self.arm_job(name, ms);
                    }
                    continue;
                }
            };
            if next.is_some() {
                self.alarms.borrow_mut().insert((*kind, name.clone()));
            }
        }
        due.len()
    }

    /// Run alarms until none is armed (at most `max` rounds).
    pub fn drain_alarms(&self, max: usize) {
        for _ in 0..max {
            if self.run_alarms() == 0 {
                return;
            }
        }
    }
}
