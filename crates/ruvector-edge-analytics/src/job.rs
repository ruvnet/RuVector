//! Async min-cut jobs (`/v1/mincut/jobs*`, ADR-351 §3 `AnalyticsJob`).
//!
//! A descriptor is pinned to one graph uid, revision and snapshot digest,
//! admitted against [`Profile::JOB`] at submit time (413 if even a job
//! cannot afford it), and moves `queued -> running -> done | failed`. It is
//! plain serde data, so the owning Durable Object stores it as a row and
//! drives `start`/`run` from an alarm. A `running` job whose isolate was
//! evicted may be restarted up to [`MAX_ATTEMPTS`] times.

use crate::cost::CostEstimate;
use crate::error::{AnalyticsError, Result};
use crate::graph::TenantGraph;
use crate::service::{plan, query, CutReport, Profile, QueryMode};
use ruvector_edge_store::ErrorCode;
use serde::{Deserialize, Serialize};

/// Restarts allowed for a job left `running` by an evicted isolate.
pub const MAX_ATTEMPTS: u32 = 3;

/// Job lifecycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JobState {
    /// Admitted, not started.
    Queued,
    /// Executing (attempt number, 1-based).
    Running {
        /// Attempt number.
        attempt: u32,
    },
    /// Finished with a report.
    Done {
        /// The answer.
        report: CutReport,
    },
    /// Finished with a typed error.
    Failed {
        /// Stable code.
        code: ErrorCode,
        /// Static detail.
        detail: String,
    },
}

/// A persisted job descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobDescriptor {
    /// Caller-visible id (`[A-Za-z0-9_-]{1,64}`).
    pub job_id: String,
    /// Graph identity (hex).
    pub graph_uid: String,
    /// Graph revision the job must run on.
    pub revision: u64,
    /// Manifest digest (hex) of the snapshot the job must run on.
    pub snapshot_digest: String,
    /// Query.
    pub mode: QueryMode,
    /// Estimate admitted at submit.
    pub estimate: CostEstimate,
    /// Lifecycle state.
    pub state: JobState,
    /// Submit time (ms since epoch, from the caller's clock port).
    pub submitted_at_ms: u64,
    /// Last transition time.
    pub updated_at_ms: u64,
}

fn valid_job_id(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

impl JobDescriptor {
    /// Admit a job against the job profile. The graph must come from
    /// [`crate::decode_graph`]: the job is pinned to the digest of the
    /// snapshot it was decoded from, never to a caller-supplied string.
    pub fn submit(job_id: &str, graph: &TenantGraph, mode: QueryMode, now_ms: u64) -> Result<Self> {
        if !valid_job_id(job_id) {
            return Err(AnalyticsError::Invalid(
                "job_id must match [A-Za-z0-9_-]{1,64}",
            ));
        }
        let estimate = plan(graph, &mode, &Profile::JOB)?;
        let Some(digest) = graph.snapshot_digest() else {
            return Err(AnalyticsError::JobState(
                "job graph must be decoded from a persisted snapshot",
            ));
        };
        Ok(JobDescriptor {
            job_id: job_id.to_owned(),
            graph_uid: hex::encode(graph.uid()),
            revision: graph.revision(),
            snapshot_digest: hex::encode(digest),
            mode,
            estimate,
            state: JobState::Queued,
            submitted_at_ms: now_ms,
            updated_at_ms: now_ms,
        })
    }

    /// `true` for `done` and `failed`.
    pub fn is_terminal(&self) -> bool {
        matches!(self.state, JobState::Done { .. } | JobState::Failed { .. })
    }

    /// `queued -> running`, or restart a `running` job after eviction.
    pub fn start(&mut self, now_ms: u64) -> Result<()> {
        let attempt = match self.state {
            JobState::Queued => 1,
            JobState::Running { attempt } if attempt < MAX_ATTEMPTS => attempt + 1,
            JobState::Running { .. } => {
                self.fail(
                    &AnalyticsError::JobState("job exceeded its restart attempts"),
                    now_ms,
                );
                return Err(AnalyticsError::JobState(
                    "job exceeded its restart attempts",
                ));
            }
            _ => return Err(AnalyticsError::JobState("job already finished")),
        };
        self.state = JobState::Running { attempt };
        self.updated_at_ms = now_ms;
        Ok(())
    }

    /// Execute a `running` job on the graph decoded from its snapshot. The
    /// graph must match the pinned uid, revision and the digest recorded by
    /// [`crate::decode_graph`]; the outcome (report or typed error) is
    /// recorded in the descriptor.
    pub fn run(&mut self, graph: &TenantGraph, now_ms: u64) -> Result<()> {
        if !matches!(self.state, JobState::Running { .. }) {
            return Err(AnalyticsError::JobState("job is not running"));
        }
        let outcome = if hex::encode(graph.uid()) != self.graph_uid
            || graph.revision() != self.revision
            || graph.snapshot_digest().map(hex::encode).as_deref()
                != Some(self.snapshot_digest.as_str())
        {
            Err(AnalyticsError::JobState(
                "graph changed since the job was submitted",
            ))
        } else {
            query(graph, &self.mode, &Profile::JOB)
        };
        match outcome {
            Ok(report) => {
                self.state = JobState::Done { report };
                self.updated_at_ms = now_ms;
            }
            Err(e) => self.fail(&e, now_ms),
        }
        Ok(())
    }

    /// Record a failure (terminal).
    pub fn fail(&mut self, e: &AnalyticsError, now_ms: u64) {
        self.state = JobState::Failed {
            code: e.code(),
            detail: e.detail().to_owned(),
        };
        self.updated_at_ms = now_ms;
    }
}

/// Where a query should run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Route {
    /// Answer on the request path.
    Inline(CostEstimate),
    /// Too large inline; submit to `/v1/mincut/jobs`.
    Job(CostEstimate),
}

/// Route a query: inline if the inline profile admits it, else a job if
/// the job profile does, else the inline error (413 for size and budget).
pub fn route(graph: &TenantGraph, mode: &QueryMode) -> Result<Route> {
    match plan(graph, mode, &Profile::INLINE) {
        Ok(est) => Ok(Route::Inline(est)),
        Err(e @ AnalyticsError::Invalid(_)) => Err(e),
        Err(inline_err) => match plan(graph, mode, &Profile::JOB) {
            Ok(est) => Ok(Route::Job(est)),
            Err(_) => Err(inline_err),
        },
    }
}
