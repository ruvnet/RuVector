//! Append-only outcome ledger (JSONL). The ledger is the loop's memory:
//! a (repo, candidate|subject) pair that was shipped, rejected, queued or is
//! in flight is never re-proposed, and per-signal-type hit rates reweight the
//! queue. It is also the ONLY source of numeric claims for READMEs.

use crate::model::{LedgerEntry, Measurement, Outcome, SignalKind, WorkKind};
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Default)]
pub struct Ledger {
    pub entries: Vec<LedgerEntry>,
}

/// Lane state for a pair, derived from its latest non-`gated` entry.
#[derive(Debug, Clone, PartialEq)]
pub enum PairState {
    Open,
    Blocked { outcome: Outcome, reason: String },
}

impl Ledger {
    pub fn parse(text: &str) -> Result<Self> {
        let mut entries = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let e: LedgerEntry = serde_json::from_str(line)
                .with_context(|| format!("ledger line {}: invalid entry", i + 1))?;
            entries.push(e);
        }
        Ok(Self { entries })
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        Self::parse(&std::fs::read_to_string(path)?)
    }

    /// Append one entry durably (never rewrites history).
    pub fn append(path: &Path, entry: &LedgerEntry) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening ledger {}", path.display()))?;
        writeln!(f, "{}", serde_json::to_string(entry)?)?;
        f.sync_all()?;
        Ok(())
    }

    /// Pair identity: same repo, and either the same observed subject, or
    /// the same target candidate doing the same work: same work kind when
    /// both sides name a subject, otherwise the same signal type (so an
    /// unrelated finding that maps to a popular candidate is not swallowed
    /// by a ledger entry about something else).
    pub fn same_pair(
        e: &LedgerEntry,
        repo: &str,
        candidate: Option<&str>,
        subject: Option<&str>,
        kind: WorkKind,
        signal: Option<SignalKind>,
    ) -> bool {
        if !e.repo.eq_ignore_ascii_case(repo) {
            return false;
        }
        if subject.is_some() && e.subject.as_deref() == subject {
            return true;
        }
        if candidate.is_none() || e.candidate.as_deref() != candidate {
            return false;
        }
        if subject.is_some() && e.subject.is_some() {
            return e.kind == kind;
        }
        match (e.signal_kind, signal) {
            (Some(a), Some(b)) => a == b,
            _ => e.kind == kind,
        }
    }

    /// Should a (repo, candidate, subject, kind) item be suppressed?
    pub fn pair_state(
        &self,
        repo: &str,
        candidate: Option<&str>,
        subject: Option<&str>,
        kind: WorkKind,
        signal: Option<SignalKind>,
        max_failures: usize,
    ) -> PairState {
        let related: Vec<&LedgerEntry> = self
            .entries
            .iter()
            .filter(|e| !matches!(e.outcome, Outcome::Gated | Outcome::Override))
            .filter(|e| Self::same_pair(e, repo, candidate, subject, kind, signal))
            .collect();
        let Some(last) = related.last() else {
            return PairState::Open;
        };
        let reason = last.reason.clone().unwrap_or_default();
        match last.outcome {
            Outcome::Shipped | Outcome::Rejected | Outcome::InProgress | Outcome::Queued => {
                PairState::Blocked {
                    outcome: last.outcome,
                    reason,
                }
            }
            Outcome::Failed => {
                let failures = related
                    .iter()
                    .filter(|e| e.outcome == Outcome::Failed)
                    .count();
                if failures >= max_failures {
                    PairState::Blocked {
                        outcome: Outcome::Failed,
                        reason: format!("failed {failures}x: {reason}"),
                    }
                } else {
                    PairState::Open
                }
            }
            Outcome::Gated | Outcome::Override => PairState::Open,
        }
    }

    /// Latest non-gated entry per (repo, subject|candidate) key.
    pub fn latest_by_key(&self) -> BTreeMap<(String, String), &LedgerEntry> {
        let mut latest = BTreeMap::new();
        for e in self
            .entries
            .iter()
            .filter(|e| !matches!(e.outcome, Outcome::Gated | Outcome::Override))
        {
            let key = (
                e.repo.to_ascii_lowercase(),
                e.subject
                    .clone()
                    .or(e.candidate.clone())
                    .unwrap_or_default(),
            );
            latest.insert(key, e);
        }
        latest
    }

    /// Repos with an in-flight lane (their single lane is taken).
    pub fn busy_repos(&self) -> BTreeMap<String, Vec<String>> {
        let mut busy: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for ((repo, name), e) in self.latest_by_key() {
            if e.outcome == Outcome::InProgress {
                busy.entry(repo).or_default().push(name);
            }
        }
        busy
    }

    /// Backlog entries whose latest state is `queued`.
    pub fn queued(&self) -> Vec<&LedgerEntry> {
        self.latest_by_key()
            .into_values()
            .filter(|e| e.outcome == Outcome::Queued)
            .collect()
    }

    /// Laplace-smoothed hit rate per signal type:
    /// (shipped + 1) / (shipped + rejected + failed + 2).
    pub fn hit_rates(&self) -> BTreeMap<SignalKind, f64> {
        let mut out = BTreeMap::new();
        for kind in SignalKind::ALL {
            let (mut s, mut n) = (0u32, 0u32);
            for e in self.entries.iter().filter(|e| e.signal_kind == Some(kind)) {
                match e.outcome {
                    Outcome::Shipped => {
                        s += 1;
                        n += 1;
                    }
                    Outcome::Rejected | Outcome::Failed => n += 1,
                    _ => {}
                }
            }
            out.insert(kind, (f64::from(s) + 1.0) / (f64::from(n) + 2.0));
        }
        out
    }

    /// New npm names created (shipped `create` entries) since `since`.
    pub fn new_packages_since(&self, since: chrono::DateTime<chrono::Utc>) -> usize {
        self.entries
            .iter()
            .filter(|e| e.kind == WorkKind::Create && e.outcome == Outcome::Shipped)
            .filter(|e| {
                chrono::DateTime::parse_from_rfc3339(&e.ts)
                    .map(|t| t.with_timezone(&chrono::Utc) >= since)
                    .unwrap_or(false)
            })
            .count()
    }

    /// Ledger measurements for a package (the only legal README numbers).
    pub fn measurements_for(&self, package: &str) -> Vec<(&LedgerEntry, &Measurement)> {
        self.entries
            .iter()
            .filter(|e| e.outcome == Outcome::Shipped && e.candidate.as_deref() == Some(package))
            .flat_map(|e| e.measurements.iter().map(move |m| (e, m)))
            .collect()
    }
}

/// Parse `metric=value:unit[@baseline]` (e.g. `recall_at_10=0.97:ratio@0.91`).
pub fn parse_measurement(s: &str) -> Result<Measurement> {
    let (metric, rest) = s
        .split_once('=')
        .context("measurement must be metric=value:unit[@baseline]")?;
    let (value, unit_base) = rest.split_once(':').context("measurement needs :unit")?;
    let (unit, baseline) = match unit_base.split_once('@') {
        Some((u, b)) => (
            u,
            Some(b.parse::<f64>().context("baseline must be a number")?),
        ),
        None => (unit_base, None),
    };
    Ok(Measurement {
        metric: metric.to_string(),
        value: value.parse().context("value must be a number")?,
        unit: unit.to_string(),
        baseline,
    })
}
