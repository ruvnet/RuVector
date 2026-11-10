//! Automatic re-queueing of rejected ledger items:
//! - heavy-dependency rejections re-open when the candidate ships a lean
//!   release (size/dep bar in `weaver.toml [gate]`);
//! - rejections with `blocked_on: [subjects]` re-open once every blocker's
//!   latest ledger entry is `shipped` (any repo).
//!
//! The rejection stays in history; the queue sees a `queued` view of it.

use crate::config::Config;
use crate::ledger::Ledger;
use crate::model::{Graph, LedgerEntry, Outcome};

/// Does a rejection's reason say the candidate was too heavy to depend on?
pub fn is_heavy_rejection(e: &LedgerEntry) -> bool {
    e.outcome == Outcome::Rejected
        && e.reason
            .as_deref()
            .map(|r| {
                let r = r.to_ascii_lowercase();
                r.contains("heavy dep") || r.contains("dependency tree") || r.contains("heavy tree")
            })
            .unwrap_or(false)
}

/// The candidate's current release clears the size/dep gate and is newer
/// than the version that was rejected.
pub fn lean_release(graph: &Graph, cfg: &Config, e: &LedgerEntry) -> Option<String> {
    let p = graph.get(e.candidate.as_deref()?)?;
    let rejected_at = e
        .version
        .as_deref()
        .and_then(|v| v.rsplit('@').next())
        .unwrap_or("");
    let g = &cfg.policy.gate;
    let lean = p.runtime_deps <= g.lean_max_runtime_deps
        && p.unpacked_size > 0
        && p.unpacked_size <= g.lean_max_unpacked_bytes;
    (p.version != rejected_at && lean).then(|| {
        format!(
            "{}@{} clears the size/dep gate ({} runtime deps, {} bytes unpacked; rejected at {})",
            p.name,
            p.version,
            p.runtime_deps,
            p.unpacked_size,
            if rejected_at.is_empty() {
                "?"
            } else {
                rejected_at
            }
        )
    })
}

/// Ledger view where heavy-dependency rejections whose candidate shipped a
/// lean release are re-queued (the rejection itself stays in history).
pub fn requeue_lean(ledger: &Ledger, graph: &Graph, cfg: &Config) -> Ledger {
    let mut out = ledger.clone();
    let latest: Vec<&LedgerEntry> = ledger.latest_by_key().into_values().collect();
    for (i, e) in ledger.entries.iter().enumerate() {
        if !is_heavy_rejection(e) || !latest.iter().any(|l| std::ptr::eq(*l, e)) {
            continue;
        }
        if let Some(why) = lean_release(graph, cfg, e) {
            let o = &mut out.entries[i];
            o.outcome = Outcome::Queued;
            o.reason = Some(format!(
                "re-queued: {why}. Was: {}",
                e.reason.as_deref().unwrap_or("")
            ));
        }
    }
    out
}

/// Blockers of a rejection that are not yet shipped (by subject, any repo).
pub fn open_blockers<'a>(ledger: &Ledger, e: &'a LedgerEntry) -> Vec<&'a str> {
    e.blocked_on
        .iter()
        .map(String::as_str)
        .filter(|b| {
            let latest = ledger.entries.iter().rev().find(|x| {
                x.subject.as_deref() == Some(*b)
                    && !matches!(x.outcome, Outcome::Gated | Outcome::Override)
            });
            latest
                .map(|x| x.outcome != Outcome::Shipped)
                .unwrap_or(true)
        })
        .collect()
}

/// Ledger view with both automatic re-queue rules applied.
pub fn requeue(ledger: &Ledger, graph: &Graph, cfg: &Config) -> Ledger {
    let mut out = requeue_lean(ledger, graph, cfg);
    let latest: Vec<&LedgerEntry> = ledger.latest_by_key().into_values().collect();
    for (i, e) in ledger.entries.iter().enumerate() {
        if e.outcome != Outcome::Rejected
            || e.blocked_on.is_empty()
            || !latest.iter().any(|l| std::ptr::eq(*l, e))
        {
            continue;
        }
        if open_blockers(ledger, e).is_empty() {
            let o = &mut out.entries[i];
            o.outcome = Outcome::Queued;
            o.reason = Some(format!(
                "re-queued: every blocker shipped ({}). Was: {}",
                e.blocked_on.join(", "),
                e.reason.as_deref().unwrap_or("")
            ));
        }
    }
    out
}
