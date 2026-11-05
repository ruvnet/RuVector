//! Stub / TODO / placeholder / simulated / fallback / brute-force markers
//! that sit near a domain keyword. Each hit is mapped through the
//! capability table to the package that already implements the real thing.

use crate::config::Capabilities;
use crate::model::{Signal, SignalKind};
use crate::util::clip;
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Lines either side of a marker searched for a domain keyword.
pub const WINDOW: usize = 3;

pub fn marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(TODO|FIXME|XXX|stub(bed)?|placeholder|simulated?|simulation only|mock implementation|not (yet )?implemented|fallback|brute[- ]?force|naive (implementation|approach|scan)|for now)\b",
        )
        .expect("marker regex")
    })
}

/// Paths that are tests, fixtures, docs or examples: markers there are noise.
pub fn is_test_or_doc(rel: &str) -> bool {
    let l = rel.to_ascii_lowercase();
    l.contains("/test/")
        || l.contains("/tests/")
        || l.contains("__tests__")
        || l.contains("__mocks__")
        || l.contains(".test.")
        || l.contains(".spec.")
        || l.contains("/fixtures/")
        || l.contains("/bench")
        || l.starts_with("test/")
        || l.starts_with("tests/")
        || l.starts_with("docs/")
        || l.contains("/docs/")
}

/// Scan one file. Hits are folded per (file, candidate): one signal with an
/// occurrence count, evidence = the first marker line.
pub fn scan_file(repo: &str, rel: &str, src: &str, caps: &Capabilities) -> Vec<Signal> {
    if is_test_or_doc(rel) {
        return vec![];
    }
    let lines: Vec<&str> = src.lines().collect();
    let mut folded: BTreeMap<String, Signal> = BTreeMap::new();
    for (i, line) in lines.iter().enumerate() {
        if !marker_re().is_match(line) {
            continue;
        }
        let lo = i.saturating_sub(WINDOW);
        let hi = (i + WINDOW + 1).min(lines.len());
        let window = lines[lo..hi].join("\n");
        let Some(hit) = caps.best_match(&window) else { continue };
        let cand = hit.capability.package.clone();
        folded
            .entry(cand.clone())
            .and_modify(|s| s.occurrences += 1)
            .or_insert_with(|| Signal {
                kind: SignalKind::StubMarker,
                repo: repo.to_string(),
                file: rel.to_string(),
                line: i + 1,
                evidence: clip(line.trim(), 200),
                subject: None,
                keyword: Some(hit.keyword.clone()),
                candidate: Some(cand),
                confidence: 0.6 * hit.capability.weight,
                severity: None,
                occurrences: 1,
            });
    }
    folded.into_values().collect()
}
