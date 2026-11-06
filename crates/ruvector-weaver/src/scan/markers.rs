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

/// Strong markers: the code says outright it is not the real thing.
pub fn marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(TODO|FIXME|XXX|stub(bed)?|placeholder|simulated|simulation only|mock implementation|not (yet )?implemented|brute[- ]?force|naive (implementation|approach|scan))\b",
        )
        .expect("marker regex")
    })
}

/// Weak markers: a degraded path that may or may not be a stand-in.
pub fn weak_marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(fallback|for now|using js implementation)\b")
            .expect("weak marker regex")
    })
}

/// Paths that are tests, fixtures, docs, examples, simulations or scripts:
/// markers there are noise for integration work.
pub fn is_test_or_doc(rel: &str) -> bool {
    let l = format!("/{}", rel.to_ascii_lowercase());
    [
        "/test/",
        "/tests/",
        "__tests__",
        "__mocks__",
        ".test.",
        ".spec.",
        "/fixtures/",
        "/bench",
        "/docs/",
        "/examples/",
        "/example/",
        "/simulation/",
        "/scripts/",
        "/demo/",
    ]
    .iter()
    .any(|p| l.contains(p))
}

/// Test files only (used to count a new package's tests).
pub fn is_test(rel: &str) -> bool {
    let l = format!("/{}", rel.to_ascii_lowercase());
    ["/test/", "/tests/", "__tests__", ".test.", ".spec."]
        .iter()
        .any(|p| l.contains(p))
}

fn is_import_line(t: &str) -> bool {
    t.starts_with("import ") || t.starts_with("export * from") || t.contains("require(")
}

/// Scan one file. Hits are folded per (file, candidate): one signal with an
/// occurrence count, evidence = the strongest (then first) marker line.
pub fn scan_file(repo: &str, rel: &str, src: &str, caps: &Capabilities) -> Vec<Signal> {
    if is_test_or_doc(rel) {
        return vec![];
    }
    let lines: Vec<&str> = src.lines().collect();
    let mut folded: BTreeMap<String, Signal> = BTreeMap::new();
    for (i, line) in lines.iter().enumerate() {
        let strength = if marker_re().is_match(line) {
            1.0
        } else if weak_marker_re().is_match(line) {
            0.6
        } else {
            continue;
        };
        if is_import_line(line.trim()) {
            continue;
        }
        let lo = i.saturating_sub(WINDOW);
        let hi = (i + WINDOW + 1).min(lines.len());
        let window = lines[lo..hi].join("\n");
        let Some(hit) = caps.best_match(&window) else {
            continue;
        };
        let cand = hit.capability.package.clone();
        let confidence = 0.6 * hit.capability.weight * strength;
        let fresh = Signal {
            kind: SignalKind::StubMarker,
            repo: repo.to_string(),
            file: rel.to_string(),
            line: i + 1,
            evidence: clip(line.trim(), 200),
            subject: None,
            keyword: Some(hit.keyword.clone()),
            candidate: Some(cand.clone()),
            confidence,
            severity: None,
            occurrences: 1,
        };
        folded
            .entry(cand)
            .and_modify(|s| {
                s.occurrences += 1;
                if confidence > s.confidence {
                    let occ = s.occurrences;
                    *s = fresh.clone();
                    s.occurrences = occ;
                }
            })
            .or_insert(fresh);
    }
    folded.into_values().collect()
}
