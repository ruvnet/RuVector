//! CPU-mode throughput model: a deny-list plus a per-family, per-thread speed
//! factor derived from the offer's `cpu_name`.
//!
//! Why: `cpu_cores_effective` alone ranked a Xeon Phi 7210 (256 slow in-order
//! Atom-class threads, 1.3 GHz) first, because $/thread was lowest. The job is
//! a dense f32 GEMM (1-N ComplEx scoring), so per-thread throughput differs by
//! 3-4x across the families vast.ai offers.
//!
//! Model: `effective throughput = cpu_cores_effective x factor`, where the
//! factor is the relative per-hardware-thread GEMM throughput of the family,
//! normalised to AMD EPYC Zen 2 (7xx2, "Rome") = 1.0. The factors are coarse
//! (IPC x typical all-core clock x SIMD width, halved for SMT siblings in the
//! same way for every family) and only need to be right in *order*: they pick
//! between offers, they never gate spend. CPU mode ranks by
//! `planning $/h / throughput` and rejects a family below `--min-cpu-speed`.
//!
//! `cpu_ghz` is deliberately ignored: live offers report e.g. 7.03 GHz for a
//! Threadripper PRO 5995WX and 1.49 GHz for a Xeon Silver 4114, so it is not a
//! usable clock. An unrecognised name gets [`UNKNOWN_FACTOR`] (a penalty, never
//! a bonus).

use serde::Serialize;

/// Factor for a CPU name that matches no family below.
pub const UNKNOWN_FACTOR: f64 = 0.6;
/// Default `--min-cpu-speed`: rejects pre-AVX2 Xeons (E5/E7 v1/v2, Westmere).
pub const DEFAULT_MIN_CPU_SPEED: f64 = 0.5;

/// Substrings (of the normalised, lower-case name) that are refused outright,
/// whatever the price: many slow threads that look cheap per thread.
pub const DENY: [(&str, &str); 7] = [
    ("xeon phi", "Xeon Phi (many-core, low single-thread)"),
    ("atom", "Atom (low single-thread)"),
    ("celeron", "Celeron (low single-thread)"),
    ("pentium", "Pentium (low single-thread)"),
    ("opteron", "Opteron (pre-Zen)"),
    ("core2", "Core 2 (pre-AVX)"),
    ("core 2", "Core 2 (pre-AVX)"),
];

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CpuPerf {
    /// Human-readable family the name matched (or "unknown").
    pub family: String,
    /// Relative per-thread throughput (EPYC Zen 2 = 1.0).
    pub factor: f64,
    /// Deny-list reason, when refused outright.
    pub denied: Option<String>,
}

/// Lower-case, strip (R)/(TM) marks and collapse whitespace.
fn normalise(name: &str) -> String {
    name.replace(['®', '™'], " ")
        .replace("(r)", " ")
        .replace("(tm)", " ")
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// First token after `key` (e.g. the model number after "epyc ").
fn token_after<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let i = s.find(key)? + key.len();
    s[i..].split_whitespace().next()
}

/// Leading ASCII digits of `t` (`"7702p"` -> `"7702"`, `"i9-14900k"` n/a).
fn digits(t: &str) -> &str {
    let end = t
        .char_indices()
        .find(|(_, c)| !c.is_ascii_digit())
        .map_or(t.len(), |(i, _)| i);
    &t[..end]
}

fn known(family: &str, factor: f64) -> CpuPerf {
    CpuPerf {
        family: family.into(),
        factor,
        denied: None,
    }
}

/// EPYC model tokens are 4 characters, e.g. `7702`, `7b13`, `7k62`, `9j14`,
/// `9654`, `4564p`: first digit = series, last digit = generation.
fn epyc(t: &str) -> Option<CpuPerf> {
    let b = t.as_bytes();
    if b.len() < 4 || !b[0].is_ascii_digit() || !b[3].is_ascii_digit() {
        return None;
    }
    Some(match (b[0], b[3]) {
        (b'7', b'1') => known("EPYC Zen1 (7xx1)", 0.75),
        (b'7', b'2') => known("EPYC Zen2 (7xx2)", 1.0),
        (b'7', b'3') => known("EPYC Zen3 (7xx3)", 1.2),
        (b'9', b'4') => known("EPYC Zen4 (9xx4)", 1.5),
        (b'9', b'5') => known("EPYC Zen5 (9xx5)", 1.7),
        (b'8', b'4') => known("EPYC Zen4c (8xx4)", 1.3),
        (b'4', b'4') => known("EPYC Zen4 AM5 (4xx4)", 1.3),
        (b'4', b'5') => known("EPYC Zen5 AM5 (4xx5)", 1.45),
        _ => return None,
    })
}

/// Ryzen / Threadripper: the first digit of the model is the generation.
fn ryzen(model: &str, threadripper: bool) -> Option<CpuPerf> {
    let d = digits(model);
    if d.len() != 4 {
        return None;
    }
    let (fam, f) = match (threadripper, d.as_bytes()[0]) {
        (true, b'1' | b'2') => ("Threadripper Zen1/Zen+", 0.8),
        (true, b'3') => ("Threadripper Zen2 (3xxx)", 1.1),
        (true, b'5') => ("Threadripper Zen3 (5xxx)", 1.3),
        (true, b'7') => ("Threadripper Zen4 (7xxx)", 1.6),
        (true, b'9') => ("Threadripper Zen5 (9xxx)", 1.8),
        (false, b'1' | b'2') => ("Ryzen Zen1/Zen+", 0.75),
        (false, b'3') => ("Ryzen Zen2 (3xxx)", 1.0),
        (false, b'5') => ("Ryzen Zen3 (5xxx)", 1.2),
        (false, b'7') => ("Ryzen Zen4 (7xxx)", 1.45),
        (false, b'9') => ("Ryzen Zen5 (9xxx)", 1.6),
        _ => return None,
    };
    Some(known(fam, f))
}

/// Xeon: E5/E7 (by `vN` suffix), Scalable (metal tier + 2nd digit = gen),
/// W, and pre-Sandy-Bridge `X5xxx`-style parts.
fn xeon(s: &str) -> Option<CpuPerf> {
    for tier in ["platinum", "gold", "silver", "bronze"] {
        if let Some(t) = token_after(s, &format!("{tier} ")) {
            let d = digits(t);
            if d.len() != 4 {
                return None;
            }
            let (gen, f) = match d.as_bytes()[1] {
                b'1' => ("Skylake-SP", 0.85),
                b'2' => ("Cascade Lake", 0.9),
                b'3' => ("Ice Lake-SP", 1.0),
                b'4' => ("Sapphire Rapids", 1.15),
                b'5' => ("Emerald Rapids", 1.2),
                b'6' => ("Granite Rapids", 1.3),
                _ => return None,
            };
            // Silver/Bronze: one AVX-512 FMA port and low all-core clocks.
            let f = if matches!(tier, "silver" | "bronze") {
                f * 0.85
            } else {
                f
            };
            return Some(known(&format!("Xeon Scalable {gen} ({tier})"), f));
        }
    }
    for series in ["e5-", "e7-", "e3-"] {
        if s.contains(series) {
            let (fam, f) = if s.ends_with(" v4") || s.contains(" v4 ") {
                ("Broadwell", 0.7)
            } else if s.ends_with(" v3") || s.contains(" v3 ") {
                ("Haswell", 0.65)
            } else if s.ends_with(" v2") || s.contains(" v2 ") {
                ("Ivy Bridge, no AVX2/FMA", 0.45)
            } else if s.ends_with(" v5") || s.ends_with(" v6") {
                ("Skylake/Kaby Lake E3", 0.75)
            } else {
                ("Sandy Bridge, no AVX2/FMA", 0.4)
            };
            return Some(known(
                &format!("Xeon {} {fam}", series.trim_end_matches('-').to_uppercase()),
                f,
            ));
        }
    }
    if let Some(t) = token_after(s, "xeon ") {
        if t.starts_with("w-")
            || t.starts_with("w3-")
            || t.starts_with("w5-")
            || t.starts_with("w7-")
            || t.starts_with("w9-")
        {
            let f = if t.starts_with("w-") { 0.9 } else { 1.15 };
            return Some(known("Xeon W", f));
        }
        let b = t.as_bytes();
        if b.len() == 5
            && matches!(b[0], b'x' | b'e' | b'l' | b'w')
            && t[1..].bytes().all(|c| c.is_ascii_digit())
        {
            return Some(known("Xeon pre-Sandy-Bridge (Westmere/Nehalem)", 0.3));
        }
    }
    None
}

/// Intel Core: `i9-14900k` -> gen 14; `i7-8700` -> gen 8; `core ultra`.
fn core(s: &str) -> Option<CpuPerf> {
    if s.contains("core ultra") {
        return Some(known("Core Ultra", 1.1));
    }
    let i = ["i9-", "i7-", "i5-", "i3-"]
        .iter()
        .find_map(|k| s.find(k).map(|i| i + 3))?;
    let d = digits(&s[i..]);
    let gen: u32 = match d.len() {
        5 => d[..2].parse().ok()?,
        4 => d[..1].parse().ok()?,
        _ => return None,
    };
    Some(match gen {
        12..=14 => known("Core 12-14th gen (hybrid)", 1.2),
        10 | 11 => known("Core 10-11th gen", 1.0),
        6..=9 => known("Core 6-9th gen", 0.85),
        _ => known("Core pre-Skylake", 0.6),
    })
}

/// Classify a `cpu_name` (`None`/blank -> unknown).
pub fn classify(cpu_name: Option<&str>) -> CpuPerf {
    let s = normalise(cpu_name.unwrap_or(""));
    for (pat, why) in DENY {
        if s.contains(pat) {
            return CpuPerf {
                family: why.into(),
                factor: 0.0,
                denied: Some(format!(
                    "cpu {:?} is deny-listed: {why}",
                    cpu_name.unwrap_or("").trim()
                )),
            };
        }
    }
    let hit = if s.contains("threadripper") {
        token_after(&s, "threadripper pro ")
            .or_else(|| token_after(&s, "threadripper "))
            .and_then(|t| ryzen(t, true))
    } else if s.contains("epyc") {
        token_after(&s, "epyc ").and_then(epyc)
    } else if s.contains("ryzen") {
        ["ryzen 9 ", "ryzen 7 ", "ryzen 5 ", "ryzen 3 "]
            .iter()
            .find_map(|k| token_after(&s, k))
            .and_then(|t| ryzen(t, false))
    } else if s.contains("xeon") {
        xeon(&s)
    } else {
        core(&s)
    };
    hit.unwrap_or_else(|| known("unknown", UNKNOWN_FACTOR))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(name: &str) -> f64 {
        classify(Some(name)).factor
    }

    #[test]
    fn deny_list_matches_live_names() {
        for n in [
            "Xeon Phi™ 7210 ",
            "Intel(R) Xeon Phi(TM) CPU 7250 @ 1.40GHz",
            "Intel Atom C3958",
            "Celeron® J4125",
            "Pentium® Gold G6400",
            "AMD Opteron(tm) Processor 6380",
        ] {
            let p = classify(Some(n));
            assert!(p.denied.is_some(), "{n}");
            assert_eq!(p.factor, 0.0);
        }
        assert!(classify(Some("AMD EPYC 7702 64-Core Processor"))
            .denied
            .is_none());
    }

    #[test]
    fn families_from_live_offer_names() {
        // Names verbatim from `POST /bundles/` (2026-09-27).
        let cases = [
            ("AMD EPYC 7702 64-Core Processor", 1.0),
            ("AMD EPYC 7K62 48-Core Processor", 1.0),
            ("AMD EPYC 7B13 64-Core Processor", 1.2),
            ("AMD EPYC 7702P 64-Core Processor", 1.0),
            ("AMD EPYC 9J14 96-Core Processor", 1.5),
            ("AMD EPYC 9654 96-Core Processor", 1.5),
            ("AMD Ryzen Threadripper PRO 5995WX 64-Cores", 1.3),
            ("AMD Ryzen Threadripper 3970X 32-Core Processor", 1.1),
            ("AMD Ryzen 9 9950X 16-Core Processor", 1.6),
            ("AMD Ryzen 9 5950X 16-Core Processor", 1.2),
            ("Xeon® E5-2686 v4 ", 0.7),
            ("Xeon® E5-2673 v3 ", 0.65),
            ("Xeon® E7-4860 v2 ", 0.45),
            ("Xeon® Gold 6430", 1.15),
            ("Xeon® Platinum 8352V ", 1.0),
            ("Xeon® Platinum 8272CL ", 0.9),
            ("Core™ i9-14900K", 1.2),
        ];
        for (n, want) in cases {
            assert!((f(n) - want).abs() < 1e-9, "{n}: {} != {want}", f(n));
        }
        assert!((f("Xeon® Silver 4114 ") - 0.85 * 0.85).abs() < 1e-9);
    }

    #[test]
    fn unknown_is_penalised_not_rewarded() {
        for n in ["", "   ", "AMD EPYC synthetic", "Mystery CPU 3000"] {
            let p = classify(Some(n));
            assert_eq!(p.factor, UNKNOWN_FACTOR, "{n}");
            assert!(p.denied.is_none());
        }
        assert_eq!(classify(None).factor, UNKNOWN_FACTOR);
        const _: () = assert!(UNKNOWN_FACTOR < 1.0);
    }

    #[test]
    fn pre_avx2_xeons_fall_below_default_floor() {
        for n in [
            "Xeon® E7-4860 v2 ",
            "Intel Xeon E5-2670 0 @ 2.60GHz",
            "Xeon X5670",
        ] {
            assert!(f(n) < DEFAULT_MIN_CPU_SPEED, "{n}: {}", f(n));
        }
        for n in [
            "Xeon® E5-2680 v4 ",
            "Xeon® E5-2699 v3 ",
            "Xeon® Silver 4114 ",
        ] {
            assert!(f(n) >= DEFAULT_MIN_CPU_SPEED, "{n}: {}", f(n));
        }
    }
}
