//! Minimal npm semver range evaluation: enough to answer "does this declared
//! range admit the patched version?" for the fix-reachability signal.
//! Supports `*`/`x`/`latest`, exact and `=`, `^`, `~`, `>= > <= <`,
//! space-separated AND, `||`, hyphen ranges and x-ranges (`1.x`, `1.2`).
//! Prerelease tags are ignored (compared on major.minor.patch).

pub type V = (u64, u64, u64);

/// Parse `1.2.3`, `v1.2.3`, `1.2.3-rc.1`; missing parts are `None`.
fn parts(s: &str) -> Option<(Option<u64>, Option<u64>, Option<u64>)> {
    let s = s.trim().trim_start_matches(['v', '=']);
    let core = s.split(['-', '+']).next()?;
    let mut it = core.split('.');
    let p = |x: Option<&str>| -> Option<Option<u64>> {
        match x {
            None | Some("x") | Some("X") | Some("*") => Some(None),
            Some(n) => n.parse().ok().map(Some),
        }
    };
    let (a, b, c) = (p(it.next())?, p(it.next())?, p(it.next())?);
    Some((a, b, c))
}

pub fn parse(v: &str) -> Option<V> {
    match parts(v)? {
        (Some(a), Some(b), Some(c)) => Some((a, b, c)),
        _ => None,
    }
}

/// One comparator set (AND) as inclusive-lower / exclusive-upper bounds.
fn bounds(set: &str) -> Option<(V, Option<V>)> {
    let set = set.trim();
    if set.is_empty() || matches!(set, "*" | "x" | "X" | "latest") {
        return Some(((0, 0, 0), None));
    }
    if let Some((a, b)) = set.split_once(" - ") {
        let lo = parts(a)?;
        let lo = (lo.0.unwrap_or(0), lo.1.unwrap_or(0), lo.2.unwrap_or(0));
        let hi = match parts(b)? {
            (Some(x), Some(y), Some(z)) => (x, y, z + 1),
            (Some(x), Some(y), None) => (x, y + 1, 0),
            (Some(x), None, _) => (x + 1, 0, 0),
            _ => return Some((lo, None)),
        };
        return Some((lo, Some(hi)));
    }
    let mut lo = (0, 0, 0);
    let mut hi: Option<V> = None;
    let tighten_hi = |hi: &mut Option<V>, v: V| {
        if hi.map(|h| v < h).unwrap_or(true) {
            *hi = Some(v);
        }
    };
    for c in set.split_whitespace() {
        let (op, rest) = ["^", "~", ">=", "<=", ">", "<", "="]
            .iter()
            .find_map(|op| c.strip_prefix(op).map(|r| (*op, r)))
            .unwrap_or(("", c));
        let p = parts(rest)?;
        let (a, b, cc) = (p.0.unwrap_or(0), p.1.unwrap_or(0), p.2.unwrap_or(0));
        match op {
            "^" => {
                lo = lo.max((a, b, cc));
                let up = if a > 0 || p.1.is_none() {
                    (a + 1, 0, 0)
                } else if b > 0 || p.2.is_none() {
                    (0, b + 1, 0)
                } else {
                    (0, 0, cc + 1)
                };
                tighten_hi(&mut hi, up);
            }
            "~" => {
                lo = lo.max((a, b, cc));
                let up = if p.1.is_none() {
                    (a + 1, 0, 0)
                } else {
                    (a, b + 1, 0)
                };
                tighten_hi(&mut hi, up);
            }
            ">=" => lo = lo.max((a, b, cc)),
            ">" => lo = lo.max((a, b, cc + 1)),
            "<" => tighten_hi(&mut hi, (a, b, cc)),
            "<=" => tighten_hi(&mut hi, (a, b, cc + 1)),
            _ => match p {
                (Some(_), Some(_), Some(_)) => {
                    lo = lo.max((a, b, cc));
                    tighten_hi(&mut hi, (a, b, cc + 1));
                }
                (Some(_), Some(_), None) => {
                    lo = lo.max((a, b, 0));
                    tighten_hi(&mut hi, (a, b + 1, 0));
                }
                (Some(_), None, _) => {
                    lo = lo.max((a, 0, 0));
                    tighten_hi(&mut hi, (a + 1, 0, 0));
                }
                _ => {}
            },
        }
    }
    Some((lo, hi))
}

/// Does `range` admit version `v`? Unparseable ranges (tags, URLs,
/// `workspace:`, `file:`) return `None`.
pub fn satisfies(range: &str, v: &str) -> Option<bool> {
    let v = parse(v)?;
    let r = range.trim();
    if r.contains(':') || r.contains('/') {
        return None;
    }
    let mut any = false;
    for set in r.split("||") {
        let (lo, hi) = bounds(set)?;
        if v >= lo && hi.map(|h| v < h).unwrap_or(true) {
            any = true;
        }
    }
    Some(any)
}

/// Can `range` resolve to a version that carries a fix first shipped in
/// `fixed`? True when some comparator set admits any version `>= fixed`:
/// either the set already starts at or above `fixed` (`^2.3.0` vs a fix in
/// 2.2.3: every admitted version descends from the fix), or it admits
/// `fixed` itself or something newer. Only a set whose exclusive upper
/// bound is at or below `fixed` keeps the fix out (`^0.9.1` vs 0.13.0).
/// Unparseable ranges return `None`, as in [`satisfies`].
pub fn reaches_fix(range: &str, fixed: &str) -> Option<bool> {
    let f = parse(fixed)?;
    let r = range.trim();
    if r.contains(':') || r.contains('/') {
        return None;
    }
    let mut any = false;
    for set in r.split("||") {
        let (lo, hi) = bounds(set)?;
        let nonempty = hi.map(|h| lo < h).unwrap_or(true);
        if nonempty && hi.map(|h| h > f).unwrap_or(true) {
            any = true;
        }
    }
    Some(any)
}

/// `^0.y.z`: a caret on a 0.x version only accepts patch/minor-locked
/// updates, so security minors are silently excluded.
pub fn is_zero_caret(range: &str) -> bool {
    let r = range.trim();
    r.starts_with("^0.") || r.starts_with("~0.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges() {
        assert_eq!(satisfies("^0.9.1", "0.13.0"), Some(false));
        assert_eq!(satisfies("^0.12.0", "0.13.0"), Some(false));
        assert_eq!(satisfies("~0.12.0", "0.13.0"), Some(false));
        assert_eq!(satisfies("^0.13.0", "0.13.0"), Some(true));
        assert_eq!(satisfies("^1.2.0", "1.9.9"), Some(true));
        assert_eq!(satisfies("^1.2.0", "2.0.0"), Some(false));
        assert_eq!(satisfies(">=0.12.0 <0.14.0", "0.13.0"), Some(true));
        assert_eq!(satisfies("0.12.x || 0.13.x", "0.13.4"), Some(true));
        assert_eq!(satisfies("1.0.0 - 1.2", "1.2.7"), Some(true));
        assert_eq!(satisfies("*", "9.0.0"), Some(true));
        assert_eq!(satisfies(">=2.0.0", "4.0.1"), Some(true));
        assert_eq!(satisfies("workspace:*", "1.0.0"), None);
        assert_eq!(satisfies("^0.0.3", "0.0.4"), Some(false));
        assert!(is_zero_caret("^0.9.1") && !is_zero_caret("^1.0.0"));
    }

    #[test]
    fn reaches_fix_counts_ranges_that_start_past_the_fix() {
        // Regression: ^2.3.0 excludes 2.2.3 but every version it admits
        // descends from the 2.2.3 fix, so the fix is reachable.
        assert_eq!(satisfies("^2.3.0", "2.2.3"), Some(false));
        assert_eq!(reaches_fix("^2.3.0", "2.2.3"), Some(true));
        assert_eq!(reaches_fix(">=2.3.0", "2.2.3"), Some(true));
        assert_eq!(reaches_fix("2.3.0", "2.2.3"), Some(true));
        // Ranges that admit the fixed version itself.
        assert_eq!(reaches_fix("^0.13.0", "0.13.0"), Some(true));
        assert_eq!(reaches_fix("^2.2.0", "2.2.3"), Some(true));
        assert_eq!(reaches_fix(">=1.0.0 <1.5.0", "1.4.0"), Some(true));
        // Ranges whose every version predates the fix.
        assert_eq!(reaches_fix("^0.9.1", "0.13.0"), Some(false));
        assert_eq!(reaches_fix("~0.12.0", "0.13.0"), Some(false));
        assert_eq!(reaches_fix("^1.0.0", "2.0.0"), Some(false));
        assert_eq!(reaches_fix("2.2.2", "2.2.3"), Some(false));
        assert_eq!(reaches_fix("<2.2.3", "2.2.3"), Some(false));
        // Any OR branch that reaches the fix is enough.
        assert_eq!(reaches_fix("^0.9.0 || ^0.13.0", "0.13.0"), Some(true));
        assert_eq!(reaches_fix("workspace:*", "1.0.0"), None);
    }
}
