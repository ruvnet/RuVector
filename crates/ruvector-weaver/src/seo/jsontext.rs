//! Format-preserving edits of a `package.json`: replace or insert one
//! top-level key without re-serialising (and re-ordering) the whole file,
//! so a metadata PR diff shows only the fields it changes.

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// Byte spans of one top-level member.
#[derive(Debug, Clone)]
struct Member {
    key: String,
    value_start: usize,
    value_end: usize,
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// End (exclusive) of the JSON string starting at `i` (which is `"`).
fn string_end(b: &[u8], mut i: usize) -> Result<usize> {
    i += 1;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2,
            b'"' => return Ok(i + 1),
            _ => i += 1,
        }
    }
    bail!("unterminated string")
}

/// End (exclusive) of the JSON value starting at `i`.
fn value_end(b: &[u8], i: usize) -> Result<usize> {
    match b.get(i) {
        Some(b'"') => string_end(b, i),
        Some(b'{') | Some(b'[') => {
            let mut depth = 0i32;
            let mut j = i;
            while j < b.len() {
                match b[j] {
                    b'"' => {
                        j = string_end(b, j)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return Ok(j + 1);
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            bail!("unbalanced value")
        }
        Some(_) => {
            let mut j = i;
            while j < b.len() && !matches!(b[j], b',' | b'}' | b']') && !b[j].is_ascii_whitespace()
            {
                j += 1;
            }
            Ok(j)
        }
        None => bail!("value expected"),
    }
}

fn members(text: &str) -> Result<(Vec<Member>, usize)> {
    let b = text.as_bytes();
    let mut i = skip_ws(b, 0);
    if b.get(i) != Some(&b'{') {
        bail!("package.json must be an object");
    }
    let mut out = Vec::new();
    i += 1;
    loop {
        i = skip_ws(b, i);
        match b.get(i) {
            Some(b'}') => return Ok((out, i)),
            Some(b',') => {
                i += 1;
                continue;
            }
            Some(b'"') => {}
            _ => bail!("unexpected byte at {i}"),
        }
        let ke = string_end(b, i)?;
        let key: String = serde_json::from_str(&text[i..ke]).context("key")?;
        i = skip_ws(b, ke);
        if b.get(i) != Some(&b':') {
            bail!("':' expected after {key:?}");
        }
        let vs = skip_ws(b, i + 1);
        let ve = value_end(b, vs)?;
        out.push(Member {
            key,
            value_start: vs,
            value_end: ve,
        });
        i = ve;
    }
}

/// Indent unit of the file (whitespace before the first member key).
fn indent_unit(text: &str) -> String {
    text.lines()
        .skip(1)
        .find(|l| l.trim_start().starts_with('"'))
        .map(|l| l[..l.len() - l.trim_start().len()].to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "  ".into())
}

fn render(v: &Value, unit: &str) -> Result<String> {
    use serde::Serialize;
    let mut buf = Vec::new();
    let fmt = serde_json::ser::PrettyFormatter::with_indent(unit.as_bytes());
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
    v.serialize(&mut ser)?;
    let s = String::from_utf8(buf)?;
    // Nested lines sit one level deeper than the top-level member.
    Ok(s.replace('\n', &format!("\n{unit}")))
}

/// Top-level value of `key`, parsed.
pub fn get(text: &str, key: &str) -> Option<Value> {
    serde_json::from_str::<Value>(text).ok()?.get(key).cloned()
}

/// Set top-level `key` to `value`. Existing keys are replaced in place;
/// new keys go after the first present key in `after` (else last).
pub fn set(text: &str, key: &str, value: &Value, after: &[&str]) -> Result<String> {
    serde_json::from_str::<Value>(text).context("package.json is not valid JSON")?;
    let (ms, close) = members(text)?;
    let unit = indent_unit(text);
    let rendered = render(value, &unit)?;
    let out = if let Some(m) = ms.iter().find(|m| m.key == key) {
        if serde_json::from_str::<Value>(&text[m.value_start..m.value_end])
            .ok()
            .as_ref()
            == Some(value)
        {
            return Ok(text.to_string());
        }
        format!(
            "{}{}{}",
            &text[..m.value_start],
            rendered,
            &text[m.value_end..]
        )
    } else {
        let anchor = after
            .iter()
            .find_map(|a| ms.iter().find(|m| m.key == *a))
            .or(ms.last());
        let entry = format!("\"{}\": {}", key, rendered);
        match anchor {
            Some(m) => format!(
                "{},\n{unit}{entry}{}",
                &text[..m.value_end],
                &text[m.value_end..]
            ),
            None => format!("{}\n{unit}{entry}\n{}", &text[..close], &text[close..]),
        }
    };
    let check: Value = serde_json::from_str(&out).context("edit produced invalid JSON")?;
    if check.get(key) != Some(value) {
        bail!("edit of {key:?} did not round-trip");
    }
    Ok(out)
}
