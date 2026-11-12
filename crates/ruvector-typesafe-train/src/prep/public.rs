//! Public intent datasets → rows, mirroring `bench/datasets/*.mjs` parsing and
//! split logic byte-for-byte (ids, CSV quirks, HWU64's hash split).
//!
//! Validation carve (plan Step 1.2, "10 % of train by sha256(id) % 10 == 0"):
//! the integer is the first 8 hex chars of sha256(id) read as u32 — the same
//! convention as the tickets `assignSplit`. Recorded in data-manifest.json.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::data::{Row, BANKING77, CLINC150, HWU64};
use crate::norm::sha256_hex;

/// One public dataset, already carved into gradient / selection / held-out.
pub struct Public {
    pub name: &'static str,
    pub train: Vec<Row>,
    pub val: Vec<Row>,
    pub heldout_texts: Vec<String>,
    pub criteria: BTreeMap<String, String>,
}

/// `humaniseLabel` from datasets/lib.mjs: `[._]+` → space, collapse, trim.
pub fn humanise(label: &str) -> String {
    let replaced: String = label
        .chars()
        .map(|c| if c == '.' || c == '_' { ' ' } else { c })
        .collect();
    replaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Hash-carved 10 % validation bucket.
pub fn is_val_bucket(id: &str) -> bool {
    let h = sha256_hex(id.as_bytes());
    u32::from_str_radix(&h[..8], 16)
        .expect("hex")
        .is_multiple_of(10)
}

fn carve(rows: Vec<Row>) -> (Vec<Row>, Vec<Row>) {
    rows.into_iter().partition(|r| !is_val_bucket(&r.id))
}

/// `splitCsvLine` from banking77.mjs (quote-aware; `""` escapes a quote).
fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut q = false;
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if q {
            if ch == '"' && chars.get(i + 1) == Some(&'"') {
                cur.push('"');
                i += 1;
            } else if ch == '"' {
                q = false;
            } else {
                cur.push(ch);
            }
        } else if ch == '"' {
            q = true;
        } else if ch == ',' {
            fields.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
        i += 1;
    }
    fields.push(cur);
    fields
}

/// JS `text.split(/\r?\n/)`.
fn js_lines(text: &str) -> Vec<&str> {
    text.split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect()
}

fn parse_banking_csv(text: &str) -> Vec<(String, String)> {
    let mut lines = js_lines(text).into_iter();
    let header = lines.next().unwrap_or("");
    let cols: Vec<String> = header.split(',').map(|c| c.trim().to_lowercase()).collect();
    let t = cols.iter().position(|c| c == "text");
    let c = cols.iter().position(|c| c == "category");
    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f = split_csv_line(line);
        if f.len() < 2 {
            continue;
        }
        if let (Some(t), Some(c)) = (t, c) {
            if let (Some(tx), Some(lb)) = (f.get(t), f.get(c)) {
                rows.push((tx.clone(), lb.clone()));
            }
        }
    }
    rows
}

pub fn banking77(train_csv: &[u8], test_csv: &[u8], categories: &[u8]) -> Result<Public> {
    let labels: Vec<String> = serde_json::from_slice(categories).context("banking77 categories")?;
    let criteria = labels.iter().map(|l| (l.clone(), humanise(l))).collect();
    let train_all = parse_banking_csv(std::str::from_utf8(train_csv)?)
        .into_iter()
        .enumerate()
        .map(|(i, (t, l))| Row::new(BANKING77, format!("b77-tr-{i}"), t, l))
        .collect();
    let heldout_texts = parse_banking_csv(std::str::from_utf8(test_csv)?)
        .into_iter()
        .map(|(t, _)| t)
        .collect();
    let (train, val) = carve(train_all);
    Ok(Public {
        name: BANKING77,
        train,
        val,
        heldout_texts,
        criteria,
    })
}

#[derive(Deserialize)]
struct ClincFull {
    train: Vec<(String, String)>,
    val: Vec<(String, String)>,
    test: Vec<(String, String)>,
    oos_train: Vec<(String, String)>,
    oos_val: Vec<(String, String)>,
    oos_test: Vec<(String, String)>,
}

pub fn clinc150(data_full: &[u8]) -> Result<Public> {
    let d: ClincFull = serde_json::from_slice(data_full).context("clinc150 data_full.json")?;
    let mk = |pairs: &[(String, String)], prefix: &str, oos: bool| -> Vec<Row> {
        pairs
            .iter()
            .enumerate()
            .map(|(i, (t, l))| {
                let label = if oos { "oos".to_string() } else { l.clone() };
                let mut r = Row::new(CLINC150, format!("{prefix}-{i}"), t.clone(), label);
                r.oos = oos;
                r
            })
            .collect()
    };
    let mut labels: Vec<String> = d.train.iter().map(|(_, l)| l.clone()).collect();
    labels.sort();
    labels.dedup();
    let criteria = labels.iter().map(|l| (l.clone(), humanise(l))).collect();
    // Official splits: CLINC150 uses its own `val` (ADR-008 §2).
    let mut train = mk(&d.train, "clinc-tr", false);
    train.extend(mk(&d.oos_train, "clinc-oostr", true));
    let mut val = mk(&d.val, "clinc-va", false);
    val.extend(mk(&d.oos_val, "clinc-oosva", true));
    let heldout_texts = d
        .test
        .iter()
        .chain(d.oos_test.iter())
        .map(|(t, _)| t.clone())
        .collect();
    Ok(Public {
        name: CLINC150,
        train,
        val,
        heldout_texts,
        criteria,
    })
}

pub fn hwu64(all_csv: &[u8]) -> Result<Public> {
    let text = std::str::from_utf8(all_csv)?;
    let mut lines = js_lines(text).into_iter().filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .unwrap_or("")
        .split(';')
        .map(|c| c.trim().to_lowercase())
        .collect();
    let idx = |n: &str| header.iter().position(|c| c == n);
    let (s, it, a) = (idx("scenario"), idx("intent"), idx("answer"));
    let (s, it, a) = (
        s.context("hwu64 header lacks scenario")?,
        it.context("hwu64 header lacks intent")?,
        a.context("hwu64 header lacks answer")?,
    );
    let maxi = s.max(it).max(a);
    let mut rows: Vec<(String, String)> = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split(';').collect();
        if parts.len() <= maxi {
            continue;
        }
        let (sc, in_, an) = (parts[s].trim(), parts[it].trim(), parts[a].trim());
        if sc.is_empty() || in_.is_empty() || an.is_empty() {
            continue;
        }
        rows.push((an.to_string(), format!("{sc}_{in_}")));
    }
    let mut labels: Vec<String> = rows.iter().map(|(_, l)| l.clone()).collect();
    labels.sort();
    labels.dedup();
    let criteria = labels.iter().map(|l| (l.clone(), humanise(l))).collect();
    let mut train_all = Vec::new();
    let mut heldout_texts = Vec::new();
    for (i, (t, l)) in rows.into_iter().enumerate() {
        // hwu64.mjs: sha256(text + '|' + label)[0] % 100 < 20 → test.
        let h = sha256_hex(format!("{t}|{l}").as_bytes());
        let first = u8::from_str_radix(&h[..2], 16).expect("hex");
        if first % 100 < 20 {
            heldout_texts.push(t);
        } else {
            train_all.push(Row::new(HWU64, format!("hwu-{i}"), t, l));
        }
    }
    let (train, val) = carve(train_all);
    Ok(Public {
        name: HWU64,
        train,
        val,
        heldout_texts,
        criteria,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_quotes_and_escapes() {
        assert_eq!(split_csv_line(r#""a, ""b""",c"#), vec![r#"a, "b""#, "c"]);
        let rows = parse_banking_csv("text,category\r\n\"hi, there\",card_arrival\n\nplain,x\n");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], ("hi, there".into(), "card_arrival".into()));
    }

    #[test]
    fn humanise_matches_js() {
        assert_eq!(humanise("card_arrival"), "card arrival");
        assert_eq!(humanise("alarm.set__x"), "alarm set x");
    }

    #[test]
    fn hwu_split_and_labels() {
        let csv = "userid;answerid;scenario;intent;answer\n1;2;alarm;set;wake me up at five\n1;3;;set;skip me\n";
        let p = hwu64(csv.as_bytes()).unwrap();
        assert_eq!(p.train.len() + p.val.len() + p.heldout_texts.len(), 1);
        assert_eq!(p.criteria.keys().collect::<Vec<_>>(), vec!["alarm_set"]);
    }

    #[test]
    fn clinc_oos_rows_are_flagged() {
        let j = r#"{"train":[["hi","greet"]],"val":[["yo","greet"]],"test":[["hey","greet"]],
            "oos_train":[["what is love","oos"]],"oos_val":[["?","oos"]],"oos_test":[["zzz","oos"]]}"#;
        let p = clinc150(j.as_bytes()).unwrap();
        assert_eq!(p.train.len(), 2);
        assert!(p.train[1].oos && p.train[1].label == "oos");
        assert_eq!(p.heldout_texts, vec!["hey", "zzz"]);
    }
}
