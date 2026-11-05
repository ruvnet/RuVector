//! Registry access behind a trait so scan / gate / propose logic is unit
//! tested against an in-memory fake and only the CLI touches the network.

use crate::http::{encode_pkg, Http};
use anyhow::Result;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub name: String,
    pub description: String,
    pub keywords: Vec<String>,
}

pub trait Registry: Sync {
    /// Does the npm name exist?
    fn exists(&self, name: &str) -> Result<bool>;
    /// `registry.npmjs.org/<name>/latest` manifest.
    fn latest(&self, name: &str) -> Result<Option<Value>>;
    /// Free-text search (already restricted by the caller's query).
    fn search(&self, text: &str) -> Result<Vec<SearchHit>>;
}

pub struct NpmRegistry {
    pub http: Http,
    probes: Mutex<BTreeMap<String, bool>>,
}

impl NpmRegistry {
    pub fn new(http: Http) -> Self {
        Self {
            http,
            probes: Mutex::new(BTreeMap::new()),
        }
    }
}

impl Registry for NpmRegistry {
    fn exists(&self, name: &str) -> Result<bool> {
        if let Some(v) = self.probes.lock().expect("probe lock").get(name) {
            return Ok(*v);
        }
        let url = format!("https://registry.npmjs.org/{}/latest", encode_pkg(name));
        let found = self.http.get_text(&url)?.is_some();
        self.probes
            .lock()
            .expect("probe lock")
            .insert(name.to_string(), found);
        Ok(found)
    }

    fn latest(&self, name: &str) -> Result<Option<Value>> {
        let url = format!("https://registry.npmjs.org/{}/latest", encode_pkg(name));
        self.http.get_json(&url)
    }

    fn search(&self, text: &str) -> Result<Vec<SearchHit>> {
        let url = format!(
            "https://registry.npmjs.org/-/v1/search?text={}&size=25",
            url_escape(text)
        );
        let Some(v) = self.http.get_json(&url)? else {
            return Ok(vec![]);
        };
        Ok(parse_search_hits(&v))
    }
}

pub fn parse_search_hits(v: &Value) -> Vec<SearchHit> {
    v["objects"]
        .as_array()
        .map(|objs| {
            objs.iter()
                .map(|o| {
                    let p = &o["package"];
                    SearchHit {
                        name: p["name"].as_str().unwrap_or_default().to_string(),
                        description: p["description"].as_str().unwrap_or_default().to_string(),
                        keywords: str_array(&p["keywords"]),
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn str_array(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

pub fn url_escape(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// In-memory registry for tests and offline runs.
#[derive(Default)]
pub struct MemRegistry {
    pub names: BTreeSet<String>,
    pub manifests: BTreeMap<String, Value>,
    pub hits: Vec<SearchHit>,
}

impl MemRegistry {
    pub fn with_names<I: IntoIterator<Item = &'static str>>(names: I) -> Self {
        Self {
            names: names.into_iter().map(String::from).collect(),
            ..Self::default()
        }
    }
}

impl Registry for MemRegistry {
    fn exists(&self, name: &str) -> Result<bool> {
        Ok(self.names.contains(name) || self.manifests.contains_key(name))
    }
    fn latest(&self, name: &str) -> Result<Option<Value>> {
        Ok(self.manifests.get(name).cloned())
    }
    fn search(&self, text: &str) -> Result<Vec<SearchHit>> {
        let t = text.to_ascii_lowercase();
        let words: Vec<&str> = t.split_whitespace().filter(|w| !w.contains(':')).collect();
        Ok(self
            .hits
            .iter()
            .filter(|h| {
                let hay = format!("{} {} {}", h.name, h.description, h.keywords.join(" "))
                    .to_ascii_lowercase();
                words.iter().any(|w| hay.contains(w))
            })
            .cloned()
            .collect())
    }
}
