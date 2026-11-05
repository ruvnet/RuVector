//! One HTTP client for every registry call: fixed User-Agent (crates.io
//! requires one), 429 / 5xx backoff honouring `Retry-After`, and an on-disk
//! JSON cache keyed by URL so re-running a tick does not re-crawl ~400
//! packages. No credentials are ever sent: every endpoint used is public.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

const UA: &str = "ruvector-weaver/0.1 (+https://github.com/ruvnet/ruvector)";
const MAX_ATTEMPTS: u32 = 7;

#[derive(Clone)]
pub struct Http {
    agent: ureq::Agent,
    cache_dir: Option<PathBuf>,
    ttl: Duration,
}

/// Result of a GET: `None` body means HTTP 404.
pub type Fetched = Option<String>;

impl Http {
    pub fn new(cache_dir: Option<PathBuf>, ttl: Duration) -> Self {
        let agent = ureq::AgentBuilder::new()
            .user_agent(UA)
            .timeout_connect(Duration::from_secs(15))
            .timeout(Duration::from_secs(60))
            .build();
        Self {
            agent,
            cache_dir,
            ttl,
        }
    }

    fn cache_path(&self, url: &str) -> Option<PathBuf> {
        let dir = self.cache_dir.as_ref()?;
        let key = hex(&Sha256::digest(url.as_bytes()));
        Some(dir.join(format!("{key}.json")))
    }

    fn cached(&self, url: &str) -> Option<Fetched> {
        let path = self.cache_path(url)?;
        let meta = std::fs::metadata(&path).ok()?;
        let age = SystemTime::now()
            .duration_since(meta.modified().ok()?)
            .unwrap_or(Duration::MAX);
        if age > self.ttl {
            return None;
        }
        let text = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str::<Fetched>(&text).ok()
    }

    fn store(&self, url: &str, body: &Fetched) {
        if let Some(path) = self.cache_path(url) {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Ok(text) = serde_json::to_string(body) {
                let _ = std::fs::write(path, text);
            }
        }
    }

    /// GET `url` as text. 404 -> `Ok(None)`. Retries 429/5xx with backoff.
    pub fn get_text(&self, url: &str) -> Result<Fetched> {
        if let Some(hit) = self.cached(url) {
            return Ok(hit);
        }
        let mut delay = Duration::from_millis(800);
        for attempt in 1..=MAX_ATTEMPTS {
            match self.agent.get(url).call() {
                Ok(resp) => {
                    let body = resp.into_string().context("reading body")?;
                    let out = Some(body);
                    self.store(url, &out);
                    return Ok(out);
                }
                Err(ureq::Error::Status(404, _)) => {
                    self.store(url, &None);
                    return Ok(None);
                }
                Err(ureq::Error::Status(code, resp)) if code == 429 || code >= 500 => {
                    let wait = resp
                        .header("retry-after")
                        .and_then(|v| v.trim().parse::<u64>().ok())
                        .map(Duration::from_secs)
                        .unwrap_or(delay)
                        .min(Duration::from_secs(120));
                    if attempt == MAX_ATTEMPTS {
                        bail!("GET {url}: HTTP {code} after {attempt} attempts");
                    }
                    std::thread::sleep(wait);
                    delay = (delay * 2).min(Duration::from_secs(60));
                }
                Err(ureq::Error::Status(code, _)) => bail!("GET {url}: HTTP {code}"),
                Err(e) => {
                    if attempt == MAX_ATTEMPTS {
                        return Err(e).with_context(|| format!("GET {url}"));
                    }
                    std::thread::sleep(delay);
                    delay = (delay * 2).min(Duration::from_secs(60));
                }
            }
        }
        unreachable!("retry loop always returns")
    }

    pub fn get_json(&self, url: &str) -> Result<Option<serde_json::Value>> {
        match self.get_text(url)? {
            Some(t) => Ok(Some(
                serde_json::from_str(&t).with_context(|| format!("JSON from {url}"))?,
            )),
            None => Ok(None),
        }
    }

    /// Download raw bytes (tarballs), capped, never cached.
    pub fn get_bytes(&self, url: &str, cap: u64) -> Result<Vec<u8>> {
        let resp = self
            .agent
            .get(url)
            .call()
            .with_context(|| format!("GET {url}"))?;
        let mut buf = Vec::new();
        resp.into_reader()
            .take(cap + 1)
            .read_to_end(&mut buf)
            .context("reading bytes")?;
        if buf.len() as u64 > cap {
            bail!("{url}: larger than {cap} bytes");
        }
        Ok(buf)
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `@scope/name` -> `@scope%2fname` for registry paths.
pub fn encode_pkg(name: &str) -> String {
    name.replace('/', "%2f")
}
