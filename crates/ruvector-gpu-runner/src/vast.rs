//! Minimal blocking vast.ai REST client. The API key lives only in this
//! struct (read from the process env) and is never logged; `Client` does not
//! implement `Debug` on purpose.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::offer::Offer;

const BASE: &str = "https://console.vast.ai/api";
/// Label prefix identifying instances created by this tool (used by `reap`).
pub const LABEL_PREFIX: &str = "rvgr-";

pub struct Client {
    agent: ureq::Agent,
    auth: String,
}

impl Client {
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("VAST_API_KEY")
            .map_err(|_| anyhow!("VAST_API_KEY not set (export it for this process only)"))?;
        if key.trim().is_empty() {
            bail!("VAST_API_KEY is empty");
        }
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(60))
            .build();
        Ok(Self {
            agent,
            auth: format!("Bearer {}", key.trim()),
        })
    }

    fn call(&self, method: &str, path: &str, body: Option<&Value>) -> Result<(u16, Value)> {
        let url = format!("{BASE}{path}");
        let req = self
            .agent
            .request(method, &url)
            .set("Authorization", &self.auth)
            .set("Accept", "application/json");
        let res = match body {
            Some(b) => req.send_json(b.clone()),
            None => req.call(),
        };
        match res {
            Ok(r) => {
                let code = r.status();
                let v = r.into_json::<Value>().unwrap_or(Value::Null);
                Ok((code, v))
            }
            Err(ureq::Error::Status(code, r)) => {
                let v = r.into_json::<Value>().unwrap_or(Value::Null);
                Ok((code, v))
            }
            Err(e) => Err(anyhow!("{method} {path}: transport error: {e}")),
        }
    }

    pub fn search_offers(&self, query: &Value) -> Result<Vec<Offer>> {
        let (code, v) = self.call("POST", "/v0/bundles/", Some(query))?;
        if code != 200 {
            bail!("offer search failed: HTTP {code}: {v}");
        }
        let offers = v.get("offers").cloned().unwrap_or(json!([]));
        serde_json::from_value(offers).context("parsing offers")
    }

    pub fn credit_usd(&self) -> Result<f64> {
        let (code, v) = self.call("GET", "/v0/users/current/", None)?;
        if code != 200 {
            bail!("users/current failed: HTTP {code}");
        }
        v.get("credit")
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow!("no credit field"))
    }

    /// Instances owned by the caller, via the paginated `/api/v1/instances/`
    /// (the v0 list endpoint returns HTTP 410 `deprecated_endpoint`). Mirrors
    /// vast-cli `_fetch_all_instances_v1`: `limit` + `after_token` paging.
    pub fn list_mine(&self) -> Result<Vec<Value>> {
        let mut all = Vec::new();
        let mut path = "/v1/instances/?limit=25".to_string();
        for _ in 0..200 {
            let (code, v) = self.call("GET", &path, None)?;
            if code != 200 || v.get("success") == Some(&Value::Bool(false)) {
                bail!("list instances failed: HTTP {code}");
            }
            if let Some(a) = v.get("instances").and_then(Value::as_array) {
                all.extend(a.iter().cloned());
            }
            match v.get("next_token").and_then(Value::as_str) {
                Some(t) if !t.is_empty() => {
                    path = format!("/v1/instances/?limit=25&after_token={}", urlencode(t));
                }
                _ => return Ok(all),
            }
        }
        bail!("list instances: too many pages")
    }

    /// Returns `None` if the instance no longer exists. v0 is what vast-cli
    /// still uses for show; on a 410 deprecation we retry the v1 path.
    pub fn show(&self, id: u64) -> Result<Option<Value>> {
        let (mut code, mut v) = self.call("GET", &format!("/v0/instances/{id}/?owner=me"), None)?;
        if is_deprecated(code, &v) {
            (code, v) = self.call("GET", &format!("/v1/instances/{id}/"), None)?;
        }
        if code == 404 {
            return Ok(None);
        }
        if code != 200 {
            bail!("show instance {id}: HTTP {code}");
        }
        let inst = v
            .get("instances")
            .or_else(|| v.get("instance"))
            .cloned()
            .unwrap_or(Value::Null);
        Ok(
            if inst.is_null() || inst.as_object().is_some_and(|o| o.is_empty()) {
                None
            } else {
                Some(inst)
            },
        )
    }

    /// PAID CALL. Rent `offer_id`; returns the new instance (contract) id.
    /// Body shape matches vast-cli `create instance` (PUT /api/v0/asks/{id}/).
    pub fn create(&self, offer_id: u64, body: &Value) -> Result<u64> {
        let (code, v) = self.call("PUT", &format!("/v0/asks/{offer_id}/"), Some(body))?;
        if code != 200 || v.get("success") != Some(&Value::Bool(true)) {
            bail!("create failed: HTTP {code}: {}", redact(&v));
        }
        v.get("new_contract")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("create response lacks new_contract"))
    }

    /// One DELETE attempt (vast-cli: DELETE /api/v0/instances/{id}/ with `{}`).
    /// 404 counts as already destroyed; a 410 deprecation retries on v1.
    fn delete_once(&self, id: u64) -> Result<()> {
        let empty = json!({});
        let (mut code, mut v) =
            self.call("DELETE", &format!("/v0/instances/{id}/"), Some(&empty))?;
        if is_deprecated(code, &v) {
            (code, v) = self.call("DELETE", &format!("/v1/instances/{id}/"), Some(&empty))?;
        }
        match code {
            404 => Ok(()),
            200 | 204 if v.get("success") != Some(&Value::Bool(false)) => Ok(()),
            _ => bail!("destroy {id}: HTTP {code}: {}", redact(&v)),
        }
    }

    /// Idempotent destroy: retries with backoff on any error, then verifies the
    /// id has disappeared from the v1 instance list. A failed destroy is a money leak,
    /// so this keeps trying for up to ~10 minutes before giving up loudly.
    pub fn destroy_verified(&self, id: u64) -> Result<()> {
        let start = Instant::now();
        let mut backoff = Duration::from_secs(2);
        loop {
            let attempt = self.delete_once(id).and_then(|_| self.wait_gone(id));
            match attempt {
                Ok(()) => return Ok(()),
                Err(e) if start.elapsed() < Duration::from_secs(600) => {
                    eprintln!("[destroy] instance {id}: {e:#}; retrying in {backoff:?}");
                    sleep(backoff);
                    backoff = (backoff * 2).min(Duration::from_secs(60));
                }
                Err(e) => {
                    bail!("DESTROY NOT CONFIRMED for instance {id} after 10 min: {e:#}. Destroy it manually NOW.")
                }
            }
        }
    }

    fn wait_gone(&self, id: u64) -> Result<()> {
        for _ in 0..12 {
            let mine = self.list_mine()?;
            if !mine
                .iter()
                .any(|i| i.get("id").and_then(Value::as_u64) == Some(id))
            {
                return Ok(());
            }
            sleep(Duration::from_secs(5));
        }
        bail!("instance {id} still listed after destroy")
    }
}

/// Strip anything that could echo request env (signed URLs) back into logs.
fn redact(v: &Value) -> String {
    let s = v.to_string();
    if s.contains("X-Goog-Signature") || s.contains("x-goog-signature") {
        "<response redacted: contained signed URL>".into()
    } else {
        s.chars().take(500).collect()
    }
}

fn is_deprecated(code: u16, v: &Value) -> bool {
    code == 410 || v.get("error").and_then(Value::as_str) == Some("deprecated_endpoint")
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}
