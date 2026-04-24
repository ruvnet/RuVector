//! Secret loading for Kalshi credentials.
//!
//! Three sources, checked in order:
//! 1. `KALSHI_SECRET_SOURCE=local` → read PEM + API key from local files
//!    (dev only). Paths default to `.kalshi/kalshi.pem` and
//!    `.kalshi/api-key.txt` under the current working directory.
//! 2. `KALSHI_SECRET_SOURCE=env`   → read from `KALSHI_API_KEY` and
//!    `KALSHI_PRIVATE_KEY_PEM` env vars.
//! 3. Default: shell out to `gcloud secrets versions access latest`.
//!
//! The result is cached in memory for 5 minutes so hot REST paths do not
//! re-invoke `gcloud` per request.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

use crate::{KalshiError, Result};

/// Loaded Kalshi credentials. The PEM is kept as a `String` rather than a
/// parsed key so the [`crate::auth::Signer`] stays the single owner of the
/// private key material.
#[derive(Clone)]
pub struct Credentials {
    pub api_key: String,
    pub private_key_pem: String,
    pub api_url: String,
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("api_key_len", &self.api_key.len())
            .field("pem_len", &self.private_key_pem.len())
            .field("api_url", &self.api_url)
            .finish()
    }
}

#[derive(Clone)]
pub struct SecretLoader {
    project: String,
    ttl: Duration,
    inner: Arc<Mutex<Option<(Credentials, Instant)>>>,
}

impl SecretLoader {
    pub fn new(project: impl Into<String>) -> Self {
        Self {
            project: project.into(),
            ttl: Duration::from_secs(300),
            inner: Arc::new(Mutex::new(None)),
        }
    }

    /// Override the cache TTL. Primarily for tests.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Load credentials, using the in-memory cache if fresh.
    pub async fn load(&self) -> Result<Credentials> {
        {
            let guard = self.inner.lock().await;
            if let Some((creds, loaded_at)) = guard.as_ref() {
                if loaded_at.elapsed() < self.ttl {
                    return Ok(creds.clone());
                }
            }
        }
        let creds = self.load_fresh().await?;
        let mut guard = self.inner.lock().await;
        *guard = Some((creds.clone(), Instant::now()));
        Ok(creds)
    }

    async fn load_fresh(&self) -> Result<Credentials> {
        match std::env::var("KALSHI_SECRET_SOURCE").as_deref() {
            Ok("local") => self.load_local().await,
            Ok("env") => self.load_env(),
            _ => self.load_gcloud().await,
        }
    }

    fn load_env(&self) -> Result<Credentials> {
        let api_key = std::env::var("KALSHI_API_KEY")
            .map_err(|_| KalshiError::Secret("KALSHI_API_KEY not set".into()))?;
        let private_key_pem = std::env::var("KALSHI_PRIVATE_KEY_PEM")
            .map_err(|_| KalshiError::Secret("KALSHI_PRIVATE_KEY_PEM not set".into()))?;
        let api_url =
            std::env::var("KALSHI_API_URL").unwrap_or_else(|_| crate::KALSHI_API_URL.to_string());
        Ok(Credentials {
            api_key,
            private_key_pem,
            api_url,
        })
    }

    async fn load_local(&self) -> Result<Credentials> {
        let pem_path = std::env::var("KALSHI_PEM_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".kalshi/kalshi.pem"));
        let api_key_path = std::env::var("KALSHI_API_KEY_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".kalshi/api-key.txt"));

        let private_key_pem = tokio::fs::read_to_string(&pem_path)
            .await
            .map_err(|e| KalshiError::Secret(format!("reading {pem_path:?}: {e}")))?;

        let api_key = if let Ok(key) = std::env::var("KALSHI_API_KEY") {
            key
        } else {
            tokio::fs::read_to_string(&api_key_path)
                .await
                .map(|s| s.trim().to_string())
                .map_err(|e| {
                    KalshiError::Secret(format!(
                        "reading {api_key_path:?} (or set KALSHI_API_KEY): {e}"
                    ))
                })?
        };

        let api_url =
            std::env::var("KALSHI_API_URL").unwrap_or_else(|_| crate::KALSHI_API_URL.to_string());

        Ok(Credentials {
            api_key,
            private_key_pem,
            api_url,
        })
    }

    async fn load_gcloud(&self) -> Result<Credentials> {
        let api_key = gcloud_secret(&self.project, "KALSHI_API_KEY").await?;
        let private_key_pem = gcloud_secret(&self.project, "KALSHI_PRIVATE_KEY_PEM").await?;
        let api_url = match gcloud_secret(&self.project, "KALSHI_API_URL").await {
            Ok(s) => s,
            Err(_) => crate::KALSHI_API_URL.to_string(),
        };
        Ok(Credentials {
            api_key: api_key.trim().to_string(),
            private_key_pem,
            api_url: api_url.trim().to_string(),
        })
    }
}

async fn gcloud_secret(project: &str, name: &str) -> Result<String> {
    let output = tokio::process::Command::new("gcloud")
        .args([
            "secrets",
            "versions",
            "access",
            "latest",
            "--secret",
            name,
            "--project",
            project,
        ])
        .output()
        .await
        .map_err(|e| KalshiError::Secret(format!("spawning gcloud for {name}: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Err(KalshiError::Secret(format!(
            "gcloud secrets access {name} failed: {stderr}"
        )));
    }
    let val = String::from_utf8(output.stdout)
        .map_err(|e| KalshiError::Secret(format!("gcloud output for {name} is not utf-8: {e}")))?;
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn env_source_reads_vars() {
        // SAFETY: env mutation in tests is serialized by running this test
        // in isolation from other env-touching tests in this module (only one).
        std::env::set_var("KALSHI_SECRET_SOURCE", "env");
        std::env::set_var("KALSHI_API_KEY", "k-env-1");
        std::env::set_var("KALSHI_PRIVATE_KEY_PEM", "PEM-PLACEHOLDER");
        std::env::remove_var("KALSHI_API_URL");

        let loader = SecretLoader::new("test").with_ttl(Duration::from_millis(10));
        let creds = loader.load().await.unwrap();
        assert_eq!(creds.api_key, "k-env-1");
        assert_eq!(creds.private_key_pem, "PEM-PLACEHOLDER");
        assert_eq!(creds.api_url, crate::KALSHI_API_URL);

        // Cache hit returns the same value immediately.
        let creds2 = loader.load().await.unwrap();
        assert_eq!(creds.api_key, creds2.api_key);

        std::env::remove_var("KALSHI_SECRET_SOURCE");
        std::env::remove_var("KALSHI_API_KEY");
        std::env::remove_var("KALSHI_PRIVATE_KEY_PEM");
    }

    #[test]
    fn debug_does_not_leak_pem() {
        let c = Credentials {
            api_key: "super-secret-1234567890".into(),
            private_key_pem: "-----BEGIN RSA PRIVATE KEY----- secret".into(),
            api_url: crate::KALSHI_API_URL.into(),
        };
        let s = format!("{c:?}");
        assert!(!s.contains("super-secret"));
        assert!(!s.contains("BEGIN RSA"));
    }
}
