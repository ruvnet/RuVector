//! Google Cloud Storage client for RVF cognitive containers
//!
//! Architecture: DashMap serves as hot in-memory cache. When `GCS_BUCKET` is
//! configured and running on GCE/Cloud Run, tokens are automatically fetched
//! from the metadata server and refreshed before expiry.
//!
//! When `GCS_BUCKET` is absent (local dev), operates as local-only.

use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Debug, Error)]
pub enum GcsError {
    #[error("Upload failed: {0}")]
    UploadFailed(String),
    #[error("Download failed: {0}")]
    DownloadFailed(String),
    #[error("Delete failed: {0}")]
    DeleteFailed(String),
    #[error("Object not found: {0}")]
    NotFound(String),
}

/// Cached access token with expiry
struct TokenCache {
    token: String,
    expires_at: std::time::Instant,
}

/// GCS client for storing RVF cognitive containers.
///
/// Write-through: local DashMap cache + GCS REST when credentials available.
/// Tokens are refreshed from GCE metadata server or `GCS_TOKEN` env var.
pub struct GcsClient {
    bucket: String,
    /// Static token from env (local dev) or None for metadata server
    static_token: Option<String>,
    /// Cached metadata server token (auto-refreshed)
    token_cache: RwLock<Option<TokenCache>>,
    http: reqwest::Client,
    /// In-memory cache (always populated)
    local_store: dashmap::DashMap<String, Vec<u8>>,
    /// Whether we're on GCE (metadata server available)
    use_metadata_server: bool,
}

impl GcsClient {
    pub fn new() -> Self {
        let bucket =
            std::env::var("GCS_BUCKET").unwrap_or_else(|_| "ruvector-brain-dev".to_string());
        let static_token = std::env::var("GCS_TOKEN").ok();
        let use_metadata_server = static_token.is_none() && std::env::var("GCS_BUCKET").is_ok();

        if static_token.is_some() {
            tracing::info!("GCS persistence enabled (static token) for bucket: {bucket}");
        } else if use_metadata_server {
            tracing::info!("GCS persistence enabled (metadata server) for bucket: {bucket}");
        } else {
            tracing::info!("GCS running in local-only mode (no GCS_BUCKET)");
        }

        Self {
            bucket,
            static_token,
            token_cache: RwLock::new(None),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
            local_store: dashmap::DashMap::new(),
            use_metadata_server,
        }
    }

    /// Whether GCS persistence is enabled
    pub fn is_persistent(&self) -> bool {
        self.static_token.is_some() || self.use_metadata_server
    }

    /// Get a valid access token, refreshing from metadata server if needed
    async fn get_token(&self) -> Option<String> {
        // Static token (env var) takes priority
        if let Some(ref token) = self.static_token {
            return Some(token.clone());
        }

        if !self.use_metadata_server {
            return None;
        }

        // Check cached token
        {
            let cache = self.token_cache.read().await;
            if let Some(ref tc) = *cache {
                // Refresh 5 minutes before expiry
                if tc.expires_at > std::time::Instant::now() + std::time::Duration::from_secs(300) {
                    return Some(tc.token.clone());
                }
            }
        }

        // Refresh from metadata server
        self.refresh_token().await
    }

    /// Fetch a new token from the GCE metadata server
    async fn refresh_token(&self) -> Option<String> {
        let url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
        let resp = self
            .http
            .get(url)
            .header("Metadata-Flavor", "Google")
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            tracing::warn!("GCE metadata token request failed: {}", resp.status());
            return None;
        }

        #[derive(serde::Deserialize)]
        struct TokenResponse {
            access_token: String,
            expires_in: u64,
        }

        let token_resp: TokenResponse = resp.json().await.ok()?;
        let expires_at =
            std::time::Instant::now() + std::time::Duration::from_secs(token_resp.expires_in);

        let token = token_resp.access_token.clone();

        // Cache the new token
        {
            let mut cache = self.token_cache.write().await;
            *cache = Some(TokenCache {
                token: token_resp.access_token,
                expires_at,
            });
        }

        tracing::debug!("GCS token refreshed, expires in {}s", token_resp.expires_in);
        Some(token)
    }

    /// Upload RVF container bytes (cache + GCS write-through)
    pub async fn upload_rvf(
        &self,
        contributor: &str,
        memory_id: &str,
        data: &[u8],
    ) -> Result<String, GcsError> {
        let path = format!("{contributor}/{memory_id}.rvf");

        // Always cache locally
        self.local_store.insert(path.clone(), data.to_vec());

        // Write-through to GCS
        if let Some(token) = self.get_token().await {
            let url = format!(
                "https://storage.googleapis.com/upload/storage/v1/b/{}/o?uploadType=media&name={}",
                self.bucket,
                urlencoding::encode(&path)
            );
            let result = self
                .http
                .post(&url)
                .bearer_auth(&token)
                .header("Content-Type", "application/octet-stream")
                .body(data.to_vec())
                .send()
                .await;
            match result {
                Ok(resp) if resp.status().as_u16() == 401 => {
                    // Token expired mid-flight, try once with fresh token
                    tracing::info!("GCS token expired on upload, refreshing...");
                    if let Some(new_token) = self.refresh_token().await {
                        let retry = self
                            .http
                            .post(&url)
                            .bearer_auth(&new_token)
                            .header("Content-Type", "application/octet-stream")
                            .body(data.to_vec())
                            .send()
                            .await;
                        if let Ok(resp) = retry {
                            if !resp.status().is_success() {
                                tracing::warn!(
                                    "GCS upload {path} retry returned {}",
                                    resp.status()
                                );
                            }
                        }
                    }
                }
                Ok(resp) if !resp.status().is_success() => {
                    tracing::warn!("GCS upload {path} returned {}", resp.status());
                }
                Err(e) => {
                    tracing::warn!("GCS upload {path} failed: {e}");
                }
                _ => {}
            }
        }

        Ok(format!("gs://{}/{}", self.bucket, path))
    }

    /// Download RVF container bytes (cache-first, then GCS)
    pub async fn download_rvf(
        &self,
        contributor: &str,
        memory_id: &str,
    ) -> Result<Vec<u8>, GcsError> {
        let path = format!("{contributor}/{memory_id}.rvf");

        // Check local cache first
        if let Some(data) = self.local_store.get(&path) {
            return Ok(data.clone());
        }

        // Try GCS
        if let Some(token) = self.get_token().await {
            let url = format!(
                "https://storage.googleapis.com/storage/v1/b/{}/o/{}?alt=media",
                self.bucket,
                urlencoding::encode(&path)
            );
            match self.http.get(&url).bearer_auth(&token).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let bytes = resp
                        .bytes()
                        .await
                        .map_err(|e| GcsError::DownloadFailed(e.to_string()))?;
                    let data = bytes.to_vec();
                    // Populate cache
                    self.local_store.insert(path, data.clone());
                    return Ok(data);
                }
                Ok(resp) if resp.status().as_u16() == 404 => {
                    return Err(GcsError::NotFound(path));
                }
                Ok(resp) => {
                    tracing::warn!("GCS download {path} returned {}", resp.status());
                }
                Err(e) => {
                    tracing::warn!("GCS download {path} failed: {e}");
                }
            }
        }

        Err(GcsError::NotFound(path))
    }

    /// Delete RVF container (cache + GCS)
    pub async fn delete_rvf(&self, contributor: &str, memory_id: &str) -> Result<(), GcsError> {
        let path = format!("{contributor}/{memory_id}.rvf");
        self.local_store.remove(&path);

        // Delete from GCS
        if let Some(token) = self.get_token().await {
            let url = format!(
                "https://storage.googleapis.com/storage/v1/b/{}/o/{}",
                self.bucket,
                urlencoding::encode(&path)
            );
            if let Err(e) = self.http.delete(&url).bearer_auth(&token).send().await {
                tracing::warn!("GCS delete {path} failed: {e}");
            }
        }

        Ok(())
    }

    /// Get bucket name
    pub fn bucket(&self) -> &str {
        &self.bucket
    }
}

impl Default for GcsClient {
    fn default() -> Self {
        Self::new()
    }
}
