//! Google Secret Manager integration

use anyhow::{anyhow, Result};
use std::env;

/// Secret Manager for accessing credentials
pub struct SecretManager {
    /// Google Cloud project ID
    project_id: String,

    /// Cached Gemini API key
    cached_gemini_key: Option<String>,
}

impl SecretManager {
    /// Create a new secret manager
    pub fn new(project_id: impl Into<String>) -> Self {
        Self {
            project_id: project_id.into(),
            cached_gemini_key: None,
        }
    }

    /// Create with default project (ruv-dev)
    pub fn default_project() -> Self {
        Self::new("ruv-dev")
    }

    /// Get the Gemini API key
    pub async fn get_gemini_api_key(&mut self) -> Result<String> {
        // Check cache first
        if let Some(ref key) = self.cached_gemini_key {
            return Ok(key.clone());
        }

        // Try environment variable first (for local dev)
        if let Ok(key) = env::var("GOOGLE_API_KEY") {
            self.cached_gemini_key = Some(key.clone());
            return Ok(key);
        }

        if let Ok(key) = env::var("GEMINI_API_KEY") {
            self.cached_gemini_key = Some(key.clone());
            return Ok(key);
        }

        // Try Google Secret Manager
        let secret_name = format!(
            "projects/{}/secrets/gemini-api-key/versions/latest",
            self.project_id
        );

        match self.access_secret(&secret_name).await {
            Ok(key) => {
                self.cached_gemini_key = Some(key.clone());
                Ok(key)
            }
            Err(e) => Err(anyhow!(
                "Failed to get Gemini API key. Set GOOGLE_API_KEY env var or configure Secret Manager: {}",
                e
            ))
        }
    }

    /// Get a secret by name
    pub async fn get_secret(&self, secret_id: &str) -> Result<String> {
        let secret_name = format!(
            "projects/{}/secrets/{}/versions/latest",
            self.project_id, secret_id
        );
        self.access_secret(&secret_name).await
    }

    /// Access a secret from Google Secret Manager
    async fn access_secret(&self, secret_name: &str) -> Result<String> {
        // Use gcloud CLI for simplicity (avoids heavy SDK dependency)
        // In production, would use google-cloud-secretmanager crate
        let output = tokio::process::Command::new("gcloud")
            .args([
                "secrets",
                "versions",
                "access",
                "latest",
                "--secret",
                &self.extract_secret_id(secret_name),
                "--project",
                &self.project_id,
            ])
            .output()
            .await?;

        if output.status.success() {
            let secret = String::from_utf8(output.stdout)?;
            Ok(secret.trim().to_string())
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(anyhow!("gcloud secrets access failed: {}", error))
        }
    }

    /// Extract secret ID from full resource name
    fn extract_secret_id(&self, secret_name: &str) -> String {
        // Parse: projects/{project}/secrets/{secret_id}/versions/{version}
        secret_name
            .split('/')
            .nth(3)
            .unwrap_or(secret_name)
            .to_string()
    }

    /// Check if running in Google Cloud environment
    pub fn is_cloud_environment() -> bool {
        env::var("K_SERVICE").is_ok() // Cloud Run
            || env::var("GOOGLE_CLOUD_PROJECT").is_ok()
            || env::var("GCP_PROJECT").is_ok()
    }

    /// Get the current project ID
    pub fn project_id(&self) -> &str {
        &self.project_id
    }

    /// Clear cached credentials (for testing)
    pub fn clear_cache(&mut self) {
        self.cached_gemini_key = None;
    }
}

/// Configuration for external services
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// π.ruv.io base URL
    pub pi_ruvio_url: String,

    /// Gemini model to use
    pub gemini_model: String,

    /// Maximum Gemini tokens
    pub gemini_max_tokens: usize,

    /// Gemini temperature
    pub gemini_temperature: f32,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            pi_ruvio_url: "https://pi.ruv.io".to_string(),
            gemini_model: "gemini-2.5-flash-preview-05-20".to_string(),
            gemini_max_tokens: 4096,
            gemini_temperature: 0.3,
        }
    }
}

impl ServiceConfig {
    /// Create config from environment
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(url) = env::var("PI_RUVIO_URL") {
            config.pi_ruvio_url = url;
        }

        if let Ok(model) = env::var("GEMINI_MODEL") {
            config.gemini_model = model;
        }

        if let Ok(tokens) = env::var("GEMINI_MAX_TOKENS") {
            if let Ok(t) = tokens.parse() {
                config.gemini_max_tokens = t;
            }
        }

        if let Ok(temp) = env::var("GEMINI_TEMPERATURE") {
            if let Ok(t) = temp.parse() {
                config.gemini_temperature = t;
            }
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_manager_creation() {
        let manager = SecretManager::new("test-project");
        assert_eq!(manager.project_id(), "test-project");
    }

    #[test]
    fn test_extract_secret_id() {
        let manager = SecretManager::new("test-project");
        let full_name = "projects/test-project/secrets/my-secret/versions/latest";
        let id = manager.extract_secret_id(full_name);
        assert_eq!(id, "my-secret");
    }

    #[test]
    fn test_service_config_default() {
        let config = ServiceConfig::default();
        assert_eq!(config.pi_ruvio_url, "https://pi.ruv.io");
        assert!(config.gemini_model.contains("gemini"));
    }
}
