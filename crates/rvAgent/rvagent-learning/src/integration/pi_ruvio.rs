//! π.ruv.io cloud brain client

use crate::discovery::{DiscoveryCategory, DiscoveryLog};
use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Client for submitting discoveries to π.ruv.io
pub struct PiRuvIoClient {
    /// HTTP client
    client: Client,

    /// Base URL
    base_url: String,

    /// Submission queue
    queue: Arc<RwLock<VecDeque<QueuedSubmission>>>,

    /// Retry configuration
    max_retries: usize,

    /// Backoff base (milliseconds)
    backoff_base_ms: u64,
}

/// A queued submission
#[derive(Debug, Clone)]
struct QueuedSubmission {
    /// The discovery to submit
    discovery: DiscoveryLog,

    /// Number of attempts made
    attempts: usize,

    /// Last error message
    last_error: Option<String>,
}

/// Request to submit a memory to π.ruv.io
#[derive(Debug, Clone, Serialize)]
pub struct BrainShareRequest {
    /// Knowledge category
    pub category: String,

    /// Short title
    pub title: String,

    /// Content
    pub content: String,

    /// Code snippet
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_snippet: Option<String>,

    /// Tags
    pub tags: Vec<String>,
}

/// Response from brain_share
#[derive(Debug, Clone, Deserialize)]
pub struct BrainShareResponse {
    /// Whether the request succeeded
    pub success: bool,

    /// Memory ID if successful
    pub memory_id: Option<String>,

    /// Error message if failed
    pub error: Option<String>,
}

/// Status of a submission
#[derive(Debug, Clone, PartialEq)]
pub enum SubmissionStatus {
    /// Waiting to be submitted
    Pending,

    /// Submission in progress
    InProgress,

    /// Successfully accepted
    Accepted { memory_id: String },

    /// Rejected by the server
    Rejected { reason: String },

    /// Failed after retries
    Failed { reason: String },
}

impl PiRuvIoClient {
    /// Create a new client
    pub fn new(base_url: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.into(),
            queue: Arc::new(RwLock::new(VecDeque::new())),
            max_retries: 3,
            backoff_base_ms: 500,
        }
    }

    /// Create client with default π.ruv.io URL
    pub fn default_client() -> Self {
        Self::new("https://pi.ruv.io")
    }

    /// Check if connected to π.ruv.io
    pub async fn check_connection(&self) -> bool {
        match self.client
            .get(format!("{}/health", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Queue a discovery for submission
    pub async fn queue_submission(&self, discovery: DiscoveryLog) -> Uuid {
        let id = discovery.id;
        let mut queue = self.queue.write().await;
        queue.push_back(QueuedSubmission {
            discovery,
            attempts: 0,
            last_error: None,
        });
        id
    }

    /// Submit a discovery directly (bypassing queue)
    pub async fn submit(&self, discovery: &DiscoveryLog) -> Result<BrainShareResponse> {
        let request = self.discovery_to_request(discovery);

        let response = self.client
            .post(format!("{}/api/v1/brain/share", self.base_url))
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            let body: BrainShareResponse = response.json().await?;
            Ok(body)
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            Err(anyhow!("π.ruv.io returned {}: {}", status, text))
        }
    }

    /// Process the submission queue
    pub async fn process_queue(&self) -> Vec<(Uuid, SubmissionStatus)> {
        let mut results = Vec::new();

        loop {
            // Get next item from queue
            let item = {
                let mut queue = self.queue.write().await;
                queue.pop_front()
            };

            let mut submission = match item {
                Some(s) => s,
                None => break,
            };

            submission.attempts += 1;

            match self.submit(&submission.discovery).await {
                Ok(response) if response.success => {
                    let status = SubmissionStatus::Accepted {
                        memory_id: response.memory_id.unwrap_or_default(),
                    };
                    results.push((submission.discovery.id, status));
                }
                Ok(response) => {
                    let reason = response.error.unwrap_or_else(|| "Unknown error".to_string());
                    let status = SubmissionStatus::Rejected { reason };
                    results.push((submission.discovery.id, status));
                }
                Err(e) => {
                    submission.last_error = Some(e.to_string());

                    if submission.attempts < self.max_retries {
                        // Requeue with exponential backoff
                        let delay = Duration::from_millis(
                            self.backoff_base_ms * 2u64.pow(submission.attempts as u32 - 1)
                        );
                        tokio::time::sleep(delay).await;

                        let mut queue = self.queue.write().await;
                        queue.push_back(submission);
                    } else {
                        let status = SubmissionStatus::Failed {
                            reason: e.to_string(),
                        };
                        results.push((submission.discovery.id, status));
                    }
                }
            }
        }

        results
    }

    /// Get queue size
    pub async fn queue_size(&self) -> usize {
        self.queue.read().await.len()
    }

    /// Convert discovery to brain_share request
    fn discovery_to_request(&self, discovery: &DiscoveryLog) -> BrainShareRequest {
        let category = match discovery.category {
            DiscoveryCategory::Optimization => "performance",
            DiscoveryCategory::Security => "security",
            DiscoveryCategory::Performance => "performance",
            DiscoveryCategory::Architecture => "architecture",
            DiscoveryCategory::Testing => "pattern",
            DiscoveryCategory::ErrorHandling => "pattern",
            DiscoveryCategory::ApiDesign => "architecture",
            DiscoveryCategory::Documentation => "pattern",
            DiscoveryCategory::Other => "pattern",
        };

        let mut content = discovery.description.clone();

        // Add method attribution
        content.push_str("\n\n## Method Attribution\n");
        content.push_str(&discovery.method_attribution());

        // Add source files
        if !discovery.source_files.is_empty() {
            content.push_str("\n\n## Source Files\n");
            for file in &discovery.source_files {
                content.push_str(&format!("- {}\n", file.display()));
            }
        }

        BrainShareRequest {
            category: category.to_string(),
            title: discovery.title.clone(),
            content,
            code_snippet: discovery.code_snippet.clone(),
            tags: discovery.tags.clone(),
        }
    }

    /// Search for similar memories in π.ruv.io
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let response = self.client
            .get(format!("{}/api/v1/brain/search", self.base_url))
            .query(&[("query", query), ("limit", &limit.to_string())])
            .send()
            .await?;

        if response.status().is_success() {
            let results: SearchResponse = response.json().await?;
            Ok(results.memories)
        } else {
            Err(anyhow!("Search failed: {}", response.status()))
        }
    }
}

/// Search response from π.ruv.io
#[derive(Debug, Deserialize)]
struct SearchResponse {
    memories: Vec<SearchResult>,
}

/// A search result from π.ruv.io
#[derive(Debug, Clone, Deserialize)]
pub struct SearchResult {
    /// Memory ID
    pub id: String,

    /// Title
    pub title: String,

    /// Category
    pub category: String,

    /// Quality score
    pub quality: f64,

    /// Similarity score
    pub similarity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = PiRuvIoClient::new("https://test.pi.ruv.io");
        assert_eq!(client.base_url, "https://test.pi.ruv.io");
    }

    #[test]
    fn test_discovery_to_request() {
        let client = PiRuvIoClient::default_client();

        let discovery = DiscoveryLog::new(
            DiscoveryCategory::Security,
            "Input Validation Pattern",
            "Discovered a robust input validation pattern",
        )
        .with_tags(vec!["security".to_string(), "validation".to_string()]);

        let request = client.discovery_to_request(&discovery);

        assert_eq!(request.category, "security");
        assert_eq!(request.title, "Input Validation Pattern");
        assert!(request.content.contains("Method Attribution"));
    }

    #[tokio::test]
    async fn test_queue_submission() {
        let client = PiRuvIoClient::default_client();

        let discovery = DiscoveryLog::new(
            DiscoveryCategory::Performance,
            "Test",
            "Test discovery",
        );

        let id = client.queue_submission(discovery).await;
        assert_eq!(client.queue_size().await, 1);
        assert!(!id.is_nil());
    }
}
