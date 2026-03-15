//! MCP resource system — providers, registry, and content types.
//!
//! Resources are read-only data sources that MCP servers expose to clients.
//! This module provides [`ResourceProvider`] for pluggable implementations,
//! [`ResourceRegistry`] for managing providers, and concrete providers
//! for static content, file system, and URI templates.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;

use crate::protocol::{McpResource, McpResourceTemplate, ResourceContent, ResourceReadResult};
use crate::{McpError, Result};

// ---------------------------------------------------------------------------
// ResourceUri
// ---------------------------------------------------------------------------

/// Parsed MCP resource URI.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceUri {
    /// The full URI string.
    pub uri: String,
    /// Scheme (e.g. "file", "memory", "template").
    pub scheme: String,
    /// Path component after `://`.
    pub path: String,
}

impl ResourceUri {
    /// Parse a URI string into components.
    pub fn parse(uri: &str) -> Result<Self> {
        if let Some((scheme, rest)) = uri.split_once("://") {
            Ok(Self {
                uri: uri.to_string(),
                scheme: scheme.to_string(),
                path: rest.to_string(),
            })
        } else {
            Ok(Self {
                uri: uri.to_string(),
                scheme: "file".to_string(),
                path: uri.to_string(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// ResourceProvider trait
// ---------------------------------------------------------------------------

/// Async provider for MCP resources.
#[async_trait]
pub trait ResourceProvider: Send + Sync {
    /// Unique scheme this provider handles (e.g. "file", "memory").
    fn scheme(&self) -> &str;

    /// List all resources this provider can serve.
    async fn list(&self) -> Result<Vec<McpResource>>;

    /// Read the content of a resource by URI.
    async fn read(&self, uri: &str) -> Result<ResourceReadResult>;

    /// URI templates this provider supports (if any).
    fn templates(&self) -> Vec<McpResourceTemplate> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// StaticResourceProvider
// ---------------------------------------------------------------------------

/// In-memory static resource provider.
pub struct StaticResourceProvider {
    resources: DashMap<String, StaticEntry>,
}

struct StaticEntry {
    name: String,
    description: Option<String>,
    mime_type: Option<String>,
    content: String,
}

impl StaticResourceProvider {
    /// Create an empty static resource provider.
    pub fn new() -> Self {
        Self {
            resources: DashMap::new(),
        }
    }

    /// Add a text resource.
    pub fn add(
        &self,
        uri: &str,
        name: &str,
        content: &str,
        mime_type: Option<&str>,
        description: Option<&str>,
    ) {
        self.resources.insert(
            uri.to_string(),
            StaticEntry {
                name: name.to_string(),
                description: description.map(|s| s.to_string()),
                mime_type: mime_type.map(|s| s.to_string()),
                content: content.to_string(),
            },
        );
    }

    /// Remove a resource.
    pub fn remove(&self, uri: &str) -> bool {
        self.resources.remove(uri).is_some()
    }

    /// Number of stored resources.
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Whether provider has no resources.
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

impl Default for StaticResourceProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ResourceProvider for StaticResourceProvider {
    fn scheme(&self) -> &str {
        "memory"
    }

    async fn list(&self) -> Result<Vec<McpResource>> {
        let mut resources: Vec<_> = self
            .resources
            .iter()
            .map(|r| McpResource {
                uri: r.key().clone(),
                name: r.value().name.clone(),
                description: r.value().description.clone(),
                mime_type: r.value().mime_type.clone(),
            })
            .collect();
        resources.sort_by(|a, b| a.uri.cmp(&b.uri));
        Ok(resources)
    }

    async fn read(&self, uri: &str) -> Result<ResourceReadResult> {
        let entry = self
            .resources
            .get(uri)
            .ok_or_else(|| McpError::resource(format!("resource not found: {}", uri)))?;
        Ok(ResourceReadResult {
            contents: vec![ResourceContent {
                uri: uri.to_string(),
                mime_type: entry.mime_type.clone(),
                text: Some(entry.content.clone()),
                blob: None,
            }],
        })
    }
}

// ---------------------------------------------------------------------------
// FileResourceProvider
// ---------------------------------------------------------------------------

/// File-system based resource provider.
pub struct FileResourceProvider {
    base_dir: String,
    files: DashMap<String, FileEntry>,
}

struct FileEntry {
    name: String,
    path: String,
    mime_type: Option<String>,
    description: Option<String>,
}

impl FileResourceProvider {
    /// Create a new file resource provider with the given base directory.
    pub fn new(base_dir: impl Into<String>) -> Self {
        Self {
            base_dir: base_dir.into(),
            files: DashMap::new(),
        }
    }

    /// Register a file resource.
    pub fn register(
        &self,
        uri: &str,
        name: &str,
        relative_path: &str,
        mime_type: Option<&str>,
        description: Option<&str>,
    ) {
        self.files.insert(
            uri.to_string(),
            FileEntry {
                name: name.to_string(),
                path: relative_path.to_string(),
                mime_type: mime_type.map(|s| s.to_string()),
                description: description.map(|s| s.to_string()),
            },
        );
    }

    /// Base directory path.
    pub fn base_dir(&self) -> &str {
        &self.base_dir
    }
}

#[async_trait]
impl ResourceProvider for FileResourceProvider {
    fn scheme(&self) -> &str {
        "file"
    }

    async fn list(&self) -> Result<Vec<McpResource>> {
        let mut resources: Vec<_> = self
            .files
            .iter()
            .map(|r| McpResource {
                uri: r.key().clone(),
                name: r.value().name.clone(),
                description: r.value().description.clone(),
                mime_type: r.value().mime_type.clone(),
            })
            .collect();
        resources.sort_by(|a, b| a.uri.cmp(&b.uri));
        Ok(resources)
    }

    async fn read(&self, uri: &str) -> Result<ResourceReadResult> {
        let entry = self
            .files
            .get(uri)
            .ok_or_else(|| McpError::resource(format!("resource not found: {}", uri)))?;

        if entry.path.contains("..") {
            return Err(McpError::resource("path traversal not allowed"));
        }

        let full_path = format!("{}/{}", self.base_dir, entry.path);
        let content = tokio::fs::read_to_string(&full_path)
            .await
            .map_err(|e| McpError::resource(format!("failed to read {}: {}", full_path, e)))?;

        Ok(ResourceReadResult {
            contents: vec![ResourceContent {
                uri: uri.to_string(),
                mime_type: entry.mime_type.clone(),
                text: Some(content),
                blob: None,
            }],
        })
    }

    fn templates(&self) -> Vec<McpResourceTemplate> {
        vec![McpResourceTemplate {
            uri_template: format!("file://{}//{{path}}", self.base_dir),
            name: "file".into(),
            description: Some("Read a file from the base directory".into()),
            mime_type: None,
        }]
    }
}

// ---------------------------------------------------------------------------
// TemplateResourceProvider
// ---------------------------------------------------------------------------

/// URI-template based resource provider with dynamic resolution.
pub struct TemplateResourceProvider {
    templates: Vec<TemplateEntry>,
    resolver: Arc<dyn TemplateResolver>,
}

struct TemplateEntry {
    uri_template: String,
    name: String,
    description: Option<String>,
    mime_type: Option<String>,
}

/// Resolves template parameters into resource content.
#[async_trait]
pub trait TemplateResolver: Send + Sync {
    /// Resolve a template URI with the given parameters.
    async fn resolve(
        &self,
        template: &str,
        params: &HashMap<String, String>,
    ) -> Result<ResourceReadResult>;
}

impl TemplateResourceProvider {
    /// Create a new template resource provider.
    pub fn new(resolver: Arc<dyn TemplateResolver>) -> Self {
        Self {
            templates: Vec::new(),
            resolver,
        }
    }

    /// Add a URI template.
    pub fn add_template(
        &mut self,
        uri_template: &str,
        name: &str,
        description: Option<&str>,
        mime_type: Option<&str>,
    ) {
        self.templates.push(TemplateEntry {
            uri_template: uri_template.to_string(),
            name: name.to_string(),
            description: description.map(|s| s.to_string()),
            mime_type: mime_type.map(|s| s.to_string()),
        });
    }
}

#[async_trait]
impl ResourceProvider for TemplateResourceProvider {
    fn scheme(&self) -> &str {
        "template"
    }

    async fn list(&self) -> Result<Vec<McpResource>> {
        Ok(vec![])
    }

    async fn read(&self, uri: &str) -> Result<ResourceReadResult> {
        let params = HashMap::new();
        self.resolver.resolve(uri, &params).await
    }

    fn templates(&self) -> Vec<McpResourceTemplate> {
        self.templates
            .iter()
            .map(|t| McpResourceTemplate {
                uri_template: t.uri_template.clone(),
                name: t.name.clone(),
                description: t.description.clone(),
                mime_type: t.mime_type.clone(),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ResourceRegistry
// ---------------------------------------------------------------------------

/// Central registry managing multiple resource providers.
pub struct ResourceRegistry {
    providers: Vec<Arc<dyn ResourceProvider>>,
}

impl ResourceRegistry {
    /// Create an empty resource registry.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Register a resource provider.
    pub fn register(&mut self, provider: Arc<dyn ResourceProvider>) {
        self.providers.push(provider);
    }

    /// List all resources from all providers.
    pub async fn list_resources(&self) -> Result<Vec<McpResource>> {
        let mut all = Vec::new();
        for provider in &self.providers {
            let resources = provider.list().await?;
            all.extend(resources);
        }
        all.sort_by(|a, b| a.uri.cmp(&b.uri));
        Ok(all)
    }

    /// List all templates from all providers.
    pub fn list_templates(&self) -> Vec<McpResourceTemplate> {
        self.providers.iter().flat_map(|p| p.templates()).collect()
    }

    /// Read a resource by URI, trying each provider.
    pub async fn read_resource(&self, uri: &str) -> Result<ResourceReadResult> {
        for provider in &self.providers {
            match provider.read(uri).await {
                Ok(result) => return Ok(result),
                Err(_) => continue,
            }
        }
        Err(McpError::resource(format!(
            "no provider can serve: {}",
            uri
        )))
    }

    /// Number of registered providers.
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }
}

impl Default for ResourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_uri_parse_with_scheme() {
        let uri = ResourceUri::parse("memory://data/key1").unwrap();
        assert_eq!(uri.scheme, "memory");
        assert_eq!(uri.path, "data/key1");
    }

    #[test]
    fn test_resource_uri_parse_bare_path() {
        let uri = ResourceUri::parse("/tmp/file.txt").unwrap();
        assert_eq!(uri.scheme, "file");
        assert_eq!(uri.path, "/tmp/file.txt");
    }

    #[test]
    fn test_resource_uri_equality() {
        let a = ResourceUri::parse("memory://a").unwrap();
        let b = ResourceUri::parse("memory://a").unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn test_static_provider_add_and_list() {
        let p = StaticResourceProvider::new();
        p.add("memory://k1", "key1", "value1", Some("text/plain"), None);
        p.add("memory://k2", "key2", "value2", None, Some("desc"));
        let list = p.list().await.unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].uri, "memory://k1");
    }

    #[tokio::test]
    async fn test_static_provider_read() {
        let p = StaticResourceProvider::new();
        p.add("memory://doc", "doc", "hello world", Some("text/plain"), None);
        let result = p.read("memory://doc").await.unwrap();
        assert_eq!(result.contents.len(), 1);
        assert_eq!(result.contents[0].text.as_deref(), Some("hello world"));
    }

    #[tokio::test]
    async fn test_static_provider_read_not_found() {
        let p = StaticResourceProvider::new();
        let err = p.read("memory://missing").await;
        assert!(err.is_err());
    }

    #[test]
    fn test_static_provider_remove() {
        let p = StaticResourceProvider::new();
        p.add("memory://x", "x", "data", None, None);
        assert_eq!(p.len(), 1);
        assert!(p.remove("memory://x"));
        assert!(p.is_empty());
        assert!(!p.remove("memory://x"));
    }

    #[test]
    fn test_static_provider_len() {
        let p = StaticResourceProvider::new();
        assert!(p.is_empty());
        p.add("memory://a", "a", "aa", None, None);
        assert_eq!(p.len(), 1);
    }

    #[tokio::test]
    async fn test_file_provider_list() {
        let p = FileResourceProvider::new("/tmp");
        p.register(
            "file:///tmp/test.txt",
            "test",
            "test.txt",
            Some("text/plain"),
            None,
        );
        let list = p.list().await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "test");
    }

    #[test]
    fn test_file_provider_base_dir() {
        let p = FileResourceProvider::new("/srv/data");
        assert_eq!(p.base_dir(), "/srv/data");
    }

    #[test]
    fn test_file_provider_templates() {
        let p = FileResourceProvider::new("/base");
        let templates = p.templates();
        assert_eq!(templates.len(), 1);
        assert!(templates[0].uri_template.contains("/base"));
    }

    #[tokio::test]
    async fn test_file_provider_path_traversal_blocked() {
        let p = FileResourceProvider::new("/tmp");
        p.register("file:///evil", "evil", "../../etc/passwd", None, None);
        let err = p.read("file:///evil").await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn test_registry_empty() {
        let reg = ResourceRegistry::new();
        let list = reg.list_resources().await.unwrap();
        assert!(list.is_empty());
        assert_eq!(reg.provider_count(), 0);
    }

    #[tokio::test]
    async fn test_registry_with_static_provider() {
        let sp = Arc::new(StaticResourceProvider::new());
        sp.add("memory://a", "a", "aaa", None, None);
        sp.add("memory://b", "b", "bbb", None, None);

        let mut reg = ResourceRegistry::new();
        reg.register(sp);
        assert_eq!(reg.provider_count(), 1);

        let list = reg.list_resources().await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn test_registry_read_resource() {
        let sp = Arc::new(StaticResourceProvider::new());
        sp.add("memory://doc", "doc", "content", None, None);

        let mut reg = ResourceRegistry::new();
        reg.register(sp);

        let result = reg.read_resource("memory://doc").await.unwrap();
        assert_eq!(result.contents[0].text.as_deref(), Some("content"));
    }

    #[tokio::test]
    async fn test_registry_read_not_found() {
        let sp = Arc::new(StaticResourceProvider::new());
        let mut reg = ResourceRegistry::new();
        reg.register(sp);

        let err = reg.read_resource("memory://missing").await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn test_registry_list_templates() {
        let fp = Arc::new(FileResourceProvider::new("/base"));
        let mut reg = ResourceRegistry::new();
        reg.register(fp);
        let templates = reg.list_templates();
        assert_eq!(templates.len(), 1);
    }

    #[tokio::test]
    async fn test_registry_multiple_providers() {
        let sp1 = Arc::new(StaticResourceProvider::new());
        sp1.add("memory://x", "x", "xx", None, None);
        let sp2 = Arc::new(StaticResourceProvider::new());
        sp2.add("memory://y", "y", "yy", None, None);

        let mut reg = ResourceRegistry::new();
        reg.register(sp1);
        reg.register(sp2);

        let list = reg.list_resources().await.unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(reg.provider_count(), 2);
    }

    #[test]
    fn test_registry_default() {
        let reg = ResourceRegistry::default();
        assert_eq!(reg.provider_count(), 0);
    }

    #[test]
    fn test_static_provider_default() {
        let p = StaticResourceProvider::default();
        assert!(p.is_empty());
    }
}
