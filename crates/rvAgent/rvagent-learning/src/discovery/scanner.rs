//! Codebase scanner for pattern discovery

use anyhow::Result;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Codebase scanner for discovering patterns
pub struct CodebaseScanner {
    /// Root directory to scan
    root: PathBuf,

    /// File extensions to include
    extensions: Vec<String>,

    /// Directories to exclude
    exclude_dirs: Vec<String>,

    /// Maximum file size to analyze (bytes)
    max_file_size: u64,
}

/// A scanned file with metadata
#[derive(Debug, Clone)]
pub struct ScannedFile {
    /// File path relative to root
    pub path: PathBuf,

    /// File content
    pub content: String,

    /// File size in bytes
    pub size: u64,

    /// Language/type inferred from extension
    pub language: Option<String>,
}

impl CodebaseScanner {
    /// Create a new scanner for a directory
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            extensions: vec![
                "rs".to_string(),
                "ts".to_string(),
                "js".to_string(),
                "py".to_string(),
                "go".to_string(),
                "java".to_string(),
                "c".to_string(),
                "cpp".to_string(),
                "h".to_string(),
            ],
            exclude_dirs: vec![
                "node_modules".to_string(),
                "target".to_string(),
                ".git".to_string(),
                "dist".to_string(),
                "build".to_string(),
                "__pycache__".to_string(),
                ".svelte-kit".to_string(),
            ],
            max_file_size: 1024 * 1024, // 1MB
        }
    }

    /// Set file extensions to include
    pub fn with_extensions(mut self, extensions: Vec<String>) -> Self {
        self.extensions = extensions;
        self
    }

    /// Set directories to exclude
    pub fn with_exclude_dirs(mut self, dirs: Vec<String>) -> Self {
        self.exclude_dirs = dirs;
        self
    }

    /// Set maximum file size
    pub fn with_max_file_size(mut self, size: u64) -> Self {
        self.max_file_size = size;
        self
    }

    /// Scan the codebase and return files
    pub async fn scan(&self) -> Result<Vec<ScannedFile>> {
        let mut files = Vec::new();
        self.scan_dir(&self.root, &mut files).await?;
        Ok(files)
    }

    /// Recursively scan a directory
    async fn scan_dir(&self, dir: &Path, files: &mut Vec<ScannedFile>) -> Result<()> {
        let mut entries = fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            // Skip excluded directories
            if path.is_dir() {
                if self.exclude_dirs.iter().any(|d| name == *d) {
                    continue;
                }
                Box::pin(self.scan_dir(&path, files)).await?;
                continue;
            }

            // Check if file matches criteria
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_string();
                if !self.extensions.contains(&ext_str) {
                    continue;
                }
            } else {
                continue;
            }

            // Check file size
            let metadata = entry.metadata().await?;
            if metadata.len() > self.max_file_size {
                continue;
            }

            // Read file content
            if let Ok(content) = fs::read_to_string(&path).await {
                let relative_path = path
                    .strip_prefix(&self.root)
                    .unwrap_or(&path)
                    .to_path_buf();

                let language = path
                    .extension()
                    .map(|e| e.to_string_lossy().to_string());

                files.push(ScannedFile {
                    path: relative_path,
                    content,
                    size: metadata.len(),
                    language,
                });
            }
        }

        Ok(())
    }

    /// Count total files that would be scanned
    pub async fn count_files(&self) -> Result<usize> {
        let files = self.scan().await?;
        Ok(files.len())
    }
}

impl ScannedFile {
    /// Get line count
    pub fn line_count(&self) -> usize {
        self.content.lines().count()
    }

    /// Check if file contains a pattern
    pub fn contains(&self, pattern: &str) -> bool {
        self.content.contains(pattern)
    }

    /// Extract functions/methods (simplified)
    pub fn extract_functions(&self) -> Vec<String> {
        let mut functions = Vec::new();

        match self.language.as_deref() {
            Some("rs") => {
                // Simple Rust function extraction
                for line in self.content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("pub fn ") || trimmed.starts_with("fn ") {
                        if let Some(name_end) = trimmed.find('(') {
                            let start = if trimmed.starts_with("pub fn ") { 7 } else { 3 };
                            let name = &trimmed[start..name_end];
                            functions.push(name.to_string());
                        }
                    }
                }
            }
            Some("ts") | Some("js") => {
                // Simple TypeScript/JavaScript function extraction
                for line in self.content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("function ") {
                        if let Some(name_end) = trimmed.find('(') {
                            let name = &trimmed[9..name_end];
                            functions.push(name.to_string());
                        }
                    } else if trimmed.contains("= (") || trimmed.contains("= async (") {
                        if let Some(eq_pos) = trimmed.find('=') {
                            let before = trimmed[..eq_pos].trim();
                            if let Some(name) = before.split_whitespace().last() {
                                functions.push(name.to_string());
                            }
                        }
                    }
                }
            }
            Some("py") => {
                // Simple Python function extraction
                for line in self.content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("def ") {
                        if let Some(name_end) = trimmed.find('(') {
                            let name = &trimmed[4..name_end];
                            functions.push(name.to_string());
                        }
                    }
                }
            }
            _ => {}
        }

        functions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scanner_creation() {
        let scanner = CodebaseScanner::new("./src")
            .with_extensions(vec!["rs".to_string()])
            .with_max_file_size(500_000);

        assert_eq!(scanner.extensions, vec!["rs"]);
        assert_eq!(scanner.max_file_size, 500_000);
    }

    #[test]
    fn test_scanned_file_functions() {
        let file = ScannedFile {
            path: PathBuf::from("test.rs"),
            content: r#"
pub fn hello() {
    println!("Hello");
}

fn private_fn() {
    // ...
}
"#.to_string(),
            size: 100,
            language: Some("rs".to_string()),
        };

        let functions = file.extract_functions();
        assert_eq!(functions.len(), 2);
        assert!(functions.contains(&"hello".to_string()));
        assert!(functions.contains(&"private_fn".to_string()));
    }
}
