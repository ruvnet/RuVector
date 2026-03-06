#!/usr/bin/env -S cargo +nightly -Zscript
//! Seed the RuVector Shared Brain with knowledge from this repository.
//!
//! Usage:
//!   BRAIN_URL=https://brain.ruv.io BRAIN_API_KEY=your-key cargo +nightly -Zscript scripts/seed-brain.rs
//!
//! Or with a local server:
//!   BRAIN_URL=http://localhost:8080 cargo +nightly -Zscript scripts/seed-brain.rs

//! ```cargo
//! [dependencies]
//! reqwest = { version = "0.12", features = ["json", "rustls-tls", "blocking"] }
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = "1.0"
//! sha3 = "0.10"
//! walkdir = "2"
//! regex = "1"
//! ```

use reqwest::blocking::Client;
use serde::Serialize;
use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
use std::path::Path;
use walkdir::WalkDir;

#[derive(Serialize)]
struct ShareRequest {
    category: String,
    title: String,
    content: String,
    tags: Vec<String>,
    code_snippet: Option<String>,
    embedding: Vec<f32>,
    witness_hash: String,
}

fn main() {
    let base_url = std::env::var("BRAIN_URL")
        .unwrap_or_else(|_| "https://brain.ruv.io".to_string());
    let api_key = std::env::var("BRAIN_API_KEY")
        .unwrap_or_else(|_| "ruvector-seed-key".to_string());

    println!("=== RuVector Shared Brain Seeder ===");
    println!("Backend: {base_url}");

    let client = Client::new();
    let mut count = 0;

    // 1. Seed ADRs
    println!("\n--- Seeding ADRs ---");
    let adr_path = Path::new("docs/adr");
    if adr_path.exists() {
        for entry in WalkDir::new(adr_path)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
        {
            let path = entry.path();
            let filename = path.file_stem().unwrap_or_default().to_string_lossy();
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    let title = extract_title(&content).unwrap_or_else(|| filename.to_string());
                    let tags = extract_adr_tags(&content);
                    let truncated = truncate(&content, 10000);
                    if let Err(e) = seed_memory(
                        &client,
                        &base_url,
                        &api_key,
                        "architecture",
                        &title,
                        &truncated,
                        &tags,
                    ) {
                        eprintln!("  Failed to seed {filename}: {e}");
                    } else {
                        println!("  Seeded: {title}");
                        count += 1;
                    }
                }
                Err(e) => eprintln!("  Failed to read {}: {e}", path.display()),
            }
        }
    }

    // 2. Seed crate READMEs
    println!("\n--- Seeding Crate READMEs ---");
    for entry in WalkDir::new("crates")
        .max_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name() == "README.md")
    {
        let path = entry.path();
        let crate_name = path
            .parent()
            .and_then(|p| p.file_name())
            .unwrap_or_default()
            .to_string_lossy();
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let title = format!("Crate: {crate_name}");
                let truncated = truncate(&content, 10000);
                if let Err(e) = seed_memory(
                    &client,
                    &base_url,
                    &api_key,
                    "convention",
                    &title,
                    &truncated,
                    &[crate_name.to_string(), "readme".to_string()],
                ) {
                    eprintln!("  Failed to seed {crate_name}: {e}");
                } else {
                    println!("  Seeded: {title}");
                    count += 1;
                }
            }
            Err(e) => eprintln!("  Failed to read {}: {e}", path.display()),
        }
    }

    // 3. Seed lib.rs doc comments
    println!("\n--- Seeding lib.rs Patterns ---");
    for entry in WalkDir::new("crates")
        .max_depth(3)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name() == "lib.rs")
    {
        let path = entry.path();
        let crate_name = path
            .ancestors()
            .nth(2)
            .and_then(|p| p.file_name())
            .unwrap_or_default()
            .to_string_lossy();
        match std::fs::read_to_string(path) {
            Ok(content) => {
                if let Some(doc_comment) = extract_doc_comment(&content) {
                    if doc_comment.len() > 50 {
                        let title = format!("Pattern: {crate_name}");
                        let truncated = truncate(&doc_comment, 10000);
                        if let Err(e) = seed_memory(
                            &client,
                            &base_url,
                            &api_key,
                            "pattern",
                            &title,
                            &truncated,
                            &[crate_name.to_string(), "pattern".to_string()],
                        ) {
                            eprintln!("  Failed to seed {crate_name}: {e}");
                        } else {
                            println!("  Seeded: {title}");
                            count += 1;
                        }
                    }
                }
            }
            Err(_) => {}
        }
    }

    // 4. Seed example READMEs
    println!("\n--- Seeding Example Solutions ---");
    for entry in WalkDir::new("examples")
        .max_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name() == "README.md")
    {
        let path = entry.path();
        let example_name = path
            .parent()
            .and_then(|p| p.file_name())
            .unwrap_or_default()
            .to_string_lossy();
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let title = format!("Example: {example_name}");
                let truncated = truncate(&content, 10000);
                if let Err(e) = seed_memory(
                    &client,
                    &base_url,
                    &api_key,
                    "solution",
                    &title,
                    &truncated,
                    &[example_name.to_string(), "example".to_string()],
                ) {
                    eprintln!("  Failed to seed {example_name}: {e}");
                } else {
                    println!("  Seeded: {title}");
                    count += 1;
                }
            }
            Err(_) => {}
        }
    }

    println!("\n=== Seeding complete: {count} memories shared ===");
}

fn seed_memory(
    client: &Client,
    base_url: &str,
    api_key: &str,
    category: &str,
    title: &str,
    content: &str,
    tags: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let embedding = hash_embedding(content);
    let witness_hash = witness_hash(&["pii_strip", "embed", "share"]);

    let req = ShareRequest {
        category: category.to_string(),
        title: title.to_string(),
        content: content.to_string(),
        tags: tags.to_vec(),
        code_snippet: None,
        embedding,
        witness_hash,
    };

    let resp = client
        .post(&format!("{base_url}/v1/memories"))
        .bearer_auth(api_key)
        .json(&req)
        .send()?;

    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("HTTP {}: {}", resp.status(), resp.text()?).into())
    }
}

fn hash_embedding(text: &str) -> Vec<f32> {
    let mut hasher = Shake256::default();
    hasher.update(b"ruvector-brain-embed:");
    hasher.update(text.as_bytes());
    let mut reader = hasher.finalize_xof();
    let mut buf = [0u8; 512];
    reader.read(&mut buf);
    let emb: Vec<f32> = buf
        .chunks(4)
        .map(|chunk| {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let raw = f32::from_le_bytes(bytes);
            (raw.rem_euclid(2.0) - 1.0).clamp(-1.0, 1.0)
        })
        .collect();
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        emb.iter().map(|x| x / norm).collect()
    } else {
        emb
    }
}

fn witness_hash(ops: &[&str]) -> String {
    let mut hasher = Shake256::default();
    for op in ops {
        hasher.update(op.as_bytes());
        hasher.update(b"|");
    }
    let mut reader = hasher.finalize_xof();
    let mut buf = [0u8; 32];
    reader.read(&mut buf);
    hex::encode(buf)
}

fn extract_title(content: &str) -> Option<String> {
    content
        .lines()
        .find(|l| l.starts_with("# "))
        .map(|l| l.trim_start_matches("# ").trim().to_string())
}

fn extract_adr_tags(content: &str) -> Vec<String> {
    let mut tags = vec!["adr".to_string()];
    if content.contains("security") || content.contains("Security") {
        tags.push("security".to_string());
    }
    if content.contains("performance") || content.contains("Performance") {
        tags.push("performance".to_string());
    }
    if content.contains("federation") || content.contains("Federation") {
        tags.push("federation".to_string());
    }
    tags
}

fn extract_doc_comment(content: &str) -> Option<String> {
    let lines: Vec<&str> = content
        .lines()
        .take_while(|l| l.starts_with("//!") || l.is_empty())
        .filter(|l| l.starts_with("//!"))
        .map(|l| l.trim_start_matches("//!").trim_start())
        .collect();
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}... [truncated]", &s[..max])
    }
}
