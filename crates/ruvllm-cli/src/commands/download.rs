//! Model download command implementation
//!
//! Downloads models from HuggingFace Hub with progress indication,
//! supporting various quantization formats optimized for Apple Silicon.

use anyhow::{Context, Result};
use bytesize::ByteSize;
use colored::Colorize;
use console::style;
use hf_hub::api::tokio::Api;
use hf_hub::{Repo, RepoType};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

use crate::models::{get_model, resolve_model_id, resolve_weights_repo, QuantPreset};

/// Run the download command
pub async fn run(
    model: &str,
    quantization: &str,
    force: bool,
    revision: Option<&str>,
    cache_dir: &str,
) -> Result<()> {
    let quant = QuantPreset::from_str(quantization)
        .ok_or_else(|| anyhow::anyhow!("Invalid quantization format: {}", quantization))?;

    // Route quantized (GGUF) requests to the repo that actually hosts GGUF
    // weights (registry `gguf_repo` twin for safetensors-only aliases).
    let base_id = resolve_model_id(model);
    let model_id = resolve_weights_repo(model, quant);

    println!();
    println!(
        "{} {} ({})",
        style("Downloading:").bold().cyan(),
        model_id,
        quant
    );
    if model_id != base_id {
        println!(
            "  {} {} hosts safetensors only; using GGUF twin repo",
            "Note:".dimmed(),
            base_id
        );
    }
    println!();

    // Get model info if available
    if let Some(model_def) = get_model(model) {
        println!("  {} {}", "Name:".dimmed(), model_def.name);
        println!("  {} {}", "Architecture:".dimmed(), model_def.architecture);
        println!("  {} {}B", "Parameters:".dimmed(), model_def.params_b);
        println!(
            "  {} ~{:.1} GB",
            "Est. Memory:".dimmed(),
            quant.estimate_memory_gb(model_def.params_b)
        );
        println!();
    }

    // Initialize HuggingFace API
    let api = Api::new().context("Failed to initialize HuggingFace API")?;

    // Create repo reference
    let repo = if let Some(rev) = revision {
        api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            rev.to_string(),
        ))
    } else {
        api.repo(Repo::new(model_id.clone(), RepoType::Model))
    };

    // List the repo's actual files and expand the quant pattern against them
    // (previously an unexpanded glob like "*Q4_K_M.gguf" was sent literally -> 404).
    let remote_files = list_repo_files(&repo, &model_id, revision)
        .await
        .with_context(|| format!("Failed to list files for {model_id} on HuggingFace"))?;
    let files_to_download = get_files_to_download(&model_id, quant, &remote_files)?;

    // Create cache directory
    let model_cache_dir = PathBuf::from(cache_dir).join("models").join(&model_id);
    tokio::fs::create_dir_all(&model_cache_dir)
        .await
        .context("Failed to create cache directory")?;

    // Download each file
    for file_name in &files_to_download {
        // Defense-in-depth: `file_name` originates from the repo's own file
        // listing (HF tree API / siblings) and is attacker-controlled for a
        // malicious repo — reject traversal before joining into the cache.
        validate_remote_file_name(file_name)
            .with_context(|| format!("Unsafe file path in {model_id} listing"))?;
        let target_path = model_cache_dir.join(file_name);
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("Failed to create cache subdirectory")?;
            ensure_under_cache_dir(&model_cache_dir, parent).with_context(|| {
                format!("Refusing to write {file_name} outside the model cache directory")
            })?;
        }

        // Check if file exists
        if target_path.exists() && !force {
            let size = tokio::fs::metadata(&target_path).await?.len();
            println!(
                "  {} {} ({})",
                style("Cached:").green(),
                file_name,
                ByteSize(size)
            );
            continue;
        }

        println!("  {} {}", style("Downloading:").yellow(), file_name);

        // Download with progress
        let downloaded_path = download_with_progress(&repo, file_name, &model_id, revision).await?;

        // Copy to cache directory
        tokio::fs::copy(&downloaded_path, &target_path)
            .await
            .context("Failed to copy file to cache")?;

        let size = tokio::fs::metadata(&target_path).await?.len();
        println!(
            "  {} {} ({})",
            style("Downloaded:").green(),
            file_name,
            ByteSize(size)
        );
    }

    println!();
    println!(
        "{} Model ready at: {}",
        style("Success!").green().bold(),
        model_cache_dir.display()
    );
    println!();

    // Print usage hint
    println!("{}", "Quick start:".bold());
    println!("  ruvllm chat {}", model);
    println!("  ruvllm serve {}", model);
    println!();

    Ok(())
}

/// Download a file with progress indication.
///
/// Tries the `hf-hub` API first; on failure (e.g. the crate's HTTP client not
/// following HuggingFace's 307 redirect to the LFS/CDN host — observed on aux files
/// like `tokenizer_config.json` in 2.1.0), falls back to a redirect-following
/// `curl -L --fail` from the HF resolve URL. curl is already the download mechanism
/// in `hub/download.rs`, so this keeps the fix dependency-free and consistent.
async fn download_with_progress(
    repo: &hf_hub::api::tokio::ApiRepo,
    file_name: &str,
    model_id: &str,
    revision: Option<&str>,
) -> Result<PathBuf> {
    // Create progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("    [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    // Download file (hf-hub API first, curl-redirect fallback on failure)
    let path = match repo.get(file_name).await {
        Ok(p) => p,
        Err(e) => download_via_curl(model_id, revision, file_name)
            .with_context(|| format!("Failed to download {} (hf-hub: {e})", file_name))?,
    };

    pb.finish_and_clear();

    Ok(path)
}

/// Fallback download via `curl -L --fail` (follows HF's 307 redirect to the CDN).
/// Writes to a temp file and returns its path; the caller copies it into the cache.
fn download_via_curl(model_id: &str, revision: Option<&str>, file_name: &str) -> Result<PathBuf> {
    let rev = revision.unwrap_or("main");
    let url = format!("https://huggingface.co/{model_id}/resolve/{rev}/{file_name}");
    let out = std::env::temp_dir().join(format!(
        "ruvllm-dl-{}-{}",
        std::process::id(),
        file_name.replace('/', "_")
    ));
    let mut args = vec![
        "-L".to_string(),     // follow redirects (the 307 hf-hub misses)
        "--fail".to_string(), // non-zero exit on HTTP error
        "-sS".to_string(),    // quiet but show errors
        "-o".to_string(),
        out.to_string_lossy().to_string(),
    ];
    let auth_config = hf_token_curl_config();
    if auth_config.is_some() {
        args.push("--config".to_string());
        args.push("-".to_string());
    }
    args.push(url.clone());
    let output = run_curl(&args, auth_config.as_deref())
        .with_context(|| format!("curl not available to fetch {url}"))?;
    if !output.status.success() {
        anyhow::bail!("curl fallback failed ({}) for {url}", output.status);
    }
    Ok(out)
}

/// List a repo's files via the hf-hub API, falling back to a `curl` of the
/// HF tree endpoint (`GET /api/models/<id>/tree/<rev>`) — the same fallback
/// idiom as `download_via_curl`, honoring `HF_TOKEN`.
async fn list_repo_files(
    repo: &hf_hub::api::tokio::ApiRepo,
    model_id: &str,
    revision: Option<&str>,
) -> Result<Vec<String>> {
    match repo.info().await {
        Ok(info) => Ok(info.siblings.into_iter().map(|s| s.rfilename).collect()),
        Err(e) => list_files_via_curl(model_id, revision)
            .with_context(|| format!("hf-hub file listing also failed: {e}")),
    }
}

/// Fallback file listing via `curl` against the HF tree API.
fn list_files_via_curl(model_id: &str, revision: Option<&str>) -> Result<Vec<String>> {
    let rev = revision.unwrap_or("main");
    let url = format!("https://huggingface.co/api/models/{model_id}/tree/{rev}?recursive=true");
    let mut args = vec!["-L".to_string(), "--fail".to_string(), "-sS".to_string()];
    let auth_config = hf_token_curl_config();
    if auth_config.is_some() {
        args.push("--config".to_string());
        args.push("-".to_string());
    }
    args.push(url.clone());
    let output = run_curl(&args, auth_config.as_deref())
        .with_context(|| format!("curl not available to list {url}"))?;
    if !output.status.success() {
        anyhow::bail!("curl file listing failed ({}) for {url}", output.status);
    }
    parse_tree_file_paths(&String::from_utf8_lossy(&output.stdout))
}

/// Build the curl-config fragment carrying the `HF_TOKEN` Authorization
/// header, or `None` when the token is unset. Passing the header through a
/// config read from stdin (`--config -`) keeps the token off curl's argv,
/// which is world-readable via `ps`/`/proc/<pid>/cmdline` (CWE-214).
fn hf_token_curl_config() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
        .map(|t| curl_auth_config(&t))
}

/// Render an Authorization header as a curl config line, escaping the
/// characters curl's double-quoted config strings treat specially so a
/// hostile/odd token value cannot inject additional config directives.
fn curl_auth_config(token: &str) -> String {
    let escaped = token
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r");
    format!("header = \"Authorization: Bearer {escaped}\"\n")
}

/// Run `curl` with `args`, feeding `stdin_config` (a curl config carrying the
/// auth header) on stdin when present. Stderr is inherited so `-sS` errors
/// stay visible; stdout is captured for callers that parse it.
fn run_curl(args: &[String], stdin_config: Option<&str>) -> Result<std::process::Output> {
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    let mut cmd = Command::new("curl");
    cmd.args(args)
        .stdin(if stdin_config.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = cmd.spawn().context("failed to spawn curl")?;
    if let Some(config) = stdin_config {
        child
            .stdin
            .take()
            .expect("stdin is piped when a config is supplied")
            .write_all(config.as_bytes())
            .context("failed to write curl config to stdin")?;
        // stdin handle dropped here -> EOF for `--config -`
    }
    child.wait_with_output().context("failed to wait for curl")
}

/// Defense-in-depth guard for file names taken from a repo's own listing
/// (HF tree API `path` / `siblings[].rfilename`): reject empty names and any
/// component that is not a plain path segment (absolute paths, `..`, `.`,
/// Windows prefixes) before the name is joined into the cache directory.
fn validate_remote_file_name(file_name: &str) -> Result<()> {
    use std::path::Component;

    if file_name.is_empty() {
        anyhow::bail!("empty file name in repo listing");
    }
    if Path::new(file_name)
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        anyhow::bail!("suspicious file path in repo listing: {file_name:?}");
    }
    Ok(())
}

/// Verify that `target_parent` (already created) canonicalizes to a location
/// under the canonicalized model cache dir — catches anything the component
/// check missed (e.g. symlinked subdirectories escaping the cache).
fn ensure_under_cache_dir(cache_root: &Path, target_parent: &Path) -> Result<()> {
    let root = cache_root
        .canonicalize()
        .context("failed to canonicalize model cache directory")?;
    let parent = target_parent
        .canonicalize()
        .context("failed to canonicalize download target directory")?;
    if !parent.starts_with(&root) {
        anyhow::bail!(
            "download target {} escapes model cache directory {}",
            parent.display(),
            root.display()
        );
    }
    Ok(())
}

/// Parse file paths out of the HF tree API JSON (`[{"type":"file","path":...},...]`).
fn parse_tree_file_paths(json: &str) -> Result<Vec<String>> {
    let entries: serde_json::Value =
        serde_json::from_str(json).context("Invalid JSON from HF tree API")?;
    let entries = entries
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("HF tree API did not return an array"))?;
    Ok(entries
        .iter()
        .filter(|e| e.get("type").and_then(|t| t.as_str()) == Some("file"))
        .filter_map(|e| e.get("path").and_then(|p| p.as_str()).map(String::from))
        .collect())
}

/// Get list of files to download for a model and quantization, resolved
/// against the repo's actual file listing.
fn get_files_to_download(
    model_id: &str,
    quant: QuantPreset,
    remote_files: &[String],
) -> Result<Vec<String>> {
    // Aux files: only the ones the repo actually has (GGUF repos typically
    // ship none — tokenizer/config are embedded in the GGUF itself).
    let mut files: Vec<String> = [
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    .iter()
    .filter(|f| remote_files.iter().any(|r| r == *f))
    .map(|f| f.to_string())
    .collect();

    // Model weights
    if model_id.to_ascii_uppercase().contains("GGUF") || quant != QuantPreset::None {
        files.extend(select_gguf_files(model_id, quant, remote_files)?);
    } else if remote_files.iter().any(|f| f == "model.safetensors") {
        files.push("model.safetensors".to_string());
    } else {
        // Sharded safetensors (model-00001-of-000NN.safetensors)
        let mut shards: Vec<String> = remote_files
            .iter()
            .filter(|f| f.starts_with("model-") && f.ends_with(".safetensors"))
            .cloned()
            .collect();
        if shards.is_empty() {
            anyhow::bail!(
                "No safetensors weights found in {model_id}.\nAvailable files:\n  {}",
                remote_files.join("\n  ")
            );
        }
        shards.sort();
        files.extend(shards);
    }

    Ok(files)
}

/// Select the GGUF weight file(s) matching `quant` from the repo listing.
///
/// Handles both single-file (`...-Q4_K_M.gguf`) and multi-part
/// (`...-q4_k_m-00001-of-00003.gguf`) layouts — multi-part is the norm for
/// larger models (e.g. Qwen2.5-14B), so all parts are downloaded in order.
/// Fails with the repo's actual GGUF inventory (or full file list) when
/// nothing matches, instead of 404ing on a glob.
pub(crate) fn select_gguf_files(
    model_id: &str,
    quant: QuantPreset,
    remote_files: &[String],
) -> Result<Vec<String>> {
    let mut matches: Vec<String> = remote_files
        .iter()
        .filter(|f| {
            quant
                .gguf_tags()
                .iter()
                .any(|tag| matches_gguf_quant(f, tag))
        })
        .cloned()
        .collect();

    if matches.is_empty() {
        let ggufs: Vec<&String> = remote_files
            .iter()
            .filter(|f| f.to_ascii_lowercase().ends_with(".gguf"))
            .collect();
        if ggufs.is_empty() {
            anyhow::bail!(
                "No GGUF files found in {model_id} (requested quantization: {quant}).\n\
                 This repo appears to host non-GGUF weights only. Available files:\n  {}\n\
                 Hint: pass a GGUF repo, or use --quantization none for safetensors.",
                remote_files.join("\n  ")
            );
        }
        anyhow::bail!(
            "No GGUF file matching quantization {quant} in {model_id}.\n\
             Available GGUF files:\n  {}",
            ggufs
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }

    // Zero-padded part numbers sort lexically (…-00001-of-00003 first).
    matches.sort();
    Ok(matches)
}

/// True if `file` is a GGUF weight file for the given lowercase quant tag,
/// either single-file (`<name>-<tag>.gguf`) or a multi-part shard
/// (`<name>-<tag>-NNNNN-of-NNNNN.gguf`). Matching is case-insensitive.
fn matches_gguf_quant(file: &str, tag: &str) -> bool {
    let f = file.to_ascii_lowercase();
    let Some(stem) = f.strip_suffix(".gguf") else {
        return false;
    };
    // Single-file: ends with the tag (allow "-tag", ".tag", or bare tag)
    if stem == tag || stem.ends_with(&format!("-{tag}")) || stem.ends_with(&format!(".{tag}")) {
        return true;
    }
    // Multi-part: ...-<tag>-NNNNN-of-NNNNN
    if let Some(idx) = stem.rfind(&format!("-{tag}-")) {
        let rest = &stem[idx + tag.len() + 2..];
        if let Some((part, total)) = rest.split_once("-of-") {
            return !part.is_empty()
                && !total.is_empty()
                && part.chars().all(|c| c.is_ascii_digit())
                && total.chars().all(|c| c.is_ascii_digit());
        }
    }
    false
}

/// Check if a model is already downloaded
pub async fn is_model_downloaded(model: &str, cache_dir: &str) -> bool {
    let model_id = resolve_model_id(model);
    let model_cache_dir = PathBuf::from(cache_dir).join("models").join(&model_id);

    // Check for tokenizer and at least one model file
    let tokenizer_exists = model_cache_dir.join("tokenizer.json").exists();
    let has_weights = tokio::fs::read_dir(&model_cache_dir)
        .await
        .ok()
        .map(|mut dir| {
            use futures::StreamExt;
            // Simplified check - just see if directory exists and has files
            true
        })
        .unwrap_or(false);

    tokenizer_exists && has_weights
}

/// Get the path to a downloaded model
pub fn get_model_path(model: &str, cache_dir: &str) -> PathBuf {
    let model_id = resolve_model_id(model);
    PathBuf::from(cache_dir).join("models").join(&model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HF tree API fixture: Qwen-style GGUF repo (lowercase, multi-part).
    const QWEN_TREE_JSON: &str = r#"[
        {"type":"file","oid":"a","size":100,"path":".gitattributes"},
        {"type":"file","oid":"b","size":100,"path":"README.md"},
        {"type":"directory","oid":"c","path":"assets"},
        {"type":"file","oid":"d","size":1,"path":"qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf"},
        {"type":"file","oid":"e","size":1,"path":"qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"},
        {"type":"file","oid":"f","size":1,"path":"qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf"},
        {"type":"file","oid":"g","size":1,"path":"qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf"},
        {"type":"file","oid":"h","size":1,"path":"qwen2.5-14b-instruct-fp16-00001-of-00008.gguf"}
    ]"#;

    fn qwen_files() -> Vec<String> {
        parse_tree_file_paths(QWEN_TREE_JSON).unwrap()
    }

    #[test]
    fn test_parse_tree_file_paths_skips_directories() {
        let files = qwen_files();
        assert_eq!(files.len(), 7);
        assert!(!files.iter().any(|f| f == "assets"));
        assert!(files.contains(&"README.md".to_string()));
    }

    #[test]
    fn test_parse_tree_rejects_bad_json() {
        assert!(parse_tree_file_paths("not json").is_err());
        assert!(parse_tree_file_paths(r#"{"error":"Repo not found"}"#).is_err());
    }

    #[test]
    fn test_glob_expansion_multi_part_sorted() {
        // The old code pushed the literal "*Q4_K_M.gguf" -> 404. The matcher
        // must instead select the real (multi-part, lowercase) filenames.
        let files = select_gguf_files(
            "Qwen/Qwen2.5-14B-Instruct-GGUF",
            QuantPreset::Q4K,
            &qwen_files(),
        )
        .unwrap();
        assert_eq!(
            files,
            vec![
                "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
                "qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf",
                "qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf",
            ]
        );
    }

    #[test]
    fn test_glob_expansion_single_file_uppercase() {
        // bartowski-style single-file uppercase naming
        let files = vec![
            "README.md".to_string(),
            "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf".to_string(),
            "microsoft_Phi-4-mini-instruct-Q8_0.gguf".to_string(),
        ];
        let selected = select_gguf_files(
            "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
            QuantPreset::Q4K,
            &files,
        )
        .unwrap();
        assert_eq!(selected, vec!["microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"]);
    }

    #[test]
    fn test_f16_matches_fp16_spelling() {
        let selected = select_gguf_files(
            "Qwen/Qwen2.5-14B-Instruct-GGUF",
            QuantPreset::F16,
            &qwen_files(),
        )
        .unwrap();
        assert_eq!(
            selected,
            vec!["qwen2.5-14b-instruct-fp16-00001-of-00008.gguf"]
        );
    }

    #[test]
    fn test_no_gguf_in_repo_fails_with_available_files() {
        // Safetensors-only repo (the phi/microsoft case) must fail early with
        // an actionable listing, not a 404 on a glob.
        let files = vec![
            "config.json".to_string(),
            "model-00001-of-00002.safetensors".to_string(),
            "model-00002-of-00002.safetensors".to_string(),
        ];
        let err = select_gguf_files("microsoft/Phi-4-mini-instruct", QuantPreset::Q4K, &files)
            .unwrap_err()
            .to_string();
        assert!(err.contains("No GGUF files found"));
        assert!(err.contains("model-00001-of-00002.safetensors"));
        assert!(err.contains("--quantization none"));
    }

    #[test]
    fn test_missing_quant_lists_available_ggufs() {
        let files = vec!["m-q8_0.gguf".to_string()];
        let err = select_gguf_files("x/y-GGUF", QuantPreset::Q4K, &files)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Q4_K_M"));
        assert!(err.contains("m-q8_0.gguf"));
    }

    #[test]
    fn test_matches_gguf_quant_shapes() {
        assert!(matches_gguf_quant("model-Q4_K_M.gguf", "q4_k_m"));
        assert!(matches_gguf_quant("model.q4_k_m.gguf", "q4_k_m"));
        assert!(matches_gguf_quant("m-q4_k_m-00001-of-00003.gguf", "q4_k_m"));
        // No false positives on other quants or non-gguf files
        assert!(!matches_gguf_quant("model-Q4_K_S.gguf", "q4_k_m"));
        assert!(!matches_gguf_quant("model-q4_k_m.safetensors", "q4_k_m"));
        assert!(!matches_gguf_quant("m-q4_k_m-partial-of-x.gguf", "q4_k_m"));
    }

    #[test]
    fn test_files_to_download_filters_aux_by_listing() {
        // GGUF repos ship no tokenizer.json/config.json — must not request them.
        let files = get_files_to_download(
            "Qwen/Qwen2.5-14B-Instruct-GGUF",
            QuantPreset::Q4K,
            &qwen_files(),
        )
        .unwrap();
        assert!(!files.contains(&"tokenizer.json".to_string()));
        assert!(
            files.iter().all(|f| !f.contains('*')),
            "no unexpanded globs"
        );
        assert_eq!(files.iter().filter(|f| f.ends_with(".gguf")).count(), 3);
    }

    #[test]
    fn test_validate_remote_file_name_rejects_hostile_names() {
        for hostile in [
            "../x",
            "/abs",
            "a/../../x",
            "",
            "..",
            "./x/../y",
            "/etc/passwd",
        ] {
            assert!(
                validate_remote_file_name(hostile).is_err(),
                "should reject {hostile:?}"
            );
        }
    }

    #[test]
    fn test_validate_remote_file_name_accepts_normal_names() {
        for ok in [
            "model-Q4_K_M.gguf",
            "tokenizer.json",
            "subdir/model-00001-of-00003.gguf",
            "a/b/c.txt",
        ] {
            assert!(
                validate_remote_file_name(ok).is_ok(),
                "should accept {ok:?}"
            );
        }
    }

    #[test]
    fn test_ensure_under_cache_dir_containment() {
        let base = std::env::temp_dir().join(format!("ruvllm-guard-test-{}", std::process::id()));
        let root = base.join("cache");
        let inside = root.join("sub");
        let outside = base.join("outside");
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::create_dir_all(&outside).unwrap();

        assert!(ensure_under_cache_dir(&root, &inside).is_ok());
        assert!(ensure_under_cache_dir(&root, &root).is_ok());
        assert!(ensure_under_cache_dir(&root, &outside).is_err());

        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn test_curl_auth_config_keeps_token_in_header_line() {
        let cfg = curl_auth_config("hf_abc123");
        assert_eq!(cfg, "header = \"Authorization: Bearer hf_abc123\"\n");
    }

    #[test]
    fn test_curl_auth_config_escapes_special_characters() {
        // Quotes, backslashes, and newlines must not break out of the quoted
        // config string (which would allow injecting extra curl directives).
        let cfg = curl_auth_config("a\"b\\c\nd\re");
        assert_eq!(
            cfg,
            "header = \"Authorization: Bearer a\\\"b\\\\c\\nd\\re\"\n"
        );
        // Exactly one config line regardless of token content.
        assert_eq!(cfg.matches('\n').count(), 1);
        assert!(cfg.ends_with('\n'));
    }

    #[test]
    fn test_files_to_download_safetensors_path() {
        let remote = vec![
            "tokenizer.json".to_string(),
            "config.json".to_string(),
            "model.safetensors".to_string(),
        ];
        let files = get_files_to_download("test/model", QuantPreset::None, &remote).unwrap();
        assert!(files.contains(&"tokenizer.json".to_string()));
        assert!(files.contains(&"model.safetensors".to_string()));
    }
}
