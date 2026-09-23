//! Quantize command implementation
//!
//! Intended to quantize models to GGUF (K-quant / Q8). The GGUF tensor writer is
//! not implemented yet, so `run` validates its arguments and fails closed
//! instead of writing an empty or header-only file (#968).

use std::path::PathBuf;

use colored::Colorize;

use ruvllm::TargetFormat;

/// Run the quantize command
///
/// The GGUF tensor writer this command needs does not exist yet: the previous
/// implementation advanced a progress bar without loading any tensors and wrote
/// either a 75-byte GGUF header declaring zero tensors (SafeTensors/PyTorch
/// input) or an empty file (GGUF input), then reported success (#968). Until a
/// real writer lands the command validates its arguments and fails closed —
/// before any output file is created — pointing at working converters.
pub async fn run(
    model: &str,
    output: &str,
    quant: &str,
    _ane_optimize: bool,
    _keep_embed_fp16: bool,
    _keep_output_fp16: bool,
    _verbose: bool,
    cache_dir: &str,
) -> anyhow::Result<()> {
    let format = TargetFormat::from_str(quant).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown quantization format: {}. Supported: q4_k_m, q5_k_m, q8_0, f16",
            quant
        )
    })?;

    let input_path = resolve_model_path(model, cache_dir)?;
    if !input_path.exists() {
        return Err(anyhow::anyhow!(
            "Input model not found: {}",
            input_path.display()
        ));
    }

    let kind = if input_path.is_dir() {
        "a HuggingFace model directory"
    } else {
        match input_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .as_deref()
        {
            Some("gguf") => "GGUF (re-quantization)",
            Some("safetensors") => "SafeTensors",
            Some("bin") | Some("pt") => "PyTorch",
            _ => "this input format",
        }
    };

    let target = if output.is_empty() {
        "<model>-<quant>.gguf".to_string()
    } else {
        output.to_string()
    };

    Err(anyhow::anyhow!(
        "`ruvllm quantize` cannot yet produce {} from {} ({}): the GGUF tensor \
         writer is not implemented, so no output was written to {}. Use llama.cpp \
         (`convert_hf_to_gguf.py` then `llama-quantize <in.gguf> <out.gguf> {}`) or \
         candle's GGUF writer instead. Tracking: https://github.com/ruvnet/ruvector/issues/968",
        format.name(),
        kind,
        input_path.display(),
        target,
        quant.to_uppercase(),
    ))
}

/// Resolve model path from identifier or path
fn resolve_model_path(model: &str, cache_dir: &str) -> anyhow::Result<PathBuf> {
    let path = PathBuf::from(model);

    // If it's already a valid path, use it
    if path.exists() {
        return Ok(path);
    }

    // Check cache directory
    let cache_path = PathBuf::from(cache_dir).join("models").join(model);
    if cache_path.exists() {
        return Ok(cache_path);
    }

    // Check for common extensions
    for ext in &["gguf", "safetensors", "bin", "pt"] {
        let with_ext = path.with_extension(ext);
        if with_ext.exists() {
            return Ok(with_ext);
        }

        let cache_with_ext = cache_path.with_extension(ext);
        if cache_with_ext.exists() {
            return Ok(cache_with_ext);
        }
    }

    // Return original path and let the caller handle the error
    Ok(path)
}

/// Print detailed format comparison
pub fn print_format_comparison() {
    println!(
        "\n{} Quantization Format Comparison:",
        "==>".bright_blue().bold()
    );
    println!();
    println!(
        "  {:<10} {:<8} {:<12} {:<12} {:<15}",
        "Format", "Bits", "Memory (0.5B)", "Quality", "Use Case"
    );
    println!("  {}", "-".repeat(60));
    println!(
        "  {:<10} {:<8} {:<12} {:<12} {:<15}",
        "Q4_K_M", "4.5", "~300 MB", "Good", "Best tradeoff"
    );
    println!(
        "  {:<10} {:<8} {:<12} {:<12} {:<15}",
        "Q5_K_M", "5.5", "~375 MB", "Better", "Higher quality"
    );
    println!(
        "  {:<10} {:<8} {:<12} {:<12} {:<15}",
        "Q8_0", "8.5", "~500 MB", "Best", "Near-lossless"
    );
    println!(
        "  {:<10} {:<8} {:<12} {:<12} {:<15}",
        "F16", "16", "~1000 MB", "Excellent", "No quant loss"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn assert_fails_closed(input_name: &str, contents: &[u8]) {
        let dir = std::env::temp_dir().join(format!("ruvllm-968-{}", uuid_like()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join(input_name);
        std::fs::write(&input, contents).unwrap();
        let output = dir.join("out.gguf");

        let err = run(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "q4_k_m",
            true,
            true,
            true,
            false,
            dir.to_str().unwrap(),
        )
        .await
        .expect_err("quantize must fail closed until a GGUF writer exists");
        assert!(err.to_string().contains("not implemented"), "{err}");
        assert!(!output.exists(), "no output artifact may be created");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn uuid_like() -> String {
        format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )
    }

    #[tokio::test]
    async fn safetensors_input_fails_closed() {
        assert_fails_closed("model.safetensors", b"{}").await;
    }

    #[tokio::test]
    async fn gguf_input_fails_closed() {
        assert_fails_closed("model.gguf", b"GGUF").await;
    }

    #[tokio::test]
    async fn unknown_format_is_rejected() {
        let err = run("x", "", "q3_zz", true, true, true, false, ".")
            .await
            .expect_err("bad format");
        assert!(err.to_string().contains("Unknown quantization format"));
    }
}
