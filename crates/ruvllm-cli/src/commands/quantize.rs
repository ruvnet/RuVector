//! Quantize command implementation
//!
//! Converts supported llama and qwen2 SafeTensors checkpoints to F16 or Q8_0 GGUF.
//! Other input and output formats are refused before writing an output artifact.

use std::path::PathBuf;

use colored::Colorize;

use ruvllm::gguf::writer::{convert_safetensors, OutputType, WriterOptions};
use ruvllm::TargetFormat;

/// Run the quantize command
///
/// Supports local SafeTensors files and indexed HuggingFace directories.
/// Existing outputs are preserved and refused; successful output is published atomically.
pub async fn run(
    model: &str,
    output: &str,
    quant: &str,
    _ane_optimize: bool,
    keep_embed_fp16: bool,
    keep_output_fp16: bool,
    verbose: bool,
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

    let supported_input = input_path.is_dir()
        || input_path.extension().and_then(|value| value.to_str()) == Some("safetensors");
    let output_type = match format {
        TargetFormat::F16 => Some(OutputType::F16),
        TargetFormat::Q8_0 => Some(OutputType::Q8_0),
        _ => None,
    };
    if let Some(output_type) = output_type.filter(|_| supported_input) {
        let output_path = if output.is_empty() {
            let name = if input_path.is_dir() {
                input_path.file_name()
            } else {
                input_path.file_stem()
            }
            .ok_or_else(|| anyhow::anyhow!("Input model has no file name"))?;
            let mut name = name.to_os_string();
            name.push(format!("-{}.gguf", format.name().to_ascii_lowercase()));
            input_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join(name)
        } else {
            PathBuf::from(output)
        };
        let summary = convert_safetensors(
            &input_path,
            &output_path,
            &WriterOptions {
                format: output_type,
                keep_embedding_f16: keep_embed_fp16,
                keep_output_f16: keep_output_fp16,
            },
        )?;
        println!(
            "Wrote {} tensors to {}",
            summary.tensor_count,
            output_path.display()
        );
        if verbose {
            println!(
                "{} elements, {} bytes, {}",
                summary.element_count,
                summary.output_bytes,
                format.name()
            );
        }
        return Ok(());
    }

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
    use std::collections::BTreeSet;
    use std::io::Cursor;
    use std::path::Path;

    use ruvllm::gguf::parser::{parse_header, parse_metadata, parse_tensor_infos};
    use ruvllm::gguf::GgufQuantType;
    use serde_json::{json, Map, Value};
    use tempfile::TempDir;

    async fn assert_fails_closed(input_name: &str, contents: &[u8]) {
        assert_format_fails_closed(input_name, contents, "q4_k_m").await;
    }

    async fn assert_format_fails_closed(input_name: &str, contents: &[u8], format: &str) {
        let dir = std::env::temp_dir().join(format!("ruvllm-968-{}", uuid_like()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join(input_name);
        std::fs::write(&input, contents).unwrap();
        let output = dir.join("out.gguf");

        let err = run(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            format,
            true,
            true,
            true,
            false,
            dir.to_str().unwrap(),
        )
        .await
        .expect_err("unsupported conversion must fail closed");
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
        for format in ["f16", "q8_0"] {
            assert_format_fails_closed("model.gguf", b"GGUF", format).await;
        }
    }

    #[tokio::test]
    async fn unknown_format_is_rejected() {
        let err = run("x", "", "q3_zz", true, true, true, false, ".")
            .await
            .expect_err("bad format");
        assert!(err.to_string().contains("Unknown quantization format"));
    }

    fn tiny_llama(directory: &Path) -> PathBuf {
        std::fs::create_dir_all(directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&json!({
                "model_type":"llama",
                "hidden_size":32,
                "intermediate_size":64,
                "num_hidden_layers":1,
                "num_attention_heads":4,
                "num_key_value_heads":2,
                "vocab_size":8,
                "max_position_embeddings":128,
                "rope_theta":10000,
                "rms_norm_eps":0.00001,
                "hidden_act":"silu",
                "tie_word_embeddings":false,
                "bos_token_id":0,
                "eos_token_id":1,
                "pad_token_id":2
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            directory.join("tokenizer.json"),
            serde_json::to_vec(&json!({
                "normalizer":null,
                "post_processor":null,
                "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"use_regex":true},
                "decoder":{"type":"ByteLevel"},
                "model":{
                    "type":"BPE",
                    "vocab":{"a":0,"b":1,"ab":2,"c":3,"d":4,"e":5,"f":6,"g":7},
                    "merges":["a b"]
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let mut header = Map::new();
        let mut weight_map = Map::new();
        let mut payload = Vec::new();
        for (ordinal, (name, shape)) in [
            ("model.embed_tokens.weight", vec![8, 32]),
            ("model.norm.weight", vec![32]),
            ("lm_head.weight", vec![8, 32]),
            ("model.layers.0.input_layernorm.weight", vec![32]),
            ("model.layers.0.post_attention_layernorm.weight", vec![32]),
            ("model.layers.0.self_attn.q_proj.weight", vec![32, 32]),
            ("model.layers.0.self_attn.k_proj.weight", vec![16, 32]),
            ("model.layers.0.self_attn.v_proj.weight", vec![16, 32]),
            ("model.layers.0.self_attn.o_proj.weight", vec![32, 32]),
            ("model.layers.0.mlp.gate_proj.weight", vec![64, 32]),
            ("model.layers.0.mlp.up_proj.weight", vec![64, 32]),
            ("model.layers.0.mlp.down_proj.weight", vec![32, 64]),
        ]
        .into_iter()
        .enumerate()
        {
            let start = payload.len();
            for index in 0..shape.iter().product::<usize>() {
                let value = (index % 31) as f32 / 64.0 - 0.25 + ordinal as f32 / 128.0;
                payload.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                name.into(),
                json!({
                    "dtype":"F32",
                    "shape":shape,
                    "data_offsets":[start, payload.len()]
                }),
            );
            weight_map.insert(name.into(), json!("weights.safetensors"));
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        let input = directory.join("weights.safetensors");
        std::fs::write(&input, bytes).unwrap();
        std::fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata":{"total_size":payload.len()},
                "weight_map":weight_map
            }))
            .unwrap(),
        )
        .unwrap();
        input
    }

    fn assert_cli_gguf(output: &Path, format: &str) {
        let bytes = std::fs::read(output).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        let mut cursor = Cursor::new(bytes.as_slice());
        let header = parse_header(&mut cursor).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 12);
        let metadata = parse_metadata(&mut cursor, header.metadata_kv_count).unwrap();
        let tensors = parse_tensor_infos(&mut cursor, header.tensor_count).unwrap();
        assert_eq!(metadata["general.architecture"].as_str(), Some("llama"));
        assert_eq!(
            metadata["general.file_type"].as_u64(),
            Some(if format == "f16" { 1 } else { 7 })
        );
        assert_eq!(metadata["tokenizer.ggml.pre"].as_str(), Some("gpt-2"));
        let data_start = (cursor.position() as usize + 31) / 32 * 32;
        let expected_matrix = if format == "f16" {
            GgufQuantType::F16
        } else {
            GgufQuantType::Q8_0
        };
        for tensor in &tensors {
            let expected = if tensor.shape.len() == 1 {
                GgufQuantType::F32
            } else {
                expected_matrix
            };
            assert_eq!(tensor.dtype, expected, "{}", tensor.name);
            let elements = tensor.shape.iter().product::<usize>();
            let payload_size = match tensor.dtype {
                GgufQuantType::F32 => elements * 4,
                GgufQuantType::F16 => elements * 2,
                GgufQuantType::Q8_0 => {
                    assert_eq!(tensor.shape[0] % 32, 0);
                    elements / 32 * 34
                }
                _ => panic!("unexpected tensor dtype"),
            };
            let start = data_start + tensor.offset as usize;
            assert_eq!(start % 32, 0);
            assert!(payload_size > 0);
            assert!(start + payload_size <= bytes.len());
            assert!(bytes[start..start + payload_size]
                .iter()
                .any(|byte| *byte != 0));
        }
        let key = tensors
            .iter()
            .find(|tensor| tensor.name == "blk.0.attn_k.weight")
            .unwrap();
        assert_eq!(key.shape, vec![32, 16]);
        let norm = tensors
            .iter()
            .find(|tensor| tensor.name == "output_norm.weight")
            .unwrap();
        assert_eq!(norm.shape, vec![32]);
    }

    fn entries(directory: &Path) -> BTreeSet<String> {
        std::fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    }

    #[tokio::test]
    async fn supported_formats_write_tensors_with_explicit_and_default_paths() {
        for format in ["f16", "q8_0"] {
            for naming in ["explicit", "default_file", "default_directory"] {
                let directory = TempDir::new().unwrap();
                let checkpoint = directory.path().join("checkpoint");
                let input = tiny_llama(&checkpoint);
                let selected_input = if naming == "default_directory" {
                    &checkpoint
                } else {
                    &input
                };
                let expected_output = match naming {
                    "explicit" => directory.path().join("selected.gguf"),
                    "default_file" => checkpoint.join(format!("weights-{format}.gguf")),
                    "default_directory" => {
                        directory.path().join(format!("checkpoint-{format}.gguf"))
                    }
                    _ => unreachable!(),
                };
                let output_argument = if naming == "explicit" {
                    expected_output.to_str().unwrap()
                } else {
                    ""
                };
                run(
                    selected_input.to_str().unwrap(),
                    output_argument,
                    format,
                    false,
                    false,
                    false,
                    false,
                    directory.path().to_str().unwrap(),
                )
                .await
                .unwrap();
                assert_cli_gguf(&expected_output, format);
            }
        }
    }

    #[tokio::test]
    async fn supported_formats_preserve_existing_output() {
        for format in ["f16", "q8_0"] {
            let directory = TempDir::new().unwrap();
            let source = tiny_llama(&directory.path().join("checkpoint"));
            let output = directory.path().join("output.gguf");
            let sentinel = b"existing output remains intact";
            std::fs::write(&output, sentinel).unwrap();
            let before = entries(directory.path());
            let error = run(
                source.to_str().unwrap(),
                output.to_str().unwrap(),
                format,
                false,
                false,
                false,
                false,
                directory.path().to_str().unwrap(),
            )
            .await
            .unwrap_err()
            .to_string();
            assert!(error.contains("output already exists"), "{error}");
            assert_eq!(std::fs::read(&output).unwrap(), sentinel);
            assert_eq!(entries(directory.path()), before);
        }
    }

    #[tokio::test]
    async fn malformed_supported_checkpoints_leave_no_output() {
        for format in ["f16", "q8_0"] {
            for malformed in ["truncated", "missing_config_field"] {
                let directory = TempDir::new().unwrap();
                let checkpoint = directory.path().join("checkpoint");
                let source = tiny_llama(&checkpoint);
                let expected_error = if malformed == "truncated" {
                    let length = std::fs::metadata(&source).unwrap().len();
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(&source)
                        .unwrap()
                        .set_len(length - 1)
                        .unwrap();
                    "truncated"
                } else {
                    let config_path = checkpoint.join("config.json");
                    let mut config: Value =
                        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
                    config.as_object_mut().unwrap().remove("hidden_size");
                    std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();
                    "hidden_size"
                };
                let output = directory.path().join("output.gguf");
                let before = entries(directory.path());
                let error = run(
                    source.to_str().unwrap(),
                    output.to_str().unwrap(),
                    format,
                    false,
                    false,
                    false,
                    false,
                    directory.path().to_str().unwrap(),
                )
                .await
                .unwrap_err()
                .to_string();
                assert!(error.contains(expected_error), "{error}");
                assert!(!output.exists());
                assert_eq!(entries(directory.path()), before);
            }
        }
    }

    #[tokio::test]
    async fn q8_cli_defaults_quantize_embeddings() {
        use clap::Parser;
        for setting in [None, Some("true"), Some("false"), Some("")] {
            let directory = TempDir::new().unwrap();
            let source = tiny_llama(&directory.path().join("checkpoint"));
            let output = directory.path().join("cli-q8.gguf");
            let mut arguments = vec![
                "ruvllm",
                "quantize",
                "--model",
                source.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                "--quant",
                "q8_0",
            ];
            if let Some(value) = setting {
                arguments.push("--keep-embed-fp16");
                if !value.is_empty() {
                    arguments.push(value);
                }
            }
            let cli = crate::Cli::try_parse_from(arguments).unwrap();
            let crate::Commands::Quantize {
                model,
                output,
                quant,
                ane_optimize,
                keep_embed_fp16,
                keep_output_fp16,
                verbose,
            } = cli.command
            else {
                panic!("quantize command required");
            };
            assert_eq!(keep_embed_fp16, matches!(setting, Some("true" | "")));
            run(
                &model,
                &output,
                &quant,
                ane_optimize,
                keep_embed_fp16,
                keep_output_fp16,
                verbose,
                directory.path().to_str().unwrap(),
            )
            .await
            .unwrap();
            let bytes = std::fs::read(output).unwrap();
            let mut cursor = Cursor::new(bytes.as_slice());
            let header = parse_header(&mut cursor).unwrap();
            parse_metadata(&mut cursor, header.metadata_kv_count).unwrap();
            let tensors = parse_tensor_infos(&mut cursor, header.tensor_count).unwrap();
            let embedding = tensors
                .iter()
                .find(|t| t.name == "token_embd.weight")
                .unwrap();
            assert_eq!(
                embedding.dtype,
                if keep_embed_fp16 {
                    GgufQuantType::F16
                } else {
                    GgufQuantType::Q8_0
                }
            );
        }
    }
}
