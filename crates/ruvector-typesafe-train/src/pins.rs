//! Every external input, pinned by immutable URL + sha256 (ADR-008 §8).
//!
//! The dataset URLs and hashes are copied verbatim from the bench loaders
//! (`bench/datasets/*.mjs`) so the trainer and the bench read identical bytes.
//! HF files use immutable revision SHAs, never `main`.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::norm::sha256_hex;

#[derive(Debug, Clone, Copy)]
pub struct Pin {
    /// Cache file name (also the file name inside `--cache`).
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
}

/// BAAI/bge-small-en-v1.5 at HF commit 5c38ec7c…: original training weights.
pub const BASE_SAFETENSORS: Pin = Pin {
    name: "bge-small-en-v1.5.safetensors",
    url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/model.safetensors",
    sha256: "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
};

/// Xenova/bge-small-en-v1.5 at HF commit ea104dac…: the engine's pinned
/// FP32 graph (`models/manifest.json` sha256 828e1496…).
pub const TEMPLATE_ONNX: Pin = Pin {
    name: "bge-small-en-v1.5.onnx",
    url: "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/ea104dacec62c0de699686887e3f920caeb4f3e3/onnx/model.onnx",
    sha256: "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
};

/// Tokenizer, byte-identical to `models/bge-small-en-v1.5/tokenizer.json`.
pub const TOKENIZER: Pin = Pin {
    name: "tokenizer.json",
    url: "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/ea104dacec62c0de699686887e3f920caeb4f3e3/tokenizer.json",
    sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
};

pub const BANKING77_TRAIN: Pin = Pin {
    name: "banking77-train.csv",
    url: "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    sha256: "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
};
pub const BANKING77_TEST: Pin = Pin {
    name: "banking77-test.csv",
    url: "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    sha256: "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
};
pub const BANKING77_CATEGORIES: Pin = Pin {
    name: "banking77-categories.json",
    url: "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json",
    sha256: "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
};
pub const CLINC150: Pin = Pin {
    name: "clinc150-data_full.json",
    url: "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
    sha256: "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
};
pub const HWU64: Pin = Pin {
    name: "hwu64-all.csv",
    url: "https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/master/AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv",
    sha256: "5f6dbf6d38fc111217924945ac59c554e0b926d5aa836ecdd0d089d2ca48e1d9",
};

pub const DATASET_PINS: [Pin; 5] = [
    BANKING77_TRAIN,
    BANKING77_TEST,
    BANKING77_CATEGORIES,
    CLINC150,
    HWU64,
];
pub const MODEL_PINS: [Pin; 3] = [BASE_SAFETENSORS, TEMPLATE_ONNX, TOKENIZER];

/// Supply-chain rule (ADR-008 §8): only these inputs may ever be loaded.
pub fn check_extension(path: &Path) -> Result<()> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let ok = name.ends_with(".safetensors")
        || name.ends_with(".onnx")
        || name.ends_with(".json")
        || name.ends_with(".jsonl")
        || name.ends_with(".csv")
        || name.ends_with(".txt");
    let banned = [".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"];
    if !ok || banned.iter().any(|b| name.ends_with(b)) {
        bail!("refusing to load {name:?}: only safetensors/onnx/json/jsonl/csv/txt inputs are allowed (ADR-008 §8)");
    }
    Ok(())
}

/// Read a file and require its sha256.
pub fn read_verified(path: &Path, sha256: &str) -> Result<Vec<u8>> {
    check_extension(path)?;
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let got = sha256_hex(&bytes);
    if got != sha256 {
        bail!(
            "sha256 mismatch for {}: expected {sha256}, got {got}",
            path.display()
        );
    }
    Ok(bytes)
}

/// Ensure `<cache>/<pin.name>` exists with the pinned hash, downloading it if
/// absent. A present-but-wrong file is an error (never silently replaced).
pub fn ensure(cache: &Path, pin: &Pin, allow_download: bool) -> Result<PathBuf> {
    fs::create_dir_all(cache).with_context(|| format!("mkdir {}", cache.display()))?;
    let path = cache.join(pin.name);
    if path.exists() {
        read_verified(&path, pin.sha256)?;
        return Ok(path);
    }
    if !allow_download {
        bail!(
            "{} missing from {} (run `openjev fetch` or pass --download)",
            pin.name,
            cache.display()
        );
    }
    eprintln!("[fetch] {} <- {}", pin.name, pin.url);
    let resp = ureq::get(pin.url)
        .call()
        .with_context(|| format!("GET {}", pin.url))?;
    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .with_context(|| format!("read body {}", pin.url))?;
    let got = sha256_hex(&bytes);
    if got != pin.sha256 {
        bail!(
            "sha256 mismatch downloading {}: expected {}, got {got}",
            pin.url,
            pin.sha256
        );
    }
    let tmp = path.with_extension("partial.txt");
    fs::write(&tmp, &bytes)?;
    fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Copy a pre-existing local file into the cache when its hash matches the
/// pin (used to seed the cache from a checkout without network).
pub fn seed_from(cache: &Path, pin: &Pin, src: &Path) -> Result<PathBuf> {
    let bytes = read_verified(src, pin.sha256)?;
    fs::create_dir_all(cache)?;
    let dst = cache.join(pin.name);
    if !dst.exists() {
        fs::write(&dst, bytes)?;
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_are_immutable_or_bench_identical() {
        for p in MODEL_PINS {
            assert!(!p.url.contains("/resolve/main/"), "{} not pinned", p.name);
            assert_eq!(p.sha256.len(), 64);
        }
        assert!(BANKING77_TEST.url.ends_with("/banking_data/test.csv"));
        assert!(BANKING77_CATEGORIES.url.ends_with("/categories.json"));
    }

    #[test]
    fn pickle_is_refused() {
        assert!(check_extension(Path::new("pytorch_model.bin")).is_err());
        assert!(check_extension(Path::new("x.pkl")).is_err());
        assert!(check_extension(Path::new("model.safetensors")).is_ok());
    }
}
