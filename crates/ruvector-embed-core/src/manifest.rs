//! Content-hash-pinned model manifest (ADR-005 §"Supply chain and release").
//!
//! Each weight file carries `{name, sha256, dims, license, source_url, added,
//! review_by}` plus a `pooling` strategy and optional `tokenizer_sha256`.
//! Load fails closed on any hash mismatch. There is no network at runtime —
//! bytes come from a local path (native) or are supplied by the caller (wasm).

use crate::error::{EmbedError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Pooling strategy declared per model. bge-* use CLS pooling; sentence-
/// transformers MiniLM uses mean pooling (ADR-002 §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    #[default]
    Mean,
    Cls,
}

/// One weight file's manifest entry. Field set mirrors ADR-005 exactly, with
/// two engine-facing additions (`pooling`, `tokenizer_sha256`) that the ADR's
/// prose implies but does not enumerate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Human model name, e.g. `bge-small-en-v1.5` or `bge-small-en-v1.5-int8`.
    pub name: String,
    /// The `.onnx` file name inside the model directory.
    pub file: String,
    /// Lower-hex SHA-256 of `file`'s bytes. Verified at load; mismatch => error.
    pub sha256: String,
    /// Embedding dimension (384 for bge-small / MiniLM).
    pub dims: usize,
    /// SPDX license id of the weights.
    pub license: String,
    /// Where the weights came from (provenance, not fetched at runtime).
    pub source_url: String,
    /// ISO date the entry was added.
    pub added: String,
    /// ISO date to re-review the pin (like a `deny.toml` ignore).
    pub review_by: String,
    /// Pooling strategy for this model.
    #[serde(default)]
    pub pooling: Pooling,
    /// Tokenizer file name (defaults to `tokenizer.json`).
    #[serde(default = "default_tokenizer_file")]
    pub tokenizer_file: String,
    /// Optional SHA-256 of the tokenizer file. When present it is verified too,
    /// closing the "pinned model, unpinned tokenizer" gap.
    #[serde(default)]
    pub tokenizer_sha256: Option<String>,
    /// Max tokens fed to the model; longer inputs are truncated (documented).
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_tokenizer_file() -> String {
    "tokenizer.json".to_string()
}

fn default_max_tokens() -> usize {
    256
}

impl ModelManifest {
    /// Parse a single manifest entry from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| EmbedError::Manifest(e.to_string()))
    }

    /// `model-name@sha256-prefix` — recorded in every receipt (ADR-002 §1).
    pub fn id(&self) -> String {
        let prefix = if self.sha256.len() >= 12 {
            &self.sha256[..12]
        } else {
            &self.sha256
        };
        format!("{}@{}", self.name, prefix)
    }

    /// Verify `bytes` hash equals the pinned model `sha256`. Fails closed.
    pub fn verify_model(&self, bytes: &[u8]) -> Result<()> {
        verify_sha256(&self.name, &self.sha256, bytes)
    }

    /// Verify `bytes` hash equals `tokenizer_sha256` if one is pinned.
    /// A `None` pin is a no-op (the gap is then documented, not silently wrong).
    pub fn verify_tokenizer(&self, bytes: &[u8]) -> Result<()> {
        match &self.tokenizer_sha256 {
            Some(expected) => verify_sha256(&format!("{} (tokenizer)", self.name), expected, bytes),
            None => Ok(()),
        }
    }
}

/// A whole manifest file: `{ "models": [ ... ] }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestFile {
    pub models: Vec<ModelManifest>,
}

impl ManifestFile {
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| EmbedError::Manifest(e.to_string()))
    }

    /// Find an entry by its `name`.
    pub fn get(&self, name: &str) -> Option<&ModelManifest> {
        self.models.iter().find(|m| m.name == name)
    }
}

/// Lower-hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

/// Verify `bytes` hash equals `expected` (case-insensitive hex). No fs, no
/// network — pure over `&[u8]`, so the mismatch test needs no model files.
pub fn verify_sha256(name: &str, expected: &str, bytes: &[u8]) -> Result<()> {
    let actual = sha256_hex(bytes);
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(EmbedError::HashMismatch {
            name: name.to_string(),
            expected: expected.to_string(),
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_known_input() {
        // SHA-256("abc") — a fixed known-answer vector.
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn verify_matches() {
        let bytes = b"hello world";
        let h = sha256_hex(bytes);
        assert!(verify_sha256("m", &h, bytes).is_ok());
        // Case-insensitive.
        assert!(verify_sha256("m", &h.to_uppercase(), bytes).is_ok());
    }

    #[test]
    fn verify_mismatch_is_typed_error_no_load() {
        let err = verify_sha256("bge", "00".repeat(32).as_str(), b"tampered").unwrap_err();
        match err {
            EmbedError::HashMismatch {
                name,
                expected,
                actual,
            } => {
                assert_eq!(name, "bge");
                assert_eq!(expected, "00".repeat(32));
                assert_eq!(actual.len(), 64);
                assert_ne!(actual, expected);
            }
            other => panic!("expected HashMismatch, got {other:?}"),
        }
    }

    #[test]
    fn manifest_verify_model_rejects_tamper() {
        let m = ModelManifest {
            name: "bge-small-en-v1.5".into(),
            file: "model.onnx".into(),
            sha256: sha256_hex(b"good weights"),
            dims: 384,
            license: "MIT".into(),
            source_url: "https://example/model".into(),
            added: "2026-09-21".into(),
            review_by: "2027-03-21".into(),
            pooling: Pooling::Cls,
            tokenizer_file: "tokenizer.json".into(),
            tokenizer_sha256: None,
            max_tokens: 256,
        };
        assert!(m.verify_model(b"good weights").is_ok());
        assert!(matches!(
            m.verify_model(b"tampered weights"),
            Err(EmbedError::HashMismatch { .. })
        ));
    }

    #[test]
    fn id_uses_12_char_prefix() {
        let m = ModelManifest {
            name: "bge-small-en-v1.5".into(),
            file: "model.onnx".into(),
            sha256: "0123456789abcdef0123456789abcdef".into(),
            dims: 384,
            license: "MIT".into(),
            source_url: "".into(),
            added: "".into(),
            review_by: "".into(),
            pooling: Pooling::Cls,
            tokenizer_file: "tokenizer.json".into(),
            tokenizer_sha256: None,
            max_tokens: 256,
        };
        assert_eq!(m.id(), "bge-small-en-v1.5@0123456789ab");
    }

    #[test]
    fn parse_manifest_file() {
        let json = r#"{
          "models": [
            {"name":"bge-small-en-v1.5","file":"model.onnx","sha256":"aa","dims":384,
             "license":"MIT","source_url":"u","added":"2026-09-21","review_by":"2027-03-21",
             "pooling":"cls"}
          ]
        }"#;
        let mf = ManifestFile::from_json(json).unwrap();
        assert_eq!(mf.models.len(), 1);
        let m = mf.get("bge-small-en-v1.5").unwrap();
        assert_eq!(m.pooling, Pooling::Cls);
        // defaults applied
        assert_eq!(m.tokenizer_file, "tokenizer.json");
        assert_eq!(m.max_tokens, 256);
        assert!(m.tokenizer_sha256.is_none());
    }
}
