//! Build a HuggingFace tokenizer from the vocabulary a GGUF file embeds, so a
//! bare `.gguf` loads without a `tokenizer.json` next to it.
//!
//! Supported: `tokenizer.ggml.model = "gpt2"` — byte-level BPE (Qwen2, Llama 3,
//! SmolLM2, GPT-2 style vocabularies): `tokenizer.ggml.tokens` +
//! `tokenizer.ggml.merges`, with control tokens (`token_type == 3`) registered as
//! special tokens and user-defined tokens (`token_type == 4`) as whole added
//! tokens. `tokenizer.ggml.pre = "qwen2"` selects Qwen's NFC normalizer and
//! split regex; other pre-tokenizer names keep the GPT-2 regex. Other tokenizer
//! models (SentencePiece `"llama"`, …) are reported as unsupported so the caller
//! can say so; a `tokenizer.json` next to the file still takes precedence for
//! every model.

use std::collections::HashMap;

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};

/// GGUF `token_type` of a control token (`<|im_start|>`, `<|eot_id|>`, …).
const TOKEN_TYPE_CONTROL: i32 = 3;

/// GGUF `token_type` of a user-defined token (Qwen's `<tool_call>`,
/// `<think>`, …): matched whole, but kept in decoded text.
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

/// Pre-tokenizer split of Qwen2/Qwen2.5/Qwen3 vocabularies (llama.cpp
/// `tokenizer.ggml.pre = "qwen2"`), verbatim from Qwen's `tokenizer.json`.
/// Unlike GPT-2's regex it splits numbers into single digits.
const QWEN2_PRE_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Why an embedded tokenizer could not be built.
#[derive(Debug, PartialEq, Eq)]
pub enum GgufTokenizerError {
    /// The file declares a tokenizer model this module does not build.
    Unsupported(String),
    /// Required metadata is missing or malformed.
    Invalid(String),
}

impl std::fmt::Display for GgufTokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(m) => write!(
                f,
                "GGUF embeds a '{m}' tokenizer, which ruvllm cannot build yet; \
                 put the model's tokenizer.json next to the .gguf"
            ),
            Self::Invalid(m) => write!(f, "invalid GGUF tokenizer metadata: {m}"),
        }
    }
}

/// Build a byte-level BPE tokenizer from GGUF vocabulary parts, with the
/// GPT-2 pre-tokenizer. Same as [`from_gguf_parts_with_pre`] with no `pre`.
pub fn from_gguf_parts(
    model: &str,
    tokens: &[String],
    merges: &[String],
    token_types: Option<&[i32]>,
) -> Result<Tokenizer, GgufTokenizerError> {
    from_gguf_parts_with_pre(model, None, tokens, merges, token_types)
}

/// Build a byte-level BPE tokenizer from GGUF vocabulary parts.
///
/// `pre` is `tokenizer.ggml.pre`; `tokens[i]` is token id `i`; `merges` are
/// `"left right"` pairs in rank order; `token_types`, when present, marks
/// control tokens as special and user-defined tokens as added tokens.
pub fn from_gguf_parts_with_pre(
    model: &str,
    pre: Option<&str>,
    tokens: &[String],
    merges: &[String],
    token_types: Option<&[i32]>,
) -> Result<Tokenizer, GgufTokenizerError> {
    if model != "gpt2" {
        return Err(GgufTokenizerError::Unsupported(model.to_string()));
    }
    if tokens.is_empty() {
        return Err(GgufTokenizerError::Invalid(
            "tokenizer.ggml.tokens is empty".into(),
        ));
    }
    let vocab: HashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let merges: Vec<(String, String)> = merges
        .iter()
        .filter_map(|m| {
            let (a, b) = m.split_once(' ')?;
            Some((a.to_string(), b.to_string()))
        })
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| GgufTokenizerError::Invalid(e.to_string()))?;
    let mut tokenizer = Tokenizer::new(bpe);
    match pre {
        // llama.cpp maps both names to its QWEN2 pre-tokenizer.
        Some("qwen2") | Some("deepseek-r1-qwen") => {
            let split = Split::new(
                SplitPattern::Regex(QWEN2_PRE_SPLIT.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|e| GgufTokenizerError::Invalid(e.to_string()))?;
            tokenizer.with_normalizer(Some(NFC));
            tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
                split.into(),
                ByteLevel::new(false, false, false).into(),
            ])));
        }
        _ => {
            tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        }
    }
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    if let Some(types) = token_types {
        let of_type = |wanted: i32| tokens.iter().zip(types).filter(move |(_, &t)| t == wanted);
        let special: Vec<AddedToken> = of_type(TOKEN_TYPE_CONTROL)
            .map(|(tok, _)| AddedToken::from(tok.clone(), true))
            .collect();
        tokenizer.add_special_tokens(&special);
        let user_defined: Vec<AddedToken> = of_type(TOKEN_TYPE_USER_DEFINED)
            .map(|(tok, _)| AddedToken::from(tok.clone(), false).normalized(false))
            .collect();
        tokenizer.add_tokens(&user_defined);
    }
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy byte-level vocabulary: single bytes (GPT-2 byte-to-unicode map
    /// for ASCII letters and space `Ġ`), a few merges, and one control token.
    fn toy() -> (Vec<String>, Vec<String>, Vec<i32>) {
        let mut tokens: Vec<String> = "abcdehlorw".chars().map(|c| c.to_string()).collect();
        tokens.push("Ġ".into());
        for t in [
            "he", "ll", "hell", "hello", "Ġw", "or", "Ġwor", "Ġworl", "Ġworld", "<|eot|>",
        ] {
            tokens.push(t.into());
        }
        let merges = [
            "h e", "l l", "he ll", "hell o", "Ġ w", "o r", "Ġw or", "Ġwor l", "Ġworl d",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut types = vec![1; tokens.len()];
        *types.last_mut().unwrap() = TOKEN_TYPE_CONTROL;
        (tokens, merges, types)
    }

    #[test]
    fn builds_a_byte_level_bpe_that_round_trips() {
        let (tokens, merges, types) = toy();
        let tok = from_gguf_parts("gpt2", &tokens, &merges, Some(&types)).unwrap();
        let enc = tok.encode("hello world", false).unwrap();
        assert_eq!(enc.get_tokens(), ["hello", "Ġworld"]);
        let ids = enc.get_ids().to_vec();
        assert_eq!(tok.decode(&ids, false).unwrap(), "hello world");
    }

    #[test]
    fn control_tokens_are_special() {
        let (tokens, merges, types) = toy();
        let tok = from_gguf_parts("gpt2", &tokens, &merges, Some(&types)).unwrap();
        let enc = tok.encode("hello<|eot|>", false).unwrap();
        assert_eq!(enc.get_tokens(), ["hello", "<|eot|>"]);
        let eot = tok.token_to_id("<|eot|>").unwrap();
        assert_eq!(eot as usize, tokens.len() - 1);
    }

    /// Digits plus merges that join them, so the GPT-2 regex (which keeps
    /// `123` as one pre-token) and Qwen's (one digit each) disagree.
    fn digits() -> (Vec<String>, Vec<String>) {
        let tokens = ["1", "2", "3", "12", "123"].map(String::from).to_vec();
        let merges = ["1 2", "12 3"].map(String::from).to_vec();
        (tokens, merges)
    }

    #[test]
    fn qwen2_pre_tokenizer_splits_single_digits() {
        let (tokens, merges) = digits();
        let qwen = from_gguf_parts_with_pre("gpt2", Some("qwen2"), &tokens, &merges, None).unwrap();
        let enc = qwen.encode("123", false).unwrap();
        assert_eq!(enc.get_tokens(), ["1", "2", "3"]);

        let gpt2 = from_gguf_parts("gpt2", &tokens, &merges, None).unwrap();
        assert_eq!(gpt2.encode("123", false).unwrap().get_tokens(), ["123"]);
    }

    #[test]
    fn user_defined_tokens_match_whole_and_survive_decode() {
        let (mut tokens, merges, mut types) = toy();
        tokens.push("<tool>".into());
        types.push(TOKEN_TYPE_USER_DEFINED);
        let tok = from_gguf_parts_with_pre("gpt2", Some("qwen2"), &tokens, &merges, Some(&types))
            .unwrap();
        let enc = tok.encode("hello<tool>", false).unwrap();
        assert_eq!(enc.get_tokens(), ["hello", "<tool>"]);
        // Not special: skip_special_tokens keeps it, unlike a control token.
        let ids = enc.get_ids().to_vec();
        assert_eq!(tok.decode(&ids, true).unwrap(), "hello<tool>");
        let eot = tok.token_to_id("<|eot|>").unwrap();
        assert_eq!(tok.decode(&[eot], true).unwrap(), "");
    }

    #[test]
    fn sentencepiece_models_are_reported_as_unsupported() {
        let (tokens, merges, _) = toy();
        let err = from_gguf_parts("llama", &tokens, &merges, None).unwrap_err();
        assert_eq!(err, GgufTokenizerError::Unsupported("llama".into()));
        assert!(err.to_string().contains("tokenizer.json"));
    }
}
