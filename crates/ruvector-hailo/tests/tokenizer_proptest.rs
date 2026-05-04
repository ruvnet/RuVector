//! Property-based fuzz tests for the WordPiece tokenizer.
//!
//! ADR-167 step 5 verification gate explicitly demanded "matches reference
//! tokenization" — the existing unit tests cover concrete examples; these
//! properties cover the algorithmic invariants over arbitrary input.

use proptest::prelude::*;
use ruvector_hailo::WordPieceTokenizer;

/// Build the same mini-vocab as the unit tests (BERT IDs convention).
fn mini_vocab() -> String {
    let mut v: Vec<String> = vec!["[PAD]".into()];
    for i in 1..100 {
        v.push(format!("[unused{}]", i));
    }
    v.extend(
        [
            "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", ",", "ru", "##v", "##ec",
            "##tor",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    v.join("\n")
}

fn tokenizer() -> WordPieceTokenizer {
    WordPieceTokenizer::from_vocab_str(&mini_vocab()).unwrap()
}

proptest! {
    /// No input should ever cause `encode` to panic, regardless of how
    /// pathological the bytes are. The fuzz includes arbitrary Unicode
    /// (emoji, RTL marks, control chars, etc.).
    #[test]
    fn encode_never_panics(text in ".*", max_seq in 1usize..256, pad in any::<bool>()) {
        let t = tokenizer();
        let _ = t.encode(&text, max_seq, pad);
    }

    /// Output length is always ≤ max_seq when not padding, exactly
    /// max_seq when padding. (Even max_seq=1 is valid — encode produces
    /// just [CLS] which fits.)
    #[test]
    fn output_length_respects_max_seq(text in ".*", max_seq in 1usize..256, pad in any::<bool>()) {
        let t = tokenizer();
        let enc = t.encode(&text, max_seq, pad);
        if pad {
            prop_assert_eq!(enc.input_ids.len(), max_seq);
            prop_assert_eq!(enc.attention_mask.len(), max_seq);
        } else {
            prop_assert!(enc.input_ids.len() <= max_seq);
            prop_assert_eq!(enc.input_ids.len(), enc.attention_mask.len());
        }
    }

    /// `actual_len` matches the count of attention=1 positions.
    #[test]
    fn actual_len_matches_attention_mask(text in ".*", max_seq in 2usize..128) {
        let t = tokenizer();
        let enc = t.encode(&text, max_seq, true);
        let real_count: usize = enc.attention_mask.iter().filter(|&&m| m == 1).count();
        prop_assert_eq!(enc.actual_len, real_count);
    }

    /// First and last *real* tokens are [CLS] and [SEP] respectively.
    /// (When max_seq is 1 the spec only fits [CLS] — handled separately.)
    #[test]
    fn cls_and_sep_bracket_real_tokens(text in ".*", max_seq in 2usize..128) {
        let t = tokenizer();
        let s = t.special_ids();
        let enc = t.encode(&text, max_seq, false);
        prop_assert_eq!(enc.input_ids[0], s.cls);
        // Last real token (before any padding) is [SEP] when there's at
        // least one real position past CLS — the encode loop reserves a
        // slot for SEP regardless of how many WordPieces fit.
        prop_assert_eq!(*enc.input_ids.last().unwrap(), s.sep);
    }

    /// Padded positions must have attention_mask == 0; unpadded ones == 1.
    #[test]
    fn pad_positions_have_zero_attention(max_seq in 4usize..64) {
        let t = tokenizer();
        // Use empty-ish text so we get padding.
        let enc = t.encode("hello", max_seq, true);
        for i in 0..max_seq {
            if i < enc.actual_len {
                prop_assert_eq!(enc.attention_mask[i], 1);
            } else {
                prop_assert_eq!(enc.attention_mask[i], 0);
                prop_assert_eq!(enc.input_ids[i], t.special_ids().pad);
            }
        }
    }

    /// Encoding the same text twice yields identical output (determinism).
    #[test]
    fn encoding_is_deterministic(text in ".*", max_seq in 2usize..64, pad in any::<bool>()) {
        let t = tokenizer();
        let a = t.encode(&text, max_seq, pad);
        let b = t.encode(&text, max_seq, pad);
        prop_assert_eq!(a.input_ids, b.input_ids);
        prop_assert_eq!(a.attention_mask, b.attention_mask);
        prop_assert_eq!(a.actual_len, b.actual_len);
    }

    /// A word that's a single vocab entry should round-trip as one
    /// token (sandwich-checked).
    #[test]
    fn known_word_is_single_token(_dummy in 0u8..1) {
        let t = tokenizer();
        let s = t.special_ids();
        let enc = t.encode("hello", 8, false);
        // [CLS] hello [SEP] = 3 tokens
        prop_assert_eq!(enc.input_ids.len(), 3);
        prop_assert_eq!(enc.input_ids[0], s.cls);
        prop_assert_ne!(enc.input_ids[1], s.unk);
        prop_assert_eq!(enc.input_ids[2], s.sep);
    }
}
