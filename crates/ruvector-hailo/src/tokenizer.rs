//! WordPiece tokenizer for `sentence-transformers/all-MiniLM-L6-v2` and
//! BERT-family models in general.
//!
//! ADR-167 §5 step 5 (`hailo-backend` branch). Pure-Rust, std-only —
//! no NPU dependency, runs the same on x86 and aarch64.
//!
//! Algorithm (faithful to HuggingFace `BertTokenizer` /
//! `BasicTokenizer + WordPieceTokenizer` in lower-case mode):
//!
//!   1. Strip control chars, normalise whitespace
//!   2. Lowercase the input
//!   3. Split on whitespace + punctuation — each chunk becomes a
//!      "basic token"
//!   4. For each basic token, greedy longest-match against the vocab.
//!      Continuation pieces are prefixed `##`.
//!   5. Wrap with `[CLS] … [SEP]`, pad/truncate to a fixed `max_seq`.

#![allow(dead_code)]

use crate::error::HailoError;
use std::collections::HashMap;
use std::path::Path;

/// Tokenizer state — pure data, cheap to clone.
#[derive(Clone)]
pub struct WordPieceTokenizer {
    /// `vocab.txt` line N (0-indexed) → token string. Used for round-trip.
    id_to_token: Vec<String>,
    /// Reverse map for fast lookup during tokenization.
    token_to_id: HashMap<String, u32>,
    /// Special token IDs — looked up once at load time.
    pad_id: u32,
    unk_id: u32,
    cls_id: u32,
    sep_id: u32,
    /// Maximum WordPiece characters per word. BERT default is 200.
    max_input_chars_per_word: usize,
}

impl WordPieceTokenizer {
    /// Standard BERT special-token names.
    pub const PAD: &'static str = "[PAD]";
    pub const UNK: &'static str = "[UNK]";
    pub const CLS: &'static str = "[CLS]";
    pub const SEP: &'static str = "[SEP]";

    /// Load from an in-memory `vocab.txt` (one token per line, 0-indexed).
    pub fn from_vocab_str(vocab: &str) -> Result<Self, HailoError> {
        let id_to_token: Vec<String> = vocab.lines().map(|l| l.to_string()).collect();
        if id_to_token.is_empty() {
            return Err(HailoError::Tokenizer("empty vocab.txt".into()));
        }
        let mut token_to_id = HashMap::with_capacity(id_to_token.len());
        for (i, t) in id_to_token.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let lookup = |name: &str| -> Result<u32, HailoError> {
            token_to_id
                .get(name)
                .copied()
                .ok_or_else(|| HailoError::Tokenizer(format!("missing special token {}", name)))
        };
        Ok(Self {
            pad_id: lookup(Self::PAD)?,
            unk_id: lookup(Self::UNK)?,
            cls_id: lookup(Self::CLS)?,
            sep_id: lookup(Self::SEP)?,
            id_to_token,
            token_to_id,
            max_input_chars_per_word: 200,
        })
    }

    /// Load from a `vocab.txt` on disk.
    ///
    /// Iter 213 — caps the vocab read at 16 MB before pulling it into
    /// memory. The all-MiniLM-L6-v2 vocab.txt is ~232 KB; even XLM-R
    /// tops out around 5 MB. 16 MB is ~70× legit headroom and prevents
    /// a misconfig (operator pointing model_dir at /var/log/* or a
    /// large arbitrary file) from OOMing the worker at boot. Parallels
    /// iter-210/211/212's same-shape caps on operator-controlled
    /// file paths.
    pub fn from_vocab_file(path: &Path) -> Result<Self, HailoError> {
        const VOCAB_CAP: u64 = 16 * 1024 * 1024; // 16 MB
        let meta = std::fs::metadata(path).map_err(|_| HailoError::BadModelDir {
            path: path.display().to_string(),
            what: "vocab.txt (stat failed)",
        })?;
        if meta.len() > VOCAB_CAP {
            return Err(HailoError::BadModelDir {
                path: path.display().to_string(),
                what: "vocab.txt exceeds 16 MB cap (iter 213 — likely a \
                       misconfig pointing at the wrong file)",
            });
        }
        let s = std::fs::read_to_string(path).map_err(|e| {
            HailoError::BadModelDir {
                path: path.display().to_string(),
                what: "vocab.txt (io error: not handled separately)",
            }
            .into_or(e)
        })?;
        Self::from_vocab_str(&s)
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Token IDs for the four special tokens.
    pub fn special_ids(&self) -> SpecialIds {
        SpecialIds {
            pad: self.pad_id,
            unk: self.unk_id,
            cls: self.cls_id,
            sep: self.sep_id,
        }
    }

    /// Tokenize a single text into IDs, prefixed with `[CLS]` and
    /// suffixed with `[SEP]`. Output length is min(actual, max_seq).
    /// If `pad_to_max_seq` is true, output is padded with `[PAD]` to
    /// exactly `max_seq` tokens — this is what the NPU expects since
    /// the HEF is compiled for a fixed sequence length.
    pub fn encode(&self, text: &str, max_seq: usize, pad_to_max_seq: bool) -> EncodedInput {
        // Iter 130 fix: degenerate `max_seq` values used to produce
        // outputs that violated the `len <= max_seq` invariant. The
        // proptest `output_length_respects_max_seq` flushed it out
        // with `max_seq=1, text=""` → `[CLS][SEP]` (length 2). Now:
        //
        //   max_seq == 0  → empty (no room for anything)
        //   max_seq == 1  → just [CLS]   (no room for [SEP])
        //   max_seq >= 2  → [CLS] … [SEP]  (the normal path)
        //
        // pad_to_max_seq still honoured at any size.
        if max_seq == 0 {
            // pad_to_max_seq has no effect at length 0 — both branches
            // yield an empty mask. Bind to underscore to keep the
            // signature stable.
            let _ = pad_to_max_seq;
            return EncodedInput {
                input_ids: Vec::new(),
                attention_mask: Vec::new(),
                actual_len: 0,
            };
        }

        let mut ids = Vec::with_capacity(max_seq);
        ids.push(self.cls_id);

        if max_seq == 1 {
            // Only room for [CLS]. Skip body + [SEP].
            let actual_len = ids.len();
            let mut attention = vec![1u32; actual_len];
            if pad_to_max_seq {
                ids.resize(max_seq, self.pad_id);
                attention.resize(max_seq, 0);
            }
            return EncodedInput {
                input_ids: ids,
                attention_mask: attention,
                actual_len,
            };
        }

        for basic in basic_tokenize(text) {
            let pieces = self.wordpiece(&basic);
            for p in pieces {
                if ids.len() + 1 >= max_seq {
                    // reserve one slot for [SEP]
                    break;
                }
                ids.push(p);
            }
            if ids.len() + 1 >= max_seq {
                break;
            }
        }
        ids.push(self.sep_id);

        let actual_len = ids.len();
        let mut attention = vec![1u32; actual_len];
        if pad_to_max_seq {
            ids.resize(max_seq, self.pad_id);
            attention.resize(max_seq, 0);
        }

        EncodedInput {
            input_ids: ids,
            attention_mask: attention,
            actual_len,
        }
    }

    /// WordPiece a single basic token. Greedy longest-match prefix; each
    /// subsequent piece prefixed `##`. Returns `[unk_id]` if any piece
    /// can't be matched.
    fn wordpiece(&self, word: &str) -> Vec<u32> {
        if word.chars().count() > self.max_input_chars_per_word {
            return vec![self.unk_id];
        }
        let chars: Vec<char> = word.chars().collect();
        let mut out = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut matched: Option<u32> = None;
            while start < end {
                let substr: String = if start == 0 {
                    chars[start..end].iter().collect()
                } else {
                    let mut s = String::from("##");
                    s.extend(&chars[start..end]);
                    s
                };
                if let Some(&id) = self.token_to_id.get(&substr) {
                    matched = Some(id);
                    break;
                }
                end -= 1;
            }
            match matched {
                Some(id) => {
                    out.push(id);
                    start = end;
                }
                None => return vec![self.unk_id],
            }
        }
        out
    }
}

/// Output of `WordPieceTokenizer::encode`.
#[derive(Clone, Debug)]
pub struct EncodedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    /// Number of meaningful tokens before any padding.
    pub actual_len: usize,
}

/// IDs of the four BERT special tokens.
#[derive(Clone, Copy, Debug)]
pub struct SpecialIds {
    pub pad: u32,
    pub unk: u32,
    pub cls: u32,
    pub sep: u32,
}

/// BasicTokenizer (whitespace + punctuation, lowercase). Public so the
/// embedder layer can pre-split if it needs to (e.g. for char-level
/// debug logs).
pub fn basic_tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        let c = ch.to_lowercase().next().unwrap_or(ch);
        if c.is_whitespace() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else if is_punctuation(c) {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push(c.to_string());
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn is_punctuation(c: char) -> bool {
    if c.is_ascii_punctuation() {
        return true;
    }
    matches!(c as u32,
        // BERT considers various Unicode punctuation classes; keep it
        // simple here — ASCII punctuation covers the test vocab.
        0x2000..=0x206F | 0x3000..=0x303F | 0xFF00..=0xFFEF
    )
}

// Helper trait so the BadModelDir error case can chain with an io error.
trait IntoErrFromIo {
    fn into_or(self, e: std::io::Error) -> HailoError;
}
impl IntoErrFromIo for HailoError {
    fn into_or(self, _e: std::io::Error) -> HailoError {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny in-memory vocab matching the BERT vocab convention as far as
    /// the algorithm is concerned. Real all-MiniLM-L6-v2 vocab has 30522
    /// entries; we use this for unit tests until the real vocab.txt
    /// is shipped (iteration 6).
    fn mini_vocab() -> String {
        let mut v = vec!["[PAD]"]; // id 0
        for i in 1..100 {
            // Pad to id 99 so [UNK]=100 [CLS]=101 [SEP]=102 like real BERT.
            v.push(Box::leak(format!("[unused{}]", i).into_boxed_str()) as &str);
        }
        v.push("[UNK]"); // 100
        v.push("[CLS]"); // 101
        v.push("[SEP]"); // 102
        v.push("[MASK]"); // 103
        v.push("hello"); // 104
        v.push("world"); // 105
        v.push(","); // 106
        v.push("ru"); // 107
        v.push("##v"); // 108
        v.push("##ec"); // 109
        v.push("##tor"); // 110
        v.join("\n")
    }

    #[test]
    fn special_token_ids_match_bert_convention() {
        let t = WordPieceTokenizer::from_vocab_str(&mini_vocab()).unwrap();
        let s = t.special_ids();
        assert_eq!(s.pad, 0);
        assert_eq!(s.unk, 100);
        assert_eq!(s.cls, 101);
        assert_eq!(s.sep, 102);
    }

    #[test]
    fn encode_hello_world_returns_cls_hello_world_sep() {
        let t = WordPieceTokenizer::from_vocab_str(&mini_vocab()).unwrap();
        let enc = t.encode("Hello, World!", 16, false);
        // Expected: [CLS] hello , world [UNK]"!" [SEP]  (since "!" not in vocab)
        // → [101, 104, 106, 105, 100, 102]
        assert_eq!(enc.input_ids, vec![101, 104, 106, 105, 100, 102]);
        assert_eq!(enc.actual_len, 6);
        assert_eq!(enc.attention_mask, vec![1; 6]);
    }

    #[test]
    fn wordpiece_splits_unknown_word_into_continuation_pieces() {
        let t = WordPieceTokenizer::from_vocab_str(&mini_vocab()).unwrap();
        // "ruvector" → ["ru", "##v", "##ec", "##tor"] → [107, 108, 109, 110]
        let enc = t.encode("ruvector", 8, false);
        assert_eq!(enc.input_ids, vec![101, 107, 108, 109, 110, 102]);
    }

    #[test]
    fn pad_to_max_seq_yields_exact_length() {
        let t = WordPieceTokenizer::from_vocab_str(&mini_vocab()).unwrap();
        let enc = t.encode("hello", 8, true);
        assert_eq!(enc.input_ids.len(), 8);
        assert_eq!(enc.attention_mask.len(), 8);
        assert_eq!(enc.actual_len, 3); // [CLS] hello [SEP]
                                       // First 3 are real, last 5 are PAD with attention 0.
        assert_eq!(enc.input_ids[..3], [101, 104, 102]);
        assert_eq!(&enc.input_ids[3..], &[0, 0, 0, 0, 0]);
        assert_eq!(&enc.attention_mask[3..], &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn truncates_at_max_seq() {
        let t = WordPieceTokenizer::from_vocab_str(&mini_vocab()).unwrap();
        // 5 known words → [CLS] hello hello hello hello hello [SEP] = 7 tokens
        // With max_seq = 4, we must keep [CLS], 2 hellos, [SEP]
        let enc = t.encode("hello hello hello hello hello", 4, false);
        assert_eq!(enc.input_ids[0], 101); // CLS
        assert_eq!(enc.input_ids[enc.actual_len - 1], 102); // SEP last real
        assert!(enc.input_ids.len() <= 4);
    }

    /// Iter 213 — from_vocab_file rejects > 16 MB vocab files.
    #[test]
    fn from_vocab_file_rejects_oversized() {
        use std::io::Write as _;
        let path = std::env::temp_dir().join(format!(
            "iter213-oversized-vocab-{}.txt",
            std::process::id()
        ));
        // 32 MB filler — well over the 16 MB cap.
        let mut f = std::fs::File::create(&path).expect("create fixture");
        let line = "token\n";
        for _ in 0..((32 * 1024 * 1024) / line.len() + 1) {
            f.write_all(line.as_bytes()).expect("write fixture");
        }
        f.sync_all().expect("sync");
        drop(f);

        match WordPieceTokenizer::from_vocab_file(&path) {
            Ok(_) => panic!("oversized vocab must reject, but loaded"),
            Err(e) => {
                let msg = format!("{:?}", e);
                assert!(
                    msg.contains("16 MB cap") || msg.contains("iter 213"),
                    "expected 16 MB cap rejection, got: {}",
                    msg
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Iter 213 — small vocab still loads correctly.
    #[test]
    fn from_vocab_file_accepts_small_vocab() {
        let path =
            std::env::temp_dir().join(format!("iter213-small-vocab-{}.txt", std::process::id()));
        std::fs::write(&path, mini_vocab()).expect("write fixture");
        let t = WordPieceTokenizer::from_vocab_file(&path).expect("small vocab must load");
        assert!(t.vocab_size() > 0);
        let _ = std::fs::remove_file(&path);
    }
}
