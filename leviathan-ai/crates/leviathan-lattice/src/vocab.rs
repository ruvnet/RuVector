//! Vocabulary management with Zeckendorf signatures

use super::{ZeckMask, VOCAB_SIZE};
use super::zeckendorf::to_zeck;
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// Vocabulary with Zeckendorf-indexed lookup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    words: Vec<String>,
    zeck_cache: Vec<ZeckMask>,
    max_size: usize,
}

impl Vocabulary {
    pub fn new(max_size: usize) -> Self {
        let mut vocab = Self {
            words: Vec::with_capacity(max_size.min(VOCAB_SIZE)),
            zeck_cache: Vec::with_capacity(max_size.min(VOCAB_SIZE)),
            max_size: max_size.min(VOCAB_SIZE),
        };

        // Reserve indices 0 and 1 for special tokens
        vocab.words.push(String::from("<PAD>")); // 0
        vocab.words.push(String::from("<EOS>")); // 1
        vocab.zeck_cache.push(0);
        vocab.zeck_cache.push(to_zeck(1));

        vocab
    }

    /// Add word to vocabulary, returns token ID
    pub fn add_word(&mut self, word: &str) -> Option<u32> {
        // Check if already exists
        for (idx, w) in self.words.iter().enumerate() {
            if w == word {
                return Some(idx as u32);
            }
        }

        // Add new word
        if self.words.len() >= self.max_size {
            return None;
        }

        let id = self.words.len() as u32;
        self.words.push(word.to_string());
        self.zeck_cache.push(to_zeck(id));
        Some(id)
    }

    /// Get token ID for word
    pub fn get_id(&self, word: &str) -> Option<u32> {
        self.words.iter().position(|w| w == word).map(|i| i as u32)
    }

    /// Get word for token ID
    pub fn get_word(&self, id: u32) -> Option<&str> {
        self.words.get(id as usize).map(|s| s.as_str())
    }

    /// Get Zeckendorf representation for token ID
    pub fn get_zeck(&self, id: u32) -> ZeckMask {
        self.zeck_cache.get(id as usize).copied().unwrap_or(0)
    }

    /// Number of words in vocabulary
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Check if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.words.len() <= 2 // Only special tokens
    }

    /// Iterate over all (id, word) pairs
    pub fn iter(&self) -> impl Iterator<Item = (u32, &str)> {
        self.words.iter().enumerate().map(|(i, w)| (i as u32, w.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_basic() {
        let mut vocab = Vocabulary::new(100);

        let id1 = vocab.add_word("hello").unwrap();
        let id2 = vocab.add_word("world").unwrap();
        let id3 = vocab.add_word("hello").unwrap(); // Duplicate

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);

        assert_eq!(vocab.get_word(id1), Some("hello"));
        assert_eq!(vocab.get_word(id2), Some("world"));
    }

    #[test]
    fn test_vocab_zeck() {
        let mut vocab = Vocabulary::new(100);
        vocab.add_word("test");

        let z = vocab.get_zeck(2); // "test" is at index 2
        assert!(z > 0);
    }
}
