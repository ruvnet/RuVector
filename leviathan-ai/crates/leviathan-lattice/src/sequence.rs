//! Sequence storage with Zeckendorf signatures

use super::{ZeckMask, MAX_SEQ};
use super::vocab::Vocabulary;
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A token sequence with precomputed Zeckendorf signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequence {
    pub tokens: Vec<u32>,
    pub signature: ZeckMask,
}

impl Sequence {
    pub fn new() -> Self {
        Self {
            tokens: Vec::with_capacity(MAX_SEQ),
            signature: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(cap),
            signature: 0,
        }
    }

    pub fn push(&mut self, token: u32) {
        self.tokens.push(token);
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Compute XOR signature from vocabulary Zeckendorf representations
    pub fn compute_signature(&mut self, vocab: &Vocabulary) {
        self.signature = 0;
        for &tok in &self.tokens {
            self.signature ^= vocab.get_zeck(tok);
        }
    }

    /// Check if this sequence matches a prefix
    pub fn matches_prefix(&self, prefix: &[u32]) -> bool {
        if prefix.len() > self.tokens.len() {
            return false;
        }
        self.tokens[..prefix.len()] == *prefix
    }
}

impl Default for Sequence {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<u32>> for Sequence {
    fn from(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            signature: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_prefix() {
        let seq = Sequence::from(vec![1, 2, 3, 4, 5]);

        assert!(seq.matches_prefix(&[1]));
        assert!(seq.matches_prefix(&[1, 2]));
        assert!(seq.matches_prefix(&[1, 2, 3, 4, 5]));
        assert!(!seq.matches_prefix(&[2, 3]));
        assert!(!seq.matches_prefix(&[1, 2, 3, 4, 5, 6]));
    }
}
