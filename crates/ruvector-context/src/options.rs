//! Caller-supplied resource and engine configuration.

use crate::{ContextIndexError, Result};
use ruvector_core::types::DbOptions;

/// Hard ceiling on [`ContextIndexOptions::max_results`].
///
/// Top K bounds a caller-driven allocation, so an unbounded "unlimited"
/// spelling such as `usize::MAX` must be rejected by configuration validation
/// rather than reaching the allocator.
pub const MAX_RESULT_LIMIT: usize = 4_096;

/// Resource and vector-engine configuration for a scoped context index.
#[derive(Debug, Clone)]
pub struct ContextIndexOptions {
    /// Base configuration copied into every exact-scope vector shard.
    pub vector: DbOptions,
    /// Maximum exact-scope shards this process may open or create.
    ///
    /// Bounds live shard handles, not files on disk: quarantined shards hold
    /// no handle and are not counted against it.
    pub max_scopes: usize,
    /// Maximum descendant shards one search may touch.
    pub max_search_shards: usize,
    /// Maximum top K accepted from a caller, capped at [`MAX_RESULT_LIMIT`].
    pub max_results: usize,
}

impl ContextIndexOptions {
    /// Validate resource bounds before opening persistent state.
    pub fn validate(&self) -> Result<()> {
        if self.vector.dimensions == 0 {
            return Err(ContextIndexError::InvalidConfiguration(
                "vector dimensions must be nonzero",
            ));
        }
        if self.max_scopes == 0 || self.max_search_shards == 0 || self.max_results == 0 {
            return Err(ContextIndexError::InvalidConfiguration(
                "resource limits must be nonzero",
            ));
        }
        if self.max_results > MAX_RESULT_LIMIT {
            return Err(ContextIndexError::InvalidConfiguration(
                "max_results exceeds the supported ceiling",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbounded_max_results_is_rejected_by_validation() {
        let options = ContextIndexOptions {
            vector: DbOptions {
                dimensions: 3,
                ..DbOptions::default()
            },
            max_scopes: 1,
            max_search_shards: 1,
            max_results: usize::MAX,
        };
        assert!(matches!(
            options.validate(),
            Err(ContextIndexError::InvalidConfiguration(_))
        ));
    }
}
