// Integration test library for ruvector-scipix
//
// This library provides the test infrastructure and utilities
// for integration testing the scipix OCR system.
//
// NOTE: The bulk of these integration tests target a `scipix-ocr` binary
// that does not exist in the current crate (the available binaries are
// `scipix-cli`, `scipix-server`, and `scipix-benchmark`). They also rely
// on real OCR models, network services, and large fixture files. They are
// gated behind the `scipix-integration-tests` feature so the default
// `cargo test --workspace` run stays green; enable the feature explicitly
// to run them once the missing binary and fixtures are in place.

// Common test utilities
#[cfg(feature = "scipix-integration-tests")]
pub mod common;

// Integration test modules
#[cfg(feature = "scipix-integration-tests")]
pub mod integration;

// Test configuration
#[cfg(test)]
mod test_config {
    use std::sync::Once;

    static INIT: Once = Once::new();

    /// Initialize test environment once
    pub fn init() {
        INIT.call_once(|| {
            // Setup test logging
            let _ = env_logger::builder().is_test(true).try_init();

            // Create test directories
            let test_dirs = vec![
                "/tmp/scipix_test",
                "/tmp/scipix_cache",
                "/tmp/scipix_results",
            ];

            for dir in test_dirs {
                std::fs::create_dir_all(dir).ok();
            }
        });
    }
}

// Convenience re-exports for tests
#[cfg(feature = "scipix-integration-tests")]
pub use common::*;
