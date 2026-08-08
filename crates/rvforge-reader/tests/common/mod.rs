//! Shared fixtures for the install/library/update tests.
//!
//! Every test gets its own roots under the system temp directory, so no test
//! reads or writes the user's real install root, state root, or receipt log —
//! and two tests running concurrently cannot see each other's library.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use rvf_forge_core::inspect::HEADER_SIZE;
use rvf_forge_core::testkit::minimal_container;
use rvforge_reader::install::InstallLayout;
use rvforge_reader::library::Library;

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// An isolated install root, state root, and scratch area.
pub struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    pub fn new(name: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir()
            .join("rvforge-reader-tests")
            .join(format!("{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("sandbox root");
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn install_root(&self) -> PathBuf {
        self.root.join("installed")
    }

    pub fn state_root(&self) -> PathBuf {
        self.root.join("state")
    }

    pub fn layout(&self) -> InstallLayout {
        InstallLayout::new(self.install_root(), self.state_root())
    }

    pub fn library(&self) -> Library {
        Library::new(self.layout())
    }

    /// A directory inside the sandbox, created.
    pub fn dir(&self, name: &str) -> PathBuf {
        let path = self.root.join(name);
        std::fs::create_dir_all(&path).expect("sandbox dir");
        path
    }

    /// A well-formed container declaring `capabilities`, written as `<name>.rvf`.
    pub fn package(&self, name: &str, capabilities: &str) -> PathBuf {
        self.write(name, &minimal_container(capabilities))
    }

    /// A container whose first segment's payload has been altered, so it walks
    /// but fails its content-hash check.
    pub fn tampered_package(&self, name: &str, capabilities: &str) -> PathBuf {
        let mut bytes = minimal_container(capabilities);
        bytes[HEADER_SIZE + 2] ^= 0xff;
        self.write(name, &bytes)
    }

    pub fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.root.join(name);
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Every lifecycle receipt recorded, as `(subject, event, outcome)`.
pub fn lifecycle_events(library: &Library) -> Vec<(String, String, String)> {
    library
        .layout()
        .lifecycle_log()
        .read_all()
        .expect("lifecycle log")
        .into_iter()
        .map(|r| (r.subject, r.event, r.outcome))
        .collect()
}

/// Whether the lifecycle log recorded `event` with `outcome` for any subject.
pub fn recorded(library: &Library, event: &str, outcome: &str) -> bool {
    lifecycle_events(library)
        .iter()
        .any(|(_, e, o)| e == event && o == outcome)
}
