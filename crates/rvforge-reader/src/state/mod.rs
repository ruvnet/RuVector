//! Encrypted state capsule layout (ADR-288).
//!
//! This module owns the directory layout and the lineage rules; [`crypto`] owns
//! the sealing, and [`seal`] / [`unseal`] are re-exported from it.
//!
//! The ADR-288 contract:
//!
//! 1. **The base RVF is immutable.** It is opened read-only, never written
//!    after signing, and any in-place modification is tampering.
//! 2. **State is a separate chain of encrypted delta segments** — a
//!    `CompressedCheckpoint` plus ordered `WitnessDelta` records — stored apart
//!    from both the base artifact and the installer payload, and deletable
//!    without touching either.
//! 3. **Every delta and checkpoint carries the base RVF identity it belongs
//!    to.** On open, a mismatch is a lineage rejection: state is refused and
//!    execution does not begin with partial state.
//!
//! ADR-288 §4 also asks that the rejection be witnessed. It is not yet:
//! [`crate::receipts`] records RVF verification results, and a state-lineage
//! rejection is a different event with no schema of its own. Today a mismatch
//! returns [`StateError::LineageRejected`] and nothing is written.
//!
//! ```text
//! <install_root>/base.rvf                  immutable, signed, read-only
//! <state_root>/install.key                 per-install key, 0600
//! <state_root>/receipts.jsonl              verification receipts
//! <state_root>/<base-identity>/
//!     checkpoint-000.ckpt                  CompressedCheckpoint
//!     delta-001.wdelta                     WitnessDelta (encrypted)
//!     delta-002.wdelta
//! ```

pub mod crypto;

use std::path::{Path, PathBuf};

pub use crypto::{recorded_identity, seal_with, unseal_with, InstallKey};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateError {
    /// The state root is inside the install root. State must be deletable
    /// without touching the immutable base artifact (ADR-288 §2).
    StateInsideInstallRoot {
        install_root: String,
        state_root: String,
    },
    /// State belongs to a different base RVF identity (ADR-288 §4).
    LineageRejected { expected: String, found: String },
    /// A base identity must be a `sha256:<hex>` content address.
    InvalidBaseIdentity(String),
    /// The capsule header is unreadable by this version.
    MalformedCapsule(String),
    /// Authentication failed: the wrong key, or altered bytes. Which one is
    /// deliberately not distinguished — answering that question is an oracle.
    OpenFailed,
    /// The AEAD refused to encrypt. Reported rather than falling back.
    SealFailed,
    /// The install key file is not a 32-byte key.
    MalformedKeyFile(String),
    /// Reading or writing the state directory failed.
    Io { path: String, message: String },
}

impl StateError {
    fn io(path: &Path, e: &std::io::Error) -> Self {
        StateError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        }
    }
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::StateInsideInstallRoot {
                install_root,
                state_root,
            } => write!(
                f,
                "state root {state_root} is inside install root {install_root}; \
                 state must be deletable independently of the base artifact"
            ),
            StateError::LineageRejected { expected, found } => write!(
                f,
                "lineage rejection: state records base identity {found}, loaded base is {expected}"
            ),
            StateError::InvalidBaseIdentity(id) => {
                write!(f, "'{id}' is not a sha256:<hex> base identity")
            }
            StateError::MalformedCapsule(why) => write!(f, "malformed state capsule: {why}"),
            StateError::OpenFailed => {
                write!(f, "the state capsule did not authenticate and was not opened")
            }
            StateError::SealFailed => write!(f, "the state capsule could not be encrypted"),
            StateError::MalformedKeyFile(path) => {
                write!(f, "{path} is not a 32-byte state key")
            }
            StateError::Io { path, message } => write!(f, "{path}: {message}"),
        }
    }
}

impl std::error::Error for StateError {}

/// Paths for one agent's state, bound to one base RVF identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCapsule {
    install_root: PathBuf,
    state_root: PathBuf,
    base_identity: String,
}

impl StateCapsule {
    pub fn new(
        install_root: impl AsRef<Path>,
        state_root: impl AsRef<Path>,
        base_identity: impl Into<String>,
    ) -> Result<Self, StateError> {
        let install_root = install_root.as_ref().to_path_buf();
        let state_root = state_root.as_ref().to_path_buf();
        let base_identity = base_identity.into();

        if !is_base_identity(&base_identity) {
            return Err(StateError::InvalidBaseIdentity(base_identity));
        }
        if state_root.starts_with(&install_root) {
            return Err(StateError::StateInsideInstallRoot {
                install_root: install_root.display().to_string(),
                state_root: state_root.display().to_string(),
            });
        }
        Ok(Self {
            install_root,
            state_root,
            base_identity,
        })
    }

    pub fn install_root(&self) -> &Path {
        &self.install_root
    }

    pub fn state_root(&self) -> &Path {
        &self.state_root
    }

    pub fn base_identity(&self) -> &str {
        &self.base_identity
    }

    /// `<state_root>/<base-identity>/`. The identity is part of the path so
    /// two lineages cannot share a directory.
    pub fn capsule_dir(&self) -> PathBuf {
        self.state_root.join(self.base_identity.replace(':', "-"))
    }

    pub fn checkpoint_path(&self, seq: u32) -> PathBuf {
        self.capsule_dir().join(format!("checkpoint-{seq:03}.ckpt"))
    }

    pub fn delta_path(&self, seq: u32) -> PathBuf {
        self.capsule_dir().join(format!("delta-{seq:03}.wdelta"))
    }

    /// The base artifact is read-only for the lifetime of its identity.
    pub fn base_artifact_path(&self) -> PathBuf {
        self.install_root.join("base.rvf")
    }

    /// Reject state that was recorded against a different base identity.
    ///
    /// ADR-288 §4 also allows migrating state whose recorded identity is an
    /// accepted *ancestor* of the loaded base. Ancestry needs the release
    /// lineage chain from the registry, so this check is exact-match only for
    /// now and rejects rather than guessing.
    pub fn check_lineage(&self, recorded_base_identity: &str) -> Result<(), StateError> {
        if recorded_base_identity == self.base_identity {
            Ok(())
        } else {
            Err(StateError::LineageRejected {
                expected: self.base_identity.clone(),
                found: recorded_base_identity.to_string(),
            })
        }
    }
}

impl StateCapsule {
    /// Seal a payload for this capsule's lineage, under the install key.
    ///
    /// # Errors
    ///
    /// Whatever [`seal`] raises, including [`StateError::Io`] if the install
    /// key cannot be read or created.
    pub fn seal(&self, plaintext: &[u8]) -> Result<Vec<u8>, StateError> {
        let key = InstallKey::load_or_create(&self.state_root)?;
        seal_with(&key, &self.base_identity, plaintext)
    }

    /// Open a payload, refusing one recorded against another base identity.
    ///
    /// # Errors
    ///
    /// [`StateError::LineageRejected`] for a foreign lineage, plus whatever
    /// [`unseal`] raises.
    pub fn unseal(&self, sealed: &[u8]) -> Result<Vec<u8>, StateError> {
        self.check_lineage(&recorded_identity(sealed)?)?;
        let key = InstallKey::load_or_create(&self.state_root)?;
        unseal_with(&key, &self.base_identity, sealed)
    }
}

fn is_base_identity(id: &str) -> bool {
    match id.strip_prefix("sha256:") {
        Some(hex) => hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()),
        None => false,
    }
}

/// Encrypt a delta or checkpoint payload under this install's key.
///
/// # Errors
///
/// [`StateError::Io`] if the key file cannot be read or created, and whatever
/// [`seal_with`] raises.
pub fn seal(base_identity: &str, plaintext: &[u8]) -> Result<Vec<u8>, StateError> {
    let key = InstallKey::load_or_create(&default_state_root())?;
    seal_with(&key, base_identity, plaintext)
}

/// Decrypt a delta or checkpoint payload. See [`seal`].
///
/// # Errors
///
/// [`StateError::Io`] if the key file cannot be read, and whatever
/// [`unseal_with`] raises — including [`StateError::OpenFailed`] for a wrong
/// key and [`StateError::LineageRejected`] for a foreign lineage.
pub fn unseal(base_identity: &str, sealed: &[u8]) -> Result<Vec<u8>, StateError> {
    let key = InstallKey::load_or_create(&default_state_root())?;
    unseal_with(&key, base_identity, sealed)
}

/// Per-user state root, kept out of the install directory.
///
/// Falls back to a relative path when the environment has no home directory
/// rather than writing somewhere surprising.
pub fn default_state_root() -> PathBuf {
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    base.unwrap_or_else(|| PathBuf::from("."))
        .join("io.ruv.rvforge.reader")
        .join("state")
}
