//! Sealing and opening state capsules (ADR-288 §4).
//!
//! # Primitive
//!
//! XChaCha20-Poly1305. It is pure Rust with no unsafe, and it is constant-time
//! on hosts without AES hardware — which the reader's Windows, macOS, and Linux
//! ARM builds cannot assume, and AES-GCM's software fallback cannot promise.
//! The 192-bit nonce also makes a random nonce per capsule safe without a
//! counter, and a counter is what a state directory the user may copy, restore,
//! or roll back would silently break.
//!
//! # Keys
//!
//! One 32-byte install key, generated on first use from the OS CSPRNG and
//! stored at `<state_root>/install.key` with `0600` permissions. Per-capsule
//! keys are HKDF-SHA256 derived from it with the base RVF identity as `info`,
//! so two lineages on one install never share a key.
//!
//! # Wire format
//!
//! ```text
//! magic "RVFSTATE" | version u8 | identity_len u16 LE | identity | nonce(24) | ciphertext‖tag
//! \_________________________ additional authenticated data ___________________/
//! ```
//!
//! The base identity is in the header in the clear so a lineage mismatch can be
//! refused without a key, and is covered by the AAD so it cannot be edited: a
//! capsule relabelled to another lineage fails authentication rather than
//! opening as that lineage's state.

use std::path::Path;

use chacha20poly1305::aead::{Aead, AeadCore, KeyInit, OsRng, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::{is_base_identity, StateError};

/// Magic bytes opening every sealed capsule.
pub const CAPSULE_MAGIC: [u8; 8] = *b"RVFSTATE";
/// Format version of the sealed capsule header.
pub const CAPSULE_VERSION: u8 = 1;
/// HKDF salt. Fixed and public; the entropy is in the install key.
const KDF_SALT: &[u8] = b"rvforge.reader.state-capsule/1";
const NONCE_LEN: usize = 24;
const HEADER_PREFIX: usize = 8 + 1 + 2;
/// Name of the per-install key file under the state root.
pub const INSTALL_KEY_FILE: &str = "install.key";

/// The per-install root key. Zeroized on drop.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct InstallKey([u8; 32]);

impl std::fmt::Debug for InstallKey {
    /// Never prints the key material.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("InstallKey(<redacted>)")
    }
}

impl InstallKey {
    /// Use an existing 32-byte key. For tests and for customer-supplied keys.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Load `<state_root>/install.key`, generating it on first use.
    ///
    /// The file is created `0600`. An existing file with wider permissions is
    /// tightened before it is read, because a key another user can read is not
    /// a key.
    ///
    /// # Errors
    ///
    /// [`StateError::Io`] if the directory or file cannot be created, read, or
    /// have its permissions set, and [`StateError::MalformedKeyFile`] if the
    /// file is not exactly 32 bytes.
    pub fn load_or_create(state_root: &Path) -> Result<Self, StateError> {
        let path = state_root.join(INSTALL_KEY_FILE);
        std::fs::create_dir_all(state_root).map_err(|e| StateError::io(&path, &e))?;

        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }

        match opts.open(&path) {
            Ok(mut file) => {
                use std::io::Write;
                let mut key = [0u8; 32];
                fill_random(&mut key);
                let write = file
                    .write_all(&key)
                    .and_then(|()| file.flush())
                    .map_err(|e| StateError::io(&path, &e));
                match write {
                    Ok(()) => Ok(Self(key)),
                    Err(e) => {
                        key.zeroize();
                        // A partially written key file would be silently wrong
                        // on the next load, so it does not survive the failure.
                        let _ = std::fs::remove_file(&path);
                        Err(e)
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                restrict_permissions(&path)?;
                let bytes = std::fs::read(&path).map_err(|e| StateError::io(&path, &e))?;
                let key: [u8; 32] = bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| StateError::MalformedKeyFile(path.display().to_string()))?;
                Ok(Self(key))
            }
            Err(e) => Err(StateError::io(&path, &e)),
        }
    }

    /// The capsule key for one base RVF lineage.
    fn capsule_key(&self, base_identity: &str) -> Key {
        let hk = Hkdf::<Sha256>::new(Some(KDF_SALT), &self.0);
        let mut out = [0u8; 32];
        hk.expand(base_identity.as_bytes(), &mut out)
            .expect("32 bytes is a valid HKDF-SHA256 output length");
        let key = Key::from(out);
        out.zeroize();
        key
    }
}

#[cfg(unix)]
fn restrict_permissions(path: &Path) -> Result<(), StateError> {
    use std::os::unix::fs::PermissionsExt;
    let meta = std::fs::metadata(path).map_err(|e| StateError::io(path, &e))?;
    let mode = meta.permissions().mode();
    if mode & 0o077 != 0 {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| StateError::io(path, &e))?;
    }
    Ok(())
}

/// Windows has no mode bits to tighten; the file inherits the state
/// directory's ACL, which is per-user under `%APPDATA%`.
#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) -> Result<(), StateError> {
    Ok(())
}

fn fill_random(buf: &mut [u8; 32]) {
    // XChaCha20Poly1305::generate_key draws from the same OS CSPRNG the nonce
    // generator uses, so the key file and the nonces share one entropy source.
    let key = XChaCha20Poly1305::generate_key(&mut OsRng);
    buf.copy_from_slice(key.as_slice());
}

/// Seal a delta or checkpoint payload for `base_identity` under `key`.
///
/// # Errors
///
/// [`StateError::InvalidBaseIdentity`] for an identity that is not a
/// `sha256:<hex>` content address, and [`StateError::SealFailed`] if the AEAD
/// refuses the payload.
pub fn seal_with(
    key: &InstallKey,
    base_identity: &str,
    plaintext: &[u8],
) -> Result<Vec<u8>, StateError> {
    if !is_base_identity(base_identity) {
        return Err(StateError::InvalidBaseIdentity(base_identity.to_string()));
    }
    let identity = base_identity.as_bytes();
    let len = u16::try_from(identity.len())
        .map_err(|_| StateError::InvalidBaseIdentity(base_identity.to_string()))?;

    let mut aad = Vec::with_capacity(HEADER_PREFIX + identity.len());
    aad.extend_from_slice(&CAPSULE_MAGIC);
    aad.push(CAPSULE_VERSION);
    aad.extend_from_slice(&len.to_le_bytes());
    aad.extend_from_slice(identity);

    let cipher = XChaCha20Poly1305::new(&key.capsule_key(base_identity));
    let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(
            &nonce,
            Payload {
                msg: plaintext,
                aad: &aad,
            },
        )
        .map_err(|_| StateError::SealFailed)?;

    let mut out = aad;
    out.extend_from_slice(nonce.as_slice());
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Open a sealed capsule that must belong to `base_identity`.
///
/// # Errors
///
/// [`StateError::MalformedCapsule`] for a header this version cannot read,
/// [`StateError::LineageRejected`] when the capsule records a different base
/// identity, and [`StateError::OpenFailed`] when authentication fails — which
/// covers both a wrong key and tampered bytes, deliberately without saying
/// which.
pub fn unseal_with(
    key: &InstallKey,
    base_identity: &str,
    sealed: &[u8],
) -> Result<Vec<u8>, StateError> {
    if sealed.len() < HEADER_PREFIX {
        return Err(StateError::MalformedCapsule("capsule is truncated".into()));
    }
    if sealed[..8] != CAPSULE_MAGIC {
        return Err(StateError::MalformedCapsule(
            "not a sealed state capsule".into(),
        ));
    }
    if sealed[8] != CAPSULE_VERSION {
        return Err(StateError::MalformedCapsule(format!(
            "capsule format version {} is not supported",
            sealed[8]
        )));
    }
    let id_len = u16::from_le_bytes([sealed[9], sealed[10]]) as usize;
    let id_end = HEADER_PREFIX + id_len;
    let body_start = id_end + NONCE_LEN;
    if sealed.len() < body_start {
        return Err(StateError::MalformedCapsule("capsule is truncated".into()));
    }
    let recorded = std::str::from_utf8(&sealed[HEADER_PREFIX..id_end])
        .map_err(|_| StateError::MalformedCapsule("base identity is not valid UTF-8".into()))?;

    // Lineage first: refusing state from another base is a decision the reader
    // makes on the header, before any key is derived (ADR-288 §4).
    if recorded != base_identity {
        return Err(StateError::LineageRejected {
            expected: base_identity.to_string(),
            found: recorded.to_string(),
        });
    }

    let cipher = XChaCha20Poly1305::new(&key.capsule_key(base_identity));
    let nonce = XNonce::from_slice(&sealed[id_end..body_start]);
    cipher
        .decrypt(
            nonce,
            Payload {
                msg: &sealed[body_start..],
                aad: &sealed[..id_end],
            },
        )
        .map_err(|_| StateError::OpenFailed)
}

/// The base identity a capsule records, without opening it.
///
/// # Errors
///
/// [`StateError::MalformedCapsule`] if the header cannot be read.
pub fn recorded_identity(sealed: &[u8]) -> Result<String, StateError> {
    if sealed.len() < HEADER_PREFIX || sealed[..8] != CAPSULE_MAGIC {
        return Err(StateError::MalformedCapsule(
            "not a sealed state capsule".into(),
        ));
    }
    let id_len = u16::from_le_bytes([sealed[9], sealed[10]]) as usize;
    let id_end = HEADER_PREFIX + id_len;
    if sealed.len() < id_end {
        return Err(StateError::MalformedCapsule("capsule is truncated".into()));
    }
    std::str::from_utf8(&sealed[HEADER_PREFIX..id_end])
        .map(str::to_string)
        .map_err(|_| StateError::MalformedCapsule("base identity is not valid UTF-8".into()))
}
