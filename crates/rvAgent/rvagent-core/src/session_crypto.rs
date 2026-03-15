//! Session encryption for data at rest using AES-256-GCM.
//!
//! This module provides secure encryption for session data, implementing C9
//! from the security audit. It uses:
//!
//! - AES-256-GCM for authenticated encryption
//! - SHA3-256 for password-based key derivation
//! - 96-bit random nonces (generated per encryption)
//! - 0600 file permissions on Unix systems
//! - UUID-based unpredictable filenames

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use rand::RngCore;
use sha3::{Digest, Sha3_256};
use std::path::Path;

/// 32-byte key for AES-256
pub type EncryptionKey = [u8; 32];

/// Generate a random encryption key using the system's secure RNG.
///
/// # Examples
///
/// ```
/// use rvagent_core::session_crypto::generate_key;
///
/// let key = generate_key();
/// assert_eq!(key.len(), 32);
/// ```
pub fn generate_key() -> EncryptionKey {
    let mut key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    key
}

/// Derive key from password using SHA3-256.
///
/// # Security Note
///
/// This is a simple key derivation function suitable for demonstration.
/// Production systems should use a proper KDF like Argon2 or PBKDF2.
///
/// # Examples
///
/// ```
/// use rvagent_core::session_crypto::derive_key;
///
/// let salt = b"some_random_salt";
/// let key = derive_key("my_password", salt);
/// assert_eq!(key.len(), 32);
/// ```
pub fn derive_key(password: &str, salt: &[u8]) -> EncryptionKey {
    let mut hasher = Sha3_256::new();
    hasher.update(password.as_bytes());
    hasher.update(salt);
    let result = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&result);
    key
}

/// Session encryption/decryption using AES-256-GCM.
///
/// Provides authenticated encryption with associated data (AEAD) for
/// session data at rest. Nonces are randomly generated and prepended
/// to the ciphertext.
pub struct SessionCrypto {
    cipher: Aes256Gcm,
}

impl SessionCrypto {
    /// Create a new session crypto instance with the given key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not exactly 32 bytes (should never happen with [`EncryptionKey`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use rvagent_core::session_crypto::{generate_key, SessionCrypto};
    ///
    /// let key = generate_key();
    /// let crypto = SessionCrypto::new(&key);
    /// ```
    pub fn new(key: &EncryptionKey) -> Self {
        Self {
            cipher: Aes256Gcm::new_from_slice(key).expect("valid key length"),
        }
    }

    /// Encrypt session data.
    ///
    /// Generates a random 96-bit nonce and prepends it to the ciphertext.
    /// The output format is: `[nonce (12 bytes)][ciphertext][auth_tag (16 bytes)]`
    ///
    /// # Examples
    ///
    /// ```
    /// use rvagent_core::session_crypto::{generate_key, SessionCrypto};
    ///
    /// let key = generate_key();
    /// let crypto = SessionCrypto::new(&key);
    /// let plaintext = b"secret session data";
    /// let encrypted = crypto.encrypt(plaintext).unwrap();
    /// assert!(encrypted.len() > plaintext.len());
    /// ```
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|_| CryptoError::EncryptionFailed)?;

        // Prepend nonce to ciphertext
        let mut result = nonce_bytes.to_vec();
        result.extend(ciphertext);
        Ok(result)
    }

    /// Decrypt session data.
    ///
    /// Expects the input format: `[nonce (12 bytes)][ciphertext][auth_tag (16 bytes)]`
    ///
    /// # Examples
    ///
    /// ```
    /// use rvagent_core::session_crypto::{generate_key, SessionCrypto};
    ///
    /// let key = generate_key();
    /// let crypto = SessionCrypto::new(&key);
    /// let plaintext = b"secret session data";
    /// let encrypted = crypto.encrypt(plaintext).unwrap();
    /// let decrypted = crypto.decrypt(&encrypted).unwrap();
    /// assert_eq!(decrypted, plaintext);
    /// ```
    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>, CryptoError> {
        if data.len() < 12 {
            return Err(CryptoError::InvalidData);
        }
        let (nonce_bytes, ciphertext) = data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        self.cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| CryptoError::DecryptionFailed)
    }

    /// Save encrypted session to file with 0600 permissions.
    ///
    /// On Unix systems, sets the file permissions to 0600 (owner read/write only).
    /// On other systems, falls back to standard file write.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rvagent_core::session_crypto::{generate_key, SessionCrypto};
    /// use std::path::Path;
    ///
    /// let key = generate_key();
    /// let crypto = SessionCrypto::new(&key);
    /// let data = b"session data";
    /// crypto.save_session(Path::new("/tmp/session.enc"), data).unwrap();
    /// ```
    pub fn save_session(&self, path: &Path, data: &[u8]) -> Result<(), CryptoError> {
        let encrypted = self.encrypt(data)?;

        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(path)
                .and_then(|mut f| f.write_all(&encrypted))
                .map_err(|e| CryptoError::IoError(e.to_string()))?;
        }

        #[cfg(not(unix))]
        {
            std::fs::write(path, &encrypted).map_err(|e| CryptoError::IoError(e.to_string()))?;
        }

        Ok(())
    }

    /// Load and decrypt session from file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rvagent_core::session_crypto::{generate_key, SessionCrypto};
    /// use std::path::Path;
    ///
    /// let key = generate_key();
    /// let crypto = SessionCrypto::new(&key);
    /// let data = crypto.load_session(Path::new("/tmp/session.enc")).unwrap();
    /// ```
    pub fn load_session(&self, path: &Path) -> Result<Vec<u8>, CryptoError> {
        let encrypted = std::fs::read(path).map_err(|e| CryptoError::IoError(e.to_string()))?;
        self.decrypt(&encrypted)
    }
}

/// Errors that can occur during cryptographic operations.
#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    /// Encryption operation failed.
    #[error("encryption failed")]
    EncryptionFailed,

    /// Decryption operation failed (likely due to wrong key or corrupted data).
    #[error("decryption failed")]
    DecryptionFailed,

    /// Invalid data format (e.g., too short to contain nonce).
    #[error("invalid data format")]
    InvalidData,

    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    IoError(String),
}

/// Generate unpredictable session filename using UUID v4.
///
/// # Examples
///
/// ```
/// use rvagent_core::session_crypto::generate_session_filename;
///
/// let filename = generate_session_filename();
/// assert!(filename.starts_with("session_"));
/// assert!(filename.ends_with(".enc"));
/// ```
pub fn generate_session_filename() -> String {
    format!("session_{}.enc", uuid::Uuid::new_v4())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_key() {
        let key1 = generate_key();
        let key2 = generate_key();

        assert_eq!(key1.len(), 32);
        assert_eq!(key2.len(), 32);
        // Keys should be different (probabilistically)
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_derive_key() {
        let salt = b"test_salt";
        let key1 = derive_key("password123", salt);
        let key2 = derive_key("password123", salt);
        let key3 = derive_key("different_password", salt);

        assert_eq!(key1.len(), 32);
        assert_eq!(key1, key2); // Same password + salt = same key
        assert_ne!(key1, key3); // Different password = different key
    }

    #[test]
    fn test_encrypt_decrypt() {
        let key = generate_key();
        let crypto = SessionCrypto::new(&key);
        let plaintext = b"secret session data";

        let encrypted = crypto.encrypt(plaintext).unwrap();
        assert!(encrypted.len() > plaintext.len()); // Nonce + tag overhead

        let decrypted = crypto.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_different_nonces() {
        let key = generate_key();
        let crypto = SessionCrypto::new(&key);
        let plaintext = b"same plaintext";

        let encrypted1 = crypto.encrypt(plaintext).unwrap();
        let encrypted2 = crypto.encrypt(plaintext).unwrap();

        // Same plaintext with different nonces = different ciphertexts
        assert_ne!(encrypted1, encrypted2);

        // Both decrypt to same plaintext
        assert_eq!(crypto.decrypt(&encrypted1).unwrap(), plaintext);
        assert_eq!(crypto.decrypt(&encrypted2).unwrap(), plaintext);
    }

    #[test]
    fn test_decrypt_with_wrong_key() {
        let key1 = generate_key();
        let key2 = generate_key();
        let crypto1 = SessionCrypto::new(&key1);
        let crypto2 = SessionCrypto::new(&key2);

        let plaintext = b"secret data";
        let encrypted = crypto1.encrypt(plaintext).unwrap();

        // Decryption with wrong key should fail
        assert!(crypto2.decrypt(&encrypted).is_err());
    }

    #[test]
    fn test_decrypt_invalid_data() {
        let key = generate_key();
        let crypto = SessionCrypto::new(&key);

        // Too short (< 12 bytes)
        assert!(matches!(
            crypto.decrypt(&[1, 2, 3]),
            Err(CryptoError::InvalidData)
        ));

        // Corrupted data
        let mut encrypted = crypto.encrypt(b"test").unwrap();
        encrypted[15] ^= 0xFF; // Flip bits in ciphertext
        assert!(matches!(
            crypto.decrypt(&encrypted),
            Err(CryptoError::DecryptionFailed)
        ));
    }

    #[test]
    fn test_save_load_session() {
        let key = generate_key();
        let crypto = SessionCrypto::new(&key);
        let data = b"session data to persist";

        let temp_dir = std::env::temp_dir();
        let filename = generate_session_filename();
        let path = temp_dir.join(&filename);

        // Save
        crypto.save_session(&path, data).unwrap();

        // Load
        let loaded = crypto.load_session(&path).unwrap();
        assert_eq!(loaded, data);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    #[cfg(unix)]
    fn test_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let key = generate_key();
        let crypto = SessionCrypto::new(&key);
        let data = b"sensitive data";

        let temp_dir = std::env::temp_dir();
        let filename = generate_session_filename();
        let path = temp_dir.join(&filename);

        crypto.save_session(&path, data).unwrap();

        let metadata = std::fs::metadata(&path).unwrap();
        let permissions = metadata.permissions();
        let mode = permissions.mode();

        // Check that permissions are 0600 (owner read/write only)
        assert_eq!(mode & 0o777, 0o600);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_generate_session_filename() {
        let filename1 = generate_session_filename();
        let filename2 = generate_session_filename();

        assert!(filename1.starts_with("session_"));
        assert!(filename1.ends_with(".enc"));
        assert!(filename2.starts_with("session_"));
        assert!(filename2.ends_with(".enc"));

        // Filenames should be different (UUID v4 collision is astronomically unlikely)
        assert_ne!(filename1, filename2);
    }

    #[test]
    fn test_empty_data() {
        let key = generate_key();
        let crypto = SessionCrypto::new(&key);
        let plaintext = b"";

        let encrypted = crypto.encrypt(plaintext).unwrap();
        let decrypted = crypto.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_large_data() {
        let key = generate_key();
        let crypto = SessionCrypto::new(&key);
        let plaintext = vec![42u8; 1024 * 1024]; // 1 MB

        let encrypted = crypto.encrypt(&plaintext).unwrap();
        let decrypted = crypto.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }
}
