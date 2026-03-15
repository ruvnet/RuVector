//! Session management for rvAgent CLI.
//!
//! Provides persistence of agent conversations across sessions using
//! UUID-based session IDs stored in `~/.rvagent/sessions/`.
//!
//! Implements session encryption at rest using AES-256-GCM (ADR-103 C9).
//! Files are written with 0o600 permissions and unpredictable (UUID) filenames.

use std::collections::HashMap;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use rvagent_core::messages::Message;

// ---------------------------------------------------------------------------
// Session action sub-commands
// ---------------------------------------------------------------------------

/// Sub-commands for `rvagent session`.
#[derive(Subcommand, Debug, Clone)]
pub enum SessionAction {
    /// List all saved sessions.
    List,
    /// Show details of a session by ID.
    Show {
        /// Session ID (UUID).
        id: String,
    },
    /// Delete a session by ID.
    Delete {
        /// Session ID (UUID).
        id: String,
    },
}

// ---------------------------------------------------------------------------
// Session data
// ---------------------------------------------------------------------------

/// Persisted agent session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier (UUID v4).
    pub id: String,
    /// When the session was created.
    pub created_at: DateTime<Utc>,
    /// When the session was last updated.
    pub updated_at: DateTime<Utc>,
    /// Model used in this session.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Arbitrary key-value state for middleware / tools.
    #[serde(default)]
    pub state: HashMap<String, serde_json::Value>,
}

impl Session {
    /// Create a new empty session.
    pub fn new(model: &str) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            created_at: now,
            updated_at: now,
            model: model.to_string(),
            messages: Vec::new(),
            state: HashMap::new(),
        }
    }

    /// Add a message and update the timestamp.
    pub fn push_message(&mut self, msg: Message) {
        self.updated_at = Utc::now();
        self.messages.push(msg);
    }
}

// ---------------------------------------------------------------------------
// Session storage
// ---------------------------------------------------------------------------

/// Returns the base directory for session storage: `~/.rvagent/sessions/`.
pub fn sessions_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not determine home directory")?;
    let dir = home.join(".rvagent").join("sessions");
    Ok(dir)
}

/// Ensure the sessions directory exists with restricted permissions.
fn ensure_sessions_dir() -> Result<PathBuf> {
    let dir = sessions_dir()?;
    if !dir.exists() {
        fs::create_dir_all(&dir)?;
        // Set directory permissions to 0o700 (owner-only).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
        }
    }
    Ok(dir)
}

/// Path to a session file given its ID.
fn session_path(id: &str) -> Result<PathBuf> {
    let dir = ensure_sessions_dir()?;
    Ok(dir.join(format!("{}.json", id)))
}

// ---------------------------------------------------------------------------
// Encryption helpers (ADR-103 C9 — AES-256-GCM structure)
// ---------------------------------------------------------------------------

/// AES-256-GCM encryption module (ADR-103 C9).
///
/// Encrypts session data at rest using AES-256-GCM with random 12-byte nonces.
/// Supports backward-compatible decryption of legacy V1 (plaintext) and
/// unencrypted session files.
mod encryption {
    use aes_gcm::{Aes256Gcm, Key, Nonce};
    use aes_gcm::aead::{Aead, KeyInit};
    use anyhow::Result;
    use rand::RngCore;

    const NONCE_LEN: usize = 12;
    const MAGIC: &[u8] = b"RVAG_ENC_V2:";

    /// Encrypt session JSON for storage using AES-256-GCM.
    pub fn encrypt_session(plaintext: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
        let cipher_key = Key::<Aes256Gcm>::from_slice(key);
        let cipher = Aes256Gcm::new(cipher_key);
        let mut nonce_bytes = [0u8; NONCE_LEN];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = cipher.encrypt(nonce, plaintext)
            .map_err(|e| anyhow::anyhow!("encryption failed: {}", e))?;
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt session data, supporting V2 (AES-256-GCM), V1 (plaintext prefix),
    /// and legacy unencrypted formats.
    pub fn decrypt_session(data: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
        let v2_magic = b"RVAG_ENC_V2:";
        let v1_magic = b"RVAG_ENC_V1:";
        if data.starts_with(v2_magic) {
            let rest = &data[v2_magic.len()..];
            if rest.len() < NONCE_LEN {
                anyhow::bail!("encrypted data too short");
            }
            let (nonce_bytes, ciphertext) = rest.split_at(NONCE_LEN);
            let cipher_key = Key::<Aes256Gcm>::from_slice(key);
            let cipher = Aes256Gcm::new(cipher_key);
            let nonce = Nonce::from_slice(nonce_bytes);
            cipher.decrypt(nonce, ciphertext)
                .map_err(|e| anyhow::anyhow!("decryption failed: {}", e))
        } else if data.starts_with(v1_magic) {
            // Legacy V1 format (plaintext with prefix)
            Ok(data[v1_magic.len()..].to_vec())
        } else {
            // Legacy unencrypted
            Ok(data.to_vec())
        }
    }
}

/// Derive the session encryption key.
///
/// If `RVAGENT_SESSION_KEY` is set, uses its UTF-8 bytes (zero-padded or
/// truncated to 32 bytes). Otherwise falls back to a deterministic default
/// key. In production, this should be replaced with a proper KDF (e.g.
/// Argon2id) or platform keychain integration.
fn session_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    if let Ok(env_key) = std::env::var("RVAGENT_SESSION_KEY") {
        let bytes = env_key.as_bytes();
        let len = bytes.len().min(32);
        key[..len].copy_from_slice(&bytes[..len]);
    }
    key
}

// ---------------------------------------------------------------------------
// Save / load
// ---------------------------------------------------------------------------

/// Save a session to disk with encryption and restricted file permissions.
pub fn save_session(session: &Session) -> Result<()> {
    let path = session_path(&session.id)?;
    let json = serde_json::to_string_pretty(session)?;
    let encrypted = encryption::encrypt_session(json.as_bytes(), &session_key())?;

    // Write atomically via temp file.
    let tmp_path = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp_path)?;
        f.write_all(&encrypted)?;
        f.sync_all()?;
    }

    // Set file permissions to 0o600 (owner read/write only) per ADR-103 C9.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o600))?;
    }

    fs::rename(&tmp_path, &path)?;
    Ok(())
}

/// Load a session from disk by ID.
pub fn load_session(id: &str) -> Result<Session> {
    let path = session_path(id)?;
    let encrypted = fs::read(&path).with_context(|| format!("session not found: {}", id))?;
    let json_bytes = encryption::decrypt_session(&encrypted, &session_key())?;
    let session: Session = serde_json::from_slice(&json_bytes)?;
    Ok(session)
}

/// List all saved session IDs with their creation timestamps.
pub fn list_sessions() -> Result<Vec<(String, DateTime<Utc>)>> {
    let dir = match sessions_dir() {
        Ok(d) if d.exists() => d,
        _ => return Ok(Vec::new()),
    };

    let mut sessions = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "json") {
            if let Ok(session) = load_session_metadata(&path) {
                sessions.push(session);
            }
        }
    }
    sessions.sort_by(|a, b| b.1.cmp(&a.1)); // newest first
    Ok(sessions)
}

/// Load only the session id and created_at without deserializing all messages.
fn load_session_metadata(path: &Path) -> Result<(String, DateTime<Utc>)> {
    let encrypted = fs::read(path)?;
    let json_bytes = encryption::decrypt_session(&encrypted, &session_key())?;

    // Deserialize the full session (small overhead for listing).
    let session: Session = serde_json::from_slice(&json_bytes)?;
    Ok((session.id, session.created_at))
}

/// Delete a session file by ID.
pub fn delete_session(id: &str) -> Result<()> {
    let path = session_path(id)?;
    if path.exists() {
        fs::remove_file(&path)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI dispatch
// ---------------------------------------------------------------------------

/// Handle a `rvagent session <action>` sub-command.
pub fn handle_session_action(action: &SessionAction) -> Result<()> {
    match action {
        SessionAction::List => {
            let sessions = list_sessions()?;
            if sessions.is_empty() {
                println!("No sessions found.");
            } else {
                println!("{:<38} {}", "SESSION ID", "CREATED");
                println!("{}", "-".repeat(60));
                for (id, created) in &sessions {
                    println!("{:<38} {}", id, created.format("%Y-%m-%d %H:%M:%S UTC"));
                }
            }
        }
        SessionAction::Show { id } => {
            let session = load_session(id)?;
            println!("Session: {}", session.id);
            println!("Created: {}", session.created_at);
            println!("Updated: {}", session.updated_at);
            println!("Model:   {}", session.model);
            println!("Messages: {}", session.messages.len());
        }
        SessionAction::Delete { id } => {
            delete_session(id)?;
            println!("Deleted session {}", id);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Override the sessions dir for testing by setting up a temp directory
    /// and directly saving/loading from it.
    fn save_to_dir(session: &Session, dir: &Path) -> Result<()> {
        let path = dir.join(format!("{}.json", session.id));
        let json = serde_json::to_string_pretty(session)?;
        let encrypted = encryption::encrypt_session(json.as_bytes(), &session_key())?;
        let mut f = fs::File::create(&path)?;
        f.write_all(&encrypted)?;
        Ok(())
    }

    fn load_from_dir(id: &str, dir: &Path) -> Result<Session> {
        let path = dir.join(format!("{}.json", id));
        let encrypted = fs::read(&path)?;
        let json_bytes = encryption::decrypt_session(&encrypted, &session_key())?;
        let session: Session = serde_json::from_slice(&json_bytes)?;
        Ok(session)
    }

    #[test]
    fn test_session_new() {
        let s = Session::new("anthropic:claude-sonnet-4-20250514");
        assert!(!s.id.is_empty());
        assert_eq!(s.model, "anthropic:claude-sonnet-4-20250514");
        assert!(s.messages.is_empty());
        // ID should be a valid UUID.
        assert!(Uuid::parse_str(&s.id).is_ok());
    }

    #[test]
    fn test_session_push_message() {
        let mut s = Session::new("test:model");
        let before = s.updated_at;
        s.push_message(Message::human("hello"));
        assert_eq!(s.messages.len(), 1);
        assert!(s.updated_at >= before);
    }

    #[test]
    fn test_session_save_load_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let mut session = Session::new("test:model");
        session.push_message(Message::human("hello"));
        session.push_message(Message::ai("hi there"));

        save_to_dir(&session, tmp.path()).unwrap();
        let loaded = load_from_dir(&session.id, tmp.path()).unwrap();

        assert_eq!(loaded.id, session.id);
        assert_eq!(loaded.model, session.model);
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0].content(), "hello");
        assert_eq!(loaded.messages[1].content(), "hi there");
    }

    #[test]
    fn test_session_serialization() {
        let session = Session::new("openai:gpt-4o");
        let json = serde_json::to_string(&session).unwrap();
        let back: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, session.id);
        assert_eq!(back.model, session.model);
    }

    #[test]
    fn test_encryption_roundtrip() {
        let key = session_key();
        let data = b"test session data";
        let encrypted = encryption::encrypt_session(data, &key).unwrap();
        let decrypted = encryption::decrypt_session(&encrypted, &key).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_encryption_legacy_plaintext() {
        let key = session_key();
        // Data without the prefix should pass through (legacy support).
        let data = b"{\"id\": \"test\"}";
        let decrypted = encryption::decrypt_session(data, &key).unwrap();
        assert_eq!(decrypted, data);
    }
}
