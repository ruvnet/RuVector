//! C2SP signed-note signing and verification (Ed25519).
//!
//! A signed note is: note text (every line non-empty, each ending in
//! `\n`), a blank line (a lone `\n`), then one or more signature lines:
//!
//! ```text
//! — <key name> <base64(key_id || signature)>\n
//! ```
//!
//! where `—` is U+2014 (em dash, UTF-8 `E2 80 94`) followed by a space,
//! and the 4-byte big-endian key ID for Ed25519 keys is
//! `SHA-256(key name || 0x0A || 0x01 || 32-byte public key)[:4]`
//! (`0x01` is the Ed25519 signature-type identifier). Signatures are
//! RFC 8032 Ed25519 over the note text (including its final newline,
//! excluding the blank separator line).
//!
//! Key strings use the Go `sumdb/note` encodings:
//! verifier `name+xxxxxxxx+base64(0x01 || pubkey)`,
//! signer `PRIVATE+KEY+name+xxxxxxxx+base64(0x01 || seed)`,
//! where `xxxxxxxx` is the key ID in lowercase hex.

use crate::{base64, checkpoint::Checkpoint, Error};
use ed25519_dalek::{Signature, Signer as _, SigningKey, Verifier as _, VerifyingKey};
use sha2::{Digest, Sha256};

/// Signature-line prefix: em dash (U+2014) then space (U+0020).
const SIG_PREFIX: &str = "\u{2014} ";
/// Signed-note algorithm identifier for Ed25519.
const ALG_ED25519: u8 = 0x01;

/// 4-byte big-endian key ID (first 4 bytes of the key hash).
pub type KeyId = [u8; 4];

fn validate_key_name(name: &str) -> Result<(), Error> {
    if name.is_empty() {
        return Err(Error::InvalidKeyName("key name MUST be non-empty"));
    }
    if name.contains('+') || name.contains('\n') || name.chars().any(char::is_whitespace) {
        return Err(Error::InvalidKeyName(
            "key name MUST NOT contain spaces, '+', or newlines",
        ));
    }
    Ok(())
}

/// `SHA-256(name || 0x0A || 0x01 || pubkey)[:4]` per C2SP signed-note.
fn key_id(name: &str, public_key: &[u8; 32]) -> KeyId {
    let mut h = Sha256::new();
    h.update(name.as_bytes());
    h.update([0x0A, ALG_ED25519]);
    h.update(public_key);
    let d = h.finalize();
    [d[0], d[1], d[2], d[3]]
}

fn hex4(id: &KeyId) -> String {
    id.iter().map(|b| format!("{b:02x}")).collect()
}

/// Note text must be non-empty, end in `\n`, and contain no blank lines.
fn validate_note_text(text: &str) -> Result<(), Error> {
    let body = text
        .strip_suffix('\n')
        .ok_or(Error::MalformedNote("text must end with a newline"))?;
    if body.is_empty() || body.split('\n').any(str::is_empty) {
        return Err(Error::MalformedNote("text lines must be non-empty"));
    }
    Ok(())
}

/// Ed25519 note signer: a key name plus signing key.
#[derive(Clone)]
pub struct NoteSigner {
    name: String,
    key: SigningKey,
    id: KeyId,
}

impl NoteSigner {
    /// Create a signer from a key name and an Ed25519 signing key.
    pub fn new(name: &str, key: SigningKey) -> Result<Self, Error> {
        validate_key_name(name)?;
        let id = key_id(name, key.verifying_key().as_bytes());
        Ok(Self {
            name: name.to_owned(),
            key,
            id,
        })
    }

    /// Create a signer from a key name and a 32-byte Ed25519 seed.
    pub fn from_seed(name: &str, seed: &[u8; 32]) -> Result<Self, Error> {
        Self::new(name, SigningKey::from_bytes(seed))
    }

    /// Parse a Go-style signer key: `PRIVATE+KEY+<name>+<hex id>+<base64>`.
    pub fn from_signer_key(skey: &str) -> Result<Self, Error> {
        let mut parts = skey.splitn(5, '+');
        let (p1, p2, name, hash, b64) = (
            parts.next().unwrap_or(""),
            parts.next().unwrap_or(""),
            parts.next().unwrap_or(""),
            parts.next().unwrap_or(""),
            parts.next().ok_or(Error::MalformedKey("expected 5 fields"))?,
        );
        if p1 != "PRIVATE" || p2 != "KEY" {
            return Err(Error::MalformedKey("missing PRIVATE+KEY prefix"));
        }
        let raw = base64::decode(b64)?;
        let (alg, seed) = raw
            .split_first()
            .ok_or(Error::MalformedKey("empty key material"))?;
        if *alg != ALG_ED25519 {
            return Err(Error::MalformedKey("unsupported algorithm (want Ed25519)"));
        }
        let seed: [u8; 32] = seed
            .try_into()
            .map_err(|_| Error::MalformedKey("Ed25519 seed must be 32 bytes"))?;
        let signer = Self::from_seed(name, &seed)?;
        if hex4(&signer.id) != hash {
            return Err(Error::KeyIdMismatch);
        }
        Ok(signer)
    }

    /// Key name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 4-byte key ID.
    pub fn key_id(&self) -> KeyId {
        self.id
    }

    /// Derive the matching verifier.
    pub fn verifier(&self) -> NoteVerifier {
        NoteVerifier {
            name: self.name.clone(),
            key: self.key.verifying_key(),
            id: self.id,
        }
    }
}

impl core::fmt::Debug for NoteSigner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Never print key material.
        f.debug_struct("NoteSigner")
            .field("name", &self.name)
            .field("key_id", &hex4(&self.id))
            .finish_non_exhaustive()
    }
}

/// Ed25519 note verifier: a key name plus verifying key.
#[derive(Debug, Clone)]
pub struct NoteVerifier {
    name: String,
    key: VerifyingKey,
    id: KeyId,
}

impl NoteVerifier {
    /// Create a verifier from a key name and an Ed25519 verifying key.
    pub fn new(name: &str, key: VerifyingKey) -> Result<Self, Error> {
        validate_key_name(name)?;
        let id = key_id(name, key.as_bytes());
        Ok(Self {
            name: name.to_owned(),
            key,
            id,
        })
    }

    /// Parse a Go-style verifier key: `<name>+<hex id>+<base64(0x01 || pubkey)>`.
    pub fn from_verifier_key(vkey: &str) -> Result<Self, Error> {
        let mut parts = vkey.splitn(3, '+');
        let name = parts.next().unwrap_or("");
        let hash = parts.next().ok_or(Error::MalformedKey("expected 3 fields"))?;
        let b64 = parts.next().ok_or(Error::MalformedKey("expected 3 fields"))?;
        let raw = base64::decode(b64)?;
        let (alg, pk) = raw
            .split_first()
            .ok_or(Error::MalformedKey("empty key material"))?;
        if *alg != ALG_ED25519 {
            return Err(Error::MalformedKey("unsupported algorithm (want Ed25519)"));
        }
        let pk: [u8; 32] = pk
            .try_into()
            .map_err(|_| Error::MalformedKey("Ed25519 public key must be 32 bytes"))?;
        let key = VerifyingKey::from_bytes(&pk)
            .map_err(|_| Error::MalformedKey("invalid Ed25519 public key"))?;
        let v = Self::new(name, key)?;
        if hex4(&v.id) != hash {
            return Err(Error::KeyIdMismatch);
        }
        Ok(v)
    }

    /// Serialize as a Go-style verifier key string
    /// (`<name>+<hex id>+<base64(0x01 || pubkey)>`), suitable for
    /// distributing to omniwitness / sumdb-note tooling.
    pub fn to_verifier_key(&self) -> String {
        let mut raw = Vec::with_capacity(33);
        raw.push(ALG_ED25519);
        raw.extend_from_slice(self.key.as_bytes());
        format!("{}+{}+{}", self.name, hex4(&self.id), base64::encode(&raw))
    }

    /// Key name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 4-byte key ID.
    pub fn key_id(&self) -> KeyId {
        self.id
    }
}

/// Sign note `text` with one or more signers, producing a full signed note.
pub fn sign(text: &str, signers: &[&NoteSigner]) -> Result<String, Error> {
    validate_note_text(text)?;
    if signers.is_empty() {
        return Err(Error::MalformedNote("at least one signer is required"));
    }
    let mut out = String::with_capacity(text.len() + 1 + signers.len() * 100);
    out.push_str(text);
    out.push('\n');
    for s in signers {
        let sig = s.key.sign(text.as_bytes());
        let mut buf = Vec::with_capacity(4 + 64);
        buf.extend_from_slice(&s.id);
        buf.extend_from_slice(&sig.to_bytes());
        out.push_str(SIG_PREFIX);
        out.push_str(&s.name);
        out.push(' ');
        out.push_str(&base64::encode(&buf));
        out.push('\n');
    }
    Ok(out)
}

/// A successfully verified note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedNote {
    /// The note text, including its final newline.
    pub text: String,
    /// Names of known keys whose signatures verified.
    pub verified_by: Vec<String>,
    /// Names on well-formed signature lines from unknown keys (ignored
    /// per spec: "Verifiers MUST ignore signatures from unknown keys").
    pub unverified: Vec<String>,
}

/// Parse and verify a signed note against a set of known verifiers.
///
/// Per the signed-note spec: well-formed signatures from unknown keys are
/// ignored; a failing signature from a *known* key (matching name AND key
/// ID) rejects the note; if no known-key signature verifies, the note is
/// rejected ([`Error::NoVerifiedSignature`]).
pub fn open(note: &str, verifiers: &[NoteVerifier]) -> Result<VerifiedNote, Error> {
    let sep = note
        .find("\n\n")
        .ok_or(Error::MalformedNote("missing blank-line separator"))?;
    let (text, rest) = note.split_at(sep + 1);
    let sigs = &rest[1..]; // skip the blank line's newline
    validate_note_text(text)?;
    let sigs = sigs
        .strip_suffix('\n')
        .ok_or(Error::MalformedNote("note must end with a newline"))?;
    if sigs.is_empty() {
        return Err(Error::MalformedNote("at least one signature line required"));
    }

    let mut verified_by = Vec::new();
    let mut unverified = Vec::new();
    for line in sigs.split('\n') {
        let rest = line
            .strip_prefix(SIG_PREFIX)
            .ok_or(Error::MalformedSignature("missing em-dash prefix"))?;
        let (name, b64) = rest
            .split_once(' ')
            .ok_or(Error::MalformedSignature("missing signature field"))?;
        validate_key_name(name)?;
        let raw = base64::decode(b64)?;
        if raw.len() < 5 {
            return Err(Error::MalformedSignature("signature must be 4+n bytes, n >= 1"));
        }
        let id: KeyId = raw[..4].try_into().expect("checked length");
        // Known key = matching name AND key ID (spec: ignore keys sharing
        // only a name or only an ID with a known key).
        match verifiers.iter().find(|v| v.name == name && v.id == id) {
            Some(v) => {
                let sig_bytes: [u8; 64] = raw[4..]
                    .try_into()
                    .map_err(|_| Error::MalformedSignature("Ed25519 signature must be 64 bytes"))?;
                v.key
                    .verify(text.as_bytes(), &Signature::from_bytes(&sig_bytes))
                    .map_err(|_| Error::InvalidSignature(name.to_owned()))?;
                verified_by.push(name.to_owned());
            }
            None => unverified.push(name.to_owned()),
        }
    }
    if verified_by.is_empty() {
        return Err(Error::NoVerifiedSignature);
    }
    Ok(VerifiedNote {
        text: text.to_owned(),
        verified_by,
        unverified,
    })
}

/// Open a signed note and parse its text as a checkpoint
/// (verify-then-parse round trip: recovers origin, tree size, root).
pub fn open_checkpoint(
    note: &str,
    verifiers: &[NoteVerifier],
) -> Result<(Checkpoint, VerifiedNote), Error> {
    let verified = open(note, verifiers)?;
    let cp = Checkpoint::parse(&verified.text)?;
    Ok((cp, verified))
}

// Unit tests for this module live in tests/unit.rs (see the note on
// `[lib] test = false` in Cargo.toml).
