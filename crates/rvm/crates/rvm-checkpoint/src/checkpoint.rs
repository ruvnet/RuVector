//! Checkpoint body per C2SP tlog-checkpoint.
//!
//! Body format (each line terminated by a single `\n`, U+000A):
//!
//! ```text
//! <origin>                  log identifier (non-empty)
//! <tree size>               ASCII decimal, no leading zeroes (except "0")
//! <root hash>               RFC 4648 §4 std base64 of the 32-byte root
//! [<extension line>...]     optional, opaque, non-empty
//! ```

use crate::{base64, note::NoteSigner, Error};
use rvm_witness::SealedSegment;

/// A transparency-log checkpoint: origin, tree size, root hash, and
/// optional opaque extension lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    origin: String,
    tree_size: u64,
    root_hash: [u8; 32],
    extensions: Vec<String>,
}

fn validate_origin(origin: &str) -> Result<(), Error> {
    if origin.is_empty() {
        return Err(Error::InvalidOrigin("origin MUST be non-empty"));
    }
    if origin.contains('\n') {
        return Err(Error::InvalidOrigin("origin must not contain a newline"));
    }
    Ok(())
}

impl Checkpoint {
    /// Create a checkpoint from raw parts.
    pub fn new(origin: &str, tree_size: u64, root_hash: [u8; 32]) -> Result<Self, Error> {
        validate_origin(origin)?;
        Ok(Self {
            origin: origin.to_owned(),
            tree_size,
            root_hash,
            extensions: Vec::new(),
        })
    }

    /// Create a checkpoint representing the log head after `seg`.
    ///
    /// Tree size is `seg.first_sequence + seg.count` (the sequence number
    /// one past the last record covered by the seal); the root hash is the
    /// seal's Merkle root.
    pub fn from_sealed_segment(origin: &str, seg: &SealedSegment) -> Result<Self, Error> {
        let tree_size = seg
            .first_sequence
            .checked_add(u64::from(seg.count))
            .ok_or(Error::TreeSizeOverflow)?;
        Self::new(origin, tree_size, seg.root)
    }

    /// Append an opaque extension line (spec: OPTIONAL, MUST be non-empty).
    pub fn push_extension(&mut self, line: &str) -> Result<(), Error> {
        if line.is_empty() {
            return Err(Error::InvalidExtension("extension lines MUST be non-empty"));
        }
        if line.contains('\n') {
            return Err(Error::InvalidExtension(
                "extension line must not contain a newline",
            ));
        }
        self.extensions.push(line.to_owned());
        Ok(())
    }

    /// Publish a chained seal's previous-seal binding (R1) as the opaque
    /// extension line `rvm.prev_seal <base64 digest>`.
    ///
    /// This is the integration seam for the kernel-side seal-chain work:
    /// the checkpoint body keeps its 3 required lines, and verifiers that
    /// do not understand the line ignore it (extension lines are opaque).
    /// Only meaningful for [`rvm_witness::seal::SEAL_VERSION_CHAINED`]
    /// seals; for unchained seals the digest field is padding and callers
    /// should not publish it.
    pub fn push_prev_seal_extension(&mut self, seg: &SealedSegment) -> Result<(), Error> {
        let line = format!("rvm.prev_seal {}", base64::encode(&seg.prev_seal_digest));
        self.push_extension(&line)
    }

    /// Log identifier (first body line).
    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// Number of leaves in the tree (second body line).
    pub fn tree_size(&self) -> u64 {
        self.tree_size
    }

    /// 32-byte Merkle root (third body line, base64-encoded on the wire).
    pub fn root_hash(&self) -> &[u8; 32] {
        &self.root_hash
    }

    /// Opaque extension lines, in order.
    pub fn extensions(&self) -> &[String] {
        &self.extensions
    }

    /// Serialize the checkpoint body (the signed-note *text*).
    ///
    /// Every line — including the last — is terminated by a single `\n`.
    pub fn marshal(&self) -> String {
        let mut s = format!(
            "{}\n{}\n{}\n",
            self.origin,
            self.tree_size,
            base64::encode(&self.root_hash)
        );
        for ext in &self.extensions {
            s.push_str(ext);
            s.push('\n');
        }
        s
    }

    /// Parse a checkpoint body (note text, including trailing newline).
    pub fn parse(text: &str) -> Result<Self, Error> {
        let body = text
            .strip_suffix('\n')
            .ok_or(Error::MalformedCheckpoint("text must end with a newline"))?;
        let mut lines = body.split('\n');
        let origin = lines
            .next()
            .filter(|l| !l.is_empty())
            .ok_or(Error::MalformedCheckpoint("missing origin line"))?;
        let size_line = lines
            .next()
            .filter(|l| !l.is_empty())
            .ok_or(Error::MalformedCheckpoint("missing tree size line"))?;
        let root_line = lines
            .next()
            .filter(|l| !l.is_empty())
            .ok_or(Error::MalformedCheckpoint("missing root hash line"))?;

        if !size_line.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Error::MalformedCheckpoint("tree size must be ASCII decimal"));
        }
        if size_line.len() > 1 && size_line.starts_with('0') {
            return Err(Error::MalformedCheckpoint("tree size has leading zeroes"));
        }
        let tree_size: u64 = size_line
            .parse()
            .map_err(|_| Error::MalformedCheckpoint("tree size does not fit in u64"))?;

        let root = base64::decode(root_line)?;
        let root_hash: [u8; 32] = root
            .try_into()
            .map_err(|_| Error::MalformedCheckpoint("root hash must be 32 bytes"))?;

        let mut cp = Self::new(origin, tree_size, root_hash)?;
        for ext in lines {
            cp.push_extension(ext)
                .map_err(|_| Error::MalformedCheckpoint("empty extension line"))?;
        }
        Ok(cp)
    }

    /// Serialize and sign as a C2SP signed note (Ed25519).
    ///
    /// Output: `marshal()`, a blank line, then one signature line
    /// `"— <name> <base64(key_id || sig)>\n"`.
    pub fn to_signed_note(&self, signer: &NoteSigner) -> String {
        crate::note::sign(&self.marshal(), &[signer])
            .expect("checkpoint body is always well-formed note text")
    }
}

/// Emit the checkpoint for the current log head from a set of sealed
/// segments: the seal whose coverage extends furthest
/// (max `first_sequence + count`) defines the head.
pub fn latest_checkpoint(origin: &str, seals: &[SealedSegment]) -> Result<Checkpoint, Error> {
    let head = seals
        .iter()
        .max_by_key(|s| s.first_sequence.saturating_add(u64::from(s.count)))
        .ok_or(Error::NoSeals)?;
    Checkpoint::from_sealed_segment(origin, head)
}

// Unit tests for this module live in tests/unit.rs (see the note on
// `[lib] test = false` in Cargo.toml).
