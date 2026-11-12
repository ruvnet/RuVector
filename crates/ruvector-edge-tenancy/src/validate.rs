//! Boundary validators (ADR-351 §4.1 step 1, §4.3).
//!
//! No Unicode normalization is performed anywhere: identifiers are compared
//! byte-for-byte. Claim components and collection names are ASCII-only, so
//! homoglyphs (fullwidth digits, Cyrillic letters, combining marks) are
//! rejected outright. Vector ids are arbitrary UTF-8, so NFC and NFD spellings
//! of the same text are **distinct** ids.

use crate::error::TenancyError;

/// Maximum length of an `org_id` / `workspace_id` claim component.
pub const MAX_CLAIM_COMPONENT_LEN: usize = 64;
/// Maximum length of an `iss` value in bytes.
pub const MAX_ISSUER_LEN: usize = 256;
/// Maximum length of a `sub` value in bytes.
pub const MAX_SUBJECT_LEN: usize = 256;
/// Maximum collection-name length.
pub const MAX_COLLECTION_NAME_LEN: usize = 63;
/// Maximum vector-id length in bytes.
pub const MAX_VECTOR_ID_BYTES: usize = 256;

/// Validate an `org_id` / `workspace_id` value: `^[A-Za-z0-9_-]{1,64}$`.
/// Rejects `:`, `/`, `|`, NUL, non-ASCII and over-long values, so the `|`
/// delimiter in the tenant-key preimage cannot be forged.
pub fn validate_claim_component(value: &str, what: &'static str) -> Result<(), TenancyError> {
    let ok = !value.is_empty()
        && value.len() <= MAX_CLAIM_COMPONENT_LEN
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if ok {
        Ok(())
    } else {
        Err(TenancyError::InvalidTenantClaim(what))
    }
}

/// Validate an `iss` value used in the tenant-key preimage: 1..=256 bytes of
/// printable ASCII (`0x21..=0x7E`) excluding the `|` preimage delimiter.
/// The issuer is also an exact match against configuration upstream; this
/// check makes the preimage unambiguous even if that check is ever relaxed.
pub fn validate_issuer(value: &str) -> Result<(), TenancyError> {
    let ok = !value.is_empty()
        && value.len() <= MAX_ISSUER_LEN
        && value
            .bytes()
            .all(|b| (0x21..=0x7E).contains(&b) && b != b'|');
    if ok {
        Ok(())
    } else {
        Err(TenancyError::InvalidTenantClaim("iss"))
    }
}

/// Validate a `sub` value (actor and membership key): 1..=256 bytes of UTF-8
/// with no control characters and no bidi/invisible formatting characters
/// (see [`is_forbidden_char`]), so it is safe to log and to use as a key.
pub fn validate_subject(value: &str) -> Result<(), TenancyError> {
    if value.is_empty() || value.len() > MAX_SUBJECT_LEN || value.chars().any(is_forbidden_char) {
        Err(TenancyError::InvalidTenantClaim("sub"))
    } else {
        Ok(())
    }
}

/// Prefix of an edge subject (ADR-351 §5.2).
pub const EDGE_SUBJECT_PREFIX: &str = "es1_";
/// Base32 characters after [`EDGE_SUBJECT_PREFIX`].
pub const EDGE_SUBJECT_HASH_LEN: usize = 26;
/// Decoded byte length of an edge-minted `jti` / `family_id`.
pub const EDGE_TOKEN_ID_BYTES: usize = 16;

/// Validate an **edge subject** (`sub` on every token kind, membership key,
/// `POST /v1/tenant/members {sub}` input): `^es1_[a-z2-7]{26}$`.
pub fn validate_edge_subject(value: &str) -> Result<(), TenancyError> {
    let ok = value.strip_prefix(EDGE_SUBJECT_PREFIX).is_some_and(|rest| {
        rest.len() == EDGE_SUBJECT_HASH_LEN
            && rest
                .bytes()
                .all(|b| b.is_ascii_lowercase() || (b'2'..=b'7').contains(&b))
    });
    if ok {
        Ok(())
    } else {
        Err(TenancyError::InvalidTenantClaim("sub"))
    }
}

/// Validate an edge-minted `jti` / `family_id`: canonical unpadded base64url
/// of exactly [`EDGE_TOKEN_ID_BYTES`] bytes (22 characters, zero trailing
/// bits).
pub fn validate_edge_token_id(value: &str, what: &'static str) -> Result<(), TenancyError> {
    match data_encoding::BASE64URL_NOPAD.decode(value.as_bytes()) {
        Ok(bytes) if bytes.len() == EDGE_TOKEN_ID_BYTES => Ok(()),
        _ => Err(TenancyError::InvalidTenantClaim(what)),
    }
}

/// Characters rejected in free-form identifiers (`sub`, vector ids):
/// Unicode `Cc` controls (C0, DEL, C1), line/paragraph separators, every
/// format (`Cf`) character, and the variation selectors (`Mn`) — anything
/// that can hide text or make two ids render identically: soft hyphen, bidi
/// marks/embeddings/overrides/
/// isolates, zero-width characters, invisible operators, deprecated format
/// controls, interlinear annotations, the BOM, variation selectors and the
/// Unicode TAG block (used for "ASCII smuggling" into LLM-visible output).
pub fn is_forbidden_char(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{00AD}'                    // SOFT HYPHEN
            | '\u{0600}'..='\u{0605}'     // Arabic number signs (Cf)
            | '\u{061C}'                  // ARABIC LETTER MARK
            | '\u{06DD}' | '\u{070F}'     // Arabic end of ayah, Syriac abbreviation mark
            | '\u{0890}' | '\u{0891}'     // Arabic pound / piastre mark above
            | '\u{08E2}'                  // ARABIC DISPUTED END OF AYAH
            | '\u{180E}'                  // MONGOLIAN VOWEL SEPARATOR
            | '\u{180B}'..='\u{180D}' | '\u{180F}' // Mongolian variation selectors
            | '\u{200B}'..='\u{200F}'     // ZWSP, ZWNJ, ZWJ, LRM, RLM
            | '\u{2028}'..='\u{202E}'     // LS, PS, LRE, RLE, PDF, LRO, RLO
            | '\u{2060}'..='\u{2064}'     // WJ, invisible operators
            | '\u{2066}'..='\u{206F}'     // isolates + deprecated format controls
            | '\u{FE00}'..='\u{FE0F}'     // variation selectors
            | '\u{FEFF}'                  // BOM / ZWNBSP
            | '\u{FFF9}'..='\u{FFFB}'     // interlinear annotations
            | '\u{110BD}' | '\u{110CD}'   // Kaithi number signs
            | '\u{13430}'..='\u{1343F}'   // Egyptian hieroglyph format controls
            | '\u{1BCA0}'..='\u{1BCA3}'   // shorthand format controls
            | '\u{1D173}'..='\u{1D17A}'   // musical symbol format controls
            | '\u{E0000}'..='\u{E007F}'   // TAG block
            | '\u{E0100}'..='\u{E01EF}' // variation selectors supplement
        )
}

/// A validated collection (or graph) name: `^[a-z0-9][a-z0-9_-]{0,62}$`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CollectionName(String);

impl CollectionName {
    /// Parse and validate.
    pub fn parse(input: &str) -> Result<Self, TenancyError> {
        let bytes = input.as_bytes();
        let first_ok = bytes
            .first()
            .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
        let rest_ok = bytes
            .iter()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_' || *b == b'-');
        if first_ok && rest_ok && bytes.len() <= MAX_COLLECTION_NAME_LEN {
            Ok(CollectionName(input.to_string()))
        } else {
            Err(TenancyError::InvalidCollectionName)
        }
    }

    /// The validated name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A validated vector id: 1-256 **bytes** of UTF-8, no forbidden characters
/// ([`is_forbidden_char`]). Not normalized: NFC and NFD forms are distinct.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorId(String);

impl VectorId {
    /// Parse and validate.
    pub fn parse(input: &str) -> Result<Self, TenancyError> {
        if input.is_empty()
            || input.len() > MAX_VECTOR_ID_BYTES
            || input.chars().any(is_forbidden_char)
        {
            return Err(TenancyError::InvalidVectorId);
        }
        Ok(VectorId(input.to_string()))
    }

    /// The validated id.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
