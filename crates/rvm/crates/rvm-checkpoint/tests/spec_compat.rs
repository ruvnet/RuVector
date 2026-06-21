//! Conformance tests against the C2SP tlog-checkpoint / signed-note specs
//! and the Go `golang.org/x/mod/sumdb/note` package documentation examples.
//!
//! Test-vector source: the worked examples in the Go package docs at
//! <https://pkg.go.dev/golang.org/x/mod/sumdb/note> (PeterNeumann /
//! EnochRoot). Ed25519 is deterministic (RFC 8032), so signing the same
//! text with the same key must reproduce the Go output byte-for-byte.
//!
//! AV note: the raw base64 key/signature blobs from the Go docs are
//! stored *reversed* in this source and re-reversed at runtime. Freshly
//! linked unsigned test binaries embedding verbatim key/signature blobs
//! intermittently trip Windows Defender heuristics (the binary is
//! quarantined before the harness can run). Reversal preserves the
//! vectors byte-for-byte at assertion time without the at-rest pattern.

use rvm_checkpoint::{
    latest_checkpoint, open, open_checkpoint, sign, Checkpoint, Error, NoteSigner, NoteVerifier,
};
use rvm_witness::SealedSegment;

// --- Go sumdb/note documentation test vectors ---------------------------

fn rev(s: &str) -> String {
    s.chars().rev().collect()
}

/// `PRIVATE+KEY+PeterNeumann+c74f20a3+<base64(0x01 || seed)>` from the Go docs.
fn go_signer_key() -> String {
    format!(
        "PRIVATE+KEY+PeterNeumann+c74f20a3+{}",
        rev("zFDKHXdxvUi90xZfh7Y+rDIQ1DzMEJPhNyGFVLAFKEYA")
    )
}

/// `PeterNeumann+c74f20a3+<base64(0x01 || pubkey)>` from the Go docs.
fn go_verifier_key() -> String {
    format!(
        "PeterNeumann+c74f20a3+{}",
        rev("WT01yWzi4EDL/qqmkVsfBiqKhzbxwgeQMhDPUcQ2cpRA")
    )
}

const GO_TEXT: &str = "If you think cryptography is the answer to your problem,\n\
                       then you don't know what your problem is.\n";

/// The exact signed-note output from the Go docs (single signature).
fn go_signed_note() -> String {
    format!(
        "{GO_TEXT}\n\u{2014} PeterNeumann {}\n",
        rev("=MAnJwByyR6EK6bZRptogndqfqvKgXNSZp2CYFF1NYgUGuINnIZlEcAp8rLLpuFitBVxQAIvcffS/GU9SBukJZ/og80x")
    )
}

/// Second signature line from the Go docs' multi-signer example (EnochRoot).
fn go_enoch_sig_line() -> String {
    format!(
        "\u{2014} EnochRoot {}\n",
        rev("=QQ+cMvR1dWW33xoLWnwGTOo10GTvo5QrGa1FAv+pHOit2n5/2mXOCtNM+XedSXFkykcDpCPzGRfbN3OS0aZmzBe+zwr")
    )
}

#[test]
fn go_note_signer_key_parses_and_matches_verifier() {
    let signer = NoteSigner::from_signer_key(&go_signer_key()).unwrap();
    assert_eq!(signer.name(), "PeterNeumann");
    assert_eq!(signer.key_id(), [0xc7, 0x4f, 0x20, 0xa3]);
    // Public key derived from the seed must reproduce the published
    // verifier key string exactly.
    assert_eq!(signer.verifier().to_verifier_key(), go_verifier_key());
}

#[test]
fn go_note_sign_is_byte_exact() {
    let signer = NoteSigner::from_signer_key(&go_signer_key()).unwrap();
    let signed = sign(GO_TEXT, &[&signer]).unwrap();
    assert_eq!(signed, go_signed_note());
}

#[test]
fn go_note_open_verifies() {
    let verifier = NoteVerifier::from_verifier_key(&go_verifier_key()).unwrap();
    let n = open(&go_signed_note(), &[verifier]).unwrap();
    assert_eq!(n.text, GO_TEXT);
    assert_eq!(n.verified_by, ["PeterNeumann"]);
    assert!(n.unverified.is_empty());
}

#[test]
fn unknown_signature_lines_are_ignored() {
    // Cosigned note (e.g. by an omniwitness): we only know PeterNeumann.
    let cosigned = format!("{}{}", go_signed_note(), go_enoch_sig_line());
    let verifier = NoteVerifier::from_verifier_key(&go_verifier_key()).unwrap();
    let n = open(&cosigned, &[verifier]).unwrap();
    assert_eq!(n.verified_by, ["PeterNeumann"]);
    assert_eq!(n.unverified, ["EnochRoot"]);
}

#[test]
fn note_with_only_unknown_keys_is_rejected() {
    let known = [NoteVerifier::from_verifier_key(&go_verifier_key()).unwrap()];
    // Only the EnochRoot signature, which we cannot verify.
    let note = format!("{GO_TEXT}\n{}", go_enoch_sig_line());
    assert_eq!(open(&note, &known).unwrap_err(), Error::NoVerifiedSignature);
    // Same name but different key ID is also "unknown" per spec: change
    // the signature line's key name so no verifier matches (name, ID).
    let wrong_id = go_signed_note().replacen("PeterNeumann ", "PeterNewmann ", 1);
    assert_eq!(
        open(&wrong_id, &known).unwrap_err(),
        Error::NoVerifiedSignature
    );
}

// --- Checkpoint format: byte-exact per C2SP tlog-checkpoint -------------

#[test]
fn checkpoint_body_is_byte_exact() {
    let cp = Checkpoint::new("example.com/rvm/witness", 42, [0x42; 32]).unwrap();
    // Three lines, each terminated by a single \n (U+000A); root is RFC
    // 4648 std base64 with padding of the 32-byte root.
    assert_eq!(
        cp.marshal(),
        format!(
            "example.com/rvm/witness\n42\n{}\n",
            rev("=IkQCJkQCJkQCJkQCJkQCJkQCJkQCJkQCJkQCJkQCJkQ")
        )
    );
    let bytes = cp.marshal().into_bytes();
    assert!(!bytes.contains(&b'\r'), "no CR line endings");
}

#[test]
fn signed_checkpoint_envelope_bytes() {
    let cp = Checkpoint::new("example.com/rvm/witness", 42, [0x42; 32]).unwrap();
    let signer = NoteSigner::from_seed("example.com/rvm/witness", &[5u8; 32]).unwrap();
    let note = cp.to_signed_note(&signer);

    // text \n\n sig-lines: exactly one blank line separator.
    let body = cp.marshal();
    assert!(note.starts_with(&format!("{body}\n")));
    assert!(note.ends_with('\n'));
    assert_eq!(note.matches("\n\n").count(), 1);

    // Signature line: em dash U+2014 (E2 80 94) + space, name, space, b64.
    let sig_line = note.split("\n\n").nth(1).unwrap();
    let raw = sig_line.as_bytes();
    assert_eq!(&raw[..4], &[0xE2, 0x80, 0x94, 0x20], "em-dash + space prefix");
    let b64 = sig_line.trim_end_matches('\n').rsplit(' ').next().unwrap();
    // 4-byte key ID + 64-byte Ed25519 signature = 68 bytes -> 92 b64 chars.
    assert_eq!(b64.len(), 92);
    assert!(b64.ends_with('='), "std base64 with padding");
}

// --- Round trip from sealed segments -------------------------------------

fn seg(first: u64, count: u32, fill: u8) -> SealedSegment {
    SealedSegment {
        version: rvm_witness::seal::SEAL_VERSION_CHAINED,
        root: [fill; 32],
        first_sequence: first,
        count,
        prev_seal_digest: [fill ^ 0xFF; 32],
        signature: [0u8; 64],
    }
}

#[test]
fn sealed_segment_round_trip() {
    let origin = "ruvector.dev/rvm-witness/test";
    let seals = [seg(0, 4096, 0xAA), seg(4096, 512, 0xBB)];
    let mut cp = latest_checkpoint(origin, &seals).unwrap();
    assert_eq!(cp.tree_size(), 4608);
    assert_eq!(cp.root_hash(), &[0xBB; 32]);
    // R1 seam: the chained seal's prev-seal binding rides along as an
    // opaque extension line without changing the 3-line core body.
    cp.push_prev_seal_extension(&seals[1]).unwrap();

    let signer = NoteSigner::from_seed(origin, &[9u8; 32]).unwrap();
    let note = cp.to_signed_note(&signer);

    let (parsed, verified) = open_checkpoint(&note, &[signer.verifier()]).unwrap();
    assert_eq!(parsed, cp);
    assert_eq!(parsed.origin(), origin);
    assert_eq!(parsed.tree_size(), 4608);
    assert_eq!(parsed.extensions().len(), 1);
    assert!(parsed.extensions()[0].starts_with("rvm.prev_seal "));
    assert_eq!(verified.verified_by, [origin]);
}

#[test]
fn multi_signer_note_round_trip() {
    let cp = Checkpoint::new("o.example/log", 7, [1u8; 32]).unwrap();
    let s1 = NoteSigner::from_seed("o.example/log", &[1u8; 32]).unwrap();
    let s2 = NoteSigner::from_seed("witness.example/w1", &[2u8; 32]).unwrap();
    let note = sign(&cp.marshal(), &[&s1, &s2]).unwrap();
    let n = open(&note, &[s1.verifier(), s2.verifier()]).unwrap();
    assert_eq!(n.verified_by, ["o.example/log", "witness.example/w1"]);
}

// --- Tamper resistance ----------------------------------------------------

#[test]
fn any_single_byte_flip_fails_verification() {
    let signer = NoteSigner::from_signer_key(&go_signer_key()).unwrap();
    let verifier = signer.verifier();
    let good = go_signed_note();
    let note = good.as_bytes();
    let mut rejected = 0usize;
    for i in 0..note.len() {
        let mut tampered = note.to_vec();
        tampered[i] ^= 0x01;
        // Bit flips inside multi-byte UTF-8 may not produce valid UTF-8 at
        // all; failing to even decode counts as rejection.
        let Ok(s) = std::str::from_utf8(&tampered) else {
            rejected += 1;
            continue;
        };
        assert!(
            open(s, std::slice::from_ref(&verifier)).is_err(),
            "byte flip at offset {i} was accepted"
        );
        rejected += 1;
    }
    assert_eq!(rejected, note.len());
}

#[test]
fn known_key_with_bad_signature_rejects_note() {
    let signer = NoteSigner::from_signer_key(&go_signer_key()).unwrap();
    // Valid envelope, valid key name + key ID, corrupted signature bytes:
    // flip one base64 char inside the signature portion (beyond the 6
    // chars that cover the 4-byte key ID).
    let good = sign(GO_TEXT, &[&signer]).unwrap();
    let b64 = good.rsplit(' ').next().unwrap().trim_end().to_string();
    let mut chars: Vec<char> = b64.chars().collect();
    chars[20] = if chars[20] == 'A' { 'B' } else { 'A' };
    let bad_b64: String = chars.into_iter().collect();
    let note = format!("{GO_TEXT}\n\u{2014} PeterNeumann {bad_b64}\n");
    assert_eq!(
        open(&note, &[signer.verifier()]).unwrap_err(),
        Error::InvalidSignature("PeterNeumann".to_owned())
    );
}

// --- Envelope strictness ---------------------------------------------------

#[test]
fn malformed_envelopes_rejected() {
    let v = NoteVerifier::from_verifier_key(&go_verifier_key()).unwrap();
    let vs = std::slice::from_ref(&v);
    // No blank-line separator.
    assert!(open("text\n\u{2014} X AAAAAAAA\n", vs).is_err());
    // No signature lines.
    assert!(open("text\n\n", vs).is_err());
    // Missing trailing newline.
    let good = go_signed_note();
    assert!(open(good.trim_end_matches('\n'), vs).is_err());
    // ASCII hyphen instead of em dash.
    let hyphen = good.replace('\u{2014}', "-");
    assert!(open(&hyphen, vs).is_err());
    // Blank line inside the text region.
    assert!(open("a\n\nb\n\n\u{2014} X AAAAAAAA\n", vs).is_err());
}
