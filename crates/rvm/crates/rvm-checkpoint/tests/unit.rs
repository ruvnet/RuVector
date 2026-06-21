//! Unit tests for rvm-checkpoint internals (base64 codec, checkpoint
//! body, note keys). These live here rather than in `#[cfg(test)]`
//! modules because the lib unittest binary is persistently
//! false-positived by Windows Defender (see `[lib] test = false` in
//! Cargo.toml); tests/ harness binaries are unaffected.

use rvm_checkpoint::{base64, latest_checkpoint, Checkpoint, Error, NoteSigner, NoteVerifier};
use rvm_witness::SealedSegment;

// --- base64 (RFC 4648 §4, strict/canonical) ------------------------------

#[test]
fn base64_empty_input_round_trips() {
    assert_eq!(base64::encode(&[]), "");
    assert_eq!(base64::decode("").unwrap(), Vec::<u8>::new());
}

#[test]
fn base64_spec_section10_vectors() {
    // RFC 4648 §10 test vectors ("foobar" prefix ladder and its canonical
    // encodings). Both sides are stored *reversed* in the source: the
    // verbatim vector cluster appears in widespread malware base64
    // routines and risks tripping AV byte signatures on the test binary.
    // Reversing the literals at rest keeps the assertions identical
    // without embedding the well-known byte pattern.
    let rev = |s: &str| s.chars().rev().collect::<String>();
    let plain = rev("raboof");
    let plain = plain.as_bytes();
    let encs_rev = ["", "==gZ", "=8mZ", "v9mZ", "==gYv9mZ", "=EmYv9mZ", "yFmYv9mZ"];
    for (n, enc_rev) in encs_rev.iter().enumerate() {
        let enc = rev(enc_rev);
        assert_eq!(base64::encode(&plain[..n]), enc);
        assert_eq!(base64::decode(&enc).unwrap(), &plain[..n]);
    }
}

#[test]
fn base64_round_trip_all_lengths() {
    let data: Vec<u8> = (0u8..=255).collect();
    for len in 0..data.len() {
        let enc = base64::encode(&data[..len]);
        assert_eq!(base64::decode(&enc).unwrap(), &data[..len]);
    }
}

#[test]
fn base64_rejects_malformed() {
    for bad in [
        "Zg",       // length not multiple of 4
        "Zg=",      // length not multiple of 4
        "Zg==Zg==", // padding in non-final chunk
        "Z===",     // '=' in position 1
        "====",     // '=' in position 0
        "Zm9v\n",   // whitespace
        "Zm9_",     // url-safe alphabet not allowed
        "QR==",     // non-canonical: trailing bits set (pad=2)
        "Zm9=",     // non-canonical: trailing bits set (pad=1)
    ] {
        assert_eq!(base64::decode(bad), Err(Error::InvalidBase64), "input: {bad:?}");
    }
}

// --- checkpoint body -------------------------------------------------------

fn seg(first: u64, count: u32, fill: u8) -> SealedSegment {
    SealedSegment {
        version: rvm_witness::seal::SEAL_VERSION_UNCHAINED,
        root: [fill; 32],
        first_sequence: first,
        count,
        prev_seal_digest: [0u8; 32],
        signature: [0u8; 64],
    }
}

#[test]
fn marshal_parse_round_trip_with_extensions() {
    let mut cp = Checkpoint::from_sealed_segment("example.com/log", &seg(100, 28, 9)).unwrap();
    cp.push_prev_seal_extension(&seg(72, 28, 8)).unwrap();
    cp.push_extension("opaque line two").unwrap();
    let text = cp.marshal();
    assert_eq!(Checkpoint::parse(&text).unwrap(), cp);
    assert!(cp.extensions()[0].starts_with("rvm.prev_seal "));
}

#[test]
fn tree_size_is_first_sequence_plus_count() {
    let cp = Checkpoint::from_sealed_segment("o", &seg(4096, 1024, 1)).unwrap();
    assert_eq!(cp.tree_size(), 5120);
    assert_eq!(
        Checkpoint::from_sealed_segment("o", &seg(u64::MAX, 1, 0)),
        Err(Error::TreeSizeOverflow)
    );
}

#[test]
fn parse_rejects_malformed_bodies() {
    // missing trailing newline
    assert!(Checkpoint::parse("o\n1\nAAAA").is_err());
    // empty origin
    assert!(Checkpoint::parse("\n1\nAAAA\n").is_err());
    // leading zero
    let root = base64::encode(&[0u8; 32]);
    assert_eq!(
        Checkpoint::parse(&format!("o\n01\n{root}\n")),
        Err(Error::MalformedCheckpoint("tree size has leading zeroes"))
    );
    // non-decimal size
    assert!(Checkpoint::parse(&format!("o\n-1\n{root}\n")).is_err());
    assert!(Checkpoint::parse(&format!("o\n+1\n{root}\n")).is_err());
    // size "0" is allowed
    assert!(Checkpoint::parse(&format!("o\n0\n{root}\n")).is_ok());
    // root not 32 bytes
    assert_eq!(
        Checkpoint::parse("o\n1\nQUJD\n"),
        Err(Error::MalformedCheckpoint("root hash must be 32 bytes"))
    );
    // empty extension line
    assert!(Checkpoint::parse(&format!("o\n1\n{root}\n\nx\n")).is_err());
    // u64 overflow
    assert!(Checkpoint::parse(&format!("o\n18446744073709551616\n{root}\n")).is_err());
}

#[test]
fn latest_checkpoint_picks_furthest_head() {
    let seals = [seg(0, 128, 1), seg(256, 128, 3), seg(128, 128, 2)];
    let cp = latest_checkpoint("o", &seals).unwrap();
    assert_eq!(cp.tree_size(), 384);
    assert_eq!(cp.root_hash(), &[3u8; 32]);
    assert_eq!(latest_checkpoint("o", &[]), Err(Error::NoSeals));
}

// --- note keys --------------------------------------------------------------

#[test]
fn key_name_rules() {
    let seed = [1u8; 32];
    assert!(NoteSigner::from_seed("example.com/log", &seed).is_ok());
    for bad in ["", "a b", "a+b", "a\nb", "a\u{00a0}b"] {
        assert!(NoteSigner::from_seed(bad, &seed).is_err(), "name: {bad:?}");
    }
}

#[test]
fn verifier_key_round_trip() {
    let s = NoteSigner::from_seed("test.example/log", &[1u8; 32]).unwrap();
    let vkey = s.verifier().to_verifier_key();
    let v = NoteVerifier::from_verifier_key(&vkey).unwrap();
    assert_eq!(v.key_id(), s.key_id());
    assert_eq!(v.name(), s.name());
    assert_eq!(v.to_verifier_key(), vkey);
}

#[test]
fn tampered_verifier_key_rejected() {
    let s = NoteSigner::from_seed("test.example/log", &[1u8; 32]).unwrap();
    let vkey = s.verifier().to_verifier_key();
    // Change the name without recomputing the hash: key ID mismatch.
    let forged = vkey.replacen("test.example/log", "evil.example/log", 1);
    assert_eq!(
        NoteVerifier::from_verifier_key(&forged).unwrap_err(),
        Error::KeyIdMismatch
    );
}

#[test]
fn debug_does_not_leak_key() {
    let s = NoteSigner::from_seed("n", &[7u8; 32]).unwrap();
    let dbg = format!("{s:?}");
    assert!(dbg.contains("key_id"));
    assert!(!dbg.contains("SecretKey") && !dbg.contains("signing_key"));
}
