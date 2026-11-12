//! ADR-288 state capsule layout and sealing.

use rvforge_reader::state::crypto::{recorded_identity, seal_with, unseal_with, InstallKey};
use rvforge_reader::state::{self, StateCapsule, StateError};

const BASE: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const OTHER: &str = "sha256:fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

fn capsule() -> StateCapsule {
    StateCapsule::new("/opt/rvforge/agent", "/var/lib/rvforge/state", BASE).unwrap()
}

#[test]
fn state_lives_outside_the_install_directory() {
    let err = StateCapsule::new("/opt/rvforge/agent", "/opt/rvforge/agent/state", BASE)
        .expect_err("state inside the install root must be refused");
    assert!(matches!(err, StateError::StateInsideInstallRoot { .. }));
}

#[test]
fn the_capsule_directory_is_keyed_by_base_identity() {
    let c = capsule();
    let dir = c.capsule_dir();
    assert!(dir.starts_with("/var/lib/rvforge/state"));
    assert!(dir.to_string_lossy().contains(&BASE.replace(':', "-")));
}

#[test]
fn checkpoint_and_delta_paths_follow_the_adr_288_layout() {
    let c = capsule();
    assert!(c
        .checkpoint_path(0)
        .to_string_lossy()
        .ends_with("checkpoint-000.ckpt"));
    assert!(c
        .delta_path(1)
        .to_string_lossy()
        .ends_with("delta-001.wdelta"));
    assert!(c
        .delta_path(12)
        .to_string_lossy()
        .ends_with("delta-012.wdelta"));
}

#[test]
fn the_base_artifact_stays_in_the_install_root() {
    let c = capsule();
    assert!(c.base_artifact_path().starts_with(c.install_root()));
    assert!(!c.base_artifact_path().starts_with(c.state_root()));
}

#[test]
fn state_from_another_lineage_is_rejected() {
    let c = capsule();
    assert!(c.check_lineage(BASE).is_ok());
    assert!(matches!(
        c.check_lineage(OTHER),
        Err(StateError::LineageRejected { .. })
    ));
}

#[test]
fn a_base_identity_must_be_a_content_address() {
    for bad in ["", "not-a-hash", "sha256:short", "md5:0123"] {
        assert!(matches!(
            StateCapsule::new("/opt/a", "/var/b", bad),
            Err(StateError::InvalidBaseIdentity(_))
        ));
    }
}

fn key(seed: u8) -> InstallKey {
    InstallKey::from_bytes([seed; 32])
}

#[test]
fn a_sealed_capsule_round_trips() {
    let k = key(1);
    let sealed = seal_with(&k, BASE, b"agent memory").unwrap();
    assert_eq!(unseal_with(&k, BASE, &sealed).unwrap(), b"agent memory");
}

#[test]
fn the_plaintext_is_not_in_the_sealed_bytes() {
    let sealed = seal_with(&key(1), BASE, b"a distinctive secret").unwrap();
    assert!(
        !sealed
            .windows(20)
            .any(|w| w == b"a distinctive secret"),
        "sealing must not write plaintext"
    );
}

#[test]
fn two_sealings_of_the_same_payload_differ() {
    let k = key(1);
    let a = seal_with(&k, BASE, b"same").unwrap();
    let b = seal_with(&k, BASE, b"same").unwrap();
    assert_ne!(a, b, "each capsule carries its own random nonce");
    assert_eq!(unseal_with(&k, BASE, &a).unwrap(), b"same");
    assert_eq!(unseal_with(&k, BASE, &b).unwrap(), b"same");
}

#[test]
fn the_capsule_header_records_the_base_identity() {
    let sealed = seal_with(&key(1), BASE, b"payload").unwrap();
    assert_eq!(recorded_identity(&sealed).unwrap(), BASE);
}

#[test]
fn state_from_another_lineage_does_not_open() {
    let k = key(1);
    let sealed = seal_with(&k, BASE, b"payload").unwrap();
    assert!(matches!(
        unseal_with(&k, OTHER, &sealed),
        Err(StateError::LineageRejected { .. })
    ));
}

#[test]
fn relabelling_a_capsule_to_another_lineage_fails_authentication() {
    let k = key(1);
    let mut sealed = seal_with(&k, BASE, b"payload").unwrap();
    // Rewrite the recorded identity in place: same length, different lineage.
    let start = sealed
        .windows(BASE.len())
        .position(|w| w == BASE.as_bytes())
        .expect("the identity is in the header");
    sealed[start..start + BASE.len()].copy_from_slice(OTHER.as_bytes());

    // The header now claims the other lineage, so it is no longer refused on
    // lineage — and the AEAD refuses it instead, because the identity is AAD.
    assert_eq!(unseal_with(&k, OTHER, &sealed), Err(StateError::OpenFailed));
}

#[test]
fn a_wrong_key_fails_cleanly() {
    let sealed = seal_with(&key(1), BASE, b"payload").unwrap();
    assert_eq!(
        unseal_with(&key(2), BASE, &sealed),
        Err(StateError::OpenFailed),
        "a wrong key is refused without saying it was the key"
    );
}

#[test]
fn tampered_ciphertext_fails_cleanly() {
    let k = key(1);
    let mut sealed = seal_with(&k, BASE, b"payload").unwrap();
    let last = sealed.len() - 1;
    sealed[last] ^= 0x01;
    assert_eq!(unseal_with(&k, BASE, &sealed), Err(StateError::OpenFailed));
}

#[test]
fn a_truncated_or_foreign_capsule_is_malformed_not_opened() {
    let k = key(1);
    for bytes in [b"".as_slice(), b"short".as_slice(), b"NOTASTATECAPSULE".as_slice()] {
        assert!(matches!(
            unseal_with(&k, BASE, bytes),
            Err(StateError::MalformedCapsule(_))
        ));
    }
}

#[test]
fn sealing_refuses_an_identity_that_is_not_a_content_address() {
    assert!(matches!(
        seal_with(&key(1), "not-a-hash", b"payload"),
        Err(StateError::InvalidBaseIdentity(_))
    ));
}

#[test]
fn the_install_key_is_stable_and_owner_only() {
    let root = std::env::temp_dir().join(format!("rvforge-key-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    let first = InstallKey::load_or_create(&root).expect("first use creates the key");
    let second = InstallKey::load_or_create(&root).expect("second use reuses it");

    // Same key: a capsule sealed by one load opens under the next.
    let sealed = seal_with(&first, BASE, b"payload").unwrap();
    assert_eq!(unseal_with(&second, BASE, &sealed).unwrap(), b"payload");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(root.join("install.key"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "the state key must be owner-only");
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_capsule_seals_and_opens_through_its_own_lineage() {
    let root = std::env::temp_dir().join(format!("rvforge-capsule-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let c = StateCapsule::new(root.join("install"), root.join("state"), BASE).unwrap();

    let sealed = c.seal(b"checkpoint").expect("seal");
    assert_eq!(c.unseal(&sealed).expect("unseal"), b"checkpoint");

    let foreign = StateCapsule::new(root.join("install"), root.join("state"), OTHER).unwrap();
    assert!(matches!(
        foreign.unseal(&sealed),
        Err(StateError::LineageRejected { .. })
    ));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_default_state_root_is_not_the_current_directory() {
    let root = state::default_state_root();
    assert!(root.to_string_lossy().contains("io.ruv.rvforge.reader"));
    assert!(root.ends_with("state"));
}
