//! P6 install-time capability contract.

use rvforge_reader::capability::{CapabilityCard, CapabilityError, CardStatus, CAPABILITY_CLASSES};

/// The manifest shape from docs/research/rvf-forge/registry-model.md.
const SAMPLE_MANIFEST: &str = r#"{
  "schemaVersion": 1,
  "type": "capability-manifest",
  "manifestId": "sha256:aaaa",
  "defaultPolicy": "deny",
  "requests": [
    { "class": "filesystem", "scope": "user-selected", "rationale": "reads documents you choose" },
    { "class": "memory", "scope": "512MiB", "rationale": "model working set" },
    { "class": "persistent-state", "scope": "encrypted-local", "rationale": "agent memory" }
  ],
  "denials": ["network", "microphone", "process", "background", "external-model-providers"],
  "manualReviewTriggers": []
}"#;

fn manifest_with(requests: &str, denials: &str, default_policy: &str) -> String {
    format!(
        r#"{{"schemaVersion":1,"manifestId":"sha256:bbbb","defaultPolicy":"{default_policy}",
            "requests":[{requests}],"denials":[{denials}]}}"#
    )
}

#[test]
fn specific_scopes_render_as_specific_sentences() {
    let card = CapabilityCard::from_manifest_json(SAMPLE_MANIFEST).expect("sample must derive");
    assert_eq!(card.status, CardStatus::Derived);
    assert_eq!(card.manifest_id.as_deref(), Some("sha256:aaaa"));
    assert_eq!(card.requests.len(), 3);

    let texts: Vec<&str> = card.requests.iter().map(|l| l.text.as_str()).collect();
    assert!(texts.contains(&"Read the files and folders you select, and nothing else"));
    assert!(texts.contains(&"Use up to 512MiB of memory"));
    assert!(texts.contains(&"Keep encrypted memory on this computer between sessions"));
}

#[test]
fn explicit_denials_render_as_specific_sentences() {
    let card = CapabilityCard::from_manifest_json(SAMPLE_MANIFEST).unwrap();
    let texts: Vec<&str> = card.cannot.iter().map(|l| l.text.as_str()).collect();
    for expected in [
        "Access the internet",
        "Use the microphone",
        "Start other programs",
        "Run in the background",
        "Contact external model providers",
    ] {
        assert!(texts.contains(&expected), "missing denial: {expected}");
    }
}

#[test]
fn unrequested_classes_are_denied_by_default() {
    let card = CapabilityCard::from_manifest_json(SAMPLE_MANIFEST).unwrap();
    let requested: Vec<&str> = card.requests.iter().map(|l| l.class.as_str()).collect();
    for class in CAPABILITY_CLASSES {
        if requested.contains(&class) {
            continue;
        }
        assert!(
            card.cannot.iter().any(|l| l.class == class),
            "class '{class}' was neither requested nor denied"
        );
    }
}

#[test]
fn no_card_line_uses_banned_prose() {
    let card = CapabilityCard::from_manifest_json(SAMPLE_MANIFEST).unwrap();
    for line in card.requests.iter().chain(card.cannot.iter()) {
        let lowered = line.text.to_ascii_lowercase();
        for banned in ["access your computer", "full access", "unrestricted access"] {
            assert!(!lowered.contains(banned), "banned prose in: {}", line.text);
        }
    }
}

#[test]
fn vague_scopes_are_rejected() {
    for scope in [
        "all",
        "all-files",
        "*",
        "**/*",
        "unrestricted",
        "any",
        "",
        "  ",
    ] {
        let json = manifest_with(
            &format!(r#"{{"class":"filesystem","scope":"{scope}"}}"#),
            "",
            "deny",
        );
        match CapabilityCard::from_manifest_json(&json) {
            Err(CapabilityError::VagueScope { class, .. }) => assert_eq!(class, "filesystem"),
            other => panic!("scope '{scope}' should be rejected, got {other:?}"),
        }
    }
}

#[test]
fn banned_rationale_prose_is_rejected() {
    let json = manifest_with(
        r#"{"class":"filesystem","scope":"user-selected","rationale":"needs to access your computer"}"#,
        "",
        "deny",
    );
    assert!(matches!(
        CapabilityCard::from_manifest_json(&json),
        Err(CapabilityError::BannedPhrase { .. })
    ));
}

#[test]
fn a_non_deny_default_policy_is_rejected() {
    let json = manifest_with(r#"{"class":"clock","scope":"deterministic"}"#, "", "allow");
    assert!(matches!(
        CapabilityCard::from_manifest_json(&json),
        Err(CapabilityError::DefaultPolicyNotDeny(_))
    ));
}

#[test]
fn unknown_capability_classes_are_rejected() {
    let json = manifest_with(r#"{"class":"telepathy","scope":"local"}"#, "", "deny");
    assert!(matches!(
        CapabilityCard::from_manifest_json(&json),
        Err(CapabilityError::UnknownClass(_))
    ));
}

#[test]
fn a_class_cannot_be_both_requested_and_denied() {
    let json = manifest_with(
        r#"{"class":"network","scope":"api.example.com"}"#,
        r#""network""#,
        "deny",
    );
    assert!(matches!(
        CapabilityCard::from_manifest_json(&json),
        Err(CapabilityError::Contradiction(_))
    ));
}

#[test]
fn a_missing_manifest_denies_everything() {
    let card = CapabilityCard::manifest_unavailable("no manifest");
    assert_eq!(card.status, CardStatus::ManifestUnavailable);
    assert!(card.requests.is_empty());
    assert_eq!(card.cannot.len(), CAPABILITY_CLASSES.len());
    assert_eq!(card.default_policy, "deny");
}

#[test]
fn malformed_json_is_a_parse_error_not_a_permissive_card() {
    assert!(matches!(
        CapabilityCard::from_manifest_json("{ not json"),
        Err(CapabilityError::Parse(_))
    ));
}
