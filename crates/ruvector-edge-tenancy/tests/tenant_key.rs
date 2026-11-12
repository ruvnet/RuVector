//! ADR-351 §4.1 / §9 "Tenancy": tenant-key derivation, charset validation,
//! separator injection and Unicode edge cases.

use proptest::prelude::*;
use ruvector_edge_tenancy::names::TENANT_KEY_LEN;
use ruvector_edge_tenancy::{
    derive_tenant_key, validate_claim_component, validate_issuer, TenancyError, TenantKey,
};

const ISS: &str = "https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev";
const ORG: &str = "3f2b8c1e-7d4a-4e9b-a1c2-5f6e7d8c9b0a";
const WS: &str = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d";

// Pins the preimage `v1|iss|org|ws` -> sha256 -> base32 lowercase[0..26].
// Changing this value re-keys every tenant.
const GOLDEN_TENANT_KEY: &str = "llnpuxs5xwlf4uk6isy5lzclps";

#[test]
fn golden_tenant_key_is_stable() {
    let k = derive_tenant_key(ISS, ORG, WS).unwrap();
    assert_eq!(k.as_str(), GOLDEN_TENANT_KEY);
}

#[test]
fn key_shape_is_26_lowercase_base32() {
    let k = derive_tenant_key(ISS, ORG, WS).unwrap();
    assert_eq!(k.as_str().len(), TENANT_KEY_LEN);
    assert!(k
        .as_str()
        .bytes()
        .all(|b| b.is_ascii_lowercase() || (b'2'..=b'7').contains(&b)));
    assert_eq!(TenantKey::parse(k.as_str()).unwrap(), k);
    assert_eq!(k.to_string(), k.as_str());
}

#[test]
fn deterministic() {
    assert_eq!(
        derive_tenant_key(ISS, ORG, WS).unwrap(),
        derive_tenant_key(ISS, ORG, WS).unwrap()
    );
}

#[test]
fn every_component_matters() {
    let base = derive_tenant_key(ISS, ORG, WS).unwrap();
    assert_ne!(
        base,
        derive_tenant_key("https://auth.cognitum.one", ORG, WS).unwrap()
    );
    assert_ne!(base, derive_tenant_key(ISS, WS, WS).unwrap());
    assert_ne!(base, derive_tenant_key(ISS, ORG, ORG).unwrap());
    // Swapping org and workspace is a different tenant.
    assert_ne!(base, derive_tenant_key(ISS, WS, ORG).unwrap());
}

#[test]
fn case_is_significant_no_folding() {
    let lower = derive_tenant_key(ISS, "abc", "ws").unwrap();
    let upper = derive_tenant_key(ISS, "ABC", "ws").unwrap();
    assert_ne!(lower, upper);
}

#[test]
fn separator_injection_is_rejected_not_collided() {
    // Without validation both would hash `v1|iss|a|b|c`.
    assert_eq!(
        derive_tenant_key(ISS, "a|b", "c"),
        Err(TenancyError::InvalidTenantClaim("org_id"))
    );
    assert_eq!(
        derive_tenant_key(ISS, "a", "b|c"),
        Err(TenancyError::InvalidTenantClaim("workspace_id"))
    );
    // An issuer carrying a delimiter cannot absorb the org component.
    assert_eq!(
        derive_tenant_key("https://x|evil", "org", "ws"),
        Err(TenancyError::InvalidTenantClaim("iss"))
    );
    // Other structural characters.
    for bad in ["a:b", "a/b", "a\0b", "a b", "a.b", "a\nb", "a%7Cb", "a\\b"] {
        assert!(derive_tenant_key(ISS, bad, "ws").is_err(), "{bad:?}");
        assert!(derive_tenant_key(ISS, "org", bad).is_err(), "{bad:?}");
    }
}

#[test]
fn charset_validator_matches_adr_section_9_list() {
    for bad in [":", "/", "|", "\0", "é", "a:b", "a/b", "a|b", "x\0"] {
        assert!(validate_claim_component(bad, "org_id").is_err(), "{bad:?}");
    }
    assert!(validate_claim_component("", "org_id").is_err());
    assert!(validate_claim_component(&"a".repeat(65), "org_id").is_err());
    assert!(validate_claim_component(&"a".repeat(64), "org_id").is_ok());
    assert!(validate_claim_component(ORG, "org_id").is_ok());
    assert!(validate_claim_component("Org_ID-9", "org_id").is_ok());
}

#[test]
fn unicode_homoglyphs_and_normalization_variants_rejected() {
    let cases = [
        "\u{FF11}23",  // fullwidth digit one
        "\u{0430}bc",  // Cyrillic a
        "e\u{0301}",   // e + combining acute (NFD)
        "\u{00E9}",    // precomposed e-acute (NFC)
        "ab\u{200D}c", // zero-width joiner
        "ab\u{200B}c", // zero-width space
        "\u{202E}cba", // right-to-left override
        "abc\u{FEFF}", // BOM
        "\u{2010}",    // Unicode hyphen, looks like '-'
        "\u{FF3F}",    // fullwidth low line, looks like '_'
        "\u{0660}",    // Arabic-indic digit zero
        "\u{212A}",    // Kelvin sign (NFKC-folds to 'K')
    ];
    for bad in cases {
        assert!(validate_claim_component(bad, "org_id").is_err(), "{bad:?}");
        assert!(derive_tenant_key(ISS, bad, "ws").is_err(), "{bad:?}");
    }
}

#[test]
fn length_is_bytes_not_chars() {
    // 64 bytes of ASCII passes; 32 two-byte chars (64 bytes) fails on charset.
    assert!(validate_claim_component(&"z".repeat(64), "w").is_ok());
    assert!(validate_claim_component(&"é".repeat(32), "w").is_err());
}

#[test]
fn issuer_validation() {
    assert!(validate_issuer(ISS).is_ok());
    assert!(validate_issuer("https://auth.cognitum.one").is_ok());
    for bad in [
        "",
        "https://a|b",
        "https://a b",
        "https://a\tb",
        "https://\u{0430}uth.cognitum.one",
        "https://a\u{7F}",
    ] {
        assert!(validate_issuer(bad).is_err(), "{bad:?}");
    }
    assert!(validate_issuer(&format!("https://{}", "a".repeat(248))).is_ok());
    assert!(validate_issuer(&format!("https://{}", "a".repeat(249))).is_err());
}

#[test]
fn tenant_key_parse_is_strict() {
    let good = derive_tenant_key(ISS, ORG, WS).unwrap();
    assert!(TenantKey::parse(&good.as_str().to_ascii_uppercase()).is_err());
    assert!(TenantKey::parse(&good.as_str()[..25]).is_err());
    assert!(TenantKey::parse(&format!("{}a", good.as_str())).is_err());
    assert!(TenantKey::parse("0000000000000000000000000a").is_err()); // 0,1,8,9 not base32
    assert!(TenantKey::parse("").is_err());
}

fn component() -> impl Strategy<Value = String> {
    "[A-Za-z0-9_-]{1,64}"
}

fn issuer() -> impl Strategy<Value = String> {
    "https://[a-z0-9.-]{1,40}(/[a-z0-9]{0,10})?"
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn prop_valid_components_always_derive(iss in issuer(), o in component(), w in component()) {
        let k = derive_tenant_key(&iss, &o, &w).unwrap();
        prop_assert_eq!(k.as_str().len(), TENANT_KEY_LEN);
        prop_assert_eq!(TenantKey::parse(k.as_str()).unwrap(), k.clone());
        prop_assert_eq!(derive_tenant_key(&iss, &o, &w).unwrap(), k);
    }

    #[test]
    fn prop_distinct_triples_distinct_keys(
        iss in issuer(), o1 in component(), w1 in component(),
        iss2 in issuer(), o2 in component(), w2 in component(),
    ) {
        prop_assume!((&iss, &o1, &w1) != (&iss2, &o2, &w2));
        prop_assert_ne!(
            derive_tenant_key(&iss, &o1, &w1).unwrap(),
            derive_tenant_key(&iss2, &o2, &w2).unwrap()
        );
    }

    #[test]
    fn prop_issuers_are_separated(iss in issuer(), iss2 in issuer(), o in component(), w in component()) {
        prop_assume!(iss != iss2);
        prop_assert_ne!(
            derive_tenant_key(&iss, &o, &w).unwrap(),
            derive_tenant_key(&iss2, &o, &w).unwrap()
        );
    }

    /// Any split of a string containing '|' into (org, ws) is rejected, so
    /// boundary-shifting collisions are impossible.
    #[test]
    fn prop_pipe_anywhere_rejected(a in "[A-Za-z0-9]{0,20}", b in "[A-Za-z0-9]{0,20}") {
        let with_pipe = format!("{}|{}", a, b);
        prop_assert!(derive_tenant_key("https://i", &with_pipe, "ws").is_err());
        prop_assert!(derive_tenant_key("https://i", "org", &with_pipe).is_err());
        let piped_issuer = ["https://i", with_pipe.as_str()].concat();
        prop_assert!(derive_tenant_key(&piped_issuer, "org", "ws").is_err());
    }

    /// Any non-ASCII character anywhere in a component is rejected.
    #[test]
    fn prop_non_ascii_rejected(a in "[a-z]{0,10}", c in any::<char>().prop_filter("non-ascii", |c| !c.is_ascii()), b in "[a-z]{0,10}") {
        let v = format!("{}{}{}", a, c, b);
        prop_assert!(validate_claim_component(&v, "org_id").is_err());
    }
}
