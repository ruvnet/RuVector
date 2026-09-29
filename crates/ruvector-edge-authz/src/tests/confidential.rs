//! Operator registry of confidential exchange clients: validated whole at
//! load, never reachable through DCR.

use super::exchange::{adapter_key, entry, thumb, ADAPTER};
use crate::confidential::{ConfidentialClients, MAX_CONFIDENTIAL_CLIENTS};
use crate::resource::GATEWAY_V1_URL;
use crate::testing::*;
use crate::ResourceAllowlist;
use p256::ecdsa::SigningKey;
use ruvector_edge_auth::AuthError;
use serde_json::{json, Value};

fn load(v: &Value) -> Result<ConfidentialClients, AuthError> {
    ConfidentialClients::from_config(&v.to_string(), &allowlist())
}

fn good() -> Value {
    entry(
        ADAPTER,
        &adapter_key(),
        &[TEAM_RESOURCE],
        "ruvector:read ruvector:write",
    )
}

#[test]
fn empty_registry_is_valid() {
    for v in ["", "   ", "[]"] {
        let c = ConfidentialClients::from_config(v, &allowlist()).unwrap();
        assert!(c.clients().is_empty());
    }
}

#[test]
fn loads_a_valid_adapter() {
    let c = load(&json!([good()])).unwrap();
    let a = c.get(ADAPTER).unwrap();
    assert_eq!(a.client_id(), ADAPTER);
    assert_eq!(a.kid(), thumb(&adapter_key()));
    assert_eq!(a.scope(), ["ruvector:read", "ruvector:write"]);
    assert!(a.allows_subject_audience(TEAM_RESOURCE));
    assert!(!a.allows_subject_audience(GATEWAY_V1_URL));
    assert!(c.get("edc-anything").is_none());
}

#[test]
fn every_invalid_registry_fails_the_load() {
    let k = adapter_key();
    let with = |f: &dyn Fn(&mut Value)| {
        let mut e = good();
        f(&mut e);
        json!([e])
    };
    let second = SigningKey::from_bytes(&[5u8; 32].into()).unwrap();
    let cases: Vec<(Value, &str)> = vec![
        (json!({"not": "an array"}), "object"),
        (
            json!([
                good(),
                entry(ADAPTER, &second, &[TEAM_RESOURCE], "ruvector:read")
            ]),
            "duplicate id",
        ),
        (with(&|e| e["client_id"] = json!("edc-0123")), "DCR prefix"),
        (with(&|e| e["client_id"] = json!("")), "empty id"),
        (with(&|e| e["client_id"] = json!("team ruv")), "id charset"),
        (
            with(&|e| e["client_id"] = json!("x".repeat(129))),
            "id length",
        ),
        (with(&|e| e["jwk"]["d"] = json!("AAAA")), "private jwk"),
        (with(&|e| e["jwk"]["k"] = json!("AAAA")), "symmetric jwk"),
        (
            with(&|e| e["jwk"]["kid"] = json!("not-the-thumbprint")),
            "kid != thumbprint",
        ),
        (
            with(&|e| {
                e["jwk"].as_object_mut().unwrap().remove("kid");
            }),
            "kid missing",
        ),
        (with(&|e| e["jwk"]["crv"] = json!("P-384")), "curve"),
        (with(&|e| e["jwk"]["alg"] = json!("RS256")), "alg"),
        (with(&|e| e["extra"] = json!(1)), "unknown member"),
        (
            with(&|e| {
                e.as_object_mut().unwrap().remove("scope");
            }),
            "scope missing",
        ),
        (
            with(&|e| e["subject_audiences"] = json!([])),
            "no audiences",
        ),
        (
            with(&|e| e["subject_audiences"] = json!([GATEWAY_V1_URL])),
            "target as subject",
        ),
        (
            with(&|e| e["subject_audiences"] = json!([TEAM_RESOURCE, TEAM_RESOURCE])),
            "repeated audience",
        ),
        (
            with(&|e| e["subject_audiences"] = json!(["https://team.ruv.io/mcp/"])),
            "non-canonical",
        ),
        (
            with(&|e| e["subject_audiences"] = json!(["https://evil.example/mcp"])),
            "uncompiled",
        ),
        (
            with(&|e| {
                e["subject_audiences"] =
                    json!([TEAM_RESOURCE, RESOURCE, TEAM_RESOURCE, RESOURCE, RESOURCE])
            }),
            "too many",
        ),
        (with(&|e| e["scope"] = json!("")), "empty ceiling"),
        (
            with(&|e| e["scope"] = json!("ruvector:read team:read")),
            "team scope in ceiling",
        ),
        (
            with(&|e| e["scope"] = json!("ruvector:read offline_access")),
            "offline_access ceiling",
        ),
        (
            with(&|e| e["scope"] = json!("ruvector:read  ruvector:write")),
            "malformed ceiling",
        ),
    ];
    for (v, why) in cases {
        assert!(load(&v).is_err(), "{why} must fail the load");
    }
    let many: Vec<Value> = (0..=MAX_CONFIDENTIAL_CLIENTS)
        .map(|i| entry(&format!("a{i}"), &k, &[TEAM_RESOURCE], "ruvector:read"))
        .collect();
    assert!(load(&json!(many)).is_err());
    let huge = format!("[{}]", " ".repeat(20 * 1024));
    assert!(ConfidentialClients::from_config(&huge, &allowlist()).is_err());
    assert!(ConfidentialClients::from_config("not json", &allowlist()).is_err());
}

#[test]
fn subject_audience_and_target_must_be_allowlisted() {
    let only_v1 = ResourceAllowlist::from_config(
        "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1 ruvector:read",
    )
    .unwrap();
    let reg = json!([good()]).to_string();
    assert!(ConfidentialClients::from_config(&reg, &only_v1).is_err());
    let only_team =
        ResourceAllowlist::from_config("https://team.ruv.io/mcp team:read team:write").unwrap();
    assert!(ConfidentialClients::from_config(&reg, &only_team).is_err());
    // Without clients the target need not be listed.
    assert!(ConfidentialClients::from_config("[]", &only_team).is_ok());
}

#[test]
fn gateway_resources_are_never_subject_audiences_and_have_no_exchange_map() {
    // `/v1/mcp` alone (and with an adapter resource) is refused: a user's
    // MCP token must never become a `/v1` token (ADR-351 §5.6, §12).
    for auds in [json!([RESOURCE]), json!([TEAM_RESOURCE, RESOURCE])] {
        let mut e = entry(ADAPTER, &adapter_key(), &[TEAM_RESOURCE], "ruvector:read");
        e["subject_audiences"] = auds.clone();
        assert!(load(&json!([e])).is_err(), "{auds}");
    }
    use crate::resource::{exchange_scope, Vocabulary};
    // Only an adapter vocabulary has a map; `ruvector:*` maps to nothing.
    for s in [
        "ruvector:read",
        "ruvector:write",
        "ruvector:admin",
        "team:read",
    ] {
        assert_eq!(exchange_scope(Vocabulary::Ruvector, s), None, "{s}");
    }
    assert_eq!(
        exchange_scope(Vocabulary::Team, "team:read"),
        Some("ruvector:read")
    );
    assert_eq!(
        exchange_scope(Vocabulary::Team, "team:write"),
        Some("ruvector:write")
    );
    for s in ["team:run", "offline_access", "mcp:invoke", "ruvector:read"] {
        assert_eq!(exchange_scope(Vocabulary::Team, s), None, "{s}");
    }
}
