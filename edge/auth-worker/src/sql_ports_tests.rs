//! SQL adapter semantics against a real (in-memory) SQLite.

use super::*;
use crate::sql::memory::MemoryDb;
use ruvector_edge_authz::authorize::ValidatedAuthorization;
use ruvector_edge_authz::client::validate_redirect_uri;
use ruvector_edge_authz::federation::UpstreamIdentity;
use ruvector_edge_authz::ResourceUrl;

fn ports() -> SqlPorts<MemoryDb> {
    let p = SqlPorts::new(MemoryDb::new());
    p.migrate().unwrap();
    p.migrate().unwrap(); // idempotent
    p
}

fn identity() -> UpstreamIdentity {
    UpstreamIdentity {
        upstream_iss: "https://auth.cognitum.one".into(),
        sub: "u".into(),
        org_id: "o".into(),
        workspace_id: "w".into(),
    }
}

fn resource() -> ResourceUrl {
    ResourceUrl::parse("https://rs.example/v1/mcp").unwrap()
}

fn validated() -> ValidatedAuthorization {
    ValidatedAuthorization {
        client_id: "edc-1".into(),
        redirect_uri: "https://c.example/cb".into(),
        scopes: vec!["ruvector:read".into()],
        state: Some("s".into()),
        code_challenge: "c".repeat(43),
        resource: resource(),
    }
}

fn refresh(hash: u8, family: &str, expires_at: u64) -> RefreshTokenRecord {
    RefreshTokenRecord {
        token_hash: [hash; 32],
        family_id: family.into(),
        client_id: "edc-1".into(),
        identity: identity(),
        resource: resource(),
        scopes: vec!["ruvector:read".into()],
        expires_at,
        family_expires_at: expires_at,
        rotated: false,
    }
}

#[test]
fn clients_round_trip_count_and_reject_duplicates() {
    let p = ports();
    let rec = ClientRecord {
        client_id: "edc-1".into(),
        redirect_uris: vec![validate_redirect_uri("https://c.example/cb").unwrap()],
        grant_types: vec!["authorization_code".into()],
        scope: vec!["ruvector:read".into()],
        client_name: Some("n".into()),
        client_id_issued_at: 5,
    };
    assert_eq!(p.client_count().unwrap(), 0);
    p.insert_client(&rec).unwrap();
    assert!(p.insert_client(&rec).is_err());
    assert_eq!(p.get_client("edc-1").unwrap(), Some(rec));
    assert_eq!(p.get_client("edc-2").unwrap(), None);
    assert_eq!(p.client_count().unwrap(), 1);
}

#[test]
fn codes_and_flows_are_taken_exactly_once() {
    let p = ports();
    let code = AuthorizationCodeRecord {
        code_hash: [7; 32],
        authorization: validated(),
        identity: identity(),
        expires_at: 100,
    };
    p.insert_code(&code).unwrap();
    assert_eq!(p.take_code(&[7; 32]).unwrap(), Some(code));
    assert_eq!(p.take_code(&[7; 32]).unwrap(), None);
    let flow = UpstreamFlowState {
        state: "st".into(),
        nonce: "n".into(),
        upstream_code_verifier: "v".into(),
        browser_binding: [1; 32],
        downstream: validated(),
        expires_at: 100,
    };
    p.insert_flow(&flow).unwrap();
    assert_eq!(p.take_flow("st").unwrap(), Some(flow));
    assert_eq!(p.take_flow("st").unwrap(), None);
}

#[test]
fn refresh_rotation_is_compare_and_set() {
    let p = ports();
    p.insert_refresh(&refresh(1, "fam", 100)).unwrap();
    assert!(!p.get_refresh(&[1; 32]).unwrap().unwrap().rotated);
    assert!(p.mark_rotated(&[1; 32]).unwrap());
    assert!(!p.mark_rotated(&[1; 32]).unwrap());
    assert!(p.get_refresh(&[1; 32]).unwrap().unwrap().rotated);
    assert!(!p.mark_rotated(&[9; 32]).unwrap());
    assert_eq!(p.get_refresh(&[9; 32]).unwrap(), None);
}

#[test]
fn family_revocation_is_idempotent_and_scoped() {
    let p = ports();
    assert!(!p.is_family_revoked("fam").unwrap());
    p.revoke_family("fam").unwrap();
    p.revoke_family("fam").unwrap();
    assert!(p.is_family_revoked("fam").unwrap());
    assert!(!p.is_family_revoked("other").unwrap());
}

#[test]
fn purge_removes_only_expired_rows() {
    let p = ports();
    p.insert_refresh(&refresh(1, "a", 50)).unwrap();
    p.insert_refresh(&refresh(2, "b", 150)).unwrap();
    p.purge_expired(100).unwrap();
    assert_eq!(p.get_refresh(&[1; 32]).unwrap(), None);
    assert!(p.get_refresh(&[2; 32]).unwrap().is_some());
}

#[test]
fn assertion_jti_is_one_time_until_it_expires() {
    let p = ports();
    let h = [7u8; 32];
    assert!(p.record_assertion(&h, 200, 100).unwrap());
    assert!(!p.record_assertion(&h, 250, 150).unwrap(), "live replay");
    assert!(!p.record_assertion(&h, 250, 199).unwrap(), "still live");
    // At or after its expiry the slot is reusable (the assertion itself is
    // then expired, so this is never a replay window).
    assert!(p.record_assertion(&h, 400, 200).unwrap());
    assert!(!p.record_assertion(&h, 500, 399).unwrap());
    assert!(p.record_assertion(&[8u8; 32], 400, 150).unwrap(), "per jti");
    p.purge_expired(400).unwrap();
    let rows =
        p.db.query("SELECT COUNT(*) FROM assertion_jtis", vec![])
            .unwrap();
    assert_eq!(rows[0][0].as_int(), Some(0));
}
