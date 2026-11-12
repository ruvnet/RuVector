//! ADR-351 §6.1 quota leases, §7 problem codes, invisible-character policy.

use proptest::prelude::*;
use ruvector_edge_tenancy::validate::is_forbidden_char;
use ruvector_edge_tenancy::{
    grant_lease, reconcile, validate_edge_subject, validate_subject, LeaseAmount, LeaseError,
    ProblemCode, QuotaError, QuotaLimits, Usage, VectorId,
};
use std::collections::HashSet;

fn limits() -> QuotaLimits {
    QuotaLimits {
        max_collections: 20,
        max_vectors: 1_000,
        max_float_budget: 100_000,
        max_bytes: 1 << 20,
        max_daily_ops: 1_000,
    }
}

fn amt(rows: u64, floats: u64, bytes: u64) -> LeaseAmount {
    LeaseAmount {
        rows,
        floats,
        bytes,
    }
}

#[test]
fn lease_reserves_consumes_and_reconciles() {
    let (mut lease, usage) = grant_lease(
        &limits(),
        &Usage::default(),
        amt(100, 1_000, 4_096),
        300,
        50,
    )
    .unwrap();
    assert_eq!(usage.vectors, 100);
    assert_eq!(usage.float_budget, 1_000);
    assert_eq!(usage.bytes, 4_096);
    assert_eq!(lease.expires_at(), 350);
    lease.consume(amt(40, 400, 1_000), 100).unwrap();
    assert_eq!(lease.remaining(), amt(60, 600, 3_096));
    // Over-consumption is all-or-nothing.
    assert_eq!(
        lease.consume(amt(61, 0, 0), 100),
        Err(LeaseError::Exhausted)
    );
    assert_eq!(lease.used(), amt(40, 400, 1_000));
    // Expiry is exact at expires_at.
    assert_eq!(lease.consume(amt(1, 1, 1), 350), Err(LeaseError::Expired));
    assert!(lease.is_expired(350) && !lease.is_expired(349));
    // Reconcile returns the unused 60/600/3096.
    let back = reconcile(&usage, &lease).unwrap();
    assert_eq!(back.vectors, 40);
    assert_eq!(back.float_budget, 400);
    assert_eq!(back.bytes, 1_000);
}

#[test]
fn lease_respects_limits_ttl_and_overflow() {
    let full = Usage {
        vectors: 950,
        ..Usage::default()
    };
    assert_eq!(
        grant_lease(&limits(), &full, amt(51, 0, 0), 60, 0).unwrap_err(),
        QuotaError::Vectors
    );
    assert_eq!(
        grant_lease(&limits(), &Usage::default(), amt(u64::MAX, 0, 0), 60, 0).unwrap_err(),
        QuotaError::Vectors
    );
    let (lease, _) = grant_lease(
        &limits(),
        &Usage::default(),
        amt(1, 1, 1),
        u64::MAX,
        u64::MAX - 5,
    )
    .unwrap();
    assert_eq!(lease.expires_at(), u64::MAX, "saturates, never wraps");
    let (lease, _) = grant_lease(&limits(), &Usage::default(), amt(1, 1, 1), 86_400, 0).unwrap();
    assert_eq!(lease.expires_at(), 600, "ttl clamped to MAX_LEASE_TTL_SECS");
    let (mut lease, _) = grant_lease(&limits(), &Usage::default(), amt(1, 1, 1), 60, 0).unwrap();
    assert_eq!(
        lease.consume(amt(u64::MAX, 0, 0), 1),
        Err(LeaseError::Exhausted)
    );
    // Reconciling against usage smaller than the reservation is a ledger bug.
    assert_eq!(
        reconcile(&Usage::default(), &lease).unwrap_err(),
        QuotaError::Underflow
    );
}

#[test]
fn every_section7_code_has_its_status() {
    let expected: &[(ProblemCode, u16, &str)] = &[
        (ProblemCode::InvalidRequest, 400, "invalid_request"),
        (ProblemCode::DimensionMismatch, 400, "dimension_mismatch"),
        (ProblemCode::NonFiniteValue, 400, "non_finite_value"),
        (ProblemCode::InvalidToken, 401, "invalid_token"),
        (ProblemCode::InsufficientScope, 403, "insufficient_scope"),
        (ProblemCode::AudienceNotAllowed, 403, "audience_not_allowed"),
        (ProblemCode::RoleRequired, 403, "role_required"),
        (ProblemCode::NotClaimed, 403, "not_claimed"),
        (ProblemCode::TenantSuspended, 403, "tenant_suspended"),
        (ProblemCode::NotFound, 404, "not_found"),
        (ProblemCode::Conflict, 409, "conflict"),
        (ProblemCode::PayloadTooLarge, 413, "payload_too_large"),
        (ProblemCode::QuotaExceeded, 413, "quota_exceeded"),
        (ProblemCode::BudgetExceeded, 413, "budget_exceeded"),
        (
            ProblemCode::IdempotencyMismatch,
            422,
            "idempotency_mismatch",
        ),
        (ProblemCode::RateLimited, 429, "rate_limited"),
        (ProblemCode::JwksUnavailable, 503, "jwks_unavailable"),
        (ProblemCode::ShardUnavailable, 503, "shard_unavailable"),
        (ProblemCode::TrustRootMismatch, 503, "trust_root_mismatch"),
        (ProblemCode::ServerError, 500, "server_error"),
    ];
    assert_eq!(expected.len(), ProblemCode::ALL.len());
    let unique: HashSet<_> = ProblemCode::ALL.iter().collect();
    assert_eq!(unique.len(), ProblemCode::ALL.len());
    for (code, status, name) in expected {
        assert!(ProblemCode::ALL.contains(code));
        assert_eq!(code.status_and_code(), (*status, *name));
    }
}

const INVISIBLE_RANGES: &[(char, char)] = &[
    ('\u{00AD}', '\u{00AD}'),
    ('\u{180E}', '\u{180E}'),
    ('\u{2061}', '\u{2064}'),
    ('\u{206A}', '\u{206F}'),
    ('\u{FE00}', '\u{FE0F}'),
    ('\u{FFF9}', '\u{FFFB}'),
    ('\u{E0000}', '\u{E007F}'),
    ('\u{E0100}', '\u{E01EF}'),
];

#[test]
fn invisible_format_characters_are_forbidden_in_ids_and_sub() {
    for &(lo, hi) in INVISIBLE_RANGES {
        for c in [lo, hi] {
            assert!(is_forbidden_char(c), "U+{:04X}", c as u32);
            assert!(VectorId::parse(&format!("doc1{c}")).is_err());
            assert!(validate_subject(&format!("u{c}")).is_err());
        }
    }
    // Look-alike pair from the finding.
    assert!(VectorId::parse("doc1").is_ok());
    assert!(VectorId::parse("doc1\u{00AD}").is_err());
    // Ordinary text and emoji stay legal.
    assert!(VectorId::parse("naïve-文書-🚀").is_ok());
    assert!(validate_edge_subject("es1_abcdefghijklmnopqrstuvwxyz").is_ok());
}

fn invisible_char() -> impl Strategy<Value = char> {
    prop::sample::select(INVISIBLE_RANGES.to_vec())
        .prop_flat_map(|(lo, hi)| (lo as u32..=hi as u32).prop_map(|u| char::from_u32(u).unwrap()))
}

proptest! {
    #[test]
    fn prop_any_invisible_char_anywhere_rejects_vector_id(
        prefix in "[a-z0-9]{0,8}", c in invisible_char(), suffix in "[a-z0-9]{0,8}"
    ) {
        let id = format!("{prefix}{c}{suffix}");
        prop_assert!(VectorId::parse(&id).is_err());
        prop_assert!(validate_subject(&id).is_err());
    }
}
