//! ADR-351 §10 quota admission and §4.3 name/id validators.

use proptest::prelude::*;
use ruvector_edge_tenancy::quota::{dimension_ok, float_cost, limits};
use ruvector_edge_tenancy::validate::is_forbidden_char;
use ruvector_edge_tenancy::{
    admit, CollectionName, ProblemCode, QuotaDelta, QuotaError, QuotaLimits, TenancyError, Usage,
    VectorId,
};

const LIMITS: QuotaLimits = QuotaLimits {
    max_collections: 20,
    max_vectors: 1_000,
    max_float_budget: 384_000,
    max_bytes: 10_000,
    max_daily_ops: 100,
};

fn at_limit() -> Usage {
    Usage {
        collections: 20,
        vectors: 1_000,
        float_budget: 384_000,
        bytes: 10_000,
        daily_ops: 100,
    }
}

#[test]
fn admit_within_limits_returns_new_usage() {
    let d = QuotaDelta {
        collections: 1,
        vectors: 10,
        float_budget: 3840,
        bytes: 100,
        ops: 1,
    };
    let u = admit(&LIMITS, &Usage::default(), &d).unwrap();
    assert_eq!(
        u,
        Usage {
            collections: 1,
            vectors: 10,
            float_budget: 3840,
            bytes: 100,
            daily_ops: 1
        }
    );
    // Exactly at the limit is allowed.
    let full = QuotaDelta {
        collections: 20,
        vectors: 1000,
        float_budget: 384_000,
        bytes: 10_000,
        ops: 100,
    };
    assert_eq!(
        admit(&LIMITS, &Usage::default(), &full).unwrap(),
        at_limit()
    );
}

#[test]
fn each_limit_reports_its_own_error() {
    let u = at_limit();
    let cases = [
        (
            QuotaDelta {
                collections: 1,
                ..Default::default()
            },
            QuotaError::Collections,
        ),
        (
            QuotaDelta {
                vectors: 1,
                ..Default::default()
            },
            QuotaError::Vectors,
        ),
        (
            QuotaDelta {
                float_budget: 1,
                ..Default::default()
            },
            QuotaError::FloatBudget,
        ),
        (
            QuotaDelta {
                bytes: 1,
                ..Default::default()
            },
            QuotaError::Bytes,
        ),
        (
            QuotaDelta {
                ops: 1,
                ..Default::default()
            },
            QuotaError::DailyOps,
        ),
    ];
    for (d, e) in cases {
        assert_eq!(admit(&LIMITS, &u, &d), Err(e));
        assert_eq!(e.problem_code(), ProblemCode::QuotaExceeded);
    }
}

#[test]
fn first_violation_in_field_order_wins() {
    let d = QuotaDelta {
        collections: 1,
        vectors: 1,
        float_budget: 1,
        bytes: 1,
        ops: 1,
    };
    assert_eq!(
        admit(&LIMITS, &at_limit(), &d),
        Err(QuotaError::Collections)
    );
    let d = QuotaDelta {
        bytes: 1,
        ops: 1,
        ..Default::default()
    };
    assert_eq!(admit(&LIMITS, &at_limit(), &d), Err(QuotaError::Bytes));
}

#[test]
fn releases_succeed_even_when_over_a_lowered_limit() {
    let over = Usage {
        collections: 25,
        vectors: 5_000,
        float_budget: 1_000_000,
        bytes: 50_000,
        daily_ops: 0,
    };
    let d = QuotaDelta {
        collections: -1,
        vectors: -100,
        float_budget: -38_400,
        bytes: -1_000,
        ops: 0,
    };
    let u = admit(&LIMITS, &over, &d).unwrap();
    assert_eq!(u.collections, 24);
    assert_eq!(u.vectors, 4_900);
    // A zero delta on an over-limit field is also fine.
    assert!(admit(&LIMITS, &over, &QuotaDelta::default()).is_ok());
}

#[test]
fn underflow_is_detected_not_wrapped() {
    let u = Usage::default();
    for d in [
        QuotaDelta {
            collections: -1,
            ..Default::default()
        },
        QuotaDelta {
            vectors: -1,
            ..Default::default()
        },
        QuotaDelta {
            float_budget: i64::MIN,
            ..Default::default()
        },
        QuotaDelta {
            bytes: -1,
            ..Default::default()
        },
    ] {
        let e = admit(&LIMITS, &u, &d).unwrap_err();
        assert_eq!(e, QuotaError::Underflow);
        assert_eq!(e.problem_code(), ProblemCode::ServerError);
    }
}

#[test]
fn overflow_is_a_limit_error_not_a_panic() {
    let huge = QuotaLimits {
        max_collections: u32::MAX,
        max_vectors: u64::MAX,
        max_float_budget: u64::MAX,
        max_bytes: u64::MAX,
        max_daily_ops: u64::MAX,
    };
    let u = Usage {
        collections: u32::MAX,
        vectors: u64::MAX,
        float_budget: u64::MAX,
        bytes: u64::MAX,
        daily_ops: u64::MAX,
    };
    let cases = [
        (
            QuotaDelta {
                collections: 1,
                ..Default::default()
            },
            QuotaError::Collections,
        ),
        (
            QuotaDelta {
                vectors: i64::MAX,
                ..Default::default()
            },
            QuotaError::Vectors,
        ),
        (
            QuotaDelta {
                float_budget: 1,
                ..Default::default()
            },
            QuotaError::FloatBudget,
        ),
        (
            QuotaDelta {
                bytes: 1,
                ..Default::default()
            },
            QuotaError::Bytes,
        ),
        (
            QuotaDelta {
                ops: u64::MAX,
                ..Default::default()
            },
            QuotaError::DailyOps,
        ),
    ];
    for (d, e) in cases {
        assert_eq!(admit(&huge, &u, &d), Err(e));
    }
    // Releasing from the maximum still works.
    let d = QuotaDelta {
        vectors: i64::MIN,
        ..Default::default()
    };
    assert_eq!(
        admit(&huge, &u, &d).unwrap().vectors,
        u64::MAX - (1u64 << 63)
    );
}

#[test]
fn float_cost_and_dimension() {
    assert_eq!(float_cost(384, 10), Some(3840));
    assert_eq!(float_cost(u32::MAX, u64::MAX), None);
    assert!(!dimension_ok(0));
    assert!(dimension_ok(1));
    assert!(dimension_ok(limits::MAX_DIMENSION));
    assert!(!dimension_ok(limits::MAX_DIMENSION + 1));
    assert_eq!(limits::MAX_COLLECTIONS, 20);
}

proptest! {
    #[test]
    fn prop_admit_is_total_and_respects_limits(
        c in any::<u32>(), v in any::<u64>(), f in any::<u64>(), b in any::<u64>(), o in any::<u64>(),
        dc in any::<i32>(), dv in any::<i64>(), df in any::<i64>(), db in any::<i64>(), dops in any::<u64>(),
    ) {
        let u = Usage { collections: c, vectors: v, float_budget: f, bytes: b, daily_ops: o };
        let d = QuotaDelta { collections: dc, vectors: dv, float_budget: df, bytes: db, ops: dops };
        if let Ok(n) = admit(&LIMITS, &u, &d) {
            prop_assert!(dc <= 0 || n.collections <= LIMITS.max_collections);
            prop_assert!(dv <= 0 || n.vectors <= LIMITS.max_vectors);
            prop_assert!(df <= 0 || n.float_budget <= LIMITS.max_float_budget);
            prop_assert!(db <= 0 || n.bytes <= LIMITS.max_bytes);
            prop_assert!(dops == 0 || n.daily_ops <= LIMITS.max_daily_ops);
            prop_assert_eq!(i128::from(n.vectors) - i128::from(v), i128::from(dv));
        }
    }
}

// ---- collection names ----

#[test]
fn collection_names() {
    for ok in ["a", "0", "docs", "my-coll_2", &"a".repeat(63)] {
        assert!(CollectionName::parse(ok).is_ok(), "{ok:?}");
    }
    for bad in [
        "",
        "-a",
        "_a",
        "A",
        "Docs",
        "a b",
        "a.b",
        "a|b",
        "a/b",
        "a:b",
        "a\0",
        &"a".repeat(64),
        "caf\u{00E9}",
        "\u{0430}",
        "a\u{200D}b",
        "\u{FF41}",
    ] {
        assert_eq!(
            CollectionName::parse(bad),
            Err(TenancyError::InvalidCollectionName),
            "{bad:?}"
        );
    }
    assert_eq!(
        TenancyError::InvalidCollectionName.problem_code(),
        ProblemCode::InvalidRequest
    );
}

// ---- vector ids ----

#[test]
fn vector_id_nfc_and_nfd_are_distinct_no_normalization() {
    let nfc = VectorId::parse("caf\u{00E9}").unwrap();
    let nfd = VectorId::parse("cafe\u{0301}").unwrap();
    assert_ne!(nfc, nfd);
    assert_eq!(nfc.as_str().as_bytes(), b"caf\xC3\xA9");
    assert_eq!(nfd.as_str().as_bytes(), b"cafe\xCC\x81");
}

#[test]
fn vector_id_length_is_bytes() {
    let emoji = "\u{1F600}"; // 4 bytes
    assert!(VectorId::parse(&emoji.repeat(64)).is_ok()); // 256 bytes
    assert!(VectorId::parse(&emoji.repeat(65)).is_err());
    assert!(VectorId::parse(&"é".repeat(128)).is_ok()); // 256 bytes
    assert!(VectorId::parse(&"é".repeat(129)).is_err());
    assert!(VectorId::parse(&"x".repeat(257)).is_err());
    assert!(VectorId::parse("").is_err());
}

#[test]
fn vector_id_forbidden_characters() {
    for bad in [
        "a\0",
        "a\n",
        "a\r",
        "a\t",
        "a\u{7F}",
        "a\u{85}",
        "a\u{9F}", // Cc: C0, DEL, C1
        "a\u{2028}",
        "a\u{2029}", // line/para separators
        "a\u{202E}b",
        "a\u{2066}b",
        "a\u{200E}",
        "\u{061C}", // bidi controls
        "a\u{200B}b",
        "a\u{200D}b",
        "a\u{2060}b",
        "\u{FEFF}a", // zero-width / BOM
    ] {
        assert_eq!(
            VectorId::parse(bad),
            Err(TenancyError::InvalidVectorId),
            "{bad:?}"
        );
    }
    // Separators, spaces and ordinary Unicode are fine in ids: they never
    // reach a preimage without being hashed as a single field.
    for ok in ["a|b", "a:b/c", "a b", "日本語", "\u{1F600}", "e\u{0301}"] {
        assert!(VectorId::parse(ok).is_ok(), "{ok:?}");
    }
    assert!(!is_forbidden_char('|'));
    assert_eq!(
        TenancyError::InvalidVectorId.problem_code(),
        ProblemCode::InvalidRequest
    );
}
