//! Parser tests (kept apart so `parse.rs` stays small).

use super::*;
use crate::schema;

#[test]
fn every_schema_constant_parses() {
    for s in schema::SHARD_SCHEMA.iter().chain(schema::LEDGER_SCHEMA) {
        assert!(
            matches!(parse(s).unwrap(), Stmt::Create { .. } | Stmt::Index { .. }),
            "{s}"
        );
    }
    for s in [
        schema::META_SELECT_ALL,
        schema::META_PUT,
        schema::META_DELETE_ALL,
        schema::VEC_PUT,
        schema::VEC_DELETE,
        schema::VEC_PAGE,
        schema::VEC_DELETE_ALL,
        schema::OPS_APPEND,
        schema::OPS_PAGE,
        schema::OPS_ACTORS,
        schema::OPS_DELETE_ALL,
        schema::FILTER_PUT,
        schema::FILTER_DELETE_ID,
        schema::FILTER_DELETE_ALL,
        schema::LMETA_SELECT_ALL,
        schema::LMETA_PUT,
        schema::MEMBER_SELECT_ALL,
        schema::MEMBER_INSERT,
        schema::MEMBER_PUT,
        schema::MEMBER_DELETE,
        schema::CATALOG_SELECT_ALL,
        schema::CATALOG_INSERT,
        schema::CATALOG_SET_STATE,
        schema::IDEM_SELECT,
        schema::IDEM_PUT,
        schema::IDEM_EXPIRED,
        schema::IDEM_DELETE,
        schema::OPS_PRUNE,
        schema::VEC_PAGE_META,
        schema::VEC_BY_IIDS,
        schema::VEC_BY_ID,
        schema::VEC_SET_IID,
        schema::CHUNK_PUT,
        schema::CHUNK_GET,
        schema::CHUNK_DELETE_FROM,
        schema::CHUNK_DELETE_BELOW,
        schema::CHUNK_DELETE_ALL,
    ] {
        parse(s).unwrap_or_else(|e| panic!("{s}: {e}"));
    }
}

#[test]
fn pk_detection() {
    match parse(schema::SHARD_SCHEMA[3]).unwrap() {
        Stmt::Create { pk, .. } => assert_eq!(pk, vec!["key", "value", "id"]),
        _ => unreachable!(),
    }
    match parse(schema::SHARD_SCHEMA[1]).unwrap() {
        Stmt::Create { pk, cols, .. } => {
            assert_eq!(pk, vec!["id"]);
            assert_eq!(cols.len(), 8);
        }
        _ => unreachable!(),
    }
}

#[test]
fn rejects_unsupported_sql() {
    for s in [
        "SELECT COUNT(*) FROM t",
        "SELECT a FROM t WHERE a = 1",
        "DROP TABLE t",
        "SELECT a FROM t; DELETE FROM t",
        "INSERT INTO t (a) SELECT a FROM u",
        "SELECT a FROM t WHERE a = ? OR b = ?",
    ] {
        assert!(parse(s).is_err(), "{s}");
    }
}

#[test]
fn in_lists_count_their_bindings() {
    let st = parse(schema::VEC_BY_IIDS).unwrap();
    assert_eq!(st.param_count(), schema::IID_BATCH);
    let st = parse("SELECT a FROM t WHERE b IN (?, ?) AND a > ? LIMIT ?").unwrap();
    assert_eq!(st.param_count(), 4);
    for s in [
        "SELECT a FROM t WHERE b IN ()",
        "SELECT a FROM t WHERE b IN (?",
    ] {
        assert!(parse(s).is_err(), "{s}");
    }
}
