//! `ResourceUrl` canonicalisation (ADR-351 §5.2/§5.4, M0 fuzz criterion):
//! the one type both the resource server (exact `aud` match, RFC 9728
//! `resource`) and the authorization server (RFC 8707 allowlist, minted
//! `aud`) compare byte-for-byte.
//!
//! Invariants for every accepted input: the canonical form is idempotent
//! (re-parsing it yields the same value), `origin() + path()` rebuilds it,
//! it is `https://` + lowercase host with no query/fragment/userinfo,
//! trailing slash, empty or dot segment, never exceeds the input length,
//! is safe inside a `WWW-Authenticate` quoted-string, and the serde
//! `try_from` path agrees with `parse`. Host case never changes the result.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ruvector_edge_auth::resource::MAX_RESOURCE_URL_LEN;
use ruvector_edge_auth::ResourceUrl;

fn check(input: &str) {
    let parsed = ResourceUrl::parse(input);
    // serde (`try_from = "String"`) must agree with `parse`.
    let json = serde_json::to_string(input).expect("a str always serialises");
    let via_serde = serde_json::from_str::<ResourceUrl>(&json);
    assert_eq!(parsed.is_ok(), via_serde.is_ok(), "{input:?}");

    let Ok(r) = parsed else {
        return;
    };
    assert_eq!(via_serde.unwrap(), r);
    let s = r.as_str();
    assert!(input.len() <= MAX_RESOURCE_URL_LEN);
    assert_eq!(s.len(), input.len(), "canonicalisation only lowercases");
    assert!(s.eq_ignore_ascii_case(input));
    assert!(s.starts_with("https://"));
    assert_eq!(format!("{}{}", r.origin(), r.path()), s);
    assert!(r.origin().len() > "https://".len());
    let host = &r.origin()["https://".len()..];
    assert!(!host.contains('/'));
    assert!(!host.bytes().any(|b| b.is_ascii_uppercase()), "{s}");
    let path = r.path();
    assert!(path.is_empty() || path.starts_with('/'));
    assert!(!path.ends_with('/') && !path.contains("//"));
    assert!(!path.split('/').any(|seg| seg == "." || seg == ".."));
    assert!(!s
        .bytes()
        .any(|b| matches!(b, b'"' | b'\\' | b'?' | b'#' | b'@')
            || b.is_ascii_control()
            || !b.is_ascii()));
    // Idempotent and stable under serde.
    assert_eq!(ResourceUrl::parse(s).unwrap(), r);
    let round: ResourceUrl = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(round, r);
    assert_eq!(r.to_string(), s);
    // Host case never matters, path case always does.
    let upper_host = format!("https://{}{}", host.to_ascii_uppercase(), path);
    assert_eq!(ResourceUrl::parse(&upper_host).unwrap(), r);
}

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        check(s);
        // Most interesting inputs lack the scheme; try them as an authority.
        if !s.starts_with("https://") && s.len() < MAX_RESOURCE_URL_LEN {
            check(&format!("https://{s}"));
        }
    }
});
