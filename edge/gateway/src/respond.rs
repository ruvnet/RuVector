//! Response helpers: JSON, RFC 9457 problems, auth challenges.

use ruvector_edge_tenancy::{problem::PROBLEM_CONTENT_TYPE, Problem, ProblemCode};
use serde::Serialize;
use worker::{Headers, Response, Result};

/// JSON 200 with `Cache-Control: no-store` when `credentialed`.
pub fn json<T: Serialize>(value: &T, credentialed: bool) -> Result<Response> {
    let mut resp = Response::from_json(value)?;
    if credentialed {
        resp.headers_mut().set("Cache-Control", "no-store")?;
    }
    Ok(resp)
}

/// Problem response, optionally with a `WWW-Authenticate` challenge.
pub fn problem(code: ProblemCode, www_authenticate: Option<String>) -> Result<Response> {
    let p = Problem::new(code);
    let headers = Headers::new();
    headers.set("Content-Type", PROBLEM_CONTENT_TYPE)?;
    headers.set("Cache-Control", "no-store")?;
    if let Some(challenge) = www_authenticate {
        headers.set("WWW-Authenticate", &challenge)?;
    }
    Ok(Response::ok(p.to_json())?
        .with_status(p.status)
        .with_headers(headers))
}
