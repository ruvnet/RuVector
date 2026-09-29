//! Port implementations over the Workers runtime: clock, RNG, HTTP.

use crate::upstream::UpstreamHttp;
use ruvector_edge_auth::{Clock, FetchError, HttpFetch, HttpResponse};
use ruvector_edge_authz::{Rng, StoreError};
use worker::{Date, Fetch, Headers, Method, Request, RequestInit, RequestRedirect};

/// `Clock` over `Date.now()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerClock;

impl Clock for WorkerClock {
    fn now_unix(&self) -> u64 {
        Date::now().as_millis() / 1000
    }
}

/// `Rng` over `crypto.getRandomValues` (getrandom `js`).
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerRng;

impl Rng for WorkerRng {
    /// Fills `buf` or returns an error; never weak bytes.
    fn fill(&self, buf: &mut [u8]) -> Result<(), StoreError> {
        #[cfg(target_arch = "wasm32")]
        {
            getrandom::getrandom(buf).map_err(|e| StoreError(e.to_string()))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = buf;
            Err(StoreError("no RNG outside wasm32".into()))
        }
    }
}

/// Global `fetch` with redirects never followed (a 3xx is returned as-is
/// and treated as a failure by the callers, which require 200).
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerFetch;

fn fetch_err(e: worker::Error) -> FetchError {
    FetchError(e.to_string())
}

async fn send(url: &str, init: &RequestInit, max: usize) -> Result<HttpResponse, FetchError> {
    let req = Request::new_with_init(url, init).map_err(fetch_err)?;
    let mut resp = Fetch::Request(req).send().await.map_err(fetch_err)?;
    let status = resp.status_code();
    // A declared length above the cap is refused before the body is read;
    // an undeclared body is truncated to `max + 1` so callers reject it.
    let declared = resp
        .headers()
        .get("Content-Length")
        .ok()
        .flatten()
        .and_then(|v| v.trim().parse::<usize>().ok());
    if declared.is_some_and(|n| n > max) {
        return Err(FetchError("response body too large".into()));
    }
    let mut body = resp.bytes().await.map_err(fetch_err)?;
    body.truncate(max.saturating_add(1));
    Ok(HttpResponse { status, body })
}

impl HttpFetch for WorkerFetch {
    async fn get(&self, url: &str, max_body_bytes: usize) -> Result<HttpResponse, FetchError> {
        let headers = Headers::new();
        headers
            .set("Accept", "application/json")
            .map_err(fetch_err)?;
        let mut init = RequestInit::new();
        init.with_method(Method::Get)
            .with_headers(headers)
            .with_redirect(RequestRedirect::Manual);
        send(url, &init, max_body_bytes).await
    }
}

impl UpstreamHttp for WorkerFetch {
    async fn post_form(&self, url: &str, body: String) -> Result<HttpResponse, FetchError> {
        let headers = Headers::new();
        headers
            .set("Content-Type", "application/x-www-form-urlencoded")
            .map_err(fetch_err)?;
        headers
            .set("Accept", "application/json")
            .map_err(fetch_err)?;
        let mut init = RequestInit::new();
        init.with_method(Method::Post)
            .with_headers(headers)
            .with_redirect(RequestRedirect::Manual)
            .with_body(Some(body.into()));
        send(url, &init, crate::upstream::MAX_UPSTREAM_BODY).await
    }
}
