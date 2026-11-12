//! Port implementations over the Workers runtime.

use ruvector_edge_auth::{Clock, FetchError, HttpFetch, HttpResponse};
use worker::{Date, Fetch, Fetcher, Headers, Method, Request, RequestInit, RequestRedirect};

/// Service binding to the `ruvector-edge-auth` Worker. Same-account
/// `workers.dev` -> `workers.dev` subrequests over the public internet are
/// refused by Cloudflare (error 1042), so the edge JWKS is fetched through
/// this binding when it is configured.
pub const EDGE_AUTH_BINDING: &str = "EDGE_AUTH";

/// `Clock` over `Date.now()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerClock;

impl Clock for WorkerClock {
    fn now_unix(&self) -> u64 {
        Date::now().as_millis() / 1000
    }
}

/// `HttpFetch` over a service binding (edge AS) or the global `fetch`
/// (upstream JWKS). Redirects are never followed; a 3xx reaches the JWKS
/// cache as a non-200 and is treated as a fetch failure.
pub enum JwksFetch {
    /// Through the `EDGE_AUTH` service binding (the URL's path is used; the
    /// host is ignored by the runtime). The **only** way the edge JWKS is
    /// fetched (ADR-351 §5.4 item 5).
    Service(Fetcher),
    /// The `EDGE_AUTH` binding is missing: every fetch fails, so the gateway
    /// answers 503 `jwks_unavailable` instead of trying a public
    /// workers.dev -> workers.dev fetch (Cloudflare 1042).
    Unbound,
    /// Public internet (upstream `auth.cognitum.one` JWKS only).
    Global,
}

/// Error of a fetch through a missing `EDGE_AUTH` binding.
pub const UNBOUND_ERROR: &str = "EDGE_AUTH service binding missing";

fn fetch_err(e: worker::Error) -> FetchError {
    FetchError(e.to_string())
}

fn init() -> Result<RequestInit, FetchError> {
    let headers = Headers::new();
    headers
        .set("Accept", "application/json")
        .map_err(fetch_err)?;
    let mut init = RequestInit::new();
    init.with_method(Method::Get)
        .with_headers(headers)
        .with_redirect(RequestRedirect::Manual);
    Ok(init)
}

impl HttpFetch for JwksFetch {
    async fn get(&self, url: &str, max_body_bytes: usize) -> Result<HttpResponse, FetchError> {
        // Checked before any runtime call, so a missing binding never
        // reaches the network.
        let service = match self {
            JwksFetch::Unbound => return Err(FetchError(UNBOUND_ERROR.into())),
            JwksFetch::Service(f) => Some(f),
            JwksFetch::Global => None,
        };
        let init = init()?;
        let mut resp = match service {
            Some(f) => f.fetch(url, Some(init)).await.map_err(fetch_err)?,
            None => {
                let req = Request::new_with_init(url, &init).map_err(fetch_err)?;
                Fetch::Request(req).send().await.map_err(fetch_err)?
            }
        };
        let status = resp.status_code();
        let body = bounded_body(&mut resp, max_body_bytes).await?;
        Ok(HttpResponse { status, body })
    }
}

/// Read at most `max + 1` body bytes: a declared `Content-Length` above
/// `max` is refused before the body is read (the `HttpFetch` contract), and
/// an undeclared body is truncated to `max + 1` so the caller rejects it.
async fn bounded_body(resp: &mut worker::Response, max: usize) -> Result<Vec<u8>, FetchError> {
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
    Ok(body)
}
