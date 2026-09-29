//! Port implementations over the Workers runtime.

use ruvector_edge_auth::{Clock, FetchError, HttpFetch, HttpResponse};
use worker::{Date, Fetch, Url};

/// `Clock` over `Date.now()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerClock;

impl Clock for WorkerClock {
    fn now_unix(&self) -> u64 {
        Date::now().as_millis() / 1000
    }
}

/// `HttpFetch` over the global `fetch`.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerFetch;

impl HttpFetch for WorkerFetch {
    async fn get(&self, url: &str) -> Result<HttpResponse, FetchError> {
        let url = Url::parse(url).map_err(|e| FetchError(e.to_string()))?;
        let mut resp = Fetch::Url(url)
            .send()
            .await
            .map_err(|e| FetchError(e.to_string()))?;
        let status = resp.status_code();
        let body = resp.bytes().await.map_err(|e| FetchError(e.to_string()))?;
        Ok(HttpResponse { status, body })
    }
}
