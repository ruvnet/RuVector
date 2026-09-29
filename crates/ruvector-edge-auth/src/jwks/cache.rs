//! In-isolate JWKS cache (ADR-351 §5.4.6) with a true single-flight fetch:
//! concurrent requests in one isolate wait for the in-flight fetch instead of
//! being turned away by the refetch rate limit.

use super::{HttpFetch, JwkSet, JwksCachePolicy, KeySource};
use crate::clock::Clock;
use crate::error::AuthError;
use core::cell::RefCell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use p256::ecdsa::VerifyingKey;
use std::collections::BTreeMap;

#[derive(Debug, Default)]
struct CacheState {
    keys: BTreeMap<String, VerifyingKey>,
    fetched_at: Option<u64>,
    last_refetch_attempt: Option<u64>,
}

/// `Some(waiters)` while a fetch is in flight.
type Flight = RefCell<Option<Vec<Waker>>>;

/// In-isolate JWKS cache implementing [`KeySource`] over an [`HttpFetch`].
/// Single-threaded (wasm isolates), hence `RefCell`; share it across requests
/// of one isolate (e.g. `thread_local!` + `Rc`).
pub struct JwksCache<F: HttpFetch, C: Clock> {
    fetch: F,
    clock: C,
    policy: JwksCachePolicy,
    state: RefCell<CacheState>,
    flight: Flight,
}

/// Ends the flight (also when the leader future is dropped mid-fetch) and
/// wakes every follower.
struct FlightGuard<'a>(&'a Flight);

impl Drop for FlightGuard<'_> {
    fn drop(&mut self) {
        let waiters = self.0.borrow_mut().take();
        for w in waiters.into_iter().flatten() {
            w.wake();
        }
    }
}

/// Resolves once no fetch is in flight.
struct JoinFlight<'a>(&'a Flight);

impl Future for JoinFlight<'_> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut slot = self.0.borrow_mut();
        match slot.as_mut() {
            None => Poll::Ready(()),
            Some(waiters) => {
                if !waiters.iter().any(|w| w.will_wake(cx.waker())) {
                    waiters.push(cx.waker().clone());
                }
                Poll::Pending
            }
        }
    }
}

impl<F: HttpFetch, C: Clock> JwksCache<F, C> {
    /// New, empty cache.
    pub fn new(fetch: F, clock: C, policy: JwksCachePolicy) -> Self {
        JwksCache {
            fetch,
            clock,
            policy,
            state: RefCell::new(CacheState::default()),
            flight: RefCell::new(None),
        }
    }

    /// The configured policy.
    pub fn policy(&self) -> &JwksCachePolicy {
        &self.policy
    }

    /// Fetch now (as the leader), or wait for the fetch already in flight
    /// (as a follower). Rate limited by [`JwksCache::fetch_allowed`]; a
    /// follower never starts a second fetch. The caller re-reads the state
    /// afterwards. Never holds a `RefCell` borrow across an `.await`.
    async fn refresh_or_join(&self, now: u64) {
        if self.flight.borrow().is_some() {
            JoinFlight(&self.flight).await;
            return;
        }
        if !self.fetch_allowed(now) {
            return;
        }
        *self.flight.borrow_mut() = Some(Vec::new());
        let _guard = FlightGuard(&self.flight);
        self.refresh(now).await;
    }

    /// One fetch attempt. A failed or unusable response never replaces a
    /// good cached set.
    async fn refresh(&self, now: u64) {
        self.state.borrow_mut().last_refetch_attempt = Some(now);
        let max = self.policy.max_body_bytes;
        let resp = match self.fetch.get(&self.policy.url, max).await {
            // Defence in depth: the port contract bounds the read at max + 1.
            Ok(r) if r.status == 200 && r.body.len() <= max => r,
            _ => return,
        };
        let Ok(mut keys) = JwkSet::parse_usable(&resp.body) else {
            return;
        };
        keys.retain(|kid, _| self.policy.kid_accepted(kid));
        if keys.is_empty() {
            return;
        }
        let mut st = self.state.borrow_mut();
        st.keys = keys;
        st.fetched_at = Some(now);
    }

    /// Spacing between attempts: `refetch_min_interval_secs` once a set has
    /// been obtained, the short `cold_retry_interval_secs` while none ever
    /// has (a transient cold-start failure must not 503 the isolate for 30 s).
    fn fetch_allowed(&self, now: u64) -> bool {
        let st = self.state.borrow();
        let interval = if st.fetched_at.is_some() {
            self.policy.refetch_min_interval_secs
        } else {
            self.policy.cold_retry_interval_secs
        };
        st.last_refetch_attempt
            .map_or(true, |t| now.saturating_sub(t) >= interval)
    }

    /// Age-bounded lookup: `Some(result)` if a set younger than `max_age`
    /// exists.
    fn lookup(&self, kid: &str, now: u64, max_age: u64) -> Option<Result<VerifyingKey, AuthError>> {
        let st = self.state.borrow();
        let fetched_at = st.fetched_at?;
        if now.saturating_sub(fetched_at) >= max_age {
            return None;
        }
        Some(st.keys.get(kid).copied().ok_or(AuthError::UnknownKid))
    }
}

impl<F: HttpFetch, C: Clock> KeySource for JwksCache<F, C> {
    /// Contract: pinned-out `kid` -> `UnknownKid` with no fetch; fresh hit ->
    /// key; stale, never fetched, or fresh-but-unknown `kid` -> one
    /// single-flight fetch (concurrent callers wait for it; rate limited per
    /// `refetch_min_interval_secs`, or `cold_retry_interval_secs` before the
    /// first success); fetch error -> last good set within
    /// `stale_if_error_secs`, else `KeysUnavailable`.
    async fn verifying_key(&self, kid: &str) -> Result<VerifyingKey, AuthError> {
        if !self.policy.kid_accepted(kid) {
            return Err(AuthError::UnknownKid);
        }
        let now = self.clock.now_unix();
        let fresh = self.policy.fresh_ttl_secs;
        let stale = self.policy.stale_if_error_secs.max(fresh);

        match self.lookup(kid, now, fresh) {
            Some(Ok(key)) => return Ok(key),
            // Fresh set without this kid: rate-limited forced refresh.
            Some(Err(_)) => {
                self.refresh_or_join(now).await;
                return self
                    .lookup(kid, now, fresh)
                    .unwrap_or(Err(AuthError::UnknownKid));
            }
            None => {}
        }
        // Stale or never fetched.
        self.refresh_or_join(now).await;
        if let Some(found) = self.lookup(kid, now, fresh) {
            return found;
        }
        self.lookup(kid, now, stale)
            .unwrap_or(Err(AuthError::KeysUnavailable))
    }
}
