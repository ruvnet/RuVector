//! Isolate-global JWKS caches, so the in-isolate TTL (ADR-351 §5.1.5)
//! spans requests instead of refetching the key set on every call.
//! Workers isolates are single-threaded, hence `thread_local!` + `Rc`.

use crate::platform::{JwksFetch, WorkerClock};
use ruvector_edge_auth::{AuthError, JwksCache, JwksCachePolicy, KeySource, VerifyingKey};
use std::cell::RefCell;
use std::rc::Rc;

type Cache = JwksCache<JwksFetch, WorkerClock>;
type Slot = RefCell<Option<(String, Rc<Cache>)>>;

thread_local! {
    static EDGE: Slot = const { RefCell::new(None) };
    static UPSTREAM: Slot = const { RefCell::new(None) };
}

/// Shared handle to an isolate-global cache.
#[derive(Clone)]
pub struct SharedKeys(Rc<Cache>);

impl KeySource for SharedKeys {
    async fn verifying_key(&self, kid: &str) -> Result<VerifyingKey, AuthError> {
        self.0.verifying_key(kid).await
    }
}

fn get_or_init(
    slot: &'static std::thread::LocalKey<Slot>,
    policy: JwksCachePolicy,
    make_fetch: impl FnOnce() -> JwksFetch,
) -> SharedKeys {
    // The slot is keyed on the whole trust root (URL + kid pin).
    let key = format!("{}|{:?}", policy.url, policy.accepted_kids);
    slot.with(|cell| {
        let mut cell = cell.borrow_mut();
        match cell.as_ref() {
            Some((k, cache)) if *k == key => SharedKeys(cache.clone()),
            _ => {
                let cache = Rc::new(JwksCache::new(make_fetch(), WorkerClock, policy));
                *cell = Some((key, cache.clone()));
                SharedKeys(cache)
            }
        }
    })
}

/// Edge AS keys, fetched through the service binding when present.
pub fn edge(url: &str, make_fetch: impl FnOnce() -> JwksFetch) -> SharedKeys {
    get_or_init(&EDGE, JwksCachePolicy::with_defaults(url), make_fetch)
}

/// Upstream (`auth.cognitum.one`) keys over the public internet, pinned to
/// `ACCEPTED_UPSTREAM_KIDS` so foreign keys never enter the cache (the
/// verifier enforces the same pin).
pub fn upstream(url: &str, accepted_kids: &[String]) -> SharedKeys {
    let policy = JwksCachePolicy::with_defaults(url).with_accepted_kids(accepted_kids.to_vec());
    get_or_init(&UPSTREAM, policy, || JwksFetch::Global)
}
