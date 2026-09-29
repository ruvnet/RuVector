//! Port implementations over the Workers runtime: clock, RNG, signer, JWKS.

use ruvector_edge_auth::{Clock, JwkSet};
use ruvector_edge_authz::{Rng, Signer, StoreError};
use worker::{Date, Env};

/// Name of the secret holding the active ES256 private scalar (32 bytes,
/// base64url). Set with `wrangler secret put`, never in `wrangler.toml`.
pub const SIGNING_KEY_SECRET: &str = "SIGNING_KEY_P256";

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
    /// Contract: fills `buf` or returns an error; never weak bytes.
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

/// ES256 signer backed by the `SIGNING_KEY_P256` secret.
pub struct EnvSigner {
    key: Option<p256::ecdsa::SigningKey>,
}

impl EnvSigner {
    /// Load the key from the environment secret.
    ///
    /// Contract: base64url-decode exactly 32 bytes and build a
    /// `p256::ecdsa::SigningKey`; any failure leaves the signer keyless so
    /// every `sign_es256` fails closed.
    pub fn from_env(env: &Env) -> Self {
        let _ = env.secret(SIGNING_KEY_SECRET);
        EnvSigner { key: None }
    }
}

impl Signer for EnvSigner {
    fn kid(&self) -> String {
        self.key
            .as_ref()
            .map(|k| ruvector_edge_auth::Jwk::from_verifying_key(k.verifying_key()).kid)
            .unwrap_or_default()
    }

    /// Contract: RFC 6979 deterministic ES256 over `signing_input`, fixed
    /// 64-byte `r || s` output.
    fn sign_es256(&self, signing_input: &[u8]) -> Result<[u8; 64], StoreError> {
        let _ = (signing_input, &self.key);
        Err(StoreError("signer not implemented".into()))
    }
}

/// Public JWKS for the active (and, during rotation, previous) key.
pub fn public_jwks(env: &Env) -> JwkSet {
    let signer = EnvSigner::from_env(env);
    let keys = signer
        .key
        .as_ref()
        .map(|k| *k.verifying_key())
        .into_iter()
        .collect::<Vec<_>>();
    ruvector_edge_authz::metadata::jwks_document(&keys)
}
