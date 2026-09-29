//! Unit tests (London school: ports mocked with mockall or in-memory fakes).

mod authorize;
mod basics;
mod code;
mod confidential;
mod dcr;
mod exchange;
mod exchange_client;
mod exchange_rules;
mod federation;
mod fixes;
mod grant;
mod refresh;
mod revoke;
mod scope_grant;
mod team_vocabulary;
mod token;

use crate::error::{OAuthError, OAuthErrorCode};

/// A request mutation and the error it must produce.
pub(crate) type Case<T> = (fn(&mut T), OAuthErrorCode);

/// Assert an error code.
#[track_caller]
pub(crate) fn assert_code<T: std::fmt::Debug>(r: Result<T, OAuthError>, code: OAuthErrorCode) {
    match r {
        Err(e) => assert_eq!(e.error, code, "{e:?}"),
        Ok(v) => panic!("expected {code:?}, got Ok({v:?})"),
    }
}
