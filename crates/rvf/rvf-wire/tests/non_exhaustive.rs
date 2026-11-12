//! #1274: stream verifier enums may grow; they must be `#[non_exhaustive]`.
//! Listing every variant plus `_` only compiles under `deny(unreachable_patterns)`
//! if the enum is `#[non_exhaustive]` for this (external) crate.
#![deny(unreachable_patterns)]

use rvf_types::ErrorCode;
use rvf_wire::stream::{IdentityStatus, StreamState};

#[test]
fn stream_state_is_non_exhaustive() {
    let name = |s: StreamState| match s {
        StreamState::AwaitDescriptor => "await",
        StreamState::Receiving => "receiving",
        StreamState::Suspended => "suspended",
        StreamState::Complete => "complete",
        StreamState::CompleteUnverified => "complete-unverified",
        StreamState::Rejected(_) => "rejected",
        StreamState::Poisoned(_) => "poisoned",
        _ => "unknown",
    };
    assert_eq!(
        name(StreamState::Poisoned(ErrorCode::StreamPoisoned)),
        "poisoned"
    );
}

#[test]
fn identity_status_is_non_exhaustive() {
    let name = |s: IdentityStatus| match s {
        IdentityStatus::Verified => "verified",
        IdentityStatus::Unverifiable => "unverifiable",
        IdentityStatus::NotPresent => "none",
        _ => "unknown",
    };
    assert_eq!(name(IdentityStatus::Verified), "verified");
}
