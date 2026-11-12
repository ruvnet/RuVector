//! #1274: server refusal/cap enums may grow; they must be `#[non_exhaustive]`.
#![deny(unreachable_patterns)]

use rvf_server::stream::{ClientCap, StreamRefusal};

#[test]
fn server_enums_are_non_exhaustive() {
    let r = |e: StreamRefusal| match e {
        StreamRefusal::Busy => 0,
        StreamRefusal::PeerUnknown => 1,
        _ => 99,
    };
    let c = |e: ClientCap| match e {
        ClientCap::PerAddress => 0,
        ClientCap::Disabled => 1,
        _ => 99,
    };
    assert_eq!(r(StreamRefusal::Busy), 0);
    assert_eq!(c(ClientCap::Disabled), 1);
}
