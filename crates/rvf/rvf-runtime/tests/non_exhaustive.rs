//! #1274: stream enums may grow; they must be `#[non_exhaustive]`.
#![deny(unreachable_patterns)]

use rvf_runtime::progressive_stream::client::{FetchEnd, RunEnd};
use rvf_runtime::progressive_stream::{QueryMode, QueryPath, StreamWitness};

#[test]
fn stream_enums_are_non_exhaustive() {
    let a = |e: FetchEnd| match e {
        FetchEnd::Eof => 0,
        FetchEnd::Aborted => 1,
        _ => 99,
    };
    let b = |e: RunEnd| match e {
        RunEnd::Complete => 0,
        RunEnd::CompleteUnverified => 1,
        RunEnd::Poisoned(_) => 2,
        RunEnd::Suspended => 3,
        _ => 99,
    };
    let c = |e: QueryMode| match e {
        QueryMode::Auto => 0,
        QueryMode::LayersOnly => 1,
        QueryMode::ScanOnly => 2,
        _ => 99,
    };
    let d = |e: QueryPath| match e {
        QueryPath::Empty => 0,
        QueryPath::ArrivedScan => 1,
        QueryPath::Layers => 2,
        _ => 99,
    };
    let e = |e: StreamWitness| match e {
        StreamWitness::DescriptorVerified(_) => 0,
        StreamWitness::DescriptorRejected(_) => 1,
        StreamWitness::LayerPublished(_) => 2,
        StreamWitness::StreamPoisoned { .. } => 3,
        StreamWitness::Suspended => 4,
        StreamWitness::Resumed => 5,
        StreamWitness::Complete(_) => 6,
        _ => 99,
    };
    assert_eq!(a(FetchEnd::Eof), 0);
    assert_eq!(b(RunEnd::Suspended), 3);
    assert_eq!(c(QueryMode::Auto), 0);
    assert_eq!(d(QueryPath::Layers), 2);
    assert_eq!(e(StreamWitness::Resumed), 5);
}
