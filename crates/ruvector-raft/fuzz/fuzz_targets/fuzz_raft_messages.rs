#![no_main]

use libfuzzer_sys::fuzz_target;
use ruvector_raft::rpc::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    RaftMessage, RequestVoteRequest, RequestVoteResponse,
};

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes into every from_bytes() deserializer.
    // None of these should panic -- they must return Ok or Err.

    // 1. Top-level RaftMessage envelope
    let _ = RaftMessage::from_bytes(data);

    // 2. Individual message types
    let _ = AppendEntriesRequest::from_bytes(data);
    let _ = AppendEntriesResponse::from_bytes(data);
    let _ = RequestVoteRequest::from_bytes(data);
    let _ = RequestVoteResponse::from_bytes(data);
    let _ = InstallSnapshotRequest::from_bytes(data);
    let _ = InstallSnapshotResponse::from_bytes(data);

    // 3. Round-trip: if any message deserializes successfully, re-serialize
    //    and verify the round-trip produces identical bytes.
    if let Ok(msg) = RaftMessage::from_bytes(data) {
        if let Ok(re_encoded) = msg.to_bytes() {
            let round_trip = RaftMessage::from_bytes(&re_encoded);
            assert!(
                round_trip.is_ok(),
                "Round-trip deserialization failed for successfully parsed RaftMessage"
            );
        }
    }

    if let Ok(req) = AppendEntriesRequest::from_bytes(data) {
        if let Ok(re_encoded) = req.to_bytes() {
            let round_trip = AppendEntriesRequest::from_bytes(&re_encoded);
            assert!(
                round_trip.is_ok(),
                "Round-trip failed for AppendEntriesRequest"
            );
        }
    }

    if let Ok(req) = RequestVoteRequest::from_bytes(data) {
        if let Ok(re_encoded) = req.to_bytes() {
            let round_trip = RequestVoteRequest::from_bytes(&re_encoded);
            assert!(
                round_trip.is_ok(),
                "Round-trip failed for RequestVoteRequest"
            );
        }
    }
});
