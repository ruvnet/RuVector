#![no_main]
//! Frame-sequence state machine, structure-aware: mutate the frames of a valid stream with
//! arbitrary ops. Invariants: every released chunk is byte-identical to the signed body and
//! released at most once; Complete implies the whole body was released.
#[path = "../common.rs"]
mod common;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rvf_wire::stream::*;

#[derive(Debug, Arbitrary)]
struct Input {
    budget: u8,
    ops: Vec<common::Op>,
}

fuzz_target!(|i: Input| {
    let b = common::base();
    let frames = common::apply(b.frames.clone(), &i.ops);
    let mut v = StreamVerifier::new(common::policy(b, i.budget as u32));
    let mut seen = std::collections::HashSet::new();
    let mut total = 0usize;
    for f in &frames {
        if let Ok(Event::Chunk { chunk, .. }) = v.push(f) {
            let o = chunk.body_offset as usize;
            assert_eq!(&b.body[o..o + chunk.bytes.len()], chunk.bytes);
            assert!(seen.insert((chunk.seg_id, chunk.chunk_index)));
            total += chunk.bytes.len();
        }
    }
    if matches!(v.state(), StreamState::Complete | StreamState::CompleteUnverified) {
        assert_eq!(total, b.body.len());
    }
});
