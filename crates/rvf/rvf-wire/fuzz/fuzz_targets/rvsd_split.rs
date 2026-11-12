#![no_main]
//! Chunk-split independence: feeding the same (possibly mutated) byte stream through the
//! frame reader whole or cut at arbitrary points yields the same verdict.
#[path = "../common.rs"]
mod common;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rvf_wire::stream::*;

#[derive(Debug, Arbitrary)]
struct Input {
    ops: Vec<common::Op>,
    cuts: Vec<u16>,
}

fn run(parts: &[&[u8]]) -> (String, Vec<(u64, u32, Vec<u8>)>, Vec<rvf_types::ErrorCode>) {
    let b = common::base();
    let mut v = StreamVerifier::new(common::policy(b, 3));
    let mut rd = FrameReader::new();
    let (mut rel, mut errs) = (vec![], vec![]);
    for p in parts {
        let _ = rd.feed(&mut v, p, &mut |r| match r {
            Ok(Event::Chunk { chunk, .. }) => rel.push((chunk.seg_id, chunk.chunk_index, chunk.bytes.to_vec())),
            Ok(_) => {}
            Err(e) => errs.push(e),
        });
    }
    (format!("{:?}", v.state()), rel, errs)
}

fuzz_target!(|i: Input| {
    let b = common::base();
    let ops: Vec<_> = i.ops.into_iter().take(6).collect();
    let bytes: Vec<u8> = common::apply(b.frames.clone(), &ops).into_iter().flatten().collect();
    let mut pts: Vec<usize> = i.cuts.iter().take(16).map(|c| *c as usize % (bytes.len() + 1)).collect();
    pts.sort_unstable();
    let (mut parts, mut last) = (Vec::new(), 0);
    for p in pts {
        parts.push(&bytes[last..p]);
        last = p;
    }
    parts.push(&bytes[last..]);
    assert_eq!(run(&[&bytes[..]]), run(&parts));
});
