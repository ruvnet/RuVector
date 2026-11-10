//! Known-answer vectors for RVSD v1 (`crates/rvf/tests/vectors/rvsd-v1-vectors.json`).
//! `RVSD_REGEN=1 cargo test -p rvf-wire --test stream_vectors` rewrites the file;
//! without it the test fails if the committed file differs from a fresh run.
mod common;
#[path = "common/veccases.rs"]
mod veccases;
#[path = "common/vecmodel.rs"]
mod vecmodel;
use common::json::*;
use common::*;
use rvf_types::{StreamEntry, StreamHeader};
use rvf_wire::stream::*;
use veccases::*;
use vecmodel::*;

const FILE_A: [u8; 16] = *b"rvsd-test-file01";
const FILE_B: [u8; 16] = *b"rvsd-other-file2";

fn vector_body(variant: u8) -> Vec<u8> {
    use rvf_types::SegmentFlags as F;
    use rvf_wire::writer::{write_segment as w, write_segment_with_algo as wa};
    let f = F::empty();
    let mut b = Vec::new();
    b.extend(w(0x07, &payload(11, 100), f, 1));
    b.extend(w(0x05, &payload(12, 64), f, 2));
    b.extend(wa(0x10, &payload(13, 5000), f, 3, 2));
    b.extend(wa(0x02, &payload(14, 4200), f, 4, 2));
    b.extend(w(0x01, &payload(15 + variant as u64, 6000), f, 5));
    b.extend(w(0x06, &payload(16, 300), f, 6));
    b.extend(root_page());
    b
}

fn entry_json(e: &StreamEntry, h_: &StreamHeader, body: &[u8]) -> J {
    use rvf_crypto::leaf_hash;
    let st = &body[e.body_offset as usize..(e.body_offset + e.stored_len) as usize];
    let n_ch = h_.chunk_count(e.stored_len).unwrap();
    let leaves: Vec<J> = (0..n_ch)
        .map(|i| {
            let a = (i as u64 * h_.chunk_size().unwrap_or(0)) as usize;
            let l = h_.chunk_len(e.stored_len, i).unwrap() as usize;
            h(&leaf_hash(&h_.file_id, e.seg_id, i, &st[a..a + l]))
        })
        .collect();
    J::Obj(vec![
        ("seg_id", n(e.seg_id)),
        ("seg_type", n(e.seg_type)),
        ("stage", n(e.stage)),
        ("requires_mask", n(e.requires_mask)),
        ("body_offset", n(e.body_offset)),
        ("stored_len", n(e.stored_len)),
        ("uncompressed_len", n(e.uncompressed_len)),
        ("chunk_count", n(n_ch)),
        ("leaf_hashes_hex", J::Arr(leaves)),
        ("chunk_table_hash_hex", h(&e.chunk_table_hash)),
    ])
}

fn stream_json(st: &Stream, body: &[u8]) -> J {
    let out = run(&Pol::std(4), &st.frames);
    assert!(
        out.errors.is_empty()
            && out.state
                == if st.name.ends_with("stage") {
                    "CompleteUnverified"
                } else {
                    "Complete"
                },
        "{} {:?}",
        st.name,
        out.errors
    );
    let hd = &st.bundle.header;
    let frames: Vec<J> = st
        .frames
        .iter()
        .zip(out.events)
        .map(|(f, e)| J::Obj(vec![("hex", h(f)), ("expect", e)]))
        .collect();
    J::Obj(vec![
        ("name", s(st.name)),
        ("chunk_log2", n(hd.chunk_log2)),
        ("flags", n(hd.flags)),
        ("epoch", n(hd.epoch)),
        ("policy_hash_hex", h(&hd.policy_hash)),
        ("rvf_identity_hex", h(&hd.rvf_identity)),
        ("descriptor_hex", h(&st.bundle.descriptor)),
        ("descriptor_hash_hex", h(&st.bundle.descriptor_hash)),
        (
            "entries",
            J::Arr(
                st.bundle
                    .entries
                    .iter()
                    .map(|e| entry_json(e, hd, body))
                    .collect(),
            ),
        ),
        ("frames", J::Arr(frames)),
        ("final_state", s(out.state)),
    ])
}

fn case_json(c: &Case, base: &Stream) -> J {
    let frames = apply(&base.frames, &c.edits);
    let o = run(&c.pol, &frames);
    let first = o.errors.first().copied();
    match (first, c.name) {
        (_, "truncate-frames") => assert_eq!(o.state, "Receiving"),
        (Some((_, r)), _) => assert_eq!(r, c.reason, "{}", c.name),
        (None, _) => panic!("{} produced no error", c.name),
    }
    assert_eq!(o.state, c.state, "{}", c.name);
    let reason = if c.name == "truncate-frames" {
        J::Null
    } else {
        s(format!("{:?}", c.reason))
    };
    let code = if c.name == "truncate-frames" {
        J::Null
    } else {
        s(format!("0x{:04x}", c.reason as u16))
    };
    J::Obj(vec![
        ("name", s(c.name)),
        ("stream", s(c.stream)),
        ("description", s(c.what)),
        ("policy", c.pol.json()),
        ("edits", J::Arr(c.edits.iter().map(edit_json).collect())),
        (
            "expect",
            J::Obj(vec![
                (
                    "first_error_frame",
                    first.map_or(J::Null, |(i, _)| n(i as u64)),
                ),
                ("reason", reason),
                ("code", code),
                ("final_state", s(o.state)),
                ("released_chunks", n(o.chunks)),
                ("released_bytes", n(o.bytes)),
                ("complete", J::Bool(false)),
            ]),
        ),
    ])
}

fn generate() -> String {
    let body = vector_body(0);
    let body_b = vector_body(1);
    let m1f = mk("m1-file", &body, FILE_A, 0xFF, false);
    let m1s = mk("m1-stage", &body, FILE_A, 0xFF, true);
    let m2f = mk("m2-file", &body, FILE_A, 12, false);
    let m2s = mk("m2-stage", &body, FILE_A, 12, true);
    let other = {
        let b = StreamDescriptorBuilder::new(FILE_B, key(0x2A))
            .epoch(7)
            .chunk_log2(12)
            .policy_hash([0xAB; 32])
            .build(&body_b)
            .unwrap();
        let frames = b.frames_file_order(&body_b).collect();
        Stream {
            name: "other",
            bundle: b,
            frames,
        }
    };
    let all = [&m1f, &m1s, &m2f, &m2s];
    let neg: Vec<J> = cases(&m1f, &m1s, &m2f, &other)
        .iter()
        .map(|c| case_json(c, all.iter().find(|s| s.name == c.stream).unwrap()))
        .collect();
    let segs: Vec<J> = m2f
        .bundle
        .entries
        .iter()
        .map(|e| {
            J::Obj(vec![
                ("seg_id", n(e.seg_id)),
                ("seg_type", n(e.seg_type)),
                ("body_offset", n(e.body_offset)),
                ("stored_len", n(e.stored_len)),
            ])
        })
        .collect();
    // retry within budget: flipped chunk then the good one again.
    let c0 = pos(&m2f.frames, 1, 3, Some(0));
    let mut retry = m2f.frames.clone();
    let good = retry[c0].clone();
    retry[c0][24] ^= 1;
    retry.insert(c0 + 1, good.clone());
    let ro = run(&Pol::std(4), &retry);
    assert_eq!((ro.errors.len(), ro.state.as_str()), (1, "Complete"));
    let retry_j = J::Obj(vec![
        ("name", s("retry-within-budget")),
        ("stream", s("m2-file")),
        (
            "description",
            s("corrupted chunk then a clean resend; one counted fault, stream completes"),
        ),
        ("policy", Pol::std(4).json()),
        (
            "edits",
            J::Arr(vec![
                edit_json(&Edit::Xor {
                    frame: c0,
                    offset: 24,
                    mask: 1,
                }),
                edit_json(&Edit::Insert {
                    at: c0 + 1,
                    bytes: good.clone(),
                }),
            ]),
        ),
        (
            "expect",
            J::Obj(vec![
                ("first_error_frame", n(c0 as u64)),
                ("reason", s("StreamChunkHashMismatch")),
                ("code", s("0x0810")),
                ("final_state", s(ro.state)),
                ("released_chunks", n(ro.chunks)),
                ("released_bytes", n(ro.bytes)),
                ("complete", J::Bool(true)),
            ]),
        ),
    ]);
    // Authentic duplicates are idempotent (budget 0: any fault would poison).
    let len = m2f.frames.len();
    let s3 = pos(&m2f.frames, 1, 3, Some(0));
    let late = Edit::Insert {
        at: len - 1,
        bytes: m2f.frames[s3].clone(),
    };
    let dup_j = variant_json(
        "duplicate-chunks-and-descriptor",
        "chunk 0 of segment 3 sent three times, a late replay of it near the end, and the descriptor twice; all idempotent, same released set",
        &m2f,
        &[late, Edit::Dup(s3), Edit::Dup(s3), Edit::Dup(0)],
        &Pol::std(0),
    );
    J::Obj(vec![
        ("format", s("rvsd-v1-vectors")),
        ("version", n(1u64)),
        (
            "generator",
            s("crates/rvf/rvf-wire/tests/stream_vectors.rs (RVSD_REGEN=1 rewrites this file)"),
        ),
        (
            "domain_tag_ascii_with_trailing_nul",
            s("RVF-v1-stream-descriptor\0"),
        ),
        (
            "test_only_key",
            J::Obj(vec![
                ("seed_hex", h(&[0x2A; 32])),
                ("public_key_hex", h(&pk(0x2A))),
                (
                    "signer_key_id_hex",
                    h(&rvf_crypto::signer_key_id(&pk(0x2A))),
                ),
                ("other_seed_hex", h(&[0x2B; 32])),
                ("other_public_key_hex", h(&pk(0x2B))),
            ]),
        ),
        ("file_id_hex", h(&FILE_A)),
        ("other_file_id_hex", h(&FILE_B)),
        (
            "body",
            J::Obj(vec![
                ("len", n(body.len() as u64)),
                ("sha256_hex", h(&rvf_types::sha256(&body))),
                ("segments", J::Arr(segs)),
                ("hex", h(&body)),
            ]),
        ),
        ("other_body_hex", h(&body_b)),
        (
            "stages_by_segment_order",
            J::Arr(STAGES.iter().map(|&x| n(x)).collect()),
        ),
        (
            "streams",
            J::Arr(vec![
                stream_json(&m1f, &body),
                stream_json(&m1s, &body),
                stream_json(&m2f, &body),
                stream_json(&m2s, &body),
            ]),
        ),
        (
            "other_stream_m2_file_descriptor_hex",
            h(&other.bundle.descriptor),
        ),
        ("positive_variants", J::Arr(vec![retry_j, dup_j])),
        ("negative", J::Arr(neg)),
    ])
    .render()
}

#[test]
fn committed_vectors_match_a_fresh_deterministic_run() {
    let fresh = generate();
    assert_eq!(fresh, generate(), "generation is not deterministic");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/vectors/rvsd-v1-vectors.json");
    if std::env::var("RVSD_REGEN").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &fresh).unwrap();
        return;
    }
    let committed =
        std::fs::read_to_string(&path).expect("vectors file missing; run with RVSD_REGEN=1");
    assert!(committed == fresh, "rvsd-v1-vectors.json differs from the generated vectors; rerun with RVSD_REGEN=1 and review the diff");
}
