//! Edit/policy/run model and stream fixtures for the RVSD vector generator.
#![allow(dead_code)]
use super::common::json::*;
use super::common::*;
use rvf_types::ErrorCode;
use rvf_wire::stream::*;

/// `file_id` the vector pins apply to (with the 0x2A test key).
pub const PIN_FILE: [u8; 16] = *b"rvsd-test-file01";
pub const STAGES: [u8; 7] = [0, 0, 2, 1, 3, 1, 2];

#[derive(Clone)]
pub enum Edit {
    Xor {
        frame: usize,
        offset: usize,
        mask: u8,
    },
    Drop(usize),
    Dup(usize),
    Swap(usize, usize),
    Keep(usize),
    Insert {
        at: usize,
        bytes: Vec<u8>,
    },
    Replace {
        frame: usize,
        bytes: Vec<u8>,
    },
    CutTail {
        frame: usize,
        bytes: usize,
    },
}

pub fn apply(frames: &[Vec<u8>], edits: &[Edit]) -> Vec<Vec<u8>> {
    let mut f = frames.to_vec();
    for e in edits {
        match e.clone() {
            Edit::Xor {
                frame,
                offset,
                mask,
            } => f[frame][offset] ^= mask,
            Edit::Drop(i) => {
                f.remove(i);
            }
            Edit::Dup(i) => f.insert(i + 1, f[i].clone()),
            Edit::Swap(a, b) => f.swap(a, b),
            Edit::Keep(k) => f.truncate(k),
            Edit::Insert { at, bytes } => f.insert(at, bytes),
            Edit::Replace { frame, bytes } => f[frame] = bytes,
            Edit::CutTail { frame, bytes } => {
                let n = f[frame].len() - bytes;
                f[frame].truncate(n);
            }
        }
    }
    f
}

pub fn edit_json(e: &Edit) -> J {
    match e.clone() {
        Edit::Xor {
            frame,
            offset,
            mask,
        } => J::Obj(vec![
            ("op", s("xor")),
            ("frame", n(frame as u64)),
            ("offset", n(offset as u64)),
            ("mask", n(mask)),
        ]),
        Edit::Drop(i) => J::Obj(vec![("op", s("drop")), ("frame", n(i as u64))]),
        Edit::Dup(i) => J::Obj(vec![("op", s("dup")), ("frame", n(i as u64))]),
        Edit::Swap(a, b) => J::Obj(vec![
            ("op", s("swap")),
            ("a", n(a as u64)),
            ("b", n(b as u64)),
        ]),
        Edit::Keep(k) => J::Obj(vec![("op", s("keep_first")), ("count", n(k as u64))]),
        Edit::Insert { at, bytes } => J::Obj(vec![
            ("op", s("insert")),
            ("at", n(at as u64)),
            ("hex", h(&bytes)),
        ]),
        Edit::Replace { frame, bytes } => J::Obj(vec![
            ("op", s("replace")),
            ("frame", n(frame as u64)),
            ("hex", h(&bytes)),
        ]),
        Edit::CutTail { frame, bytes } => J::Obj(vec![
            ("op", s("cut_tail_bytes")),
            ("frame", n(frame as u64)),
            ("bytes", n(bytes as u64)),
        ]),
    }
}

#[derive(Clone)]
pub struct Pol {
    pub trusted: Vec<[u8; 32]>,
    pub budget: u32,
    pub pin: Option<EpochPin>,
    pub now: Option<u64>,
    pub max_dup: Option<u64>,
}
impl Pol {
    pub fn std(budget: u32) -> Self {
        Pol {
            trusted: vec![pk(0x2A)],
            budget,
            pin: None,
            now: None,
            max_dup: None,
        }
    }
    pub fn json(&self) -> J {
        let pin = match self.pin {
            None => J::Null,
            Some(p) => J::Obj(vec![
                ("epoch", n(p.epoch)),
                ("descriptor_hash_hex", h(&p.descriptor_hash)),
                (
                    "signer_key_id_hex",
                    h(&rvf_crypto::signer_key_id(&pk(0x2A))),
                ),
                ("file_id_hex", h(&PIN_FILE)),
            ]),
        };
        J::Obj(vec![
            (
                "trusted_public_keys_hex",
                J::Arr(self.trusted.iter().map(|k| h(k)).collect()),
            ),
            ("fault_budget", n(self.budget)),
            ("epoch_pin", pin),
            ("now_ns", self.now.map_or(J::Null, J::Num)),
            ("max_duplicate_frames", self.max_dup.map_or(J::Null, J::Num)),
        ])
    }
}

pub struct Out {
    pub events: Vec<J>,
    pub errors: Vec<(usize, ErrorCode)>,
    pub state: String,
    pub chunks: u64,
    pub bytes: u64,
}

pub fn state_name(s: StreamState) -> String {
    match s {
        StreamState::Poisoned(e) => format!("Poisoned:{e:?}"),
        StreamState::Rejected(e) => format!("Rejected:{e:?}"),
        o => format!("{o:?}"),
    }
}

pub fn run(p: &Pol, frames: &[Vec<u8>]) -> Out {
    let pin = p.pin;
    let sid = rvf_crypto::signer_key_id(&pk(0x2A));
    let f = move |sg: &[u8; 8], fid: &[u8; 16]| {
        if *sg == sid && *fid == PIN_FILE {
            pin
        } else {
            None
        }
    };
    let mut pol = StreamPolicy::new(&p.trusted);
    pol.fault_budget = p.budget;
    pol.now_ns = p.now;
    pol.max_duplicate_frames = p.max_dup;
    if p.pin.is_some() {
        pol.epoch_pin = Some(&f);
    }
    let mut v = StreamVerifier::new(pol);
    let mut o = Out {
        events: vec![],
        errors: vec![],
        state: String::new(),
        chunks: 0,
        bytes: 0,
    };
    for (i, fr) in frames.iter().enumerate() {
        let ev = match v.push(fr) {
            Ok(Event::DescriptorVerified {
                descriptor_hash,
                seg_count,
                body_len,
            }) => J::Obj(vec![
                ("event", s("DescriptorVerified")),
                ("descriptor_hash_hex", h(&descriptor_hash)),
                ("seg_count", n(seg_count)),
                ("body_len", n(body_len)),
            ]),
            Ok(Event::Duplicate {
                seg_id,
                chunk_index,
            }) => J::Obj(vec![
                ("event", s("Duplicate")),
                ("seg_id", n(seg_id)),
                ("chunk_index", n(chunk_index)),
            ]),
            Ok(Event::Unverified {
                seg_id,
                chunk_index,
            }) => J::Obj(vec![
                ("event", s("Unverified")),
                ("seg_id", n(seg_id)),
                ("chunk_index", n(chunk_index)),
            ]),
            Ok(Event::DuplicateDescriptor) => J::Obj(vec![("event", s("DuplicateDescriptor"))]),
            Ok(Event::TableVerified { seg_id }) => {
                J::Obj(vec![("event", s("TableVerified")), ("seg_id", n(seg_id))])
            }
            Ok(Event::Chunk {
                chunk,
                segment_verified,
                stage_verified,
                complete,
            }) => {
                o.chunks += 1;
                o.bytes += chunk.bytes.len() as u64;
                J::Obj(vec![
                    ("event", s("Chunk")),
                    ("seg_id", n(chunk.seg_id)),
                    ("chunk_index", n(chunk.chunk_index)),
                    ("body_offset", n(chunk.body_offset)),
                    ("len", n(chunk.bytes.len() as u64)),
                    ("segment_verified", J::Bool(segment_verified)),
                    ("stage_verified", stage_verified.map_or(J::Null, n)),
                    (
                        "complete",
                        complete.map_or(J::Null, |c| s(format!("{c:?}"))),
                    ),
                ])
            }
            Err(e) => {
                o.errors.push((i, e));
                J::Obj(vec![
                    ("event", s("Error")),
                    ("reason", s(format!("{e:?}"))),
                    ("code", s(format!("0x{:04x}", e as u16))),
                ])
            }
            Ok(other) => panic!("test model has no mapping for event {other:?}"),
        };
        o.events.push(ev);
    }
    o.state = state_name(v.state());
    o
}

pub struct Stream {
    pub name: &'static str,
    pub bundle: StreamBundle,
    pub frames: Vec<Vec<u8>>,
}

pub fn mk(name: &'static str, body: &[u8], file_id: [u8; 16], log2: u8, stage: bool) -> Stream {
    let mut b = StreamDescriptorBuilder::new(file_id, key(0x2A))
        .epoch(7)
        .chunk_log2(log2)
        .policy_hash([0xAB; 32]);
    if stage {
        b = b.stages(&STAGES).requires_lower_stages(true);
    }
    let bundle = b.build(body).unwrap();
    let frames = if stage {
        bundle.frames_stage_order(body).collect()
    } else {
        bundle.frames_file_order(body).collect()
    };
    Stream {
        name,
        bundle,
        frames,
    }
}

/// A positive variant: base stream plus edits, expected to end in `Complete`.
pub fn variant_json(
    name: &'static str,
    what: &'static str,
    base: &Stream,
    edits: &[Edit],
    pol: &Pol,
) -> J {
    let frames = apply(&base.frames, edits);
    let o = run(pol, &frames);
    assert_eq!(o.state, "Complete", "{name}");
    let first = o.errors.first().copied();
    J::Obj(vec![
        ("name", s(name)),
        ("stream", s(base.name)),
        ("description", s(what)),
        ("policy", pol.json()),
        ("edits", J::Arr(edits.iter().map(edit_json).collect())),
        ("events", J::Arr(o.events)),
        (
            "expect",
            J::Obj(vec![
                ("error_count", n(o.errors.len() as u64)),
                (
                    "first_error_frame",
                    first.map_or(J::Null, |(i, _)| n(i as u64)),
                ),
                (
                    "reason",
                    first.map_or(J::Null, |(_, e)| s(format!("{e:?}"))),
                ),
                (
                    "code",
                    first.map_or(J::Null, |(_, e)| s(format!("0x{:04x}", e as u16))),
                ),
                ("final_state", s(o.state)),
                ("released_chunks", n(o.chunks)),
                ("released_bytes", n(o.bytes)),
                ("complete", J::Bool(true)),
            ]),
        ),
    ])
}
