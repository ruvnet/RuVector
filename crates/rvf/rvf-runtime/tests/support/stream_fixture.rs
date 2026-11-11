//! Stream-ready fixture shared by the progressive-stream tests.
#![allow(dead_code)]

use ed25519_dalek::SigningKey;
use rvf_runtime::progressive_stream::build::{
    build_stream_store, descriptor_for, BuildParams, StreamDataset,
};
use rvf_wire::stream::StreamBundle;
use std::path::PathBuf;

pub struct Fixture {
    pub _dir: tempfile::TempDir,
    pub path: PathBuf,
    pub ds: StreamDataset,
    pub body: Vec<u8>,
    pub bundle: StreamBundle,
    pub key: SigningKey,
    pub public: [u8; 32],
    pub file_id: [u8; 16],
}

pub fn fixture_with(n: usize, dim: usize, clusters: usize, seed: u64, epoch: u64) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream.rvf");
    let ds = StreamDataset::clustered(n, dim, clusters, 0.4, seed);
    let p = BuildParams {
        vec_batch: 700,
        chunk_log2: 12,
        epoch,
        ..Default::default()
    };
    let secret = [seed as u8 ^ 0x5a; 32];
    let built = build_stream_store(&path, &ds, &p, &secret).unwrap();
    let key = SigningKey::from_bytes(&[0xA7; 32]);
    let (body, bundle) = descriptor_for(&path, built.file_id, key.clone(), &p).unwrap();
    Fixture {
        _dir: dir,
        path,
        ds,
        body,
        bundle,
        public: key.verifying_key().to_bytes(),
        key,
        file_id: built.file_id,
    }
}

pub fn fixture() -> Fixture {
    fixture_with(2600, 16, 8, 11, 1)
}

/// Frames in delivery (stage) order, descriptor first.
pub fn frames(f: &Fixture) -> Vec<Vec<u8>> {
    f.bundle.frames_stage_order(&f.body).collect()
}

/// Every visible vector must equal the dataset's vector (authenticity).
pub fn assert_authentic(
    ing: &rvf_runtime::progressive_stream::StreamIngest<'_>,
    ds: &StreamDataset,
) {
    for (id, v) in ing.vectors().iter() {
        assert_eq!(
            v,
            ds.vector(id as usize),
            "visible vector {id} is not authentic"
        );
    }
}
