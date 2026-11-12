//! Descriptor decoding: rejections, epoch/expiry/pin handling and signed-but-malformed tables.
mod common;
use common::*;
use rvf_types::stream::*;
use rvf_types::ErrorCode as E;
use rvf_wire::stream::*;

fn bundle(body: &[u8], log2: u8) -> StreamBundle {
    StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(log2)
        .build(body)
        .unwrap()
}

#[test]
fn descriptor_rejections() {
    let body = sample_body(1);
    let b = bundle(&body, 12);
    // exhaustive bit flips on a two-segment descriptor (debug-build Ed25519 is slow)
    let small = {
        let f = rvf_types::SegmentFlags::empty();
        let mut s = rvf_wire::writer::write_segment(0x07, &payload(1, 100), f, 1);
        s.extend(root_page());
        s
    };
    let sb = bundle(&small, 12);
    {
        let keys = [pk(1)];
        let pol = policy(&keys);
        assert!(decode_descriptor(&sb.descriptor, &pol).is_ok());
        for bit in 0..sb.descriptor.len() * 8 {
            let mut d = sb.descriptor.clone();
            d[bit / 8] ^= 1 << (bit % 8);
            assert!(decode_descriptor(&d, &pol).is_err(), "flip {bit} accepted");
        }
    }
    let keys = [pk(1)];
    let pol = policy(&keys);
    assert!(decode_descriptor(&b.descriptor, &pol).is_ok());
    // truncation / extension
    for n in 0..b.descriptor.len() {
        assert!(decode_descriptor(&b.descriptor[..n], &pol).is_err());
    }
    let mut ext = b.descriptor.clone();
    ext.push(0);
    assert_eq!(
        decode_descriptor(&ext, &pol).unwrap_err(),
        E::StreamDescriptorTruncated
    );
    // unknown signer / wrong key
    let other = [pk(2)];
    assert_eq!(
        decode_descriptor(&b.descriptor, &policy(&other)).unwrap_err(),
        E::UnknownSigner
    );
    let none: [[u8; 32]; 0] = [];
    assert_eq!(
        decode_descriptor(&b.descriptor, &policy(&none)).unwrap_err(),
        E::UnknownSigner
    );
    // limits
    let mut p = policy(&keys);
    p.max_segments = 2;
    assert_eq!(
        decode_descriptor(&b.descriptor, &p).unwrap_err(),
        E::StreamLimitExceeded
    );
    let mut p = policy(&keys);
    p.max_descriptor_bytes = 100;
    assert_eq!(
        decode_descriptor(&b.descriptor, &p).unwrap_err(),
        E::StreamLimitExceeded
    );
    let mut p = policy(&keys);
    p.max_class_bytes[StreamClass::Executable as usize] = 100;
    assert_eq!(
        decode_descriptor(&b.descriptor, &p).unwrap_err(),
        E::StreamLimitExceeded
    );
    let mut p = policy(&keys);
    p.max_segment_bytes = 1000;
    assert_eq!(
        decode_descriptor(&b.descriptor, &p).unwrap_err(),
        E::StreamLimitExceeded
    );
    let mut p = policy(&keys);
    p.expected_policy_hash = Some([9; 32]);
    assert_eq!(
        decode_descriptor(&b.descriptor, &p).unwrap_err(),
        E::StreamPolicyMismatch
    );
}

#[test]
fn epoch_expiry_and_pin() {
    let body = sample_body(1);
    let keys = [pk(1)];
    let b5 = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .epoch(5)
        .expires_ns(1000)
        .build(&body)
        .unwrap();
    let mut p = policy(&keys);
    p.now_ns = Some(999);
    assert!(decode_descriptor(&b5.descriptor, &p).is_ok());
    p.now_ns = Some(1000);
    assert_eq!(
        decode_descriptor(&b5.descriptor, &p).unwrap_err(),
        E::StreamExpired
    );
    let sid = rvf_crypto::signer_key_id(&pk(1));
    let pin = |e: u64, h: [u8; 32]| {
        move |s: &[u8; 8], f: &[u8; 16]| {
            (*s == sid && *f == FILE_ID).then_some(EpochPin {
                epoch: e,
                descriptor_hash: h,
            })
        }
    };
    let f = pin(6, [0; 32]);
    let mut p = policy(&keys);
    p.epoch_pin = Some(&f);
    assert_eq!(
        decode_descriptor(&b5.descriptor, &p).unwrap_err(),
        E::StreamStaleEpoch
    );
    let f = pin(5, b5.descriptor_hash);
    p.epoch_pin = Some(&f);
    assert!(decode_descriptor(&b5.descriptor, &p).is_ok());
    // same epoch, different content (different policy hash)
    let b5b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .epoch(5)
        .policy_hash([1; 32])
        .build(&body)
        .unwrap();
    assert_eq!(
        decode_descriptor(&b5b.descriptor, &p).unwrap_err(),
        E::StreamEpochConflict
    );
    let f = pin(4, b5.descriptor_hash);
    p.epoch_pin = Some(&f);
    assert!(decode_descriptor(&b5b.descriptor, &p).is_ok());
}

#[test]
fn signed_but_malformed_tables_are_rejected() {
    use ed25519_dalek::SigningKey;
    let body = sample_body(1);
    let b = bundle(&body, 12);
    let keys = [pk(1)];
    let pol = policy(&keys);
    let resign = |mutate: &dyn Fn(&mut Vec<StreamEntry>, &mut StreamHeader)| {
        let (mut es, mut h) = (b.entries.clone(), b.header);
        mutate(&mut es, &mut h);
        h.seg_count = es.len() as u32;
        h.table_len = h.seg_count * 72;
        let mut d = h.to_bytes().to_vec();
        for e in &es {
            d.extend_from_slice(&e.to_bytes());
        }
        d.extend_from_slice(&64u16.to_le_bytes());
        let sk: SigningKey = key(1);
        let sig = rvf_crypto::sign_descriptor(&d, &sk);
        d.extend_from_slice(&sig);
        decode_descriptor(&d, &pol).unwrap_err()
    };
    use E::{StreamLimitExceeded as Lim, StreamTableInvalid as Bad};
    assert_eq!(resign(&|es, _| es.swap(0, 1)), Bad); // unsorted
    assert_eq!(resign(&|es, _| es[1].body_offset += 64), Bad); // gap
    assert_eq!(resign(&|es, _| es[1].body_offset -= 64), Bad); // overlap
    assert_eq!(resign(&|es, _| es[2].seg_id = es[1].seg_id), Bad); // duplicate id
    assert_eq!(resign(&|_, h| h.body_len -= 1), Bad); // body_len lie
    assert_eq!(resign(&|es, _| es[0].stored_len = 10), Bad); // < 64
    assert_eq!(resign(&|es, _| es[0].stored_len = 0), Bad);
    assert_eq!(resign(&|es, _| es[0].stage = 32), Bad);
    assert_eq!(
        resign(&|es, _| {
            es[0].stage = 1;
            es[0].requires_mask = 2
        }),
        Bad
    ); // self-require
    assert_eq!(
        resign(&|es, _| {
            es[0].stage = 1;
            es[0].requires_mask = 1;
            es[1].stage = 0;
            es[1].requires_mask = 2
        }),
        Bad
    ); // cycle
    assert_eq!(
        resign(&|es, _| {
            let n = es.len();
            es[n - 1].seg_type = 0x01;
            es[0].seg_type = 0xFE
        }),
        Bad
    ); // root page not last
    assert_eq!(resign(&|es, _| es.clear()), Bad);
    assert_eq!(resign(&|es, _| es[0].uncompressed_len = u64::MAX), Lim);
    assert_eq!(resign(&|es, _| es[0].stored_len = u64::MAX / 2), Lim);
    assert_eq!(resign(&|es, _| es[0].stored_len = u64::MAX), Lim);
    assert_eq!(resign(&|_, h| h.body_len = u64::MAX), Lim);
}
