//! ADR-353 over real sockets: server endpoints, range semantics, abuse
//! limits, the throttled end-to-end run, and a lying HTTP mirror.

mod support;

use rvf_runtime::progressive_stream::client::{ClientConfig, RunEnd, StreamClient, Throttled};
use rvf_runtime::progressive_stream::StreamIngest;
use rvf_runtime::{QueryOptions, RvfStore};
use rvf_server::stream_client::{HttpTransport, TcpTransport};
use rvf_wire::stream::{StreamPolicy, StreamState};
use std::time::Instant;
use support::*;

fn n() -> usize {
    std::env::var("OFER_STREAM_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000)
}

fn authentic(ing: &StreamIngest<'_>, f: &Fixture) {
    for (id, v) in ing.vectors().iter() {
        assert_eq!(v, f.ds.vector(id as usize));
    }
}

#[test]
fn descriptor_and_range_semantics_and_abuse_limits() {
    let f = fixture(2000, 16, 8, 3);
    let r = start(&f, |c| {
        c.max_concurrent_streams = 1;
        c.client_cap = rvf_server::stream::ClientCap::Disabled;
        c.max_range_bytes = 10_000;
    });
    let (st, h, d) = raw_get(r.http, "/v1/stream/descriptor", None);
    assert_eq!(st, 200);
    assert_eq!(d, r.server.descriptor());
    assert!(h.iter().any(|(k, _)| k == "x-rvf-descriptor-sha256"));
    let total = r.server.stream_len();

    let (st, h, b) = raw_get(r.http, "/v1/stream/body", Some("bytes=100-199"));
    assert_eq!(st, 206);
    assert_eq!(b, r.server.read_range(100, 200));
    assert!(h.contains(&("content-range".into(), format!("bytes 100-199/{total}"))));

    // Per-response cap: an open-ended range is clamped to max_range_bytes.
    let (st, h, b) = raw_get(r.http, "/v1/stream/body", Some("bytes=5-"));
    assert_eq!(st, 206);
    assert_eq!(b.len(), 10_000);
    assert!(h.contains(&("content-range".into(), format!("bytes 5-10004/{total}"))));

    for bad in [
        "bytes=0-1,5-6",
        "bytes=-5",
        "bytes=20-10",
        "pages=0-1",
        "bytes=99999999999-",
    ] {
        let (st, _, _) = raw_get(r.http, "/v1/stream/body", Some(bad));
        assert_eq!(st, 416, "{bad}");
    }

    // Concurrent-stream limit (max 1): with the only permit taken, a body
    // request is refused; once released it is served.
    let permit = r.server.try_permit(None).expect("one permit");
    assert!(r.server.try_permit(None).is_err());
    let (st, _, _) = raw_get(r.http, "/v1/stream/body", Some("bytes=0-9"));
    assert_eq!(st, 429);
    // The descriptor endpoint is not a stream and is unaffected.
    assert_eq!(raw_get(r.http, "/v1/stream/descriptor", None).0, 200);
    drop(permit);
    assert_eq!(raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0, 206);
}

#[test]
fn end_to_end_over_a_throttled_link_equals_a_buffered_open() {
    let nn = n();
    let f = fixture(
        nn,
        128.min(if nn > 20_000 { 128 } else { 32 }),
        64.min(nn / 40).max(8),
        7,
    );
    let r = start(&f, |_| {});
    let desc = read_descriptor(r.http);
    let keys = [public_key()];
    let rates: Vec<f64> = std::env::var("OFER_STREAM_MBPS")
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|_| vec![1000.0]);
    let store = RvfStore::open_readonly(&f.rvf).unwrap();
    assert!(
        store.index_ready(),
        "persisted index loaded: the buffered side uses the same graph"
    );
    let qs: Vec<Vec<f32>> = (0..30)
        .map(|i| f.ds.vector(i * 97 % f.ds.len()).to_vec())
        .collect();
    eprintln!(
        "body {} bytes, stream {} bytes, descriptor {} bytes",
        std::fs::metadata(&f.rvf).unwrap().len(),
        r.server.stream_len(),
        desc.len()
    );
    for mbps in rates {
        let t = Throttled {
            inner: HttpTransport::new(r.http.to_string(), "/v1/stream/body"),
            mbps,
        };
        let mut c = StreamClient::new(StreamPolicy::new(&keys), t, ClientConfig::default());
        let t0 = Instant::now();
        let mut first = [None::<f64>; 3];
        let mut unsealed_answers = 0;
        let end = c.run(&mut |ing| {
            let a = ing.query(&qs[0], 10, 256);
            let l = ing.layers();
            for (i, on) in [l.layer_a, l.layer_b, l.layer_c].into_iter().enumerate() {
                if on && first[i].is_none() {
                    first[i] = Some(t0.elapsed().as_secs_f64());
                }
            }
            for r in &a.envelope.results {
                assert!(ing.vectors().contains(r.id));
            }
            if !a.sealed {
                unsealed_answers += 1;
                assert_ne!(
                    a.envelope.quality,
                    rvf_types::quality::ResponseQuality::Verified
                );
            }
            authentic(ing, &f);
        });
        let total = t0.elapsed().as_secs_f64();
        assert_eq!(end, RunEnd::CompleteUnverified);
        assert_eq!(c.ingest.state(), StreamState::CompleteUnverified);
        let mut same = 0;
        for q in &qs {
            let a: Vec<u64> = c
                .ingest
                .query(q, 10, 256)
                .envelope
                .results
                .iter()
                .map(|x| x.id)
                .collect();
            let b: Vec<u64> = store
                .query(q, 10, &QueryOptions::default())
                .unwrap()
                .iter()
                .map(|x| x.id)
                .collect();
            same += (a == b) as usize;
        }
        eprintln!(
            "{mbps:>6} Mbps: A@{:.2}s B@{:.2}s C@{:.2}s complete@{total:.2}s, {unsealed_answers} unsealed answers, identical top-10 {same}/{}",
            first[0].unwrap_or(f64::NAN), first[1].unwrap_or(f64::NAN), first[2].unwrap_or(f64::NAN), qs.len()
        );
        assert!(same * 10 >= qs.len() * 9, "{same}/{}", qs.len());
        assert!(first[0].unwrap() <= first[1].unwrap() && first[1].unwrap() <= first[2].unwrap());
    }
}

#[test]
fn tcp_message_pair_delivers_the_same_stream() {
    let f = fixture(2000, 16, 8, 3);
    let r = start(&f, |c| c.tcp_max_response = 200_000);
    let keys = [public_key()];
    let mut c = StreamClient::new(
        StreamPolicy::new(&keys),
        TcpTransport::new(r.tcp.to_string(), 1 << 20),
        ClientConfig::default(),
    );
    assert_eq!(
        c.run(&mut |ing| authentic(ing, &f)),
        RunEnd::CompleteUnverified
    );
    assert_eq!(c.stats.bytes_fetched, r.server.stream_len());
    let d = TcpTransport::new(r.tcp.to_string(), 1)
        .descriptor()
        .unwrap();
    assert_eq!(d, r.server.descriptor());
}

#[test]
fn http_mirror_lies_never_publish_unverified_data() {
    let f = fixture(2000, 16, 8, 3);
    let r = start(&f, |_| {});
    let honest = {
        let mut v = Vec::new();
        r.server.stream_len();
        v.extend_from_slice(&r.server.read_range(0, r.server.stream_len()));
        v
    };
    let keys = [public_key()];
    let cut = Some(honest.len() / 3);
    for lie in [
        Lie::None,
        Lie::LengthTooBig,
        Lie::LengthTooSmall,
        Lie::RangeStartsAtZero,
        Lie::IgnoresRange,
        Lie::WrongTotal,
    ] {
        let addr = mirror(honest.clone(), lie, cut);
        let mut c = StreamClient::new(
            StreamPolicy::new(&keys),
            HttpTransport::new(addr.to_string(), "/v1/stream/body"),
            ClientConfig {
                max_stalls: 3,
                frame_retries: 3,
            },
        );
        let end = c.run(&mut |ing| authentic(ing, &f));
        authentic(&c.ingest, &f);
        eprintln!(
            "mirror {lie:?}: {end:?}, drops {}, fetched {}",
            c.stats.drops, c.stats.bytes_fetched
        );
        match lie {
            // Control: an honest mirror that drops once resumes to Complete.
            Lie::None => assert_eq!(end, RunEnd::CompleteUnverified),
            // Harmless or self-correcting lies may still complete, but only
            // with fully verified content.
            Lie::LengthTooBig | Lie::LengthTooSmall => {
                if end.all_verified() {
                    assert_eq!(c.ingest.vectors().len(), f.ds.len());
                }
            }
            // Resuming against a mirror that ignores / misreports Range must
            // be refused by the transport, so the stream cannot complete.
            Lie::RangeStartsAtZero | Lie::IgnoresRange | Lie::WrongTotal => {
                assert!(!end.all_verified(), "{lie:?}");
                assert!(c.stats.drops >= 1);
            }
        }
        if !end.all_verified() {
            assert!(!matches!(
                c.ingest.state(),
                StreamState::Complete | StreamState::CompleteUnverified
            ));
        }
    }
    // Bit flips behind correct headers: refused by the verifier.
    let mut flipped = honest.clone();
    let at = flipped.len() / 2;
    flipped[at] ^= 0x01;
    let addr = mirror(flipped, Lie::None, None);
    let mut c = StreamClient::new(
        StreamPolicy::new(&keys),
        HttpTransport::new(addr.to_string(), "/v1/stream/body"),
        ClientConfig {
            max_stalls: 3,
            frame_retries: 3,
        },
    );
    let end = c.run(&mut |ing| authentic(ing, &f));
    assert!(!end.all_verified());
    assert!(c.stats.refusals >= 1);
}
