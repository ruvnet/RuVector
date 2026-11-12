//! Verified progressive streaming: StreamIngest + client over an in-memory
//! transport (the wire variants live in rvf-server's tests).

#[path = "support/stream_fixture.rs"]
mod fx;

use fx::*;
use rvf_runtime::progressive_stream::client::{ClientConfig, MemTransport, RunEnd, StreamClient};
use rvf_runtime::progressive_stream::{QueryMode, QueryPath};
use rvf_runtime::{QueryOptions, RvfStore};
use rvf_wire::stream::{IdentityStatus, StreamPolicy, StreamState};

fn stream_bytes(f: &Fixture) -> Vec<u8> {
    frames(f).concat()
}

fn exact_top(
    ds: &rvf_runtime::progressive_stream::build::StreamDataset,
    q: &[f32],
    k: usize,
) -> Vec<u64> {
    let mut d: Vec<(f32, u64)> = (0..ds.len())
        .map(|i| {
            (
                ds.vector(i)
                    .iter()
                    .zip(q)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>(),
                i as u64,
            )
        })
        .collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d.iter().take(k).map(|x| x.1).collect()
}

#[test]
fn queries_before_complete_return_only_verified_unsealed_data() {
    let f = fixture();
    let keys = [f.public];
    let mut c = StreamClient::new(
        StreamPolicy::new(&keys),
        MemTransport::new(stream_bytes(&f)),
        ClientConfig::default(),
    );
    let q = f.ds.vector(1234).to_vec();
    let mut seen_layers = (false, false, false);
    let mut queries = 0;
    let mut before_a_empty = true;
    let end = c.run(&mut |ing| {
        let a = ing.query(&q, 10, 64);
        queries += 1;
        if !a.sealed {
            // Unsealed answers never claim Verified quality.
            assert_ne!(
                a.envelope.quality,
                rvf_types::quality::ResponseQuality::Verified
            );
        }
        let l = ing.layers();
        // The layers are never searched before Layer A is verified; the exact
        // scan over verified vectors may answer earlier and says so.
        let layers_only = ing.query_with(&q, 10, 64, QueryMode::LayersOnly);
        if !l.layer_a && !layers_only.envelope.results.is_empty() {
            before_a_empty = false;
        }
        if !l.layer_c && !a.envelope.results.is_empty() {
            assert_eq!(a.path, QueryPath::ArrivedScan);
            assert!(a.coverage_vectors <= 1.0 && !a.sealed);
            assert_eq!(
                a.envelope.results[0].retrieval_quality,
                rvf_types::quality::RetrievalQuality::BruteForceBudgeted
            );
        }
        for r in &a.envelope.results {
            assert!(
                ing.vectors().contains(r.id),
                "result {} is not a verified vector",
                r.id
            );
        }
        seen_layers = (
            seen_layers.0 | l.layer_a,
            seen_layers.1 | l.layer_b,
            seen_layers.2 | l.layer_c,
        );
        assert_authentic(ing, &f.ds);
    });
    assert_eq!(end, RunEnd::CompleteUnverified);
    assert!(
        before_a_empty,
        "nothing is searchable before Layer A is verified"
    );
    assert_eq!(seen_layers, (true, true, true));
    assert!(queries > 10);
    let st = c.ingest.state();
    assert_eq!(st, StreamState::CompleteUnverified);
    // Stage order is not file order, so identity needs the persisted body.
    assert_eq!(
        c.ingest.verifier().identity_status(),
        Some(IdentityStatus::Unverifiable)
    );
    c.ingest
        .finish_identity(|off, buf| {
            buf.copy_from_slice(&f.body[off as usize..off as usize + buf.len()]);
            true
        })
        .unwrap();
    let a = c.ingest.query(&q, 10, 256);
    assert!(a.sealed);
    assert_eq!(
        a.envelope.quality,
        rvf_types::quality::ResponseQuality::Verified
    );
    assert!(a.envelope.degradation.is_none());
    assert_eq!(a.visible_vectors, f.ds.len());
}

#[test]
fn completed_stream_answers_like_a_buffered_open_of_the_same_file() {
    let f = fixture();
    let keys = [f.public];
    let mut c = StreamClient::new(
        StreamPolicy::new(&keys),
        MemTransport::new(stream_bytes(&f)),
        ClientConfig::default(),
    );
    assert_eq!(c.run(&mut |_| {}), RunEnd::CompleteUnverified);
    let store = RvfStore::open_readonly(&f.path).unwrap();
    assert!(
        store.index_ready(),
        "persisted Layer C is loaded, index held constant"
    );
    let mut same = 0;
    let mut rec_stream = 0.0;
    let mut rec_store = 0.0;
    let nq = 40;
    for i in 0..nq {
        let q = f.ds.vector(i * 61 % f.ds.len()).to_vec();
        let a: Vec<u64> = c
            .ingest
            .query(&q, 10, 256)
            .envelope
            .results
            .iter()
            .map(|r| r.id)
            .collect();
        let b: Vec<u64> = store
            .query(&q, 10, &QueryOptions::default())
            .unwrap()
            .iter()
            .map(|r| r.id)
            .collect();
        let truth = exact_top(&f.ds, &q, 10);
        same += (a == b) as usize;
        rec_stream += a.iter().filter(|x| truth.contains(x)).count() as f64 / 10.0;
        rec_store += b.iter().filter(|x| truth.contains(x)).count() as f64 / 10.0;
    }
    eprintln!(
        "identical top-10 lists: {same}/{nq}; recall stream {:.3} store {:.3}",
        rec_stream / nq as f64,
        rec_store / nq as f64
    );
    assert!(same as f64 >= 0.9 * nq as f64, "{same}/{nq}");
}

#[test]
fn suspend_and_resume_at_chunk_granularity_costs_no_resend() {
    let f = fixture();
    let keys = [f.public];
    let bytes = stream_bytes(&f);
    let total = bytes.len() as u64;
    let mut t = MemTransport::new(bytes);
    t.cut_after = Some(total as usize / 7 + 13);
    let mut c = StreamClient::new(
        StreamPolicy::new(&keys),
        t,
        ClientConfig {
            max_stalls: 4,
            frame_retries: 5,
        },
    );
    assert_eq!(c.run(&mut |_| {}), RunEnd::CompleteUnverified);
    assert!(c.stats.drops >= 6, "{:?}", c.stats);
    assert_eq!(c.stats.bytes_fetched, total, "resume re-sends nothing");
    assert_authentic(&c.ingest, &f.ds);
}
