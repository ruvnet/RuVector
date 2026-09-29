use super::*;
use crate::bytes::tests::Segs;
use crate::metric::{distance, dot, l2_sq};

fn sample(dim: usize, rows: usize) -> Vec<f32> {
    (0..dim * rows)
        .map(|i| ((i as f32) * 0.618).sin() * 3.0)
        .collect()
}

fn roundtrip(q: &QuantParams) -> QuantParams {
    let mut buf = Vec::new();
    q.write(&mut buf);
    let n = buf.len();
    let mut src = Segs(vec![buf]);
    QuantParams::read(&mut Reader::new(&mut src, n)).unwrap()
}

#[test]
fn roundtrip_error_is_bounded_by_half_step() {
    let dim = 16;
    let s = sample(dim, 50);
    let q = QuantParams::train(Metric::L2, dim, &s, 3).unwrap();
    assert_eq!((q.dim(), q.epoch(), q.kind()), (dim, 3, QuantKind::PerDim));
    let mut code = vec![0u8; dim];
    let mut back = vec![0f32; dim];
    for row in s.chunks_exact(dim) {
        q.encode(row, &mut code);
        q.decode(&code, &mut back);
        for i in 0..dim {
            assert!((row[i] - back[i]).abs() <= q.scale()[i] * 0.5 + 1e-5);
        }
    }
}

#[test]
fn asymmetric_distance_matches_dequantized() {
    let dim = 19;
    let s = sample(dim, 20);
    for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
        let q = QuantParams::train(metric, dim, &s, 1).unwrap();
        let (a, b) = (&s[..dim], &s[dim..2 * dim]);
        let mut code = vec![0u8; dim];
        let mut deq = vec![0f32; dim];
        q.encode(b, &mut code);
        q.decode(&code, &mut deq);
        let pq = q.prepare(a);
        let expect = match metric {
            Metric::L2 => l2_sq(a, &deq),
            Metric::Dot => -dot(a, &deq),
            Metric::Cosine => 1.0 - dot(a, &deq) / norm(a),
        };
        assert!((pq.distance(&code) - expect).abs() < 1e-3, "{metric:?}");
        let exact = distance(metric, a, b);
        let approx = if metric == Metric::L2 {
            pq.distance(&code).sqrt()
        } else {
            pq.distance(&code)
        };
        assert!(
            (approx - exact).abs() < 0.01 * exact.abs() + 0.05,
            "{metric:?} {approx} {exact}"
        );
    }
}

#[test]
fn rejects_bad_params_and_clamps() {
    assert!(QuantParams::fixed_range(Metric::L2, 4, 1.0, 1.0, 0).is_err());
    assert!(QuantParams::fixed_range(Metric::L2, 0, 0.0, 1.0, 0).is_err());
    assert!(QuantParams::fixed_range(Metric::L2, MAX_DIM + 1, 0.0, 1.0, 0).is_err());
    assert!(QuantParams::fixed_range(Metric::L2, 2, 0.0, f32::INFINITY, 0).is_err());
    assert!(QuantParams::train(Metric::L2, 4, &[1.0; 6], 0).is_err());
    assert!(QuantParams::train(Metric::L2, 2, &[1.0, f32::NAN], 0).is_err());
    assert!(QuantParams::train(Metric::Cosine, 2, &[0.0, 0.0], 0).is_err());
    let q = QuantParams::cosine_fixed(2, 0).unwrap();
    let mut c = [0u8; 2];
    q.encode(&[3.0, -4.0], &mut c);
    assert_eq!(c, [204, 25]); // -0.8 normalises to -0.79999995
    let t = QuantParams::train(Metric::L2, 2, &[0.0, 5.0, 1.0, 5.0], 0).unwrap();
    t.encode(&[9.0, -9.0], &mut c);
    assert_eq!(c, [255, 0]);
    assert_eq!(roundtrip(&t), t);
}

#[test]
fn constant_training_dimension_still_encodes_variation() {
    // dim 1 is always 0.0 in the sample (e.g. a 1k reservoir that never
    // saw it move); later rows use it.
    let s: Vec<f32> = (0..100)
        .flat_map(|i| [(i as f32 / 99.0) * 2.0 - 1.0, 0.0])
        .collect();
    let q = QuantParams::train(Metric::L2, 2, &s, 1).unwrap();
    assert!(
        (q.scale()[1] - q.scale()[0]).abs() < 1e-9,
        "widest step reused"
    );
    let (mut c, mut back) = ([0u8; 2], [0f32; 2]);
    for x in [0.0f32, 0.4, -0.4, 0.9] {
        q.encode(&[0.0, x], &mut c);
        q.decode(&c, &mut back);
        assert!(
            (back[1] - x).abs() <= q.scale()[1] * 0.5 + 1e-6,
            "{x} -> {}",
            back[1]
        );
    }
    // A single training row: every dimension is degenerate.
    let one = QuantParams::train(Metric::L2, 2, &[3.0, 0.0], 1).unwrap();
    assert!(one.scale().iter().all(|s| s.is_finite() && *s > 0.0));
    one.encode(&[3.0, 0.0], &mut c);
    assert_eq!(c, [128, 128]);
    one.encode(&[3.1, 0.0], &mut c);
    assert!(c[0] > 128, "variation around a constant is kept");
    assert_eq!(roundtrip(&one), one);
}

#[test]
fn oversized_dim_in_payload_is_malformed_on_every_platform() {
    let mut buf = Vec::new();
    buf.put_u8(Metric::L2.code());
    buf.put_u8(2);
    buf.put_u32(1 << 16);
    buf.put_u64(0);
    buf.extend(std::iter::repeat(0u8).take(64));
    let n = buf.len();
    let mut src = Segs(vec![buf]);
    let err = QuantParams::read(&mut Reader::new(&mut src, n)).unwrap_err();
    assert_eq!(err, DecodeError::Malformed("quant dim"));
}
