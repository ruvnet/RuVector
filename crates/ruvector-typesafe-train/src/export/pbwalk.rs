//! Minimal protobuf span walker: locates each graph initializer's float payload
//! (`TensorProto.raw_data` or packed `float_data`) as a byte range of the
//! ORIGINAL file, so the transplant can overwrite weights in place.
//!
//! Why not decode → mutate → re-encode with prost: prost drops any field its
//! schema does not model, so a re-encode could silently change the graph.
//! Patching byte ranges of equal length leaves every other byte identical —
//! the identity transplant is byte-for-byte the template.

use std::collections::HashMap;
use std::ops::Range;

use anyhow::{bail, Result};

const MODEL_GRAPH: u64 = 7;
const GRAPH_INITIALIZER: u64 = 5;
const TENSOR_NAME: u64 = 8;
const TENSOR_FLOAT_DATA: u64 = 4;
const TENSOR_RAW_DATA: u64 = 9;

fn varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let Some(&b) = buf.get(*pos) else {
            bail!("truncated varint")
        };
        *pos += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    bail!("varint overflow")
}

/// Iterate `(field, wire_type, payload_range)` over one message's bytes.
fn fields(buf: &[u8], range: Range<usize>) -> Result<Vec<(u64, u8, Range<usize>)>> {
    let mut out = Vec::new();
    let mut pos = range.start;
    while pos < range.end {
        let key = varint(buf, &mut pos)?;
        let (field, wt) = (key >> 3, (key & 7) as u8);
        let start = pos;
        match wt {
            0 => {
                varint(buf, &mut pos)?;
            }
            1 => pos += 8,
            2 => {
                let len = varint(buf, &mut pos)? as usize;
                let s = pos;
                pos += len;
                out.push((field, wt, s..pos));
                continue;
            }
            5 => pos += 4,
            _ => bail!("unsupported wire type {wt} at byte {start}"),
        }
        if pos > range.end {
            bail!("field overruns message");
        }
        out.push((field, wt, start..pos));
    }
    if pos != range.end {
        bail!("message length mismatch");
    }
    Ok(out)
}

/// Initializer name → byte range of its little-endian f32 payload.
pub fn float_payload_spans(model: &[u8]) -> Result<HashMap<String, Range<usize>>> {
    let graph = fields(model, 0..model.len())?
        .into_iter()
        .find(|(f, wt, _)| *f == MODEL_GRAPH && *wt == 2)
        .map(|(_, _, r)| r);
    let Some(graph) = graph else {
        bail!("ModelProto has no graph")
    };
    let mut spans = HashMap::new();
    for (f, wt, init) in fields(model, graph)? {
        if f != GRAPH_INITIALIZER || wt != 2 {
            continue;
        }
        let mut name = None;
        let mut payload = None;
        for (tf, twt, r) in fields(model, init)? {
            match (tf, twt) {
                (TENSOR_NAME, 2) => name = Some(String::from_utf8(model[r].to_vec())?),
                (TENSOR_RAW_DATA, 2) | (TENSOR_FLOAT_DATA, 2) => payload = Some(r),
                _ => {}
            }
        }
        if let (Some(n), Some(p)) = (name, payload) {
            spans.insert(n, p);
        }
    }
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use tract_onnx::pb::{GraphProto, ModelProto, TensorProto};

    #[test]
    fn finds_raw_and_packed_float_payloads() {
        let raw = TensorProto {
            name: "a".into(),
            dims: vec![2],
            data_type: 1,
            raw_data: [1f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
            ..Default::default()
        };
        let packed = TensorProto {
            name: "b".into(),
            dims: vec![3],
            data_type: 1,
            float_data: vec![3.0, 4.0, 5.0],
            ..Default::default()
        };
        let m = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![raw, packed],
                ..Default::default()
            }),
            ..Default::default()
        };
        let bytes = m.encode_to_vec();
        let spans = float_payload_spans(&bytes).unwrap();
        let get = |n: &str| -> Vec<f32> {
            bytes[spans[n].clone()]
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        assert_eq!(get("a"), vec![1.0, 2.0]);
        assert_eq!(get("b"), vec![3.0, 4.0, 5.0]);
    }
}
