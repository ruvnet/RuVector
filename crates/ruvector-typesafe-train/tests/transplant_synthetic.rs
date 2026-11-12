//! Transplant on a synthetic ONNX graph shaped like the Xenova export: named
//! embeddings/biases/LayerNorms, anonymous transposed `onnx::MatMul_*` weights
//! consumed by `/encoder/layer.N/.../MatMul` nodes, plus an int64 constant.

use prost::Message;
use ruvector_typesafe_train::export::transplant::{map_initializers, transplant, transpose, Named};
use ruvector_typesafe_train::model::BertConfig;
use tract_onnx::pb::{GraphProto, ModelProto, NodeProto, TensorProto};

fn base(seed: u64) -> Named {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    BertConfig::tiny(20)
        .param_shapes()
        .into_iter()
        .map(|(n, shape)| {
            let len = shape.iter().product();
            (n, (shape, (0..len).map(|_| next()).collect()))
        })
        .collect()
}

fn f32_tensor(name: &str, dims: &[usize], v: &[f32]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: dims.iter().map(|&d| d as i64).collect(),
        data_type: 1,
        raw_data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ..Default::default()
    }
}

/// Build the template; `rename` optionally corrupts one consumer node name.
fn template(b: &Named, rename: Option<&str>, extra: Option<TensorProto>) -> Vec<u8> {
    let mut inits = Vec::new();
    let mut nodes = Vec::new();
    let mut anon = 1525;
    for (name, (shape, v)) in b {
        let is_linear =
            name.ends_with(".weight") && shape.len() == 2 && name.starts_with("encoder.");
        if is_linear {
            let id = format!("onnx::MatMul_{anon}");
            anon += 1;
            inits.push(f32_tensor(
                &id,
                &[shape[1], shape[0]],
                &transpose(v, shape[0], shape[1]),
            ));
            let path = ruvector_typesafe_train::export::transplant::module_path(name);
            let node_name = match rename {
                Some(r) if name == r => "/encoder/layer.9/bogus/MatMul".to_string(),
                _ => format!("{path}MatMul"),
            };
            nodes.push(NodeProto {
                name: node_name,
                op_type: "MatMul".into(),
                input: vec!["x".into(), id],
                output: vec![format!("y{anon}")],
                ..Default::default()
            });
        } else {
            inits.push(f32_tensor(name, shape, v));
        }
    }
    inits.push(TensorProto {
        name: "onnx::Slice_214".into(),
        dims: vec![1],
        data_type: 7,
        raw_data: 5i64.to_le_bytes().to_vec(),
        ..Default::default()
    });
    inits.extend(extra);
    ModelProto {
        ir_version: 8,
        producer_name: "synthetic".into(),
        graph: Some(GraphProto {
            node: nodes,
            initializer: inits,
            name: "g".into(),
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

#[test]
fn identity_is_byte_identical_and_bijective() {
    let b = base(1);
    let t = template(&b, None, None);
    let (out, rep) = transplant(&t, &b, &b).unwrap();
    assert!(rep.byte_identical_to_template);
    assert_eq!(out, t);
    assert_eq!(rep.float_initializers, b.len());
    assert_eq!(rep.anonymous_transposed, 12); // 6 linears × 2 layers
    assert_eq!(
        rep.skipped_non_float,
        vec![("onnx::Slice_214".to_string(), 7)]
    );
    assert_eq!(rep.max_abs_delta_vs_base, 0.0);
}

#[test]
fn fine_tuned_values_land_transposed_where_recorded() {
    let b = base(2);
    let mut new = b.clone();
    for (_, v) in new.values_mut() {
        for x in v.iter_mut() {
            *x += 0.25;
        }
    }
    let t = template(&b, None, None);
    let (out, rep) = transplant(&t, &b, &new).unwrap();
    assert!(!rep.byte_identical_to_template && (rep.max_abs_delta_vs_base - 0.25).abs() < 1e-6);
    assert_eq!(out.len(), t.len(), "in-place patch keeps the file size");
    let m = ModelProto::decode(out.as_slice()).unwrap();
    let g = m.graph.unwrap();
    let q = "encoder.layer.1.attention.self.query.weight";
    let map = rep.mappings.iter().find(|m| m.tensor == q).unwrap();
    assert!(
        map.transposed
            && map
                .cross_check
                .contains("/encoder/layer.1/attention/self/query/")
    );
    let init = g
        .initializer
        .iter()
        .find(|i| i.name == map.initializer)
        .unwrap();
    let got: Vec<f32> = init
        .raw_data
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let (shape, v) = &new[q];
    assert_eq!(got, transpose(v, shape[0], shape[1]));
    // untouched int64 constant survives
    assert!(g
        .initializer
        .iter()
        .any(|i| i.name == "onnx::Slice_214" && i.raw_data == 5i64.to_le_bytes()));
}

#[test]
fn unmatched_float_initializer_aborts() {
    let b = base(3);
    let stray = f32_tensor("stray", &[3], &[9.0, 9.0, 9.0]);
    let t = template(&b, None, Some(stray));
    let err = transplant(&t, &b, &b).unwrap_err().to_string();
    assert!(
        err.contains("stray") && err.contains("0 base matches"),
        "{err}"
    );
}

#[test]
fn node_name_disagreement_aborts() {
    let b = base(4);
    let t = template(&b, Some("encoder.layer.0.output.dense.weight"), None);
    let err = transplant(&t, &b, &b).unwrap_err().to_string();
    assert!(
        err.contains("consumers") && err.contains("/encoder/layer.0/output/dense/"),
        "{err}"
    );
}

#[test]
fn ambiguous_value_match_aborts() {
    let mut b = base(5);
    let dup = b["encoder.layer.0.output.LayerNorm.weight"].clone();
    b.insert("encoder.layer.1.output.LayerNorm.weight".into(), dup);
    let t = template(&b, None, None);
    let m = ModelProto::decode(t.as_slice()).unwrap();
    let err = map_initializers(&m, &b).unwrap_err().to_string();
    assert!(err.contains("2 base matches"), "{err}");
}

#[test]
fn missing_template_tensor_breaks_bijection() {
    let b = base(6);
    let mut smaller = b.clone();
    smaller.remove("embeddings.token_type_embeddings.weight");
    let t = template(&smaller, None, None);
    let m = ModelProto::decode(t.as_slice()).unwrap();
    let err = map_initializers(&m, &b).unwrap_err().to_string();
    assert!(
        err.contains("embeddings.token_type_embeddings.weight mapped 0 times"),
        "{err}"
    );
}
