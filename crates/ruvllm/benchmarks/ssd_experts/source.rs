//! Single-shard benchmark source using positional reads. The runner pins the
//! model digest and the upstream safetensors validator checks the full layout.
use super::TensorSource;
use super::safetensors_layout::{SafetensorsLayoutEntry, validate_safetensors_layout};
use crate::error::InferenceError;
use serde::Deserialize;
use std::{collections::HashMap, fs::File, io::Read, os::unix::fs::FileExt, path::Path, sync::Arc};

#[derive(Deserialize)]
struct Meta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}
pub(crate) struct BenchFile {
    file: Arc<File>,
    base: u64,
    tensors: HashMap<String, Meta>,
}
fn bad(msg: impl Into<String>) -> InferenceError {
    InferenceError::InvalidSafetensors(msg.into())
}

impl BenchFile {
    pub(crate) fn open(path: &Path) -> Result<Self, InferenceError> {
        let mut file = File::open(path)?;
        let len = file.metadata()?.len();
        let mut prefix = [0; 8];
        file.read_exact(&mut prefix)?;
        let n = u64::from_le_bytes(prefix);
        if n > 100_000_000 || n + 8 > len {
            return Err(bad("invalid header length"));
        }
        let mut bytes = vec![0; n as usize];
        file.read_exact(&mut bytes)?;
        let mut values: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes).map_err(|e| bad(e.to_string()))?;
        values.remove("__metadata__");
        let tensors: HashMap<String, Meta> = values
            .into_iter()
            .map(|(k, v)| {
                Ok((
                    k,
                    serde_json::from_value(v).map_err(|e| bad(e.to_string()))?,
                ))
            })
            .collect::<Result<_, InferenceError>>()?;
        let entries = tensors
            .iter()
            .map(|(k, v)| SafetensorsLayoutEntry {
                name: k,
                dtype: &v.dtype,
                shape: &v.shape,
                start: v.data_offsets[0],
                end: v.data_offsets[1],
            })
            .collect::<Vec<_>>();
        validate_safetensors_layout(
            &path.display().to_string(),
            (len - n - 8)
                .try_into()
                .map_err(|_| bad("file size overflow"))?,
            &entries,
        )?;
        Ok(Self {
            file: Arc::new(file),
            base: 8 + n,
            tensors,
        })
    }
    fn meta(&self, name: &str) -> Result<&Meta, InferenceError> {
        self.tensors
            .get(name)
            .ok_or_else(|| bad(format!("missing {name}")))
    }
}

impl TensorSource for BenchFile {
    fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
        Ok(self.tensors.contains_key(name))
    }
    fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
        Ok(self.tensors.get(name).map(|m| m.shape.clone()))
    }
    fn tensor_dtype(&mut self, name: &str) -> Result<Option<String>, InferenceError> {
        Ok(self.tensors.get(name).map(|m| m.dtype.clone()))
    }
    fn bench_extent(
        &mut self,
        name: &str,
        shape: &[usize],
    ) -> Result<(Arc<File>, u64, usize), InferenceError> {
        let m = self.meta(name)?;
        if shape.len() != 3 || shape[0] == 0 || shape != m.shape || m.dtype != "BF16" {
            return Err(bad(format!("unsupported expert tensor {name}")));
        }
        Ok((
            self.file.clone(),
            self.base + m.data_offsets[0] as u64,
            (m.data_offsets[1] - m.data_offsets[0]) / shape[0],
        ))
    }
    fn get_f32_tensor_owned(
        &mut self,
        name: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
        let m = self.meta(name)?;
        let width = match m.dtype.as_str() {
            "BF16" | "F16" => 2,
            "F32" => 4,
            _ => return Err(bad(format!("unsupported {} in {name}", m.dtype))),
        };
        let [start, end] = m.data_offsets;
        let mut result = Vec::with_capacity((end - start) / width);
        let mut buf = [0; 32768];
        let mut off = start;
        while off < end {
            let n = (end - off).min(buf.len());
            self.file
                .read_exact_at(&mut buf[..n], self.base + off as u64)?;
            for b in buf[..n].chunks_exact(width) {
                let v = match m.dtype.as_str() {
                    "BF16" => super::half_bits::bf16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])),
                    "F16" => super::half_bits::f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])),
                    _ => f32::from_le_bytes(b.try_into().unwrap()),
                };
                if !v.is_finite() {
                    return Err(bad(format!("nonfinite value in {name}")));
                }
                result.push(v);
            }
            off += n;
        }
        Ok((result, m.shape.clone()))
    }
}
