//! Bounded CPU expert cache for the isolated benchmark. Input extents come
//! from the same validated, pinned BF16 file descriptors as the resident path.
use crate::error::InferenceError;
use crate::weights::TensorSource;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, ErrorKind};
use std::os::unix::fs::FileExt;
use std::sync::Arc;

pub(crate) type Extent = (Arc<File>, u64, usize);

#[derive(Debug)]
pub(crate) struct Pager {
    gate: Extent,
    down: Extent,
    capacity: usize,
    entries: VecDeque<(usize, Vec<f32>)>,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) evictions: u64,
    pub(crate) bytes: u64,
}

impl Pager {
    pub(crate) fn new<T: TensorSource + ?Sized>(
        source: &mut T,
        prefix: &str,
        experts: usize,
        hidden: usize,
        intermediate: usize,
        top_k: usize,
        capacity: usize,
    ) -> Result<Self, InferenceError> {
        if capacity < top_k || capacity > experts {
            return Err(InferenceError::InvalidInput(
                "invalid expert cache capacity".into(),
            ));
        }
        let rows = intermediate
            .checked_mul(2)
            .ok_or_else(|| InferenceError::InvalidInput("expert shape overflow".into()))?;
        let gate = source.bench_extent(
            &format!("{prefix}.mlp.experts.gate_up_proj"),
            &[experts, rows, hidden],
        )?;
        let down = source.bench_extent(
            &format!("{prefix}.mlp.experts.down_proj"),
            &[experts, hidden, intermediate],
        )?;
        Ok(Self {
            gate,
            down,
            capacity,
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            bytes: 0,
        })
    }

    pub(crate) fn slots(&self) -> usize {
        self.capacity
    }

    fn read(extent: &Extent, expert: usize, output: &mut Vec<f32>) -> io::Result<()> {
        let (file, start, stride) = extent;
        let offset = (expert as u64)
            .checked_mul(*stride as u64)
            .and_then(|v| start.checked_add(v))
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "expert offset overflow"))?;
        // Keep temporary conversion storage independent of model and cache size.
        let mut buf = [0u8; 32768];
        let mut consumed = 0;
        while consumed < *stride {
            let len = (*stride - consumed).min(buf.len());
            file.read_exact_at(&mut buf[..len], offset + consumed as u64)?;
            for pair in buf[..len].chunks_exact(2) {
                let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
                if !value.is_finite() {
                    return Err(io::Error::new(
                        ErrorKind::InvalidData,
                        "nonfinite expert weight",
                    ));
                }
                output.push(value);
            }
            consumed += len;
        }
        Ok(())
    }

    pub(crate) fn resolve(&mut self, expert: usize) -> io::Result<&[f32]> {
        if let Some(i) = self.entries.iter().position(|(id, _)| *id == expert) {
            let entry = self.entries.remove(i).unwrap();
            self.entries.push_back(entry);
            self.hits += 1;
        } else {
            let mut values = if self.entries.len() == self.capacity {
                self.evictions += 1;
                self.entries.pop_front().unwrap().1
            } else {
                Vec::with_capacity((self.gate.2 + self.down.2) / 2)
            };
            values.clear();
            Self::read(&self.gate, expert, &mut values)?;
            Self::read(&self.down, expert, &mut values)?;
            self.bytes += (self.gate.2 + self.down.2) as u64;
            self.misses += 1;
            self.entries.push_back((expert, values));
        }
        Ok(&self.entries.back().unwrap().1)
    }
}
