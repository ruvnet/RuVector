//! Bounded SafeTensors input validation and authenticated tensor reads.

use crate::error::{Result, RuvLLMError};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

const MAX_JSON_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_JSON_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TENSORS: usize = 100_000;
const MAX_SHARDS: usize = 1024;
const MAX_RANK: usize = 4;
const HASH_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SourceDtype {
    F32,
    F16,
    BF16,
}

impl SourceDtype {
    pub(super) fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct SourceTensor {
    pub(super) shape: Vec<usize>,
    pub(super) dtype: SourceDtype,
    pub(super) file: PathBuf,
    pub(super) offset: u64,
    pub(super) byte_len: u64,
    sha256: [u8; 32],
}

#[derive(Debug)]
struct SourceFile {
    handle: Mutex<File>,
    byte_len: u64,
}

#[derive(Debug)]
pub(super) struct Checkpoint {
    pub(super) root: PathBuf,
    pub(super) tensors: BTreeMap<String, SourceTensor>,
    files: BTreeMap<PathBuf, Arc<SourceFile>>,
}

impl Checkpoint {
    pub(super) fn tensor_reader(&self, tensor: &SourceTensor) -> Result<TensorReader> {
        let file = self
            .files
            .get(&tensor.file)
            .ok_or_else(|| invalid("tensor does not belong to an opened checkpoint"))?;
        let end = tensor
            .offset
            .checked_add(tensor.byte_len)
            .ok_or_else(|| invalid("tensor file offset overflow"))?;
        if end > file.byte_len {
            return Err(invalid("tensor span exceeds its source file"));
        }
        Ok(TensorReader {
            file: Arc::clone(file),
            offset: tensor.offset,
            remaining: tensor.byte_len,
            digest: Sha256::new(),
            expected: tensor.sha256,
        })
    }
}

/// The caller must finish each reader before publishing converted output.
pub(super) struct TensorReader {
    file: Arc<SourceFile>,
    offset: u64,
    remaining: u64,
    digest: Sha256,
    expected: [u8; 32],
}

impl TensorReader {
    pub(super) fn finish(self) -> Result<()> {
        if self.remaining != 0 {
            return Err(invalid(
                "conversion did not consume the complete source tensor",
            ));
        }
        let handle = self
            .file
            .handle
            .lock()
            .map_err(|_| invalid("source file lock is poisoned"))?;
        if handle.metadata()?.len() != self.file.byte_len {
            return Err(invalid("source file length changed during conversion"));
        }
        let actual: [u8; 32] = self.digest.finalize().into();
        if actual != self.expected {
            return Err(invalid("source tensor changed after preflight"));
        }
        Ok(())
    }
}

impl Read for TensorReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() || self.remaining == 0 {
            return Ok(0);
        }
        let count = self.remaining.min(buffer.len() as u64) as usize;
        let mut handle = self
            .file
            .handle
            .lock()
            .map_err(|_| io::Error::other("source file lock is poisoned"))?;
        if handle.metadata()?.len() != self.file.byte_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "source file length changed during conversion",
            ));
        }
        handle.seek(SeekFrom::Start(self.offset))?;
        let count = handle.read(&mut buffer[..count])?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "source tensor was truncated during conversion",
            ));
        }
        self.digest.update(&buffer[..count]);
        self.offset += count as u64;
        self.remaining -= count as u64;
        Ok(count)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct Header(BTreeMap<String, TensorInfo>);

impl<'de> Deserialize<'de> for Header {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = Header;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a SafeTensors header object with unique names")
            }

            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Header, M::Error> {
                let mut tensors = BTreeMap::new();
                let mut seen_metadata = false;
                while let Some(name) = map.next_key::<String>()? {
                    if name == "__metadata__" {
                        if seen_metadata {
                            return Err(de::Error::custom("duplicate __metadata__ key"));
                        }
                        seen_metadata = true;
                        let _: UniqueMap<String> = map.next_value()?;
                    } else {
                        if name.is_empty() || tensors.contains_key(&name) {
                            return Err(de::Error::custom("empty or duplicate tensor name"));
                        }
                        if tensors.len() >= MAX_TENSORS {
                            return Err(de::Error::custom("too many tensors in header"));
                        }
                        tensors.insert(name, map.next_value::<TensorInfo>()?);
                    }
                }
                Ok(Header(tensors))
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

struct UniqueMap<V>(BTreeMap<String, V>);

impl<'de, V: Deserialize<'de>> Deserialize<'de> for UniqueMap<V> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct UniqueVisitor<V>(PhantomData<V>);
        impl<'de, V: Deserialize<'de>> Visitor<'de> for UniqueVisitor<V> {
            type Value = UniqueMap<V>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an object with unique nonempty keys")
            }

            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut values = BTreeMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    if key.is_empty() || values.contains_key(&key) {
                        return Err(de::Error::custom("empty or duplicate JSON key"));
                    }
                    if values.len() >= MAX_TENSORS {
                        return Err(de::Error::custom("too many JSON keys"));
                    }
                    values.insert(key, map.next_value()?);
                }
                Ok(UniqueMap(values))
            }
        }
        deserializer.deserialize_map(UniqueVisitor(PhantomData))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    metadata: Option<IndexMetadata>,
    weight_map: UniqueMap<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IndexMetadata {
    total_size: Option<u64>,
}

pub(super) fn open(input: &Path) -> Result<Checkpoint> {
    let input = fs::canonicalize(input)?;
    let is_directory = input.is_dir();
    let root = if is_directory {
        input.clone()
    } else {
        input
            .parent()
            .ok_or_else(|| invalid("checkpoint path has no parent directory"))?
            .to_path_buf()
    };
    let mut checkpoint = Checkpoint {
        root,
        tensors: BTreeMap::new(),
        files: BTreeMap::new(),
    };
    let mut metadata_bytes = 0;
    if !is_directory {
        require_safetensors(&input)?;
        checkpoint.add_file(&input, None, &mut metadata_bytes)?;
    } else {
        let single = input.join("model.safetensors");
        let index_path = input.join("model.safetensors.index.json");
        match (single.try_exists()?, index_path.try_exists()?) {
            (true, false) => {
                let single = contained_path(&input, "model.safetensors")?;
                checkpoint.add_file(&single, None, &mut metadata_bytes)?;
            }
            (false, true) => {
                let index_path = contained_path(&input, "model.safetensors.index.json")?;
                let bytes = read_json_file(&index_path)?;
                metadata_bytes = bytes.len() as u64;
                let index: Index = serde_json::from_slice(&bytes)
                    .map_err(|error| invalid(format!("invalid SafeTensors index: {error}")))?;
                drop(bytes);
                if index.weight_map.0.is_empty() {
                    return Err(invalid("SafeTensors index contains no tensors"));
                }
                let mut shards: BTreeMap<PathBuf, BTreeMap<String, ()>> = BTreeMap::new();
                let mut spellings = BTreeMap::new();
                for (name, relative) in index.weight_map.0 {
                    let path = contained_path(&input, &relative)?;
                    require_safetensors(&path)?;
                    if let Some(previous) = spellings.insert(path.clone(), relative.clone()) {
                        if previous != relative {
                            return Err(invalid("index refers to a shard through multiple paths"));
                        }
                    }
                    shards.entry(path).or_default().insert(name, ());
                    if shards.len() > MAX_SHARDS {
                        return Err(invalid("SafeTensors index exceeds the shard limit"));
                    }
                }
                for (path, expected) in shards {
                    checkpoint.add_file(&path, Some(&expected), &mut metadata_bytes)?;
                }
                if let Some(expected_bytes) = index.metadata.and_then(|value| value.total_size) {
                    let actual_bytes = checkpoint
                        .tensors
                        .values()
                        .try_fold(0u64, |total, tensor| total.checked_add(tensor.byte_len));
                    if actual_bytes != Some(expected_bytes) {
                        return Err(invalid(
                            "index total_size does not match tensor payload bytes",
                        ));
                    }
                }
            }
            (true, true) => {
                return Err(invalid(
                    "ambiguous checkpoint: both a single file and shard index exist",
                ))
            }
            (false, false) => {
                return Err(invalid(
                    "directory requires model.safetensors or model.safetensors.index.json",
                ))
            }
        }
    }
    Ok(checkpoint)
}

impl Checkpoint {
    fn add_file(
        &mut self,
        path: &Path,
        expected: Option<&BTreeMap<String, ()>>,
        total_metadata_bytes: &mut u64,
    ) -> Result<()> {
        let mut file = File::open(path)?;
        let file_metadata = file.metadata()?;
        if !file_metadata.is_file() {
            return Err(invalid("SafeTensors input must be a regular file"));
        }
        let file_len = file_metadata.len();
        let mut length = [0u8; 8];
        file.read_exact(&mut length)?;
        let header_len = u64::from_le_bytes(length);
        let data_start = header_len
            .checked_add(8)
            .ok_or_else(|| invalid("SafeTensors header length overflow"))?;
        if header_len == 0 || header_len > MAX_JSON_BYTES || data_start > file_len {
            return Err(invalid(
                "SafeTensors header is empty, oversized, or truncated",
            ));
        }
        *total_metadata_bytes = total_metadata_bytes
            .checked_add(header_len)
            .filter(|total| *total <= MAX_TOTAL_JSON_BYTES)
            .ok_or_else(|| invalid("checkpoint exceeds the metadata byte limit"))?;
        let mut bytes = allocate_bytes(header_len)?;
        file.read_exact(&mut bytes)?;
        if bytes.first() != Some(&b'{') {
            return Err(invalid("SafeTensors header must start with an object"));
        }
        let Header(tensors) = serde_json::from_slice::<Header>(&bytes)
            .map_err(|error| invalid(format!("invalid SafeTensors header: {error}")))?;
        drop(bytes);
        if tensors.is_empty() {
            return Err(invalid("SafeTensors file contains no tensors"));
        }
        if let Some(expected) = expected {
            if !tensors.keys().eq(expected.keys()) {
                return Err(invalid(format!(
                    "shard {} has missing or unindexed tensors",
                    path.display()
                )));
            }
        }
        let mut spans = Vec::with_capacity(tensors.len());
        for (name, info) in tensors {
            if info.shape.is_empty() || info.shape.len() > MAX_RANK || info.shape.contains(&0) {
                return Err(invalid(format!(
                    "unsupported empty or excessive-rank shape for {name}"
                )));
            }
            let dtype = match info.dtype.as_str() {
                "F32" => SourceDtype::F32,
                "F16" => SourceDtype::F16,
                "BF16" => SourceDtype::BF16,
                other => {
                    return Err(invalid(format!(
                        "unsupported SafeTensors dtype {other} for {name}"
                    )))
                }
            };
            let byte_len = info
                .shape
                .iter()
                .try_fold(dtype.byte_width() as u64, |size, &dimension| {
                    size.checked_mul(dimension as u64)
                })
                .ok_or_else(|| invalid(format!("shape byte size overflows for {name}")))?;
            let [start, end] = info.data_offsets;
            if end.checked_sub(start) != Some(byte_len) {
                return Err(invalid(format!(
                    "tensor byte span does not match shape and dtype for {name}"
                )));
            }
            spans.push((start, end, name, info.shape, dtype));
        }
        spans.sort_unstable_by_key(|span| span.0);
        let mut next_offset = 0;
        for (start, end, _, _, _) in &spans {
            if *start != next_offset {
                return Err(invalid(
                    "SafeTensors payload contains overlapping or gapped spans",
                ));
            }
            next_offset = *end;
        }
        if data_start.checked_add(next_offset) != Some(file_len) {
            return Err(invalid(
                "SafeTensors payload is truncated or has trailing bytes",
            ));
        }
        for (start, end, name, shape, dtype) in spans {
            if self.tensors.contains_key(&name) {
                return Err(invalid(format!("duplicate checkpoint tensor {name}")));
            }
            if self.tensors.len() >= MAX_TENSORS {
                return Err(invalid("checkpoint exceeds the tensor count limit"));
            }
            let byte_len = end - start;
            let offset = data_start
                .checked_add(start)
                .ok_or_else(|| invalid("tensor file offset overflow"))?;
            file.seek(SeekFrom::Start(offset))?;
            let sha256 = hash_span(&mut file, byte_len)?;
            self.tensors.insert(
                name,
                SourceTensor {
                    shape,
                    dtype,
                    file: path.to_path_buf(),
                    offset,
                    byte_len,
                    sha256,
                },
            );
        }
        if file.metadata()?.len() != file_len {
            return Err(invalid("source file changed length during preflight"));
        }
        self.files.insert(
            path.to_path_buf(),
            Arc::new(SourceFile {
                handle: Mutex::new(file),
                byte_len: file_len,
            }),
        );
        Ok(())
    }
}

fn hash_span(file: &mut File, mut remaining: u64) -> Result<[u8; 32]> {
    let mut digest = Sha256::new();
    let mut bytes = [0u8; HASH_CHUNK_BYTES];
    while remaining != 0 {
        let count = remaining.min(bytes.len() as u64) as usize;
        file.read_exact(&mut bytes[..count])?;
        digest.update(&bytes[..count]);
        remaining -= count as u64;
    }
    Ok(digest.finalize().into())
}

fn contained_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    if relative.is_empty()
        || relative.contains('\\')
        || relative.contains(':')
        || relative_path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(invalid(
            "index shard paths must be relative without traversal",
        ));
    }
    let path = fs::canonicalize(root.join(relative_path))?;
    if !path.starts_with(root) {
        return Err(invalid("index shard path escapes the checkpoint directory"));
    }
    Ok(path)
}

fn require_safetensors(path: &Path) -> Result<()> {
    if path.extension().and_then(|value| value.to_str()) != Some("safetensors") {
        return Err(invalid(
            "input must be a SafeTensors file or checkpoint directory",
        ));
    }
    Ok(())
}

fn read_json_file(path: &Path) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_JSON_BYTES {
        return Err(invalid(
            "checkpoint index is not a regular bounded JSON file",
        ));
    }
    let mut bytes = allocate_bytes(metadata.len())?;
    file.read_exact(&mut bytes)?;
    if file.metadata()?.len() != metadata.len() {
        return Err(invalid("checkpoint index changed length while being read"));
    }
    Ok(bytes)
}

fn allocate_bytes(length: u64) -> Result<Vec<u8>> {
    let length =
        usize::try_from(length).map_err(|_| invalid("metadata size exceeds address space"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| RuvLLMError::OutOfMemory("cannot allocate checkpoint metadata".into()))?;
    bytes.resize(length, 0);
    Ok(bytes)
}

fn invalid(message: impl Into<String>) -> RuvLLMError {
    RuvLLMError::Quantization(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn fixture(directory: &Path, name: &str, header: &str, payload: &[u8]) -> PathBuf {
        let path = directory.join(name);
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(payload).unwrap();
        path
    }

    fn index(directory: &Path, contents: &str) {
        fs::write(directory.join("model.safetensors.index.json"), contents).unwrap();
    }

    #[test]
    fn single_file_preserves_supported_dtype_spans() {
        let directory = TempDir::new().unwrap();
        let header = r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F16","shape":[2],"data_offsets":[8,12]},"c":{"dtype":"BF16","shape":[2],"data_offsets":[12,16]}}"#;
        let payload: Vec<u8> = (0..16).collect();
        let path = fixture(directory.path(), "model.safetensors", header, &payload);
        let checkpoint = open(&path).unwrap();
        assert_eq!(checkpoint.root, fs::canonicalize(directory.path()).unwrap());
        assert_eq!(checkpoint.tensors.len(), 3);
        for (name, dtype, start, end) in [
            ("a", SourceDtype::F32, 0, 8),
            ("b", SourceDtype::F16, 8, 12),
            ("c", SourceDtype::BF16, 12, 16),
        ] {
            let tensor = &checkpoint.tensors[name];
            assert_eq!(tensor.dtype, dtype);
            assert_eq!(tensor.shape, [2]);
            assert_eq!(tensor.offset, 8 + header.len() as u64 + start as u64);
            assert_eq!(tensor.byte_len, (end - start) as u64);
            let mut reader = checkpoint.tensor_reader(tensor).unwrap();
            let mut data = Vec::new();
            reader.read_to_end(&mut data).unwrap();
            reader.finish().unwrap();
            assert_eq!(data, payload[start..end]);
        }
        assert_eq!(open(directory.path()).unwrap().tensors.len(), 3);
    }

    #[test]
    fn two_shard_index_has_exact_tensor_membership_and_size() {
        let directory = TempDir::new().unwrap();
        fixture(
            directory.path(),
            "one.safetensors",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &1f32.to_le_bytes(),
        );
        fixture(
            directory.path(),
            "two.safetensors",
            r#"{"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &2f32.to_le_bytes(),
        );
        index(
            directory.path(),
            r#"{"metadata":{"total_size":8},"weight_map":{"a":"one.safetensors","b":"two.safetensors"}}"#,
        );
        let checkpoint = open(directory.path()).unwrap();
        assert_eq!(
            checkpoint
                .tensors
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        for (name, expected) in [("a", 1f32), ("b", 2f32)] {
            let mut reader = checkpoint.tensor_reader(&checkpoint.tensors[name]).unwrap();
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes).unwrap();
            assert_eq!(f32::from_le_bytes(bytes), expected);
            reader.finish().unwrap();
        }
    }

    #[test]
    fn malformed_headers_shapes_and_payload_spans_fail_closed() {
        let directory = TempDir::new().unwrap();
        let cases = [
            (
                "duplicate tensor",
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
                8,
            ),
            (
                "duplicate metadata",
                r#"{"__metadata__":{},"__metadata__":{},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                4,
            ),
            (
                "duplicate metadata entry",
                r#"{"__metadata__":{"format":"pt","format":"pt"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                4,
            ),
            (
                "duplicate field",
                r#"{"x":{"dtype":"F32","dtype":"F16","shape":[1],"data_offsets":[0,4]}}"#,
                4,
            ),
            (
                "unknown field",
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":true}}"#,
                4,
            ),
            (
                "unsupported dtype",
                r#"{"x":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}"#,
                4,
            ),
            (
                "scalar",
                r#"{"x":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
                4,
            ),
            (
                "zero dimension",
                r#"{"x":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}}"#,
                0,
            ),
            (
                "rank limit",
                r#"{"x":{"dtype":"F32","shape":[1,1,1,1,1],"data_offsets":[0,4]}}"#,
                4,
            ),
            (
                "overflow",
                r#"{"x":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,4]}}"#,
                4,
            ),
            (
                "mismatched span",
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}}"#,
                3,
            ),
            (
                "reversed span",
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
                4,
            ),
            (
                "overlap",
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                4,
            ),
            (
                "gap",
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
                8,
            ),
            (
                "trailing payload",
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                8,
            ),
            (
                "truncated payload",
                r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
                4,
            ),
            ("empty checkpoint", "{}", 0),
            ("invalid json", "{", 0),
        ];
        for (control, header, payload_len) in cases {
            let path = fixture(
                directory.path(),
                "invalid.safetensors",
                header,
                &vec![0; payload_len],
            );
            assert!(
                open(&path).is_err(),
                "control unexpectedly accepted: {control}"
            );
        }
    }

    #[test]
    fn short_or_oversized_headers_fail_without_payload_allocation() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("invalid.safetensors");
        fs::write(&path, [0u8; 7]).unwrap();
        assert!(open(&path).is_err());
        fs::write(&path, [4u64.to_le_bytes().as_slice(), b"{"].concat()).unwrap();
        assert!(open(&path).is_err());
        let mut file = File::create(&path).unwrap();
        file.write_all(&(MAX_JSON_BYTES + 1).to_le_bytes()).unwrap();
        file.set_len(MAX_JSON_BYTES + 9).unwrap();
        assert!(open(&path).is_err());
    }

    #[test]
    fn malformed_index_membership_and_size_fail_closed() {
        let directory = TempDir::new().unwrap();
        fixture(
            directory.path(),
            "one.safetensors",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        fixture(
            directory.path(),
            "two.safetensors",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        let cases = [
            r#"{"weight_map":{"a":"one.safetensors","a":"two.safetensors"}}"#,
            r#"{"weight_map":{"missing":"one.safetensors"}}"#,
            r#"{"weight_map":{"a":"missing.safetensors"}}"#,
            r#"{"weight_map":{"a":"one.safetensors","b":"two.safetensors"}}"#,
            r#"{"metadata":{"total_size":8},"weight_map":{"a":"one.safetensors"}}"#,
            r#"{"weight_map":{}}"#,
            r#"{"weight_map":{"a":"one.safetensors"},"weight_map":{"a":"one.safetensors"}}"#,
        ];
        for contents in cases {
            index(directory.path(), contents);
            assert!(
                open(directory.path()).is_err(),
                "accepted index: {contents}"
            );
        }
        fixture(
            directory.path(),
            "one.safetensors",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"extra":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
            &[0; 8],
        );
        index(
            directory.path(),
            r#"{"weight_map":{"a":"one.safetensors"}}"#,
        );
        assert!(open(directory.path()).is_err());
    }

    #[test]
    fn ambiguous_directory_and_untrusted_paths_fail_closed() {
        let directory = TempDir::new().unwrap();
        let root = fs::canonicalize(directory.path()).unwrap();
        for relative in [
            "",
            "..",
            "../outside.safetensors",
            "/outside.safetensors",
            "a/../../outside.safetensors",
            "a\\..\\outside.safetensors",
            "C:outside.safetensors",
        ] {
            assert!(contained_path(&root, relative).is_err());
        }
        fixture(
            directory.path(),
            "model.safetensors",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        index(
            directory.path(),
            r#"{"weight_map":{"a":"model.safetensors"}}"#,
        );
        assert!(open(directory.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn shard_symlink_cannot_escape_checkpoint_root() {
        let directory = TempDir::new().unwrap();
        let root = directory.path().join("checkpoint");
        fs::create_dir(&root).unwrap();
        let outside = fixture(
            directory.path(),
            "outside.safetensors",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        std::os::unix::fs::symlink(outside, root.join("link.safetensors")).unwrap();
        index(&root, r#"{"weight_map":{"a":"link.safetensors"}}"#);
        assert!(open(&root).is_err());
    }

    #[test]
    fn changed_tensor_bytes_and_incomplete_reads_fail_authentication() {
        let directory = TempDir::new().unwrap();
        let path = fixture(
            directory.path(),
            "model.safetensors",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &1f32.to_le_bytes(),
        );
        let checkpoint = open(&path).unwrap();
        let tensor = &checkpoint.tensors["a"];
        assert!(checkpoint.tensor_reader(tensor).unwrap().finish().is_err());
        let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(tensor.offset)).unwrap();
        file.write_all(&2f32.to_le_bytes()).unwrap();
        let mut reader = checkpoint.tensor_reader(tensor).unwrap();
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).unwrap();
        assert!(reader.finish().is_err());
        file.set_len(tensor.offset + 2).unwrap();
        let mut reader = checkpoint.tensor_reader(tensor).unwrap();
        assert!(reader.read_to_end(&mut Vec::new()).is_err());
    }

    #[test]
    fn independent_readers_stay_bounded_and_do_not_share_cursors() {
        let directory = TempDir::new().unwrap();
        let path = fixture(
            directory.path(),
            "model.safetensors",
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
            &[1, 2, 3, 4, 5, 6, 7, 8],
        );
        let checkpoint = open(&path).unwrap();
        let mut first = checkpoint.tensor_reader(&checkpoint.tensors["a"]).unwrap();
        let mut second = checkpoint.tensor_reader(&checkpoint.tensors["b"]).unwrap();
        let mut bytes = [0; 2];
        first.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes, [1, 2]);
        second.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes, [5, 6]);
        first.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes, [3, 4]);
        second.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes, [7, 8]);
        assert_eq!(first.read(&mut bytes).unwrap(), 0);
        first.finish().unwrap();
        second.finish().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn replacing_a_shard_path_does_not_replace_its_opened_source() {
        let directory = TempDir::new().unwrap();
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let path = fixture(
            directory.path(),
            "model.safetensors",
            header,
            &1f32.to_le_bytes(),
        );
        let checkpoint = open(&path).unwrap();
        fs::rename(&path, directory.path().join("original.safetensors")).unwrap();
        fixture(
            directory.path(),
            "model.safetensors",
            header,
            &2f32.to_le_bytes(),
        );
        let mut reader = checkpoint.tensor_reader(&checkpoint.tensors["a"]).unwrap();
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes).unwrap();
        reader.finish().unwrap();
        assert_eq!(f32::from_le_bytes(bytes), 1f32);
    }
}
