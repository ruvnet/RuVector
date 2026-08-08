//! Authoritative `META_SEG` v1 encoding.
//!
//! The decoder is deliberately allocation-bounded and rejects non-canonical
//! ordering. This makes the same bytes suitable for hashing and signing.

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

pub const META_MAGIC: &[u8; 8] = b"RVFMETA\0";
pub const META_COMPLETE_MAGIC: &[u8; 8] = b"RVFDONE\0";
pub const META_COMPLETION_FOOTER_SIZE: usize = 40;
pub const META_VERSION: u16 = 1;
pub const META_FLAG_FULL_SNAPSHOT: u16 = 1;
pub const MAX_META_DELTAS: usize = 64;
pub const MAX_META_DECODED_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_META_FIELDS: usize = 65_536;
pub const MAX_META_RECORDS: usize = 16_777_216;
pub const MAX_META_NAME_BYTES: usize = 4_096;
pub const MAX_META_VALUE_BYTES: usize = 16 * 1024 * 1024;

/// Smallest number of encoded bytes each sequence element can occupy. Counts
/// read from the payload are attacker-controlled, so these bound how many
/// elements the remaining input could actually hold (see `Cursor::capacity_for`).
const MIN_SCHEMA_ENTRY_BYTES: usize = 6;
const MIN_FILE_METADATA_BYTES: usize = 3;
const MIN_RECORD_BYTES: usize = 13;
const MIN_FIELD_BYTES: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MetadataType {
    String = 1,
    Bytes = 2,
    I64 = 3,
    U64 = 4,
    F64 = 5,
    Bool = 6,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    Null,
    String(String),
    Bytes(Vec<u8>),
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
    DeleteField,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataSchemaEntry {
    pub field_id: u16,
    pub value_type: MetadataType,
    pub nullable: bool,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FileMetadataRecord {
    pub key: String,
    pub value: MetadataValue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetadataRecordOp {
    Upsert(Vec<(u16, MetadataValue)>),
    DeleteRecord,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetadataRecord {
    pub vector_id: u64,
    pub operation: MetadataRecordOp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetadataSegment {
    pub generation: u64,
    pub base_generation: Option<u64>,
    pub full_snapshot: bool,
    pub schema: Vec<MetadataSchemaEntry>,
    pub file_metadata: Vec<FileMetadataRecord>,
    pub records: Vec<MetadataRecord>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataDecodeError {
    Truncated,
    InvalidMagic,
    InvalidChecksum,
    InvalidVersion,
    InvalidTag,
    InvalidUtf8,
    InvalidOrdering,
    InvalidSchema,
    NonFiniteFloat,
    ResourceLimit,
    TrailingBytes,
}

#[cfg(feature = "alloc")]
impl MetadataSegment {
    pub fn encode(&self) -> Result<Vec<u8>, MetadataDecodeError> {
        validate_segment(self)?;
        let mut out = Vec::new();
        out.extend_from_slice(META_MAGIC);
        out.extend_from_slice(&META_VERSION.to_le_bytes());
        out.extend_from_slice(
            &(if self.full_snapshot {
                META_FLAG_FULL_SNAPSHOT
            } else {
                0
            })
            .to_le_bytes(),
        );
        out.extend_from_slice(&self.generation.to_le_bytes());
        out.extend_from_slice(&self.base_generation.unwrap_or(u64::MAX).to_le_bytes());
        put_u32(&mut out, self.schema.len())?;
        put_u32(&mut out, self.file_metadata.len())?;
        put_u32(&mut out, self.records.len())?;
        for entry in &self.schema {
            out.extend_from_slice(&entry.field_id.to_le_bytes());
            out.push(entry.value_type as u8);
            out.push(u8::from(entry.nullable));
            put_bytes_u16(&mut out, entry.name.as_bytes())?;
        }
        for entry in &self.file_metadata {
            put_bytes_u16(&mut out, entry.key.as_bytes())?;
            encode_value(&mut out, &entry.value)?;
        }
        for record in &self.records {
            out.extend_from_slice(&record.vector_id.to_le_bytes());
            match &record.operation {
                MetadataRecordOp::Upsert(fields) => {
                    out.push(0);
                    put_u32(&mut out, fields.len())?;
                    for (field_id, value) in fields {
                        out.extend_from_slice(&field_id.to_le_bytes());
                        encode_value(&mut out, value)?;
                    }
                }
                MetadataRecordOp::DeleteRecord => {
                    out.push(1);
                    out.extend_from_slice(&0u32.to_le_bytes());
                }
            }
        }
        if out.len().saturating_add(META_COMPLETION_FOOTER_SIZE) > MAX_META_DECODED_BYTES {
            return Err(MetadataDecodeError::ResourceLimit);
        }
        let digest = crate::sha256::sha256(&out);
        out.extend_from_slice(META_COMPLETE_MAGIC);
        out.extend_from_slice(&digest);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, MetadataDecodeError> {
        if bytes.len() > MAX_META_DECODED_BYTES {
            return Err(MetadataDecodeError::ResourceLimit);
        }
        if bytes.len() < META_COMPLETION_FOOTER_SIZE {
            return Err(MetadataDecodeError::Truncated);
        }
        let body_len = bytes.len() - META_COMPLETION_FOOTER_SIZE;
        if &bytes[body_len..body_len + 8] != META_COMPLETE_MAGIC
            || bytes[body_len + 8..] != crate::sha256::sha256(&bytes[..body_len])
        {
            return Err(MetadataDecodeError::InvalidChecksum);
        }
        let mut cursor = Cursor::new(&bytes[..body_len]);
        if cursor.take(8)? != META_MAGIC {
            return Err(MetadataDecodeError::InvalidMagic);
        }
        if cursor.u16()? != META_VERSION {
            return Err(MetadataDecodeError::InvalidVersion);
        }
        let flags = cursor.u16()?;
        let generation = cursor.u64()?;
        let raw_base = cursor.u64()?;
        let schema_count = cursor.count(MAX_META_FIELDS)?;
        let file_count = cursor.count(MAX_META_FIELDS)?;
        let record_count = cursor.count(MAX_META_RECORDS)?;
        let mut schema =
            Vec::with_capacity(cursor.capacity_for(schema_count, MIN_SCHEMA_ENTRY_BYTES));
        for _ in 0..schema_count {
            let field_id = cursor.u16()?;
            let value_type = decode_type(cursor.u8()?)?;
            let nullable = match cursor.u8()? {
                0 => false,
                1 => true,
                _ => return Err(MetadataDecodeError::InvalidSchema),
            };
            let name = cursor.string_u16(MAX_META_NAME_BYTES)?;
            schema.push(MetadataSchemaEntry {
                field_id,
                value_type,
                nullable,
                name,
            });
        }
        let mut file_metadata =
            Vec::with_capacity(cursor.capacity_for(file_count, MIN_FILE_METADATA_BYTES));
        for _ in 0..file_count {
            let key = cursor.string_u16(MAX_META_NAME_BYTES)?;
            let value = decode_value(&mut cursor)?;
            file_metadata.push(FileMetadataRecord { key, value });
        }
        let mut records = Vec::with_capacity(cursor.capacity_for(record_count, MIN_RECORD_BYTES));
        for _ in 0..record_count {
            let vector_id = cursor.u64()?;
            let op = cursor.u8()?;
            let field_count = cursor.count(MAX_META_FIELDS)?;
            let operation = match op {
                0 => {
                    let mut fields =
                        Vec::with_capacity(cursor.capacity_for(field_count, MIN_FIELD_BYTES));
                    for _ in 0..field_count {
                        fields.push((cursor.u16()?, decode_value(&mut cursor)?));
                    }
                    MetadataRecordOp::Upsert(fields)
                }
                1 if field_count == 0 => MetadataRecordOp::DeleteRecord,
                _ => return Err(MetadataDecodeError::InvalidTag),
            };
            records.push(MetadataRecord {
                vector_id,
                operation,
            });
        }
        if cursor.remaining() != 0 {
            return Err(MetadataDecodeError::TrailingBytes);
        }
        let segment = Self {
            generation,
            base_generation: (raw_base != u64::MAX).then_some(raw_base),
            full_snapshot: flags & META_FLAG_FULL_SNAPSHOT != 0,
            schema,
            file_metadata,
            records,
        };
        validate_segment(&segment)?;
        Ok(segment)
    }
}

#[cfg(feature = "alloc")]
fn validate_segment(segment: &MetadataSegment) -> Result<(), MetadataDecodeError> {
    if segment.full_snapshot != segment.base_generation.is_none() {
        return Err(MetadataDecodeError::InvalidSchema);
    }
    if let Some(base) = segment.base_generation {
        if segment.generation
            != base
                .checked_add(1)
                .ok_or(MetadataDecodeError::InvalidSchema)?
        {
            return Err(MetadataDecodeError::InvalidSchema);
        }
    }
    let mut previous = None;
    for entry in &segment.schema {
        if previous.is_some_and(|id| id >= entry.field_id)
            || entry.name.is_empty()
            || entry.name.len() > MAX_META_NAME_BYTES
        {
            return Err(MetadataDecodeError::InvalidOrdering);
        }
        previous = Some(entry.field_id);
    }
    for pair in segment.file_metadata.windows(2) {
        if pair[0].key >= pair[1].key {
            return Err(MetadataDecodeError::InvalidOrdering);
        }
    }
    for pair in segment.records.windows(2) {
        if pair[0].vector_id >= pair[1].vector_id {
            return Err(MetadataDecodeError::InvalidOrdering);
        }
    }
    for record in &segment.records {
        if let MetadataRecordOp::Upsert(fields) = &record.operation {
            for pair in fields.windows(2) {
                if pair[0].0 >= pair[1].0 {
                    return Err(MetadataDecodeError::InvalidOrdering);
                }
            }
            for (field_id, value) in fields {
                let schema = segment
                    .schema
                    .iter()
                    .find(|entry| entry.field_id == *field_id)
                    .ok_or(MetadataDecodeError::InvalidSchema)?;
                validate_value(value, schema)?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "alloc")]
fn validate_value(
    value: &MetadataValue,
    schema: &MetadataSchemaEntry,
) -> Result<(), MetadataDecodeError> {
    let valid = match value {
        MetadataValue::Null => schema.nullable,
        MetadataValue::String(v) => {
            schema.value_type == MetadataType::String && v.len() <= MAX_META_VALUE_BYTES
        }
        MetadataValue::Bytes(v) => {
            schema.value_type == MetadataType::Bytes && v.len() <= MAX_META_VALUE_BYTES
        }
        MetadataValue::I64(_) => schema.value_type == MetadataType::I64,
        MetadataValue::U64(_) => schema.value_type == MetadataType::U64,
        MetadataValue::F64(v) => schema.value_type == MetadataType::F64 && v.is_finite(),
        MetadataValue::Bool(_) => schema.value_type == MetadataType::Bool,
        MetadataValue::DeleteField => true,
    };
    valid
        .then_some(())
        .ok_or(MetadataDecodeError::InvalidSchema)
}

#[cfg(feature = "alloc")]
fn encode_value(out: &mut Vec<u8>, value: &MetadataValue) -> Result<(), MetadataDecodeError> {
    match value {
        MetadataValue::Null => out.push(0),
        MetadataValue::String(v) => {
            out.push(1);
            put_bytes_u32(out, v.as_bytes())?;
        }
        MetadataValue::Bytes(v) => {
            out.push(2);
            put_bytes_u32(out, v)?;
        }
        MetadataValue::I64(v) => {
            out.push(3);
            out.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::U64(v) => {
            out.push(4);
            out.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::F64(v) if v.is_finite() => {
            out.push(5);
            out.extend_from_slice(&v.to_le_bytes());
        }
        MetadataValue::F64(_) => return Err(MetadataDecodeError::NonFiniteFloat),
        MetadataValue::Bool(v) => {
            out.push(6);
            out.push(u8::from(*v));
        }
        MetadataValue::DeleteField => out.push(7),
    }
    Ok(())
}

#[cfg(feature = "alloc")]
fn decode_value(cursor: &mut Cursor<'_>) -> Result<MetadataValue, MetadataDecodeError> {
    Ok(match cursor.u8()? {
        0 => MetadataValue::Null,
        1 => MetadataValue::String(cursor.string_u32(MAX_META_VALUE_BYTES)?),
        2 => MetadataValue::Bytes(cursor.bytes_u32(MAX_META_VALUE_BYTES)?.to_vec()),
        3 => MetadataValue::I64(i64::from_le_bytes(cursor.array()?)),
        4 => MetadataValue::U64(cursor.u64()?),
        5 => {
            let value = f64::from_le_bytes(cursor.array()?);
            if !value.is_finite() {
                return Err(MetadataDecodeError::NonFiniteFloat);
            }
            MetadataValue::F64(value)
        }
        6 => MetadataValue::Bool(match cursor.u8()? {
            0 => false,
            1 => true,
            _ => return Err(MetadataDecodeError::InvalidTag),
        }),
        7 => MetadataValue::DeleteField,
        _ => return Err(MetadataDecodeError::InvalidTag),
    })
}

fn decode_type(value: u8) -> Result<MetadataType, MetadataDecodeError> {
    match value {
        1 => Ok(MetadataType::String),
        2 => Ok(MetadataType::Bytes),
        3 => Ok(MetadataType::I64),
        4 => Ok(MetadataType::U64),
        5 => Ok(MetadataType::F64),
        6 => Ok(MetadataType::Bool),
        _ => Err(MetadataDecodeError::InvalidTag),
    }
}

#[cfg(feature = "alloc")]
fn put_u32(out: &mut Vec<u8>, len: usize) -> Result<(), MetadataDecodeError> {
    out.extend_from_slice(
        &u32::try_from(len)
            .map_err(|_| MetadataDecodeError::ResourceLimit)?
            .to_le_bytes(),
    );
    Ok(())
}
#[cfg(feature = "alloc")]
fn put_bytes_u16(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), MetadataDecodeError> {
    out.extend_from_slice(
        &u16::try_from(bytes.len())
            .map_err(|_| MetadataDecodeError::ResourceLimit)?
            .to_le_bytes(),
    );
    out.extend_from_slice(bytes);
    Ok(())
}
#[cfg(feature = "alloc")]
fn put_bytes_u32(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), MetadataDecodeError> {
    put_u32(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}

#[cfg(feature = "alloc")]
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}
#[cfg(feature = "alloc")]
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
    fn take(&mut self, len: usize) -> Result<&'a [u8], MetadataDecodeError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(MetadataDecodeError::ResourceLimit)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(MetadataDecodeError::Truncated)?;
        self.offset = end;
        Ok(value)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], MetadataDecodeError> {
        self.take(N)?
            .try_into()
            .map_err(|_| MetadataDecodeError::Truncated)
    }
    fn u8(&mut self) -> Result<u8, MetadataDecodeError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, MetadataDecodeError> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    fn u32(&mut self) -> Result<u32, MetadataDecodeError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64, MetadataDecodeError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn count(&mut self, max: usize) -> Result<usize, MetadataDecodeError> {
        let count = self.u32()? as usize;
        if count > max {
            return Err(MetadataDecodeError::ResourceLimit);
        }
        Ok(count)
    }
    /// Preallocation size for a sequence whose length came from the payload.
    ///
    /// A declared count is attacker-controlled and capped only by a format
    /// ceiling, so reserving it directly lets a tiny blob drive a huge
    /// allocation. Each element consumes at least `min_element_bytes`, so the
    /// unread remainder is a hard upper bound on how many can be present
    /// (ADR-280 §9). An under-reservation only costs the decoder a regrowth.
    fn capacity_for(&self, count: usize, min_element_bytes: usize) -> usize {
        count.min(self.remaining() / min_element_bytes)
    }
    fn bytes_u32(&mut self, max: usize) -> Result<&'a [u8], MetadataDecodeError> {
        let len = self.u32()? as usize;
        if len > max {
            return Err(MetadataDecodeError::ResourceLimit);
        }
        self.take(len)
    }
    fn string_u16(&mut self, max: usize) -> Result<String, MetadataDecodeError> {
        let len = self.u16()? as usize;
        if len > max {
            return Err(MetadataDecodeError::ResourceLimit);
        }
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| MetadataDecodeError::InvalidUtf8)
    }
    fn string_u32(&mut self, max: usize) -> Result<String, MetadataDecodeError> {
        String::from_utf8(self.bytes_u32(max)?.to_vec())
            .map_err(|_| MetadataDecodeError::InvalidUtf8)
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use alloc::vec;
    #[test]
    fn round_trip_distinct_states_and_file_metadata() {
        let segment = MetadataSegment {
            generation: 7,
            base_generation: None,
            full_snapshot: true,
            schema: vec![
                MetadataSchemaEntry {
                    field_id: 1,
                    value_type: MetadataType::String,
                    nullable: true,
                    name: "title".into(),
                },
                MetadataSchemaEntry {
                    field_id: 2,
                    value_type: MetadataType::Bool,
                    nullable: false,
                    name: "live".into(),
                },
            ],
            file_metadata: vec![FileMetadataRecord {
                key: "rvf.embedding.space_id".into(),
                value: MetadataValue::String("abc".into()),
            }],
            records: vec![
                MetadataRecord {
                    vector_id: 10,
                    operation: MetadataRecordOp::Upsert(vec![
                        (1, MetadataValue::Null),
                        (2, MetadataValue::Bool(true)),
                    ]),
                },
                MetadataRecord {
                    vector_id: 11,
                    operation: MetadataRecordOp::DeleteRecord,
                },
            ],
        };
        assert_eq!(
            MetadataSegment::decode(&segment.encode().unwrap()).unwrap(),
            segment
        );
    }
    /// A header may claim counts near the format ceilings over a payload far
    /// too small to hold them. Decoding must fail cleanly on the truncated
    /// body without reserving capacity for the claimed counts.
    #[test]
    fn rejects_oversized_counts_without_preallocating() {
        fn crafted(schema_count: u32, file_count: u32, record_count: u32) -> Vec<u8> {
            let mut body = Vec::new();
            body.extend_from_slice(META_MAGIC);
            body.extend_from_slice(&META_VERSION.to_le_bytes());
            body.extend_from_slice(&META_FLAG_FULL_SNAPSHOT.to_le_bytes());
            body.extend_from_slice(&1u64.to_le_bytes());
            body.extend_from_slice(&u64::MAX.to_le_bytes());
            body.extend_from_slice(&schema_count.to_le_bytes());
            body.extend_from_slice(&file_count.to_le_bytes());
            body.extend_from_slice(&record_count.to_le_bytes());
            let digest = crate::sha256::sha256(&body);
            body.extend_from_slice(META_COMPLETE_MAGIC);
            body.extend_from_slice(&digest);
            body
        }

        // An 80-byte blob claiming the maximum record count. Reserving
        // `MAX_META_RECORDS` records would allocate hundreds of MiB; the
        // remaining zero bytes can hold none of them.
        let blob = crafted(0, 0, MAX_META_RECORDS as u32);
        assert_eq!(blob.len(), 80);
        assert_eq!(
            MetadataSegment::decode(&blob),
            Err(MetadataDecodeError::Truncated)
        );
        assert_eq!(
            MetadataSegment::decode(&crafted(MAX_META_FIELDS as u32, 0, 0)),
            Err(MetadataDecodeError::Truncated)
        );
        assert_eq!(
            MetadataSegment::decode(&crafted(0, MAX_META_FIELDS as u32, 0)),
            Err(MetadataDecodeError::Truncated)
        );
        // A count past the ceiling is still a resource-limit rejection.
        assert_eq!(
            MetadataSegment::decode(&crafted(0, 0, MAX_META_RECORDS as u32 + 1)),
            Err(MetadataDecodeError::ResourceLimit)
        );
    }

    #[test]
    fn rejects_non_monotonic_delta_and_non_finite_float() {
        let mut segment = MetadataSegment {
            generation: 3,
            base_generation: Some(1),
            full_snapshot: false,
            schema: vec![],
            file_metadata: vec![],
            records: vec![],
        };
        assert_eq!(segment.encode(), Err(MetadataDecodeError::InvalidSchema));
        segment.generation = 2;
        segment.schema.push(MetadataSchemaEntry {
            field_id: 1,
            value_type: MetadataType::F64,
            nullable: false,
            name: "score".into(),
        });
        segment.records.push(MetadataRecord {
            vector_id: 1,
            operation: MetadataRecordOp::Upsert(vec![(1, MetadataValue::F64(f64::NAN))]),
        });
        assert!(segment.encode().is_err());
    }
}
