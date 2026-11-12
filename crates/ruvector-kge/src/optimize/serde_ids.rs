//! Serialize KGE proposal/knobs ids as JSON **strings** while keeping the
//! internal `u64` (ADR-004 receipts). The ids are content hashes spanning the
//! full `u64` range; a JSON number above `2^53` loses precision in a JS reader,
//! so any consumer matching a `championId` against a receipt would mis-compare.
//! Emitting them as strings keeps the match exact across the boundary. The
//! round-trip is lossless: `deserialize` parses the string back to the same
//! `u64`.

/// A required `u64` id as a JSON string.
pub(crate) mod id_str {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &u64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&v.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// An optional `u64` id as a JSON string, or `null` when absent.
pub(crate) mod opt_id_str {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<u64>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(x) => s.serialize_some(&x.to_string()),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
        match Option::<String>::deserialize(d)? {
            Some(s) => s.parse().map(Some).map_err(serde::de::Error::custom),
            None => Ok(None),
        }
    }
}
