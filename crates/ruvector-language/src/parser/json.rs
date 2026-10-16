//! Streaming JSON validation before conversion to a typed wire contract.
use super::{MAX_JSON_BYTES, MAX_JSON_DEPTH};
use crate::error::{Error, Result};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use std::fmt;

const MAX_VALUES: usize = 1_000_000;
const MAX_COLLECTION: usize = 100_000;

/// A shared input boundary for programs, snapshots, contexts, and model files.
/// Every object, including nested property maps, rejects duplicate JSON keys.
pub fn strict_json(source: &str) -> Result<Value> {
    if source.len() > MAX_JSON_BYTES {
        return Err(Error::new(
            "source_limit",
            "JSON exceeds 16 MiB source limit",
        ));
    }
    let mut remaining = MAX_VALUES;
    let mut deserializer = serde_json::Deserializer::from_str(source);
    let value = Seed {
        depth: 0,
        remaining: &mut remaining,
    }
    .deserialize(&mut deserializer)
    .map_err(|e| Error::new("json_parse", e.to_string()))?;
    deserializer
        .end()
        .map_err(|e| Error::new("json_parse", e.to_string()))?;
    Ok(value)
}

struct Seed<'a> {
    depth: usize,
    remaining: &'a mut usize,
}
impl<'de> DeserializeSeed<'de> for Seed<'_> {
    type Value = Value;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Value, D::Error> {
        if self.depth > MAX_JSON_DEPTH {
            return Err(de::Error::custom("JSON nesting exceeds 64 levels"));
        }
        if *self.remaining == 0 {
            return Err(de::Error::custom("JSON value limit exceeded"));
        }
        *self.remaining -= 1;
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Seed<'_> {
    type Value = Value;
    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("bounded JSON without duplicate keys")
    }
    fn visit_bool<E: de::Error>(self, value: bool) -> std::result::Result<Value, E> {
        Ok(Value::Bool(value))
    }
    fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<Value, E> {
        Ok(Value::Number(value.into()))
    }
    fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<Value, E> {
        Ok(Value::Number(value.into()))
    }
    fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| de::Error::custom("nonfinite number"))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Value, E> {
        Ok(Value::String(value.to_owned()))
    }
    fn visit_string<E: de::Error>(self, value: String) -> std::result::Result<Value, E> {
        Ok(Value::String(value))
    }
    fn visit_unit<E: de::Error>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_none<E: de::Error>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> std::result::Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = access.next_element_seed(Seed {
            depth: self.depth + 1,
            remaining: self.remaining,
        })? {
            if values.len() == MAX_COLLECTION {
                return Err(de::Error::custom("JSON array exceeds 100000 entries"));
            }
            values.push(value);
        }
        Ok(Value::Array(values))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> std::result::Result<Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = access.next_key::<String>()? {
            if values.len() == MAX_COLLECTION {
                return Err(de::Error::custom("JSON object exceeds 100000 entries"));
            }
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key {key}")));
            }
            let value = access.next_value_seed(Seed {
                depth: self.depth + 1,
                remaining: self.remaining,
            })?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}
