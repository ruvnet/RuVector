//! Ports. The index is a small ordered key-value store: the synchronous KV
//! API of a SQLite-backed Durable Object (one per scope,
//! [`crate::keys::registry_do_name`]) satisfies it directly, and so does an
//! R2 listing for read paths. R2 object bytes never pass through this crate:
//! the service returns keys and the Worker streams.

pub use ruvector_edge_auth::Clock;
pub use ruvector_edge_store::StoreError;
pub use ruvector_edge_tenancy::EntropySource;

/// Ordered KV port. Methods take `&self` (the DO handle is a shared JS
/// object); implementations use interior mutability.
pub trait KvStore {
    /// Read one value.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError>;
    /// Write one value.
    fn put(&self, key: &str, value: &[u8]) -> Result<(), StoreError>;
    /// Delete one value (absent is not an error).
    fn delete(&self, key: &str) -> Result<(), StoreError>;
    /// Up to `limit` entries whose key starts with `prefix` and sorts
    /// strictly after `start_after`, ascending by key.
    fn list(
        &self,
        prefix: &str,
        start_after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, Vec<u8>)>, StoreError>;
}

impl<T: KvStore + ?Sized> KvStore for &T {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
        (**self).get(key)
    }
    fn put(&self, key: &str, value: &[u8]) -> Result<(), StoreError> {
        (**self).put(key, value)
    }
    fn delete(&self, key: &str) -> Result<(), StoreError> {
        (**self).delete(key)
    }
    fn list(
        &self,
        prefix: &str,
        start_after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, Vec<u8>)>, StoreError> {
        (**self).list(prefix, start_after, limit)
    }
}
