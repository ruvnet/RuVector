//! M3 platform ports and their Workers implementations: R2 (`DATA`, bucket
//! `ruvector-edge-data`), the two Queues producers (`INGEST_QUEUE` →
//! `ruvector-edge-ingest`, `AUDIT_QUEUE` → `ruvector-edge-audit`) and
//! Workers AI (`AI`, `@cf/baai/bge-small-en-v1.5`). Native tests implement
//! the same traits in memory (`m3_mem`).
//!
//! Every R2 key is built by the caller from server-derived components
//! (tenant key, uids, minted ids); these ports never see request paths.

use ruvector_edge_snapshot::{EmbedError, EmbedOptions, EmbeddingPort, ObjectFacts};
use ruvector_edge_store::{ErrorCode, OpError};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::future::Future;

/// R2 bucket binding.
pub const DATA_BINDING: &str = "DATA";
/// Ingest queue producer binding.
pub const INGEST_BINDING: &str = "INGEST_QUEUE";
/// Audit queue producer binding.
pub const AUDIT_BINDING: &str = "AUDIT_QUEUE";
/// Workers AI binding.
pub const AI_BINDING: &str = "AI";
/// Ingest queue name (consumer dispatch).
pub const INGEST_QUEUE: &str = "ruvector-edge-ingest";
/// Audit queue name (consumer dispatch).
pub const AUDIT_QUEUE: &str = "ruvector-edge-audit";

/// Object storage failure (retryable).
pub fn storage_err() -> OpError {
    OpError::new(ErrorCode::ShardUnavailable, "object storage unavailable")
}

/// R2 port.
#[allow(async_fn_in_trait)] // Workers futures are !Send.
pub trait Blob {
    /// Store `bytes` at `key` (with its sha256 when known, so R2 keeps it).
    async fn put(&self, key: &str, bytes: Vec<u8>, sha256: Option<[u8; 32]>)
        -> Result<(), OpError>;
    /// Whole object, `None` if absent.
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, OpError>;
    /// `len` bytes at `offset`, `None` if absent.
    async fn get_range(&self, key: &str, offset: u64, len: u64)
        -> Result<Option<Vec<u8>>, OpError>;
    /// Size and stored sha256, `None` if absent.
    async fn head(&self, key: &str) -> Result<Option<ObjectFacts>, OpError>;
    /// sha256 and size of the whole object, streamed (never buffered
    /// whole), `None` if absent.
    async fn sha256_stream(&self, key: &str) -> Result<Option<([u8; 32], u64)>, OpError>;
    /// Remove an object (absent is fine).
    async fn delete(&self, key: &str) -> Result<(), OpError>;
    /// Start a multipart upload; returns its R2 upload id.
    async fn mp_begin(&self, key: &str) -> Result<String, OpError>;
    /// Upload part `n` (1-based); returns its etag.
    async fn mp_part(
        &self,
        key: &str,
        upload: &str,
        n: u16,
        bytes: Vec<u8>,
    ) -> Result<String, OpError>;
    /// Complete with `(n, etag)` parts; returns the object size.
    async fn mp_complete(
        &self,
        key: &str,
        upload: &str,
        parts: &[(u16, String)],
    ) -> Result<u64, OpError>;
    /// Abort a multipart upload (best effort).
    async fn mp_abort(&self, key: &str, upload: &str) -> Result<(), OpError>;
}

/// Producer queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueueName {
    /// `ruvector-edge-ingest`.
    Ingest,
    /// `ruvector-edge-audit`.
    Audit,
}

/// Queues producer port.
#[allow(async_fn_in_trait)]
pub trait Queues {
    /// Send one JSON message.
    async fn send(&self, q: QueueName, body: Json) -> Result<(), OpError>;
}

/// R2 over a Workers bucket binding.
pub struct R2(pub worker::Bucket);

impl Blob for R2 {
    async fn put(&self, key: &str, bytes: Vec<u8>, sha: Option<[u8; 32]>) -> Result<(), OpError> {
        let mut b = self.0.put(key, bytes);
        if let Some(s) = sha {
            b = b.sha256(s.to_vec());
        }
        b.execute().await.map(|_| ()).map_err(|_| storage_err())
    }

    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, OpError> {
        let Some(obj) = self.0.get(key).execute().await.map_err(|_| storage_err())? else {
            return Ok(None);
        };
        match obj.body() {
            Some(b) => b.bytes().await.map(Some).map_err(|_| storage_err()),
            None => Ok(Some(Vec::new())),
        }
    }

    async fn get_range(
        &self,
        key: &str,
        offset: u64,
        len: u64,
    ) -> Result<Option<Vec<u8>>, OpError> {
        let range = worker::Range::OffsetWithLength {
            offset,
            length: len,
        };
        let got = self.0.get(key).range(range).execute().await;
        let Some(obj) = got.map_err(|_| storage_err())? else {
            return Ok(None);
        };
        match obj.body() {
            Some(b) => b.bytes().await.map(Some).map_err(|_| storage_err()),
            None => Ok(Some(Vec::new())),
        }
    }

    async fn head(&self, key: &str) -> Result<Option<ObjectFacts>, OpError> {
        let Some(obj) = self.0.head(key).await.map_err(|_| storage_err())? else {
            return Ok(None);
        };
        let sha256 = obj
            .checksum()
            .sha256
            .and_then(|v| <[u8; 32]>::try_from(v.as_slice()).ok());
        Ok(Some(ObjectFacts {
            size: obj.size(),
            sha256,
        }))
    }

    async fn sha256_stream(&self, key: &str) -> Result<Option<([u8; 32], u64)>, OpError> {
        use futures_util::StreamExt;
        use sha2::{Digest, Sha256};
        let Some(obj) = self.0.get(key).execute().await.map_err(|_| storage_err())? else {
            return Ok(None);
        };
        let mut h = Sha256::new();
        let mut n = 0u64;
        if let Some(body) = obj.body() {
            let mut chunks = body.stream().map_err(|_| storage_err())?;
            while let Some(chunk) = chunks.next().await {
                let chunk = chunk.map_err(|_| storage_err())?;
                n += chunk.len() as u64;
                h.update(&chunk);
            }
        }
        Ok(Some((h.finalize().into(), n)))
    }

    async fn delete(&self, key: &str) -> Result<(), OpError> {
        self.0.delete(key).await.map_err(|_| storage_err())
    }

    async fn mp_begin(&self, key: &str) -> Result<String, OpError> {
        let up = self.0.create_multipart_upload(key).execute().await;
        Ok(up.map_err(|_| storage_err())?.upload_id().await)
    }

    async fn mp_part(
        &self,
        key: &str,
        upload: &str,
        n: u16,
        bytes: Vec<u8>,
    ) -> Result<String, OpError> {
        let up = self
            .0
            .resume_multipart_upload(key, upload)
            .map_err(|_| storage_err())?;
        let part = up.upload_part(n, bytes).await.map_err(|_| storage_err())?;
        Ok(part.etag())
    }

    async fn mp_complete(
        &self,
        key: &str,
        upload: &str,
        parts: &[(u16, String)],
    ) -> Result<u64, OpError> {
        let up = self
            .0
            .resume_multipart_upload(key, upload)
            .map_err(|_| storage_err())?;
        let parts = parts
            .iter()
            .map(|(n, e)| worker::UploadedPart::new(*n, e.clone()));
        let obj = up.complete(parts).await.map_err(|_| storage_err())?;
        Ok(obj.size())
    }

    async fn mp_abort(&self, key: &str, upload: &str) -> Result<(), OpError> {
        let up = self
            .0
            .resume_multipart_upload(key, upload)
            .map_err(|_| storage_err())?;
        up.abort().await.map_err(|_| storage_err())
    }
}

/// Queues over the two producer bindings.
pub struct WorkerQueues<'a>(pub &'a worker::Env);

impl Queues for WorkerQueues<'_> {
    async fn send(&self, q: QueueName, body: Json) -> Result<(), OpError> {
        let binding = match q {
            QueueName::Ingest => INGEST_BINDING,
            QueueName::Audit => AUDIT_BINDING,
        };
        let queue = self.0.queue(binding).map_err(|_| storage_err())?;
        // JSON text, not a structured clone: no JS Map / float-number
        // ambiguity when the consumer decodes it.
        queue
            .send(body.to_string())
            .await
            .map_err(|_| storage_err())
    }
}

/// Workers AI embedding port. The `AI` binding is resolved per call, so a
/// request that embeds nothing never touches it.
pub struct WorkersAi<'a>(pub &'a worker::Env);

#[derive(Serialize)]
struct AiIn<'a> {
    text: &'a [&'a str],
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    truncate_inputs: bool,
}

#[derive(Deserialize)]
struct AiOut {
    data: Vec<Vec<f32>>,
}

impl EmbeddingPort for WorkersAi<'_> {
    fn embed(
        &self,
        model: &str,
        texts: &[&str],
        options: EmbedOptions,
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbedError>> {
        let input = AiIn {
            text: texts,
            truncate_inputs: options.truncate_inputs,
        };
        let model = model.to_string();
        async move {
            let ai = self
                .0
                .ai(AI_BINDING)
                .map_err(|_| EmbedError::Port("ai binding"))?;
            let out: AiOut = ai
                .run(model, input)
                .await
                .map_err(|_| EmbedError::Port("workers ai"))?;
            Ok(out.data)
        }
    }
}
