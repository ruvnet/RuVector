//! [`BlobStore`] over the R2 bucket binding (ADR-351 §6.3). Object bytes
//! are streamed or held one part at a time; nothing here buffers a whole
//! package except `get_range` for a bounded range.

use crate::registry_ports::BlobStore;
use crate::registry_wire::RvfError;
use futures_util::StreamExt;
use ruvector_edge_registry::upload::ObjectEvidence;
use sha2::{Digest, Sha256};
use worker::{Bucket, Conditional, FixedLengthStream, Range, UploadedPart};

/// R2 binding of the `ruvector-edge-data` bucket.
pub const R2_BINDING: &str = "EDGE_DATA";

/// The bucket.
pub struct R2(pub Bucket);

fn se(_: worker::Error) -> RvfError {
    RvfError::storage()
}

fn gone(e: &worker::Error) -> bool {
    let m = e.to_string();
    m.contains("10024") || m.contains("NoSuchUpload") || m.contains("does not exist")
}

impl BlobStore for R2 {
    async fn mp_create(&self, key: &str) -> Result<String, RvfError> {
        let up = self
            .0
            .create_multipart_upload(key)
            .execute()
            .await
            .map_err(se)?;
        Ok(up.upload_id().await)
    }

    async fn mp_part(
        &self,
        key: &str,
        upload: &str,
        n: u16,
        bytes: Vec<u8>,
    ) -> Result<String, RvfError> {
        let up = self.0.resume_multipart_upload(key, upload).map_err(se)?;
        Ok(up.upload_part(n, bytes).await.map_err(se)?.etag())
    }

    async fn mp_complete(
        &self,
        key: &str,
        upload: &str,
        parts: &[(u16, String)],
    ) -> Result<(), RvfError> {
        let up = self.0.resume_multipart_upload(key, upload).map_err(se)?;
        let parts = parts.iter().map(|(n, e)| UploadedPart::new(*n, e.clone()));
        up.complete(parts).await.map(|_| ()).map_err(se)
    }

    async fn mp_abort(&self, key: &str, upload: &str) -> Result<(), RvfError> {
        let up = self.0.resume_multipart_upload(key, upload).map_err(se)?;
        match up.abort().await {
            Ok(()) => Ok(()),
            // An already completed or aborted upload (R2 NoSuchUpload,
            // 10024) refuses the abort: nothing left to do.
            Err(e) if gone(&e) => Ok(()),
            Err(e) => Err(se(e)),
        }
    }

    async fn size(&self, key: &str) -> Result<Option<u64>, RvfError> {
        Ok(self.0.head(key).await.map_err(se)?.map(|o| o.size()))
    }

    async fn get_range(
        &self,
        key: &str,
        offset: u64,
        len: u64,
    ) -> Result<Option<Vec<u8>>, RvfError> {
        let range = Range::OffsetWithLength {
            offset,
            length: len,
        };
        let Some(obj) = self.0.get(key).range(range).execute().await.map_err(se)? else {
            return Ok(None);
        };
        let body = obj.body().ok_or_else(RvfError::storage)?;
        body.bytes().await.map(Some).map_err(se)
    }

    async fn put_checked(
        &self,
        key: &str,
        bytes: Vec<u8>,
        sha256: [u8; 32],
    ) -> Result<ObjectEvidence, RvfError> {
        let obj = self
            .0
            .put(key, bytes)
            .sha256(sha256.to_vec())
            .execute()
            .await
            .map_err(se)?;
        Ok(ObjectEvidence {
            sha256,
            size: obj.ok_or_else(RvfError::storage)?.size(),
        })
    }

    async fn copy_create_only(
        &self,
        from: &str,
        to: &str,
        sha256: [u8; 32],
    ) -> Result<bool, RvfError> {
        let Some(src) = self.0.get(from).execute().await.map_err(se)? else {
            return Err(RvfError::storage());
        };
        let size = src.size();
        let stream = src
            .body()
            .ok_or_else(RvfError::storage)?
            .stream()
            .map_err(se)?;
        let only_if = Conditional {
            etag_does_not_match: Some("*".into()),
            ..Default::default()
        };
        let put = self
            .0
            .put(to, FixedLengthStream::wrap(stream, size))
            .sha256(sha256.to_vec())
            .only_if(only_if)
            .execute()
            .await
            .map_err(se)?;
        Ok(put.is_some())
    }

    async fn measure(&self, key: &str) -> Result<Option<ObjectEvidence>, RvfError> {
        let Some(obj) = self.0.get(key).execute().await.map_err(se)? else {
            return Ok(None);
        };
        let mut stream = obj
            .body()
            .ok_or_else(RvfError::storage)?
            .stream()
            .map_err(se)?;
        let (mut h, mut size) = (Sha256::new(), 0u64);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(se)?;
            size += chunk.len() as u64;
            h.update(&chunk);
        }
        Ok(Some(ObjectEvidence {
            sha256: h.finalize().into(),
            size,
        }))
    }

    async fn delete(&self, key: &str) -> Result<(), RvfError> {
        self.0.delete(key).await.map_err(se)
    }
}
