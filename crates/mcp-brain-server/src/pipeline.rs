//! RVF container construction pipeline (ADR-075 Phase 5).
//!
//! Assembles memory data into a multi-segment RVF container using rvf-wire.
//! Each container has at minimum VEC + META + WITNESS segments, plus optional
//! DiffPrivacyProof and RedactionLog segments when those features are active.

use rvf_types::SegmentFlags;

/// Input data for building an RVF container.
pub struct RvfPipelineInput<'a> {
    pub memory_id: &'a str,
    pub embedding: &'a [f32],
    pub title: &'a str,
    pub content: &'a str,
    pub tags: &'a [String],
    pub category: &'a str,
    pub contributor_id: &'a str,
    pub witness_chain: Option<&'a [u8]>,
    pub dp_proof_json: Option<&'a str>,
    pub redaction_log_json: Option<&'a str>,
}

/// Build an RVF container from pipeline input.
/// Returns the serialized container bytes (concatenated 64-byte-aligned segments).
/// Returns an error if metadata serialization fails (prevents silent data loss).
pub fn build_rvf_container(input: &RvfPipelineInput<'_>) -> Result<Vec<u8>, String> {
    let flags = SegmentFlags::empty();
    let mut container = Vec::new();
    let mut seg_id: u64 = 1;

    // Segment 1: VEC (0x01) — embedding as f32 little-endian bytes
    {
        let mut payload = Vec::with_capacity(input.embedding.len() * 4);
        for &val in input.embedding {
            payload.extend_from_slice(&val.to_le_bytes());
        }
        let seg = rvf_wire::write_segment(0x01, &payload, flags, seg_id);
        container.extend_from_slice(&seg);
        seg_id += 1;
    }

    // Segment 2: META (0x07) — JSON metadata
    {
        let meta = serde_json::json!({
            "memory_id": input.memory_id,
            "title": input.title,
            "content": input.content,
            "tags": input.tags,
            "category": input.category,
            "contributor_id": input.contributor_id,
        });
        let payload = serde_json::to_vec(&meta)
            .map_err(|e| format!("Failed to serialize RVF metadata: {e}"))?;
        let seg = rvf_wire::write_segment(0x07, &payload, flags, seg_id);
        container.extend_from_slice(&seg);
        seg_id += 1;
    }

    // Segment 3: WITNESS (0x0A) — witness chain bytes (if present)
    if let Some(chain) = input.witness_chain {
        let seg = rvf_wire::write_segment(0x0A, chain, flags, seg_id);
        container.extend_from_slice(&seg);
        seg_id += 1;
    }

    // Segment 4: DiffPrivacyProof (0x34) — proof JSON bytes (if DP enabled)
    if let Some(proof) = input.dp_proof_json {
        let seg = rvf_wire::write_segment(0x34, proof.as_bytes(), flags, seg_id);
        container.extend_from_slice(&seg);
        seg_id += 1;
    }

    // Segment 5: RedactionLog (0x35) — redaction JSON bytes (if PII stripped)
    if let Some(log) = input.redaction_log_json {
        let seg = rvf_wire::write_segment(0x35, log.as_bytes(), flags, seg_id);
        container.extend_from_slice(&seg);
        let _ = seg_id; // suppress unused warning on last increment
    }

    Ok(container)
}

/// Count the number of segments in a serialized RVF container.
/// Walks the container by reading 64-byte headers and skipping payloads.
pub fn count_segments(container: &[u8]) -> usize {
    let mut count = 0;
    let mut offset = 0;
    while offset + 64 <= container.len() {
        // Read payload_length from header bytes [16..24] (little-endian u64)
        let payload_len = u64::from_le_bytes(
            container[offset + 16..offset + 24]
                .try_into()
                .unwrap_or([0u8; 8]),
        ) as usize;
        let padded = rvf_wire::calculate_padded_size(64, payload_len);
        count += 1;
        offset += padded;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rvf_container_has_segments() {
        let embedding = vec![0.1f32, 0.2, 0.3, 0.4];
        let tags = vec!["test".to_string()];
        let witness_chain = rvf_crypto::create_witness_chain(&[
            rvf_crypto::WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(b"test"),
                timestamp_ns: 1000,
                witness_type: 0x01,
            },
        ]);
        let dp_proof = r#"{"epsilon":1.0,"delta":1e-5}"#;
        let redaction_log = r#"{"entries":[],"total_redactions":0}"#;

        let input = RvfPipelineInput {
            memory_id: "test-id",
            embedding: &embedding,
            title: "Test Title",
            content: "Test content",
            tags: &tags,
            category: "pattern",
            contributor_id: "test-contributor",
            witness_chain: Some(&witness_chain),
            dp_proof_json: Some(dp_proof),
            redaction_log_json: Some(redaction_log),
        };

        let container = build_rvf_container(&input).expect("build should succeed");
        let seg_count = count_segments(&container);
        // VEC + META + WITNESS + DP_PROOF + REDACTION_LOG = 5 segments
        assert!(seg_count >= 3, "expected >= 3 segments, got {seg_count}");
        assert_eq!(seg_count, 5);
    }

    #[test]
    fn test_rvf_container_minimal() {
        let embedding = vec![1.0f32; 128];
        let tags = vec![];
        let input = RvfPipelineInput {
            memory_id: "min-id",
            embedding: &embedding,
            title: "Minimal",
            content: "Content",
            tags: &tags,
            category: "solution",
            contributor_id: "anon",
            witness_chain: None,
            dp_proof_json: None,
            redaction_log_json: None,
        };
        let container = build_rvf_container(&input).expect("build should succeed");
        let seg_count = count_segments(&container);
        // VEC + META = 2 segments (no witness, no DP, no redaction)
        assert_eq!(seg_count, 2);
    }
}
