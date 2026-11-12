//! Bulk-import job: crash mid-import, resume from the persisted cursor, and
//! reach the same final state as an uninterrupted run.

mod common;
use common::*;
use ruvector_edge_snapshot::types::sha256;
use ruvector_edge_snapshot::*;
use std::collections::{BTreeMap, BTreeSet};

/// A shard that applies upserts under `op_id` idempotency.
#[derive(Default, PartialEq, Debug)]
struct MockShard {
    rows: BTreeMap<String, (Vec<u32>, Option<String>)>,
    ops: BTreeSet<String>,
    applied: u64,
    deduped: u64,
}

impl MockShard {
    fn apply(&mut self, op_id: &str, batch: &ImportBatch) {
        if !self.ops.insert(op_id.to_string()) {
            self.deduped += 1;
            return;
        }
        self.applied += 1;
        for r in &batch.rows {
            let bits = r.values.iter().map(|v| v.to_bits()).collect();
            self.rows.insert(r.id.clone(), (bits, r.metadata.clone()));
        }
    }
}

fn file(n: u64, dim: u16) -> Vec<u8> {
    let mut f = Vec::new();
    let mut sink = |b: &[u8]| f.extend_from_slice(b);
    let mut x = RvfExporter::new(dim, Metric::L2, 37).unwrap();
    for r in rows(n, dim, 21) {
        x.push(&r, &mut sink).unwrap();
    }
    x.finish(&mut sink);
    f
}

fn spec(dim: u16) -> ImportSpec {
    ImportSpec {
        dim,
        metric: Metric::L2,
        limits: ImportLimits {
            max_batch_rows: 16,
            ..ImportLimits::default()
        },
    }
}

fn new_job(id: &str, f: &[u8]) -> ImportJob {
    ImportJob::new(id, TENANT_A, UID, "up_1", f.len() as u64, sha256(&[f])).unwrap()
}

fn facts(f: &[u8]) -> ObjectFacts {
    ObjectFacts {
        size: f.len() as u64,
        sha256: Some(sha256(&[f])),
    }
}

/// One delivery with the caller's current `limits` (the job pins the
/// first delivery's); `crash_after` stops after that many batches were
/// applied to the shard, *before* the last one is committed.
fn deliver_with(
    job: &mut ImportJob,
    f: &[u8],
    shard: &mut MockShard,
    crash_after: Option<usize>,
    limits: ImportLimits,
) -> bool {
    let summary = inspect_tail(&f[f.len() - 2048..], f.len() as u64, &limits).unwrap();
    let (cursor, pinned) = job.deliver(facts(f), &summary, limits).unwrap();
    let s = ImportSpec {
        limits: pinned,
        ..spec(summary.dim)
    };
    let mut imp = RvfImporter::new(s, summary, cursor).unwrap();
    let mut applied = 0;
    for piece in f[cursor.byte_offset as usize..].chunks(333) {
        imp.feed(piece).unwrap();
        while let Some(b) = imp.next_batch().unwrap() {
            shard.apply(&job.op_id(b.seq).unwrap(), &b);
            applied += 1;
            if crash_after == Some(applied) {
                return false; // process dies: batch applied, cursor not recorded
            }
            assert!(job.commit_batch(&b).unwrap());
        }
    }
    job.complete(&imp.finish().unwrap()).unwrap();
    true
}

fn deliver(job: &mut ImportJob, f: &[u8], shard: &mut MockShard, crash: Option<usize>) -> bool {
    deliver_with(job, f, shard, crash, spec(6).limits)
}

#[test]
fn resume_after_crash_yields_identical_final_state() {
    let dim = 6;
    let f = file(1_000, dim);

    let mut clean_job = new_job("job_1", &f);
    let mut clean = MockShard::default();
    assert!(deliver(&mut clean_job, &f, &mut clean, None));
    assert_eq!(clean_job.state, JobState::Done);
    assert_eq!(clean.rows.len(), 1_000);

    for crash_points in [vec![1], vec![5, 3], vec![20, 1, 17]] {
        let mut job = new_job("job_1", &f);
        let mut shard = MockShard::default();
        for c in &crash_points {
            assert!(!deliver(&mut job, &f, &mut shard, Some(*c)));
            // Only the persisted bytes survive the crash.
            job = ImportJob::decode(&job.encode()).unwrap();
            assert_eq!(job.state, JobState::Running);
        }
        assert!(deliver(&mut job, &f, &mut shard, None));
        assert_eq!(job.state, JobState::Done);
        assert_eq!(job.attempts as usize, crash_points.len() + 1);
        assert_eq!(shard.rows, clean.rows);
        assert_eq!(shard.ops, clean.ops, "same deterministic op_ids");
        assert_eq!(shard.applied, clean.applied);
        assert_eq!(
            shard.deduped as usize,
            crash_points.len(),
            "each crash replays exactly one batch"
        );
        assert_eq!(job.cursor, clean_job.cursor);
    }
}

#[test]
fn resume_mid_record_skips_consumed_rows() {
    // 37 rows per record and 16 per batch: batches end mid-record.
    let f = file(100, 4);
    let mut job = new_job("job_2", &f);
    let mut shard = MockShard::default();
    assert!(!deliver(&mut job, &f, &mut shard, Some(2)));
    let c = job.cursor;
    assert_eq!((c.row_in_record, c.batch_seq, c.rows_done), (16, 1, 16));
    assert!(deliver(&mut job, &f, &mut shard, None));
    assert_eq!(shard.rows.len(), 100);
}

#[test]
fn redelivery_uses_limits_pinned_at_job_start() {
    // Quota 1000 at start; after a crash at 480 rows the tenant's *remaining*
    // quota is 520 and a deploy changed the batch size. The resumed job must
    // keep comparing against the start quota and regenerate identical batches.
    let f = file(1_000, 6);
    let start = ImportLimits {
        max_rows: 1_000,
        ..spec(6).limits
    };
    let mut clean_job = new_job("job_p", &f);
    let mut clean = MockShard::default();
    assert!(deliver_with(&mut clean_job, &f, &mut clean, None, start));

    let mut job = new_job("job_p", &f);
    let mut shard = MockShard::default();
    assert!(!deliver_with(&mut job, &f, &mut shard, Some(31), start));
    job = ImportJob::decode(&job.encode()).unwrap();
    let later = ImportLimits {
        max_rows: 520,
        max_batch_rows: 64,
        ..start
    };
    assert!(deliver_with(&mut job, &f, &mut shard, None, later));
    assert_eq!(shard.rows, clean.rows);
    assert_eq!(shard.ops, clean.ops);
    assert_eq!(shard.deduped, 1);
    let b = job.binding.unwrap();
    assert_eq!((b.max_rows, b.max_batch_rows), (1_000, 16));
}

#[test]
fn job_is_bound_to_its_upload() {
    let f = file(200, 6);
    let lim = spec(6).limits;
    let mut job = new_job("job_b", &f);
    assert_eq!(job.staging_key(), format!("staging/{TENANT_A}/up_1"));
    let other = ImportJob::new("job_b", TENANT_B, UID, "up_1", 1, [0; 32]).unwrap();
    assert_ne!(other.staging_key(), job.staging_key());
    assert!(ImportJob::new("j", TENANT_A, UID, "../up", 1, [0; 32]).is_err());

    let sum = inspect_tail(&f, f.len() as u64, &lim).unwrap();
    let wrong_size = ObjectFacts {
        size: 1,
        ..facts(&f)
    };
    assert_eq!(
        job.deliver(wrong_size, &sum, lim).unwrap_err(),
        JobError::UploadMismatch("size")
    );
    let wrong_sha = ObjectFacts {
        sha256: Some([7; 32]),
        ..facts(&f)
    };
    assert_eq!(
        job.deliver(wrong_sha, &sum, lim).unwrap_err(),
        JobError::UploadMismatch("sha256")
    );
    assert_eq!(job.state, JobState::Queued, "refusals change nothing");
    job.deliver(facts(&f), &sum, lim).unwrap();

    // The object at the staging key changed between deliveries (same size,
    // no stored checksum): the pinned manifest hash refuses it.
    let n = f.len();
    let mut other_sum = inspect_tail(&f, n as u64, &lim).unwrap();
    other_sum.manifest_hash[0] ^= 1;
    let no_sha = ObjectFacts {
        size: n as u64,
        sha256: None,
    };
    assert_eq!(
        job.deliver(no_sha, &other_sum, lim).unwrap_err(),
        JobError::UploadMismatch("manifest")
    );
}

#[test]
fn full_stream_sha256_is_checked_at_completion() {
    let f = file(50, 6);
    let lim = spec(6).limits;
    // Declared digest differs from the bytes; R2 stored no checksum.
    let mut job = ImportJob::new("job_s", TENANT_A, UID, "up_1", f.len() as u64, [3; 32]).unwrap();
    let sum = inspect_tail(&f, f.len() as u64, &lim).unwrap();
    let no_sha = ObjectFacts {
        size: f.len() as u64,
        sha256: None,
    };
    let (cursor, pinned) = job.deliver(no_sha, &sum, lim).unwrap();
    let s = ImportSpec {
        limits: pinned,
        ..spec(6)
    };
    let mut imp = RvfImporter::new(s, sum, cursor).unwrap();
    imp.feed(&f).unwrap();
    while let Some(b) = imp.next_batch().unwrap() {
        job.commit_batch(&b).unwrap();
    }
    let totals = imp.finish().unwrap();
    assert_eq!(totals.sha256, Some(sha256(&[&f])));
    assert_eq!(
        job.complete(&totals).unwrap_err(),
        JobError::UploadMismatch("sha256")
    );
    job.fail(FailCode::Integrity).unwrap();
}

#[test]
fn state_machine_rejects_illegal_transitions_and_gaps() {
    let f = file(10, 6);
    let mut job = new_job("job_3", &f);
    let b = |seq| ImportBatch {
        seq,
        rows: vec![],
        cursor_after: ImportCursor {
            batch_seq: seq + 1,
            ..ImportCursor::default()
        },
    };
    assert!(matches!(
        job.commit_batch(&b(0)),
        Err(JobError::IllegalTransition { from: "queued" })
    ));
    assert!(job.op_id(0).is_err(), "no op ids before the binding exists");
    let lim = spec(6).limits;
    let sum = inspect_tail(&f, f.len() as u64, &lim).unwrap();
    job.deliver(facts(&f), &sum, lim).unwrap();
    assert!(job.commit_batch(&b(0)).unwrap());
    assert!(
        !job.commit_batch(&b(0)).unwrap(),
        "replayed batch is a no-op"
    );
    assert_eq!(
        job.commit_batch(&b(2)).unwrap_err(),
        JobError::BatchGap {
            expected: 1,
            got: 2
        }
    );
    job.fail(FailCode::Integrity).unwrap();
    assert_eq!(job.state, JobState::Failed(FailCode::Integrity));
    assert!(job.deliver(facts(&f), &sum, lim).is_err());
    assert!(job.fail(FailCode::Cancelled).is_err());
    assert_eq!(ImportJob::decode(&job.encode()).unwrap(), job);
    assert!(ImportJob::new("../x", TENANT_A, UID, "u", 1, [0; 32]).is_err());
    assert!(ImportJob::new("j", "tenant", UID, "u", 1, [0; 32]).is_err());
    let mut bytes = job.encode();
    bytes.push(0);
    assert!(ImportJob::decode(&bytes).is_err());
}

#[test]
fn op_ids_are_deterministic_and_bound_to_the_batch_contents() {
    let bind = JobBinding {
        manifest_hash: [1; 32],
        max_batch_rows: 500,
        max_rows: 10,
    };
    let a = batch_op_id("job_1", &bind, 7);
    assert_eq!(a, batch_op_id("job_1", &bind, 7));
    assert_ne!(a, batch_op_id("job_1", &bind, 8));
    assert_ne!(a, batch_op_id("job_2", &bind, 7));
    let other_file = JobBinding {
        manifest_hash: [2; 32],
        ..bind
    };
    let other_batching = JobBinding {
        max_batch_rows: 499,
        ..bind
    };
    assert_ne!(a, batch_op_id("job_1", &other_file, 7));
    assert_ne!(a, batch_op_id("job_1", &other_batching, 7));
    assert_eq!(a.len(), 26);
    assert!(a
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()));
}
