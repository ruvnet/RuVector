# ADR-109: Backup and Disaster Recovery Strategy

**Status**: Accepted, Implemented
**Date**: 2026-03-15
**Implemented**: 2026-03-15
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-059 (Shared Brain Google Cloud), ADR-064 (Pi Brain Infrastructure)

## 1. Context

The π.ruv.io shared brain system stores critical data:
- **958+ knowledge memories** with embeddings, witness chains, and quality scores
- **57 contributors** with reputation histories
- **124,507 graph edges** representing knowledge relationships
- **8 Brainpedia pages** with delta logs and evidence chains
- **947 votes** for federated learning preference pairs

Currently, **NO scheduled backups exist**:
- Cloud Scheduler API is disabled
- Firestore Point-in-Time Recovery (PITR) is disabled
- Delete protection is disabled
- No dedicated backup bucket exists
- No automated export jobs are configured

This creates unacceptable risk: a single misconfiguration, code bug, or malicious actor could destroy months of contributed knowledge with no recovery path.

## 2. Decision

Implement a multi-layer backup strategy with:
1. **Firestore PITR** for short-term recovery (7-day window)
2. **Daily automated exports** to GCS for long-term retention
3. **Weekly full exports** with 90-day retention
4. **Cross-region replication** for disaster recovery
5. **Delete protection** on critical databases

## 3. Architecture

### 3.1 Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

| Scenario | RPO | RTO | Method |
|----------|-----|-----|--------|
| Accidental delete (single doc) | 0 | 5 min | Firestore PITR |
| Accidental bulk delete | 0 | 15 min | Firestore PITR |
| Database corruption | 24 hrs | 1 hr | Daily GCS export |
| Regional outage | 24 hrs | 4 hrs | Cross-region restore |
| Complete project loss | 7 days | 24 hrs | Weekly GCS export |

### 3.2 Firestore Configuration

```yaml
# Enable via gcloud
pointInTimeRecoveryEnablement: POINT_IN_TIME_RECOVERY_ENABLED
deleteProtectionState: DELETE_PROTECTION_ENABLED
```

Commands to enable:
```bash
# Enable PITR (7-day recovery window)
gcloud firestore databases update --project=ruv-dev \
  --point-in-time-recovery-enablement=ENABLED

# Enable delete protection
gcloud firestore databases update --project=ruv-dev \
  --delete-protection-state=ENABLED
```

### 3.3 GCS Backup Buckets

| Bucket | Location | Class | Retention | Purpose |
|--------|----------|-------|-----------|---------|
| `ruvector-backups-daily` | US-CENTRAL1 | Standard | 30 days | Daily exports |
| `ruvector-backups-weekly` | US | Nearline | 90 days | Weekly archives |
| `ruvector-backups-dr` | EU | Coldline | 365 days | Disaster recovery |

Create buckets:
```bash
# Daily backups (same region as service)
gcloud storage buckets create gs://ruvector-backups-daily \
  --project=ruv-dev --location=us-central1 --uniform-bucket-level-access

# Weekly archives (multi-region)
gcloud storage buckets create gs://ruvector-backups-weekly \
  --project=ruv-dev --location=us --storage-class=nearline \
  --uniform-bucket-level-access

# DR copy (cross-region)
gcloud storage buckets create gs://ruvector-backups-dr \
  --project=ruv-dev --location=eu --storage-class=coldline \
  --uniform-bucket-level-access
```

### 3.4 Cloud Scheduler Jobs

#### Daily Export (02:00 UTC)
```bash
# Enable Cloud Scheduler API
gcloud services enable cloudscheduler.googleapis.com --project=ruv-dev

# Create service account for exports
gcloud iam service-accounts create firestore-backup \
  --project=ruv-dev \
  --display-name="Firestore Backup Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding ruv-dev \
  --member="serviceAccount:firestore-backup@ruv-dev.iam.gserviceaccount.com" \
  --role="roles/datastore.importExportAdmin"

gcloud projects add-iam-policy-binding ruv-dev \
  --member="serviceAccount:firestore-backup@ruv-dev.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Create daily export job
gcloud scheduler jobs create http firestore-daily-backup \
  --project=ruv-dev --location=us-central1 \
  --schedule="0 2 * * *" \
  --uri="https://firestore.googleapis.com/v1/projects/ruv-dev/databases/(default):exportDocuments" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"outputUriPrefix":"gs://ruvector-backups-daily/firestore/daily"}' \
  --oauth-service-account-email=firestore-backup@ruv-dev.iam.gserviceaccount.com
```

#### Weekly Full Export (Sunday 03:00 UTC)
```bash
gcloud scheduler jobs create http firestore-weekly-backup \
  --project=ruv-dev --location=us-central1 \
  --schedule="0 3 * * 0" \
  --uri="https://firestore.googleapis.com/v1/projects/ruv-dev/databases/(default):exportDocuments" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"outputUriPrefix":"gs://ruvector-backups-weekly/firestore/weekly"}' \
  --oauth-service-account-email=firestore-backup@ruv-dev.iam.gserviceaccount.com
```

### 3.5 GCS Object Lifecycle

```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
```

Apply lifecycle policies:
```bash
# Daily bucket: 30-day retention
gcloud storage buckets update gs://ruvector-backups-daily \
  --lifecycle-file=lifecycle-30d.json

# Weekly bucket: 90-day retention
gcloud storage buckets update gs://ruvector-backups-weekly \
  --lifecycle-file=lifecycle-90d.json

# DR bucket: 365-day retention
gcloud storage buckets update gs://ruvector-backups-dr \
  --lifecycle-file=lifecycle-365d.json
```

### 3.6 Cross-Region Replication

Use Storage Transfer Service for DR copies:
```bash
gcloud transfer jobs create \
  gs://ruvector-backups-weekly \
  gs://ruvector-backups-dr \
  --project=ruv-dev \
  --name="weekly-dr-replication" \
  --schedule-starts="2026-03-16T04:00:00Z" \
  --schedule-repeats-every="P7D"
```

### 3.7 GCS RVF Container Backups

The RVF containers in `ruvector-brain-us-central1` also need protection:
```bash
# Enable object versioning for immutability
gcloud storage buckets update gs://ruvector-brain-us-central1 \
  --versioning

# Create cross-region backup
gcloud transfer jobs create \
  gs://ruvector-brain-us-central1 \
  gs://ruvector-backups-dr/rvf-containers \
  --project=ruv-dev \
  --name="rvf-daily-replication" \
  --schedule-starts="2026-03-16T05:00:00Z" \
  --schedule-repeats-every="P1D"
```

### 3.8 Secrets Backup

Export secrets to encrypted backup:
```bash
# Create secrets backup script (run manually monthly)
#!/bin/bash
DATE=$(date +%Y%m%d)
SECRETS=(brain-api-key brain-signing-key cloudflare-api-token huggingface-token)

for secret in "${SECRETS[@]}"; do
  gcloud secrets versions access latest --secret=$secret --project=ruv-dev \
    | gcloud kms encrypt --key=projects/ruv-dev/locations/global/keyRings/backup/cryptoKeys/secrets \
      --plaintext-file=- --ciphertext-file=secrets-$secret-$DATE.enc
done

gsutil cp secrets-*.enc gs://ruvector-backups-weekly/secrets/$DATE/
rm secrets-*.enc
```

## 4. Monitoring and Alerts

### 4.1 Backup Job Monitoring

```bash
# Create alert policy for failed backup jobs
gcloud alpha monitoring policies create \
  --project=ruv-dev \
  --display-name="Backup Job Failure Alert" \
  --condition-filter='resource.type="cloud_scheduler_job" AND metric.type="cloudjobs.googleapis.com/job/completed_count" AND metric.labels.status!="SUCCESS"' \
  --notification-channels=<channel-id>
```

### 4.2 Storage Monitoring

Monitor backup bucket sizes to detect anomalies:
```bash
# Alert if daily backup size drops >50% (possible data loss)
# Alert if weekly backup fails to appear
```

## 5. Recovery Procedures

### 5.1 Point-in-Time Recovery (Firestore PITR)

```bash
# Restore to specific timestamp
gcloud firestore databases restore \
  --source-database="(default)" \
  --destination-database="(default)-restored" \
  --source-backup-time="2026-03-15T10:00:00Z" \
  --project=ruv-dev
```

### 5.2 Import from GCS Export

```bash
# Import from daily backup
gcloud firestore import \
  gs://ruvector-backups-daily/firestore/daily/2026-03-15T02:00:00_12345/ \
  --project=ruv-dev
```

### 5.3 Full Disaster Recovery

1. Create new GCP project (if needed)
2. Import Firestore from DR bucket
3. Copy RVF containers from DR bucket
4. Restore secrets from encrypted backup
5. Deploy Cloud Run service
6. Update DNS records

## 6. Implementation Plan

| Phase | Task | Priority | Effort |
|-------|------|----------|--------|
| 1 | Enable Firestore PITR | Critical | 5 min |
| 2 | Enable delete protection | Critical | 2 min |
| 3 | Create backup buckets | High | 10 min |
| 4 | Enable Cloud Scheduler API | High | 2 min |
| 5 | Create service account | High | 5 min |
| 6 | Configure daily export job | High | 10 min |
| 7 | Configure weekly export job | Medium | 5 min |
| 8 | Set up lifecycle policies | Medium | 10 min |
| 9 | Configure cross-region transfer | Medium | 15 min |
| 10 | Set up monitoring alerts | Medium | 20 min |
| 11 | Document recovery procedures | Low | 30 min |
| 12 | Test recovery (quarterly drill) | Low | 2 hrs |

**Total initial setup: ~2 hours**
**Ongoing cost: ~$5-15/month** (storage + scheduler jobs)

## 7. Cost Estimate

| Component | Monthly Cost |
|-----------|--------------|
| Firestore PITR | $0 (included) |
| Daily bucket (30d × ~10MB) | ~$0.10 |
| Weekly bucket (90d × ~50MB) | ~$0.50 |
| DR bucket (365d × ~200MB) | ~$1.00 |
| Cloud Scheduler (31 jobs/month) | ~$0.10 |
| Storage Transfer Service | $0 (free tier) |
| **Total** | **~$2-5/month** |

## 8. Consequences

### Positive
- **Recovery capability**: Can recover from any failure within RPO/RTO targets
- **Compliance ready**: Audit trail of backups for compliance requirements
- **Peace of mind**: No risk of catastrophic data loss
- **Low cost**: Under $5/month for comprehensive protection

### Negative
- **Added complexity**: More infrastructure to manage
- **Operational overhead**: Quarterly recovery drills required
- **Storage costs**: Small but non-zero ongoing expense

### Neutral
- Backup data is encrypted at rest (GCS default)
- Export/import operations are eventually consistent

## 9. References

- [Firestore Point-in-Time Recovery](https://cloud.google.com/firestore/docs/pitr)
- [Firestore Export/Import](https://cloud.google.com/firestore/docs/manage-data/export-import)
- [Cloud Scheduler Documentation](https://cloud.google.com/scheduler/docs)
- [GCS Object Lifecycle](https://cloud.google.com/storage/docs/lifecycle)
- [Storage Transfer Service](https://cloud.google.com/storage-transfer-service)
