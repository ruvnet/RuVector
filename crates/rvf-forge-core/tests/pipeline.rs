//! End-to-end tests over the public API: inspect an RVF, verify it, build a
//! manifest, checksum the outputs, and emit provenance.
//!
//! Every fixture is constructed byte by byte in-test via
//! `rvf_forge_core::testkit`, so nothing here depends on a checked-in binary
//! whose provenance a reader would have to take on faith.

use rvf_forge_core::inspect::HEADER_SIZE;
use rvf_forge_core::testkit::{minimal_container, signed_segment, unsigned_segment, TestKeypair};
use rvf_forge_core::{
    build_manifest, checksums, inspect, inspect_bytes, provenance, verify, verify_bytes,
    verify_checksums, AppMetadata, BuildSpec, BuilderIdentity, BuilderKind, CapabilityClass,
    CapabilityGrant, CapabilityPolicy, CheckKind, ErrorCode, Outcome, PackagingMode,
    ProvenanceInputs, RuntimeProfile, SigningIdentity, SigningPosture, Target, VerifyOptions,
};
use rvf_types::SegmentType;

/// A container with a META segment, a signed WASM segment, and a root manifest.
fn signed_agent(kp: &TestKeypair) -> Vec<u8> {
    let mut data = unsigned_segment(
        SegmentType::Meta,
        b"rvf.capabilities=network,persistent-state",
        1,
    );
    data.extend(signed_segment(
        SegmentType::Wasm,
        b"\0asm\x01\0\0\0agent bytecode",
        2,
        kp,
    ));
    data.extend(unsigned_segment(SegmentType::Manifest, b"root", 3));
    data
}

fn app() -> AppMetadata {
    AppMetadata {
        name: "Example Agent".into(),
        version: "1.0.0".into(),
        publisher: "Example Inc".into(),
        license: Some("MIT".into()),
        description: Some("An example".into()),
    }
}

fn spec_for(identity: &str) -> BuildSpec {
    BuildSpec {
        rvf_identity: identity.to_string(),
        app: app(),
        targets: vec![
            Target::WindowsX64,
            Target::MacosUniversal,
            Target::LinuxAppimage,
        ],
        packaging_mode: PackagingMode::Embedded,
        runtime_profile: RuntimeProfile::Wasm,
        capability_policy: CapabilityPolicy::deny_all()
            .grant(
                CapabilityClass::Network,
                CapabilityGrant::scoped(["api.example.com"]),
            )
            .grant(
                CapabilityClass::PersistentState,
                CapabilityGrant::unscoped(),
            ),
        rvm_version: "0.4.0".into(),
        rvm_commit: "cafebabe".into(),
        compatibility_matrix_revision: Some("matrix-2026-08-01".into()),
    }
}

#[test]
fn a_signed_agent_inspects_verifies_and_builds() {
    let kp = TestKeypair::deterministic(11);
    let rvf = signed_agent(&kp);

    let inspection = inspect_bytes(&rvf).unwrap();
    assert_eq!(inspection.segments.len(), 3);
    assert_eq!(inspection.executable_segment_count, 1);
    assert!(inspection.all_executables_signed());
    assert_eq!(
        inspection.declared_capabilities,
        vec![CapabilityClass::Network, CapabilityClass::PersistentState]
    );
    assert!(inspection.root_manifest().is_some());

    let report = verify_bytes(&rvf, &VerifyOptions::with_trusted_keys(vec![kp.public])).unwrap();
    assert!(report.ok, "{:?}", report.failures());
    assert_eq!(report.rvf_identity, inspection.rvf_identity);

    let manifest = build_manifest(&spec_for(&inspection.rvf_identity)).unwrap();
    assert_eq!(manifest.rvf_identity, inspection.rvf_identity);
    assert_eq!(
        manifest.runtime_contract().rvf_identity,
        inspection.rvf_identity
    );
}

#[test]
fn the_manifest_is_byte_identical_for_the_same_spec() {
    let identity = "ab".repeat(32);
    let first = build_manifest(&spec_for(&identity))
        .unwrap()
        .to_canonical_json()
        .unwrap();
    let second = build_manifest(&spec_for(&identity))
        .unwrap()
        .to_canonical_json()
        .unwrap();
    assert_eq!(first, second);

    // And stable across a shuffled target list.
    let mut shuffled = spec_for(&identity);
    shuffled.targets.reverse();
    assert_eq!(
        build_manifest(&shuffled)
            .unwrap()
            .to_canonical_json()
            .unwrap(),
        first
    );
}

#[test]
fn one_rvf_yields_one_identity_for_every_target() {
    let kp = TestKeypair::deterministic(12);
    let rvf = signed_agent(&kp);
    let identity = inspect_bytes(&rvf).unwrap().rvf_identity;

    // ADR-291 acceptance criterion 2: every package from one build carries the
    // same rvfIdentity, whichever targets were requested.
    for targets in [
        vec![Target::WindowsX64],
        vec![Target::MacosIntel, Target::MacosArm64],
        vec![Target::RvmAppliance],
    ] {
        let mut spec = spec_for(&identity);
        spec.targets = targets;
        let contract = build_manifest(&spec).unwrap().runtime_contract();
        assert_eq!(contract.rvf_identity, identity);
        assert_eq!(contract.state_schema_version, 1);
        assert_eq!(contract.witness_schema_version, 1);
    }
}

#[test]
fn tampering_with_a_signed_segment_is_detected() {
    let kp = TestKeypair::deterministic(13);
    let clean = signed_agent(&kp);
    let opts = VerifyOptions::with_trusted_keys(vec![kp.public]);
    assert!(verify_bytes(&clean, &opts).unwrap().ok);

    // Flip a byte inside the WASM payload and repair its content hash, so that
    // only the signature is left to catch the change.
    let wasm = inspect_bytes(&clean).unwrap().segments[1].clone();
    assert_eq!(wasm.segment_type_name, "wasm");
    let header_at = wasm.offset as usize;
    let payload_at = header_at + HEADER_SIZE;
    let payload_len = wasm.payload_length as usize;

    let mut tampered = clean.clone();
    tampered[payload_at + 10] ^= 0xff;
    let repaired = rvf_wire::hash::compute_content_hash(
        wasm.checksum_algo,
        &tampered[payload_at..payload_at + payload_len],
    );
    // content_hash lives at header offset 0x28 and is 16 bytes wide.
    tampered[header_at + 0x28..header_at + 0x38].copy_from_slice(&repaired);

    let report = verify_bytes(&tampered, &opts).unwrap();
    assert!(
        !report.ok,
        "a repaired content hash must not launder tampering"
    );
    let failed = report.failures();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].check, CheckKind::Signature);
    assert_eq!(
        report.into_result().unwrap_err().code(),
        ErrorCode::VerifyFailed
    );
}

#[test]
fn tampering_without_repairing_the_hash_fails_the_hash_check() {
    let kp = TestKeypair::deterministic(14);
    let mut rvf = signed_agent(&kp);
    let wasm = inspect_bytes(&rvf).unwrap().segments[1].clone();
    rvf[wasm.offset as usize + HEADER_SIZE + 5] ^= 0xff;

    let report = verify_bytes(&rvf, &VerifyOptions::with_trusted_keys(vec![kp.public])).unwrap();
    assert!(!report.ok);
    let kinds: Vec<CheckKind> = report.failures().iter().map(|r| r.check).collect();
    assert!(kinds.contains(&CheckKind::ContentHash), "{kinds:?}");
}

#[test]
fn an_unsigned_executable_is_refused_by_default() {
    let mut rvf = unsigned_segment(SegmentType::Meta, b"name=agent", 1);
    rvf.extend(unsigned_segment(SegmentType::Wasm, b"\0asm\x01\0\0\0", 2));
    rvf.extend(unsigned_segment(SegmentType::Manifest, b"root", 3));

    let inspection = inspect_bytes(&rvf).unwrap();
    assert_eq!(inspection.unsigned_executable_segments, vec![2]);
    assert!(!inspection.all_executables_signed());

    let report = verify_bytes(&rvf, &VerifyOptions::default()).unwrap();
    assert!(!report.ok);
    let err = report.into_result().unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnsignedSegment);
    assert!(err.to_string().contains("wasm"), "{err}");
}

#[test]
fn every_verification_result_produces_a_record() {
    let kp = TestKeypair::deterministic(15);
    let rvf = signed_agent(&kp);
    let report = verify_bytes(&rvf, &VerifyOptions::with_trusted_keys(vec![kp.public])).unwrap();

    // One root-manifest check, one content hash per segment, one
    // executable-signed check and one signature check for the WASM segment.
    let count = |k: CheckKind| report.records.iter().filter(|r| r.check == k).count();
    assert_eq!(count(CheckKind::RootManifestPresent), 1);
    assert_eq!(count(CheckKind::ContentHash), 3);
    assert_eq!(count(CheckKind::ExecutableSigned), 1);
    assert_eq!(count(CheckKind::Signature), 1);
    assert!(report.records.iter().all(|r| !r.detail.is_empty()));
    assert!(report.records.iter().all(|r| r.outcome != Outcome::Fail));

    // The report round-trips through JSON, which is its wire form.
    let json = serde_json::to_string(&report).unwrap();
    let back: rvf_forge_core::VerificationReport = serde_json::from_str(&json).unwrap();
    assert_eq!(back, report);
}

#[test]
fn an_unsupported_target_is_refused_before_any_build_work() {
    let err = Target::parse("linux-rpm").unwrap_err();
    assert_eq!(err.code(), ErrorCode::UnsupportedTarget);

    let mut spec = spec_for(&"ab".repeat(32));
    spec.targets.clear();
    assert_eq!(
        build_manifest(&spec).unwrap_err().code(),
        ErrorCode::Manifest
    );
}

#[test]
fn checksums_round_trip_and_feed_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let kp = TestKeypair::deterministic(16);
    let rvf = signed_agent(&kp);

    std::fs::write(dir.path().join("Agent.rvf"), &rvf).unwrap();
    std::fs::write(dir.path().join("AgentSetup.exe"), b"nsis installer").unwrap();
    std::fs::write(dir.path().join("Agent.dmg"), b"disk image").unwrap();

    let artifacts = checksums(dir.path(), &["Agent.rvf", "AgentSetup.exe", "Agent.dmg"]).unwrap();
    assert_eq!(artifacts.len(), 3);
    assert!(verify_checksums(&artifacts, dir.path())
        .unwrap()
        .iter()
        .all(|r| r.matched));

    let inspection = inspect(dir.path().join("Agent.rvf")).unwrap();
    let report = verify(
        dir.path().join("Agent.rvf"),
        &VerifyOptions::with_trusted_keys(vec![kp.public]),
    )
    .unwrap();
    assert!(report.ok);

    let manifest = build_manifest(&spec_for(&inspection.rvf_identity)).unwrap();
    let record = provenance(&ProvenanceInputs {
        rvf_identity: inspection.rvf_identity.clone(),
        build_manifest_hash: manifest.manifest_hash().unwrap(),
        source_revision: "cafebabe".into(),
        builder: BuilderIdentity {
            id: "forge-worker".into(),
            kind: BuilderKind::Hosted,
            worker_id: Some("job-1".into()),
        },
        signing: SigningIdentity {
            posture: SigningPosture::KmsOrHsm,
            identity: Some("CN=Example Inc".into()),
        },
        started_at: "2026-08-03T10:00:00Z".into(),
        finished_at: "2026-08-03T10:04:12Z".into(),
        artifacts: artifacts.clone(),
        compatibility_matrix_revision: manifest.compatibility_matrix_revision.clone(),
    })
    .unwrap();

    // Invariant 5: the record traces back to input, runtime, revision,
    // builder, and signing identity.
    assert_eq!(record.rvf_identity, inspection.rvf_identity);
    assert_eq!(
        record.build_manifest_hash,
        manifest.manifest_hash().unwrap()
    );
    assert_eq!(record.source_revision, "cafebabe");
    assert_eq!(record.artifacts.len(), 3);
    assert_eq!(
        record.compatibility_matrix_revision.as_deref(),
        Some("matrix-2026-08-01")
    );
    assert_eq!(record.record_hash().unwrap().len(), 64);
}

#[test]
fn a_replaced_artifact_fails_checksum_verification() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("AgentSetup.exe"), b"genuine").unwrap();
    let manifest = checksums(dir.path(), &["AgentSetup.exe"]).unwrap();

    std::fs::write(dir.path().join("AgentSetup.exe"), b"malicious").unwrap();
    let results = verify_checksums(&manifest, dir.path()).unwrap();
    assert_eq!(results.len(), 1);
    assert!(!results[0].matched);
}

#[test]
fn a_truncated_container_is_invalid_rather_than_silently_short() {
    let kp = TestKeypair::deterministic(17);
    let rvf = signed_agent(&kp);
    let truncated = &rvf[..rvf.len() - 70];

    let err = inspect_bytes(truncated).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidRvf);
    assert_eq!(
        verify_bytes(truncated, &VerifyOptions::default())
            .unwrap_err()
            .code(),
        ErrorCode::InvalidRvf
    );
}

#[test]
fn a_missing_file_is_an_io_error() {
    let err = inspect("/nonexistent/agent.rvf").unwrap_err();
    assert_eq!(err.code(), ErrorCode::Io);
    assert_eq!(
        verify("/nonexistent/agent.rvf", &VerifyOptions::default())
            .unwrap_err()
            .code(),
        ErrorCode::Io
    );
}

#[test]
fn the_declared_capability_set_defaults_to_deny() {
    let rvf = minimal_container("");
    let inspection = inspect_bytes(&rvf).unwrap();
    assert!(inspection.declared_capabilities.is_empty());

    let policy = CapabilityPolicy::deny_all();
    for class in CapabilityClass::ALL {
        assert!(!policy.is_granted(class), "{class}");
    }

    // A policy that withholds a class the artifact declared is reported.
    let declared = inspect_bytes(&minimal_container("network,gpu"))
        .unwrap()
        .declared_capabilities;
    let narrow =
        CapabilityPolicy::deny_all().grant(CapabilityClass::Network, CapabilityGrant::unscoped());
    assert_eq!(
        narrow.undeclared_grants(&declared),
        vec![CapabilityClass::Gpu]
    );
}
