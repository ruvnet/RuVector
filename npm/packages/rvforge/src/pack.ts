/**
 * `rvforge pack agent.rvf` — the publisher-side validation gate (requirements
 * P4 "Package", ADR-294 §4).
 *
 * Pack runs every item on the ADR-294 validation list that is decidable from
 * the artifact and the project file alone, then assembles the two registry
 * objects a publish is made of: an unsigned `CapabilityManifest` and a draft
 * `Release`. Nothing here signs, uploads, or executes.
 *
 * Two properties are load-bearing:
 *
 * - **Every check runs, then the aggregate decides.** A publisher fixing four
 *   problems should see four problems, not discover them one release at a
 *   time. Checks collect into a report; the report's failures raise a single
 *   error naming all of them.
 * - **A vague capability scope is a review trigger, not a failure.** ADR-294
 *   §8 makes unrestricted filesystem or network access *require human review*;
 *   it does not make the package invalid. Pack records the trigger on the
 *   manifest and lowers the security profile, and the release still packs.
 *
 * The validation list itself lives in `./pack-checks`; this module runs it and
 * turns the result into registry objects.
 */

import { basename } from 'node:path';

import { compatibilityMatrixRevision } from './compat';
import { sha256Canonical } from './hash';
import type { SoftwareInventory } from './inventory';
import {
  assertNoFailures,
  capabilityPolicyCheck,
  executableSegmentCheck,
  externalServicesCheck,
  licenseCheck,
  memoryRequirementsCheck,
  modelProvenanceCheck,
  runtimeCompatibilityCheck,
  signatureCheck,
  structureCheck,
  type CheckStatus,
  type PackCheck,
} from './pack-checks';
import {
  REVIEW_TRIGGER_CLASSES,
  VAGUE_SCOPES,
  publisherIdFor,
  type CapabilityRequest,
  type RvforgeProject,
} from './project';
import { objectId, type CapabilityManifest, type Release } from './registry';
import { validateRvf, type ValidateResult } from './validate';
import { FORGE_PACKAGE_NAME, forgeVersion } from './version';

export type { CheckStatus, PackCheck } from './pack-checks';

/** How much scrutiny the declared contract earns (rendered by `publish`). */
export type SecurityProfile = 'restricted' | 'standard' | 'review-required';

export interface PackOptions {
  rvfPath: string;
  project: RvforgeProject;
  /** Permit unsigned executable segments (development packs). */
  allowUnsigned?: boolean;
  /** Overrides the clock; used by tests for deterministic release ids. */
  publishedAt?: string;
}

export interface PackResult {
  rvfPath: string;
  validation: ValidateResult;
  checks: PackCheck[];
  manualReviewTriggers: string[];
  securityProfile: SecurityProfile;
  capabilityManifest: CapabilityManifest;
  softwareInventory: SoftwareInventory;
  provenance: { id: string; record: PackProvenance };
  /** Draft release: unsigned, `predecessor` unresolved, trust level `published`. */
  release: Release;
  warnings: string[];
}

/** What a pack was decided from; content-addressed into `Release.provenance`. */
export interface PackProvenance {
  schemaVersion: 1;
  type: 'pack-provenance';
  tool: { name: string; version: string };
  rvfIdentity: string;
  rvfSize: number;
  rvfFileId: string;
  compatibilityMatrixRevision: string;
  checks: Array<{ id: string; status: CheckStatus }>;
}

/**
 * Run the pack pipeline.
 *
 * @throws {ForgeError} the first failing check's code, with every failure in
 * its detail bag; `FORGE_E_INVALID_RVF` or `FORGE_E_UNSIGNED_SEGMENT` earlier
 * if the artifact does not parse at all.
 */
export async function runPack(options: PackOptions): Promise<PackResult> {
  const { project } = options;
  const validation = await validateRvf(options.rvfPath, {
    deep: true,
    allowUnsigned: options.allowUnsigned,
  });

  const triggers = manualReviewTriggers(project.capabilities.requests);
  const checks: PackCheck[] = [
    structureCheck(validation),
    signatureCheck(validation, Boolean(options.allowUnsigned)),
    executableSegmentCheck(validation),
    modelProvenanceCheck(project),
    capabilityPolicyCheck(project, triggers),
    runtimeCompatibilityCheck(project),
    memoryRequirementsCheck(project),
    externalServicesCheck(project),
    licenseCheck(project),
  ];

  const inventory = packInventory(project, validation);
  checks.push({
    id: 'software-inventory',
    title: 'Software inventory',
    status: 'pass',
    detail:
      `generated: ${inventory.components.length} component(s), ${inventory.files.length} file(s), ` +
      `${inventory.rvfSegments.count} RVF segment(s)`,
  });

  assertNoFailures(checks, options.rvfPath);

  const capabilityManifest = buildCapabilityManifest(project, triggers);
  const provenanceRecord = packProvenance(validation, checks);
  const provenanceId = objectId(provenanceRecord as unknown as Record<string, unknown>);
  const release = draftRelease({
    project,
    validation,
    capabilityManifestId: capabilityManifest.manifestId,
    softwareInventoryId: objectId(inventory as unknown as Record<string, unknown>),
    provenanceId,
    publishedAt: options.publishedAt ?? new Date().toISOString(),
  });

  return {
    rvfPath: options.rvfPath,
    validation,
    checks,
    manualReviewTriggers: triggers,
    securityProfile: securityProfileFor(project, triggers),
    capabilityManifest,
    softwareInventory: inventory,
    provenance: { id: provenanceId, record: provenanceRecord },
    release,
    warnings: validation.warnings,
  };
}

/**
 * Scopes and classes that force human review (ADR-294 §8).
 *
 * Two sources: a class whose mere request is a trigger (process creation,
 * inter-agent delegation, physical device control), and a scope too broad to
 * describe concretely — which is the same condition requirements P6 bans from
 * the install UX, detected here rather than at render time.
 */
export function manualReviewTriggers(requests: readonly CapabilityRequest[]): string[] {
  const triggers = new Set<string>();
  for (const request of requests) {
    if ((REVIEW_TRIGGER_CLASSES as readonly string[]).includes(request.class)) {
      triggers.add(`class:${request.class}`);
    }
    const scope = request.scope.trim().toLowerCase();
    if (VAGUE_SCOPES.includes(scope) || scope.startsWith('all-') || scope === '**') {
      triggers.add(`vague-scope:${request.class}:${request.scope.trim()}`);
    }
  }
  return [...triggers].sort();
}

/** SHA256 over the canonical capability contract — the policy's identity. */
export const capabilityContractHash = (project: RvforgeProject): string =>
  sha256Canonical({
    defaultPolicy: project.capabilities.defaultPolicy,
    denials: [...project.capabilities.denials].sort(),
    requests: [...project.capabilities.requests].sort((a, b) => a.class.localeCompare(b.class)),
  });

/**
 * The software inventory a pack produces.
 *
 * `bundles` and per-bundle files are empty by construction: pack inspects one
 * `.rvf`, it does not stage a bundle. `forge build` fills those in.
 */
export function packInventory(project: RvforgeProject, validation: ValidateResult): SoftwareInventory {
  return {
    schemaVersion: 1,
    components: [
      { name: FORGE_PACKAGE_NAME, version: forgeVersion(), role: 'packer' },
      {
        name: project.listing.name,
        version: project.version,
        role: 'payload',
        sha256: validation.identity.sha256,
        sizeBytes: validation.size,
      },
      {
        name: 'rvm',
        version: project.runtimeRequirements.rvmVersionMin,
        role: 'runtime',
        profile: project.runtimeRequirements.profiles.join(','),
      },
    ],
    bundles: [],
    files: [
      {
        path: basename(validation.path),
        sha256: validation.identity.sha256,
        size: validation.size,
        role: 'rvf',
      },
    ],
    rvfSegments: {
      count: validation.segments.count,
      executableCount: validation.segments.executableCount,
      byType: validation.segments.byType,
    },
    capabilityPolicyHash: capabilityContractHash(project),
    compatibilityMatrixRevision: compatibilityMatrixRevision(),
  };
}

/** Assemble the unsigned capability manifest with its content id filled in. */
export function buildCapabilityManifest(
  project: RvforgeProject,
  triggers: readonly string[],
): CapabilityManifest {
  const unidentified = {
    schemaVersion: 1 as const,
    type: 'capability-manifest' as const,
    defaultPolicy: 'deny' as const,
    requests: [...project.capabilities.requests]
      .map((r) => ({ class: r.class, scope: r.scope.trim(), rationale: r.rationale.trim() }))
      .sort((a, b) => `${a.class}:${a.scope}`.localeCompare(`${b.class}:${b.scope}`)),
    denials: [...project.capabilities.denials].sort(),
    manualReviewTriggers: [...triggers].sort(),
    signatures: [],
  };
  return { ...unidentified, manifestId: objectId(unidentified) };
}

function packProvenance(validation: ValidateResult, checks: readonly PackCheck[]): PackProvenance {
  return {
    schemaVersion: 1,
    type: 'pack-provenance',
    tool: { name: FORGE_PACKAGE_NAME, version: forgeVersion() },
    rvfIdentity: `sha256:${validation.identity.sha256}`,
    rvfSize: validation.size,
    rvfFileId: validation.identity.fileId,
    compatibilityMatrixRevision: compatibilityMatrixRevision(),
    checks: checks.map((check) => ({ id: check.id, status: check.status })),
  };
}

function draftRelease(input: {
  project: RvforgeProject;
  validation: ValidateResult;
  capabilityManifestId: string;
  softwareInventoryId: string;
  provenanceId: string;
  publishedAt: string;
}): Release {
  const { project, validation } = input;
  const unidentified = {
    schemaVersion: 1 as const,
    type: 'release' as const,
    package: { name: project.listing.name, publisherId: publisherIdFor(project.publisher) },
    version: project.version,
    // Resolved against the registry's package head at publish time; a draft
    // has no registry to resolve against.
    predecessor: null,
    rvfIdentity: `sha256:${validation.identity.sha256}`,
    rvfSize: validation.size,
    capabilityManifest: input.capabilityManifestId,
    runtimeProfiles: [...project.runtimeRequirements.profiles].sort(),
    compatibility: {
      rvmVersionMin: project.runtimeRequirements.rvmVersionMin,
      stateSchemaVersion: project.runtimeRequirements.stateSchemaVersion,
      witnessSchemaVersion: project.runtimeRequirements.witnessSchemaVersion,
    },
    modelManifest: {
      location: project.modelManifest.location,
      digests: [...project.modelManifest.digests].sort(),
    },
    softwareInventory: input.softwareInventoryId,
    // Both require the quarantined runtime; pack never invents a report it
    // did not produce (ADR-294 §7: trust levels name evidence).
    evaluationReport: null,
    securityReport: null,
    provenance: input.provenanceId,
    trustLevel: 'published' as const,
    rollbackSafeUntilStateSchema: project.state.rollbackSafeUntilStateSchema,
    listing: {
      category: project.listing.category,
      description: project.listing.description,
      priceModel: project.listing.priceModel,
    },
    publishedAt: input.publishedAt,
    signatures: [],
  };
  return { ...unidentified, releaseId: objectId(unidentified) };
}

/** Security profile the store renders, derived from the declared contract. */
export function securityProfileFor(
  project: RvforgeProject,
  triggers: readonly string[],
): SecurityProfile {
  if (triggers.length > 0) return 'review-required';
  const requestsNetwork = project.capabilities.requests.some((r) => r.class === 'network');
  return requestsNetwork ? 'standard' : 'restricted';
}
