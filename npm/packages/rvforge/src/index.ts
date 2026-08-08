/**
 * `@ruvector/rvforge` — RVForge programmatic API.
 *
 * The CLI in `./cli` is a thin router over these functions; anything the
 * command line can do is available here for use inside a build pipeline.
 */

export {
  EXIT_CODES,
  ForgeError,
  ForgeErrorCode,
  exitCodeFor,
  redact,
  redactDetails,
  toForgeError,
  type ForgeErrorDetails,
  type ForgeErrorJson,
} from './errors';

export {
  CAPABILITY_CLASSES,
  PACKAGING_MODES,
  RUNTIME_PROFILES,
  SUPPORTED_TARGETS,
  TARGET_ARTIFACTS,
  TARGET_WORKER,
  resolveTarget,
  resolveTargets,
  type AppMetadata,
  type BuildManifest,
  type BuildTarget,
  type CapabilityClass,
  type CapabilityPolicy,
  type ForgeConfig,
  type PackagingMode,
  type ProvenanceFile,
  type ProvenanceRecord,
  type RuntimeContract,
  type RuntimeProfile,
  type RvfIdentityRecord,
  type SigningRef,
  type SigningRefs,
  type WorkerFamily,
} from './types';

export {
  canonicalJson,
  canonicalJsonFile,
  sha256Buffer,
  sha256Canonical,
  sha256File,
  sha256String,
} from './hash';

export {
  EXECUTABLE_SEGMENT_TYPES,
  ROOT_MANIFEST_MAGIC_BYTES,
  ROOT_MANIFEST_SIZE,
  SEGMENT_HEADER_SIZE,
  SEGMENT_MAGIC_BYTES,
  SEGMENT_TYPE_NAMES,
  crc32c,
  parseRootManifest,
  parseSegmentHeader,
  type RvfRootManifest,
  type RvfSegmentHeader,
} from './rvf';

export {
  RVM_COMPATIBILITY_MATRIX,
  buildManifest,
  capabilityPolicyHash,
  defaultCapabilityPolicy,
  manifestFileContents,
  manifestHash,
  normalizeCapabilityPolicy,
  serializeManifest,
  type BuildManifestInput,
} from './manifest';

export {
  CONFIG_FILENAME,
  defaultConfig,
  loadConfig,
  runInit,
  validateConfig,
  type InitOptions,
  type InitResult,
} from './config';

export {
  estimateBuild,
  formatBytes,
  RUNNER_COST_PER_MINUTE,
  type BuildEstimate,
  type EstimateInput,
  type TargetEstimate,
} from './estimate';

export {
  COMPATIBILITY_MATRIX,
  RUNTIME_PROFILE_MATRIX_KEY,
  TARGET_INSTALLER_FORMATS,
  TARGET_PLATFORM,
  assertCompatible,
  checkCompatibility,
  closestSupported,
  compatibilityMatrixRevision,
  type AssertCompatibleInput,
  type CompatCheck,
  type CompatRequest,
  type CompatSuggestion,
  type CompatibilityMatrix,
  type MatrixPlatform,
  type MatrixRuntimeProfile,
  type MatrixStatus,
} from './compat';

export {
  BUNDLES_DIR,
  assertEmbeddedRvfIdentical,
  assertLocator,
  buildLocator,
  stageBundles,
  verifyLocator,
  type BundleEntry,
  type BundleLayout,
  type LocatorCheck,
  type ReaderSlot,
  type RvfLocator,
} from './package-modes';

export {
  buildInventory,
  roleFor,
  type InventoryBundle,
  type InventoryComponent,
  type InventoryEntry,
  type InventoryInput,
  type InventoryRole,
  type SoftwareInventory,
} from './inventory';

export {
  RECEIPTS_FILENAME,
  appendReceipt,
  asSubject,
  createWitnessReceipt,
  lastReceiptFor,
  readReceipts,
  receiptContentId,
  receiptLine,
  verifyReceiptChain,
  type ChainProblem,
  type ChainVerification,
  type CreateReceiptInput,
  type WitnessActorKind,
  type WitnessEvent,
  type WitnessOutcome,
  type WitnessReceipt,
  type WitnessSignature,
} from './witness';

export { validateRvf, type SegmentSummary, type ValidateOptions, type ValidateResult } from './validate';
export {
  CHECKSUMS_FILE,
  INVENTORY_FILE,
  MANIFEST_FILE,
  PROVENANCE_FILE,
  runBuild,
  type BuildOptions,
  type BuildResult,
} from './build';
export { planSubmit, runSubmit, type SubmitOptions, type SubmitPlan, type SubmitResult } from './submit';
export { assertBuildId, runStatus, type StatusOptions, type StatusResult } from './status';
export { runDownload, type DownloadOptions, type DownloadResult } from './download';
export {
  MANIFEST_FILENAME,
  PROVENANCE_FILENAME,
  runVerify,
  type FileCheck,
  type VerifyOptions,
  type VerifyResult,
} from './verify';

export {
  API_TOKEN_ENV,
  API_URL_ENV,
  DEFAULT_API_URL,
  ForgeApiClient,
  type BuildArtifact,
  type BuildRecord,
  type BuildState,
  type ForgeApiOptions,
  type SubmitRequest,
} from './api';

export {
  MANIFEST_CAPABILITY_CLASSES,
  PROJECT_FILENAME,
  REVIEW_TRIGGER_CLASSES,
  VAGUE_SCOPES,
  defaultDenials,
  defaultProject,
  loadProject,
  publisherIdFor,
  validateProject,
  writeProject,
  type CapabilityRequest,
  type ManifestCapabilityClass,
  type ProjectInitResult,
  type ProjectListing,
  type ProjectPublisher,
  type ProjectRuntimeRequirements,
  type RvforgeProject,
} from './project';

export {
  assertKeyFilePermissions,
  generatePublisherKey,
  keyIdFor,
  loadPublisherKey,
  signObjectId,
  verifyObjectId,
  writePublisherKey,
  type PublisherKey,
  type PublisherKeyFile,
} from './keys';

export {
  DEFAULT_REGISTRY_DIR,
  LOG_ENTRIES_FILE,
  PUBLISHER_FILENAME,
  RELEASES_FILENAME,
  TREE_HEAD_FILE,
  WITNESS_DIRNAME,
  appendLogEntry,
  appendRelease,
  appendWitnessReceipt,
  digestOf,
  getObject,
  inclusionProof,
  initRegistry,
  objectId,
  objectPath,
  packagePath,
  publisherPath,
  putObject,
  readLog,
  readReleaseIndex,
  readTreeHead,
  readWitnessChain,
  verifyInclusionProof,
  verifyLog,
  witnessIndexPath,
  type CapabilityManifest,
  type InclusionProof,
  type ObjectSignature,
  type PublisherRecord,
  type Release,
  type ReleaseIndexEntry,
  type RegistryObject,
  type TransparencyLogEntry,
  type TreeHead,
} from './registry';

export {
  inclusionPath,
  leafHash,
  nodeHash,
  root as merkleRoot,
  verifyInclusion,
  type Hash,
} from './merkle';

export {
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

export {
  buildCapabilityManifest,
  capabilityContractHash,
  manualReviewTriggers,
  packInventory,
  runPack,
  securityProfileFor,
  type PackOptions,
  type PackProvenance,
  type PackResult,
  type SecurityProfile,
} from './pack';

export {
  QUARANTINE_REASON,
  runTestAgent,
  type TestAgentOptions,
  type TestAgentResult,
  type TestCategory,
  type TestStatus,
} from './test-agent';

export {
  publishSummary,
  renderPublishSummary,
  runPublish,
  type PublishOptions,
  type PublishResult,
  type PublishSummary,
} from './publish';

export { forgeVersion, FORGE_PACKAGE_NAME } from './version';
