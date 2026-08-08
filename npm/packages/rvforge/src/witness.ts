/**
 * Witness receipts — required output §3.11, registry data model v0.1.
 *
 * A receipt records that something happened to a subject: a build produced it,
 * a verification checked it. Receipts hash-chain per subject through
 * `prevReceipt`, so the sequence of events on one RVF identity is a chain
 * whose links cannot be reordered, removed, or edited without the recomputed
 * ids ceasing to match.
 *
 * Identity follows the registry rules: an object's id is `sha256:<hex>` over
 * its canonical JSON. `receiptId` and `signatures` are excluded from that
 * encoding — `receiptId` because a field cannot contain its own hash, and
 * `signatures` because a countersignature added later must not change the id
 * of the thing it signs.
 *
 * Signatures are *slots*. Forge holds signing references, never key material,
 * so it emits an empty `signatures` array for a signing worker to fill.
 */

import { appendFile, readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { canonicalJson, sha256String } from './hash';

export const RECEIPTS_FILENAME = 'receipts.jsonl';

/**
 * Events forge witnesses.
 *
 * `build` and `verify` come from the local pipeline; the rest are the registry
 * data model's vocabulary, carried here so a receipt forge writes into a
 * registry uses the same names a Reader will read.
 *
 * **Deviation from registry-model.md**: `publish` is not in the model's event
 * list, which runs `build|verify|install|update|capability-grant|
 * capability-denial|revocation-check`. Appending a release is an event worth a
 * receipt and none of those names fits it, so forge adds one — to be
 * reconciled with `crates/rvforge-registry` rather than left implicit.
 */
export type WitnessEvent =
  | 'build'
  | 'verify'
  | 'publish'
  | 'install'
  | 'update'
  | 'capability-grant'
  | 'capability-denial'
  | 'revocation-check';
export type WitnessOutcome = 'pass' | 'fail' | 'denied';
export type WitnessActorKind = 'builder' | 'reader' | 'registry' | 'rvm';

export interface WitnessSignature {
  keyId: string;
  role: 'publisher' | 'registry';
  sig: string;
}

export interface WitnessReceipt {
  schemaVersion: 1;
  type: 'witness-receipt';
  /** `sha256:<hex>` over the canonical JSON, less `receiptId` and `signatures`. */
  receiptId: string;
  /** What the receipt is about — the RVF identity, as `sha256:<hex>`. */
  subject: string;
  event: WitnessEvent;
  outcome: WitnessOutcome;
  actor: { kind: WitnessActorKind; id: string };
  evidence: Record<string, string | number | boolean | null>;
  /** RFC 3339 UTC. */
  timestamp: string;
  /** Previous receipt id for this subject, or null for the first. */
  prevReceipt: string | null;
  /** Slots only; forge never signs. */
  signatures: WitnessSignature[];
}

export interface CreateReceiptInput {
  subject: string;
  event: WitnessEvent;
  outcome: WitnessOutcome;
  actor: { kind: WitnessActorKind; id: string };
  evidence?: Record<string, string | number | boolean | null>;
  prevReceipt?: string | null;
  /** Overrides the clock; used by tests for deterministic ids. */
  timestamp?: string;
}

/** Normalise a bare hex digest into the `sha256:<hex>` form ids use. */
export const asSubject = (sha256: string): string =>
  sha256.startsWith('sha256:') ? sha256 : `sha256:${sha256}`;

/** Content id of a receipt: `sha256:<hex>` over its canonical JSON. */
export function receiptContentId(receipt: Omit<WitnessReceipt, 'receiptId'>): string {
  const { signatures: _signatures, ...rest } = receipt;
  return `sha256:${sha256String(canonicalJson(rest))}`;
}

/** Build a receipt with its content id filled in. */
export function createWitnessReceipt(input: CreateReceiptInput): WitnessReceipt {
  const unidentified: Omit<WitnessReceipt, 'receiptId'> = {
    schemaVersion: 1,
    type: 'witness-receipt',
    subject: asSubject(input.subject),
    event: input.event,
    outcome: input.outcome,
    actor: input.actor,
    evidence: input.evidence ?? {},
    timestamp: input.timestamp ?? new Date().toISOString(),
    prevReceipt: input.prevReceipt ?? null,
    signatures: [],
  };
  return { ...unidentified, receiptId: receiptContentId(unidentified) };
}

/** Exact line forge appends to `receipts.jsonl`. */
export const receiptLine = (receipt: WitnessReceipt): string => `${canonicalJson(receipt)}\n`;

/** Read the receipt chain from a build directory; absent file means no chain. */
export async function readReceipts(dir: string): Promise<WitnessReceipt[]> {
  let raw: string;
  try {
    raw = await readFile(join(dir, RECEIPTS_FILENAME), 'utf8');
  } catch {
    return [];
  }
  return raw
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => JSON.parse(line) as WitnessReceipt);
}

/** Id of the newest receipt for `subject`, or null when the chain is empty. */
export function lastReceiptFor(
  receipts: readonly WitnessReceipt[],
  subject: string,
): string | null {
  const wanted = asSubject(subject);
  for (let i = receipts.length - 1; i >= 0; i--) {
    if (receipts[i].subject === wanted) return receipts[i].receiptId;
  }
  return null;
}

/** Append a receipt, chaining it onto whatever is already recorded. */
export async function appendReceipt(dir: string, input: CreateReceiptInput): Promise<WitnessReceipt> {
  const existing = await readReceipts(dir);
  const receipt = createWitnessReceipt({
    ...input,
    prevReceipt: input.prevReceipt ?? lastReceiptFor(existing, input.subject),
  });
  await appendFile(join(dir, RECEIPTS_FILENAME), receiptLine(receipt), 'utf8');
  return receipt;
}

export interface ChainProblem {
  index: number;
  receiptId: string;
  reason: 'id-mismatch' | 'broken-link' | 'orphan-root' | 'malformed';
  detail: string;
}

export interface ChainVerification {
  ok: boolean;
  count: number;
  /** Subjects present, sorted, each with its chain length. */
  subjects: Array<{ subject: string; length: number }>;
  problems: ChainProblem[];
}

/**
 * Verify the hash chain.
 *
 * Two independent checks per receipt: the recorded `receiptId` must equal the
 * id recomputed from the content, and `prevReceipt` must name the previous
 * receipt for the same subject (null for the first). Recomputing the id is
 * what catches an edited receipt; checking the link is what catches a removed
 * or reordered one.
 */
export function verifyReceiptChain(receipts: readonly WitnessReceipt[]): ChainVerification {
  const problems: ChainProblem[] = [];
  const previousBySubject = new Map<string, string>();
  const lengthBySubject = new Map<string, number>();

  receipts.forEach((receipt, index) => {
    const id = receipt?.receiptId ?? '(missing)';
    if (!receipt || receipt.type !== 'witness-receipt' || typeof receipt.subject !== 'string') {
      problems.push({ index, receiptId: id, reason: 'malformed', detail: 'not a witness receipt' });
      return;
    }

    const { receiptId, ...content } = receipt;
    const recomputed = receiptContentId(content);
    if (recomputed !== receiptId) {
      problems.push({
        index,
        receiptId,
        reason: 'id-mismatch',
        detail: `content hashes to ${recomputed}`,
      });
    }

    const expectedPrev = previousBySubject.get(receipt.subject) ?? null;
    if (receipt.prevReceipt !== expectedPrev) {
      problems.push({
        index,
        receiptId,
        reason: expectedPrev === null ? 'orphan-root' : 'broken-link',
        detail: `prevReceipt is ${String(receipt.prevReceipt)}, expected ${String(expectedPrev)}`,
      });
    }

    previousBySubject.set(receipt.subject, receiptId);
    lengthBySubject.set(receipt.subject, (lengthBySubject.get(receipt.subject) ?? 0) + 1);
  });

  return {
    ok: problems.length === 0,
    count: receipts.length,
    subjects: [...lengthBySubject.entries()]
      .map(([subject, length]) => ({ subject, length }))
      .sort((a, b) => a.subject.localeCompare(b.subject)),
    problems,
  };
}
