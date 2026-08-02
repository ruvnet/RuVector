# ADR-009: RVF Version 1 Wire Contract

| Field | Value |
|---|---|
| Status | Accepted |
| Date | 2026-08-02 |
| Authors | RuVector Architecture Team |
| Reviewers | Repository maintainers |
| Supersedes | Wire layout sections of ADR-004-rvf-format and ADR-005-rvf-cognitive-container |
| Related | RVF research specification, rvf-types, rvf-wire, rvf-manifest |

## 1. Context

Two mutually incompatible binary layouts have both been described as "RVF" in
this repository.

The first comes from ADR-004 (`docs/architecture/decisions/ADR-004-rvf-format.md`)
and ADR-005 (`ADR-005-rvf-cognitive-container.md`). Those documents describe a
container that begins with a fixed 64-byte file header at offset zero, from
which a reader learns the file's identity, version, and the location of
everything else. Under that model, parsing starts at byte zero and proceeds
forward.

The second is the layout that the shipped crates actually implement. An RVF
file is an append-only stream of independently verifiable segments, each
beginning with its own 64-byte header and each starting on a 64-byte boundary.
There is no file-level header at offset zero at all. Instead, the file's
identity and directory live in the newest MANIFEST_SEG, and a reader finds that
segment by inspecting the file's *tail*: the Level-0 root manifest is exactly
4096 bytes and occupies the final 4096 bytes of the newest manifest payload.
This is what `rvf_wire::find_latest_manifest` does, and it is the layout every
RVF artifact in the wild already uses.

A second, subtler ambiguity compounds the first. The format's magic values are
documented by their mnemonics, "RVFS" for segments and "RVM0" for the root
manifest. Those mnemonics are the *big-endian* rendering of the numeric
constants `SEGMENT_MAGIC = 0x52564653` and `ROOT_MANIFEST_MAGIC = 0x52564D30`.
RVF serializes every multi-byte integer little-endian, so the four bytes that
actually appear at the start of a segment are `53 46 56 52`, and the four bytes
at the start of a root manifest are `30 4D 56 52`. Neither matches the ASCII
spelling of its own mnemonic. Documentation and pseudocode that compare
`header[0:4]` against `b'RVFS'` describe a reader that would reject every real
RVF file.

The risk in leaving this unresolved is not merely editorial. Someone reading the
old ADRs or the old pseudocode could reasonably conclude the shipped writers are
buggy and "fix" them to emit big-endian magic or to prepend a header at offset
zero. Either change would make every existing artifact unreadable, and because
segment content hashes and manifest signatures cover these bytes, it would also
invalidate signatures over data that had not otherwise changed. The wire format
is a compatibility surface with deployed readers, and it needs to be treated as
one.

## 2. Decision

### 2.1 Canonical file structure

RVF version 1 has no fixed header at offset zero. A v1 file is a stream of
independently verifiable segments, each beginning on a 64-byte-aligned boundary
with its own 64-byte segment header. Segments are appended; earlier bytes are
never rewritten in place.

The latest valid MANIFEST_SEG is the single source of truth for the file's
contents. Older manifests remain in the file and remain parseable, but a reader
that finds a newer valid manifest must prefer it.

The Level-0 root manifest is exactly 4096 bytes — one OS page — and occupies the
final 4096 bytes of the latest manifest segment's payload. Readers therefore
discover a file by inspecting its tail: read the last 4096 bytes, check for the
root manifest magic, and verify the trailing CRC32C. If that fast path fails,
fall back to scanning backward from the end of the file at 64-byte boundaries,
looking for a segment header whose type is MANIFEST_SEG. A reader must not
require, and must not assume the presence of, any structure at offset zero.

### 2.2 Canonical byte order and magic values

All multi-byte integers in the RVF v1 wire format are little-endian unless a
field is explicitly documented otherwise.

| Purpose | Numeric constant | Mnemonic (big-endian rendering) | Wire bytes (little-endian) |
|---|---|---|---|
| Segment header | `0x52564653` | `RVFS` | `53 46 56 52` (reads as "SFVR") |
| Level-0 root manifest | `0x52564D30` | `RVM0` | `30 4D 56 52` (reads as "0MVR") |

The mnemonic is a naming convention for humans. The wire bytes are the contract.
Any code that compares raw bytes must use the exported constants
`rvf_types::SEGMENT_MAGIC_BYTES` and `rvf_types::ROOT_MANIFEST_MAGIC_BYTES`
rather than a hand-written ASCII literal. Code that compares the parsed `u32`
must use `SEGMENT_MAGIC` / `ROOT_MANIFEST_MAGIC` and must decode with
`u32::from_le_bytes`.

### 2.3 Version stability

Version 1 writers must continue to emit exactly the bytes that are already
deployed. Changing any literal byte of the v1 wire format — the magic values,
the field order within the segment header, the size or placement of the root
manifest, the alignment rule — is a new format version, not a fix.

Introducing a format version 2 requires, before any v2 writer ships:

- a version discriminator that a v1 reader can detect and reject cleanly rather
  than misparse;
- dual-version readers that accept both v1 and v2 artifacts;
- golden byte vectors committed for both versions, so neither can drift silently;
- an explicit statement of signature and content-hash compatibility, covering
  whether v1 signatures remain verifiable over migrated data and, if not, how
  artifacts are re-signed.

### 2.4 Normative sources

The normative description of the RVF v1 wire format is, in order of precedence:

1. This ADR.
2. `docs/research/rvf/wire/binary-layout.md`.
3. The exported constants and codecs in the `rvf-types`, `rvf-wire`, and
   `rvf-manifest` crates.
4. The golden byte-vector tests in `crates/rvf/rvf-wire/tests/wire_contract_golden.rs`
   and the constant tests in `crates/rvf/rvf-types/src/constants.rs`.

The offset-zero header diagrams in ADR-004 and ADR-005 are historical records of
a design that was not shipped. They must not be used to implement a reader or a
writer.

## 3. Rationale

Codifying the shipped behavior, rather than migrating the shipped behavior to
match the older ADRs, is the cheaper and safer direction for three reasons.

The tail-discovered manifest is what makes append-only writing work. A file
header at offset zero would have to be rewritten on every commit to point at the
new manifest, which reintroduces in-place mutation, torn-write windows, and a
single point of corruption that invalidates the entire file. Discovering the
manifest from the tail means a crash mid-append leaves the previous manifest
intact and still authoritative; recovery is "scan backward until something
verifies," which degrades gracefully rather than failing absolutely.

Little-endian throughout matches the hardware every RVF reader runs on, so
headers can be read by direct load rather than byte-swapping, and 64-byte
segment alignment matches both the AVX-512 register width and the cache line.
Reversing the magic to make it "read nicely" in a hex dump would buy readability
for humans and cost a byte-swap on the hot path for machines, in a format whose
whole point is zero-copy access.

Freezing the bytes is what makes the format a contract at all. Content hashes and
manifest signatures are computed over these exact bytes. A change that looks
cosmetic — reversing four magic bytes — silently invalidates every signature in
every existing artifact, with no error message that points at the cause. The
constants are cheap to keep and expensive to change, so we keep them.

## 4. Consequences

**Positive.** There is now a single normative answer to "what does an RVF file
look like," and it matches what the code does, so a new contributor reading the
docs and a new contributor reading the crates arrive at the same place. The
exported `*_MAGIC_BYTES` constants remove the recurring endianness mistake from
the class of bugs that can be written. Golden vectors turn an accidental wire
change from a silent compatibility break into a failing test in CI.

**Negative.** The older ADRs remain in the tree with layout sections that are now
explicitly historical, which is a small ongoing source of confusion for anyone
who reads them without the superseding note. The mnemonic-versus-wire-bytes
distinction is genuinely counterintuitive and will keep needing to be explained;
we accept that cost in exchange for not breaking deployed artifacts.

**Neutral.** The golden vectors pin `Level0Root::default()` and the canonical
empty segment header specifically. Fields that a default root leaves zeroed are
covered by the trailing CRC32C but are not independently pinned; extending
coverage to populated manifests is future work, not a gap in the contract.

## 5. Security requirements

Magic values are structural sentinels. They tell a reader where a record
plausibly begins; they carry no authority whatsoever. Finding `53 46 56 52` at a
64-byte boundary means "a segment header may start here," not "this is a
trustworthy segment."

Before acting on any segment, a reader must validate, in this order: the format
version (rejecting unsupported versions rather than guessing); the segment type
(unknown types are preserved but not interpreted); every declared length against
the actual remaining bytes, so a declared payload length can never induce a read
past the end of the mapping; the 64-byte alignment of the segment start; the
content hash over the payload, compared in constant time; and, where the segment
carries one, the signature chain.

For the Level-0 root manifest specifically, the CRC32C at offset `0xFFC` covers
bytes `0x000..0xFFC` and must be verified before any offset or length in the
manifest is dereferenced. CRC32C is an integrity check against corruption, not
an authentication mechanism; it must never be treated as evidence of
authenticity.

No executable payload — embedded kernel images, eBPF programs, or WASM modules
carried in KERNEL, EBPF, or profile segments — may be activated on the strength
of a matching magic value and a matching content hash alone. Activation requires
a verified signature from a trusted key and an explicit policy decision by the
host. A content hash proves the bytes are the bytes the writer wrote; it says
nothing about whether the writer was authorized.

## 6. Acceptance criteria

This ADR is satisfied when all of the following hold in CI:

1. A golden byte-vector test serializes the canonical empty segment — a META
   segment with an empty payload, segment id 0, no flags, and a SHAKE-256
   content hash — and asserts the full 64-byte array. Its first four bytes are
   `53 46 56 52`, and bytes `0x28..0x38` are the NIST-published SHAKE-256 value
   for empty input truncated to 128 bits, `46b9dd2b0ba88d13233b3feb743eeb24`.

2. A golden byte-vector test serializes the default Level-0 root manifest and
   asserts that it is exactly 4096 bytes, that it begins
   `30 4D 56 52 01 00 00 00`, and that its trailing CRC32C at offset `0xFFC` is
   `FF DD 18 14`.

3. A tail-scanning test builds an RVF byte stream through the public writer API,
   writes it to disk, and locates the root manifest through the reader's tail
   discovery — while asserting that offset zero holds an ordinary segment and is
   not parseable as a root manifest, demonstrating that no fixed offset-zero
   header is required.

4. `rvf_types::SEGMENT_MAGIC_BYTES` and `rvf_types::ROOT_MANIFEST_MAGIC_BYTES`
   are exported, and their tests assert both the numeric constant and the exact
   little-endian byte array, including that the byte array is *not* equal to the
   ASCII mnemonic.

5. No documentation or pseudocode in the tree compares v1 wire bytes against a
   literal ASCII string such as `b'RVFS'` or `b'RVM0'`; such comparisons use the
   exact byte sequences or the exported constants.

6. The wire-layout sections of ADR-004 and ADR-005 carry a prominent note
   marking them superseded by this ADR.

## 7. Revision history

| Date | Change |
|---|---|
| 2026-08-02 | Codified the shipped append-only segment stream and tail-discovered manifest as the normative RVF v1 wire contract; exported exact magic byte constants; added golden byte vectors; superseded the wire-layout sections of ADR-004 and ADR-005. |
