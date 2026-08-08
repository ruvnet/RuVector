# ADR-009: RVF Version 1 Wire Contract

| Field | Value |
|---|---|
| Status | Accepted |
| Date | 2026-08-02 |
| Authors | RuVector Architecture Team |
| Reviewers | Repository maintainers |
| Supersedes | Wire layout sections of ADR-004-rvf-format and ADR-005-rvf-cognitive-container |
| Related | RVF research specification, rvf-types, rvf-wire, rvf-manifest, rvf-runtime |

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
beginning with its own 64-byte header. There is no file-level header at offset
zero at all. Instead, the file's identity and directory live in the newest
MANIFEST_SEG, and a reader finds that segment by inspecting the file's *tail*.

Within that second layout there are two variants, and the original text of this
ADR described only one of them. `rvf-wire` and `rvf-manifest` pad every segment
to a 64-byte boundary and can emit a 4096-byte Level-0 root manifest at the tail;
`rvf-runtime` — the crate behind `rvf-cli`, and therefore the writer that
produces most RVF files that exist — pads nothing and emits no root manifest at
all. `rvf-runtime` depends on neither `rvf-wire` nor `rvf-manifest`; the two
paths were built separately and never reconciled. Describing the wire variant as
though it were the whole format made this ADR wrong about the files people
actually have, which §2.1 now corrects.

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
independently verifiable segments, each beginning with its own 64-byte segment
header. Segments are appended; earlier bytes are never rewritten in place.

The latest valid MANIFEST_SEG is the single source of truth for the file's
contents. Older manifests remain in the file and remain parseable, but a reader
that finds a newer valid manifest must prefer it. Because the stream is
append-only, "latest" means highest byte offset.

**Two container shapes exist in v1, and both are conformant.** They are produced
by two independent writer paths that share the segment header layout and the
magic values but differ in padding, content-hash algorithm, and whether a
Level-0 root manifest is emitted at all.

*Runtime containers* are what the `rvf-runtime` crate writes, and therefore what
the `rvf` CLI produces. Segments are packed back to back with no padding: each
segment begins wherever the previous one ended, so in general only the segment at
offset zero is 64-byte aligned. The header's `alignment_pad` field is zero. There
is no Level-0 root manifest anywhere in the file. Content hashes use the
runtime's legacy CRC32-rotation hash, labelled `checksum_algo = 0`.

*Wire containers* are what the `rvf-wire` and `rvf-manifest` crates write. Every
segment payload is zero-padded to the next 64-byte boundary and the pad length is
recorded in `alignment_pad`, so every segment does start on a 64-byte boundary.
Content hashes use the rvf-wire algorithm registry (`1` = XXH3-128,
`2` = SHAKE-256/128). A wire container may additionally carry the Level-0 root
manifest.

The Level-0 root manifest is **optional in v1**. Where it is present it is
exactly 4096 bytes — one OS page — and occupies the final 4096 bytes of the
latest manifest segment's payload. A container shorter than 4096 bytes cannot
carry one by construction: a freshly created store is 162 bytes, and a store of
24 sixteen-dimension vectors is 2304 bytes, so small containers never have one
regardless of which writer produced them.

A conformant reader must therefore implement both discovery paths:

1. **Root-manifest fast path.** If the file is at least 4096 bytes long, read the
   final 4096 bytes and check for the root manifest magic `30 4D 56 52` and a
   valid trailing CRC32C at offset `0xFFC`. If both hold, use it. Skip this step
   entirely for files shorter than 4096 bytes rather than seeking to a negative
   offset.
2. **Manifest-segment fallback.** Otherwise — and whenever the fast path fails —
   scan backward from the end of the file for a segment header whose magic is
   `53 46 56 52` and whose type byte is MANIFEST_SEG (`0x05`), and take the
   highest-offset candidate that parses.

The backward scan must step **one byte at a time, not 64 bytes at a time**.
A 64-byte stride finds manifests only in wire containers. In a runtime container
it steps straight past the newest manifest, and if an older manifest happens to
sit at offset zero — which it does in every store that has been written to more
than once — the scan silently returns that stale manifest instead of failing.
The reader then reports the store's first epoch, typically zero vectors, with no
error. A wrong answer that looks like a right one is the worst available failure
mode, and it is the one a 64-byte stride produces.

Because a byte-wise scan can match magic bytes occurring inside a payload, a
candidate is not a manifest until it parses. Reject a declared payload length
that overflows or extends past the end of the file, reject a segment directory
whose declared entry count cannot fit within the declared payload, and only then
treat the record as authoritative.

A reader must not require, and must not assume the presence of, any structure at
offset zero, a Level-0 root manifest anywhere in the file, or 64-byte alignment
of any segment other than the first.

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
   `rvf-manifest` crates, which define the wire-container shape and the Level-0
   root manifest codec.
4. The writer and reader in the `rvf-runtime` crate — `store.rs`,
   `write_path.rs`, and `read_path.rs` — which define the runtime-container
   shape. `rvf-runtime` is the writer behind `rvf-cli` and therefore the origin
   of most RVF artifacts in existence. Any description of v1 that omits it does
   not describe the files people actually have; omitting it here is what made
   earlier revisions of this ADR misleading.
5. The golden byte-vector tests in `crates/rvf/rvf-wire/tests/wire_contract_golden.rs`
   and the constant tests in `crates/rvf/rvf-types/src/constants.rs`.

Sources 3 and 4 disagree on segment padding and on the `checksum_algo` registry.
That disagreement is a fact about v1, not an error to be resolved by preferring
one source: each is normative for the containers it produces, and a reader that
handles only one of them is incomplete. Unifying the two paths would be a
format-version change under §2.3, not a fix.

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

The larger cost is that v1 is now documented as two container shapes rather than
one, and every reader has to carry both discovery paths. That is a real tax on
implementors, and it is the honest description of what shipped. Recording it is
strictly better than the alternative this ADR previously chose, which was to
document the shape one writer produced and leave implementors to discover the
other by having their reader return an empty store.

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
past the end of the mapping; the content hash over the payload, computed with the
algorithm named by `checksum_algo` and compared in constant time; and, where the
segment carries one, the signature chain.

Segment alignment must not be used as a validation criterion. Runtime containers
pack segments without padding (§2.1), so rejecting an unaligned segment start
rejects valid data. Alignment is a performance property of wire containers, not a
security property of the format.

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
   `FF DD 18 14`. This pins the root manifest codec wherever a root manifest is
   written; it does not assert that every container carries one.

3. For **wire containers**, a tail-scanning test builds an RVF byte stream
   through the `rvf-wire` writer API, writes it to disk, and locates the root
   manifest through the reader's tail discovery — while asserting that offset
   zero holds an ordinary segment and is not parseable as a root manifest,
   demonstrating that no fixed offset-zero header is required.
   (`rvf-wire/tests/wire_contract_golden.rs::root_manifest_is_discovered_from_the_tail_without_an_offset_zero_header`.)

3a. For **runtime containers**, reopen tests demonstrate that a store written by
   `rvf-runtime` — which carries no root manifest and no segment padding — is
   recovered through the byte-wise backward manifest-segment scan, including
   when the newest manifest lies beyond the initial tail-scan window.
   (`rvf-runtime/src/store.rs::reopen_with_manifest_beyond_64kb_tail_window`,
   and the lifecycle tests in `crates/rvf/tests/rvf-integration/tests/`
   `runtime_lifecycle.rs`, `e2e_store_lifecycle.rs`, and `rvf_cli_smoke.rs`.)
   No test may assert that a runtime-produced container contains the root
   manifest magic or that its segments are 64-byte aligned; neither is true.

4. `rvf_types::SEGMENT_MAGIC_BYTES` and `rvf_types::ROOT_MANIFEST_MAGIC_BYTES`
   are exported, and their tests assert both the numeric constant and the exact
   little-endian byte array, including that the byte array is *not* equal to the
   ASCII mnemonic.

5. No documentation or pseudocode in the tree compares v1 wire bytes against a
   literal ASCII string such as `b'RVFS'` or `b'RVM0'`; such comparisons use the
   exact byte sequences or the exported constants.

6. The wire-layout sections of ADR-004 and ADR-005 carry a prominent note
   marking them superseded by this ADR.

7. No documentation describing the RVF v1 container layout — this ADR,
   `docs/research/rvf/wire/binary-layout.md`, and `docs/research/rvf/spec/` —
   presents the Level-0 root manifest as mandatory or universally present, and
   no v1 reader pseudocode in those documents scans backward at a 64-byte stride
   or seeks to `EOF - 4096` without first checking that the file is at least
   4096 bytes long.

## 7. Revision history

| Date | Change |
|---|---|
| 2026-08-02 | Codified the shipped append-only segment stream and tail-discovered manifest as the normative RVF v1 wire contract; exported exact magic byte constants; added golden byte vectors; superseded the wire-layout sections of ADR-004 and ADR-005. |
| 2026-08-03 | Corrected §2.1 to describe both v1 container shapes: `rvf-runtime` containers, which carry no Level-0 root manifest and no segment padding, and `rvf-wire`/`rvf-manifest` containers, which carry both. Made the root manifest explicitly optional, noted that a container under 4096 bytes cannot carry one, and specified the two-path reader algorithm — including that the manifest-segment fallback must scan byte-wise rather than at a 64-byte stride, because a 64-byte stride silently returns a stale manifest in runtime containers. Removed segment alignment as a reader validation criterion (§5). Added `rvf-runtime` to the normative sources (§2.4). Split acceptance criterion 3 so it is true of each writer path, and added criteria 3a and 7. Raised by an external implementor building an independent RVF parser for the rvQR project against this ADR alone (issue #775): their reader found zero occurrences of the root manifest magic in a CLI-produced container. The spec's claims were only testable once someone implemented against it from outside, which is how the gap surfaced. |
