# Python portfolio recovery evidence — 2026-10-13

Owner: Codex /root. PR: https://github.com/ruvnet/RuVector/pull/1205.
Repair the existing interrupted 0.2.0 release before selecting a new package.

## Immutable inputs

Source: fccf36e9a76d35bce6750dfd3b25802cae3d683b.
Run: 38169271248; artifact: 11713787141.
Artifact SHA256: b4c92cae86c753305abef7aad395fdd9d91c7e84c3c887342defa7a3e4607241.
Inventory SHA256: ca4c4b47df73a26619c048ed1eb02013bfd96484aeb88e20fe16464ef1e0a32d.

At 11:25 UTC only attention and cluster 0.2.0 wheels matched the 22-file
inventory on PyPI; 20 files were missing. No upload was attempted during these
repairs. The native dependency remains exactly ruvector 0.1.3.
This checkpoint does not claim publication.

## Repair evidence

- Run https://github.com/ruvnet/RuVector/actions/runs/38390477781:
  62 recovery contracts, frozen native audit, Linux and macOS installed-artifact
  acceptance passed. Windows stopped on inventory hash mismatch.
- A scoped LF checkout rule preserves the byte-pinned inventory. A clean Git
  index checkout with core.autocrlf=true retained its exact SHA256. The strict
  hash check remains unchanged.
- PRs now run all three preparation platforms and the frozen audit. Publication
  still requires explicit main dispatch, passing prerequisites and protected
  pypi environment.
- Seven targeted npm upgrades clear npm audit. Seven functional consumer groups
  pass against a clean 144-record affected dependency closure. Reproduce with
  npm/tests/prepare-security-consumer.py and security-dependency-consumer.cjs.
  Full workspace installation is not claimed: unrelated lock drift remains.
- Quinn 0.11.18 replaces 0.11.15 for five RustSec advisories with only required
  RNG changes. The original SDK all-target normal/build closure contains
  neither Quinn nor Reqwest. Local Cargo 1.90 hit an internal resolver panic
  before compilation; hosted current-stable checks remain required.
- Replication releases its DashMap guard before accessing another entry.
  All 27 tests pass; the new same-shard regression fails within five seconds
  against the original implementation.
- Server readiness follows socket binding. The strict test retains its child
  stdout pipe through requests. All six tests pass; the affected HTTP case
  passed 50 fresh-process runs.
- Only core-and-rest CI moves to Ubuntu 24.04 for ORT ABI requirements.
  Hosted whole-shard validation remains required.

An independent review agent approved these code, dependency, checkout and
workflow changes. No assertions or release protections were weakened.
Additional failures must be repaired and reviewed before promotion.

## Additional gate repair

Head fbf30da67aa3da87db6c8c34a6124f5a499d904d passed Windows
recovery preflight and both hosted Cargo vulnerability audits.
The remaining cargo-deny failure identified unmaintained unsafe-libyaml.

Only rvagent-middleware used serde_yaml directly. It now uses pinned
serde-saphyr 1.3.0 (MIT OR Apache-2.0) with only deserialization enabled.
The crate explicitly declares Rust 1.89; the frozen Python SDK is unaffected.
A private visitor retains string-only metadata without numeric-key coercion.
Existing 4KiB frontmatter bounds are supplemented by depth, alias, node,
event and scalar expansion limits. The exact-source parser harness passed
18 existing and five new compatibility/adversarial tests; this excludes
three middleware integration tests, so full hosted crate CI remains required.
An independent review approved the parser, feature restrictions and MSRV
declaration. No advisory exception was added.

## Promotion and rollback

Do not merge or publish with failed or pending required checks. Reconcile PyPI
under the publication lock; upload only byte-identical missing originals,
stop on any failure and never blindly retry an uncertain upload. Then verify
all 22 hashes and fresh PyPI-only installation on Linux, macOS and Windows.
Existing tutorial: https://gist.github.com/ruvnet/2491c7631bf3e186e815196f52fd8954.

Code rollback is a reviewed revert of PR1205. PyPI files cannot be overwritten.
On post-release failure stop promotion, assess yanking with repository owners,
and release a separately reviewed patch version. Never replace frozen 0.2.0
files with rebuilt artifacts.
