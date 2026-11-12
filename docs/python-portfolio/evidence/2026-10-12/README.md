# Portfolio recovery evidence, 2026-10-12

This repair addresses the incomplete 0.2.0 release before adding another Python
library. It does not rebuild or replace immutable published versions.

## Observed state

The original supervisor stopped after a publication failure. PR1188 merged as
`fccf36e9a76d35bce6750dfd3b25802cae3d683b`. The actual main release
[run38169271248](https://github.com/ruvnet/RuVector/actions/runs/38169271248)
passed build and native audit, then stopped on PyPI HTTP429: too many new projects.

On October12, two original wheels are present and match their frozen SHA256:
attention and cluster0.2.0. Their source archives and nine other complete target
releases remain missing: 20 of22 distribution files are absent. This is not a
completed portfolio release. The checked in inventory records exact registry
filenames and hashes; no token or private customer data is included.

The original runtime is0.1.3. Main now requests0.1.4, so rebuilding0.2.0 from main
would produce different dependency metadata. Recovery uses only the original
archive, artifact11713787141, SHA256
`b4c92cae86c753305abef7aad395fdd9d91c7e84c3c887342defa7a3e4607241`.
The actual downloaded ZIP and all22 extracted file hashes were verified anew.

## Validation

| Check | Observed result |
| --- | --- |
| Recovery adversarial contracts |62 passed on Linux; Windows permission behavior simulated, not a native Windows run |
| Original installed wheel contracts |103 passed, CPython3.12.3 Linuxx86_64 |
| Original README examples |11 passed against installed original artifacts |
| Published attention tutorial |Passed from fresh PyPI installation: attention0.2.0, runtime0.1.3, NumPy2.2.6 |
| Workflow validation |actionlint1.7.7 passed; official release binary checksum verified; external shellcheck/pyflakes integrations disabled |
| Python dependency audit |pip-audit2.10.1 reported no known issues in resolved dependencies; nine unpublished companion distributions could not be looked up |
| Read-only source scan |Pinned Ruflo3.25.6 reported zero findings for the recovery helper; no advisory feed timestamp supplied |
| Secret scan |detect-secrets1.5.0 flagged four public provenance hashes, manually confirmed false positives; no credentials present |
| Independent review |Approved exact helper and workflow after correcting Windows mode handling and artifact download directory layout |
| Live registry preflight |Two existing hashes match,20 missing original files selected, zero uploads |

The native audit from the original run is preserved as historical evidence.
Recovery additionally gates upload on a fresh frozen-source native audit, fresh
original artifact installation on Linux/macOS/Windows, and dependency checks.
Hosted platform gates and post-publication PyPI consumer tests remain to run.

## Failure and promotion policy

PR and default dispatch checks cannot publish. An explicit main dispatch with
`publish=true` enters the existing protected pypi environment only after
platform and audit gates pass. Ordinary and recovery portfolio publishers share
one concurrency group. Under that lock the registry is checked again; existing
matching files are skipped by selection, never by trusting a duplicate-upload
error. Hash conflicts, yanked files, metadata mismatches, unsafe paths and registry
errors stop selection.

There is no established upload429 cooldown in official guidance. The24hour rule
in [Trusted Publisher guidance](https://docs.pypi.org/trusted-publishers/troubleshooting/)
concerns publisher registration, not this upload error. No rapid retry, alternate
account or IP workaround is configured. A failed upload leaves a partial registry
state and requires a new read-only inventory before another bounded attempt.

Disabling the recovery entry point rolls back execution without deleting or
overwriting published versions. The exact file ledger and receipts stay available.
Windows relies on the trusted hosted runner ACL rather than POSIX permission bits.
Checksums detect changes; they do not replace protected source review.

## Candidate decision

[candidate-decision.json](candidate-decision.json) records the requested weighted
ranking and fresh npm window September12 through October11. Filtered retrieval
is the leading future extension of an existing package; no new library is
selected while the original release is incomplete. Counts are downloads, not
unique users, and usefulness/readiness scores are engineering estimates.

Public tested tutorial: https://gist.github.com/ruvnet/2491c7631bf3e186e815196f52fd8954

The posted code was fetched back from GitHub, matched byte for byte and ran successfully. See tutorial-ledger.json for its immutable revision and code digest.
