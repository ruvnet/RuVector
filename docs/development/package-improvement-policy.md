# Bounded package improvement

This policy supplements, and does not replace, repository instructions in
`CLAUDE.md` and `docs/development/CONTRIBUTING.md`. It grants no release authority.

1. Verify the package name, source directory, version, public API, dependency
   ranges, supported platforms and consumer before changing source. Keep unknowns
   explicit; registry versions and current source versions may differ.
2. Use one owner and an isolated worktree with an expiring lease. Record the
   source commit, fixed reference and last accepted baseline. Respect the agreed
   time, retry, compute and external-write budgets.
3. Choose a real defect or measured bottleneck. Correctness fixes require an
   executable reproducer; quantitative optimization claims require at least a
   10% measured improvement and zero applicable regressions. Compare identical
   cache, script, platform and dependency-version policies.
4. Test actual public consumer behavior, not only contracts or graph annotations.
   Persistence claims require write, process exit, restart and retrieval. Distinguish
   graceful restart from crash durability, and mocked from native execution.
5. Preserve acceptance tests and independent evaluation. Record raw results,
   source and artifact hashes, failures, rollback instructions and later retained
   revalidation. An unchanged pass is not a new improvement.
6. Submit a narrow reviewable PR. Existing maintainer review, CI, signing and
   release gates remain authoritative. Never auto-merge, publish or bypass gates
   as a consequence of a passing loop. Never manufacture dependencies or downloads.
7. Record validated, non-sensitive findings in the authorized project memory and
   verify retrieval. Memory is evidence to recheck, not execution authority.

## Packaging guidance

Keep package-specific `AGENTS.md` beside its manifest and preserve existing
instructions. Track source-documented, pack-verified and released states
separately. Inspect `npm pack --ignore-scripts --json` contents and compressed /
unpacked sizes before claiming guidance inclusion. This inspection skips build
hooks and is not a release or runtime validation. A package with missing generated
entry points must not be represented as a usable release artifact.

Document supported build and test commands per package. Do not edit installed
`node_modules` as source, silently upgrade native dependencies or introduce new
services or credentials. Keep an operator stop control and use bounded retries.
