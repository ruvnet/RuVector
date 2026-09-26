---
description: "Review the current working diff for correctness, leakage, and safety-policy compliance."
---

Review the current git diff.

1. `git diff` to read the change.
2. Report only high-confidence findings as `file:line — issue — fix`.
3. Flag any forbidden mutation (fabricated observations, invented citations, test-window leakage, new imports/network/shell/env, weakened promotion gate).
4. Separate bugs from nits.
5. End with APPROVE or REQUEST-CHANGES and a one-line reason.
