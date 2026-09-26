---
description: "Health-check the harness: kernel load, MCP wiring, pipeline build, host adapter."
---

Run a full health check and print a PASS/FAIL table.

1. Kernel loads and `kernelInfo().version` matches package.json.
2. The MCP server starts and lists its tools.
3. The detect→extract→embed→score pipeline runs over the bundled fixtures.
4. The configured host adapter is present.

Exit non-zero if any check fails.
