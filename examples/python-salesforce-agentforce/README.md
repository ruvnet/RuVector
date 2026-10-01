# RuVector + Salesforce Agentforce (example)

**Everything under this directory is illustrative, not validated against a real Salesforce
org or the current Salesforce Metadata API schema.** This session's task boundary was "mock
the Salesforce APIs, do not hit a real org" (ADR-352), so these metadata sketches were written
from Salesforce's public documentation on External Services and Named Credentials, not confirmed
by importing them into a live org's Setup UI or by a Metadata API `describe` call. Treat them as
a starting point — review and adjust in your own org (ideally in a scratch/sandbox org first)
before deploying anything here.

## What this demonstrates

`ruvector`'s Salesforce integration (`ruvector[salesforce]`, see
`docs/adr/ADR-352-ruvector-python-sdk-cli-mcp.md`'s "Integrations" section for the full writeup
of what was and wasn't verified) exposes three vector operations — search, upsert, and a
grounding-context join — as plain HTTP actions, fronted by a generated OpenAPI 3.0 document. The
extension point used is **External Services + OpenAPI**, confirmed to be the self-service,
no-special-access path (Data Cloud's retrieval surface doesn't accept an arbitrary external
vector store, and Agentforce's own MCP client is Beta/AE-tier-gated — neither was buildable or
verifiable in this session; see the ADR for the research that established this).

## Steps (sketch — verify each against your own org)

1. **Run the `ruvector` MCP server with Salesforce actions enabled:**

   ```bash
   export RUVECTOR_SALESFORCE_ACTION_TOKEN="<a real secret, not this placeholder>"
   export RUVECTOR_ENABLE_SALESFORCE_ACTIONS=1
   ruvector serve --http --host 0.0.0.0 --port 8420
   ```

   The OpenAPI document this External Service imports is served at
   `http://<host>:8420/salesforce/openapi.json` (no auth required on that one GET — see
   `ruvector.salesforce_routes` for why only the three action routes are bearer-gated).

2. **Create a Named Credential** pointing at that host — `namedCredentials/` has a sketch. The
   action token goes in the Named Credential's auth configuration, not in any metadata file
   committed here.

3. **Register an External Service** from the OpenAPI URL above —
   `externalServiceRegistrations/` has a sketch of the registration metadata. Salesforce's UI
   path (Setup → External Services → New External Service → "From API Specification") is the
   reliable way to do this interactively; the metadata file is for tracking the result in source
   control afterward, not for a blind deploy.

4. **Expose the resulting actions to an Agentforce agent** via Setup → Agentforce Studio, adding
   the registered External Service's operations (`search`, `upsert`, `ground`) as agent actions.
   This step is pure org configuration with no `ruvector`-specific metadata to track here.

## What's explicitly out of scope here

- **Agentforce MCP registration** — not built. Documented in the ADR as a beta/AE-tier-gated
  option for a future session with real access to verify against, not something this example
  pretends to demonstrate.
- **Data Cloud "bring your own retriever"** — not applicable; its grounding surface queries
  Data Cloud's own indexed data, not an arbitrary external vector store over HTTP (see the ADR's
  research findings).
- **A `GenAiFunction` metadata sketch** — deliberately not included. It's a newer metadata type
  whose exact current XML schema this session could not confirm without a live org or a current
  Metadata API reference, and fabricating one here would misrepresent confidence this session
  doesn't have. Register agent actions through the Agentforce Studio UI (step 4 above) instead.
