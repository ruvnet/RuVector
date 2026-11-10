# RuVector on Cloudflare for ChatGPT

Tenant scoped MCP tools and a compact MCP Apps UI. The Worker runs real RVF exact search and weighted mincut WASM. It stores numeric vectors in D1 and authenticates each request with an issuer signed JWT and a server managed membership table.

## Capability status

| Capability | Status | Boundary |
| --- | --- | --- |
| RVF exact search and action ranking | Implemented, tested with `@ruvector/rvf-wasm@0.1.9` | Maximum 500 vectors per search, 1536 dimensions, no embedding generation |
| Weighted graph mincut | Implemented, tested with `@ruvector/mincut-wasm@2.3.0` | Maximum 100 edges per call |
| MCP Apps console | Bundled, build validated | ChatGPT rendering still needs a live acceptance test |
| MetaHarness Darwin | Signed service binding adapter | No evaluation Worker is provisioned; no promotion tool |
| Autogenous | Signed service binding adapter | Proposal only; no learning Worker is provisioned |
| ruvLLM | Signed service binding adapter | The current `ruvllm-wasm` package has placeholder inference methods; a validated inference Worker is required |
| Persistent HNSW, other RuVector WASM modules | Not exposed | Require a tenant partitioned index service and per module compatibility tests |

The UI reports available capabilities from the deployed bindings. It never labels a pending service as ready. Distances from action ranking are not calibrated probabilities.

## Tenant and security model

The JWT must verify against `OAUTH_ISSUER`, `OAUTH_AUDIENCE`, and `OAUTH_JWKS_URL`, contain `sub`, `iat`, `exp`, and include `ruvector.read`. `OAUTH_AUDIENCE` is the canonical HTTPS `/mcp` resource identifier advertised in protected resource metadata and carried into the token audience. Writes also require `ruvector.write`. An administrator provisions `(issuer, subject, tenant_id, role)` in D1. Every tool checks membership and role, then binds the authorized tenant into its queries. Tool supplied tenant IDs cannot grant access. D1 enforces the collection dimension and 500 record capacity. The atomic daily usage counter defaults to 1000 calls per tenant.

The D1 database in the Cognitum Cloudflare account was created with US jurisdiction. This is an initial pilot choice. Regional tenant requirements need separate jurisdiction scoped stores before accepting data for those tenants. No customer records or memberships have been inserted by this package.

The optional service bindings only call a fixed internal path. Requests carry a SHA256 HMAC over an envelope containing tenant, issuer, actor, operation, timestamp, request ID, and bounded input. A downstream service must verify the HMAC, enforce a short timestamp window and request ID replay protection, and use the signed tenant as its data boundary. Do not enable a binding before its corresponding Worker passes that contract.

## Build and test

From this directory:

```sh
npm ci --workspaces=false
npm test
npm run typecheck
npm run build
```

The build copies pinned package WASM files to ignored `src/vendor/` and runs a Wrangler dry run. The resulting module bundle is about 1.84 MiB before compression. `schema.sql` can be applied to a local D1 database with:

```sh
npx wrangler d1 execute ruvector-chatgpt --local --file schema.sql
```

## Live deployment gate

1. Choose an OAuth 2.1 compatible issuer whose JWT access tokens carry the exact advertised `/mcp` resource URL as audience and `ruvector.read` or `ruvector.write` scopes. Set real `OAUTH_ISSUER`, `OAUTH_JWKS_URL`, and `OAUTH_AUDIENCE` values in the Worker configuration. The current placeholders deliberately fail closed. Verify the issuer advertises authorization and token endpoints, S256 PKCE, a supported ChatGPT client registration method, and preservation of the `resource` parameter at both authorization and token exchange. Existing Cognitum `ruview` audience tokens cannot be used for this MCP resource.
2. Apply `schema.sql` to the provisioned D1 database with `npx wrangler d1 execute ruvector-chatgpt --remote --file schema.sql`. Provision memberships through an administrator controlled path. Do not derive membership from a tool argument or an unverified email header.
3. Deploy with `npm run deploy` using a Cloudflare deployment credential scoped to this Worker and D1 binding. Add service bindings and `SERVICE_SIGNING_KEY` only after each downstream Worker verifies the signed envelope. Never commit a signing key.
4. Inspect `/health`, `/.well-known/oauth-protected-resource/mcp`, a 401 challenge without a token, MCP initialization, tool discovery, the UI resource, and authorized calls. Verify a valid token from tenant A gets 403 for tenant B, a viewer cannot upsert, and an editor with write scope can upsert and search. Refresh the ChatGPT plugin connection after a tool schema change.

The MCP endpoint is `https://ruvector-chatgpt.cognitum-consulting-mail.workers.dev/mcp` once this Worker is deployed. This URL is an intended route, not a claim of a live deployment.

## Rollback

Keep the previous Worker version and D1 backup. Roll back the Worker first. Schema changes must be additive until old and new Worker versions both pass the acceptance set. Disable optional service bindings independently if their downstream validation fails.
