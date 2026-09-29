//! Registry tools on `/v1/mcp` (ADR-351 §3, M5): `rvf_list` and `rvf_get`
//! (read), `rvf_import` (write, gated like `vector_upsert`: `ruvector:write`
//! and an editor role, else the HTTP 403 step-up). There is deliberately no
//! publish tool: public publish is `/v1` only.

use crate::backend::Backend;
use crate::mcp::McpExtra;
use crate::registry_ports::{root, scope, BlobStore, RegistryRpc};
use crate::registry_routes::{registry_caller, target};
use crate::registry_wire::{CallerWire, RootCall, RootOut, RvfError, ScopeCall, ScopeOut};
use crate::rvf_upload::Deps;
use ruvector_edge_registry::{PackageName, RegistryError};
use ruvector_edge_store::CallerContext;
use ruvector_edge_tenancy::ProblemCode;
use serde::Deserialize;
use serde_json::{json, Map, Value as Json};

/// The registry tools over one request's ports.
pub struct RvfTools<'a, 'd, B, R, S>(pub &'a Deps<'d, B, R, S>);

fn args<T: for<'de> Deserialize<'de>>(m: Map<String, Json>) -> Result<T, RvfError> {
    serde_json::from_value(Json::Object(m))
        .map_err(|_| RvfError::new(ProblemCode::InvalidRequest, "malformed arguments"))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListArgs {
    #[serde(default)]
    package: Option<String>,
    #[serde(default)]
    cursor: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GetArgs {
    package: String,
    version: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportArgs {
    collection: String,
    package: String,
    version: String,
    #[serde(default)]
    offset: u64,
}

fn tool(name: &str, description: &str, props: Json, required: &[&str], write: bool) -> Json {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object", "properties": props, "required": required,
            "additionalProperties": false,
        },
        "annotations": {
            "readOnlyHint": !write,
            "destructiveHint": write,
            "idempotentHint": true,
            "openWorldHint": false,
        },
    })
}

/// `tools/list` entries.
pub fn tools() -> Vec<Json> {
    let pkg = json!({ "type": "string", "description": "@scope/name" });
    let ver = json!({ "type": "string", "description": "exact SemVer version" });
    vec![
        tool(
            "rvf_list",
            "Without a package: the RVF scopes your tenant owns. With one: its versions you can pull, newest first (paginated).",
            json!({ "package": pkg, "cursor": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100 } }),
            &[],
            false,
        ),
        tool(
            "rvf_get",
            "Manifest of one RVF package version: dimension, metric, vector count, segments, SHA-256, visibility, yank.",
            json!({ "package": pkg, "version": ver }),
            &["package", "version"],
            false,
        ),
        tool(
            "rvf_import",
            "Import an RVF package version into a collection (same dimension and metric); vectors are upserted by id. Large packages import in windows: repeat with offset = next_offset until it is null.",
            json!({ "collection": { "type": "string" }, "package": pkg, "version": ver, "offset": { "type": "integer", "minimum": 0 } }),
            &["collection", "package", "version"],
            true,
        ),
    ]
}

impl<B: Backend, R: RegistryRpc, S: BlobStore> McpExtra for RvfTools<'_, '_, B, R, S> {
    fn tools(&self) -> Vec<Json> {
        tools()
    }

    async fn call(
        &self,
        ctx: &CallerContext,
        name: &str,
        a: Map<String, Json>,
        now: u64,
    ) -> Option<Result<Json, RvfError>> {
        let d = self.0;
        let run = async {
            let c = registry_caller(d.b, ctx).await?;
            let cw = CallerWire::from(&c);
            match name {
                "rvf_list" => {
                    let ListArgs {
                        package,
                        cursor,
                        limit,
                    } = args(a)?;
                    let Some(p) = package else {
                        if !c.caps.contains(ruvector_edge_auth::Capability::Read) {
                            return Err(RegistryError::Forbidden(
                                ruvector_edge_auth::Capability::Read,
                            )
                            .into());
                        }
                        let RootOut::Scopes { scopes } =
                            root(d.r, RootCall::Scopes { caller: cw }).await?
                        else {
                            return Err(RvfError::unexpected());
                        };
                        return Ok(json!({ "scopes": scopes }));
                    };
                    let n = PackageName::parse(&p)
                        .map_err(|e| RvfError::from(RegistryError::from(e)))?;
                    let call = ScopeCall::Versions {
                        caller: cw,
                        name: n.to_string(),
                        cursor,
                        limit: limit.unwrap_or(0),
                    };
                    match scope(d.r, n.scope(), call).await? {
                        ScopeOut::Versions { page } => {
                            serde_json::to_value(page).map_err(|_| RvfError::unexpected())
                        }
                        _ => Err(RvfError::unexpected()),
                    }
                }
                "rvf_get" => {
                    let GetArgs { package, version } = args(a)?;
                    let t = target(&package, &version)?;
                    let call = ScopeCall::Get {
                        caller: cw,
                        at: t.coords(),
                    };
                    match scope(d.r, t.name.scope(), call).await? {
                        ScopeOut::Manifest { manifest } => {
                            serde_json::to_value(manifest).map_err(|_| RvfError::unexpected())
                        }
                        _ => Err(RvfError::unexpected()),
                    }
                }
                _ => {
                    let ImportArgs {
                        collection,
                        package,
                        version,
                        offset,
                    } = args(a)?;
                    let body = json!({ "package": package, "version": version, "offset": offset })
                        .to_string();
                    crate::rvf_import::import(d, ctx, &c, &collection, body.as_bytes(), now)
                        .await
                        .map(|(_, v)| v)
                }
            }
        };
        matches!(name, "rvf_list" | "rvf_get" | "rvf_import").then_some(())?;
        Some(run.await)
    }
}
