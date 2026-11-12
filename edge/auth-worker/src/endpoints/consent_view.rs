//! HTML of the consent page (ADR-351 §5.6): a single server-rendered card,
//! no script, no external resources. Every interpolated value goes through
//! [`escape`]; the only request-derived text is what the page always showed
//! (client name, redirect host — plus the loopback port — resource and the
//! granted scopes). Headers and the cookie live in [`super::consent::page`].

use super::consent::{escape, CONSENT_PATH};
use ruvector_edge_authz::authorize::{redirect_is_verified, ValidatedAuthorization};
use ruvector_edge_authz::client::{ClientRecord, GRANT_REFRESH};
use ruvector_edge_authz::refresh::{FAMILY_MAX_LIFETIME_SECS, OFFLINE_ACCESS};
use ruvector_edge_authz::resource::{exchange_disclosure, GATEWAY_V1_URL, TEAM_RESOURCE_URL};

/// Seconds the Continue button stays inert after the page renders (CSS
/// only; blunts click-jacking and double-click tricks without script).
pub const ARM_DELAY_SECS: u32 = 1;

/// Friendly names of the compiled resources; anything else shows its host.
fn resource_label(url: &str) -> Option<&'static str> {
    const MCP_SUFFIX: &str = "/mcp";
    if url == TEAM_RESOURCE_URL {
        Some("RuFlo AI Team")
    } else if url == GATEWAY_V1_URL {
        Some("RuVector data API")
    } else if url.strip_suffix(MCP_SUFFIX) == Some(GATEWAY_V1_URL) {
        Some("RuVector MCP tools")
    } else {
        None
    }
}

/// Plain-language text of a scope (`None` for an unknown scope, which is
/// then shown by its code).
fn scope_text(scope: &str) -> Option<&'static str> {
    Some(match scope {
        "ruvector:read" => "Read your vectors, collections and graphs",
        "ruvector:write" => "Add, change and delete vectors; create collections",
        "ruvector:admin" => "Manage members, access controls and restores",
        "team:read" => "View your AI Team work",
        "team:write" => "Create and change AI Team work",
        "team:run" => "Run AI Team agents on your behalf",
        _ => return None,
    })
}

/// What an exchanged `ruvector:*` scope lets team.ruv.io do.
fn exchange_effect(rv: &str) -> &'static str {
    match rv {
        "ruvector:read" => "read your RuVector data",
        "ruvector:write" => "add, change and delete vectors and collections in your RuVector data",
        _ => "use your RuVector data",
    }
}

/// One permission row: plain text, the code as secondary text, and an
/// optional extra line (already HTML).
fn row(code: &str, text: &str, extra: &str) -> String {
    format!(
        "<li data-scope=\"{c}\"><span class=\"pt\">{t}</span>{extra}\
<code>{c}</code></li>",
        c = escape(code),
        t = escape(text),
    )
}

/// Row disclosing long-lived access. Shown whenever the client registered
/// the refresh grant — refresh tokens follow `grant_types`, not the
/// `offline_access` scope (ADR-351 §5.3) — so the page always tells the
/// user when access outlives the 15-minute access token.
pub fn offline_access_notice() -> String {
    row(
        OFFLINE_ACCESS,
        &format!(
            "Stay signed in (up to {} days)",
            FAMILY_MAX_LIFETIME_SECS / 86_400
        ),
        "<span class=\"ps\">Keeps access with a refresh token until you sign \
out or revoke it.</span>",
    )
}

/// The permission rows. Every `team:*` scope granted for team.ruv.io that
/// the compiled exchange map carries into a `…/v1` token gets an "Also"
/// line naming its `ruvector:*` effect (ADR-351 §5.6, §16.1), so a family
/// consented now needs no new consent when the M1 exchange ships.
pub fn permission_rows(client: &ClientRecord, auth: &ValidatedAuthorization) -> String {
    let pairs = exchange_disclosure(&auth.resource, &auth.scopes);
    let mut rows: String = auth
        .scopes
        .iter()
        .filter(|s| s.as_str() != OFFLINE_ACCESS)
        .map(|s| {
            let also = pairs
                .iter()
                .find(|(team, _)| *team == s.as_str())
                .map(|(_, rv)| {
                    format!(
                        "<span class=\"ps also\">Also: {} (<code>{}</code>)</span>",
                        exchange_effect(rv),
                        escape(rv)
                    )
                })
                .unwrap_or_default();
            row(s, scope_text(s).unwrap_or(s), &also)
        })
        .collect();
    if client.allows_grant(GRANT_REFRESH) {
        rows.push_str(&offline_access_notice());
    } else if auth.scopes.iter().any(|s| s == OFFLINE_ACCESS) {
        rows.push_str(&row(
            OFFLINE_ACCESS,
            "Offline access",
            "<span class=\"ps\">This app is not issued a refresh token.</span>",
        ));
    }
    rows
}

/// Explicit statement that team.ruv.io reaches the user's RuVector data
/// (ADR-351 §5.6, §16.1). Empty for every grant without an exchange entry.
pub fn ruvector_data_notice(auth: &ValidatedAuthorization) -> String {
    if exchange_disclosure(&auth.resource, &auth.scopes).is_empty() {
        return String::new();
    }
    "<p class=\"n\"><strong>Includes your RuVector data.</strong> The team.ruv.io \
service can access your RuVector data on your behalf with these permissions \
(see “Also” above).</p>"
        .to_string()
}

/// Whether a client name could imitate another app (anything outside
/// printable ASCII, e.g. Cyrillic homoglyphs of a Latin name).
pub fn is_suspicious_name(name: &str) -> bool {
    !name.bytes().all(|b| (0x20..0x7f).contains(&b))
}

/// Where the user returns after signing in, as HTML: loopback redirects
/// are "an app running on this computer", anything else shows its host.
fn destination(redirect_uri: &str) -> String {
    use std::net::{Ipv4Addr, Ipv6Addr};
    use url::Host;
    let Ok(url) = url::Url::parse(redirect_uri) else {
        return String::new();
    };
    let loopback = url.scheme() == "http"
        && match url.host() {
            Some(Host::Ipv4(ip)) => ip == Ipv4Addr::LOCALHOST,
            Some(Host::Ipv6(ip)) => ip == Ipv6Addr::LOCALHOST,
            _ => false,
        };
    let host = url.host_str().unwrap_or_default();
    if loopback {
        let at = match url.port() {
            Some(p) => format!("{host}:{p}"),
            None => host.to_string(),
        };
        return format!(
            "<strong>an app running on this computer</strong> \
<span class=\"sub\">({})</span>",
            escape(&at)
        );
    }
    format!("<strong class=\"host\">{}</strong>", escape(host))
}

/// Warning blocks shown above the buttons.
fn warnings(auth: &ValidatedAuthorization, name: &str) -> String {
    let warn = |title: &str, body: &str| {
        format!(
            "<div class=\"w\"><span class=\"wi\" aria-hidden=\"true\">!</span>\
<p><strong>{title}</strong> {body}</p></div>"
        )
    };
    let mut out = String::new();
    if !redirect_is_verified(&auth.redirect_uri) {
        out.push_str(&warn(
            "Unverified application.",
            "Anyone can register an application here; this one sends you back \
to a site that is not a known connector. Continue only if you set it up yourself.",
        ));
    }
    if is_suspicious_name(name) {
        out.push_str(&warn(
            "Unusual characters.",
            "This application's name contains characters outside plain ASCII \
and may imitate another app. Check where it sends you back to (shown above).",
        ));
    }
    out
}

/// Inline stylesheet; `__ARM__` is replaced by [`ARM_DELAY_SECS`]. The
/// `.p` animation and `@keyframes arm` are the click-jacking arm delay.
const CSS: &str = ":root{color-scheme:light dark;--bg:#f3f4f6;--card:#fff;--fg:#111827;\
--mu:#4b5563;--bd:#d1d5db;--pri:#1a56db;--pfg:#fff;--fo:#1a56db;--wbg:#fdf2f2;\
--wbd:#c81e1e;--wfg:#7f1d1d;--nbg:#eff6ff;--nbd:#1a56db;--av:#dbeafe;--avfg:#1e3a8a}\
@media (prefers-color-scheme:dark){:root{--bg:#0b0f17;--card:#151b26;--fg:#e5e7eb;\
--mu:#a3acb9;--bd:#2f3a4d;--pri:#2563eb;--pfg:#fff;--fo:#93c5fd;--wbg:#2a1215;\
--wbd:#f87171;--wfg:#fecaca;--nbg:#0f1d33;--nbd:#60a5fa;--av:#1e3a8a;--avfg:#dbeafe}}\
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);\
font:16px/1.5 system-ui,-apple-system,\"Segoe UI\",sans-serif}\
main{max-width:30rem;margin:2.5rem auto;padding:1.75rem;background:var(--card);\
border:1px solid var(--bd);border-radius:.9rem}\
.brand{display:flex;align-items:center;gap:.4rem;font-weight:600;font-size:.9rem;\
color:var(--mu);margin-bottom:1.25rem}.brand svg{width:1.1rem;height:1.1rem}\
.app{display:flex;gap:.9rem;align-items:flex-start}\
.av{flex:none;width:2.75rem;height:2.75rem;border-radius:.7rem;display:flex;\
align-items:center;justify-content:center;font-weight:700;font-size:1.25rem;\
background:var(--av);color:var(--avfg)}\
h1{font-size:1.3rem;line-height:1.3;margin:0 0 .2rem;overflow-wrap:anywhere}\
h2{font-size:.95rem;margin:1.25rem 0 .5rem}\
.mu,.sub,code,.ps{color:var(--mu)}.mu{font-size:.8rem;margin:0}\
dl{margin:1.25rem 0 0;display:grid;gap:.6rem}dt{font-size:.8rem;color:var(--mu)}\
dd{margin:0;overflow-wrap:anywhere}.sub,.ps{display:block;font-size:.85rem}\
.host{font-size:1.05rem}code{font:.75rem ui-monospace,SFMono-Regular,Menlo,monospace}\
ul{list-style:none;margin:0;padding:0;border:1px solid var(--bd);border-radius:.6rem}\
li{padding:.6rem .8rem;border-top:1px solid var(--bd)}li:first-child{border-top:0}\
li>code{display:block;margin-top:.1rem}.pt{font-weight:500}\
.n{margin:.9rem 0 0;padding:.6rem .8rem;border-left:4px solid var(--nbd);\
background:var(--nbg);border-radius:.3rem;font-size:.9rem}\
.w{display:flex;gap:.6rem;align-items:flex-start;margin:1rem 0 0;padding:.6rem .8rem;\
border-left:4px solid var(--wbd);background:var(--wbg);color:var(--wfg);border-radius:.3rem}\
.w p{margin:0;font-size:.9rem}.wi{flex:none;width:1.3rem;height:1.3rem;border-radius:50%;\
background:var(--wbd);color:var(--card);font-weight:700;display:flex;\
align-items:center;justify-content:center;font-size:.85rem}\
.act{display:flex;gap:.6rem;margin-top:1.5rem;align-items:center}form{margin:0}\
.b{display:inline-block;padding:.65rem 1.1rem;border-radius:.5rem;\
border:1px solid var(--bd);background:transparent;color:var(--fg);text-align:center;\
text-decoration:none;font:inherit;font-weight:600;cursor:pointer}\
.p{background:var(--pri);color:var(--pfg);border-color:var(--pri);\
animation:arm __ARM__s steps(1,end)}\
@keyframes arm{from{pointer-events:none;opacity:.5}to{pointer-events:auto;opacity:1}}\
a:focus-visible,button:focus-visible{outline:3px solid var(--fo);outline-offset:2px}\
footer{margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--bd);font-size:.85rem}\
footer p{margin:0}\
@media (max-width:32rem){main{margin:0;min-height:100vh;border:0;border-radius:0;\
padding:1.25rem 1rem}.act{flex-direction:column;align-items:stretch}.b{width:100%}}";

/// Cognitum wordmark glyph (inline SVG, `currentColor`, no external refs).
const MARK: &str = "<svg viewBox=\"0 0 24 24\" aria-hidden=\"true\" focusable=\"false\">\
<path fill=\"currentColor\" d=\"M12 2 3 7v10l9 5 9-5V7l-9-5zm0 2.3 6.9 3.8v7.8L12 \
19.7l-6.9-3.8V8.1L12 4.3zm0 3.2a4.5 4.5 0 1 0 3.2 7.7l-1.4-1.4A2.5 2.5 0 1 1 12 9.5c.7 \
0 1.3.3 1.8.7l1.4-1.4A4.5 4.5 0 0 0 12 7.5z\"/></svg>";

/// Values the page shows beyond the client and the authorization.
pub struct ViewParams<'a> {
    /// Upstream `state` of this flow (hidden field).
    pub flow: &'a str,
    /// Consent form token (hidden field).
    pub token: &'a str,
    /// URL of the Cancel link (`access_denied` to the redirect URI).
    pub cancel_url: &'a str,
    /// This AS's issuer URL (its host is shown in the footer).
    pub issuer: &'a str,
}

/// The consent page HTML.
pub fn render(client: &ClientRecord, auth: &ValidatedAuthorization, v: &ViewParams<'_>) -> String {
    let name = client
        .client_name
        .as_deref()
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .unwrap_or("An unnamed application");
    let initial: String = name
        .chars()
        .next()
        .map(|c| c.to_uppercase().collect())
        .unwrap_or_default();
    let resource = auth.resource.as_str();
    let resource_html = match resource_label(resource) {
        Some(label) => format!(
            "<strong>{}</strong><span class=\"sub\">{}</span>",
            escape(label),
            escape(resource)
        ),
        None => {
            let host = url::Url::parse(resource)
                .ok()
                .and_then(|u| u.host_str().map(str::to_string))
                .unwrap_or_default();
            format!(
                "<strong>{}</strong><span class=\"sub\">{}</span>",
                escape(&host),
                escape(resource)
            )
        }
    };
    let issuer_host = url::Url::parse(v.issuer)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .unwrap_or_default();
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
<meta name=\"color-scheme\" content=\"light dark\">\
<title>Authorize {name} · Cognitum</title><style>{css}</style></head><body>\
<main><div class=\"brand\">{mark}<span>Cognitum</span></div>\
<div class=\"app\"><span class=\"av\" aria-hidden=\"true\">{initial}</span><div>\
<h1>{name} wants access to your account</h1>\
<p class=\"mu\">This name is provided by the app itself and is not verified \
by Cognitum.</p></div></div>\
<dl><div><dt>After you sign in, you return to</dt><dd>{dest}</dd></div>\
<div><dt>It will access</dt><dd>{resource_html}</dd></div></dl>\
<h2>This will let it:</h2><ul>{rows}</ul>{ruvector_data}{warnings}\
<div class=\"act\"><form method=\"post\" action=\"{action}\">\
<input type=\"hidden\" name=\"flow\" value=\"{flow}\">\
<input type=\"hidden\" name=\"consent\" value=\"{token}\">\
<button class=\"b p\" type=\"submit\">Continue to sign in with Cognitum</button></form>\
<a class=\"b\" href=\"{cancel}\">Cancel</a></div>\
<footer><p>You can revoke access at any time.</p>\
<p class=\"mu\">{issuer_host}</p></footer></main></body></html>",
        css = CSS.replace("__ARM__", &ARM_DELAY_SECS.to_string()),
        mark = MARK,
        name = escape(name),
        initial = escape(&initial),
        dest = destination(&auth.redirect_uri),
        rows = permission_rows(client, auth),
        ruvector_data = ruvector_data_notice(auth),
        warnings = warnings(auth, name),
        action = CONSENT_PATH,
        flow = escape(v.flow),
        token = escape(v.token),
        cancel = escape(v.cancel_url),
        issuer_host = escape(&issuer_host),
    )
}
