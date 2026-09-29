//! Consent page presentation (ADR-351 §5.6): friendly resource labels,
//! plain-language scopes, destination wording, warnings, theming, and the
//! static-page security invariants. Writes review samples when
//! `CONSENT_SAMPLE_DIR` is set.

use super::*;

/// The resource is named in plain language with its URL as secondary text,
/// and every scope row reads as a sentence with the code beneath it.
#[test]
fn consent_page_uses_friendly_labels_and_plain_scope_text() {
    let (w, id, team) = team_world();
    let html = consent_html(
        &w,
        &scoped_query(&id, &team, "team:read team:write team:run"),
    );
    assert!(
        html.contains(
            "<strong>RuFlo AI Team</strong><span class=\"sub\">https://team.ruv.io/mcp</span>"
        ),
        "{html}"
    );
    for (scope, text) in [
        ("team:read", "View your AI Team work"),
        ("team:write", "Create and change AI Team work"),
        ("team:run", "Run AI Team agents on your behalf"),
    ] {
        let row = row_of(&html, scope).unwrap();
        assert!(
            row.contains(&format!("<span class=\"pt\">{text}</span>")),
            "{row}"
        );
        assert!(row.contains(&format!("<code>{scope}</code>")), "{row}");
    }
    let gw = crate::config::tests::RESOURCE;
    let html = consent_html(&w, &scoped_query(&id, gw, "ruvector:read"));
    assert!(
        html.contains("<strong>RuVector MCP tools</strong>"),
        "{html}"
    );
    assert!(
        row_of(&html, "ruvector:read")
            .unwrap()
            .contains("Read your vectors, collections and graphs"),
        "{html}"
    );
    let v1 = ruvector_edge_authz::resource::GATEWAY_V1_URL;
    let html = consent_html(&w, &scoped_query(&id, v1, "ruvector:read"));
    assert!(
        html.contains("<strong>RuVector data API</strong>"),
        "{html}"
    );
    // Headline, title and the self-asserted-name note.
    let html = consent_html(&w, &scoped_query(&id, v1, "ruvector:read"));
    assert!(html.contains("<title>Authorize An unnamed application · Cognitum</title>"));
    assert!(html.contains("<h1>An unnamed application wants access to your account</h1>"));
    assert!(
        html.contains("This name is provided by the app itself"),
        "{html}"
    );
}

/// Loopback redirects are described as an app on this computer (with the
/// port), https redirects by their host; the verified connector gets no
/// warning.
#[test]
fn consent_page_describes_loopback_and_https_destinations() {
    let w = World::new();
    let r = w.register(json!({
        "redirect_uris": ["http://127.0.0.1/callback"],
        "client_name": "Local CLI",
    }));
    assert_eq!(r.status, 201, "{}", String::from_utf8_lossy(&r.body));
    let id = body_json(&r)["client_id"].as_str().unwrap().to_string();
    let q = authorize_query(&id, RESOURCE).replace(
        &url::form_urlencoded::byte_serialize(REDIRECT.as_bytes()).collect::<String>(),
        &url::form_urlencoded::byte_serialize(b"http://127.0.0.1:58962/callback")
            .collect::<String>(),
    );
    let html = consent_html(&w, &q);
    assert!(
        html.contains(
            "<strong>an app running on this computer</strong> \
<span class=\"sub\">(127.0.0.1:58962)</span>"
        ),
        "{html}"
    );
    assert!(!html.contains("Unverified application"), "{html}");
    assert!(!html.contains("only if you trust"), "{html}");
    let (page, _) = page_for(&w, &w.client_id());
    let html = String::from_utf8(page.body).unwrap();
    assert!(
        html.contains("<strong class=\"host\">claude.ai</strong>"),
        "{html}"
    );
}

/// Security invariants of the redesigned page: no script or external
/// resource, the arm-delay animation, dark mode, focus styles, the same
/// hidden fields and POST action, footer, and the unchanged CSP.
#[test]
fn consent_page_is_static_themed_and_armed() {
    let w = World::new();
    let (page, _) = page_for(&w, &w.client_id());
    let html = String::from_utf8(page.body.clone()).unwrap();
    let lower = html.to_ascii_lowercase();
    for banned in [
        "<script",
        "url(",
        "data:",
        "<link",
        "<img",
        "@import",
        "http-equiv",
    ] {
        assert!(!lower.contains(banned), "{banned}: {html}");
    }
    assert!(html.contains("<html lang=\"en\">"));
    assert!(html.contains("<title>Authorize Test &lt;b&gt;Client&lt;/b&gt; · Cognitum</title>"));
    assert!(html.contains("<main>") && html.contains("<h1>") && html.contains("<ul>"));
    assert!(html.contains("animation:arm 1s steps(1,end)"), "{html}");
    assert!(html.contains(
        "@keyframes arm{from{pointer-events:none;opacity:.5}to{pointer-events:auto;opacity:1}}"
    ));
    assert!(html.contains("@media (prefers-color-scheme:dark)"));
    assert!(html.contains(":focus-visible"));
    assert!(html.contains("<form method=\"post\" action=\"/authorize/consent\">"));
    assert!(html.contains(
        "<button class=\"b p\" type=\"submit\">Continue to sign in with Cognitum</button>"
    ));
    assert_eq!(html.matches("<input type=\"hidden\"").count(), 2);
    assert!(html.contains("You can revoke access at any time."));
    assert!(
        html.contains("ruvector-edge-auth.example.workers.dev"),
        "{html}"
    );
    let cancel = html.split("<a class=\"b\" href=\"").nth(1).unwrap();
    assert!(
        cancel.starts_with("https://claude.ai/api/mcp/auth_callback?"),
        "{cancel}"
    );
    assert!(cancel.contains("error=access_denied"), "{cancel}");
    assert_eq!(
        page.header("Content-Security-Policy").unwrap(),
        "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; \
         base-uri 'none'; form-action 'self' https://auth.cognitum.one"
    );
    assert_eq!(page.header("Referrer-Policy"), Some("no-referrer"));
    assert_eq!(page.header("X-Frame-Options"), Some("DENY"));
}

/// Consent page for a hand-built client and authorization (lets the test
/// cover names DCR would refuse).
fn direct_page(
    name: &str,
    redirect: &str,
    resource: &str,
    scopes: &[&str],
    refresh: bool,
) -> String {
    use ruvector_edge_authz::client::{validate_redirect_uri, ClientRecord};
    let mut grants = vec!["authorization_code".to_string()];
    if refresh {
        grants.push("refresh_token".to_string());
    }
    let client = ClientRecord {
        client_id: "edc-sample".into(),
        redirect_uris: vec![validate_redirect_uri(redirect).unwrap()],
        grant_types: grants,
        scope: scopes.iter().map(|s| s.to_string()).collect(),
        client_name: Some(name.into()),
        client_id_issued_at: T0,
    };
    let auth = ruvector_edge_authz::authorize::ValidatedAuthorization {
        client_id: client.client_id.clone(),
        redirect_uri: redirect.into(),
        scopes: client.scope.clone(),
        state: Some("client-state".into()),
        code_challenge: CHALLENGE.into(),
        resource: ruvector_edge_authz::ResourceUrl::parse(resource).unwrap(),
    };
    let form = crate::endpoints::consent::ConsentForm {
        flow: "flow-state-abcdefghijklmnop",
        token: "consent-token",
        upstream_origin: "https://auth.cognitum.one",
        issuer: "https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev",
    };
    let cancel = format!("{redirect}?error=access_denied&state=client-state");
    let r = crate::endpoints::consent::page(&client, &auth, &form, &cancel, "c=v".into());
    String::from_utf8(r.body).unwrap()
}

/// Warnings stay above the buttons with a marker; a non-ASCII name is
/// escaped and flagged. With `CONSENT_SAMPLE_DIR` set, the rendered samples
/// are written there for review.
#[test]
fn consent_page_warnings_and_samples() {
    const TEAM: &str = "https://team.ruv.io/mcp";
    let loopback = "http://127.0.0.1:58962/callback";
    let team_scopes = ["team:read", "team:write", "offline_access"];
    let refresh = direct_page(
        "RuFlo AI Team canary validation",
        loopback,
        TEAM,
        &team_scopes,
        true,
    );
    let no_refresh = direct_page(
        "RuFlo AI Team canary validation",
        loopback,
        TEAM,
        &team_scopes[..2],
        false,
    );
    let evil = direct_page(
        "\u{0421}laude <script>",
        "https://evil.example/cb",
        crate::config::tests::RESOURCE,
        &["ruvector:read", "ruvector:write"],
        true,
    );
    assert!(refresh.contains("Stay signed in (up to 90 days)"));
    assert!(!no_refresh.contains("data-scope=\"offline_access\""));
    assert!(no_refresh.contains("Includes your RuVector data"));
    assert!(!refresh.contains("class=\"w\""), "{refresh}");
    assert!(evil.contains("Unverified application."), "{evil}");
    assert!(evil.contains("Unusual characters."), "{evil}");
    assert!(evil.contains("<span class=\"wi\" aria-hidden=\"true\">!</span>"));
    assert!(evil.contains("\u{0421}laude &lt;script&gt;"));
    assert!(!evil.to_ascii_lowercase().contains("<script"));
    assert!(evil.contains("<strong class=\"host\">evil.example</strong>"));
    let warn = evil.find("class=\"w\"").unwrap();
    assert!(warn < evil.find("<div class=\"act\">").unwrap());
    if let Ok(dir) = std::env::var("CONSENT_SAMPLE_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        for (file, html) in [
            ("team-loopback-refresh.html", &refresh),
            ("team-loopback-no-refresh.html", &no_refresh),
            ("unverified-https-non-ascii.html", &evil),
        ] {
            std::fs::write(format!("{dir}/{file}"), html).unwrap();
        }
    }
}
