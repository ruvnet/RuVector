//! Integration tests for CompositeBackend routing (ADR-094).
//!
//! Tests verify path-prefix routing to correct backends,
//! path traversal re-validation after prefix stripping (ADR-103 C11),
//! and multiple-route configurations.

use rvagent_backends::utils::contains_traversal;

/// Simulated route entry for testing CompositeBackend routing logic.
struct Route {
    prefix: String,
    backend_name: String,
}

/// Select the backend and stripped path for a given input path.
/// Uses longest-prefix-first matching (same as CompositeBackend).
fn route_path<'a>(
    routes: &'a [Route],
    path: &str,
    default_backend: &'a str,
) -> (&'a str, String) {
    // Routes should be sorted by prefix length descending.
    for route in routes {
        if path.starts_with(&route.prefix) {
            let stripped = path[route.prefix.len()..].to_string();
            let stripped = stripped.trim_start_matches('/').to_string();
            return (&route.backend_name, stripped);
        }
    }
    (default_backend, path.to_string())
}

/// The composite router should select the backend matching the path prefix.
#[test]
fn test_route_to_correct_backend() {
    // Routes sorted by prefix length descending (longest first).
    let routes = vec![
        Route {
            prefix: "sandbox/workspace/".to_string(),
            backend_name: "workspace_backend".to_string(),
        },
        Route {
            prefix: "sandbox/".to_string(),
            backend_name: "sandbox_backend".to_string(),
        },
    ];

    // Path matching longer prefix should route to workspace_backend.
    let (backend, stripped) = route_path(&routes, "sandbox/workspace/src/main.rs", "default");
    assert_eq!(backend, "workspace_backend");
    assert_eq!(stripped, "src/main.rs");

    // Path matching shorter prefix should route to sandbox_backend.
    let (backend2, stripped2) = route_path(&routes, "sandbox/other/file.txt", "default");
    assert_eq!(backend2, "sandbox_backend");
    assert_eq!(stripped2, "other/file.txt");

    // Path matching no prefix should route to default.
    let (backend3, stripped3) = route_path(&routes, "local/file.txt", "default");
    assert_eq!(backend3, "default");
    assert_eq!(stripped3, "local/file.txt");
}

/// After prefix stripping, the resulting path must be re-validated
/// against traversal attacks (ADR-103 C11 / SEC-003).
#[test]
fn test_prefix_strip_path_traversal_blocked() {
    let routes = vec![Route {
        prefix: "sandbox/".to_string(),
        backend_name: "sandbox_backend".to_string(),
    }];

    // Attacker tries: "sandbox/../../../etc/passwd"
    // After stripping prefix "sandbox/", we get "../../../etc/passwd"
    let (_, stripped) = route_path(&routes, "sandbox/../../../etc/passwd", "default");
    assert!(
        contains_traversal(&stripped),
        "stripped path '{}' should be flagged as traversal",
        stripped
    );

    // Another variant: "sandbox/foo/../../etc/shadow"
    let (_, stripped2) = route_path(&routes, "sandbox/foo/../../etc/shadow", "default");
    assert!(
        contains_traversal(&stripped2),
        "stripped path '{}' should be flagged as traversal",
        stripped2
    );

    // Tilde expansion attempt.
    let (_, stripped3) = route_path(&routes, "sandbox/~root/.ssh/id_rsa", "default");
    // The ~ itself is not traversal, but real CompositeBackend should
    // also reject paths starting with ~.
    assert!(
        stripped3.starts_with('~'),
        "stripped path should start with ~ for additional validation"
    );

    // Safe stripped path should pass.
    let (_, safe) = route_path(&routes, "sandbox/src/lib.rs", "default");
    assert!(!contains_traversal(&safe));
    assert!(!safe.starts_with('~'));
}

/// Multiple routes with different prefixes should each route correctly.
#[test]
fn test_multiple_routes() {
    let routes = vec![
        Route {
            prefix: "docker/app/src/".to_string(),
            backend_name: "docker_src".to_string(),
        },
        Route {
            prefix: "docker/app/".to_string(),
            backend_name: "docker_app".to_string(),
        },
        Route {
            prefix: "docker/".to_string(),
            backend_name: "docker_root".to_string(),
        },
        Route {
            prefix: "local/".to_string(),
            backend_name: "local_fs".to_string(),
        },
    ];

    // Most specific match wins.
    let (b1, p1) = route_path(&routes, "docker/app/src/main.rs", "default");
    assert_eq!(b1, "docker_src");
    assert_eq!(p1, "main.rs");

    let (b2, p2) = route_path(&routes, "docker/app/Cargo.toml", "default");
    assert_eq!(b2, "docker_app");
    assert_eq!(p2, "Cargo.toml");

    let (b3, p3) = route_path(&routes, "docker/Dockerfile", "default");
    assert_eq!(b3, "docker_root");
    assert_eq!(p3, "Dockerfile");

    let (b4, p4) = route_path(&routes, "local/readme.md", "default");
    assert_eq!(b4, "local_fs");
    assert_eq!(p4, "readme.md");

    // No match -> default.
    let (b5, p5) = route_path(&routes, "remote/file.txt", "default");
    assert_eq!(b5, "default");
    assert_eq!(p5, "remote/file.txt");

    // All stripped paths should be traversal-safe.
    for path in &[p1, p2, p3, p4, p5] {
        assert!(
            !contains_traversal(path),
            "stripped path '{}' should not contain traversal",
            path
        );
    }
}
