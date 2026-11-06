//! Import extraction for JS/TS sources. Only syntactic import forms count
//! (`import … from`, side-effect `import '…'`, `require()`, `require.resolve()`,
//! dynamic `import()`); package names that merely appear in comments or
//! strings are ignored, which keeps 404 / undeclared signals precise.

use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq)]
pub struct Import {
    /// Normalised package name (`@scope/name` or `name`).
    pub package: String,
    pub line: usize,
    pub dynamic: bool,
    pub text: String,
}

const BUILTINS: &[&str] = &[
    "assert",
    "async_hooks",
    "buffer",
    "child_process",
    "cluster",
    "console",
    "constants",
    "crypto",
    "dgram",
    "dns",
    "domain",
    "events",
    "fs",
    "http",
    "http2",
    "https",
    "inspector",
    "module",
    "net",
    "os",
    "path",
    "perf_hooks",
    "process",
    "punycode",
    "querystring",
    "readline",
    "repl",
    "stream",
    "string_decoder",
    "sys",
    "timers",
    "tls",
    "trace_events",
    "tty",
    "url",
    "util",
    "v8",
    "vm",
    "wasi",
    "worker_threads",
    "zlib",
    "test",
];

fn static_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?:\bfrom\s*|^\s*import\s*)['"]([^'"\n]+)['"]"#).expect("static import re")
    })
}

fn dynamic_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"\b(?:require(?:\.resolve)?|import)\s*\(\s*['"]([^'"\n]+)['"]"#)
            .expect("dynamic import re")
    })
}

/// `@scope/name/deep/path` -> `@scope/name`; relative, `node:`, builtins,
/// URLs and malformed specifiers -> `None`.
pub fn package_name(spec: &str) -> Option<String> {
    let spec = spec.trim();
    if spec.is_empty()
        || spec.starts_with('.')
        || spec.starts_with('/')
        || spec.starts_with('#')
        || spec.starts_with('~')
        || spec.contains(':')
        || spec.contains('$')
        || spec.contains(' ')
    {
        return None;
    }
    let mut parts = spec.split('/');
    let first = parts.next()?;
    let name = if first.starts_with('@') {
        let second = parts.next()?;
        if second.is_empty() {
            return None;
        }
        format!("{first}/{second}")
    } else {
        first.to_string()
    };
    if BUILTINS.contains(&name.as_str()) {
        return None;
    }
    let ok = name.chars().all(|c| {
        c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_' | '.' | '@' | '/')
    });
    ok.then_some(name)
}

/// Strip `//` and `/* */` comments while keeping line structure, so line
/// numbers stay exact. Naive about comment markers inside strings, which is
/// fine for import detection (import specifiers never contain them).
pub fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut in_block = false;
    for line in src.lines() {
        let mut rest = line;
        let mut kept = String::new();
        loop {
            if in_block {
                match rest.find("*/") {
                    Some(i) => {
                        in_block = false;
                        rest = &rest[i + 2..];
                    }
                    None => break,
                }
            } else {
                let lc = find_line_comment(rest);
                let bc = rest.find("/*");
                match (lc, bc) {
                    (Some(l), Some(b)) if b < l => {
                        kept.push_str(&rest[..b]);
                        in_block = true;
                        rest = &rest[b + 2..];
                    }
                    (Some(l), _) => {
                        kept.push_str(&rest[..l]);
                        break;
                    }
                    (None, Some(b)) => {
                        kept.push_str(&rest[..b]);
                        in_block = true;
                        rest = &rest[b + 2..];
                    }
                    (None, None) => {
                        kept.push_str(rest);
                        break;
                    }
                }
            }
        }
        out.push_str(&kept);
        out.push('\n');
    }
    out
}

/// `//` not preceded by `:` (keeps `https://` in strings intact).
fn find_line_comment(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    (0..b.len().saturating_sub(1))
        .find(|&i| b[i] == b'/' && b[i + 1] == b'/' && (i == 0 || b[i - 1] != b':'))
}

pub fn extract(src: &str) -> Vec<Import> {
    let code = strip_comments(src);
    let originals: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for (i, line) in code.lines().enumerate() {
        let mut push = |spec: &str, dynamic: bool| {
            if let Some(package) = package_name(spec) {
                out.push(Import {
                    package,
                    line: i + 1,
                    dynamic,
                    text: originals
                        .get(i)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_default(),
                });
            }
        };
        for c in static_re().captures_iter(line) {
            push(&c[1], false);
        }
        for c in dynamic_re().captures_iter(line) {
            push(&c[1], true);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalises_names() {
        assert_eq!(
            package_name("@ruvector/rvf/dist/x.js").as_deref(),
            Some("@ruvector/rvf")
        );
        assert_eq!(package_name("ruvector/hooks").as_deref(), Some("ruvector"));
        assert_eq!(package_name("./local"), None);
        assert_eq!(package_name("node:fs"), None);
        assert_eq!(package_name("fs"), None);
    }

    #[test]
    fn ignores_comment_mentions() {
        let src = "// uses @ruvector/learning-wasm\n/* import x from '@ruvector/a' */\nconst m = await import('@ruvector/ast' as string).catch(() => null);\n";
        let got = extract(src);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].package, "@ruvector/ast");
        assert_eq!(got[0].line, 3);
        assert!(got[0].dynamic);
    }
}
