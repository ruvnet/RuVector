# C5: Sandbox Path Restriction Contract

**Status**: ✅ Implemented
**ADR**: ADR-103 C5
**Security Code**: SEC-023
**Crate**: `rvagent-backends`
**Module**: `sandbox`

## Overview

The Sandbox Path Restriction Contract (C5/SEC-023) is a mandatory security contract that ensures all filesystem operations within a sandbox are confined to the sandbox root directory. Any attempt to access files outside the sandbox MUST fail with a `PathEscapesSandbox` error.

## Security Properties

### Mandatory Enforcement

All sandbox implementations MUST:

1. **Confine all filesystem access to `sandbox_root()`**
   - No operations may access files outside the designated root
   - Path validation is mandatory before any filesystem access

2. **Reject path traversal attempts**
   - `../` segments that escape the sandbox
   - Absolute paths pointing outside the sandbox
   - Symlinks that resolve outside the sandbox

3. **Use `validate_path()` before filesystem operations**
   - Canonicalize paths to resolve `.`, `..`, and symlinks
   - Check that canonicalized path starts with sandbox root
   - Return `PathEscapesSandbox` error for violations

4. **Fail securely on violations**
   - Never silently allow escape attempts
   - Provide clear error messages for debugging
   - Log security violations for audit

## Implementation

### Core Types

```rust
/// Sandbox-specific errors (ADR-103 C5)
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("Path escapes sandbox root: {0}")]
    PathEscapesSandbox(String),

    #[error("Command execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Sandbox initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Timeout exceeded")]
    Timeout,

    #[error("IO error: {0}")]
    IoError(String),
}
```

### BaseSandbox Trait

The `BaseSandbox` trait defines the mandatory path restriction contract:

```rust
pub trait BaseSandbox: Send + Sync {
    /// The root path of the sandbox filesystem.
    /// All file operations MUST be confined to this root.
    fn sandbox_root(&self) -> &Path;

    /// Validate that a path is within the sandbox (MANDATORY).
    ///
    /// # Security Contract (SEC-023)
    /// - MUST reject paths outside sandbox_root
    /// - MUST canonicalize paths to resolve symlinks and .. components
    /// - MUST return PathEscapesSandbox error for violations
    fn validate_path(&self, path: &Path) -> Result<PathBuf, SandboxError> {
        // Canonicalize to resolve symlinks and .. components
        let canonical = path.canonicalize()
            .map_err(|e| SandboxError::IoError(format!("Failed to canonicalize {}: {}", path.display(), e)))?;

        let root = self.sandbox_root().canonicalize()
            .map_err(|e| SandboxError::InitializationFailed(format!("Failed to canonicalize root: {}", e)))?;

        // Check if canonical path starts with root
        if !canonical.starts_with(&root) {
            return Err(SandboxError::PathEscapesSandbox(
                format!("{} is outside sandbox root {}", canonical.display(), root.display())
            ));
        }

        Ok(canonical)
    }

    /// Check if a path is within the sandbox root (legacy method).
    fn is_path_confined(&self, path: &Path) -> bool {
        self.validate_path(path).is_ok()
    }
}
```

### LocalSandbox Implementation

`LocalSandbox` provides a concrete implementation with strict security properties:

```rust
pub struct LocalSandbox {
    id: String,
    root: PathBuf,
    config: SandboxConfig,
    created_at: std::time::Instant,
}

impl LocalSandbox {
    pub fn new(root: PathBuf) -> Result<Self, SandboxError> {
        // Create root directory if it doesn't exist
        if !root.exists() {
            std::fs::create_dir_all(&root)
                .map_err(|e| SandboxError::InitializationFailed(
                    format!("Failed to create sandbox root {}: {}", root.display(), e)
                ))?;
        }

        // Verify root is a directory
        if !root.is_dir() {
            return Err(SandboxError::InitializationFailed(
                format!("{} is not a directory", root.display())
            ));
        }

        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            root,
            config: SandboxConfig::default(),
            created_at: std::time::Instant::now(),
        })
    }
}
```

#### Command Execution Security (SEC-005)

Commands execute with:
- Working directory = sandbox root
- Sanitized environment (only HOME and PATH)
- Output size limits to prevent DoS

```rust
fn execute_sync(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse {
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(command);
    cmd.current_dir(&self.root);  // Confine to sandbox

    // Sanitize environment (SEC-005)
    cmd.env_clear();
    cmd.env("HOME", &self.root);
    cmd.env("PATH", "/usr/bin:/bin");

    // Execute with output truncation
    // ...
}
```

## Security Test Suite

Comprehensive tests verify all escape vectors are blocked:

### Path Validation Tests

```rust
#[test]
fn test_validate_path_rejects_parent_directory_escape() {
    let sandbox = LocalSandbox::new(temp_dir).unwrap();
    let escape = temp_dir.join("../etc/passwd");

    let result = sandbox.validate_path(&escape);
    assert!(matches!(result, Err(SandboxError::PathEscapesSandbox(_))));
}

#[test]
fn test_validate_path_rejects_symlink_escape() {
    let sandbox = LocalSandbox::new(temp_dir).unwrap();
    let link = temp_dir.join("evil_link");
    symlink("/etc/passwd", &link).unwrap();

    let result = sandbox.validate_path(&link);
    assert!(matches!(result, Err(SandboxError::PathEscapesSandbox(_))));
}
```

### Command Execution Tests

```rust
#[test]
fn test_execute_confined_to_sandbox_root() {
    let sandbox = LocalSandbox::new(temp_dir).unwrap();
    fs::write(temp_dir.join("test.txt"), "sandbox file").unwrap();

    let response = sandbox.execute_sync("cat test.txt", None);
    assert_eq!(response.exit_code, Some(0));
    assert!(response.output.contains("sandbox file"));
}

#[test]
fn test_execute_environment_sanitized() {
    let sandbox = LocalSandbox::new(temp_dir).unwrap();
    let response = sandbox.execute_sync("env | sort", None);

    let lines: Vec<&str> = response.output.lines().collect();
    assert_eq!(lines.len(), 2); // Only HOME and PATH
}
```

## Attack Vectors Mitigated

### 1. Parent Directory Traversal (`../`)

**Attack**: Access files outside sandbox via `../etc/passwd`

**Mitigation**: Path canonicalization resolves `..` segments, then `starts_with()` check fails

```rust
let escape = sandbox_root.join("../etc/passwd");
sandbox.validate_path(&escape) // Error: PathEscapesSandbox
```

### 2. Absolute Paths

**Attack**: Direct access via `/etc/passwd`

**Mitigation**: Canonicalization and `starts_with()` check

```rust
sandbox.validate_path("/etc/passwd") // Error: PathEscapesSandbox
```

### 3. Symlink Escape

**Attack**: Create symlink pointing outside sandbox

**Mitigation**: Canonicalization follows symlinks, exposing real path

```rust
symlink("/etc/passwd", sandbox_root.join("evil_link"));
sandbox.validate_path(sandbox_root.join("evil_link")) // Error: PathEscapesSandbox
```

### 4. Complex Path Manipulation

**Attack**: Mix of `.`, `..`, symlinks to confuse validation

**Mitigation**: Full canonicalization handles all cases

```rust
let complex = sandbox_root.join("./foo/../../../etc/passwd");
sandbox.validate_path(&complex) // Error: PathEscapesSandbox
```

## Usage Examples

### Basic Sandbox Creation

```rust
use rvagent_backends::{LocalSandbox, BaseSandbox};

// Create sandbox with auto-created root
let sandbox = LocalSandbox::new(PathBuf::from("/tmp/my_sandbox"))?;

// Validate paths before use
let safe_path = sandbox.validate_path(Path::new("/tmp/my_sandbox/file.txt"))?;
let content = fs::read_to_string(safe_path)?;
```

### Custom Configuration

```rust
use rvagent_backends::{LocalSandbox, SandboxConfig};

let config = SandboxConfig {
    timeout_secs: 60,
    max_output_size: 1024 * 1024, // 1MB
    work_dir: None,
};

let sandbox = LocalSandbox::new_with_config(root_path, config)?;
```

### Safe File Operations

```rust
// ALWAYS validate before filesystem access
fn safe_read_file(sandbox: &impl BaseSandbox, path: &str) -> Result<String, SandboxError> {
    let path = Path::new(path);

    // Validate path is within sandbox
    let validated_path = sandbox.validate_path(path)?;

    // Safe to read now
    Ok(fs::read_to_string(validated_path)
        .map_err(|e| SandboxError::IoError(e.to_string()))?)
}
```

## Integration with Backend Protocol

`LocalSandbox` implements both `BaseSandbox` and `SandboxBackend`:

```rust
#[async_trait]
impl SandboxBackend for LocalSandbox {
    async fn execute(&self, command: &str, timeout: Option<u32>) -> ExecuteResponse {
        self.execute_sync(command, timeout)
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn sandbox_root(&self) -> &Path {
        &self.root
    }
}
```

All file operations from `Backend` trait use validated paths.

## Testing

Run the comprehensive security test suite:

```bash
# All sandbox tests
cargo test -p rvagent-backends sandbox

# Security-specific tests
cargo test --test sandbox_security_tests

# With verbose output
cargo test -p rvagent-backends sandbox -- --nocapture
```

Expected: All 20+ security tests pass, covering:
- Path validation (allowed and rejected cases)
- Multiple escape vectors (parent dirs, symlinks, absolute paths)
- Command execution confinement
- Environment sanitization
- Output size limits

## Security Checklist

Before deploying a sandbox backend:

- [ ] `validate_path()` called before ALL filesystem operations
- [ ] Paths are canonicalized before validation
- [ ] `starts_with(sandbox_root)` check enforced
- [ ] `PathEscapesSandbox` errors returned on violations
- [ ] Command execution confined to sandbox root
- [ ] Environment sanitized (only safe variables)
- [ ] Output size limits enforced
- [ ] All security tests pass
- [ ] No mock-based tests (only real filesystem tests)

## Performance Characteristics

- **Path validation**: O(1) after canonicalization
- **Canonicalization**: Filesystem-dependent (typically <1ms)
- **Memory overhead**: ~100 bytes per sandbox instance
- **No caching**: Every operation validates (security > performance)

## Future Enhancements

Potential improvements (not required for C5):

1. **cgroups integration** for resource limits
2. **seccomp filters** for syscall restrictions
3. **namespace isolation** for stronger confinement
4. **Audit logging** for security events
5. **Policy-based validation** with custom rules

## References

- **ADR-103**: Review Amendments (C5 specification)
- **SEC-023**: Sandbox Path Restriction Contract
- **SEC-005**: Environment Sanitization
- `crates/rvAgent/rvagent-backends/src/sandbox.rs`: Implementation
- `tests/sandbox_security_tests.rs`: Security test suite

---

**Last Updated**: 2026-03-15
**Status**: ✅ Complete and tested
