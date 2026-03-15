//! Comprehensive security tests for C5: Sandbox Path Restriction Contract.
//!
//! Tests all path escape vectors and validates the mandatory security contract.
//! Run with: cargo test -p rvagent-backends --test sandbox_security_tests

#[cfg(test)]
mod sandbox_security {
    use rvagent_backends::{LocalSandbox, BaseSandbox, SandboxError};
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn test_validate_path_allows_files_within_sandbox() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Create test file
        let allowed_file = temp.path().join("allowed.txt");
        fs::write(&allowed_file, "safe content").unwrap();

        let result = sandbox.validate_path(&allowed_file);
        assert!(result.is_ok(), "Should allow files within sandbox");
        assert_eq!(result.unwrap(), allowed_file.canonicalize().unwrap());
    }

    #[test]
    fn test_validate_path_rejects_parent_directory_escape() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Attempt to escape via ../
        let escape_attempt = temp.path().join("../etc/passwd");

        let result = sandbox.validate_path(&escape_attempt);
        assert!(result.is_err(), "Should reject ../ escape attempts");

        match result {
            Err(SandboxError::PathEscapesSandbox(msg)) => {
                assert!(msg.contains("outside sandbox root"), "Error message should explain the violation");
            }
            _ => panic!("Expected PathEscapesSandbox error"),
        }
    }

    #[test]
    fn test_validate_path_rejects_multiple_parent_escapes() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        let escape_attempts = vec![
            temp.path().join(".."),
            temp.path().join("../.."),
            temp.path().join("../../.."),
            temp.path().join("foo/../../.."),
            temp.path().join("./../../etc"),
        ];

        for escape in escape_attempts {
            let result = sandbox.validate_path(&escape);
            assert!(
                result.is_err(),
                "Should reject escape: {}",
                escape.display()
            );
        }
    }

    #[test]
    fn test_validate_path_rejects_absolute_paths_outside_sandbox() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Absolute paths outside sandbox
        let outside_paths = vec![
            Path::new("/etc/passwd"),
            Path::new("/tmp/evil"),
            Path::new("/var/log/system.log"),
        ];

        for path in outside_paths {
            // This will fail either at canonicalize (if file doesn't exist)
            // or at starts_with check (if it does exist)
            let result = sandbox.validate_path(path);
            assert!(
                result.is_err(),
                "Should reject absolute path outside sandbox: {}",
                path.display()
            );
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_validate_path_rejects_symlink_escape() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Create symlink pointing outside sandbox
        let link_path = temp.path().join("evil_symlink");
        std::os::unix::fs::symlink("/etc/passwd", &link_path).unwrap();

        let result = sandbox.validate_path(&link_path);
        assert!(result.is_err(), "Should reject symlinks pointing outside sandbox");

        match result {
            Err(SandboxError::PathEscapesSandbox(msg)) => {
                assert!(msg.contains("outside sandbox root"));
            }
            _ => panic!("Expected PathEscapesSandbox error for symlink escape"),
        }
    }

    #[test]
    fn test_validate_path_allows_nested_directories() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Create deeply nested structure
        let nested = temp.path().join("level1/level2/level3");
        fs::create_dir_all(&nested).unwrap();
        let deep_file = nested.join("deep.txt");
        fs::write(&deep_file, "nested content").unwrap();

        let result = sandbox.validate_path(&deep_file);
        assert!(result.is_ok(), "Should allow deeply nested paths within sandbox");
    }

    #[test]
    fn test_validate_path_normalizes_dot_segments() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        let file = temp.path().join("test.txt");
        fs::write(&file, "test").unwrap();

        // Path with redundant ./ and .. segments that resolve within sandbox
        let weird_path = temp.path().join("./subdir/../test.txt");

        let result = sandbox.validate_path(&weird_path);
        assert!(result.is_ok(), "Should handle normalized paths");
        assert_eq!(result.unwrap(), file.canonicalize().unwrap());
    }

    #[test]
    fn test_execute_confined_to_sandbox_root() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Create file in sandbox
        fs::write(temp.path().join("test.txt"), "sandbox file").unwrap();

        // Command runs with cwd = sandbox root
        let response = sandbox.execute_sync("cat test.txt", None);
        assert_eq!(response.exit_code, Some(0));
        assert!(response.output.contains("sandbox file"));
    }

    #[test]
    fn test_execute_cannot_access_parent_directories() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        // Try to access parent directory
        let response = sandbox.execute_sync("cat ../etc/passwd", None);

        // Command should fail (path doesn't exist from sandbox perspective)
        assert_ne!(response.exit_code, Some(0));
        assert!(
            response.output.contains("No such file") || response.output.contains("cannot access")
        );
    }

    #[test]
    fn test_execute_environment_sanitized() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        let response = sandbox.execute_sync("env | sort", None);
        assert_eq!(response.exit_code, Some(0));

        // Only HOME and PATH should be set (SEC-005)
        let lines: Vec<&str> = response.output.lines().collect();
        assert_eq!(
            lines.len(),
            2,
            "Environment should only have HOME and PATH, found: {:?}",
            lines
        );
        assert!(lines.iter().any(|l| l.starts_with("HOME=")));
        assert!(lines.iter().any(|l| l.starts_with("PATH=")));
    }

    #[test]
    fn test_execute_respects_max_output_size() {
        let temp = TempDir::new().unwrap();
        let config = rvagent_backends::SandboxConfig {
            timeout_secs: 30,
            max_output_size: 100, // Very small limit
            work_dir: None,
        };
        let sandbox = LocalSandbox::new_with_config(temp.path().to_path_buf(), config).unwrap();

        // Generate output larger than limit
        let response = sandbox.execute_sync("seq 1 1000", None);
        assert_eq!(response.exit_code, Some(0));
        assert!(response.truncated, "Output should be truncated");
        assert_eq!(response.output.len(), 100);
    }

    #[test]
    fn test_is_path_confined_legacy_api() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        let allowed = temp.path().join("allowed.txt");
        fs::write(&allowed, "test").unwrap();

        assert!(sandbox.is_path_confined(&allowed));

        // Escape attempts
        assert!(!sandbox.is_path_confined(&temp.path().join("../etc/passwd")));
        assert!(!sandbox.is_path_confined(Path::new("/etc/passwd")));
    }

    #[test]
    fn test_sandbox_creation_creates_missing_root() {
        let temp = TempDir::new().unwrap();
        let new_root = temp.path().join("new_sandbox");

        assert!(!new_root.exists());

        let sandbox = LocalSandbox::new(new_root.clone()).unwrap();

        assert!(new_root.exists());
        assert!(new_root.is_dir());
        assert_eq!(sandbox.sandbox_root(), &new_root);
    }

    #[test]
    fn test_sandbox_rejects_file_as_root() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("not_a_dir");
        fs::write(&file, "test").unwrap();

        let result = LocalSandbox::new(file);
        assert!(result.is_err());

        match result {
            Err(SandboxError::InitializationFailed(msg)) => {
                assert!(msg.contains("not a directory"));
            }
            _ => panic!("Expected InitializationFailed error"),
        }
    }

    #[test]
    fn test_sandbox_id_is_unique() {
        let temp = TempDir::new().unwrap();
        let sandbox1 = LocalSandbox::new(temp.path().to_path_buf()).unwrap();
        let sandbox2 = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        assert_ne!(sandbox1.sandbox_id(), sandbox2.sandbox_id());
        assert!(!sandbox1.sandbox_id().is_empty());
    }

    #[test]
    fn test_validate_path_error_contains_helpful_message() {
        let temp = TempDir::new().unwrap();
        let sandbox = LocalSandbox::new(temp.path().to_path_buf()).unwrap();

        let escape = temp.path().join("../outside");
        let result = sandbox.validate_path(&escape);

        match result {
            Err(SandboxError::PathEscapesSandbox(msg)) => {
                assert!(msg.contains("outside sandbox root"));
                assert!(msg.contains(temp.path().to_str().unwrap()));
            }
            _ => panic!("Expected detailed error message"),
        }
    }
}
