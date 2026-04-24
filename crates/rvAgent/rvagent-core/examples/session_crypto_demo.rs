//! Demonstration of session encryption at rest (C9).
//!
//! This example shows:
//! - Key generation and derivation
//! - Encrypting/decrypting session data
//! - Saving/loading encrypted sessions from files
//! - File permissions on Unix systems
//! - UUID-based unpredictable filenames

use rvagent_core::session_crypto::{
    derive_key, generate_key, generate_session_filename, SessionCrypto,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Session Encryption at Rest Demo (C9) ===\n");

    // 1. Generate random encryption key
    println!("1. Generating random AES-256 key...");
    let key = generate_key();
    println!("   Generated 32-byte key: {:02x?}...", &key[..8]);

    // 2. Derive key from password
    println!("\n2. Deriving key from password...");
    let salt = b"some_random_salt_value_here";
    let derived_key = derive_key("my_secure_password", salt);
    println!("   Derived key: {:02x?}...", &derived_key[..8]);

    // 3. Create session crypto instance
    println!("\n3. Creating SessionCrypto instance...");
    let crypto = SessionCrypto::new(&key);
    println!("   Ready to encrypt/decrypt");

    // 4. Encrypt session data
    println!("\n4. Encrypting session data...");
    let session_data = serde_json::json!({
        "user_id": "user_12345",
        "session_id": "sess_67890",
        "created_at": "2024-03-15T10:30:00Z",
        "state": {
            "current_task": "security_audit",
            "context": ["encryption", "authentication", "authorization"]
        }
    });
    let plaintext = serde_json::to_vec(&session_data)?;
    println!("   Plaintext size: {} bytes", plaintext.len());

    let encrypted = crypto.encrypt(&plaintext)?;
    println!(
        "   Encrypted size: {} bytes (includes 12-byte nonce + 16-byte auth tag)",
        encrypted.len()
    );
    println!("   Overhead: {} bytes", encrypted.len() - plaintext.len());

    // 5. Decrypt session data
    println!("\n5. Decrypting session data...");
    let decrypted = crypto.decrypt(&encrypted)?;
    let recovered_data: serde_json::Value = serde_json::from_slice(&decrypted)?;
    println!(
        "   Recovered data: {}",
        serde_json::to_string_pretty(&recovered_data)?
    );

    // 6. Demonstrate different nonces for same plaintext
    println!("\n6. Encrypting same data twice (different nonces)...");
    let encrypted1 = crypto.encrypt(&plaintext)?;
    let encrypted2 = crypto.encrypt(&plaintext)?;
    println!("   Ciphertext 1: {:02x?}...", &encrypted1[..20]);
    println!("   Ciphertext 2: {:02x?}...", &encrypted2[..20]);
    println!("   Are they different? {}", encrypted1 != encrypted2);

    // 7. Save encrypted session to file
    println!("\n7. Saving encrypted session to file...");
    let temp_dir = std::env::temp_dir();
    let filename = generate_session_filename();
    let session_path = temp_dir.join(&filename);
    println!("   Filename: {}", filename);
    println!("   Full path: {}", session_path.display());

    crypto.save_session(&session_path, &plaintext)?;
    println!("   Session saved successfully");

    // 8. Check file permissions (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = std::fs::metadata(&session_path)?;
        let permissions = metadata.permissions();
        let mode = permissions.mode();
        println!("   File permissions: {:o} (should be 600)", mode & 0o777);
    }

    // 9. Load encrypted session from file
    println!("\n8. Loading encrypted session from file...");
    let loaded_data = crypto.load_session(&session_path)?;
    let loaded_session: serde_json::Value = serde_json::from_slice(&loaded_data)?;
    println!(
        "   Loaded data matches original: {}",
        loaded_data == plaintext
    );
    println!(
        "   Loaded session: {}",
        serde_json::to_string_pretty(&loaded_session)?
    );

    // 10. Demonstrate wrong key failure
    println!("\n9. Testing decryption with wrong key...");
    let wrong_key = generate_key();
    let wrong_crypto = SessionCrypto::new(&wrong_key);
    match wrong_crypto.decrypt(&encrypted) {
        Ok(_) => println!("   ERROR: Should have failed!"),
        Err(e) => println!("   ✓ Decryption correctly failed: {}", e),
    }

    // 11. Demonstrate corrupted data detection
    println!("\n10. Testing corrupted data detection...");
    let mut corrupted = encrypted.clone();
    corrupted[20] ^= 0xFF; // Flip bits in ciphertext
    match crypto.decrypt(&corrupted) {
        Ok(_) => println!("   ERROR: Should have detected corruption!"),
        Err(e) => println!("   ✓ Corruption correctly detected: {}", e),
    }

    // Cleanup
    println!("\n11. Cleaning up...");
    std::fs::remove_file(&session_path)?;
    println!("   Temporary file removed");

    println!("\n=== Demo Complete ===");
    println!("\nKey features demonstrated:");
    println!("  ✓ AES-256-GCM authenticated encryption");
    println!("  ✓ Random nonce generation (96-bit)");
    println!("  ✓ SHA3-256 password-based key derivation");
    println!("  ✓ UUID-based unpredictable filenames");
    println!("  ✓ 0600 file permissions on Unix");
    println!("  ✓ Authentication tag verification (16 bytes)");
    println!("  ✓ Corruption and wrong-key detection");

    Ok(())
}
