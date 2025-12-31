//! Node identity management with Ed25519 keypairs

use wasm_bindgen::prelude::*;
use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};
use sha2::{Sha256, Digest};
use rand::rngs::OsRng;

/// Node identity with Ed25519 keypair
#[wasm_bindgen]
pub struct WasmNodeIdentity {
    signing_key: SigningKey,
    node_id: String,
    site_id: String,
    fingerprint: Option<String>,
}

#[wasm_bindgen]
impl WasmNodeIdentity {
    /// Generate a new node identity
    #[wasm_bindgen]
    pub fn generate(site_id: &str) -> Result<WasmNodeIdentity, JsValue> {
        let mut csprng = OsRng;
        let signing_key = SigningKey::generate(&mut csprng);

        // Derive node ID from public key
        let verifying_key = signing_key.verifying_key();
        let node_id = Self::derive_node_id(&verifying_key);

        Ok(WasmNodeIdentity {
            signing_key,
            node_id,
            site_id: site_id.to_string(),
            fingerprint: None,
        })
    }

    /// Restore identity from secret key bytes
    #[wasm_bindgen(js_name = fromSecretKey)]
    pub fn from_secret_key(secret_key: &[u8], site_id: &str) -> Result<WasmNodeIdentity, JsValue> {
        if secret_key.len() != 32 {
            return Err(JsValue::from_str("Secret key must be 32 bytes"));
        }

        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(secret_key);

        let signing_key = SigningKey::from_bytes(&key_bytes);
        let verifying_key = signing_key.verifying_key();
        let node_id = Self::derive_node_id(&verifying_key);

        Ok(WasmNodeIdentity {
            signing_key,
            node_id,
            site_id: site_id.to_string(),
            fingerprint: None,
        })
    }

    /// Get the node's unique identifier
    #[wasm_bindgen(js_name = nodeId)]
    pub fn node_id(&self) -> String {
        self.node_id.clone()
    }

    /// Get the site ID
    #[wasm_bindgen(js_name = siteId)]
    pub fn site_id(&self) -> String {
        self.site_id.clone()
    }

    /// Get the public key as hex string
    #[wasm_bindgen(js_name = publicKeyHex)]
    pub fn public_key_hex(&self) -> String {
        hex::encode(self.signing_key.verifying_key().as_bytes())
    }

    /// Get the public key as bytes
    #[wasm_bindgen(js_name = publicKeyBytes)]
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.signing_key.verifying_key().as_bytes().to_vec()
    }

    /// Export secret key (for backup)
    #[wasm_bindgen(js_name = exportSecretKey)]
    pub fn export_secret_key(&self) -> Vec<u8> {
        self.signing_key.to_bytes().to_vec()
    }

    /// Sign a message
    #[wasm_bindgen]
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        let signature = self.signing_key.sign(message);
        signature.to_bytes().to_vec()
    }

    /// Verify a signature
    #[wasm_bindgen]
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> bool {
        if signature.len() != 64 {
            return false;
        }

        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(signature);

        match Signature::from_bytes(&sig_bytes) {
            sig => self.signing_key.verifying_key().verify(message, &sig).is_ok(),
        }
    }

    /// Verify a signature from another node
    #[wasm_bindgen(js_name = verifyFrom)]
    pub fn verify_from(public_key: &[u8], message: &[u8], signature: &[u8]) -> bool {
        if public_key.len() != 32 || signature.len() != 64 {
            return false;
        }

        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(public_key);

        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(signature);

        let verifying_key = match VerifyingKey::from_bytes(&key_bytes) {
            Ok(k) => k,
            Err(_) => return false,
        };

        let signature = Signature::from_bytes(&sig_bytes);
        verifying_key.verify(message, &signature).is_ok()
    }

    /// Set browser fingerprint for anti-sybil
    #[wasm_bindgen(js_name = setFingerprint)]
    pub fn set_fingerprint(&mut self, fingerprint: &str) {
        self.fingerprint = Some(fingerprint.to_string());
    }

    /// Get browser fingerprint
    #[wasm_bindgen(js_name = getFingerprint)]
    pub fn get_fingerprint(&self) -> Option<String> {
        self.fingerprint.clone()
    }

    /// Derive node ID from public key
    fn derive_node_id(verifying_key: &VerifyingKey) -> String {
        let mut hasher = Sha256::new();
        hasher.update(verifying_key.as_bytes());
        let hash = hasher.finalize();

        // Use first 16 bytes as node ID (base58 encoded)
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&hash[..16]);

        // Simple hex encoding for now
        format!("node-{}", hex::encode(&id_bytes[..8]))
    }
}

/// Browser fingerprint generator for anti-sybil protection
#[wasm_bindgen]
pub struct BrowserFingerprint;

#[wasm_bindgen]
impl BrowserFingerprint {
    /// Generate anonymous uniqueness score
    /// This doesn't track users, just ensures one node per browser
    #[wasm_bindgen]
    pub async fn generate() -> Result<String, JsValue> {
        let window = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window object"))?;

        let navigator = window.navigator();
        let screen = window.screen()
            .map_err(|_| JsValue::from_str("No screen object"))?;

        let mut components = Vec::new();

        // Hardware signals (non-identifying)
        components.push(format!("{}", navigator.hardware_concurrency()));
        components.push(format!("{}x{}", screen.width().unwrap_or(0), screen.height().unwrap_or(0)));

        // Timezone offset
        let date = js_sys::Date::new_0();
        components.push(format!("{}", date.get_timezone_offset()));

        // Language
        if let Some(lang) = navigator.language() {
            components.push(lang);
        }

        // Platform
        if let Ok(platform) = navigator.platform() {
            components.push(platform);
        }

        // Hash all components
        let combined = components.join("|");
        let mut hasher = Sha256::new();
        hasher.update(combined.as_bytes());
        let hash = hasher.finalize();

        Ok(hex::encode(hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_generation() {
        let identity = WasmNodeIdentity::generate("test-site").unwrap();
        assert!(identity.node_id().starts_with("node-"));
        assert_eq!(identity.site_id(), "test-site");
    }

    #[test]
    fn test_sign_verify() {
        let identity = WasmNodeIdentity::generate("test-site").unwrap();
        let message = b"Hello, EdgeNet!";

        let signature = identity.sign(message);
        assert_eq!(signature.len(), 64);

        let is_valid = identity.verify(message, &signature);
        assert!(is_valid);

        // Tampered message should fail
        let is_valid = identity.verify(b"Tampered", &signature);
        assert!(!is_valid);
    }

    #[test]
    fn test_export_import() {
        let identity1 = WasmNodeIdentity::generate("test-site").unwrap();
        let secret_key = identity1.export_secret_key();

        let identity2 = WasmNodeIdentity::from_secret_key(&secret_key, "test-site").unwrap();

        assert_eq!(identity1.node_id(), identity2.node_id());
        assert_eq!(identity1.public_key_hex(), identity2.public_key_hex());
    }
}
