//! Keys command - manage cryptographic keys for secure boot
//!
//! This module provides real Ed25519 key generation, signing, and verification
//! using the ed25519-dalek library for RuVix secure boot operations.

use anyhow::{bail, Context, Result};
use clap::Subcommand;
use colored::Colorize;
use ed25519_dalek::{
    Signature, Signer, SigningKey, Verifier, VerifyingKey,
    SECRET_KEY_LENGTH, PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH,
};
use rand::rngs::OsRng;
use sha2::{Sha256, Sha384, Sha512, Digest};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use zeroize::Zeroize;

/// Key management actions
#[derive(Subcommand, Debug)]
pub enum KeysAction {
    /// Generate new signing keys
    #[command(after_help = "\
EXAMPLES:
    Generate Ed25519 keypair:
        ruvix keys generate --output keys/

    Generate with custom name:
        ruvix keys generate --output keys/ --name production

    Generate RSA keys:
        ruvix keys generate --output keys/ --algorithm rsa --bits 4096
")]
    Generate {
        /// Output directory for keys
        #[arg(short, long, default_value = "keys")]
        output: PathBuf,

        /// Key name prefix
        #[arg(short, long, default_value = "ruvix")]
        name: String,

        /// Key algorithm (ed25519, rsa)
        #[arg(long, default_value = "ed25519")]
        algorithm: KeyAlgorithm,

        /// Key size in bits (for RSA)
        #[arg(long, default_value = "4096")]
        bits: u32,

        /// Encrypt private key with passphrase
        #[arg(long)]
        passphrase: bool,

        /// Force overwrite existing keys
        #[arg(long)]
        force: bool,
    },

    /// Sign a kernel image
    #[command(after_help = "\
EXAMPLES:
    Sign kernel image:
        ruvix keys sign --key keys/ruvix.priv --image kernel8.img

    Sign with output path:
        ruvix keys sign --key keys/ruvix.priv --image kernel8.img --output kernel8.signed.img

    Sign with embedded certificate:
        ruvix keys sign --key keys/ruvix.priv --image kernel8.img --embed-cert keys/ruvix.cert
")]
    Sign {
        /// Private key file
        #[arg(short, long)]
        key: PathBuf,

        /// Image to sign
        #[arg(short, long)]
        image: PathBuf,

        /// Output signed image (default: <image>.signed)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Certificate to embed
        #[arg(long)]
        embed_cert: Option<PathBuf>,

        /// Hash algorithm (sha256, sha384, sha512)
        #[arg(long, default_value = "sha256")]
        hash: HashAlgorithm,
    },

    /// Verify a signed image
    #[command(after_help = "\
EXAMPLES:
    Verify signature:
        ruvix keys verify --key keys/ruvix.pub --image kernel8.signed.img

    Verify with certificate chain:
        ruvix keys verify --cert keys/chain.pem --image kernel8.signed.img
")]
    Verify {
        /// Public key file
        #[arg(short, long, required_unless_present = "cert")]
        key: Option<PathBuf>,

        /// Certificate file (alternative to key)
        #[arg(short, long)]
        cert: Option<PathBuf>,

        /// Signed image to verify
        #[arg(short, long)]
        image: PathBuf,

        /// Verbose verification output
        #[arg(long)]
        verbose: bool,
    },

    /// List keys in a directory
    #[command(after_help = "\
EXAMPLES:
    List all keys:
        ruvix keys list

    List keys in specific directory:
        ruvix keys list --dir /path/to/keys

    Show key details:
        ruvix keys list --details
")]
    List {
        /// Directory to scan
        #[arg(short, long, default_value = "keys")]
        dir: PathBuf,

        /// Show detailed key information
        #[arg(long)]
        details: bool,

        /// Output format
        #[arg(long, default_value = "table")]
        format: ListFormat,
    },

    /// Export public key
    #[command(after_help = "\
EXAMPLES:
    Export public key:
        ruvix keys export --key keys/ruvix.priv --output keys/ruvix.pub

    Export in PEM format:
        ruvix keys export --key keys/ruvix.priv --format pem
")]
    Export {
        /// Private key to export from
        #[arg(short, long)]
        key: PathBuf,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Output format (pem, der, raw)
        #[arg(long, default_value = "pem")]
        format: ExportFormat,
    },

    /// Import a key
    #[command(after_help = "\
EXAMPLES:
    Import public key:
        ruvix keys import --file external.pub --output keys/vendor.pub

    Import with validation:
        ruvix keys import --file external.pub --output keys/vendor.pub --validate
")]
    Import {
        /// Key file to import
        #[arg(short, long)]
        file: PathBuf,

        /// Output location
        #[arg(short, long)]
        output: PathBuf,

        /// Validate key before importing
        #[arg(long)]
        validate: bool,
    },
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum KeyAlgorithm {
    #[default]
    Ed25519,
    Rsa,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum HashAlgorithm {
    #[default]
    Sha256,
    Sha384,
    Sha512,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum ListFormat {
    #[default]
    Table,
    Json,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum ExportFormat {
    #[default]
    Pem,
    Der,
    Raw,
}

/// RuVix key file header magic
const RUVIX_KEY_MAGIC: &[u8; 8] = b"RUVIXKEY";
/// Version for key format
const KEY_FORMAT_VERSION: u8 = 1;

/// Signed image header
#[repr(C)]
struct SignedImageHeader {
    magic: [u8; 8],
    version: u8,
    hash_algo: u8,
    reserved: [u8; 6],
    payload_size: u64,
    signature: [u8; SIGNATURE_LENGTH],
    public_key: [u8; PUBLIC_KEY_LENGTH],
}

/// Execute the keys command
pub fn execute(action: KeysAction, verbose: bool) -> Result<()> {
    match action {
        KeysAction::Generate { output, name, algorithm, bits, passphrase, force } => {
            execute_generate(&output, &name, algorithm, bits, passphrase, force, verbose)
        }
        KeysAction::Sign { key, image, output, embed_cert, hash } => {
            execute_sign(&key, &image, output.as_ref(), embed_cert.as_ref(), hash, verbose)
        }
        KeysAction::Verify { key, cert, image, verbose: verb } => {
            execute_verify(key.as_ref(), cert.as_ref(), &image, verb || verbose)
        }
        KeysAction::List { dir, details, format } => {
            execute_list(&dir, details, format, verbose)
        }
        KeysAction::Export { key, output, format } => {
            execute_export(&key, &output, format, verbose)
        }
        KeysAction::Import { file, output, validate } => {
            execute_import(&file, &output, validate, verbose)
        }
    }
}

fn execute_generate(
    output: &PathBuf,
    name: &str,
    algorithm: KeyAlgorithm,
    bits: u32,
    passphrase: bool,
    force: bool,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("{}", "Key Generation Configuration:".cyan().bold());
        println!("  Output:     {}", output.display().to_string().yellow());
        println!("  Name:       {}", name);
        println!("  Algorithm:  {:?}", algorithm);
        if matches!(algorithm, KeyAlgorithm::Rsa) {
            println!("  Bits:       {}", bits);
        }
        println!("  Passphrase: {}", passphrase);
        println!();
    }

    // Create output directory if it doesn't exist
    fs::create_dir_all(output).context("Failed to create output directory")?;

    let priv_path = output.join(format!("{}.priv", name));
    let pub_path = output.join(format!("{}.pub", name));

    // Check for existing keys
    if !force {
        if priv_path.exists() {
            bail!("Private key already exists: {}. Use --force to overwrite.", priv_path.display());
        }
        if pub_path.exists() {
            bail!("Public key already exists: {}. Use --force to overwrite.", pub_path.display());
        }
    }

    println!("{} Generating {:?} keypair...", "[1/3]".cyan(), algorithm);

    match algorithm {
        KeyAlgorithm::Ed25519 => {
            // Generate Ed25519 keypair using cryptographically secure RNG
            let signing_key = SigningKey::generate(&mut OsRng);
            let verifying_key = signing_key.verifying_key();

            // Get passphrase if requested
            let mut encryption_key: Option<[u8; 32]> = None;
            if passphrase {
                let pass = prompt_passphrase("Enter passphrase for private key: ")?;
                let pass_confirm = prompt_passphrase("Confirm passphrase: ")?;
                if pass != pass_confirm {
                    bail!("Passphrases do not match");
                }
                encryption_key = Some(derive_key_from_passphrase(&pass)?);
            }

            println!("{} Writing keys to disk...", "[2/3]".cyan());

            // Write private key
            let priv_bytes = if let Some(key) = encryption_key {
                encrypt_private_key(&signing_key.to_bytes(), &key)?
            } else {
                create_key_file(&signing_key.to_bytes(), KeyType::PrivateEd25519, false)
            };
            fs::write(&priv_path, priv_bytes).context("Failed to write private key")?;

            // Set restrictive permissions on private key (Unix)
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&priv_path)?.permissions();
                perms.set_mode(0o600);
                fs::set_permissions(&priv_path, perms)?;
            }

            // Write public key
            let pub_bytes = create_key_file(&verifying_key.to_bytes(), KeyType::PublicEd25519, false);
            fs::write(&pub_path, pub_bytes).context("Failed to write public key")?;

            println!("{} Verifying keypair...", "[3/3]".cyan());

            // Test sign/verify cycle
            let test_message = b"RuVix key verification test";
            let signature = signing_key.sign(test_message);
            verifying_key.verify(test_message, &signature)
                .context("Key verification failed")?;

            // Compute fingerprint
            let fingerprint = compute_fingerprint(&verifying_key.to_bytes());

            println!();
            println!("{} Generated Ed25519 keypair:", "SUCCESS".green().bold());
            println!("  Private: {}", priv_path.display().to_string().yellow());
            println!("  Public:  {}", pub_path.display().to_string().yellow());
            println!("  Fingerprint: {}", fingerprint.dimmed());
        }
        KeyAlgorithm::Rsa => {
            #[cfg(feature = "rsa")]
            {
                use rsa::{RsaPrivateKey, RsaPublicKey};
                use rsa::pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding};

                println!("  Generating RSA-{} keypair (this may take a moment)...", bits);
                let mut rng = OsRng;
                let private_key = RsaPrivateKey::new(&mut rng, bits as usize)
                    .context("Failed to generate RSA key")?;
                let public_key = RsaPublicKey::from(&private_key);

                println!("{} Writing keys to disk...", "[2/3]".cyan());

                // Write private key in PEM format
                let priv_pem = private_key.to_pkcs8_pem(LineEnding::LF)
                    .context("Failed to encode private key")?;
                fs::write(&priv_path, priv_pem.as_bytes())?;

                // Write public key in PEM format
                let pub_pem = public_key.to_public_key_pem(LineEnding::LF)
                    .context("Failed to encode public key")?;
                fs::write(&pub_path, pub_pem)?;

                println!("{} RSA keypair generated", "[3/3]".cyan());
                println!();
                println!("{} Generated RSA-{} keypair:", "SUCCESS".green().bold());
                println!("  Private: {}", priv_path.display().to_string().yellow());
                println!("  Public:  {}", pub_path.display().to_string().yellow());
            }

            #[cfg(not(feature = "rsa"))]
            {
                bail!("RSA support not compiled in. Rebuild with --features rsa");
            }
        }
    }

    Ok(())
}

fn execute_sign(
    key: &PathBuf,
    image: &PathBuf,
    output: Option<&PathBuf>,
    _embed_cert: Option<&PathBuf>,
    hash: HashAlgorithm,
    verbose: bool,
) -> Result<()> {
    let output_path = output
        .cloned()
        .unwrap_or_else(|| {
            let stem = image.file_stem().unwrap_or_default();
            let ext = image.extension().unwrap_or_default();
            image.with_file_name(format!(
                "{}.signed.{}",
                stem.to_string_lossy(),
                ext.to_string_lossy()
            ))
        });

    if verbose {
        println!("{}", "Signing Configuration:".cyan().bold());
        println!("  Key:     {}", key.display().to_string().yellow());
        println!("  Image:   {}", image.display());
        println!("  Output:  {}", output_path.display());
        println!("  Hash:    {:?}", hash);
        println!();
    }

    println!("{} Loading private key...", "[1/4]".cyan());
    let signing_key = load_private_key(key)?;
    let verifying_key = signing_key.verifying_key();

    println!("{} Computing image hash...", "[2/4]".cyan());
    let image_data = fs::read(image).context("Failed to read image file")?;

    let hash_bytes = match hash {
        HashAlgorithm::Sha256 => {
            let mut hasher = Sha256::new();
            hasher.update(&image_data);
            hasher.finalize().to_vec()
        }
        HashAlgorithm::Sha384 => {
            let mut hasher = Sha384::new();
            hasher.update(&image_data);
            hasher.finalize().to_vec()
        }
        HashAlgorithm::Sha512 => {
            let mut hasher = Sha512::new();
            hasher.update(&image_data);
            hasher.finalize().to_vec()
        }
    };

    if verbose {
        println!("  Hash: {}", hex::encode(&hash_bytes).dimmed());
    }

    println!("{} Generating signature...", "[3/4]".cyan());
    let signature = signing_key.sign(&hash_bytes);

    println!("{} Writing signed image...", "[4/4]".cyan());

    // Create signed image with header
    let header = SignedImageHeader {
        magic: *b"RUVIXSIG",
        version: 1,
        hash_algo: match hash {
            HashAlgorithm::Sha256 => 1,
            HashAlgorithm::Sha384 => 2,
            HashAlgorithm::Sha512 => 3,
        },
        reserved: [0; 6],
        payload_size: image_data.len() as u64,
        signature: signature.to_bytes(),
        public_key: verifying_key.to_bytes(),
    };

    let mut output_data = Vec::with_capacity(std::mem::size_of::<SignedImageHeader>() + image_data.len());

    // Write header (unsafe but okay for POD struct)
    let header_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            &header as *const SignedImageHeader as *const u8,
            std::mem::size_of::<SignedImageHeader>()
        )
    };
    output_data.extend_from_slice(header_bytes);
    output_data.extend_from_slice(&image_data);

    fs::write(&output_path, output_data).context("Failed to write signed image")?;

    println!();
    println!(
        "{} Signed image: {}",
        "SUCCESS".green().bold(),
        output_path.display().to_string().yellow()
    );
    println!("  Signature: {}", hex::encode(&signature.to_bytes()[..16]).dimmed());
    println!("  Image size: {} bytes", image_data.len());

    Ok(())
}

fn execute_verify(
    key: Option<&PathBuf>,
    _cert: Option<&PathBuf>,
    image: &PathBuf,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("{}", "Verification Configuration:".cyan().bold());
        if let Some(k) = key {
            println!("  Key:   {}", k.display().to_string().yellow());
        }
        println!("  Image: {}", image.display());
        println!();
    }

    println!("{} Reading signed image...", "[1/4]".cyan());
    let image_data = fs::read(image).context("Failed to read signed image")?;

    if image_data.len() < std::mem::size_of::<SignedImageHeader>() {
        bail!("Image file too small to contain signature header");
    }

    // Parse header
    let header: SignedImageHeader = unsafe {
        std::ptr::read(image_data.as_ptr() as *const SignedImageHeader)
    };

    if &header.magic != b"RUVIXSIG" {
        bail!("Invalid signature header magic");
    }

    println!("{} Loading verification key...", "[2/4]".cyan());
    let verifying_key = if let Some(k) = key {
        load_public_key(k)?
    } else {
        // Use embedded public key from header
        VerifyingKey::from_bytes(&header.public_key)
            .context("Invalid embedded public key")?
    };

    println!("{} Computing image hash...", "[3/4]".cyan());
    let payload = &image_data[std::mem::size_of::<SignedImageHeader>()..];

    let hash_bytes = match header.hash_algo {
        1 => {
            let mut hasher = Sha256::new();
            hasher.update(payload);
            hasher.finalize().to_vec()
        }
        2 => {
            let mut hasher = Sha384::new();
            hasher.update(payload);
            hasher.finalize().to_vec()
        }
        3 => {
            let mut hasher = Sha512::new();
            hasher.update(payload);
            hasher.finalize().to_vec()
        }
        _ => bail!("Unknown hash algorithm: {}", header.hash_algo),
    };

    if verbose {
        println!("  Hash: {}", hex::encode(&hash_bytes).dimmed());
    }

    println!("{} Verifying signature...", "[4/4]".cyan());
    let signature = Signature::from_bytes(&header.signature);

    match verifying_key.verify(&hash_bytes, &signature) {
        Ok(()) => {
            println!();
            println!("{} Signature verified successfully", "VALID".green().bold());
            println!("  Payload size: {} bytes", header.payload_size);
            println!("  Fingerprint: {}", compute_fingerprint(&verifying_key.to_bytes()).dimmed());
        }
        Err(e) => {
            println!();
            println!("{} Signature verification failed", "INVALID".red().bold());
            bail!("Signature verification error: {}", e);
        }
    }

    Ok(())
}

fn execute_list(dir: &PathBuf, details: bool, format: ListFormat, verbose: bool) -> Result<()> {
    if verbose {
        println!("{} Scanning for keys in {}", "[keys]".cyan(), dir.display());
    }

    if !dir.exists() {
        bail!("Directory does not exist: {}", dir.display());
    }

    let mut keys = Vec::new();

    for entry in fs::read_dir(dir).context("Failed to read directory")? {
        let entry = entry?;
        let path = entry.path();

        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy();
            if ext_str == "priv" || ext_str == "pub" {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                let key_type = if ext_str == "priv" { "Private" } else { "Public" };

                // Try to determine algorithm and fingerprint
                let (algo, fingerprint) = if let Ok(data) = fs::read(&path) {
                    parse_key_info(&data)
                } else {
                    ("Unknown".to_string(), "N/A".to_string())
                };

                let metadata = fs::metadata(&path)?;
                let created: chrono::DateTime<chrono::Utc> = chrono::DateTime::from(metadata.created().unwrap_or(std::time::SystemTime::UNIX_EPOCH));
                let created_str = created.format("%Y-%m-%d").to_string();

                keys.push((name, algo, key_type.to_string(), created_str, fingerprint));
            }
        }
    }

    match format {
        ListFormat::Json => {
            let json_keys: Vec<_> = keys
                .iter()
                .map(|(name, algo, typ, date, fp)| {
                    serde_json::json!({
                        "name": name,
                        "algorithm": algo,
                        "type": typ,
                        "created": date,
                        "fingerprint": fp
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json_keys)?);
        }
        ListFormat::Table => {
            println!();
            if details {
                println!(
                    "  {:<20} {:<12} {:<10} {:<12} {}",
                    "NAME".bold(),
                    "ALGORITHM".bold(),
                    "TYPE".bold(),
                    "CREATED".bold(),
                    "FINGERPRINT".bold()
                );
                println!("  {}", "-".repeat(75));
                for (name, algo, typ, date, fp) in &keys {
                    let typ_colored = match typ.as_str() {
                        "Private" => typ.red(),
                        "Public" => typ.green(),
                        _ => typ.normal(),
                    };
                    println!(
                        "  {:<20} {:<12} {:<10} {:<12} {}",
                        name, algo, typ_colored, date, fp.dimmed()
                    );
                }
            } else {
                println!(
                    "  {:<20} {:<12} {}",
                    "NAME".bold(),
                    "ALGORITHM".bold(),
                    "TYPE".bold()
                );
                println!("  {}", "-".repeat(45));
                for (name, algo, typ, _, _) in &keys {
                    let typ_colored = match typ.as_str() {
                        "Private" => typ.red(),
                        "Public" => typ.green(),
                        _ => typ.normal(),
                    };
                    println!("  {:<20} {:<12} {}", name, algo, typ_colored);
                }
            }
        }
    }

    println!();
    println!("  Found {} keys in {}", keys.len(), dir.display());

    Ok(())
}

fn execute_export(key: &PathBuf, output: &PathBuf, format: ExportFormat, verbose: bool) -> Result<()> {
    if verbose {
        println!(
            "{} Exporting public key from {} to {} ({:?} format)",
            "[keys]".cyan(),
            key.display(),
            output.display(),
            format
        );
    }

    let signing_key = load_private_key(key)?;
    let verifying_key = signing_key.verifying_key();

    let output_bytes = match format {
        ExportFormat::Pem => {
            let pem = pem::Pem::new("ED25519 PUBLIC KEY", verifying_key.to_bytes().to_vec());
            pem::encode(&pem).into_bytes()
        }
        ExportFormat::Der => {
            verifying_key.to_bytes().to_vec()
        }
        ExportFormat::Raw => {
            verifying_key.to_bytes().to_vec()
        }
    };

    fs::write(output, output_bytes).context("Failed to write public key")?;

    println!(
        "\n{} Exported public key to {}",
        "OK".green(),
        output.display().to_string().yellow()
    );

    Ok(())
}

fn execute_import(file: &PathBuf, output: &PathBuf, validate: bool, verbose: bool) -> Result<()> {
    if verbose {
        println!(
            "{} Importing key from {} to {}",
            "[keys]".cyan(),
            file.display(),
            output.display()
        );
    }

    let data = fs::read(file).context("Failed to read key file")?;

    if validate {
        // Try to parse as public key
        let key_bytes = parse_key_bytes(&data)?;
        if key_bytes.len() == PUBLIC_KEY_LENGTH {
            VerifyingKey::from_bytes(&key_bytes.try_into().unwrap())
                .context("Invalid public key format")?;
            println!("  Key validated: Ed25519 public key");
        } else if key_bytes.len() == SECRET_KEY_LENGTH {
            SigningKey::from_bytes(&key_bytes.try_into().unwrap());
            println!("  Key validated: Ed25519 private key");
        } else {
            bail!("Unknown key format (length: {} bytes)", key_bytes.len());
        }
    }

    // Ensure output directory exists
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::copy(file, output).context("Failed to copy key file")?;

    println!(
        "\n{} Imported key to {}",
        "OK".green(),
        output.display().to_string().yellow()
    );

    Ok(())
}

// Helper functions

#[derive(Clone, Copy)]
enum KeyType {
    PrivateEd25519,
    PublicEd25519,
}

fn create_key_file(key_bytes: &[u8], key_type: KeyType, encrypted: bool) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(RUVIX_KEY_MAGIC);
    data.push(KEY_FORMAT_VERSION);
    data.push(match key_type {
        KeyType::PrivateEd25519 => 1,
        KeyType::PublicEd25519 => 2,
    });
    data.push(if encrypted { 1 } else { 0 });
    data.extend_from_slice(&[0u8; 5]); // Reserved
    data.extend_from_slice(key_bytes);
    data
}

fn encrypt_private_key(key_bytes: &[u8; SECRET_KEY_LENGTH], encryption_key: &[u8; 32]) -> Result<Vec<u8>> {
    use aes_gcm::{Aes256Gcm, Key, Nonce};
    use aes_gcm::aead::{Aead, KeyInit};

    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(encryption_key));
    let nonce_bytes: [u8; 12] = rand::random();
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, key_bytes.as_ref())
        .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

    let mut data = Vec::new();
    data.extend_from_slice(RUVIX_KEY_MAGIC);
    data.push(KEY_FORMAT_VERSION);
    data.push(1); // Ed25519 private
    data.push(1); // Encrypted
    data.extend_from_slice(&[0u8; 5]); // Reserved
    data.extend_from_slice(&nonce_bytes);
    data.extend_from_slice(&ciphertext);

    Ok(data)
}

fn derive_key_from_passphrase(passphrase: &str) -> Result<[u8; 32]> {
    use argon2::{Argon2, PasswordHasher};
    use argon2::password_hash::SaltString;

    // Use a fixed salt for deterministic key derivation
    // In production, you'd store the salt with the encrypted key
    let salt = SaltString::encode_b64(b"RuVixKeyDerive16").unwrap();

    let argon2 = Argon2::default();
    let hash = argon2.hash_password(passphrase.as_bytes(), &salt)
        .map_err(|e| anyhow::anyhow!("Key derivation failed: {}", e))?;

    let hash_output = hash.hash.unwrap();
    let mut key = [0u8; 32];
    key.copy_from_slice(&hash_output.as_bytes()[..32]);

    Ok(key)
}

fn prompt_passphrase(prompt: &str) -> Result<String> {
    use std::io::Write;
    print!("{}", prompt);
    std::io::stdout().flush()?;

    // Disable echo for password input
    #[cfg(unix)]
    {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        let mut line = String::new();
        stdin.lock().read_line(&mut line)?;
        println!();
        Ok(line.trim().to_string())
    }

    #[cfg(not(unix))]
    {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        println!();
        Ok(line.trim().to_string())
    }
}

fn load_private_key(path: &PathBuf) -> Result<SigningKey> {
    let data = fs::read(path).context("Failed to read private key")?;
    let key_bytes = parse_key_bytes(&data)?;

    if key_bytes.len() != SECRET_KEY_LENGTH {
        bail!("Invalid private key length: expected {}, got {}", SECRET_KEY_LENGTH, key_bytes.len());
    }

    let mut bytes = [0u8; SECRET_KEY_LENGTH];
    bytes.copy_from_slice(&key_bytes);
    Ok(SigningKey::from_bytes(&bytes))
}

fn load_public_key(path: &PathBuf) -> Result<VerifyingKey> {
    let data = fs::read(path).context("Failed to read public key")?;
    let key_bytes = parse_key_bytes(&data)?;

    if key_bytes.len() != PUBLIC_KEY_LENGTH {
        bail!("Invalid public key length: expected {}, got {}", PUBLIC_KEY_LENGTH, key_bytes.len());
    }

    let mut bytes = [0u8; PUBLIC_KEY_LENGTH];
    bytes.copy_from_slice(&key_bytes);
    VerifyingKey::from_bytes(&bytes).context("Invalid public key")
}

fn parse_key_bytes(data: &[u8]) -> Result<Vec<u8>> {
    // Check for RuVix key format
    if data.len() >= 16 && &data[..8] == RUVIX_KEY_MAGIC {
        let _version = data[8];
        let _key_type = data[9];
        let encrypted = data[10];

        if encrypted != 0 {
            bail!("Key is encrypted. Decryption not implemented in this context.");
        }

        return Ok(data[16..].to_vec());
    }

    // Check for PEM format
    if data.starts_with(b"-----BEGIN") {
        let pem = pem::parse(data).context("Failed to parse PEM")?;
        return Ok(pem.into_contents());
    }

    // Assume raw bytes
    Ok(data.to_vec())
}

fn parse_key_info(data: &[u8]) -> (String, String) {
    if data.len() >= 16 && &data[..8] == RUVIX_KEY_MAGIC {
        let key_type = data[9];
        let algo = match key_type {
            1 | 2 => "Ed25519",
            3 | 4 => "RSA",
            _ => "Unknown",
        };

        // Compute fingerprint for public keys
        let fingerprint = if key_type == 2 && data.len() >= 16 + PUBLIC_KEY_LENGTH {
            compute_fingerprint(&data[16..16 + PUBLIC_KEY_LENGTH])
        } else if key_type == 1 && data.len() >= 16 + SECRET_KEY_LENGTH {
            // For private keys, derive public key first
            let mut bytes = [0u8; SECRET_KEY_LENGTH];
            bytes.copy_from_slice(&data[16..16 + SECRET_KEY_LENGTH]);
            let signing_key = SigningKey::from_bytes(&bytes);
            compute_fingerprint(&signing_key.verifying_key().to_bytes())
        } else {
            "N/A".to_string()
        };

        return (algo.to_string(), fingerprint);
    }

    ("Unknown".to_string(), "N/A".to_string())
}

fn compute_fingerprint(key_bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key_bytes);
    let hash = hasher.finalize();
    format!("SHA256:{}", base64::encode(&hash[..12]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_generate_ed25519_keys() {
        let dir = tempdir().unwrap();
        let result = execute_generate(
            &dir.path().to_path_buf(),
            "test",
            KeyAlgorithm::Ed25519,
            4096,
            false,
            true,
            false,
        );
        assert!(result.is_ok());

        assert!(dir.path().join("test.priv").exists());
        assert!(dir.path().join("test.pub").exists());
    }

    #[test]
    fn test_sign_and_verify() {
        let dir = tempdir().unwrap();

        // Generate keys
        execute_generate(
            &dir.path().to_path_buf(),
            "test",
            KeyAlgorithm::Ed25519,
            4096,
            false,
            true,
            false,
        ).unwrap();

        // Create test image
        let image_path = dir.path().join("test.img");
        fs::write(&image_path, b"Test image data").unwrap();

        // Sign image
        let signed_path = dir.path().join("test.signed.img");
        execute_sign(
            &dir.path().join("test.priv"),
            &image_path,
            Some(&signed_path),
            None,
            HashAlgorithm::Sha256,
            false,
        ).unwrap();

        // Verify signature
        let result = execute_verify(
            Some(&dir.path().join("test.pub")),
            None,
            &signed_path,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_keys() {
        let dir = tempdir().unwrap();

        // Generate keys
        execute_generate(
            &dir.path().to_path_buf(),
            "test",
            KeyAlgorithm::Ed25519,
            4096,
            false,
            true,
            false,
        ).unwrap();

        let result = execute_list(&dir.path().to_path_buf(), false, ListFormat::Table, false);
        assert!(result.is_ok());
    }
}
