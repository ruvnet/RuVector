use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;
use validator::Validate;

/// Maximum number of bytes accepted from a remote image URL.
pub const MAX_REMOTE_FETCH_BYTES: usize = 20 * 1024 * 1024;

/// Timeout applied to outbound requests made on behalf of API clients.
pub const OUTBOUND_TIMEOUT: Duration = Duration::from_secs(15);

/// Returns true only for globally routable unicast addresses.
pub(crate) fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            !(v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
                || v4.is_documentation()
                || o[0] == 0
                || (o[0] == 100 && (o[1] & 0xc0) == 64) // 100.64.0.0/10
                || (o[0] == 192 && o[1] == 0 && o[2] == 0) // 192.0.0.0/24
                || (o[0] == 198 && (o[1] & 0xfe) == 18) // 198.18.0.0/15
                || o[0] >= 240)
        }
        IpAddr::V6(v6) => {
            let seg = v6.segments();
            // IPv4-mapped, IPv4-compatible and NAT64 forms embed an IPv4 address.
            let embedded = seg[0..5] == [0, 0, 0, 0, 0] && (seg[5] == 0xffff || seg[5] == 0)
                || (seg[0] == 0x64 && seg[1] == 0xff9b && seg[2..6] == [0, 0, 0, 0]);
            if embedded && !v6.is_unspecified() && !v6.is_loopback() {
                let o = v6.octets();
                return is_public_ip(IpAddr::from([o[12], o[13], o[14], o[15]]));
            }
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (seg[0] & 0xfe00) == 0xfc00 // unique local fc00::/7
                || (seg[0] & 0xffc0) == 0xfe80 // link local fe80::/10
                || (seg[0] & 0xffc0) == 0xfec0 // site local fec0::/10
                || seg[0] == 0x2001 && seg[1] == 0x0db8) // documentation
        }
    }
}

/// Build an HTTP client for `url` that only connects to public addresses.
///
/// The host is resolved once, every resolved address must be public, and the
/// client is pinned to the checked address so a second DNS answer cannot
/// redirect it. Redirects are disabled and a timeout is applied.
pub(crate) async fn public_client(url: &str) -> anyhow::Result<(reqwest::Client, reqwest::Url)> {
    let parsed = reqwest::Url::parse(url)?;
    if !matches!(parsed.scheme(), "http" | "https") {
        anyhow::bail!("unsupported URL scheme");
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("URL has no host"))?
        .to_string();
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| anyhow::anyhow!("URL has no port"))?;

    let literal = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse::<IpAddr>()
        .ok();
    let addrs: Vec<SocketAddr> = match literal {
        Some(ip) => vec![SocketAddr::new(ip, port)],
        None => tokio::net::lookup_host((host.as_str(), port))
            .await?
            .collect(),
    };
    if addrs.is_empty() || !addrs.iter().all(|a| is_public_ip(a.ip())) {
        anyhow::bail!("URL resolves to a non-public address");
    }

    let mut builder = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(OUTBOUND_TIMEOUT);
    if literal.is_none() {
        builder = builder.resolve(&host, addrs[0]);
    }
    Ok((builder.build()?, parsed))
}

/// Read a response body, failing once it exceeds `max` bytes.
pub(crate) async fn read_bounded(
    mut response: reqwest::Response,
    max: usize,
) -> anyhow::Result<Vec<u8>> {
    if response
        .content_length()
        .map_or(false, |len| len > max as u64)
    {
        anyhow::bail!("remote content too large");
    }
    let mut out = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if out.len() + chunk.len() > max {
            anyhow::bail!("remote content too large");
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

/// Text/Image OCR request
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TextRequest {
    /// Image source (base64, URL, or multipart)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub src: Option<String>,

    /// Base64 encoded image data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base64: Option<String>,

    /// Image URL
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(url)]
    pub url: Option<String>,

    /// Request metadata
    #[serde(default)]
    pub metadata: RequestMetadata,
}

impl TextRequest {
    /// Get image data from request
    pub async fn get_image_data(&self) -> anyhow::Result<Vec<u8>> {
        if let Some(base64_data) = &self.base64 {
            // Decode base64
            use base64::Engine;
            let decoded = base64::engine::general_purpose::STANDARD.decode(base64_data)?;
            Ok(decoded)
        } else if let Some(url) = &self.url {
            // Download from URL (public addresses only, bounded size)
            let (client, parsed) = public_client(url).await?;
            let response = client.get(parsed).send().await?.error_for_status()?;
            read_bounded(response, MAX_REMOTE_FETCH_BYTES).await
        } else {
            anyhow::bail!("No image data provided")
        }
    }
}

/// Digital ink strokes request
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct StrokesRequest {
    /// Array of stroke data
    #[validate(length(min = 1))]
    pub strokes: Vec<Stroke>,

    /// Request metadata
    #[serde(default)]
    pub metadata: RequestMetadata,
}

/// Single stroke in digital ink
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stroke {
    /// X coordinates
    pub x: Vec<f64>,

    /// Y coordinates
    pub y: Vec<f64>,

    /// Optional timestamps
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t: Option<Vec<f64>>,
}

/// Legacy LaTeX equation request
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LatexRequest {
    /// Image source
    #[serde(skip_serializing_if = "Option::is_none")]
    pub src: Option<String>,

    /// Base64 encoded image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base64: Option<String>,

    /// Image URL
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(url)]
    pub url: Option<String>,

    /// Request metadata
    #[serde(default)]
    pub metadata: RequestMetadata,
}

/// PDF processing request
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PdfRequest {
    /// PDF file URL
    #[validate(url)]
    pub url: String,

    /// Conversion options
    #[serde(default)]
    pub options: PdfOptions,

    /// Webhook URL for completion notification
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(url)]
    pub webhook_url: Option<String>,

    /// Request metadata
    #[serde(default)]
    pub metadata: RequestMetadata,
}

/// PDF processing options
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PdfOptions {
    /// Output format
    #[serde(default = "default_format")]
    pub format: String,

    /// Enable OCR
    #[serde(default)]
    pub enable_ocr: bool,

    /// Include images
    #[serde(default = "default_true")]
    pub include_images: bool,

    /// Page range (e.g., "1-5")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_range: Option<String>,
}

fn default_format() -> String {
    "mmd".to_string()
}

fn default_true() -> bool {
    true
}

/// Request metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RequestMetadata {
    /// Output formats
    #[serde(default = "default_formats")]
    pub formats: Vec<String>,

    /// Include confidence scores
    #[serde(default)]
    pub include_confidence: bool,

    /// Enable math mode
    #[serde(default = "default_true")]
    pub enable_math: bool,

    /// Language hint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

fn default_formats() -> Vec<String> {
    vec!["text".to_string()]
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn test_text_request_validation() {
        let request = TextRequest {
            src: None,
            base64: Some("SGVsbG8gV29ybGQ=".to_string()),
            url: None,
            metadata: RequestMetadata::default(),
        };

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_strokes_request_validation() {
        let request = StrokesRequest {
            strokes: vec![Stroke {
                x: vec![0.0, 1.0, 2.0],
                y: vec![0.0, 1.0, 0.0],
                t: None,
            }],
            metadata: RequestMetadata::default(),
        };

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_empty_strokes_validation() {
        let request = StrokesRequest {
            strokes: vec![],
            metadata: RequestMetadata::default(),
        };

        assert!(request.validate().is_err());
    }

    #[test]
    fn test_pdf_request_validation() {
        let request = PdfRequest {
            url: "https://example.com/document.pdf".to_string(),
            options: PdfOptions::default(),
            webhook_url: None,
            metadata: RequestMetadata::default(),
        };

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_invalid_url() {
        let request = PdfRequest {
            url: "not-a-url".to_string(),
            options: PdfOptions::default(),
            webhook_url: None,
            metadata: RequestMetadata::default(),
        };

        assert!(request.validate().is_err());
    }

    /// Spawn a loopback HTTP listener that answers one request with `body`.
    /// Returns its URL and a receiver that yields `true` if it was contacted.
    pub(crate) fn loopback_listener(body: Vec<u8>) -> (String, std::sync::mpsc::Receiver<bool>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/img.png", listener.local_addr().unwrap());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
            while std::time::Instant::now() < deadline {
                if let Ok((mut sock, _)) = listener.accept() {
                    let _ = sock.set_nonblocking(false);
                    let mut buf = [0u8; 4096];
                    let _ = sock.read(&mut buf);
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = sock.write_all(head.as_bytes());
                    let _ = sock.write_all(&body);
                    let _ = tx.send(true);
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            let _ = tx.send(false);
        });
        (url, rx)
    }

    #[tokio::test]
    async fn test_image_url_to_loopback_is_rejected() {
        let (url, contacted) = loopback_listener(b"internal-only".to_vec());
        let request = TextRequest {
            src: None,
            base64: None,
            url: Some(url),
            metadata: RequestMetadata::default(),
        };

        assert!(request.get_image_data().await.is_err());
        assert!(
            !contacted.recv().unwrap(),
            "loopback service must not be contacted"
        );
    }

    #[test]
    fn test_non_public_addresses_are_rejected() {
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "::1",
            "::",
            "fd00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "::ffff:169.254.169.254",
            "64:ff9b::a9fe:a9fe",
        ] {
            assert!(!is_public_ip(ip.parse().unwrap()), "{ip} must be rejected");
        }
        for ip in ["93.184.216.34", "2606:4700::1111", "::ffff:8.8.8.8"] {
            assert!(is_public_ip(ip.parse().unwrap()), "{ip} must be allowed");
        }
    }

    #[tokio::test]
    async fn test_oversized_remote_body_is_rejected() {
        let (url, _contacted) = loopback_listener(vec![0u8; 4096]);
        let response = reqwest::get(&url).await.unwrap();
        assert!(read_bounded(response, 1024).await.is_err());
    }
}
