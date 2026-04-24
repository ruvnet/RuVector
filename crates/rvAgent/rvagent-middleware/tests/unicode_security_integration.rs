//! Integration tests for Unicode Security Module (C7)
//!
//! Demonstrates comprehensive security checks against Unicode-based attacks.

use rvagent_middleware::{
    AgentState, Message, Middleware, PipelineConfig, RunnableConfig, Runtime, ToolCall,
    UnicodeSecurityChecker, UnicodeSecurityConfig, UnicodeSecurityMiddleware,
};

#[tokio::test]
async fn test_real_world_bidi_attack() {
    // Simulate a real BiDi override attack on a filename
    let mw = UnicodeSecurityMiddleware::strict()
        .with_input_sanitization(true)
        .with_output_sanitization(true);

    let state = AgentState {
        messages: vec![
            // Attacker tries to disguise evil.exe as safe.txt
            Message::tool(
                "Downloaded: safe\u{202E}exe.txt", // Displays as "safeexe.txt" (reversed)
                "tc-1",
                "filesystem",
            ),
        ],
        todos: vec![],
        extensions: Default::default(),
    };

    let runtime = Runtime::new();
    let config = RunnableConfig::default();

    let update = mw.abefore_agent(&state, &runtime, &config).await;
    assert!(update.is_some());

    let new_msgs = update.unwrap().messages.unwrap();
    // BiDi should be stripped
    assert!(!new_msgs[0].content.contains('\u{202E}'));
    assert_eq!(new_msgs[0].content, "Downloaded: safeexe.txt");
}

#[tokio::test]
async fn test_real_world_homoglyph_attack() {
    // Simulate a phishing attack with Cyrillic lookalikes
    let checker = UnicodeSecurityChecker::strict();

    // Attacker creates fake "paypal.com" with Cyrillic 'а' (U+0430)
    let phishing_url = "pаypal.com"; // Contains Cyrillic 'а'
    let issues = checker.check(phishing_url);

    // Should detect confusable and homoglyph attack
    let has_confusable = issues
        .iter()
        .any(|issue| matches!(issue, rvagent_middleware::UnicodeIssue::Confusable { .. }));
    let has_homoglyph = issues.iter().any(|issue| {
        matches!(
            issue,
            rvagent_middleware::UnicodeIssue::HomoglyphAttack { .. }
        )
    });

    assert!(has_confusable || has_homoglyph);
}

#[tokio::test]
async fn test_real_world_zero_width_steganography() {
    // Simulate hiding secret data in zero-width characters
    let mw = UnicodeSecurityMiddleware::strict()
        .with_user_input_check(true)
        .with_input_sanitization(true);

    // User input with hidden zero-width characters encoding secret data
    let state = AgentState {
        messages: vec![Message::user(
            "Innocent\u{200B}text\u{200C}with\u{200D}hidden\u{200B}data",
        )],
        todos: vec![],
        extensions: Default::default(),
    };

    let runtime = Runtime::new();
    let config = RunnableConfig::default();

    let update = mw.abefore_agent(&state, &runtime, &config).await;
    assert!(update.is_some());

    let new_msgs = update.unwrap().messages.unwrap();
    // All zero-width should be stripped
    assert_eq!(new_msgs[0].content, "Innocenttextwithhiddendata");
}

#[tokio::test]
async fn test_tool_call_argument_sanitization() {
    // Test that tool call arguments are checked for Unicode attacks
    let mw = UnicodeSecurityMiddleware::strict();

    let state = AgentState {
        messages: vec![{
            let mut msg = Message::assistant("");
            msg.tool_calls = vec![
                ToolCall {
                    id: "tc-1".to_string(),
                    name: "write_file".to_string(),
                    args: serde_json::json!({
                        "path": "safe\u{202E}exe.txt",
                        "content": "malicious content"
                    }),
                },
                ToolCall {
                    id: "tc-2".to_string(),
                    name: "browser_navigate".to_string(),
                    args: serde_json::json!({
                        "url": "pаypal.com" // Cyrillic 'а'
                    }),
                },
            ];
            msg
        }],
        todos: vec![],
        extensions: Default::default(),
    };

    let runtime = Runtime::new();
    let config = RunnableConfig::default();

    // Should detect issues in tool call arguments (logs warnings)
    let update = mw.abefore_agent(&state, &runtime, &config).await;
    // With sanitize_inputs = true by default, this should be None
    // because sanitize() is only applied to message content, not tool args
    assert!(update.is_none());
}

#[tokio::test]
async fn test_mixed_script_detection_in_identifiers() {
    // Test detection of mixed scripts in variable names or identifiers
    let checker = UnicodeSecurityChecker::strict();

    let mixed_code = "let userName = 'test'; let userNаme = 'fake';"; // Second has Cyrillic 'а'
    let issues = checker.check(mixed_code);

    let has_mixed = issues
        .iter()
        .any(|issue| matches!(issue, rvagent_middleware::UnicodeIssue::MixedScript { .. }));

    assert!(has_mixed);
}

#[tokio::test]
async fn test_safe_multilingual_content_unmodified() {
    // Test that legitimate multilingual content is not flagged
    // when using permissive config
    let mw = UnicodeSecurityMiddleware::new(UnicodeSecurityConfig::permissive())
        .with_output_sanitization(false);

    let state = AgentState {
        messages: vec![Message::tool(
            "Hello, 世界! Привет! مرحبا", // Multi-script greeting
            "tc-1",
            "translator",
        )],
        todos: vec![],
        extensions: Default::default(),
    };

    let runtime = Runtime::new();
    let config = RunnableConfig::default();

    let update = mw.abefore_agent(&state, &runtime, &config).await;
    // Permissive mode doesn't check mixed scripts or confusables
    assert!(update.is_none());
}

#[tokio::test]
async fn test_ascii_only_fast_path() {
    // Test that pure ASCII content takes the fast path
    let checker = UnicodeSecurityChecker::strict();

    let ascii_text = "Hello, world! This is pure ASCII with numbers 12345.";
    assert!(UnicodeSecurityChecker::is_ascii_safe(ascii_text));

    let issues = checker.check(ascii_text);
    assert!(issues.is_empty());

    let safe_result = checker.is_safe(ascii_text);
    assert!(safe_result);
}

#[tokio::test]
async fn test_comprehensive_attack_scenario() {
    // Combine multiple attack vectors in one message
    let mw = UnicodeSecurityMiddleware::strict()
        .with_output_sanitization(true)
        .with_user_input_check(true)
        .with_input_sanitization(true);

    let state = AgentState {
        messages: vec![
            Message::user("Visit pаypal.com\u{200B}now!"), // Homoglyph + zero-width
            Message::tool(
                "Downloaded: evil\u{202E}txt.exe", // BiDi override
                "tc-1",
                "filesystem",
            ),
        ],
        todos: vec![],
        extensions: Default::default(),
    };

    let runtime = Runtime::new();
    let config = RunnableConfig::default();

    let update = mw.abefore_agent(&state, &runtime, &config).await;
    assert!(update.is_some());

    let new_msgs = update.unwrap().messages.unwrap();
    // User message: zero-width stripped
    assert_eq!(new_msgs[0].content, "Visit pаypal.comnow!"); // Confusable remains
                                                             // Tool message: BiDi stripped
    assert_eq!(new_msgs[1].content, "Downloaded: eviltxt.exe");
}

#[test]
fn test_all_dangerous_bidi_controls() {
    let checker = UnicodeSecurityChecker::strict();

    // Test each BiDi control individually
    let attacks = vec![
        ("LRE", "\u{202A}"),
        ("RLE", "\u{202B}"),
        ("PDF", "\u{202C}"),
        ("LRO", "\u{202D}"),
        ("RLO", "\u{202E}"), // Most dangerous
        ("LRI", "\u{2066}"),
        ("RLI", "\u{2067}"),
        ("FSI", "\u{2068}"),
        ("PDI", "\u{2069}"),
    ];

    for (name, bidi) in attacks {
        let malicious = format!("safe{}file.txt", bidi);
        let issues = checker.check(&malicious);

        assert!(!issues.is_empty(), "{} control should be detected", name);
        assert!(
            !checker.is_safe(&malicious),
            "{} should fail safety check",
            name
        );

        let sanitized = checker.sanitize(&malicious);
        assert_eq!(sanitized, "safefile.txt", "{} should be stripped", name);
    }
}

#[test]
fn test_all_zero_width_characters() {
    let checker = UnicodeSecurityChecker::strict();

    let zero_widths = vec![
        ("ZWSP", "\u{200B}"),
        ("ZWNJ", "\u{200C}"),
        ("ZWJ", "\u{200D}"),
        ("LRM", "\u{200E}"),
        ("RLM", "\u{200F}"),
        ("WJ", "\u{2060}"),
        ("BOM", "\u{FEFF}"),
    ];

    for (name, zw) in zero_widths {
        let hidden = format!("secret{}data", zw);
        let issues = checker.check(&hidden);

        assert!(!issues.is_empty(), "{} should be detected", name);

        let sanitized = checker.sanitize(&hidden);
        assert_eq!(sanitized, "secretdata", "{} should be stripped", name);
    }
}

#[test]
fn test_cyrillic_latin_confusables() {
    let checker = UnicodeSecurityChecker::strict();

    // Common phishing targets
    let phishing_domains = vec![
        "pаypal.com",    // Cyrillic 'а'
        "googlе.com",    // Cyrillic 'е'
        "аpple.com",     // Cyrillic 'а'
        "micrоsoft.com", // Cyrillic 'о'
    ];

    for domain in phishing_domains {
        let issues = checker.check(domain);

        let has_confusable = issues
            .iter()
            .any(|issue| matches!(issue, rvagent_middleware::UnicodeIssue::Confusable { .. }));

        assert!(has_confusable, "Should detect confusable in '{}'", domain);
    }
}

#[test]
fn test_issue_display_formatting() {
    use rvagent_middleware::UnicodeIssue;

    // Test that issue display includes useful information
    let issue1 = UnicodeIssue::BidiControl {
        char: '\u{202E}',
        position: 10,
        unicode: "U+202E".to_string(),
    };
    let display1 = issue1.to_string();
    assert!(display1.contains("202E"));
    assert!(display1.contains("BiDi"));
    assert!(display1.contains("10"));

    let issue2 = UnicodeIssue::Confusable {
        char: 'а',
        looks_like: 'a',
        position: 5,
    };
    let display2 = issue2.to_string();
    assert!(display2.contains("looks like"));
    assert!(display2.contains("5"));

    let issue3 = UnicodeIssue::MixedScript {
        scripts: vec!["Latin".to_string(), "Cyrillic".to_string()],
    };
    let display3 = issue3.to_string();
    assert!(display3.contains("Latin"));
    assert!(display3.contains("Cyrillic"));
}

#[test]
fn test_performance_ascii_fast_path() {
    let checker = UnicodeSecurityChecker::strict();

    // Large ASCII text should be fast
    let large_ascii = "a".repeat(100_000);
    let start = std::time::Instant::now();
    let is_safe = UnicodeSecurityChecker::is_ascii_safe(&large_ascii);
    let elapsed = start.elapsed();

    assert!(is_safe);
    // ASCII check should be very fast (< 10ms for 100k chars)
    assert!(elapsed.as_millis() < 10);
}

#[test]
fn test_config_strict_vs_permissive() {
    let strict = UnicodeSecurityChecker::strict();
    let permissive = UnicodeSecurityChecker::new(UnicodeSecurityConfig::permissive());

    let text = "pаypal.com\u{200B}"; // Confusable + zero-width

    // Strict should detect both
    let strict_issues = strict.check(text);
    assert!(strict_issues.len() >= 2); // At least confusable and zero-width

    // Permissive should only detect zero-width (BiDi and zero-width always checked)
    let permissive_issues = permissive.check(text);
    let has_confusable = permissive_issues
        .iter()
        .any(|issue| matches!(issue, rvagent_middleware::UnicodeIssue::Confusable { .. }));
    assert!(!has_confusable); // Should not check confusables
}
