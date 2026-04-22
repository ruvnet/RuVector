// Quick integration test - run with: ANTHROPIC_API_KEY=sk-... cargo test -p rvagent-backends --test live_anthropic_test
use rvagent_backends::AnthropicClient;
use rvagent_core::messages::Message;
use rvagent_core::models::{resolve_model, ChatModel};

#[tokio::test]
async fn test_live_anthropic_call() {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("Skipping live test: ANTHROPIC_API_KEY not set");
        return;
    }

    let config = resolve_model("anthropic:claude-sonnet-4-20250514");
    let client = AnthropicClient::new(config).expect("failed to create client");

    let messages = vec![Message::human("What is 2+2? Reply with just the number.")];

    let response = client.complete(&messages).await.expect("API call failed");
    let content = response.content();
    println!("Response: {}", content);
    assert!(
        content.contains("4"),
        "Expected '4' in response, got: {}",
        content
    );
}
