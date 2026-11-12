//! Tests for the serve command: the real router and middleware, driven by a
//! scripted backend whose token counts and finish reasons are known.

use super::*;
use ruvllm::error::Result as RuvResult;
use ruvllm::{
    GeneratedToken, GenerationOutput, ModelConfig, ModelInfo, RuvLLMError, SpecialTokens,
    TokenStream, Tokenizer,
};
use std::sync::Mutex;
use tower::ServiceExt;

/// One token per whitespace-separated word, so counts are predictable.
struct WordTokenizer;

impl Tokenizer for WordTokenizer {
    fn encode(&self, text: &str) -> RuvResult<Vec<u32>> {
        Ok(text.split_whitespace().map(|_| 7).collect())
    }

    fn decode(&self, _tokens: &[u32]) -> RuvResult<String> {
        Ok(String::new())
    }

    fn vocab_size(&self) -> usize {
        16
    }

    fn special_tokens(&self) -> SpecialTokens {
        SpecialTokens::default()
    }
}

/// Replays scripted deltas, then ends with `outcome` (an `Err` is a
/// generation failure). Records the prompt and params it was called with.
struct ScriptedBackend {
    deltas: Vec<&'static str>,
    completion_tokens: usize,
    outcome: Result<FinishReason, &'static str>,
    context: usize,
    tokenizer: WordTokenizer,
    calls: Mutex<Vec<(String, GenerateParams)>>,
    /// Replay the deltas until `on_token` returns false, then set `cancelled`.
    endless: bool,
    cancelled: AtomicBool,
}

impl ScriptedBackend {
    fn new(deltas: Vec<&'static str>, outcome: Result<FinishReason, &'static str>) -> Self {
        Self {
            completion_tokens: deltas.len() + 1,
            deltas,
            outcome,
            context: 4096,
            tokenizer: WordTokenizer,
            calls: Mutex::new(Vec::new()),
            endless: false,
            cancelled: AtomicBool::new(false),
        }
    }

    fn last_call(&self) -> (String, GenerateParams) {
        self.calls.lock().unwrap().last().cloned().expect("no call")
    }
}

impl LlmBackend for ScriptedBackend {
    fn load_model(&mut self, _model_id: &str, _config: ModelConfig) -> RuvResult<()> {
        Ok(())
    }

    fn generate(&self, _prompt: &str, _params: GenerateParams) -> RuvResult<String> {
        Err(RuvLLMError::InvalidOperation(
            "serve must use generate_detailed".into(),
        ))
    }

    fn generate_stream(
        &self,
        _prompt: &str,
        _params: GenerateParams,
    ) -> RuvResult<Box<dyn Iterator<Item = RuvResult<GeneratedToken>> + Send + '_>> {
        Err(RuvLLMError::InvalidOperation(
            "serve must use generate_detailed".into(),
        ))
    }

    fn generate_stream_v2(&self, _prompt: &str, _params: GenerateParams) -> RuvResult<TokenStream> {
        Err(RuvLLMError::InvalidOperation(
            "serve must use generate_detailed".into(),
        ))
    }

    fn generate_detailed(
        &self,
        prompt: &str,
        params: GenerateParams,
        on_token: &mut dyn FnMut(&GeneratedToken) -> bool,
    ) -> RuvResult<GenerationOutput> {
        self.calls
            .lock()
            .unwrap()
            .push((prompt.to_string(), params.clone()));
        'replay: loop {
            for delta in &self.deltas {
                let token = GeneratedToken {
                    id: 7,
                    text: delta.to_string(),
                    logprob: None,
                    is_special: false,
                };
                if !on_token(&token) {
                    self.cancelled.store(true, Ordering::Relaxed);
                    break 'replay;
                }
            }
            if !self.endless {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let finish_reason = self
            .outcome
            .map_err(|msg| RuvLLMError::Generation(msg.into()))?;
        Ok(GenerationOutput {
            text: self.deltas.concat(),
            prompt_tokens: self.tokenizer.encode(prompt)?.len(),
            completion_tokens: self.completion_tokens,
            finish_reason,
        })
    }

    fn get_embeddings(&self, _text: &str) -> RuvResult<Vec<f32>> {
        Ok(Vec::new())
    }

    fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        Some(&self.tokenizer)
    }

    fn is_model_loaded(&self) -> bool {
        true
    }

    fn model_info(&self) -> Option<ModelInfo> {
        Some(ModelInfo {
            name: "scripted".into(),
            architecture: ModelArchitecture::Llama,
            num_parameters: 0,
            vocab_size: 16,
            hidden_size: 8,
            num_layers: 1,
            max_context_length: self.context,
            quantization: None,
            memory_usage: 0,
        })
    }

    fn unload_model(&mut self) {}
}

fn state_with(backend: Option<Arc<dyn LlmBackend>>, allow_mock: bool) -> SharedState {
    Arc::new(RwLock::new(ServerState {
        model_id: "served/model".into(),
        backend,
        allow_mock,
        chat_template: ChatTemplate::Qwen,
        request_count: 0,
        total_tokens: 0,
        start_time: Instant::now(),
    }))
}

fn loaded(backend: &Arc<ScriptedBackend>) -> SharedState {
    let backend: Arc<dyn LlmBackend> = backend.clone();
    state_with(Some(backend), false)
}

struct Reply {
    status: StatusCode,
    mode: String,
    body: Vec<u8>,
}

impl Reply {
    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).expect("JSON body")
    }

    /// The payload of every SSE `data:` line, in order.
    fn sse_data(&self) -> Vec<String> {
        String::from_utf8_lossy(&self.body)
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(str::to_string)
            .collect()
    }

    /// SSE payloads other than `[DONE]`, parsed.
    fn sse_json(&self) -> Vec<serde_json::Value> {
        self.sse_data()
            .iter()
            .filter(|d| d.as_str() != "[DONE]")
            .map(|d| serde_json::from_str(d).expect("SSE JSON"))
            .collect()
    }
}

async fn request(state: SharedState, method: &str, uri: &str, body: serde_json::Value) -> Reply {
    let request = axum::http::Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    let request = if method == "GET" {
        request.body(axum::body::Body::empty())
    } else {
        request.body(axum::body::Body::from(body.to_string()))
    }
    .unwrap();
    let response = build_router(state).oneshot(request).await.unwrap();
    let status = response.status();
    let mode = response.headers()["x-ruvllm-mode"]
        .to_str()
        .unwrap()
        .to_string();
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap()
        .to_vec();
    Reply { status, mode, body }
}

async fn chat(state: SharedState, body: serde_json::Value) -> Reply {
    request(state, "POST", "/v1/chat/completions", body).await
}

fn user(content: &str) -> serde_json::Value {
    serde_json::json!([{ "role": "user", "content": content }])
}

fn qwen_prompt(content: &str) -> String {
    ChatTemplate::Qwen.format(&[ruvllm::ChatMessage::user(content)])
}

#[tokio::test]
async fn non_stream_reports_backend_usage_and_length() {
    let backend = Arc::new(ScriptedBackend::new(
        vec!["The", " capital", " is"],
        Ok(FinishReason::Length),
    ));
    let state = loaded(&backend);
    let reply = chat(
        state.clone(),
        serde_json::json!({ "model": "anything", "messages": user("Capital of France?"), "max_tokens": 4 }),
    )
    .await;

    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.mode, "model");
    let json = reply.json();
    let prompt_tokens = qwen_prompt("Capital of France?").split_whitespace().count();
    assert_eq!(json["choices"][0]["finish_reason"], "length");
    assert_eq!(json["choices"][0]["message"]["content"], "The capital is");
    assert_eq!(json["usage"]["prompt_tokens"], prompt_tokens);
    assert_eq!(json["usage"]["completion_tokens"], 4);
    assert_eq!(json["usage"]["total_tokens"], prompt_tokens + 4);
    assert_eq!(json["model"], "served/model");
    assert!(json.get("system_fingerprint").is_none());

    // The backend got the ChatML prompt and the request's max_tokens.
    let (prompt, params) = backend.last_call();
    assert_eq!(prompt, qwen_prompt("Capital of France?"));
    assert_eq!(params.max_tokens, 4);

    let metrics = request(state, "GET", "/metrics", serde_json::Value::Null).await;
    assert_eq!(metrics.json()["tokens_total"], prompt_tokens + 4);
    assert_eq!(metrics.json()["requests_total"], 1);
}

#[tokio::test]
async fn non_stream_end_of_sequence_and_stop_sequence_are_stop() {
    for reason in [FinishReason::EndOfSequence, FinishReason::Stop] {
        let backend = Arc::new(ScriptedBackend::new(vec!["Paris"], Ok(reason)));
        let reply = chat(
            loaded(&backend),
            serde_json::json!({ "model": "m", "messages": user("hi") }),
        )
        .await;
        assert_eq!(reply.status, StatusCode::OK);
        assert_eq!(
            reply.json()["choices"][0]["finish_reason"],
            "stop",
            "{:?}",
            reason
        );
    }
}

#[tokio::test]
async fn generation_error_is_a_500_not_assistant_content() {
    let backend = Arc::new(ScriptedBackend::new(vec![], Err("kv cache exploded")));
    let reply = chat(
        loaded(&backend),
        serde_json::json!({ "model": "m", "messages": user("hi") }),
    )
    .await;

    assert_eq!(reply.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(reply.mode, "model");
    let json = reply.json();
    assert_eq!(json["error"]["type"], "server_error");
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("kv cache exploded"));
    assert!(json.get("choices").is_none());
}

#[tokio::test]
async fn stream_sends_deltas_then_the_backend_finish_reason_and_usage() {
    let backend = Arc::new(ScriptedBackend::new(
        vec!["早上", "好", "。"],
        Ok(FinishReason::Length),
    ));
    let reply = chat(
        loaded(&backend),
        serde_json::json!({
            "model": "m",
            "messages": user("Translate"),
            "max_tokens": 4,
            "stream": true,
            "stream_options": { "include_usage": true }
        }),
    )
    .await;

    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.sse_data().last().map(String::as_str), Some("[DONE]"));
    let events = reply.sse_json();
    assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
    let text: String = events
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "早上好。");

    let finishes: Vec<_> = events
        .iter()
        .filter(|e| !e["choices"][0]["finish_reason"].is_null())
        .collect();
    assert_eq!(finishes.len(), 1);
    assert_eq!(finishes[0]["choices"][0]["finish_reason"], "length");

    let usage = events.last().unwrap();
    assert_eq!(usage["choices"], serde_json::json!([]));
    let prompt_tokens = qwen_prompt("Translate").split_whitespace().count();
    assert_eq!(usage["usage"]["prompt_tokens"], prompt_tokens);
    assert_eq!(usage["usage"]["completion_tokens"], 4);
}

#[tokio::test]
async fn stream_end_of_sequence_is_stop_without_usage_by_default() {
    let backend = Arc::new(ScriptedBackend::new(
        vec!["Paris"],
        Ok(FinishReason::EndOfSequence),
    ));
    let reply = chat(
        loaded(&backend),
        serde_json::json!({ "model": "m", "messages": user("hi"), "stream": true }),
    )
    .await;
    let events = reply.sse_json();
    let last = events.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert!(events.iter().all(|e| e.get("usage").is_none()));
}

#[tokio::test]
async fn stream_error_sends_an_error_event_and_no_finish_chunk() {
    let backend = Arc::new(ScriptedBackend::new(vec!["partial"], Err("decode failed")));
    let reply = chat(
        loaded(&backend),
        serde_json::json!({ "model": "m", "messages": user("hi"), "stream": true }),
    )
    .await;

    assert_eq!(reply.mode, "model");
    assert_eq!(reply.sse_data().last().map(String::as_str), Some("[DONE]"));
    let events = reply.sse_json();
    let error = events.iter().find(|e| e.get("error").is_some()).unwrap();
    assert!(error["error"]["message"]
        .as_str()
        .unwrap()
        .contains("decode failed"));
    assert!(events
        .iter()
        .all(|e| e["choices"][0]["finish_reason"].is_null()));
    assert!(!String::from_utf8_lossy(&reply.body).contains("mock"));
}

#[tokio::test]
async fn stop_accepts_a_string_or_a_list() {
    for (stop, expected) in [
        (serde_json::json!("\n"), vec!["\n"]),
        (serde_json::json!(["END", "STOP"]), vec!["END", "STOP"]),
    ] {
        let backend = Arc::new(ScriptedBackend::new(vec!["x"], Ok(FinishReason::Stop)));
        let reply = chat(
            loaded(&backend),
            serde_json::json!({ "model": "m", "messages": user("hi"), "stop": stop }),
        )
        .await;
        assert_eq!(reply.status, StatusCode::OK);
        assert_eq!(backend.last_call().1.stop_sequences, expected);
    }
}

#[tokio::test]
async fn invalid_requests_are_400_before_generation() {
    let backend = Arc::new(ScriptedBackend::new(vec!["x"], Ok(FinishReason::Stop)));
    let bad = [
        serde_json::json!({ "model": "m", "messages": [{ "role": "tool", "content": "x" }] }),
        serde_json::json!({ "model": "m", "messages": [] }),
        serde_json::json!({ "model": "m", "messages": user("hi"), "max_tokens": 0 }),
    ];
    for body in bad {
        let reply = chat(loaded(&backend), body.clone()).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST, "{}", body);
        assert_eq!(reply.json()["error"]["type"], "invalid_request_error");
    }
    assert!(backend.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn prompt_that_fills_the_context_is_400_even_when_streaming() {
    let mut backend = ScriptedBackend::new(vec!["x"], Ok(FinishReason::Stop));
    backend.context = 4;
    let backend = Arc::new(backend);
    for stream in [false, true] {
        let reply = chat(
            loaded(&backend),
            serde_json::json!({ "model": "m", "messages": user("a b c d e"), "stream": stream }),
        )
        .await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
        assert_eq!(reply.json()["error"]["code"], "context_length_exceeded");
    }
    assert!(backend.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn no_model_without_allow_mock_is_503() {
    let reply = chat(
        state_with(None, false),
        serde_json::json!({ "model": "m", "messages": user("hi") }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(reply.mode, "mock");
    assert_eq!(reply.json()["error"]["code"], "model_not_loaded");
}

#[tokio::test]
async fn mock_mode_is_labeled_placeholder_text() {
    let state = state_with(None, true);
    let reply = chat(
        state.clone(),
        serde_json::json!({ "model": "m", "messages": user("Hello, write some code") }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
    assert_eq!(reply.mode, "mock");
    let json = reply.json();
    assert_eq!(json["system_fingerprint"], MOCK_FINGERPRINT);
    let content = json["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.starts_with("[ruvllm mock mode]"), "{}", content);
    assert!(content.contains("not model output"));
    assert_eq!(json["usage"]["total_tokens"], 0);

    let reply = chat(
        state,
        serde_json::json!({ "model": "m", "messages": user("hi"), "stream": true }),
    )
    .await;
    assert_eq!(reply.mode, "mock");
    let events = reply.sse_json();
    assert!(events
        .iter()
        .all(|e| e["system_fingerprint"] == MOCK_FINGERPRINT));
    let text: String = events
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert!(text.starts_with("[ruvllm mock mode]"));
}

#[tokio::test]
async fn responses_and_health_are_labelled_mock_without_a_model() {
    let reply = request(
        state_with(None, true),
        "GET",
        "/health",
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(reply.mode, "mock");
    let json = reply.json();
    assert_eq!(json["mode"], "mock");
    assert_eq!(json["status"], "degraded");
}

#[tokio::test]
async fn health_is_model_mode_with_a_backend() {
    let backend = Arc::new(ScriptedBackend::new(vec![], Ok(FinishReason::Stop)));
    let reply = request(loaded(&backend), "GET", "/health", serde_json::Value::Null).await;
    assert_eq!(reply.mode, "model");
    assert_eq!(reply.json()["status"], "healthy");
}

#[test]
fn finish_reasons_map_to_openai_strings() {
    assert_eq!(openai_finish_reason(FinishReason::Length), Some("length"));
    assert_eq!(openai_finish_reason(FinishReason::Stop), Some("stop"));
    assert_eq!(
        openai_finish_reason(FinishReason::EndOfSequence),
        Some("stop")
    );
    assert_eq!(openai_finish_reason(FinishReason::Error), None);
}

fn messages(json: serde_json::Value) -> Vec<RequestMessage> {
    serde_json::from_value(json).unwrap()
}

#[test]
fn build_prompt_uses_the_model_template() {
    let msgs = messages(serde_json::json!([
        { "role": "developer", "content": "You are helpful." },
        { "role": "user", "content": [
            { "type": "text", "text": "Hello!" }, { "type": "text", "text": "Hi." }
        ] },
        { "role": "assistant", "content": null },
    ]));
    let prompt = build_prompt(&ChatTemplate::Qwen, &msgs).unwrap();
    assert_eq!(
        prompt,
        "<|im_start|>system\nYou are helpful.<|im_end|>\n\
         <|im_start|>user\nHello!\nHi.<|im_end|>\n\
         <|im_start|>assistant\n<|im_end|>\n<|im_start|>assistant\n"
    );

    for (bad, says) in [
        (
            serde_json::json!([{ "role": "tool", "content": "{}" }]),
            "'tool'",
        ),
        (serde_json::json!([{ "role": "user" }]), "needs content"),
        (
            serde_json::json!([{ "role": "user", "content": [
                { "type": "image_url", "image_url": { "url": "x" } }
            ] }]),
            "'image_url'",
        ),
    ] {
        let err = build_prompt(&ChatTemplate::Qwen, &messages(bad)).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains(says), "{}", err.message);
    }
}

#[tokio::test]
async fn malformed_bodies_get_an_openai_error_not_axum_text() {
    let backend = Arc::new(ScriptedBackend::new(vec!["x"], Ok(FinishReason::Stop)));
    for body in [
        "{\"messages\": [",
        "{\"model\": \"m\"}",
        "{\"messages\": [{\"role\": \"user\", \"content\": 5}]}",
    ] {
        let request = axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body))
            .unwrap();
        let response = build_router(loaded(&backend))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["type"], "invalid_request_error", "{body}");
    }
    assert!(backend.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn dropping_a_non_streaming_request_cancels_its_generation() {
    let mut backend = ScriptedBackend::new(vec!["x"], Ok(FinishReason::Stop));
    backend.endless = true;
    let backend = Arc::new(backend);
    let job = prepare_job(
        &ChatTemplate::Qwen,
        serde_json::from_value(serde_json::json!({ "messages": user("hi") })).unwrap(),
    )
    .unwrap();
    let dyn_backend: Arc<dyn LlmBackend> = backend.clone();
    let request = complete(
        loaded(&backend),
        dyn_backend,
        ResponseMeta::new("m".into()),
        job,
    );
    // The client gives up: the handler future is dropped mid-generation.
    let timed_out = tokio::time::timeout(std::time::Duration::from_millis(50), request).await;
    assert!(timed_out.is_err());
    for _ in 0..200 {
        if backend.cancelled.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("generation kept running after the request was dropped");
}

#[test]
fn chat_template_prefers_the_model_file_then_architecture_then_id() {
    struct Templated(Option<ChatTemplate>, ModelArchitecture);
    impl LlmBackend for Templated {
        fn load_model(&mut self, _: &str, _: ModelConfig) -> RuvResult<()> {
            Ok(())
        }
        fn generate(&self, _: &str, _: GenerateParams) -> RuvResult<String> {
            Ok(String::new())
        }
        fn generate_stream(
            &self,
            _: &str,
            _: GenerateParams,
        ) -> RuvResult<Box<dyn Iterator<Item = RuvResult<GeneratedToken>> + Send + '_>> {
            Ok(Box::new(std::iter::empty()))
        }
        fn generate_stream_v2(&self, _: &str, _: GenerateParams) -> RuvResult<TokenStream> {
            Err(RuvLLMError::InvalidOperation("unused".into()))
        }
        fn chat_template(&self) -> Option<ChatTemplate> {
            self.0.clone()
        }
        fn get_embeddings(&self, _: &str) -> RuvResult<Vec<f32>> {
            Ok(Vec::new())
        }
        fn tokenizer(&self) -> Option<&dyn Tokenizer> {
            None
        }
        fn is_model_loaded(&self) -> bool {
            true
        }
        fn model_info(&self) -> Option<ModelInfo> {
            Some(ModelInfo {
                name: "t".into(),
                architecture: self.1,
                num_parameters: 0,
                vocab_size: 0,
                hidden_size: 0,
                num_layers: 0,
                max_context_length: 0,
                quantization: None,
                memory_usage: 0,
            })
        }
        fn unload_model(&mut self) {}
    }

    let from_file = Templated(Some(ChatTemplate::Llama3), ModelArchitecture::Qwen);
    assert_eq!(
        select_chat_template(Some(&from_file), "x/qwen"),
        ChatTemplate::Llama3
    );
    // A local path with no "qwen" in it still gets ChatML for a Qwen model.
    let qwen = Templated(None, ModelArchitecture::Qwen);
    assert_eq!(
        select_chat_template(Some(&qwen), "/models/model.gguf"),
        ChatTemplate::Qwen
    );
    assert_eq!(
        select_chat_template(None, "mistralai/Mistral-7B"),
        ChatTemplate::Mistral
    );
}

#[test]
fn mock_text_says_it_is_not_model_output() {
    let text = mock_text("org/model");
    assert!(text.starts_with("[ruvllm mock mode]"));
    assert!(text.contains("org/model"));
    assert!(text.contains("not model output"));
}

#[test]
fn test_detect_architecture() {
    assert_eq!(
        detect_architecture("mistralai/Mistral-7B"),
        ruvllm::ModelArchitecture::Mistral
    );
    assert_eq!(
        detect_architecture("Qwen/Qwen2.5-14B"),
        ruvllm::ModelArchitecture::Qwen
    );
}
