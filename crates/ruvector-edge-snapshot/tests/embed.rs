//! Embedding batcher limits and the port contract (mock, no network).

use ruvector_edge_snapshot::*;
use std::cell::RefCell;
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
    }
}

struct MockAi {
    calls: RefCell<Vec<usize>>,
    dim: usize,
}

impl EmbeddingPort for MockAi {
    fn embed(
        &self,
        model: &str,
        texts: &[&str],
        options: EmbedOptions,
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, EmbedError>> {
        assert_eq!(model, BGE_SMALL_MODEL);
        assert!(!options.truncate_inputs, "off unless the Worker opts in");
        self.calls.borrow_mut().push(texts.len());
        let out = texts
            .iter()
            .map(|t| vec![t.len() as f32; self.dim])
            .collect();
        async move { Ok(out) }
    }
}

#[test]
fn plans_respect_text_and_token_limits() {
    let b = EmbeddingBatcher::default();
    let texts: Vec<String> = (0..250)
        .map(|i| format!("document number {i} about vectors"))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let plan = b.plan(&refs).unwrap();
    assert_eq!(
        plan.iter().map(|p| p.range.len()).collect::<Vec<_>>(),
        [100, 100, 50]
    );
    assert_eq!(plan.last().unwrap().range.end, 250);

    let tight = EmbeddingBatcher {
        limits: EmbedLimits {
            max_tokens_per_batch: 40,
            ..EmbedLimits::default()
        },
        ..b.clone()
    };
    let plan = tight.plan(&refs[..20]).unwrap();
    for p in &plan {
        let sum: usize = refs[p.range.clone()]
            .iter()
            .map(|t| estimate_tokens(t))
            .sum();
        assert_eq!(sum, p.est_tokens);
        assert!(p.est_tokens <= 40);
    }
    let covered: usize = plan.iter().map(|p| p.range.len()).sum();
    assert_eq!(covered, 20);
    assert!(plan.windows(2).all(|w| w[0].range.end == w[1].range.start));
}

#[test]
fn over_limit_inputs_are_refused_with_their_index() {
    let b = EmbeddingBatcher::default();
    assert_eq!(
        b.plan(&["ok", "  "]).unwrap_err(),
        EmbedError::EmptyText { index: 1 }
    );
    let long = "a".repeat(9 * 1024);
    assert_eq!(
        b.plan(&["ok", &long]).unwrap_err(),
        EmbedError::TextTooLong { index: 1 }
    );
    let many_words = "word ".repeat(600);
    assert_eq!(
        b.plan(&[&many_words]).unwrap_err(),
        EmbedError::TooManyTokens { index: 0 }
    );
    let punct = "!".repeat(600);
    assert_eq!(
        b.plan(&[&punct]).unwrap_err(),
        EmbedError::TooManyTokens { index: 0 }
    );
    let texts = vec!["x"; 501];
    assert_eq!(b.plan(&texts).unwrap_err(), EmbedError::TooManyTexts(501));
    assert!(b.plan(&[]).unwrap().is_empty());
}

#[test]
fn token_estimate_is_conservative() {
    assert_eq!(estimate_tokens("hello"), 2 + 2);
    assert_eq!(estimate_tokens("a b c"), 2 + 3);
    assert_eq!(estimate_tokens("hi, there!"), 2 + (1 + 1) + (2 + 1));
    // "h" (no vowel: 1/char) + "é" (non-ASCII) + "llo".
    assert_eq!(estimate_tokens("héllo"), 2 + 1 + 1 + 1);
}

#[test]
fn token_estimate_counts_digit_and_vowelless_runs_per_character() {
    // WordPiece splits hex / base64 / UUIDs / digit runs into 1–2-char
    // pieces; the old ceil(len/4) estimate under-counted them 2–4×.
    let hex = "3f9a2b7c".repeat(8); // 64 chars
    assert_eq!(estimate_tokens(&hex), 2 + 64);
    let uuid = "123e4567-e89b-12d3-a456-426614174000";
    assert_eq!(estimate_tokens(uuid), 2 + 32 + 4);
    assert_eq!(estimate_tokens("xkcd tsk"), 2 + 4 + 3);
    // Long rare words: ceil(len/3) ≥ their WordPiece count (~8 here).
    assert_eq!(estimate_tokens("antidisestablishmentarianism"), 2 + 10);
    // ~2,000 hex chars no longer pass as ~500 tokens: refused up front
    // instead of failing a whole 100-text model call.
    let b = EmbeddingBatcher::default();
    let long_hex = "3f9a2b7c".repeat(250);
    assert_eq!(
        b.plan(&[long_hex.as_str()]).unwrap_err(),
        EmbedError::TooManyTokens { index: 0 }
    );
}

#[test]
fn embed_all_calls_port_per_batch_and_validates() {
    let b = EmbeddingBatcher::default();
    let texts: Vec<String> = (0..230).map(|i| format!("t{i}")).collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let ai = MockAi {
        calls: RefCell::new(vec![]),
        dim: BGE_SMALL_DIM,
    };
    let out = block_on(embed_all(&ai, &b, &refs)).unwrap();
    assert_eq!(out.len(), 230);
    assert!(out.iter().all(|v| v.len() == 384));
    assert_eq!(out[229][0], "t229".len() as f32);
    assert_eq!(*ai.calls.borrow(), [100, 100, 30]);

    let wrong = MockAi {
        calls: RefCell::new(vec![]),
        dim: 768,
    };
    assert_eq!(
        block_on(embed_all(&wrong, &b, &refs)).unwrap_err(),
        EmbedError::BadVector { index: 0 }
    );
}

#[test]
fn accept_checks_count_dim_and_finiteness() {
    let b = EmbeddingBatcher::default();
    let batch = EmbedBatch {
        range: 10..12,
        est_tokens: 4,
    };
    assert_eq!(
        b.accept(&batch, vec![vec![0.0; 384]]).unwrap_err(),
        EmbedError::ResponseCount
    );
    let mut v = vec![vec![0.0; 384], vec![0.0; 384]];
    v[1][5] = f32::NAN;
    assert_eq!(
        b.accept(&batch, v).unwrap_err(),
        EmbedError::BadVector { index: 11 }
    );
    assert!(b.accept(&batch, vec![vec![0.5; 384]; 2]).is_ok());
}
