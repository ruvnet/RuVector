//! Integration tests for the decision engine (ADR-003), over the deterministic
//! `HashEmbedder`; one test renames it to exercise the calibration path (gated
//! off for `@test-double` ids).

use super::*;
use crate::embedder::Embedder;
use crate::hash_embedder::HashEmbedder;
use crate::{Answer, Criterion, DecisionRequest, Head, Question, TypesafeError};
use std::collections::BTreeMap;

const DIMS: usize = 256;

fn structured(what: &str, examples: &[&str], not_for: Option<&str>) -> Criterion {
    Criterion::Structured {
        what: what.into(),
        not_for: not_for.map(str::to_string),
        examples: examples.iter().map(|s| s.to_string()).collect(),
    }
}

fn choice_request(state: &str, criteria: Vec<(&str, Criterion)>) -> DecisionRequest {
    let mut map = BTreeMap::new();
    for (k, c) in criteria {
        map.insert(k.to_string(), c);
    }
    let mut questions = BTreeMap::new();
    questions.insert(
        "q".to_string(),
        Question::Choice {
            instructions: String::new(),
            criteria: map,
        },
    );
    DecisionRequest {
        state: state.into(),
        questions,
    }
}

fn as_choice(a: &Answer) -> (&str, &BTreeMap<String, f32>, &crate::AnswerMeta) {
    match a {
        Answer::Choice {
            choice,
            probabilities,
            meta,
        } => (choice, probabilities, meta),
        other => panic!("expected choice, got {other:?}"),
    }
}

fn topic_request(state: &str) -> DecisionRequest {
    choice_request(
        state,
        vec![
            (
                "weather",
                structured(
                    "weather forecast",
                    &["sunny rain clouds storm", "temperature tomorrow humid"],
                    None,
                ),
            ),
            (
                "sports",
                structured(
                    "sports match scores",
                    &["football goal player match", "tennis racket serve court"],
                    None,
                ),
            ),
            (
                "cooking",
                structured(
                    "cooking recipe kitchen",
                    &["bake oven flour dough", "simmer sauce garlic onion"],
                    None,
                ),
            ),
        ],
    )
}

fn training_examples() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            "weather",
            &[
                "rain clouds storm today",
                "sunny warm forecast tomorrow",
                "humid temperature rising",
                "snow cold winter freeze",
                "wind gust breeze strong",
                "fog mist morning damp",
                "thunder lightning downpour",
                "hail sleet icy roads",
                "heatwave scorching dry summer",
                "drizzle overcast grey sky",
            ],
        ),
        (
            "sports",
            &[
                "football goal striker match",
                "tennis serve racket court",
                "basketball dunk hoop score",
                "cricket bat wicket over",
                "hockey puck rink slapshot",
                "golf putt fairway birdie",
                "rugby scrum tackle tryline",
                "baseball pitch home run",
                "boxing punch jab knockout",
                "swimming lap freestyle relay",
            ],
        ),
    ]
}

fn build_probe_training() -> Vec<LabeledExample> {
    let mut ex = Vec::new();
    for (label, samples) in training_examples() {
        for s in samples {
            ex.push(LabeledExample {
                text: (*s).into(),
                label: label.into(),
            });
        }
    }
    ex
}

fn two_topic_request(state: &str) -> DecisionRequest {
    choice_request(
        state,
        vec![
            (
                "weather",
                structured("weather forecast", &["rain clouds sunny storm"], None),
            ),
            (
                "sports",
                structured("sports match", &["football tennis goal court"], None),
            ),
        ],
    )
}

struct ProdLike {
    inner: HashEmbedder,
    id: String,
}

impl ProdLike {
    fn new(dims: usize) -> Self {
        Self {
            inner: HashEmbedder::new(dims),
            id: format!("hash-bow-{dims}@prod"),
        }
    }
}

impl Embedder for ProdLike {
    fn embed(&self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
        self.inner.embed(texts)
    }
    fn dims(&self) -> usize {
        self.inner.dims()
    }
    fn id(&self) -> &str {
        &self.id
    }
}

fn many_examples() -> Vec<LabeledExample> {
    // 120 examples so the every-5th calibration slice clears the 20-example
    // floor. Texts must be UNIQUE: the bank is append-only and dedupes by
    // content, so a repeated `(question,label,text)` never adds a second row.
    // The trailing `number {i}` keeps every example distinct.
    let weather = ["rain", "storm", "sunny", "cloud", "snow", "wind"];
    let sports = ["goal", "match", "court", "score", "tackle", "serve"];
    let mut ex = Vec::new();
    for i in 0..60 {
        let w = weather[i % weather.len()];
        let s = sports[i % sports.len()];
        ex.push(LabeledExample {
            text: format!("{w} weather forecast today number {i}"),
            label: "weather".into(),
        });
        ex.push(LabeledExample {
            text: format!("{s} sports match play number {i}"),
            label: "sports".into(),
        });
    }
    ex
}

mod decide;
mod train;
