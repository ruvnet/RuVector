//! Request / response shapes. Field names follow the Jev wire contract
//! (`state`, `questions`, `choice`, `probabilities`, `confidence`, `score`,
//! `legend`, `noul`); the extra fields (`abstain`, `calibrated`, `head`,
//! `model`, `temperature`) are additive (ADR-003 "what every answer carries").

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One option of a `choice` question: a bare description or the structured form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Criterion {
    Text(String),
    Structured {
        what: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        not_for: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        examples: Vec<String>,
    },
}

impl Criterion {
    pub fn what(&self) -> &str {
        match self {
            Criterion::Text(t) => t,
            Criterion::Structured { what, .. } => what,
        }
    }
    pub fn not_for(&self) -> Option<&str> {
        match self {
            Criterion::Text(_) => None,
            Criterion::Structured { not_for, .. } => not_for.as_deref(),
        }
    }
    pub fn examples(&self) -> &[String] {
        match self {
            Criterion::Text(_) => &[],
            Criterion::Structured { examples, .. } => examples,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Pick one of up to 255 options. `criteria` is ordered (BTreeMap) so
    /// probabilities are reproducible across runs and targets.
    Choice {
        #[serde(default)]
        instructions: String,
        criteria: BTreeMap<String, Criterion>,
    },
    /// Ordinal legend, e.g. ["Calm", "Irritated", "Angry"].
    Score {
        #[serde(default)]
        instructions: String,
        legend: Vec<String>,
    },
    /// A 0–1 predicate ("the sender needs a response soon").
    Noul { instructions: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub state: String,
    pub questions: BTreeMap<String, Question>,
}

/// Which head produced an answer (ADR-003). Recorded in every receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Head {
    NearestPrototype,
    LinearProbe,
    Logistic,
    /// Zero-label similarity fallback for `noul`: never reported as a probability.
    SimilarityUncalibrated,
}

/// Fields every answer carries in addition to the Jev-shaped payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnswerMeta {
    pub confidence: f32,
    pub abstain: f32,
    pub calibrated: bool,
    pub head: Head,
    /// Embedder id: model name @ manifest hash (ADR-002 §1).
    pub model: String,
    pub temperature: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Answer {
    Choice {
        choice: String,
        /// Sums to 1 over the options (Jev contract); abstain mass is in `meta`.
        probabilities: BTreeMap<String, f32>,
        #[serde(flatten)]
        meta: AnswerMeta,
    },
    Score {
        score: usize,
        legend: String,
        probabilities: Vec<f32>,
        #[serde(flatten)]
        meta: AnswerMeta,
    },
    Noul {
        noul: f32,
        #[serde(flatten)]
        meta: AnswerMeta,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub embed_calls: u32,
    pub texts_embedded: u32,
    pub state_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub answers: BTreeMap<String, Answer>,
    pub usage: Usage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jev_shaped_request_round_trips() {
        let json = r#"{"state":"my card was charged twice","questions":{
          "dept":{"type":"choice","instructions":"route","criteria":{
            "billing":"charges and refunds",
            "fraud":{"what":"unauthorised use","not_for":"duplicate charges","examples":["someone used my card"]}}},
          "mood":{"type":"score","legend":["Calm","Irritated","Angry"]},
          "urgent":{"type":"noul","instructions":"the sender needs a response soon"}}}"#;
        let req: DecisionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.questions.len(), 3);
        match &req.questions["dept"] {
            Question::Choice { criteria, .. } => {
                assert_eq!(criteria["billing"].what(), "charges and refunds");
                assert_eq!(criteria["fraud"].not_for(), Some("duplicate charges"));
                assert_eq!(criteria["fraud"].examples().len(), 1);
            }
            other => panic!("unexpected {other:?}"),
        }
        let back = serde_json::to_string(&req).unwrap();
        let again: DecisionRequest = serde_json::from_str(&back).unwrap();
        assert_eq!(req, again);
    }
}
