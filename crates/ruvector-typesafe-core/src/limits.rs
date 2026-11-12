//! Input limits (ADR-005): rejected with a typed error, never silently truncated.

use crate::{DecisionRequest, Question, Result, TypesafeError};

pub const MAX_STATE_BYTES: usize = 16 * 1024;
pub const MAX_OPTIONS: usize = 255;
pub const MAX_DESCRIPTION_BYTES: usize = 1024;
pub const MAX_EXAMPLES_PER_OPTION: usize = 10;
pub const MAX_EXAMPLE_BYTES: usize = 512;
pub const MAX_QUESTIONS: usize = 64;

pub fn validate(req: &DecisionRequest) -> Result<()> {
    if req.state.len() > MAX_STATE_BYTES {
        return Err(TypesafeError::Limit("state exceeds 16 KiB"));
    }
    if req.questions.is_empty() {
        return Err(TypesafeError::Invalid("no questions".into()));
    }
    if req.questions.len() > MAX_QUESTIONS {
        return Err(TypesafeError::Limit("more than 64 questions"));
    }
    for (id, q) in &req.questions {
        match q {
            Question::Choice { criteria, .. } => {
                if criteria.len() < 2 {
                    return Err(TypesafeError::Invalid(format!(
                        "question {id}: fewer than 2 options"
                    )));
                }
                if criteria.len() > MAX_OPTIONS {
                    return Err(TypesafeError::Limit("more than 255 options"));
                }
                for c in criteria.values() {
                    if c.what().len() > MAX_DESCRIPTION_BYTES
                        || c.not_for().is_some_and(|n| n.len() > MAX_DESCRIPTION_BYTES)
                    {
                        return Err(TypesafeError::Limit("option description exceeds 1 KiB"));
                    }
                    if c.examples().len() > MAX_EXAMPLES_PER_OPTION {
                        return Err(TypesafeError::Limit("more than 10 examples for one option"));
                    }
                    if c.examples().iter().any(|e| e.len() > MAX_EXAMPLE_BYTES) {
                        return Err(TypesafeError::Limit("example exceeds 512 bytes"));
                    }
                }
            }
            Question::Score { legend, .. } => {
                if legend.len() < 2 || legend.len() > MAX_OPTIONS {
                    return Err(TypesafeError::Invalid(format!(
                        "question {id}: legend needs 2..=255 buckets"
                    )));
                }
            }
            Question::Noul { instructions } => {
                if instructions.trim().is_empty() {
                    return Err(TypesafeError::Invalid(format!(
                        "question {id}: noul needs instructions"
                    )));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Criterion;
    use std::collections::BTreeMap;

    fn req(state: &str) -> DecisionRequest {
        let mut criteria = BTreeMap::new();
        criteria.insert("a".into(), Criterion::Text("alpha".into()));
        criteria.insert("b".into(), Criterion::Text("beta".into()));
        let mut questions = BTreeMap::new();
        questions.insert(
            "q".into(),
            Question::Choice {
                instructions: String::new(),
                criteria,
            },
        );
        DecisionRequest {
            state: state.into(),
            questions,
        }
    }

    #[test]
    fn accepts_a_normal_request() {
        assert!(validate(&req("hello")).is_ok());
    }

    #[test]
    fn rejects_oversized_state_instead_of_truncating() {
        let big = "x".repeat(MAX_STATE_BYTES + 1);
        assert!(matches!(validate(&req(&big)), Err(TypesafeError::Limit(_))));
    }
}
