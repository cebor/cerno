//! The one path all three primitives take.
//!
//! Noul, Choice and Score differ only in what the options are and how the resulting
//! distribution is named. Everything between — labelling, prompting, the single host call,
//! folding token variants, substituting the floor, calibrating, normalising — is shared.

use crate::{labels, math, prompt};
use cerno_host::{FirstTokenDistribution, FirstTokenRequest, HostError, ModelHost};
use cerno_types::{
    Answer, Calibration, ErrorCode, LevelProbability, MAX_LEVELS, MAX_OPTIONS, MIN_LEVELS,
    OptionProbability, Question, QuestionKind, SystemOneRequest,
};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("{message}")]
    Invalid {
        code: ErrorCode,
        message: String,
        question_id: Option<String>,
    },

    /// The model answered, but with nothing resembling one of the labels we offered.
    ///
    /// Distinct from a host failure: the runtime worked, the model simply did not follow the
    /// instruction. Almost always this means the model is too small or too chatty for the job,
    /// which is what the benchmark in `cerno-bench` exists to catch before deployment.
    #[error(
        "model answered question {question_id:?} with none of the labels {expected:?}; \
         it offered {observed:?} instead"
    )]
    NoLabelMatched {
        question_id: String,
        expected: Vec<String>,
        observed: Vec<String>,
    },

    #[error(transparent)]
    Host(#[from] HostError),
}

impl EngineError {
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Invalid { code, .. } => *code,
            Self::NoLabelMatched { .. } => ErrorCode::NoLabelMatched,
            Self::Host(HostError::Timeout(_)) => ErrorCode::HostTimeout,
            Self::Host(_) => ErrorCode::HostUnavailable,
        }
    }

    pub fn question_id(&self) -> Option<&str> {
        match self {
            Self::Invalid { question_id, .. } => question_id.as_deref(),
            Self::NoLabelMatched { question_id, .. } => Some(question_id),
            Self::Host(_) => None,
        }
    }

    fn invalid(code: ErrorCode, question_id: Option<&str>, message: impl Into<String>) -> Self {
        Self::Invalid {
            code,
            message: message.into(),
            question_id: question_id.map(str::to_string),
        }
    }
}

/// What a question looks like once the primitive has been stripped away.
struct Ballot<'a> {
    question: Option<&'a str>,
    /// The option texts shown to the model, in request order.
    options: Vec<String>,
}

/// One question's resolved distribution over its labels.
struct Tally {
    /// Probability per option, in request order.
    probabilities: Vec<f64>,
    /// The logprob actually fed into the softmax, per label. A floor substitution appears here
    /// like any other value; `truncated` is what says one happened.
    logprobs: BTreeMap<String, f64>,
    confidence: f64,
    truncated: bool,
    input_tokens: u32,
}

pub struct Engine {
    host: Arc<dyn ModelHost>,
    keep_alive: Option<String>,
}

impl Engine {
    pub fn new(host: Arc<dyn ModelHost>, keep_alive: Option<String>) -> Self {
        Self { host, keep_alive }
    }

    /// The largest option count this engine can answer, given what its host reports.
    pub fn max_options(&self) -> usize {
        MAX_OPTIONS.min(self.host.capabilities().max_top_logprobs)
    }

    pub fn host_name(&self) -> &str {
        self.host.name()
    }

    /// Reject a request that cannot be answered, before any model is loaded.
    pub fn validate(&self, request: &SystemOneRequest) -> Result<(), EngineError> {
        if request.state.trim().is_empty() {
            return Err(EngineError::invalid(
                ErrorCode::EmptyState,
                None,
                "state must not be empty",
            ));
        }

        if request.questions.is_empty() {
            return Err(EngineError::invalid(
                ErrorCode::NoQuestions,
                None,
                "at least one question is required",
            ));
        }

        if let Some(calibration) = request.calibration {
            let t = calibration.temperature;
            if !t.is_finite() || t <= 0.0 {
                return Err(EngineError::invalid(
                    ErrorCode::InvalidCalibration,
                    None,
                    format!("calibration temperature must be finite and above zero, got {t}"),
                ));
            }
        }

        let mut seen = BTreeMap::new();
        for question in &request.questions {
            if seen.insert(question.id.as_str(), ()).is_some() {
                return Err(EngineError::invalid(
                    ErrorCode::DuplicateQuestionId,
                    Some(&question.id),
                    format!("question id {:?} appears more than once", question.id),
                ));
            }
            self.validate_question(question)?;
        }

        Ok(())
    }

    fn validate_question(&self, question: &Question) -> Result<(), EngineError> {
        let id = Some(question.id.as_str());
        let max = self.max_options();

        match &question.kind {
            QuestionKind::Noul(text) => {
                if text.trim().is_empty() {
                    return Err(EngineError::invalid(
                        ErrorCode::EmptyQuestion,
                        id,
                        "a noul question must not be empty",
                    ));
                }
            }
            QuestionKind::Choice(spec) => {
                if spec.options.len() < 2 {
                    return Err(EngineError::invalid(
                        ErrorCode::TooFewOptions,
                        id,
                        format!(
                            "a choice needs at least 2 options, got {}",
                            spec.options.len()
                        ),
                    ));
                }
                if spec.options.len() > max {
                    return Err(EngineError::invalid(
                        ErrorCode::TooManyOptions,
                        id,
                        format!(
                            "a choice may have at most {max} options, got {}; the host reports \
                             only {max} ranked tokens, so further options cannot be observed. \
                             Split them across a first question that picks a group and a second \
                             that picks within it.",
                            spec.options.len()
                        ),
                    ));
                }
                if spec.options.iter().any(|o| o.trim().is_empty()) {
                    return Err(EngineError::invalid(
                        ErrorCode::EmptyQuestion,
                        id,
                        "choice options must not be empty",
                    ));
                }
            }
            QuestionKind::Score(spec) => {
                let count = spec.levels.count();
                if count < MIN_LEVELS as usize || count > MAX_LEVELS as usize {
                    return Err(EngineError::invalid(
                        ErrorCode::InvalidLevels,
                        id,
                        format!(
                            "a score rubric needs between {MIN_LEVELS} and {MAX_LEVELS} levels, \
                             got {count}"
                        ),
                    ));
                }
                if spec.levels.legend().iter().any(|l| l.trim().is_empty()) {
                    return Err(EngineError::invalid(
                        ErrorCode::EmptyQuestion,
                        id,
                        "score level labels must not be empty",
                    ));
                }
            }
        }

        Ok(())
    }

    /// Ask one question and type its answer.
    pub async fn answer(
        &self,
        state: &str,
        question: &Question,
        model: &str,
        calibration: Calibration,
    ) -> Result<(Answer, u32), EngineError> {
        self.validate_question(question)?;

        let ballot = ballot_for(&question.kind);
        let tally = self
            .tally(state, &ballot, model, calibration, &question.id)
            .await?;

        let answer = match &question.kind {
            QuestionKind::Noul(_) => Answer::Noul {
                // Option 0 is "Yes" by construction; see `ballot_for`.
                noul: tally.probabilities[0],
                confidence: tally.confidence,
                raw_logprobs: tally.logprobs,
                truncated: tally.truncated,
            },

            QuestionKind::Choice(spec) => {
                let winner = math::argmax(&tally.probabilities);
                Answer::Choice {
                    choice: spec.options[winner].clone(),
                    index: winner,
                    confidence: tally.confidence,
                    probabilities: spec
                        .options
                        .iter()
                        .zip(&tally.probabilities)
                        .map(|(option, p)| OptionProbability {
                            option: option.clone(),
                            probability: *p,
                        })
                        .collect(),
                    raw_logprobs: tally.logprobs,
                    truncated: tally.truncated,
                }
            }

            QuestionKind::Score(spec) => {
                let legend = spec.levels.legend();
                let winner = math::argmax(&tally.probabilities);
                Answer::Score {
                    score: (winner + 1) as u8,
                    expected_score: math::expected_level(&tally.probabilities),
                    legend: legend[winner].clone(),
                    confidence: tally.confidence,
                    probabilities: legend
                        .iter()
                        .zip(&tally.probabilities)
                        .enumerate()
                        .map(|(i, (text, p))| LevelProbability {
                            level: (i + 1) as u8,
                            legend: text.clone(),
                            probability: *p,
                        })
                        .collect(),
                    raw_logprobs: tally.logprobs,
                    truncated: tally.truncated,
                }
            }
        };

        Ok((answer, tally.input_tokens))
    }

    /// The shared middle: prompt, one host call, read the labels back out.
    async fn tally(
        &self,
        state: &str,
        ballot: &Ballot<'_>,
        model: &str,
        calibration: Calibration,
        question_id: &str,
    ) -> Result<Tally, EngineError> {
        let label_set = labels::labels(ballot.options.len());
        let lettered: Vec<(&str, &str)> = label_set
            .iter()
            .copied()
            .zip(ballot.options.iter().map(String::as_str))
            .collect();

        let distribution = self
            .host
            .first_token(FirstTokenRequest {
                model: model.to_string(),
                system: Some(prompt::SYSTEM.to_string()),
                user: prompt::user_turn(state, ballot.question, &lettered),
                top_logprobs: self.max_options(),
                keep_alive: self.keep_alive.clone(),
            })
            .await?;

        read_labels(&distribution, &label_set, calibration, question_id)
    }
}

/// Reduce a question to its options. This is the only place the three primitives differ.
fn ballot_for(kind: &QuestionKind) -> Ballot<'_> {
    match kind {
        // "Yes" first, so the noul probability is always option 0.
        QuestionKind::Noul(question) => Ballot {
            question: Some(question),
            options: vec!["Yes".to_string(), "No".to_string()],
        },
        QuestionKind::Choice(spec) => Ballot {
            question: spec.question.as_deref(),
            options: spec.options.clone(),
        },
        QuestionKind::Score(spec) => Ballot {
            question: spec.question.as_deref(),
            options: spec.levels.legend(),
        },
    }
}

/// Map a token distribution onto the labels we offered.
fn read_labels(
    distribution: &FirstTokenDistribution,
    label_set: &[&str],
    calibration: Calibration,
    question_id: &str,
) -> Result<Tally, EngineError> {
    let mut logprobs = Vec::with_capacity(label_set.len());
    let mut observed_any = false;
    let mut truncated = false;

    for label in label_set {
        // Every token spelling this label counts toward it; see `math::logsumexp`.
        let variants: Vec<f64> = distribution
            .tokens
            .iter()
            .filter(|(token, _)| labels::matches(token, label))
            .map(|(_, lp)| *lp)
            .collect();

        if variants.is_empty() {
            // Ranked below everything the host reported, so the floor is a strict upper bound.
            truncated = true;
            logprobs.push(distribution.floor);
        } else {
            observed_any = true;
            logprobs.push(math::logsumexp(&variants));
        }
    }

    if !observed_any {
        return Err(EngineError::NoLabelMatched {
            question_id: question_id.to_string(),
            expected: label_set.iter().map(|l| l.to_string()).collect(),
            observed: distribution
                .tokens
                .iter()
                .take(5)
                .map(|(t, _)| t.clone())
                .collect(),
        });
    }

    let probabilities = math::softmax(&logprobs, calibration.temperature);
    let confidence = math::confidence(&probabilities);

    Ok(Tally {
        logprobs: label_set
            .iter()
            .map(|l| l.to_string())
            .zip(logprobs)
            .collect(),
        probabilities,
        confidence,
        truncated,
        input_tokens: distribution.input_tokens,
    })
}
