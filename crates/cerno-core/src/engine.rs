//! The one path all three primitives take.
//!
//! Noul, Choice and Score differ only in what the options are and how the resulting
//! distribution is named. Everything between — labelling, prompting, the single host call,
//! folding token variants, substituting the floor, calibrating, normalising — is shared.

use crate::{labels, math, prompt};
use cerno_host::{FirstTokenDistribution, FirstTokenRequest, HostError, ModelHost};
use cerno_types::{
    Answer, Calibration, ErrorCode, LevelProbability, MAX_LEVELS, MAX_OPTIONS, MAX_QUESTIONS,
    MIN_LEVELS, OptionProbability, Question, QuestionKind, SystemOneRequest,
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

    /// The host answered 404 to a question naming `model`.
    ///
    /// Every runtime answers a model it does not have that way, and without `strict_models`
    /// that is where a misspelt `model` surfaces — the caller's to fix, so not a 5xx inviting a
    /// retry. But a host URL missing its `/v1` answers 404 to every model as well, so the
    /// message says what to check when it is not the model.
    #[error(
        "model {model:?} is not known to {host} ({source}); if every model is refused this way, \
         the host URL is probably wrong"
    )]
    UnknownModel {
        model: String,
        host: String,
        source: HostError,
    },

    #[error(transparent)]
    Host(#[from] HostError),
}

impl EngineError {
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Invalid { code, .. } => *code,
            Self::NoLabelMatched { .. } => ErrorCode::NoLabelMatched,
            Self::UnknownModel { .. } => ErrorCode::UnknownModel,
            Self::Host(HostError::Timeout(_)) => ErrorCode::HostTimeout,
            Self::Host(_) => ErrorCode::HostUnavailable,
        }
    }

    pub fn question_id(&self) -> Option<&str> {
        match self {
            Self::Invalid { question_id, .. } => question_id.as_deref(),
            Self::NoLabelMatched { question_id, .. } => Some(question_id),
            Self::UnknownModel { .. } | Self::Host(_) => None,
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
    /// The labels that fell outside the reported window and were given the floor.
    truncated_labels: Vec<String>,
    /// The probability the model itself put on the observed labels, before any normalising.
    label_mass: f64,
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

        if request.questions.len() > MAX_QUESTIONS {
            return Err(EngineError::invalid(
                ErrorCode::TooManyQuestions,
                None,
                format!(
                    "a request may have at most {MAX_QUESTIONS} questions, got {}; send the rest \
                     in a second request",
                    request.questions.len()
                ),
            ));
        }

        let mut seen = BTreeMap::new();
        for question in &request.questions {
            if question.id.trim().is_empty() {
                return Err(EngineError::invalid(
                    ErrorCode::EmptyQuestionId,
                    Some(&question.id),
                    "question ids must not be empty; the answer is found by its id",
                ));
            }
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
                if let Some(twice) = repeated(&spec.options) {
                    return Err(EngineError::invalid(
                        ErrorCode::DuplicateOption,
                        id,
                        format!(
                            "option {twice:?} appears more than once; the model would see it \
                             twice and split its probability between the copies"
                        ),
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
                let legend = spec.levels.legend();
                if legend.iter().any(|l| l.trim().is_empty()) {
                    return Err(EngineError::invalid(
                        ErrorCode::EmptyQuestion,
                        id,
                        "score level labels must not be empty",
                    ));
                }
                if let Some(twice) = repeated(&legend) {
                    return Err(EngineError::invalid(
                        ErrorCode::DuplicateOption,
                        id,
                        format!(
                            "level {twice:?} appears more than once; the model could not tell \
                             the two apart"
                        ),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Ask one question and type its answer. The `u32` is the prompt's token count.
    pub async fn answer(
        &self,
        state: &str,
        question: &Question,
        model: &str,
        calibration: Calibration,
    ) -> Result<(Answer, u32), EngineError> {
        let (answer, distribution) = self
            .answer_with_distribution(state, question, model, calibration)
            .await?;
        Ok((answer, distribution.input_tokens))
    }

    /// [`Engine::answer`], also handing back the distribution the answer was read from.
    ///
    /// The answer renormalises over the offered labels, so it cannot say whether the model's
    /// most likely token was a label at all. The distribution can, which is what the benchmark
    /// measures fidelity by.
    pub async fn answer_with_distribution(
        &self,
        state: &str,
        question: &Question,
        model: &str,
        calibration: Calibration,
    ) -> Result<(Answer, FirstTokenDistribution), EngineError> {
        self.validate_question(question)?;

        let ballot = ballot_for(&question.kind);
        let (tally, distribution) = self
            .tally(state, &ballot, model, calibration, &question.id)
            .await?;

        let answer = match &question.kind {
            QuestionKind::Noul(_) => Answer::Noul {
                // Option 0 is "Yes" by construction; see `ballot_for`.
                noul: tally.probabilities[0],
                raw_logprobs: tally.logprobs,
                truncated: !tally.truncated_labels.is_empty(),
                truncated_labels: tally.truncated_labels,
                label_mass: tally.label_mass,
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
                    truncated: !tally.truncated_labels.is_empty(),
                    truncated_labels: tally.truncated_labels,
                    label_mass: tally.label_mass,
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
                    truncated: !tally.truncated_labels.is_empty(),
                    truncated_labels: tally.truncated_labels,
                    label_mass: tally.label_mass,
                }
            }
        };

        Ok((answer, distribution))
    }

    /// The shared middle: prompt, one host call, read the labels back out.
    async fn tally(
        &self,
        state: &str,
        ballot: &Ballot<'_>,
        model: &str,
        calibration: Calibration,
        question_id: &str,
    ) -> Result<(Tally, FirstTokenDistribution), EngineError> {
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
            .await
            .map_err(|error| match error {
                HostError::Status { status: 404, .. } => EngineError::UnknownModel {
                    model: model.to_string(),
                    host: self.host.name().to_string(),
                    source: error,
                },
                other => EngineError::Host(other),
            })?;

        let tally = read_labels(&distribution, &label_set, calibration, question_id)?;
        Ok((tally, distribution))
    }
}

/// The first text that appears twice, compared the way the model sees it: trimmed.
fn repeated(texts: &[String]) -> Option<&str> {
    let mut seen = std::collections::BTreeSet::new();
    texts.iter().map(|t| t.trim()).find(|t| !seen.insert(*t))
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
            question: asked(&spec.question),
            options: spec.options.clone(),
        },
        QuestionKind::Score(spec) => Ballot {
            question: asked(&spec.question),
            options: spec.levels.legend(),
        },
    }
}

/// The question text, if there is one worth asking. A blank question is the same as none: it
/// would otherwise put an empty `QUESTION:` line in front of the options.
fn asked(question: &Option<String>) -> Option<&str> {
    question.as_deref().filter(|q| !q.trim().is_empty())
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
    let mut truncated_labels = Vec::new();
    // Observed labels only: a floor is a bound, and adding bounds would overstate the mass.
    let mut label_mass = 0.0;

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
            truncated_labels.push(label.to_string());
            logprobs.push(distribution.floor);
        } else {
            observed_any = true;
            let logprob = math::logsumexp(&variants);
            label_mass += logprob.exp();
            logprobs.push(logprob);
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
        truncated_labels,
        // Reported logprobs over-sum a little from rounding; the share cannot pass 1.
        label_mass: label_mass.min(1.0),
    })
}
