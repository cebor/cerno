//! Wire types for the cerno HTTP API.
//!
//! This crate is deliberately dependency-light: it carries `serde` always and `utoipa` only
//! behind the `schema` feature. That is what lets `cerno-sdk` share the exact types the server
//! serialises without pulling the engine or the OpenAPI machinery along with them.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg(feature = "schema")]
use utoipa::ToSchema;

/// Maximum number of answer options in a single question.
///
/// Bounded by Ollama's `top_logprobs` ceiling of 20: the engine reads the distribution over the
/// first generated token, so an option whose label is not among those 20 entries has no
/// observable probability. See [`Answer`]'s `truncated` flag for the partial case.
pub const MAX_OPTIONS: usize = 20;

/// Most questions in a single request.
///
/// Every question is a forward pass, and they share one concurrency limit with every other
/// caller. Without a bound, one request could queue thousands of passes ahead of everyone else.
pub const MAX_QUESTIONS: usize = 32;

/// Fewest levels a [`ScoreSpec`] rubric may have. One level is not a judgement.
pub const MIN_LEVELS: u8 = 2;

/// Most levels a [`ScoreSpec`] rubric may have, matching JEV's rubric range of 2..=10.
pub const MAX_LEVELS: u8 = 10;

// ---------------------------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------------------------

/// A shared state plus the questions to ask about it.
///
/// Unknown fields are refused rather than ignored: a misspelt `calibraton` would otherwise
/// be dropped without a word, and the answers would come back uncalibrated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct SystemOneRequest {
    /// The context every question is asked against.
    pub state: String,

    /// Model name or configured alias. Falls back to the server's default when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Overrides the calibration applied to label logits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<Calibration>,

    /// The questions. Ids must be unique within a request.
    pub questions: Vec<Question>,
}

/// Temperature scaling applied to label logits before the softmax.
///
/// `1.0` leaves the model's raw distribution untouched. Values above 1 flatten it, which is the
/// usual correction for instruct-tuned models that answer clear cases at a probability of 1.0.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct Calibration {
    pub temperature: f64,
}

impl Default for Calibration {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

/// One question, identified so its answer can be found in the response map. It names exactly one
/// primitive.
//
// Read through `RawQuestion` rather than derived: with `flatten`, serde would take the first
// primitive key it met and silently drop a second one, and it cannot refuse unknown fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "RawQuestion")]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct Question {
    pub id: String,
    #[serde(flatten)]
    pub kind: QuestionKind,
}

/// The wire form of a [`Question`], before it is checked to name exactly one primitive.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawQuestion {
    id: String,
    #[serde(default)]
    noul: Option<String>,
    #[serde(default)]
    choice: Option<ChoiceSpec>,
    #[serde(default)]
    score: Option<ScoreSpec>,
}

impl TryFrom<RawQuestion> for Question {
    type Error = String;

    fn try_from(raw: RawQuestion) -> Result<Self, Self::Error> {
        let kind = match (raw.noul, raw.choice, raw.score) {
            (Some(noul), None, None) => QuestionKind::Noul(noul),
            (None, Some(choice), None) => QuestionKind::Choice(choice),
            (None, None, Some(score)) => QuestionKind::Score(score),
            _ => {
                return Err(format!(
                    "question {:?} must have exactly one of `noul`, `choice` or `score`",
                    raw.id
                ));
            }
        };
        Ok(Self { id: raw.id, kind })
    }
}

/// The three primitives. Externally tagged, so the wire form is `{"id": .., "noul": ..}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub enum QuestionKind {
    /// How likely the answer is yes.
    Noul(String),
    /// One of up to [`MAX_OPTIONS`] options.
    Choice(ChoiceSpec),
    /// A position on a rubric of [`MIN_LEVELS`]..=[`MAX_LEVELS`] levels.
    Score(ScoreSpec),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ChoiceSpec {
    /// What is being asked. Omit when the options speak for themselves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    pub options: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ScoreSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    pub levels: LevelSpec,
}

/// A rubric given either as a plain count or as the text of each level.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub enum LevelSpec {
    /// `"levels": 5` — the engine generates the legend ("1" through "5").
    Count(u8),
    /// `"levels": ["gar nicht", "wenig", ...]` — the strings become the legend.
    Labels(Vec<String>),
}

impl LevelSpec {
    /// The legend, one entry per level, lowest first.
    pub fn legend(&self) -> Vec<String> {
        match self {
            Self::Count(n) => (1..=*n).map(|i| i.to_string()).collect(),
            Self::Labels(labels) => labels.clone(),
        }
    }

    pub fn count(&self) -> usize {
        match self {
            Self::Count(n) => *n as usize,
            Self::Labels(labels) => labels.len(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct SystemOneResponse {
    /// Answers keyed by the question id they were asked under.
    pub answers: BTreeMap<String, Answer>,
    /// The model actually used, after alias resolution.
    pub model: String,
    pub usage: Usage,
    pub timing_ms: Timing,
}

/// A typed answer.
///
/// Every variant carries `raw_logprobs`, `truncated` and `truncated_labels` so a caller can redo
/// the normalisation itself — calibration is a convenience here, never a place where information
/// is lost.
///
/// A label in `truncated_labels` fell outside the host's reporting window, so its entry in
/// `raw_logprobs` is the weakest reported logprob: an upper bound, not an observation.
/// `truncated` is true exactly when that list is not empty. Servers predating the list omit
/// it, which reads as empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub enum Answer {
    Noul {
        /// Probability that the answer is yes, in `0.0..=1.0`.
        noul: f64,
        confidence: f64,
        raw_logprobs: BTreeMap<String, f64>,
        truncated: bool,
        #[serde(default)]
        truncated_labels: Vec<String>,
    },
    Choice {
        /// The winning option, verbatim as it was supplied.
        choice: String,
        /// Its position in the request's `options`.
        index: usize,
        confidence: f64,
        /// Every option with its probability, in request order.
        probabilities: Vec<OptionProbability>,
        raw_logprobs: BTreeMap<String, f64>,
        truncated: bool,
        #[serde(default)]
        truncated_labels: Vec<String>,
    },
    Score {
        /// The most likely level, 1-based.
        score: u8,
        /// The probability-weighted mean level — often more useful than the argmax.
        expected_score: f64,
        /// The legend entry for `score`.
        legend: String,
        confidence: f64,
        probabilities: Vec<LevelProbability>,
        raw_logprobs: BTreeMap<String, f64>,
        truncated: bool,
        #[serde(default)]
        truncated_labels: Vec<String>,
    },
}

impl Answer {
    /// Whether some label fell outside the host's `top_logprobs` window, making its probability
    /// an upper bound rather than an observation.
    pub fn truncated(&self) -> bool {
        match self {
            Self::Noul { truncated, .. }
            | Self::Choice { truncated, .. }
            | Self::Score { truncated, .. } => *truncated,
        }
    }

    /// The labels whose logprob is a bound rather than an observation. Empty unless
    /// [`Answer::truncated`].
    pub fn truncated_labels(&self) -> &[String] {
        match self {
            Self::Noul {
                truncated_labels, ..
            }
            | Self::Choice {
                truncated_labels, ..
            }
            | Self::Score {
                truncated_labels, ..
            } => truncated_labels,
        }
    }

    /// How peaked the distribution is, in `0.0..=1.0`. See `cerno_core::math::confidence`.
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Noul { confidence, .. }
            | Self::Choice { confidence, .. }
            | Self::Score { confidence, .. } => *confidence,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct OptionProbability {
    pub option: String,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct LevelProbability {
    pub level: u8,
    pub legend: String,
    pub probability: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct Usage {
    /// Prompt tokens summed over every question in the request.
    pub input_tokens: u32,
    pub questions: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct Timing {
    pub total: u64,
}

// ---------------------------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ModelsResponse {
    pub models: Vec<ModelInfo>,
    /// Which entry answers a request that names no model.
    pub default: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ModelInfo {
    /// The name callers pass as `model`.
    pub alias: String,
    /// The host-level model this resolves to.
    pub model: String,
    pub calibration: Calibration,
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ErrorResponse {
    pub code: ErrorCode,
    pub message: String,
    /// Set when exactly one question is at fault.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question_id: Option<String>,
}

/// Machine-readable failure reasons. Callers branch on these, not on `message`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub enum ErrorCode {
    /// More than [`MAX_OPTIONS`] options. v1 does not split these across passes.
    TooManyOptions,
    /// More than [`MAX_QUESTIONS`] questions in one request.
    TooManyQuestions,
    /// The body is not a request at all: malformed JSON, a wrong type, an unknown field, or a
    /// question naming more or fewer than one primitive.
    InvalidRequest,
    /// Option list empty, or a single option — there is nothing to decide.
    TooFewOptions,
    /// Level count outside [`MIN_LEVELS`]..=[`MAX_LEVELS`].
    InvalidLevels,
    EmptyState,
    EmptyQuestion,
    DuplicateQuestionId,
    NoQuestions,
    UnknownModel,
    InvalidCalibration,
    /// The model answered with no recognisable label — it is not following the instruction.
    NoLabelMatched,
    HostUnavailable,
    HostTimeout,
    /// A failure inside cerno itself. Not the caller's doing, and not the model's.
    Internal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn question(value: serde_json::Value) -> Result<Question, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn each_primitive_reads_back_from_its_wire_form() {
        assert!(matches!(
            question(json!({"id": "u", "noul": "Urgent?"}))
                .unwrap()
                .kind,
            QuestionKind::Noul(_)
        ));
        assert!(matches!(
            question(json!({"id": "t", "choice": {"options": ["a", "b"]}}))
                .unwrap()
                .kind,
            QuestionKind::Choice(_)
        ));
        assert!(matches!(
            question(json!({"id": "s", "score": {"levels": 5}}))
                .unwrap()
                .kind,
            QuestionKind::Score(_)
        ));
    }

    /// With a derived `flatten`, the first key would win and the second would vanish.
    #[test]
    fn a_question_naming_two_primitives_is_refused() {
        let err = question(json!({
            "id": "x",
            "noul": "Urgent?",
            "choice": {"options": ["a", "b"]}
        }))
        .unwrap_err();

        assert!(err.to_string().contains("exactly one"), "{err}");
    }

    #[test]
    fn a_question_naming_no_primitive_is_refused() {
        assert!(question(json!({"id": "x"})).is_err());
    }

    /// A typo must fail loudly rather than drop the field it was meant to be.
    #[test]
    fn unknown_fields_are_refused_at_every_level() {
        assert!(question(json!({"id": "x", "chioce": {"options": ["a", "b"]}})).is_err());
        assert!(question(json!({"id": "x", "choice": {"options": ["a"], "opts": []}})).is_err());

        let request = json!({
            "state": "s",
            "calibraton": {"temperature": 2.0},
            "questions": [{"id": "u", "noul": "Urgent?"}]
        });
        assert!(serde_json::from_value::<SystemOneRequest>(request).is_err());
    }

    /// A question serialises to the same shape it is read from.
    #[test]
    fn a_question_round_trips() {
        let wire = json!({"id": "t", "choice": {"question": "Which?", "options": ["a", "b"]}});

        let back = serde_json::to_value(question(wire.clone()).unwrap()).unwrap();

        assert_eq!(back, wire);
    }

    /// A server that predates `truncated_labels` omits it; that must read as empty.
    #[test]
    fn a_missing_truncated_labels_reads_as_empty() {
        let answer: Answer = serde_json::from_value(json!({
            "type": "noul",
            "noul": 0.9,
            "confidence": 0.5,
            "raw_logprobs": {"A": -0.1, "B": -2.4},
            "truncated": false
        }))
        .unwrap();

        assert!(answer.truncated_labels().is_empty());
    }
}
