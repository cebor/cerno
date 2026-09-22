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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct Calibration {
    pub temperature: f64,
}

impl Default for Calibration {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

/// One question, identified so its answer can be found in the response map.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct Question {
    pub id: String,
    #[serde(flatten)]
    pub kind: QuestionKind,
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
#[cfg_attr(feature = "schema", derive(ToSchema))]
pub struct ChoiceSpec {
    /// What is being asked. Omit when the options speak for themselves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    pub options: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
/// Every variant carries `raw_logprobs` and `truncated` so a caller can redo the normalisation
/// itself — calibration is a convenience here, never a place where information is lost.
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
