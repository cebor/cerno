//! Reading answers back out.

use crate::Error;
use cerno_types::{Answer, SystemOneResponse, Usage};

/// The answers to one request, with accessors that fail loudly on the wrong id or type.
///
/// Reaching for `noul("team")` when `team` was a choice is a programming mistake, not a runtime
/// condition, and the error says which id and which types rather than handing back a default.
#[derive(Debug, Clone)]
pub struct Answers {
    response: SystemOneResponse,
}

fn name_of(answer: &Answer) -> &'static str {
    match answer {
        Answer::Noul { .. } => "noul",
        Answer::Choice { .. } => "choice",
        Answer::Score { .. } => "score",
    }
}

impl Answers {
    pub(crate) fn new(response: SystemOneResponse) -> Self {
        Self { response }
    }

    /// The model that answered, after alias resolution.
    pub fn model(&self) -> &str {
        &self.response.model
    }

    pub fn usage(&self) -> Usage {
        self.response.usage
    }

    pub fn timing_ms(&self) -> u64 {
        self.response.timing_ms.total
    }

    /// The raw answer for `id`.
    pub fn get(&self, id: &str) -> Result<&Answer, Error> {
        self.response
            .answers
            .get(id)
            .ok_or_else(|| Error::MissingAnswer(id.to_string()))
    }

    /// Probability that the answer to `id` is yes.
    pub fn noul(&self, id: &str) -> Result<f64, Error> {
        match self.get(id)? {
            Answer::Noul { noul, .. } => Ok(*noul),
            other => Err(self.wrong_type(id, "noul", other)),
        }
    }

    /// The winning option for `id`.
    pub fn choice(&self, id: &str) -> Result<&str, Error> {
        match self.get(id)? {
            Answer::Choice { choice, .. } => Ok(choice),
            other => Err(self.wrong_type(id, "choice", other)),
        }
    }

    /// The winning level for `id`, 1-based.
    pub fn score(&self, id: &str) -> Result<u8, Error> {
        match self.get(id)? {
            Answer::Score { score, .. } => Ok(*score),
            other => Err(self.wrong_type(id, "score", other)),
        }
    }

    /// The probability-weighted mean level for `id`.
    pub fn expected_score(&self, id: &str) -> Result<f64, Error> {
        match self.get(id)? {
            Answer::Score { expected_score, .. } => Ok(*expected_score),
            other => Err(self.wrong_type(id, "score", other)),
        }
    }

    /// The legend entry the winning level carries.
    pub fn legend(&self, id: &str) -> Result<&str, Error> {
        match self.get(id)? {
            Answer::Score { legend, .. } => Ok(legend),
            other => Err(self.wrong_type(id, "score", other)),
        }
    }

    /// How peaked the distribution behind `id` was, in `0.0..=1.0`.
    pub fn confidence(&self, id: &str) -> Result<f64, Error> {
        Ok(self.get(id)?.confidence())
    }

    /// Whether some label for `id` fell outside the host's reporting window, which makes its
    /// probability an upper bound rather than an observation.
    pub fn truncated(&self, id: &str) -> Result<bool, Error> {
        Ok(self.get(id)?.truncated())
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.response.answers.keys().map(String::as_str)
    }

    pub fn into_inner(self) -> SystemOneResponse {
        self.response
    }

    fn wrong_type(&self, id: &str, expected: &'static str, actual: &Answer) -> Error {
        Error::WrongType {
            id: id.to_string(),
            expected,
            actual: name_of(actual),
        }
    }
}
