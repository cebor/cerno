//! Assembling a request.

use crate::{Client, Error, response::Answers};
use cerno_types::{
    Calibration, ChoiceSpec, LevelSpec, Question, QuestionKind, ScoreSpec, SystemOneRequest,
};

/// A score rubric: either a plain count or the text of each level.
///
/// The conversions exist so `.score("sev", "How bad?", 5)` and
/// `.score("sev", "How bad?", ["low", "high"])` both read naturally at the call site.
pub struct Levels(pub(crate) LevelSpec);

impl From<u8> for Levels {
    fn from(count: u8) -> Self {
        Self::count(count.into())
    }
}

impl Levels {
    /// A rubric of `count` generated levels. `From<u8>` covers a literal at the call site; this
    /// takes a count read from somewhere else, so a value too large for any rubric still reaches
    /// the service and is refused there with the bounds, rather than wrapping on the way.
    pub fn count(count: u32) -> Self {
        Self(LevelSpec::Count(count))
    }

    /// A rubric from level texts, lowest first.
    pub fn labels<S: AsRef<str>>(labels: impl IntoIterator<Item = S>) -> Self {
        Self(LevelSpec::Labels(collect(labels)))
    }
}

// Implemented for concrete container types rather than a blanket `IntoIterator`: a blanket impl
// overlaps `From<u8>`, because nothing stops a future `u8: IntoIterator`. These cover the shapes
// that actually appear at a call site, and `Levels::labels` covers the rest.
macro_rules! levels_from_labels {
    ($($t:ty),* $(,)?) => {
        $(impl From<$t> for Levels {
            fn from(labels: $t) -> Self {
                Self::labels(labels)
            }
        })*
    };
}

levels_from_labels!(Vec<String>, Vec<&str>, &[&str], &[String]);

impl<const N: usize> From<[&str; N]> for Levels {
    fn from(labels: [&str; N]) -> Self {
        Self::labels(labels)
    }
}

impl<const N: usize> From<[String; N]> for Levels {
    fn from(labels: [String; N]) -> Self {
        Self::labels(labels)
    }
}

/// A request under construction.
#[must_use = "a builder does nothing until send() is called"]
pub struct SystemOne<'a> {
    client: &'a Client,
    request: SystemOneRequest,
}

impl<'a> SystemOne<'a> {
    pub(crate) fn new(client: &'a Client, state: String) -> Self {
        Self {
            client,
            request: SystemOneRequest {
                state,
                model: None,
                calibration: None,
                questions: Vec::new(),
            },
        }
    }

    /// Name a model or a configured alias. The service's default applies otherwise.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.request.model = Some(model.into());
        self
    }

    /// Scale the label logits before they are normalised. Above 1 flattens, below 1 sharpens.
    pub fn calibration(mut self, temperature: f64) -> Self {
        self.request.calibration = Some(Calibration { temperature });
        self
    }

    /// How likely the answer to `question` is yes.
    pub fn noul(self, id: impl Into<String>, question: impl Into<String>) -> Self {
        self.push(id, QuestionKind::Noul(question.into()))
    }

    /// One of `options`.
    pub fn choice<S: AsRef<str>>(
        self,
        id: impl Into<String>,
        question: impl Into<String>,
        options: impl IntoIterator<Item = S>,
    ) -> Self {
        let spec = ChoiceSpec {
            question: Some(question.into()),
            options: collect(options),
        };
        self.push(id, QuestionKind::Choice(spec))
    }

    /// One of `options`, where the options speak for themselves.
    pub fn choice_of<S: AsRef<str>>(
        self,
        id: impl Into<String>,
        options: impl IntoIterator<Item = S>,
    ) -> Self {
        let spec = ChoiceSpec {
            question: None,
            options: collect(options),
        };
        self.push(id, QuestionKind::Choice(spec))
    }

    /// A position on a rubric: `5` for five generated levels, or a list of level texts.
    pub fn score(
        self,
        id: impl Into<String>,
        question: impl Into<String>,
        levels: impl Into<Levels>,
    ) -> Self {
        let spec = ScoreSpec {
            question: Some(question.into()),
            levels: levels.into().0,
        };
        self.push(id, QuestionKind::Score(spec))
    }

    /// The request as it will be sent. Useful for logging and for testing a builder chain
    /// without a server.
    pub fn body(&self) -> &SystemOneRequest {
        &self.request
    }

    pub async fn send(self) -> Result<Answers, Error> {
        let response = self.client.post_systemone(&self.request).await?;
        Ok(Answers::new(response))
    }

    fn push(mut self, id: impl Into<String>, kind: QuestionKind) -> Self {
        self.request.questions.push(Question {
            id: id.into(),
            kind,
        });
        self
    }
}

fn collect<S: AsRef<str>>(items: impl IntoIterator<Item = S>) -> Vec<String> {
    items.into_iter().map(|s| s.as_ref().to_string()).collect()
}
