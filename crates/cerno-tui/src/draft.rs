//! A question as it is being typed, before it becomes a request.
//!
//! A draft keeps the text of *every* field, not only the ones its current kind uses. Switching a
//! question from choice to score and back has to leave the options you typed where they were —
//! losing them because a radio button moved is the kind of small betrayal that makes a form
//! annoying to use.

use cerno_sdk::{Levels, SystemOne};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Noul, Kind::Choice, Kind::Score];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Noul => "noul",
            Kind::Choice => "choice",
            Kind::Score => "score",
        }
    }

    /// The next kind, wrapping. Used by the left/right keys in the editor.
    pub fn next(self) -> Self {
        match self {
            Kind::Noul => Kind::Choice,
            Kind::Choice => Kind::Score,
            Kind::Score => Kind::Noul,
        }
    }

    pub fn previous(self) -> Self {
        self.next().next()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionDraft {
    pub id: String,
    pub kind: Kind,
    pub question: String,
    /// Raw text of the options field, comma separated.
    #[serde(default)]
    pub options: String,
    /// Raw text of the levels field: a count, or comma-separated level texts.
    #[serde(default)]
    pub levels: String,
}

impl Default for QuestionDraft {
    fn default() -> Self {
        Self {
            id: String::new(),
            kind: Kind::Noul,
            question: String::new(),
            options: String::new(),
            levels: "5".to_string(),
        }
    }
}

impl QuestionDraft {
    /// Add this draft to a request being built.
    ///
    /// Nothing is validated here. The service decides what it can answer, and its refusals are
    /// already written to be acted on; a second copy of those rules in the client would be a
    /// copy that eventually disagrees.
    pub fn apply<'a>(&self, builder: SystemOne<'a>) -> SystemOne<'a> {
        match self.kind {
            Kind::Noul => builder.noul(&self.id, &self.question),

            Kind::Choice => {
                let options = split_list(&self.options);
                if self.question.trim().is_empty() {
                    builder.choice_of(&self.id, options)
                } else {
                    builder.choice(&self.id, self.question.trim(), options)
                }
            }

            Kind::Score => {
                builder.score(&self.id, self.question.trim(), parse_levels(&self.levels))
            }
        }
    }

    /// One line describing the draft for the question list.
    pub fn summary(&self) -> String {
        match self.kind {
            Kind::Noul => self.question.trim().to_string(),
            Kind::Choice => split_list(&self.options).join(" | "),
            Kind::Score => match parse_levels_raw(&self.levels) {
                ParsedLevels::Count(n) => format!("{n} levels"),
                ParsedLevels::Labels(labels) => labels.join(" | "),
            },
        }
    }
}

/// Split a comma-separated field into trimmed, non-empty entries.
///
/// Empty entries are dropped rather than preserved: a trailing comma is how a list looks
/// mid-typing, and it should not turn into an empty option the service then rejects.
pub fn split_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(str::to_string)
        .collect()
}

/// What the levels field means, before it becomes an SDK type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedLevels {
    Count(u32),
    Labels(Vec<String>),
}

/// Read the levels field: a bare number is a count, anything else is a list of level texts.
pub fn parse_levels_raw(raw: &str) -> ParsedLevels {
    let trimmed = raw.trim();

    // Wider than any rubric, so "300" stays a count the service refuses by its bounds instead of
    // turning into a one-level list of the text "300".
    if let Ok(count) = trimmed.parse::<u32>() {
        return ParsedLevels::Count(count);
    }

    let labels = split_list(raw);
    if labels.is_empty() {
        // An empty field is still a score question; let the service say what is wrong with it
        // rather than inventing a rubric here.
        return ParsedLevels::Count(0);
    }
    ParsedLevels::Labels(labels)
}

pub fn parse_levels(raw: &str) -> Levels {
    match parse_levels_raw(raw) {
        ParsedLevels::Count(n) => Levels::count(n),
        ParsedLevels::Labels(labels) => Levels::labels(labels),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_field_is_split_and_trimmed() {
        assert_eq!(split_list("IT, Facility, HR"), vec!["IT", "Facility", "HR"]);
        assert_eq!(split_list("IT,Facility"), vec!["IT", "Facility"]);
        assert_eq!(split_list("  spaced  ,  out  "), vec!["spaced", "out"]);
    }

    /// A trailing comma is what a list looks like mid-typing.
    #[test]
    fn empty_entries_are_dropped_rather_than_sent() {
        assert_eq!(split_list("IT, Facility,"), vec!["IT", "Facility"]);
        assert_eq!(split_list("IT,,Facility"), vec!["IT", "Facility"]);
        assert!(split_list("").is_empty());
        assert!(split_list("   ,  , ").is_empty());
    }

    #[test]
    fn a_bare_number_is_a_level_count() {
        assert_eq!(parse_levels_raw("5"), ParsedLevels::Count(5));
        assert_eq!(parse_levels_raw("  10 "), ParsedLevels::Count(10));
    }

    #[test]
    fn anything_else_is_a_list_of_level_texts() {
        assert_eq!(
            parse_levels_raw("gering, mittel, hoch"),
            ParsedLevels::Labels(vec!["gering".into(), "mittel".into(), "hoch".into()])
        );
    }

    #[test]
    fn a_number_too_large_for_a_rubric_is_still_a_count() {
        assert_eq!(parse_levels_raw("300"), ParsedLevels::Count(300));
    }

    /// "3 stars" is not the number 3, and guessing it is would silently drop the word.
    #[test]
    fn a_number_with_text_is_a_label_not_a_count() {
        assert_eq!(
            parse_levels_raw("3 stars"),
            ParsedLevels::Labels(vec!["3 stars".into()])
        );
    }

    /// An empty rubric is the service's business to reject, not something to paper over.
    #[test]
    fn an_empty_levels_field_becomes_a_count_the_service_will_refuse() {
        assert_eq!(parse_levels_raw(""), ParsedLevels::Count(0));
        assert_eq!(parse_levels_raw("  ,  "), ParsedLevels::Count(0));
    }

    /// Switching kinds must not discard what was typed under the other kind.
    #[test]
    fn a_draft_keeps_every_field_across_a_kind_change() {
        let mut draft = QuestionDraft {
            id: "q".into(),
            kind: Kind::Choice,
            question: "Which team?".into(),
            options: "IT, Facility".into(),
            levels: "gering, hoch".into(),
        };

        draft.kind = Kind::Score;
        assert_eq!(draft.options, "IT, Facility");

        draft.kind = Kind::Choice;
        assert_eq!(draft.levels, "gering, hoch");
        assert_eq!(draft.summary(), "IT | Facility");
    }

    #[test]
    fn kinds_cycle_in_both_directions() {
        assert_eq!(Kind::Noul.next(), Kind::Choice);
        assert_eq!(Kind::Score.next(), Kind::Noul);
        assert_eq!(Kind::Noul.previous(), Kind::Score);

        for kind in Kind::ALL {
            assert_eq!(kind.next().previous(), kind);
        }
    }

    #[test]
    fn summaries_describe_each_kind() {
        let noul = QuestionDraft {
            id: "u".into(),
            question: "Is this urgent?".into(),
            ..Default::default()
        };
        assert_eq!(noul.summary(), "Is this urgent?");

        let score = QuestionDraft {
            kind: Kind::Score,
            levels: "5".into(),
            ..Default::default()
        };
        assert_eq!(score.summary(), "5 levels");
    }
}
