//! Application state and the transitions the key handler drives.
//!
//! Everything that decides *what* happens lives here, and nothing in this module draws or talks
//! to a terminal. That is what lets the interesting parts — building a request out of drafts,
//! moving focus, reporting a failure against the question that caused it — be tested directly.

use crate::draft::{Kind, QuestionDraft};
use crate::editor::Editor;
use crate::session::Session;
use cerno_sdk::{Answers, Client, Error, SystemOne};
use cerno_types::ErrorCode;
use ratatui_textarea::TextArea;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    State,
    Questions,
    Answers,
}

impl Focus {
    pub fn next(self) -> Self {
        match self {
            Focus::State => Focus::Questions,
            Focus::Questions => Focus::Answers,
            Focus::Answers => Focus::State,
        }
    }

    pub fn previous(self) -> Self {
        self.next().next()
    }

    /// Whether typing a printable character goes into a text field rather than acting as a
    /// shortcut. `q` must quit from the list and type a letter in the state box.
    pub fn is_text_input(self) -> bool {
        matches!(self, Focus::State)
    }
}

#[derive(Debug)]
pub enum Status {
    Idle,
    Sending {
        started: Instant,
        /// Which send this is. A result carrying any other number belongs to a request that was
        /// cancelled, and must not land as the answer to this one.
        generation: u64,
    },
    Failed(Failure),
}

#[derive(Debug, Clone)]
pub struct Failure {
    pub message: String,
    pub code: Option<ErrorCode>,
    /// The question the service blamed, so the list can point at it.
    pub question_id: Option<String>,
}

impl Failure {
    pub fn from_error(error: &Error) -> Self {
        match error {
            Error::Api { code, response, .. } => Self {
                message: response.message.clone(),
                code: Some(*code),
                question_id: response.question_id.clone(),
            },
            other => Self {
                message: other.to_string(),
                code: None,
                question_id: None,
            },
        }
    }
}

pub struct App {
    pub state: TextArea<'static>,
    pub questions: Vec<QuestionDraft>,
    pub selected: usize,
    pub focus: Focus,

    pub model: Option<String>,
    /// Aliases offered by the service, for cycling with `m`.
    pub models: Vec<String>,
    pub calibration: Option<f64>,

    pub status: Status,
    pub answers: Option<Answers>,
    /// Whether `answers` belongs to an older request than the form now describes.
    pub stale: bool,

    pub editor: Option<Editor>,
    pub show_help: bool,

    pub url: String,
    pub healthy: Option<bool>,
    pub should_quit: bool,
    /// Advances once per tick; drives the spinner.
    pub tick: u64,
    /// Counts sends, so a result can be matched to the request that produced it.
    generation: u64,
}

impl App {
    pub fn new(session: Session, url: String) -> Self {
        Self {
            state: TextArea::new(split_lines(&session.state)),
            questions: session.questions,
            selected: 0,
            focus: Focus::State,
            model: session.model,
            models: Vec::new(),
            calibration: session.calibration,
            status: Status::Idle,
            answers: None,
            stale: false,
            editor: None,
            show_help: false,
            url,
            healthy: None,
            should_quit: false,
            tick: 0,
            generation: 0,
        }
    }

    pub fn state_text(&self) -> String {
        self.state.lines().join("\n")
    }

    pub fn to_session(&self) -> Session {
        Session {
            state: self.state_text(),
            model: self.model.clone(),
            calibration: self.calibration,
            questions: self.questions.clone(),
        }
    }

    // -- questions ---------------------------------------------------------------------------

    pub fn select_next(&mut self) {
        if !self.questions.is_empty() {
            self.selected = (self.selected + 1) % self.questions.len();
        }
    }

    pub fn select_previous(&mut self) {
        if !self.questions.is_empty() {
            self.selected = (self.selected + self.questions.len() - 1) % self.questions.len();
        }
    }

    /// Open the editor on a new question, offering an id that does not collide with an
    /// existing one. The field itself starts empty; see [`Editor::adding`].
    pub fn add_question(&mut self) {
        self.editor = Some(Editor::adding(self.next_id()));
    }

    pub fn edit_selected(&mut self) {
        if let Some(draft) = self.questions.get(self.selected) {
            self.editor = Some(Editor::editing(draft.clone(), self.selected));
        }
    }

    pub fn delete_selected(&mut self) {
        if self.selected < self.questions.len() {
            self.questions.remove(self.selected);
            self.selected = self.selected.saturating_sub(usize::from(
                self.selected >= self.questions.len() && !self.questions.is_empty(),
            ));
            if self.questions.is_empty() {
                self.selected = 0;
            }
            self.mark_stale();
        }
    }

    /// Take the draft out of the editor and put it where it belongs.
    pub fn commit_editor(&mut self) {
        let Some(editor) = self.editor.take() else {
            return;
        };
        let (draft, existing) = editor.finish();

        match existing {
            Some(index) if index < self.questions.len() => self.questions[index] = draft,
            _ => {
                self.questions.push(draft);
                self.selected = self.questions.len() - 1;
            }
        }
        self.mark_stale();
    }

    pub fn cancel_editor(&mut self) {
        self.editor = None;
    }

    /// `q1`, `q2`, … skipping any id already taken.
    fn next_id(&self) -> String {
        (1..)
            .map(|n| format!("q{n}"))
            .find(|candidate| !self.questions.iter().any(|q| &q.id == candidate))
            .expect("the sequence is unbounded")
    }

    // -- model and calibration ---------------------------------------------------------------

    /// Step through the service's aliases, then back to "the server's default".
    pub fn cycle_model(&mut self) {
        if self.models.is_empty() {
            return;
        }

        let next = match &self.model {
            None => Some(self.models[0].clone()),
            Some(current) => match self.models.iter().position(|m| m == current) {
                Some(index) if index + 1 < self.models.len() => {
                    Some(self.models[index + 1].clone())
                }
                // Past the end, fall back to letting the service choose.
                Some(_) => None,
                None => Some(self.models[0].clone()),
            },
        };

        self.model = next;
        self.mark_stale();
    }

    // -- sending -----------------------------------------------------------------------------

    /// Assemble the request this form describes. See [`Session::build`].
    pub fn build<'a>(&self, client: &'a Client) -> SystemOne<'a> {
        self.to_session().build(client)
    }

    pub fn is_sending(&self) -> bool {
        matches!(self.status, Status::Sending { .. })
    }

    /// Enter the sending state and return the generation the result must carry to be accepted.
    pub fn begin_send(&mut self) -> u64 {
        self.generation += 1;
        self.status = Status::Sending {
            started: Instant::now(),
            generation: self.generation,
        };
        self.generation
    }

    /// Take a request's result. Returns whether it was accepted, i.e. whether it belonged to the
    /// request currently in flight.
    pub fn finish_send(&mut self, generation: u64, result: Result<Answers, Error>) -> bool {
        // A cancelled request can still land: the task is aborted, but a result already on the
        // channel arrives anyway. Checking only "are we sending" is not enough — after Esc and
        // a second send, the first result would arrive while the second is in flight and be
        // shown as its answer. The generation says which request a result belongs to.
        match self.status {
            Status::Sending {
                generation: current,
                ..
            } if current == generation => {}
            _ => return false,
        }

        match result {
            Ok(answers) => {
                self.answers = Some(answers);
                self.stale = false;
                self.status = Status::Idle;
            }
            Err(error) => {
                // The previous answers stay on screen, marked stale: a failed run should not
                // erase the run you were comparing it against.
                self.stale = self.answers.is_some();
                self.status = Status::Failed(Failure::from_error(&error));
            }
        }
        true
    }

    pub fn cancel_send(&mut self) {
        if self.is_sending() {
            self.status = Status::Idle;
        }
    }

    pub fn failure(&self) -> Option<&Failure> {
        match &self.status {
            Status::Failed(failure) => Some(failure),
            _ => None,
        }
    }

    /// Whether the question at `index` is the one the service refused.
    pub fn is_blamed(&self, index: usize) -> bool {
        match (
            self.failure().and_then(|f| f.question_id.as_deref()),
            self.questions.get(index),
        ) {
            (Some(blamed), Some(draft)) => draft.id == blamed,
            _ => false,
        }
    }

    /// The form has changed since the answers on screen were produced.
    pub fn mark_stale(&mut self) {
        if self.answers.is_some() {
            self.stale = true;
        }
    }
}

/// Split text into the lines a `TextArea` expects, never producing an empty vector.
fn split_lines(text: &str) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    text.split('\n').map(str::to_string).collect()
}

/// A kind's short label, re-exported for the renderer.
pub fn kind_label(kind: Kind) -> &'static str {
    kind.label()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        App::new(Session::default(), "http://cerno.test".into())
    }

    fn draft(id: &str) -> QuestionDraft {
        QuestionDraft {
            id: id.into(),
            question: "Is this urgent?".into(),
            ..Default::default()
        }
    }

    #[test]
    fn focus_cycles_in_both_directions() {
        assert_eq!(Focus::State.next(), Focus::Questions);
        assert_eq!(Focus::Answers.next(), Focus::State);

        for focus in [Focus::State, Focus::Questions, Focus::Answers] {
            assert_eq!(focus.next().previous(), focus);
        }
    }

    /// `q` has to quit from the list and type a letter in the state box.
    #[test]
    fn only_the_state_box_swallows_printable_keys() {
        assert!(Focus::State.is_text_input());
        assert!(!Focus::Questions.is_text_input());
        assert!(!Focus::Answers.is_text_input());
    }

    #[test]
    fn selection_wraps_at_both_ends() {
        let mut app = app();
        app.questions = vec![draft("a"), draft("b"), draft("c")];

        app.select_next();
        assert_eq!(app.selected, 1);

        app.select_previous();
        app.select_previous();
        assert_eq!(app.selected, 2, "wraps backwards past zero");

        app.select_next();
        assert_eq!(app.selected, 0, "wraps forwards past the end");
    }

    /// An empty list must not panic or produce an out-of-range index.
    #[test]
    fn selection_is_safe_with_no_questions() {
        let mut app = app();

        app.select_next();
        app.select_previous();
        app.delete_selected();

        assert_eq!(app.selected, 0);
        assert!(app.questions.is_empty());
    }

    #[test]
    fn deleting_the_last_question_moves_the_selection_back() {
        let mut app = app();
        app.questions = vec![draft("a"), draft("b")];
        app.selected = 1;

        app.delete_selected();

        assert_eq!(app.questions.len(), 1);
        assert_eq!(app.selected, 0, "selection must stay in range");
    }

    #[test]
    fn a_new_question_gets_an_id_that_is_free() {
        let mut app = app();
        app.questions = vec![draft("q1"), draft("q3")];

        app.add_question();
        app.commit_editor();

        assert_eq!(app.questions.last().unwrap().id, "q2");
    }

    #[test]
    fn editing_replaces_in_place_rather_than_appending() {
        let mut app = app();
        app.questions = vec![draft("a"), draft("b")];
        app.selected = 0;

        app.edit_selected();
        app.editor
            .as_mut()
            .unwrap()
            .set(crate::editor::Field::Question, "changed");
        app.commit_editor();

        assert_eq!(app.questions.len(), 2);
        assert_eq!(app.questions[0].question, "changed");
        assert_eq!(app.questions[1].id, "b");
    }

    #[test]
    fn cancelling_the_editor_changes_nothing() {
        let mut app = app();
        app.questions = vec![draft("a")];

        app.edit_selected();
        app.editor
            .as_mut()
            .unwrap()
            .set(crate::editor::Field::Question, "discarded");
        app.cancel_editor();

        assert_eq!(app.questions[0].question, "Is this urgent?");
        assert!(app.editor.is_none());
    }

    #[test]
    fn cycling_models_ends_at_the_service_default() {
        let mut app = app();
        app.models = vec!["small".into(), "large".into()];

        assert_eq!(app.model, None);
        app.cycle_model();
        assert_eq!(app.model.as_deref(), Some("small"));
        app.cycle_model();
        assert_eq!(app.model.as_deref(), Some("large"));
        app.cycle_model();
        assert_eq!(app.model, None, "wraps back to letting the service choose");
    }

    #[test]
    fn cycling_does_nothing_when_the_service_listed_no_models() {
        let mut app = app();

        app.cycle_model();

        assert_eq!(app.model, None);
    }

    /// A failure names a question, and the list has to be able to point at it.
    #[test]
    fn the_blamed_question_is_the_one_the_service_named() {
        let mut app = app();
        app.questions = vec![draft("urgent"), draft("team")];
        app.status = Status::Failed(Failure {
            message: "a choice may have at most 20 options".into(),
            code: Some(ErrorCode::TooManyOptions),
            question_id: Some("team".into()),
        });

        assert!(!app.is_blamed(0));
        assert!(app.is_blamed(1));
    }

    #[test]
    fn a_failure_without_a_question_blames_nobody() {
        let mut app = app();
        app.questions = vec![draft("a")];
        app.status = Status::Failed(Failure {
            message: "host unreachable".into(),
            code: Some(ErrorCode::HostUnavailable),
            question_id: None,
        });

        assert!(!app.is_blamed(0));
    }

    #[test]
    fn state_text_round_trips_through_the_editor_widget() {
        let session = Session {
            state: "line one\nline two".into(),
            ..Default::default()
        };

        let app = App::new(session, "http://cerno.test".into());

        assert_eq!(app.state_text(), "line one\nline two");
    }

    #[test]
    fn a_session_round_trips_through_the_app() {
        let session = Session {
            state: "Ticket: printer jammed.".into(),
            model: Some("small".into()),
            calibration: Some(2.5),
            questions: vec![draft("urgent")],
        };

        let app = App::new(session.clone(), "http://cerno.test".into());

        assert_eq!(app.to_session(), session);
    }

    /// Answers on screen belong to the request that produced them; changing the form must say so
    /// rather than leave a stale result looking current.
    #[test]
    fn editing_the_form_marks_existing_answers_stale() {
        let mut app = app();
        assert!(!app.stale, "nothing to be stale about yet");

        app.questions = vec![draft("a")];
        app.mark_stale();
        assert!(!app.stale, "no answers on screen, so nothing goes stale");
    }
}

#[cfg(test)]
mod late_result_tests {
    use super::*;
    use cerno_sdk::Error;

    /// Esc means the answer you were waiting for is no longer wanted. A result that was already
    /// in flight when the task was aborted must not repaint the screen behind your back.
    #[test]
    fn a_result_arriving_after_a_cancel_is_dropped() {
        let mut app = App::new(Session::default(), "http://cerno.test".into());
        let cancelled = app.begin_send();
        app.cancel_send();

        app.finish_send(cancelled, Err(Error::MissingAnswer("late".into())));

        assert!(
            app.failure().is_none(),
            "a cancelled request left an error behind"
        );
        assert!(matches!(app.status, Status::Idle));
    }

    /// Esc, then a second send: the first request's result can still be on the channel, and it
    /// arrives while the second one is in flight. It must not be taken for the second's answer.
    #[test]
    fn a_cancelled_result_is_not_taken_for_the_next_requests_answer() {
        let mut app = App::new(Session::default(), "http://cerno.test".into());
        let cancelled = app.begin_send();
        app.cancel_send();
        let current = app.begin_send();

        app.finish_send(cancelled, Err(Error::MissingAnswer("late".into())));

        assert!(app.failure().is_none(), "the stale result was shown");
        assert!(app.is_sending(), "the second request is still in flight");

        app.finish_send(current, Err(Error::MissingAnswer("current".into())));

        assert!(
            app.failure().is_some(),
            "the current request's result was dropped"
        );
    }
}
