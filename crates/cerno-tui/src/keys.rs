//! Turning key presses into state changes.
//!
//! Kept apart from the event loop so it can be exercised directly: the loop owns the terminal
//! and the network, this owns the decisions.

use crate::app::{App, Focus};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

/// Something the event loop has to do that this module cannot: talk to the network, or stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    None,
    Send,
    Quit,
}

/// Calibration is stepped rather than typed. Exploring how flat a model needs to be is a
/// nudge-and-look loop, and a number field would make each nudge a five-key affair.
const TEMPERATURE_STEP: f64 = 0.25;
const TEMPERATURE_RANGE: std::ops::RangeInclusive<f64> = 0.25..=8.0;

pub fn handle(app: &mut App, key: KeyEvent) -> Action {
    // Windows terminals report press and release; acting on both would double every keystroke.
    if key.kind == KeyEventKind::Release {
        return Action::None;
    }

    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

    // Ctrl+C always quits, whatever has focus.
    if ctrl && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('q')) {
        return Action::Quit;
    }

    if app.show_help {
        app.show_help = false;
        return Action::None;
    }

    if app.editor.is_some() {
        return handle_editor(app, key);
    }

    if ctrl && matches!(key.code, KeyCode::Char('s')) {
        return if app.is_sending() {
            Action::None
        } else {
            Action::Send
        };
    }

    match key.code {
        // Esc abandons a request in flight; there is nothing else for it to close here.
        KeyCode::Esc => {
            app.cancel_send();
            return Action::None;
        }
        KeyCode::Tab => {
            app.focus = app.focus.next();
            return Action::None;
        }
        KeyCode::BackTab => {
            app.focus = app.focus.previous();
            return Action::None;
        }
        _ => {}
    }

    if app.focus == Focus::State {
        // Everything else belongs to the text box, including `q` and `a`.
        app.state.input(key);
        app.mark_stale();
        return Action::None;
    }

    handle_shortcut(app, key)
}

fn handle_shortcut(app: &mut App, key: KeyEvent) -> Action {
    match key.code {
        KeyCode::Char('q') => return Action::Quit,
        KeyCode::Char('?') => app.show_help = true,

        KeyCode::Up | KeyCode::Char('k') => app.select_previous(),
        KeyCode::Down | KeyCode::Char('j') => app.select_next(),

        KeyCode::Char('a') => app.add_question(),
        KeyCode::Char('e') | KeyCode::Enter => app.edit_selected(),
        KeyCode::Char('d') => app.delete_selected(),

        KeyCode::Char('m') => app.cycle_model(),

        KeyCode::Char('t') => {
            let next = app.calibration.unwrap_or(1.0) + TEMPERATURE_STEP;
            app.calibration = Some(next.min(*TEMPERATURE_RANGE.end()));
            app.mark_stale();
        }
        KeyCode::Char('T') => {
            let next = app.calibration.unwrap_or(1.0) - TEMPERATURE_STEP;
            app.calibration = Some(next.max(*TEMPERATURE_RANGE.start()));
            app.mark_stale();
        }
        // Back to whatever the service or the model alias says.
        KeyCode::Char('c') => {
            app.calibration = None;
            app.mark_stale();
        }

        KeyCode::Char('1') => app.focus = Focus::State,
        KeyCode::Char('2') => app.focus = Focus::Questions,

        _ => {}
    }
    Action::None
}

fn handle_editor(app: &mut App, key: KeyEvent) -> Action {
    match key.code {
        KeyCode::Esc => {
            app.cancel_editor();
            Action::None
        }
        KeyCode::Enter => {
            app.commit_editor();
            Action::None
        }
        _ => {
            if let Some(editor) = app.editor.as_mut() {
                editor.input(key);
            }
            Action::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draft::{Kind, QuestionDraft};
    use crate::session::Session;

    fn app() -> App {
        App::new(Session::default(), "http://cerno.test".into())
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }

    fn with_questions() -> App {
        let mut app = app();
        app.questions = vec![
            QuestionDraft {
                id: "a".into(),
                ..Default::default()
            },
            QuestionDraft {
                id: "b".into(),
                kind: Kind::Choice,
                ..Default::default()
            },
        ];
        app.focus = Focus::Questions;
        app
    }

    #[test]
    fn ctrl_c_quits_from_anywhere() {
        let mut app = app();
        assert_eq!(handle(&mut app, ctrl(KeyCode::Char('c'))), Action::Quit);

        app.focus = Focus::State;
        assert_eq!(handle(&mut app, ctrl(KeyCode::Char('c'))), Action::Quit);

        app.add_question();
        assert_eq!(handle(&mut app, ctrl(KeyCode::Char('c'))), Action::Quit);
    }

    /// The whole reason `Focus::is_text_input` exists.
    #[test]
    fn q_types_a_letter_in_the_state_box_and_quits_outside_it() {
        let mut app = app();
        app.focus = Focus::State;

        assert_eq!(handle(&mut app, key(KeyCode::Char('q'))), Action::None);
        assert_eq!(app.state_text(), "q");

        app.focus = Focus::Questions;
        assert_eq!(handle(&mut app, key(KeyCode::Char('q'))), Action::Quit);
    }

    /// Likewise `a` — it must not open the editor while someone is typing a word.
    #[test]
    fn shortcuts_do_not_fire_while_typing_the_state() {
        let mut app = app();
        app.focus = Focus::State;

        for ch in "against".chars() {
            handle(&mut app, key(KeyCode::Char(ch)));
        }

        assert_eq!(app.state_text(), "against");
        assert!(app.editor.is_none());
    }

    #[test]
    fn ctrl_s_asks_the_loop_to_send() {
        let mut app = with_questions();

        assert_eq!(handle(&mut app, ctrl(KeyCode::Char('s'))), Action::Send);
    }

    /// A second Ctrl+S while one request is in flight would queue a duplicate.
    #[test]
    fn ctrl_s_is_ignored_while_a_request_is_in_flight() {
        let mut app = with_questions();
        app.begin_send();

        assert_eq!(handle(&mut app, ctrl(KeyCode::Char('s'))), Action::None);
    }

    #[test]
    fn esc_cancels_a_request_in_flight() {
        let mut app = with_questions();
        app.begin_send();
        assert!(app.is_sending());

        handle(&mut app, key(KeyCode::Esc));

        assert!(!app.is_sending());
    }

    #[test]
    fn tab_moves_between_panes_in_both_directions() {
        let mut app = app();
        assert_eq!(app.focus, Focus::State);

        handle(&mut app, key(KeyCode::Tab));
        assert_eq!(app.focus, Focus::Questions);

        handle(&mut app, key(KeyCode::BackTab));
        assert_eq!(app.focus, Focus::State);
    }

    #[test]
    fn the_question_list_navigates_and_edits() {
        let mut app = with_questions();

        handle(&mut app, key(KeyCode::Down));
        assert_eq!(app.selected, 1);

        handle(&mut app, key(KeyCode::Char('e')));
        assert!(app.editor.is_some());

        handle(&mut app, key(KeyCode::Esc));
        assert!(app.editor.is_none());

        handle(&mut app, key(KeyCode::Char('d')));
        assert_eq!(app.questions.len(), 1);
    }

    #[test]
    fn a_opens_the_editor_and_enter_commits_it() {
        let mut app = with_questions();

        handle(&mut app, key(KeyCode::Char('a')));
        assert!(app.editor.is_some());

        handle(&mut app, key(KeyCode::Enter));

        assert!(app.editor.is_none());
        assert_eq!(app.questions.len(), 3);
    }

    /// Keys typed into the modal must reach its fields, not the shortcuts underneath.
    #[test]
    fn the_editor_swallows_shortcut_keys_while_it_is_open() {
        let mut app = with_questions();
        app.add_question();

        handle(&mut app, key(KeyCode::Char('d')));

        assert_eq!(app.questions.len(), 2, "`d` must not delete while editing");
        assert!(app.editor.is_some());
    }

    #[test]
    fn temperature_steps_up_and_down_and_clears() {
        let mut app = with_questions();

        handle(&mut app, key(KeyCode::Char('t')));
        assert_eq!(app.calibration, Some(1.25));

        handle(&mut app, key(KeyCode::Char('t')));
        assert_eq!(app.calibration, Some(1.5));

        handle(&mut app, key(KeyCode::Char('T')));
        assert_eq!(app.calibration, Some(1.25));

        handle(&mut app, key(KeyCode::Char('c')));
        assert_eq!(app.calibration, None, "back to the alias default");
    }

    /// The service rejects a temperature of zero or below; the form should never send one.
    #[test]
    fn temperature_stays_inside_a_usable_range() {
        let mut app = with_questions();

        for _ in 0..40 {
            handle(&mut app, key(KeyCode::Char('T')));
        }
        assert_eq!(app.calibration, Some(0.25));

        for _ in 0..80 {
            handle(&mut app, key(KeyCode::Char('t')));
        }
        assert_eq!(app.calibration, Some(8.0));
    }

    #[test]
    fn help_opens_and_any_key_closes_it() {
        let mut app = with_questions();

        handle(&mut app, key(KeyCode::Char('?')));
        assert!(app.show_help);

        handle(&mut app, key(KeyCode::Char('x')));
        assert!(!app.show_help);
    }

    /// Key releases arrive on Windows terminals and would otherwise double every press.
    #[test]
    fn key_releases_are_ignored() {
        let mut app = with_questions();
        let mut release = key(KeyCode::Char('d'));
        release.kind = KeyEventKind::Release;

        handle(&mut app, release);

        assert_eq!(app.questions.len(), 2);
    }

    #[test]
    fn the_number_keys_jump_to_a_pane() {
        let mut app = with_questions();

        handle(&mut app, key(KeyCode::Char('1')));
        assert_eq!(app.focus, Focus::State);
    }
}
