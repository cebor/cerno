//! What the screen actually says, rendered through ratatui's test backend.
//!
//! These assert on text rather than on exact layout: the point is that a probability, a winner
//! and a truncation warning reach the user, not that a box is 40 columns wide.

use cerno_sdk::{Answers, SystemOneResponse};
use cerno_tui::app::{App, Failure, Focus, Status};
use cerno_tui::draft::{Kind, QuestionDraft};
use cerno_tui::render;
use cerno_tui::session::Session;
use cerno_types::ErrorCode;
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::style::Color;

fn answers_from_conformance() -> Answers {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/conformance/cases.json");
    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

    let response: SystemOneResponse =
        serde_json::from_value(cases["responses"][0]["body"].clone()).unwrap();
    Answers::from(response)
}

fn truncated_answer() -> Answers {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/conformance/cases.json");
    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

    let response: SystemOneResponse =
        serde_json::from_value(cases["responses"][1]["body"].clone()).unwrap();
    Answers::from(response)
}

fn app_with_answers() -> App {
    let mut app = App::new(
        Session {
            state: "Ticket: Serverraum-Klima ausgefallen.".into(),
            questions: vec![
                QuestionDraft {
                    id: "urgent".into(),
                    kind: Kind::Noul,
                    question: "Ist das dringend?".into(),
                    ..Default::default()
                },
                QuestionDraft {
                    id: "team".into(),
                    kind: Kind::Choice,
                    question: "Welches Team?".into(),
                    options: "IT, Facility, HR".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        "http://localhost:3000".into(),
    );
    app.answers = Some(answers_from_conformance());
    app.healthy = Some(true);
    app
}

/// Render at a generous size and return the screen as lines of text.
fn screen(app: &App) -> Vec<String> {
    screen_sized(app, 120, 32)
}

fn screen_sized(app: &App, width: u16, height: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| render::draw(frame, app)).unwrap();

    let buffer = terminal.backend().buffer().clone();
    (0..height)
        .map(|y| {
            (0..width)
                .filter_map(|x| buffer.cell((x, y)).map(|c| c.symbol().to_string()))
                .collect::<String>()
        })
        .collect()
}

fn contains(lines: &[String], needle: &str) -> bool {
    lines.iter().any(|line| line.contains(needle))
}

#[test]
fn the_panes_and_the_state_text_are_on_screen() {
    let lines = screen(&app_with_answers());

    assert!(contains(&lines, "State (1)"), "{lines:#?}");
    assert!(contains(&lines, "Questions (2)"));
    assert!(contains(&lines, "Answers"));
    assert!(contains(&lines, "Serverraum-Klima"));
}

#[test]
fn the_question_list_shows_each_id_kind_and_summary() {
    let lines = screen(&app_with_answers());

    assert!(contains(&lines, "urgent"), "{lines:#?}");
    assert!(contains(&lines, "noul"));
    assert!(contains(&lines, "Ist das dringend?"));
    assert!(contains(&lines, "IT | Facility | HR"));
    assert!(contains(&lines, "+ add question"));
}

#[test]
fn a_noul_answer_shows_its_probability_against_yes_and_no() {
    let lines = screen(&app_with_answers());

    assert!(contains(&lines, "noul 0.9911"), "{lines:#?}");
    assert!(contains(&lines, "Yes"));
    assert!(contains(&lines, "No"));
    assert!(contains(&lines, "99.1%"));
}

#[test]
fn a_choice_answer_names_the_winner_and_bars_every_option() {
    let lines = screen(&app_with_answers());

    assert!(contains(&lines, "choice → Facility"), "{lines:#?}");
    assert!(contains(&lines, "98.6%"));
    assert!(contains(&lines, "1.1%"));
    // Every option is listed, not only the winner — the distribution is the point.
    assert!(contains(&lines, "IT"));
    assert!(contains(&lines, "HR"));
}

#[test]
fn a_score_answer_shows_the_level_its_legend_and_the_expected_value() {
    let lines = screen(&app_with_answers());

    assert!(contains(&lines, "score → 5"), "{lines:#?}");
    assert!(contains(&lines, "kritisch"));
    assert!(contains(&lines, "expected 4.88"));
}

#[test]
fn bars_are_actually_drawn() {
    let lines = screen(&app_with_answers());

    assert!(
        contains(&lines, "█"),
        "no filled block on screen: {lines:#?}"
    );
    assert!(contains(&lines, "░"), "no empty block on screen");
}

/// The flag exists so a caller can tell a bound from an observation; the screen has to say so.
#[test]
fn a_truncated_answer_is_called_out() {
    let mut app = app_with_answers();
    app.answers = Some(truncated_answer());

    let lines = screen(&app);

    assert!(contains(&lines, "truncated"), "{lines:#?}");
    assert!(contains(&lines, "upper bound"));
}

#[test]
fn the_status_bar_shows_the_service_and_its_health() {
    let lines = screen(&app_with_answers());

    assert!(contains(&lines, "http://localhost:3000"), "{lines:#?}");
    assert!(contains(&lines, "default model"));
}

#[test]
fn an_unreachable_service_says_so() {
    let mut app = app_with_answers();
    app.healthy = Some(false);

    let lines = screen(&app);

    assert!(contains(&lines, "unreachable"), "{lines:#?}");
}

#[test]
fn a_request_in_flight_shows_a_spinner_and_how_to_cancel() {
    let mut app = app_with_answers();
    app.begin_send();

    let lines = screen(&app);

    assert!(contains(&lines, "sending"), "{lines:#?}");
    assert!(contains(&lines, "Esc cancels"));
}

/// The service names the question it refused; the list has to point at it, in red.
#[test]
fn the_refused_question_is_marked_in_red() {
    let mut app = app_with_answers();
    app.status = Status::Failed(Failure {
        message: "a choice may have at most 20 options, got 21".into(),
        code: Some(ErrorCode::TooManyOptions),
        question_id: Some("team".into()),
    });

    let mut terminal = Terminal::new(TestBackend::new(120, 32)).unwrap();
    terminal.draw(|frame| render::draw(frame, &app)).unwrap();
    let buffer = terminal.backend().buffer().clone();

    let lines = screen(&app);
    assert!(
        contains(&lines, "too many options".to_lowercase().as_str())
            || contains(&lines, "at most 20 options"),
        "{lines:#?}"
    );

    // Only the left half: the same id also appears in the answers pane on the right, and that
    // copy is not the one under test.
    let left = 0..60u16;
    let row = (0..32)
        .find(|y| {
            left.clone()
                .filter_map(|x| buffer.cell((x, *y)).map(|c| c.symbol().to_string()))
                .collect::<String>()
                .contains("team")
        })
        .expect("the refused question should be in the question list");

    let has_red = left.clone().any(|x| {
        buffer
            .cell((x, row))
            .map(|c| c.style().fg == Some(Color::Red))
            .unwrap_or(false)
    });
    assert!(has_red, "the refused question was not marked, row {row}");

    // And the question that was not refused must be left alone.
    let other_row = (0..32)
        .find(|y| {
            left.clone()
                .filter_map(|x| buffer.cell((x, *y)).map(|c| c.symbol().to_string()))
                .collect::<String>()
                .contains("urgent")
        })
        .expect("the other question should be in the question list");
    let other_has_red = left.clone().any(|x| {
        buffer
            .cell((x, other_row))
            .map(|c| c.style().fg == Some(Color::Red))
            .unwrap_or(false)
    });
    assert!(
        !other_has_red,
        "an innocent question was marked, row {other_row}"
    );
}

#[test]
fn stale_answers_are_labelled_rather_than_cleared() {
    let mut app = app_with_answers();
    app.stale = true;

    let lines = screen(&app);

    assert!(contains(&lines, "Answers (stale)"), "{lines:#?}");
    // The numbers are still there to compare against.
    assert!(contains(&lines, "Facility"));
}

#[test]
fn an_empty_form_explains_what_to_do() {
    let app = App::new(Session::default(), "http://localhost:3000".into());

    let lines = screen(&app);

    assert!(contains(&lines, "Add a question"), "{lines:#?}");
}

#[test]
fn the_help_overlay_lists_the_keys() {
    let mut app = app_with_answers();
    app.show_help = true;

    let lines = screen(&app);

    assert!(contains(&lines, "Keys"), "{lines:#?}");
    assert!(contains(&lines, "Ctrl+S"));
    assert!(contains(&lines, "cancel a request"));
}

#[test]
fn the_editor_modal_shows_the_fields_for_its_kind() {
    let mut app = app_with_answers();
    app.focus = Focus::Questions;
    app.selected = 1;
    app.edit_selected();

    let lines = screen(&app);

    assert!(contains(&lines, "Edit question"), "{lines:#?}");
    assert!(contains(&lines, "id"));
    assert!(
        contains(&lines, "options"),
        "a choice must offer its options field"
    );
    assert!(contains(&lines, "[choice]"), "the current kind is marked");
}

/// A cramped terminal must still render rather than panic on a negative width.
#[test]
fn a_small_terminal_does_not_panic() {
    let app = app_with_answers();

    for (width, height) in [(40u16, 12u16), (30, 10), (20, 8)] {
        let lines = screen_sized(&app, width, height);
        assert_eq!(lines.len(), height as usize);
    }
}
