//! The TUI builds the same requests as the three SDKs.
//!
//! `spec/conformance/cases.json` already describes each question neutrally — `kind`, `question`,
//! `options`, `levels` — which is exactly the shape of a `QuestionDraft`. Driving the form from
//! those cases hangs the fourth client on the same shared truth as the other three, so a change
//! to the wire format cannot quietly leave the TUI behind.

use cerno_sdk::Client;
use cerno_tui::app::App;
use cerno_tui::draft::{Kind, QuestionDraft};
use cerno_tui::session::Session;
use serde_json::Value;

fn cases() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/conformance/cases.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn kind_from(name: &str) -> Kind {
    match name {
        "noul" => Kind::Noul,
        "choice" => Kind::Choice,
        "score" => Kind::Score,
        other => panic!("unknown kind {other}"),
    }
}

/// Turn one conformance question into the fields a user would have typed.
fn draft_from(question: &Value) -> QuestionDraft {
    let levels = match &question["levels"] {
        Value::Null => String::new(),
        Value::Number(n) => n.to_string(),
        Value::Array(labels) => labels
            .iter()
            .map(|l| l.as_str().unwrap())
            .collect::<Vec<_>>()
            .join(", "),
        other => panic!("unexpected levels {other}"),
    };

    let options = question["options"]
        .as_array()
        .map(|list| {
            list.iter()
                .map(|o| o.as_str().unwrap())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();

    QuestionDraft {
        id: question["id"].as_str().unwrap().to_string(),
        kind: kind_from(question["kind"].as_str().unwrap()),
        question: question["question"].as_str().unwrap_or("").to_string(),
        options,
        levels,
    }
}

fn app_from(case: &Value) -> App {
    let session = Session {
        state: case["state"].as_str().unwrap().to_string(),
        model: case["model"].as_str().map(str::to_string),
        calibration: case["calibration"].as_f64(),
        questions: case["questions"]
            .as_array()
            .unwrap()
            .iter()
            .map(draft_from)
            .collect(),
    };
    App::new(session, "http://cerno.test".into())
}

#[test]
fn the_form_produces_the_same_body_as_the_sdks() {
    let cases = cases();
    let client = Client::new("http://unused.invalid").unwrap();

    for case in cases["requests"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let app = app_from(case);

        let produced = serde_json::to_value(app.build(&client).body()).unwrap();

        assert_eq!(produced, case["expect_body"], "case {name:?}");
    }
}

/// A round trip through the session file must not change what gets sent — the whole point of
/// saving it is to come back to the same request.
#[test]
fn a_saved_and_reloaded_form_sends_the_same_body() {
    let cases = cases();
    let client = Client::new("http://unused.invalid").unwrap();
    let dir = tempfile::tempdir().unwrap();

    for case in cases["requests"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let path = dir.path().join(format!("{}.json", name.replace(' ', "-")));

        let before = app_from(case);
        before.to_session().save_to(&path).unwrap();

        let after = App::new(Session::load_from(&path), "http://cerno.test".into());

        assert_eq!(
            serde_json::to_value(after.build(&client).body()).unwrap(),
            case["expect_body"],
            "case {name:?} changed across a save and reload"
        );
    }
}

/// Multi-line state is the normal case — people paste tickets — and the newlines have to reach
/// the service intact rather than being flattened by the text widget.
#[test]
fn multi_line_state_survives_the_text_widget() {
    let client = Client::new("http://unused.invalid").unwrap();
    let state = "Ticket 4711\n\nDrucker im 2. Stock\nnimmt nichts mehr an.";

    let app = App::new(
        Session {
            state: state.to_string(),
            questions: vec![QuestionDraft {
                id: "urgent".into(),
                kind: Kind::Noul,
                question: "Dringend?".into(),
                ..Default::default()
            }],
            ..Default::default()
        },
        "http://cerno.test".into(),
    );

    let body = serde_json::to_value(app.build(&client).body()).unwrap();

    assert_eq!(body["state"], state);
}
