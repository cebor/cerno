//! Engine behaviour against a scripted host.
//!
//! The distributions here are shaped like the ones measured against real Ollama models,
//! including the awkward ones: a label pushed out of the top-20 window, a tokeniser that spells
//! one answer two ways, and a model that ignores the instruction entirely.

use cerno_core::{Engine, EngineError};
use cerno_host::{
    FirstTokenDistribution, FirstTokenRequest, HostCapabilities, HostError, ModelHost,
};
use cerno_types::{
    Answer, Calibration, ChoiceSpec, ErrorCode, LevelSpec, MAX_QUESTIONS, Question, QuestionKind,
    ScoreSpec, SystemOneRequest,
};
use std::sync::{Arc, Mutex};

/// A host that replays one distribution and records the prompt it was asked with.
struct ScriptedHost {
    tokens: Vec<(String, f64)>,
    seen: Mutex<Vec<FirstTokenRequest>>,
}

impl ScriptedHost {
    fn new(tokens: &[(&str, f64)]) -> Arc<Self> {
        Arc::new(Self {
            tokens: tokens.iter().map(|(t, l)| (t.to_string(), *l)).collect(),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn last_prompt(&self) -> String {
        self.seen.lock().unwrap().last().unwrap().user.clone()
    }
}

#[async_trait::async_trait]
impl ModelHost for ScriptedHost {
    fn capabilities(&self) -> HostCapabilities {
        HostCapabilities {
            max_top_logprobs: 20,
        }
    }

    fn name(&self) -> &str {
        "scripted"
    }

    async fn first_token(
        &self,
        req: FirstTokenRequest,
    ) -> Result<FirstTokenDistribution, HostError> {
        self.seen.lock().unwrap().push(req);

        let floor = self
            .tokens
            .iter()
            .map(|(_, lp)| *lp)
            .fold(f64::INFINITY, f64::min);

        Ok(FirstTokenDistribution {
            tokens: self.tokens.clone(),
            floor,
            input_tokens: 110,
            latency: std::time::Duration::from_millis(50),
        })
    }
}

fn engine(tokens: &[(&str, f64)]) -> (Engine, Arc<ScriptedHost>) {
    let host = ScriptedHost::new(tokens);
    (Engine::new(host.clone(), None), host)
}

fn noul(id: &str, q: &str) -> Question {
    Question {
        id: id.into(),
        kind: QuestionKind::Noul(q.into()),
    }
}

fn choice(id: &str, options: &[&str]) -> Question {
    Question {
        id: id.into(),
        kind: QuestionKind::Choice(ChoiceSpec {
            question: Some("Which team?".into()),
            options: options.iter().map(|s| s.to_string()).collect(),
        }),
    }
}

fn score(id: &str, levels: LevelSpec) -> Question {
    Question {
        id: id.into(),
        kind: QuestionKind::Score(ScoreSpec {
            question: Some("How severe?".into()),
            levels,
        }),
    }
}

// -------------------------------------------------------------------------------------------
// The three primitives
// -------------------------------------------------------------------------------------------

/// "Yes" is option A by construction, so the noul is the probability of the A label.
#[tokio::test]
async fn noul_reports_the_probability_of_yes() {
    let (engine, host) = engine(&[("A", -0.1), ("B", -2.4)]);

    let (answer, tokens) = engine
        .answer(
            "Server room at 31C.",
            &noul("urgent", "Is this urgent?"),
            "m",
            Calibration::default(),
        )
        .await
        .unwrap();

    let Answer::Noul {
        noul,
        truncated,
        raw_logprobs,
        ..
    } = answer
    else {
        panic!("expected a noul, got {answer:?}");
    };
    assert!(noul > 0.9, "got {noul}");
    assert!(!truncated);
    assert_eq!(raw_logprobs.get("A"), Some(&-0.1));
    assert_eq!(tokens, 110);
    assert!(host.last_prompt().contains("A) Yes\nB) No"));
}

/// A noul that is really a "no" puts the mass on B, and the reported probability follows.
#[tokio::test]
async fn noul_below_a_half_when_the_model_says_no() {
    let (engine, _) = engine(&[("B", -0.05), ("A", -3.0)]);

    let (answer, _) = engine
        .answer(
            "Room booking question.",
            &noul("urgent", "Is this urgent?"),
            "m",
            Calibration::default(),
        )
        .await
        .unwrap();

    let Answer::Noul { noul, .. } = answer else {
        panic!()
    };
    assert!(noul < 0.1, "got {noul}");
}

#[tokio::test]
async fn choice_returns_the_winning_option_with_every_probability_in_request_order() {
    let (engine, host) = engine(&[("A", -5.2), ("B", -0.1), ("C", -9.0)]);

    let (answer, _) = engine
        .answer(
            "Printer jam.",
            &choice("team", &["IT", "Facility", "HR"]),
            "m",
            Calibration::default(),
        )
        .await
        .unwrap();

    let Answer::Choice {
        choice,
        index,
        probabilities,
        ..
    } = answer
    else {
        panic!()
    };
    assert_eq!(choice, "Facility");
    assert_eq!(index, 1);
    assert_eq!(
        probabilities
            .iter()
            .map(|p| p.option.as_str())
            .collect::<Vec<_>>(),
        vec!["IT", "Facility", "HR"]
    );
    assert!((probabilities.iter().map(|p| p.probability).sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(host.last_prompt().contains("A) IT\nB) Facility\nC) HR"));
}

#[tokio::test]
async fn score_reports_argmax_expected_value_and_legend() {
    // Mass split between level 4 (D) and level 5 (E), leaning on 4.
    let (engine, _) = engine(&[
        ("A", -12.0),
        ("B", -10.0),
        ("C", -6.0),
        ("D", -0.3),
        ("E", -1.5),
    ]);

    let (answer, _) = engine
        .answer(
            "Outage.",
            &score("sev", LevelSpec::Count(5)),
            "m",
            Calibration::default(),
        )
        .await
        .unwrap();

    let Answer::Score {
        score,
        expected_score,
        legend,
        probabilities,
        ..
    } = answer
    else {
        panic!()
    };
    assert_eq!(score, 4);
    assert_eq!(legend, "4");
    // Between the two levels holding the mass, nearer the more likely one.
    assert!((4.0..4.5).contains(&expected_score), "got {expected_score}");
    assert_eq!(probabilities.len(), 5);
    assert_eq!(probabilities[0].level, 1);
}

#[tokio::test]
async fn score_uses_caller_supplied_level_text_as_the_legend() {
    let (engine, host) = engine(&[("A", -4.0), ("B", -0.1), ("C", -3.0)]);
    let levels = LevelSpec::Labels(vec!["low".into(), "medium".into(), "high".into()]);

    let (answer, _) = engine
        .answer(
            "Ticket.",
            &score("sev", levels),
            "m",
            Calibration::default(),
        )
        .await
        .unwrap();

    let Answer::Score { legend, score, .. } = answer else {
        panic!()
    };
    assert_eq!(score, 2);
    assert_eq!(legend, "medium");
    assert!(host.last_prompt().contains("A) low\nB) medium\nC) high"));
}

// -------------------------------------------------------------------------------------------
// The measured edge cases
// -------------------------------------------------------------------------------------------

/// Measured against `gemma4:26b`: when the model is certain, the losing label drops out of the
/// top-20 window entirely. The answer must still come back, flagged, rather than failing.
#[tokio::test]
async fn a_label_outside_the_window_falls_back_to_the_floor_and_flags_truncation() {
    let (engine, _) = engine(&[("B", -0.001), ("**", -14.0), ("N", -15.5)]);

    let (answer, _) = engine
        .answer(
            "Room booking.",
            &noul("urgent", "Is this urgent?"),
            "m",
            Calibration::default(),
        )
        .await
        .unwrap();

    let Answer::Noul {
        noul,
        truncated,
        truncated_labels,
        raw_logprobs,
        ..
    } = answer
    else {
        panic!()
    };
    assert!(
        truncated,
        "A was not reported, so the answer is an upper bound"
    );
    // The answer names which label is the bound, so a caller can tell it from B's observation.
    assert_eq!(truncated_labels, vec!["A".to_string()]);
    // A took the floor, which is the weakest reported entry.
    assert_eq!(raw_logprobs.get("A"), Some(&-15.5));
    assert!(noul < 0.001, "got {noul}");
}

/// One answer, several spellings. Both tokens must count toward the same label rather than one
/// being picked and the other silently dropped.
#[tokio::test]
async fn token_variants_of_one_label_are_folded_together() {
    // "A" and " A" each hold half the mass; "B" alone holds slightly more than either.
    let (engine, _) = engine(&[("B", -0.7), ("A", -1.4), (" A", -1.4)]);

    let (answer, _) = engine
        .answer(
            "Ticket.",
            &noul("q", "Urgent?"),
            "m",
            Calibration::default(),
        )
        .await
        .unwrap();

    let Answer::Noul {
        noul, truncated, ..
    } = answer
    else {
        panic!()
    };
    assert!(!truncated, "A was observed, in two spellings");
    // Folded, A holds e^-1.4 * 2 = 0.493 against B's e^-0.7 = 0.497 — almost a tie.
    assert!((0.45..0.55).contains(&noul), "got {noul}");
}

/// A model that answers in prose instead of a letter is not usable, and the error says so
/// concretely enough to act on.
#[tokio::test]
async fn an_answer_with_no_recognisable_label_is_an_error_that_names_what_came_back() {
    let (engine, _) = engine(&[("Sure", -0.2), ("Well", -1.0), ("The", -2.0)]);

    let err = engine
        .answer(
            "Ticket.",
            &noul("urgent", "Urgent?"),
            "m",
            Calibration::default(),
        )
        .await
        .unwrap_err();

    assert_eq!(err.code(), ErrorCode::NoLabelMatched);
    assert_eq!(err.question_id(), Some("urgent"));
    let EngineError::NoLabelMatched {
        observed, expected, ..
    } = &err
    else {
        panic!()
    };
    assert_eq!(expected, &vec!["A".to_string(), "B".to_string()]);
    assert!(observed.contains(&"Sure".to_string()), "{observed:?}");
}

/// Calibration must flatten without reordering, and must not touch the reported logprobs.
#[tokio::test]
async fn calibration_flattens_probabilities_but_leaves_logprobs_and_ranking_alone() {
    let tokens = [("A", -0.005), ("B", -5.246), ("C", -11.5)];

    let (cold_engine, _) = engine(&tokens);
    let (raw, _) = cold_engine
        .answer(
            "s",
            &choice("team", &["IT", "HR", "Sales"]),
            "m",
            Calibration { temperature: 1.0 },
        )
        .await
        .unwrap();

    let (warm_engine, _) = engine(&tokens);
    let (warm, _) = warm_engine
        .answer(
            "s",
            &choice("team", &["IT", "HR", "Sales"]),
            "m",
            Calibration { temperature: 3.0 },
        )
        .await
        .unwrap();

    let Answer::Choice {
        probabilities: rp,
        raw_logprobs: rl,
        choice: rc,
        ..
    } = raw
    else {
        panic!()
    };
    let Answer::Choice {
        probabilities: wp,
        raw_logprobs: wl,
        choice: wc,
        ..
    } = warm
    else {
        panic!()
    };

    assert_eq!(rc, wc, "calibration must not change the winner");
    assert_eq!(rl, wl, "calibration must not rewrite the evidence");
    assert!(
        wp[0].probability < rp[0].probability,
        "{} vs {}",
        wp[0].probability,
        rp[0].probability
    );
    assert!(wp[1].probability > rp[1].probability);
}

/// The prompt is what the model actually sees; pin its shape end to end.
#[tokio::test]
async fn the_prompt_carries_context_question_and_the_letter_instruction() {
    let (engine, host) = engine(&[("A", -0.1), ("B", -2.0)]);

    engine
        .answer(
            "Printer is jammed.",
            &noul("q", "Is this urgent?"),
            "m",
            Calibration::default(),
        )
        .await
        .unwrap();

    let prompt = host.last_prompt();
    assert!(prompt.starts_with("CONTEXT:\nPrinter is jammed."));
    assert!(prompt.contains("QUESTION: Is this urgent?"));
    assert!(prompt.ends_with("Answer with one letter only."));
}

// -------------------------------------------------------------------------------------------
// Validation
// -------------------------------------------------------------------------------------------

fn request(questions: Vec<Question>) -> SystemOneRequest {
    SystemOneRequest {
        state: "A ticket.".into(),
        model: None,
        calibration: None,
        questions,
    }
}

#[tokio::test]
async fn validation_rejects_more_options_than_the_host_can_report() {
    let (engine, _) = engine(&[("A", -0.1)]);
    let many: Vec<String> = (0..21).map(|i| format!("option {i}")).collect();
    let refs: Vec<&str> = many.iter().map(String::as_str).collect();

    let err = engine
        .validate(&request(vec![choice("team", &refs)]))
        .unwrap_err();

    assert_eq!(err.code(), ErrorCode::TooManyOptions);
    assert_eq!(err.question_id(), Some("team"));
    // The message has to tell the caller what to do instead.
    assert!(err.to_string().contains("Split them"), "{err}");
}

#[tokio::test]
async fn validation_accepts_exactly_the_maximum() {
    let (engine, _) = engine(&[("A", -0.1)]);
    let many: Vec<String> = (0..20).map(|i| format!("option {i}")).collect();
    let refs: Vec<&str> = many.iter().map(String::as_str).collect();

    assert!(
        engine
            .validate(&request(vec![choice("team", &refs)]))
            .is_ok()
    );
}

#[tokio::test]
async fn validation_rejects_rubrics_outside_two_to_ten_levels() {
    let (engine, _) = engine(&[("A", -0.1)]);

    for bad in [LevelSpec::Count(1), LevelSpec::Count(11)] {
        let err = engine
            .validate(&request(vec![score("sev", bad)]))
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidLevels);
    }

    for good in [LevelSpec::Count(2), LevelSpec::Count(10)] {
        assert!(engine.validate(&request(vec![score("sev", good)])).is_ok());
    }
}

#[tokio::test]
async fn validation_rejects_duplicate_ids_empty_state_and_no_questions() {
    let (engine, _) = engine(&[("A", -0.1)]);

    let dupes = request(vec![noul("same", "a?"), noul("same", "b?")]);
    assert_eq!(
        engine.validate(&dupes).unwrap_err().code(),
        ErrorCode::DuplicateQuestionId
    );

    let mut empty_state = request(vec![noul("q", "a?")]);
    empty_state.state = "   \n ".into();
    assert_eq!(
        engine.validate(&empty_state).unwrap_err().code(),
        ErrorCode::EmptyState
    );

    assert_eq!(
        engine.validate(&request(vec![])).unwrap_err().code(),
        ErrorCode::NoQuestions
    );
}

/// One request must not be able to queue an unbounded number of forward passes.
#[tokio::test]
async fn validation_caps_the_number_of_questions() {
    let (engine, host) = engine(&[("A", -0.1)]);
    let questions = |n: usize| (0..n).map(|i| noul(&format!("q{i}"), "a?")).collect();

    assert!(engine.validate(&request(questions(MAX_QUESTIONS))).is_ok());
    assert_eq!(
        engine
            .validate(&request(questions(MAX_QUESTIONS + 1)))
            .unwrap_err()
            .code(),
        ErrorCode::TooManyQuestions
    );
    assert!(
        host.seen.lock().unwrap().is_empty(),
        "validation must not reach the host"
    );
}

#[tokio::test]
async fn validation_rejects_a_choice_with_fewer_than_two_options() {
    let (engine, _) = engine(&[("A", -0.1)]);

    let err = engine
        .validate(&request(vec![choice("team", &["IT"])]))
        .unwrap_err();

    assert_eq!(err.code(), ErrorCode::TooFewOptions);
}

#[tokio::test]
async fn validation_rejects_a_nonsensical_calibration_temperature() {
    let (engine, _) = engine(&[("A", -0.1)]);

    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut req = request(vec![noul("q", "a?")]);
        req.calibration = Some(Calibration { temperature: bad });

        assert_eq!(
            engine.validate(&req).unwrap_err().code(),
            ErrorCode::InvalidCalibration,
            "temperature {bad} should be rejected"
        );
    }
}
