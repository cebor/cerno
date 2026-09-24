//! Runs the shared conformance cases from `spec/conformance/cases.json`.
//!
//! Every SDK runs these same cases. Request cases check that a builder chain produces the exact
//! JSON the service expects; response cases check that a body is read back into the right
//! values; error cases check the failure mapping. If the three clients ever disagree, they
//! disagree here first.

use cerno_sdk::{Client, Error, ErrorCode, Levels};
use serde_json::Value;

fn cases() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/conformance/cases.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn client(url: &str) -> Client {
    Client::new(url).unwrap()
}

/// Build each request case through the public builder and compare the serialised body.
#[test]
fn request_cases_produce_the_expected_body() {
    let cases = cases();
    let client = client("http://unused.invalid");

    for case in cases["requests"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut builder = client.systemone(case["state"].as_str().unwrap());

        if let Some(model) = case["model"].as_str() {
            builder = builder.model(model);
        }
        if let Some(temperature) = case["calibration"].as_f64() {
            builder = builder.calibration(temperature);
        }

        for question in case["questions"].as_array().unwrap() {
            let id = question["id"].as_str().unwrap();
            let q = question["question"].as_str();

            builder = match question["kind"].as_str().unwrap() {
                "noul" => builder.noul(id, q.unwrap()),
                "choice" => {
                    let options: Vec<&str> = question["options"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|o| o.as_str().unwrap())
                        .collect();
                    match q {
                        Some(q) => builder.choice(id, q, options),
                        None => builder.choice_of(id, options),
                    }
                }
                "score" => {
                    let raw = &question["levels"];
                    let levels: Levels = match raw.as_u64() {
                        Some(count) => (count as u8).into(),
                        None => Levels::labels(
                            raw.as_array().unwrap().iter().map(|l| l.as_str().unwrap()),
                        ),
                    };
                    builder.score(id, q.unwrap(), levels)
                }
                other => panic!("unknown kind {other} in case {name}"),
            };
        }

        let produced = serde_json::to_value(builder.body()).unwrap();
        assert_eq!(produced, case["expect_body"], "case {name:?}");
    }
}

#[tokio::test]
async fn response_cases_read_back_the_expected_values() {
    let cases = cases();

    for case in cases["responses"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();

        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/systemone")
            .with_header("content-type", "application/json")
            .with_body(case["body"].to_string())
            .create_async()
            .await;

        let answers = client(&server.url())
            .systemone("state")
            .noul("ignored", "the mock answers regardless")
            .send()
            .await
            .unwrap_or_else(|e| panic!("case {name:?}: {e}"));

        let expect = &case["expect"];
        if let Some(model) = expect["model"].as_str() {
            assert_eq!(answers.model(), model, "case {name:?}");
        }
        if let Some(usage) = expect["usage"].as_object() {
            assert_eq!(
                answers.usage().input_tokens as u64,
                usage["input_tokens"].as_u64().unwrap(),
                "case {name:?}"
            );
        }

        for (id, want) in expect["answers"].as_object().unwrap() {
            match want["type"].as_str().unwrap() {
                "noul" => {
                    if let Some(v) = want["noul"].as_f64() {
                        assert_eq!(answers.noul(id).unwrap(), v, "case {name:?} id {id}");
                    }
                }
                "choice" => {
                    assert_eq!(
                        answers.choice(id).unwrap(),
                        want["choice"].as_str().unwrap(),
                        "case {name:?} id {id}"
                    );
                    if let Some(index) = want["index"].as_u64() {
                        assert_eq!(
                            answers.index(id).unwrap() as u64,
                            index,
                            "case {name:?} id {id}"
                        );
                    }
                    assert!(answers.confidence(id).unwrap() > 0.0, "case {name:?}");
                }
                "score" => {
                    assert_eq!(
                        answers.score(id).unwrap() as u64,
                        want["score"].as_u64().unwrap(),
                        "case {name:?} id {id}"
                    );
                    if let Some(legend) = want["legend"].as_str() {
                        assert_eq!(answers.legend(id).unwrap(), legend, "case {name:?}");
                    }
                }
                other => panic!("unknown answer type {other}"),
            }

            if let Some(truncated) = want["truncated"].as_bool() {
                assert_eq!(answers.truncated(id).unwrap(), truncated, "case {name:?}");
            }
            if let Some(mass) = want["label_mass"].as_f64() {
                assert_eq!(
                    answers.label_mass(id).unwrap(),
                    mass,
                    "case {name:?} id {id}"
                );
            }
            if let Some(labels) = want["truncated_labels"].as_array() {
                let labels: Vec<&str> = labels.iter().map(|l| l.as_str().unwrap()).collect();
                assert_eq!(
                    answers.truncated_labels(id).unwrap(),
                    labels.as_slice(),
                    "case {name:?} id {id}"
                );
            }
        }
    }
}

#[tokio::test]
async fn error_cases_map_to_typed_errors() {
    let cases = cases();

    for case in cases["errors"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let status = case["status"].as_u64().unwrap() as u16;
        let expected_code: ErrorCode =
            serde_json::from_value(case["body"]["code"].clone()).unwrap();

        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/systemone")
            .with_status(status as usize)
            .with_header("content-type", "application/json")
            .with_body(case["body"].to_string())
            .create_async()
            .await;

        let err = client(&server.url())
            .systemone("state")
            .noul("q", "Urgent?")
            .send()
            .await
            .expect_err(&format!("case {name:?} should fail"));

        match err {
            Error::Api {
                status: got,
                code,
                response,
            } => {
                assert_eq!(got, status, "case {name:?}");
                assert_eq!(code, expected_code, "case {name:?}");
                assert_eq!(
                    response.question_id.as_deref(),
                    case["body"]["question_id"].as_str(),
                    "case {name:?}"
                );
            }
            other => panic!("case {name:?} gave {other:?}"),
        }
    }
}

/// A body that is not cerno's - a gateway's HTML page, some other JSON, a 2xx that is not an
/// answer - is reported as exactly that, never as a cerno error code it does not carry.
#[tokio::test]
async fn unexpected_cases_are_reported_as_unexpected() {
    let cases = cases();

    for case in cases["unexpected"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let status = case["status"].as_u64().unwrap() as u16;

        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/systemone")
            .with_status(status as usize)
            .with_body(case["body_text"].as_str().unwrap())
            .create_async()
            .await;

        let err = client(&server.url())
            .systemone("state")
            .noul("q", "Urgent?")
            .send()
            .await
            .expect_err(&format!("case {name:?} should fail"));

        assert!(
            matches!(err, Error::Unexpected { status: got, .. } if got == status),
            "case {name:?} gave {err:?}"
        );
    }
}

/// A newer service can send a code this client has never heard of. That is still the service
/// speaking, not a proxy, and its code and message must survive.
#[tokio::test]
async fn unknown_code_cases_keep_their_code_and_message() {
    let cases = cases();

    for case in cases["unknown_codes"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let status = case["status"].as_u64().unwrap() as u16;
        let body = &case["body"];

        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/systemone")
            .with_status(status as usize)
            .with_body(body.to_string())
            .create_async()
            .await;

        let err = client(&server.url())
            .systemone("state")
            .noul("q", "Urgent?")
            .send()
            .await
            .expect_err(&format!("case {name:?} should fail"));

        let Error::UnknownCode {
            status: got,
            code,
            message,
            question_id,
        } = err
        else {
            panic!("case {name:?} gave {err:?}")
        };
        assert_eq!(got, status, "case {name:?}");
        assert_eq!(code, body["code"].as_str().unwrap(), "case {name:?}");
        assert_eq!(message, body["message"].as_str().unwrap(), "case {name:?}");
        assert_eq!(
            question_id.as_deref(),
            body["question_id"].as_str(),
            "case {name:?}"
        );
    }
}

/// Asking for the wrong type is a programming mistake and must say so precisely.
#[tokio::test]
async fn reading_an_answer_as_the_wrong_type_names_both_types() {
    let cases = cases();
    let body = cases["responses"][0]["body"].to_string();

    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/systemone")
        .with_body(body)
        .create_async()
        .await;

    let answers = client(&server.url())
        .systemone("state")
        .noul("urgent", "Urgent?")
        .send()
        .await
        .unwrap();

    let err = answers.choice("urgent").unwrap_err();
    assert!(
        matches!(&err, Error::WrongType { id, expected: "choice", actual: "noul" } if id == "urgent"),
        "{err:?}"
    );

    // A noul carries no confidence, as in JEV, so asking for one is the same mistake.
    let err = answers.confidence("urgent").unwrap_err();
    assert!(
        matches!(
            &err,
            Error::WrongType {
                expected: "choice or score",
                actual: "noul",
                ..
            }
        ),
        "{err:?}"
    );

    assert!(matches!(
        answers.noul("nope").unwrap_err(),
        Error::MissingAnswer(_)
    ));
}

/// `into_inner` and `From` are the two directions of the same door; a response should survive
/// going through both.
#[tokio::test]
async fn a_response_can_be_wrapped_without_going_over_the_network() {
    let cases = cases();
    let body = cases["responses"][0]["body"].clone();
    let response: cerno_sdk::SystemOneResponse = serde_json::from_value(body).unwrap();

    let answers = cerno_sdk::Answers::from(response);

    assert_eq!(answers.model(), "gemma4:e2b-it-qat");
    assert_eq!(answers.choice("team").unwrap(), "Facility");
    assert_eq!(answers.score("sev").unwrap(), 5);
    assert_eq!(answers.into_inner().usage.questions, 3);
}

/// All three SDKs agree: a service that cannot be reached is not up, rather than an error.
#[tokio::test]
async fn health_is_false_when_the_service_cannot_be_reached() {
    // Bind then drop, so the port is known to be closed. A fixed port such as 9 is not: behind
    // some sandboxes and firewalls the connection attempt hangs until it times out instead of
    // being refused, and this test took 30 s to pass.
    let closed = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap()
    };
    assert!(!client(&format!("http://{closed}")).health().await);
}
