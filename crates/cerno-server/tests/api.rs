//! End-to-end tests through the real router, with mockito standing in for the model host.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use cerno_core::Engine;
use cerno_host::HostKind;
use cerno_server::config::{CalibrationEntry, Config, ModelEntry};
use cerno_server::{AppState, build_router};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::Duration;
use tower::ServiceExt;

/// An Ollama-shaped reply whose first token is `B`, with A, C and D also ranked.
fn chat_reply() -> String {
    json!({
        "model": "test",
        "message": {"role": "assistant", "content": "B"},
        "done": true,
        "prompt_eval_count": 57,
        "logprobs": [{
            "token": "B",
            "logprob": -0.1,
            "top_logprobs": [
                {"token": "B", "logprob": -0.1},
                {"token": "A", "logprob": -2.4},
                {"token": "C", "logprob": -6.0},
                {"token": "D", "logprob": -9.0},
                {"token": "E", "logprob": -11.0}
            ]
        }]
    })
    .to_string()
}

fn config(url: &str, strict: bool) -> Config {
    Config {
        bind: "127.0.0.1:0".parse().unwrap(),
        host: HostKind::Ollama,
        host_url: url.to_string(),
        host_api_key: None,
        default_model: "small".into(),
        models: BTreeMap::from([(
            "small".to_string(),
            ModelEntry {
                model: "gemma4:e2b-it-qat".into(),
                calibration: CalibrationEntry { temperature: 1.0 },
            },
        )]),
        strict_models: strict,
        max_concurrent_questions: 4,
        keep_alive: Some("5m".into()),
        host_timeout: Duration::from_secs(5),
    }
}

fn app(config: Config) -> axum::Router {
    let host = cerno_host::connect(
        config.host,
        &config.host_url,
        config.host_api_key.clone(),
        config.host_timeout,
    )
    .unwrap();
    let engine = Engine::new(host, config.keep_alive.clone());
    build_router(AppState::new(engine, config))
}

async fn post(app: axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn get(app: axum::Router, path: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();

    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

#[tokio::test]
async fn health_answers_without_touching_the_host() {
    // No mock is registered: a request reaching Ollama would fail the test.
    let (status, body) = get(app(config("http://127.0.0.1:9", false)), "/health").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "ok");
}

/// One state, three primitives, one request.
#[tokio::test]
async fn systemone_answers_every_primitive_in_a_single_request() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/api/chat")
        .with_header("content-type", "application/json")
        .with_body(chat_reply())
        .expect(3)
        .create_async()
        .await;

    let (status, body) = post(
        app(config(&server.url(), false)),
        "/v1/systemone",
        json!({
            "state": "Printer is jammed.",
            "questions": [
                {"id": "urgent", "noul": "Is this urgent?"},
                {"id": "team", "choice": {"question": "Which team?", "options": ["IT", "Facility", "HR"]}},
                {"id": "sev", "score": {"question": "How severe?", "levels": 4}}
            ]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    mock.assert_async().await;

    // The mock always answers "B", which is option 2 for every primitive.
    assert_eq!(body["answers"]["urgent"]["type"], "noul");
    assert!(
        body["answers"]["urgent"]["noul"].as_f64().unwrap() < 0.5,
        "B is 'No'"
    );

    assert_eq!(body["answers"]["team"]["choice"], "Facility");
    assert_eq!(body["answers"]["team"]["index"], 1);
    assert_eq!(
        body["answers"]["team"]["probabilities"]
            .as_array()
            .unwrap()
            .len(),
        3
    );

    assert_eq!(body["answers"]["sev"]["score"], 2);
    assert_eq!(body["answers"]["sev"]["legend"], "2");

    // Alias resolution reaches the response, and usage sums over all three questions.
    assert_eq!(body["model"], "gemma4:e2b-it-qat");
    assert_eq!(body["usage"]["questions"], 3);
    assert_eq!(body["usage"]["input_tokens"], 57 * 3);
}

/// Every answer must carry the evidence it was derived from.
#[tokio::test]
async fn answers_carry_raw_logprobs_and_the_truncation_flag() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/api/chat")
        .with_body(chat_reply())
        .create_async()
        .await;

    let (_, body) = post(
        app(config(&server.url(), false)),
        "/v1/systemone",
        json!({"state": "s", "questions": [{"id": "q", "noul": "Urgent?"}]}),
    )
    .await;

    let answer = &body["answers"]["q"];
    assert_eq!(answer["raw_logprobs"]["A"], -2.4);
    assert_eq!(answer["raw_logprobs"]["B"], -0.1);
    assert_eq!(answer["truncated"], false);
}

/// The mock reports only A..E, so a 6-option choice leaves F outside the window.
#[tokio::test]
async fn an_unreported_label_is_flagged_as_truncated() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/api/chat")
        .with_body(chat_reply())
        .create_async()
        .await;

    let (status, body) = post(
        app(config(&server.url(), false)),
        "/v1/systemone",
        json!({
            "state": "s",
            "questions": [{"id": "q", "choice": {"options": ["a","b","c","d","e","f"]}}]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["answers"]["q"]["truncated"], true);
    // F fell back to the floor, the weakest reported entry.
    assert_eq!(body["answers"]["q"]["raw_logprobs"]["F"], -11.0);
}

#[tokio::test]
async fn a_request_with_too_many_options_is_rejected_before_the_host_is_called() {
    let mut server = mockito::Server::new_async().await;
    // Zero calls expected: validation must happen before any model is loaded.
    let mock = server
        .mock("POST", "/api/chat")
        .expect(0)
        .create_async()
        .await;

    let options: Vec<String> = (0..21).map(|i| format!("option {i}")).collect();
    let (status, body) = post(
        app(config(&server.url(), false)),
        "/v1/systemone",
        json!({"state": "s", "questions": [{"id": "q", "choice": {"options": options}}]}),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["code"], "too_many_options");
    assert_eq!(body["question_id"], "q");
    mock.assert_async().await;
}

#[tokio::test]
async fn invalid_requests_report_a_machine_readable_code() {
    let server = mockito::Server::new_async().await;
    let url = server.url();

    let cases = [
        (
            json!({"state": "", "questions": [{"id": "q", "noul": "a?"}]}),
            "empty_state",
        ),
        (json!({"state": "s", "questions": []}), "no_questions"),
        (
            json!({"state": "s", "questions": [{"id": "d", "noul": "a?"}, {"id": "d", "noul": "b?"}]}),
            "duplicate_question_id",
        ),
        (
            json!({"state": "s", "questions": [{"id": "q", "score": {"levels": 11}}]}),
            "invalid_levels",
        ),
        (
            json!({"state": "s", "questions": [{"id": "q", "choice": {"options": ["only"]}}]}),
            "too_few_options",
        ),
        (
            json!({"state": "s", "calibration": {"temperature": 0.0},
                   "questions": [{"id": "q", "noul": "a?"}]}),
            "invalid_calibration",
        ),
    ];

    for (request, expected) in cases {
        let (status, body) = post(app(config(&url, false)), "/v1/systemone", request).await;

        assert_eq!(
            status,
            StatusCode::UNPROCESSABLE_ENTITY,
            "{expected}: {body}"
        );
        assert_eq!(body["code"], expected, "{body}");
    }
}

/// A model answering in prose is an operator problem, not a caller problem.
#[tokio::test]
async fn a_model_that_ignores_the_instruction_is_a_bad_gateway() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/api/chat")
        .with_body(
            json!({
                "prompt_eval_count": 10,
                "logprobs": [{"token": "Sure", "top_logprobs": [
                    {"token": "Sure", "logprob": -0.2},
                    {"token": "Well", "logprob": -1.5}
                ]}]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let (status, body) = post(
        app(config(&server.url(), false)),
        "/v1/systemone",
        json!({"state": "s", "questions": [{"id": "q", "noul": "Urgent?"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["code"], "no_label_matched");
    assert_eq!(body["question_id"], "q");
    // The message has to name what actually came back, or it is not actionable.
    assert!(body["message"].as_str().unwrap().contains("Sure"), "{body}");
}

/// The same request through an OpenAI-compatible host: the answer comes out identically, and the
/// body carries what makes the distribution readable on that runtime.
#[tokio::test]
async fn an_openai_compatible_host_answers_the_same_request() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer sk-test")
        .match_body(mockito::Matcher::PartialJson(json!({
            "model": "gemma4:e2b-it-qat",
            "logprobs": true,
            "max_tokens": 1,
            "top_k": -1,
            "chat_template_kwargs": {"enable_thinking": false},
        })))
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "B"},
                    "logprobs": {"content": [{
                        "token": "B",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "B", "logprob": -0.1},
                            {"token": "A", "logprob": -2.4},
                            {"token": "C", "logprob": -6.0},
                            {"token": "D", "logprob": -9.0}
                        ]
                    }]}
                }],
                "usage": {"prompt_tokens": 57, "completion_tokens": 1}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let mut config = config(&format!("{}/v1", server.url()), false);
    config.host = HostKind::Vllm;
    config.host_api_key = Some("sk-test".into());

    let (status, body) = post(
        app(config),
        "/v1/systemone",
        json!({
            "state": "Printer is jammed.",
            "questions": [{"id": "team", "choice": {"question": "Which team?", "options": ["IT", "Facility", "HR"]}}]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    mock.assert_async().await;
    assert_eq!(body["answers"]["team"]["choice"], "Facility", "{body}");
}

/// A refused connection and a host that never answers are different faults, and a caller that
/// retries needs to tell them apart: one means Ollama is not running, the other means it is busy
/// or wedged.
#[tokio::test]
async fn a_refused_connection_is_a_bad_gateway() {
    // Bind then drop, so the port is known to be closed rather than merely unlikely to be open.
    let closed = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap()
    };

    let (status, body) = post(
        app(config(&format!("http://{closed}"), false)),
        "/v1/systemone",
        json!({"state": "s", "questions": [{"id": "q", "noul": "Urgent?"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
    assert_eq!(body["code"], "host_unavailable");
}

#[tokio::test]
async fn a_host_that_never_answers_is_a_gateway_timeout() {
    // Accept the connection and then do nothing, which is what a wedged runtime looks like.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((stream, _)) = listener.accept().await {
            held.push(stream);
        }
    });

    let mut config = config(&format!("http://{addr}"), false);
    config.host_timeout = Duration::from_millis(300);

    let (status, body) = post(
        app(config),
        "/v1/systemone",
        json!({"state": "s", "questions": [{"id": "q", "noul": "Urgent?"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert_eq!(body["code"], "host_timeout");
}

#[tokio::test]
async fn strict_mode_rejects_an_unconfigured_model() {
    let server = mockito::Server::new_async().await;

    let (status, body) = post(
        app(config(&server.url(), true)),
        "/v1/systemone",
        json!({"state": "s", "model": "nope:1b", "questions": [{"id": "q", "noul": "a?"}]}),
    )
    .await;

    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["code"], "unknown_model");
}

/// The request's calibration must win over the alias's.
#[tokio::test]
async fn a_request_calibration_overrides_the_configured_one() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/api/chat")
        .with_body(chat_reply())
        .expect_at_least(2)
        .create_async()
        .await;
    let url = server.url();

    let body = |t: f64| {
        json!({
            "state": "s",
            "calibration": {"temperature": t},
            "questions": [{"id": "q", "noul": "Urgent?"}]
        })
    };

    let (_, sharp) = post(app(config(&url, false)), "/v1/systemone", body(1.0)).await;
    let (_, flat) = post(app(config(&url, false)), "/v1/systemone", body(5.0)).await;

    let sharp_noul = sharp["answers"]["q"]["noul"].as_f64().unwrap();
    let flat_noul = flat["answers"]["q"]["noul"].as_f64().unwrap();

    // Flattening pulls the probability toward 0.5 without crossing it.
    assert!(
        flat_noul > sharp_noul,
        "{flat_noul} should exceed {sharp_noul}"
    );
    assert!(flat_noul < 0.5);
    // The evidence is untouched by calibration.
    assert_eq!(
        sharp["answers"]["q"]["raw_logprobs"],
        flat["answers"]["q"]["raw_logprobs"]
    );
}

#[tokio::test]
async fn models_lists_the_configured_aliases_and_the_default() {
    let server = mockito::Server::new_async().await;

    let (status, body) = get(app(config(&server.url(), false)), "/v1/models").await;
    let body: Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["default"], "small");
    assert_eq!(body["models"][0]["alias"], "small");
    assert_eq!(body["models"][0]["model"], "gemma4:e2b-it-qat");
}

#[tokio::test]
async fn the_openapi_document_is_served_and_describes_the_endpoint() {
    let server = mockito::Server::new_async().await;

    let (status, body) = get(app(config(&server.url(), false)), "/openapi.json").await;
    let doc: Value = serde_json::from_str(&body).unwrap();

    assert_eq!(status, StatusCode::OK);
    assert!(doc["paths"]["/v1/systemone"]["post"].is_object(), "{body}");
    assert!(doc["components"]["schemas"]["SystemOneRequest"].is_object());
}
