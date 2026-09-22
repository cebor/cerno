//! Guards the checked-in API surface.

use cerno_server::ApiDoc;
use serde_json::Value;
use utoipa::OpenApi;

fn repo_path(relative: &str) -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR is crates/cerno-server; the spec lives at the workspace root.
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

/// The SDKs are written against `spec/openapi.json`. If it drifts from the code, three clients
/// break at once and nothing says so — hence a test rather than a convention.
#[test]
fn the_checked_in_spec_matches_the_code() {
    let generated = ApiDoc::openapi().to_pretty_json().unwrap();
    let path = repo_path("spec/openapi.json");
    let checked_in = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} is missing: {e}", path.display()));

    assert_eq!(
        checked_in.trim(),
        generated.trim(),
        "spec/openapi.json is stale — regenerate it with:\n\
         cargo run -p cerno-server --bin cerno-openapi > spec/openapi.json"
    );
}

/// The conformance file is the shared truth every SDK is tested against, so a malformed or
/// truncated one would quietly weaken all three test suites.
#[test]
fn the_conformance_cases_are_well_formed() {
    let path = repo_path("spec/conformance/cases.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} is missing: {e}", path.display()));
    let cases: Value = serde_json::from_str(&text).unwrap();

    let requests = cases["requests"].as_array().expect("requests array");
    let responses = cases["responses"].as_array().expect("responses array");
    let errors = cases["errors"].as_array().expect("errors array");

    assert!(!requests.is_empty() && !responses.is_empty() && !errors.is_empty());

    for case in requests {
        assert!(case["name"].is_string(), "{case}");
        assert!(case["state"].is_string(), "{case}");
        assert!(case["questions"].is_array(), "{case}");
        assert!(case["expect_body"].is_object(), "{case}");
    }

    for case in responses {
        assert!(case["name"].is_string(), "{case}");
        assert!(case["body"].is_object(), "{case}");
        assert!(case["expect"].is_object(), "{case}");
    }

    for case in errors {
        assert!(case["status"].is_u64(), "{case}");
        assert!(case["body"]["code"].is_string(), "{case}");
    }
}

/// Request cases describe a body an SDK must produce. Deserialising each one through the
/// server's own types proves the target is reachable before any SDK is written against it.
#[test]
fn every_request_case_body_is_a_valid_request() {
    let text = std::fs::read_to_string(repo_path("spec/conformance/cases.json")).unwrap();
    let cases: Value = serde_json::from_str(&text).unwrap();

    for case in cases["requests"].as_array().unwrap() {
        let parsed: Result<cerno_types::SystemOneRequest, _> =
            serde_json::from_value(case["expect_body"].clone());

        assert!(
            parsed.is_ok(),
            "case {:?} does not deserialise: {:?}",
            case["name"],
            parsed.err()
        );
    }
}

/// And response cases describe a body an SDK must parse, so the server's types must produce it.
#[test]
fn every_response_case_body_is_a_valid_response() {
    let text = std::fs::read_to_string(repo_path("spec/conformance/cases.json")).unwrap();
    let cases: Value = serde_json::from_str(&text).unwrap();

    for case in cases["responses"].as_array().unwrap() {
        let parsed: Result<cerno_types::SystemOneResponse, _> =
            serde_json::from_value(case["body"].clone());

        assert!(
            parsed.is_ok(),
            "case {:?} does not deserialise: {:?}",
            case["name"],
            parsed.err()
        );
    }
}
