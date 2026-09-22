//! HTTP handlers.

use crate::error::ApiError;
use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use cerno_types::{
    Answer, Calibration, ErrorCode, ErrorResponse, ModelInfo, ModelsResponse, SystemOneRequest,
    SystemOneResponse, Timing, Usage,
};
use std::collections::BTreeMap;
use std::time::Instant;
use tokio::task::JoinSet;

/// Liveness. Registered outside the traced router, so health checks never reach the logs.
pub async fn health() -> &'static str {
    "ok"
}

/// The models this server will answer for.
#[utoipa::path(
    get,
    path = "/v1/models",
    responses((status = 200, body = ModelsResponse)),
    tag = "cerno",
)]
pub async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        models: state
            .config
            .models
            .iter()
            .map(|(alias, entry)| ModelInfo {
                alias: alias.clone(),
                model: entry.model.clone(),
                calibration: entry.calibration.into(),
            })
            .collect(),
        default: state.config.default_model.clone(),
    })
}

/// Ask several questions about one state.
///
/// Every question is one forward pass, and they are independent, so they run concurrently up to
/// the configured limit rather than one after another.
#[utoipa::path(
    post,
    path = "/v1/systemone",
    request_body = SystemOneRequest,
    responses(
        (status = 200, body = SystemOneResponse),
        (status = 422, body = ErrorResponse, description = "the request cannot be answered as written"),
        (status = 502, body = ErrorResponse, description = "the model or its runtime failed"),
        (status = 504, body = ErrorResponse, description = "the host timed out"),
    ),
    tag = "cerno",
)]
pub async fn systemone(
    State(state): State<AppState>,
    Json(request): Json<SystemOneRequest>,
) -> Result<Json<SystemOneResponse>, ApiError> {
    let started = Instant::now();

    let requested = request
        .model
        .clone()
        .unwrap_or_else(|| state.config.default_model.clone());

    let (model, model_calibration) = state.config.resolve(&requested).ok_or_else(|| {
        ApiError::new(
            ErrorCode::UnknownModel,
            format!(
                "model {requested:?} is not configured and strict_models is on; known: {:?}",
                state.config.known_models()
            ),
        )
    })?;

    // Request beats alias beats identity.
    let calibration: Calibration = request.calibration.unwrap_or(model_calibration);

    // Validate every question before loading a model, so a bad request costs nothing.
    state.engine.validate(&request)?;

    let mut tasks = JoinSet::new();
    for question in request.questions.clone() {
        let engine = state.engine.clone();
        let semaphore = state.semaphore.clone();
        let state_text = request.state.clone();
        let model = model.clone();

        tasks.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .expect("semaphore is never closed");
            let id = question.id.clone();
            let result = engine
                .answer(&state_text, &question, &model, calibration)
                .await;
            (id, result)
        });
    }

    let mut answers: BTreeMap<String, Answer> = BTreeMap::new();
    let mut input_tokens: u32 = 0;

    while let Some(joined) = tasks.join_next().await {
        let (id, result) = joined.map_err(|e| {
            ApiError::new(
                ErrorCode::HostUnavailable,
                format!("answering question {e} failed to complete"),
            )
        })?;

        let (answer, tokens) = result?;
        input_tokens = input_tokens.saturating_add(tokens);
        answers.insert(id, answer);
    }

    Ok(Json(SystemOneResponse {
        usage: Usage {
            input_tokens,
            questions: answers.len(),
        },
        answers,
        model,
        timing_ms: Timing {
            total: started.elapsed().as_millis() as u64,
        },
    }))
}
