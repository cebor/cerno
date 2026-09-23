//! HTTP handlers.

use crate::error::{ApiError, ApiJson};
use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use cerno_types::{
    Answer, Calibration, ErrorCode, ErrorResponse, ModelInfo, ModelsResponse, SystemOneRequest,
    SystemOneResponse, Timing, Usage,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
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
///
/// The request succeeds or fails as a whole: if any question fails, the error names it and the
/// answers to the others are discarded. The same holds when the whole request, waiting for a
/// free slot included, outlasts the server's request timeout: that is a 504 `host_timeout`.
#[utoipa::path(
    post,
    path = "/v1/systemone",
    request_body = SystemOneRequest,
    responses(
        (status = 200, body = SystemOneResponse),
        (status = 400, body = ErrorResponse, description = "the body is not a valid request"),
        (status = 422, body = ErrorResponse, description = "the request cannot be answered as written"),
        (status = 502, body = ErrorResponse, description = "the model or its runtime failed"),
        (status = 504, body = ErrorResponse, description = "the host timed out"),
        (status = 500, body = ErrorResponse, description = "a failure inside cerno"),
    ),
    tag = "cerno",
)]
pub async fn systemone(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<SystemOneRequest>,
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

    // Shared by every question's task rather than copied into each: a state can be large.
    let state_text: Arc<str> = Arc::from(request.state.as_str());
    let deadline = started + state.config.request_timeout;

    let mut tasks = JoinSet::new();
    // Which question each task answers. A task that panics comes back as a bare JoinError, so
    // the id has to be recoverable from the task alone.
    let mut question_of = HashMap::new();
    for question in request.questions.clone() {
        let engine = state.engine.clone();
        let semaphore = state.semaphore.clone();
        let state_text = state_text.clone();
        let model = model.clone();

        let id = question.id.clone();
        let handle = tasks.spawn(async move {
            let _permit = semaphore
                .acquire_owned()
                .await
                .expect("semaphore is never closed");
            engine
                .answer(&state_text, &question, &model, calibration)
                .await
        });
        question_of.insert(handle.id(), id);
    }

    let mut answers: BTreeMap<String, Answer> = BTreeMap::new();
    let mut input_tokens: u32 = 0;

    // Returning early drops the JoinSet, which aborts every question still running or queued,
    // and gives their permits back.
    let out_of_time = || ApiError {
        code: ErrorCode::HostTimeout,
        message: format!(
            "the request did not finish within {:?}; ask fewer questions at once, or raise \
             CERNO_REQUEST_TIMEOUT_SECS",
            state.config.request_timeout
        ),
        question_id: None,
    };

    while let Some(joined) = tokio::time::timeout_at(deadline.into(), tasks.join_next_with_id())
        .await
        .map_err(|_| out_of_time())?
    {
        let (task, result) = joined.map_err(|e| {
            let id = question_of.get(&e.id()).cloned();
            ApiError {
                code: ErrorCode::Internal,
                message: format!("answering question {id:?} failed to complete: {e}"),
                question_id: id,
            }
        })?;

        let (answer, tokens) = result?;
        let id = question_of
            .remove(&task)
            .expect("every task was registered when it was spawned");
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
