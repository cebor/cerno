//! Router assembly.

use crate::routes;
use crate::state::AppState;
use axum::Router;
use axum::routing::{get, post};
use cerno_types::{
    Answer, Calibration, ChoiceSpec, ErrorCode, ErrorResponse, LevelProbability, LevelSpec,
    ModelInfo, ModelsResponse, OptionProbability, Question, QuestionKind, ScoreSpec,
    SystemOneRequest, SystemOneResponse, Timing, Usage,
};
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
    paths(routes::systemone, routes::models),
    components(schemas(
        SystemOneRequest, SystemOneResponse, Question, QuestionKind, ChoiceSpec, ScoreSpec,
        LevelSpec, Calibration, Answer, OptionProbability, LevelProbability, Usage, Timing,
        ModelsResponse, ModelInfo, ErrorResponse, ErrorCode,
    )),
    tags((name = "cerno", description = "System-one decisions over local models")),
    info(
        title = "cerno",
        description = "Noul, Choice and Score against a locally hosted model. \
                       Each question is one forward pass: the answer is read from the \
                       probability distribution over the first generated token.",
    ),
)]
pub struct ApiDoc;

/// Build the application.
///
/// `/health` is registered on the outer router, outside `TraceLayer`, because `Router::layer`
/// only wraps routes registered before it. Health checks therefore produce no log lines at any
/// `RUST_LOG` level, which keeps a container's logs about requests that mean something.
pub fn build_router(state: AppState) -> Router {
    let traced = Router::new()
        .route("/v1/systemone", post(routes::systemone))
        .route("/v1/models", get(routes::models))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    Router::new()
        .route("/health", get(routes::health))
        .merge(SwaggerUi::new("/docs").url("/openapi.json", ApiDoc::openapi()))
        .merge(traced)
}
