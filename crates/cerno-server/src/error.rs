//! Mapping engine failures onto HTTP.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use cerno_core::EngineError;
use cerno_types::{ErrorCode, ErrorResponse};

/// The HTTP status a failure deserves.
///
/// The split that matters is between "the request cannot be answered as written" (422, the
/// caller fixes it) and "the model or its runtime let us down" (5xx, the caller retries).
/// `NoLabelMatched` sits on the 5xx side deliberately: the request was well-formed, the model
/// simply did not follow the instruction, and that is an operator's problem — usually the wrong
/// model — not the caller's.
pub fn status_for(code: ErrorCode) -> StatusCode {
    match code {
        ErrorCode::TooManyOptions
        | ErrorCode::TooManyQuestions
        | ErrorCode::TooFewOptions
        | ErrorCode::InvalidLevels
        | ErrorCode::EmptyState
        | ErrorCode::EmptyQuestion
        | ErrorCode::DuplicateQuestionId
        | ErrorCode::NoQuestions
        | ErrorCode::UnknownModel
        | ErrorCode::InvalidCalibration => StatusCode::UNPROCESSABLE_ENTITY,

        ErrorCode::NoLabelMatched | ErrorCode::HostUnavailable => StatusCode::BAD_GATEWAY,
        ErrorCode::HostTimeout => StatusCode::GATEWAY_TIMEOUT,
        ErrorCode::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// An error on its way out of a handler.
pub struct ApiError {
    pub code: ErrorCode,
    pub message: String,
    pub question_id: Option<String>,
}

impl ApiError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            question_id: None,
        }
    }
}

impl From<EngineError> for ApiError {
    fn from(err: EngineError) -> Self {
        Self {
            code: err.code(),
            question_id: err.question_id().map(str::to_string),
            message: err.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = status_for(self.code);

        if status.is_server_error() {
            tracing::error!(code = ?self.code, question_id = ?self.question_id, "{}", self.message);
        } else {
            tracing::debug!(code = ?self.code, question_id = ?self.question_id, "{}", self.message);
        }

        (
            status,
            Json(ErrorResponse {
                code: self.code,
                message: self.message,
                question_id: self.question_id,
            }),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_mistakes_are_unprocessable() {
        for code in [
            ErrorCode::TooManyOptions,
            ErrorCode::TooManyQuestions,
            ErrorCode::TooFewOptions,
            ErrorCode::InvalidLevels,
            ErrorCode::EmptyState,
            ErrorCode::NoQuestions,
            ErrorCode::DuplicateQuestionId,
            ErrorCode::InvalidCalibration,
            ErrorCode::UnknownModel,
        ] {
            assert_eq!(
                status_for(code),
                StatusCode::UNPROCESSABLE_ENTITY,
                "{code:?}"
            );
        }
    }

    /// A well-formed request that the model fluffed is not the caller's fault to fix.
    #[test]
    fn model_and_host_failures_are_gateway_errors() {
        assert_eq!(
            status_for(ErrorCode::NoLabelMatched),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            status_for(ErrorCode::HostUnavailable),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            status_for(ErrorCode::HostTimeout),
            StatusCode::GATEWAY_TIMEOUT
        );
    }

    #[test]
    fn a_failure_inside_cerno_is_an_internal_error() {
        assert_eq!(
            status_for(ErrorCode::Internal),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
