//! The cerno HTTP service.
//!
//! [`build_router`] serves `POST /v1/systemone`, `GET /v1/models` and `/health`, with the
//! Swagger UI at `/docs` and the OpenAPI document at `/openapi.json`. The handlers validate a
//! request whole before any host call and hand it to [`cerno_core::Engine`], which does the
//! actual work over whichever [`cerno_host`] adapter [`Config`] selects.
//!
//! [`ApiDoc`] is also what the `cerno-openapi` binary prints into `spec/openapi.json`, the
//! document the SDKs are written against.

pub mod config;
pub mod error;
pub mod router;
pub mod routes;
pub mod state;

pub use config::Config;
pub use router::{ApiDoc, build_router};
pub use state::AppState;
