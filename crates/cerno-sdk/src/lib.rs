//! Rust client for the cerno service.
//!
//! ```no_run
//! # async fn example() -> Result<(), cerno_sdk::Error> {
//! use cerno_sdk::Client;
//!
//! let client = Client::new("http://localhost:3000")?;
//! let answers = client
//!     .systemone("Ticket: server room at 31C, rising.")
//!     .noul("urgent", "Is this urgent?")
//!     .choice("team", "Which team?", ["IT", "Facility", "HR"])
//!     .score("sev", "How severe?", 5)
//!     .send()
//!     .await?;
//!
//! assert!(answers.noul("urgent")? > 0.8);
//! assert_eq!(answers.choice("team")?, "Facility");
//! # Ok(())
//! # }
//! ```
//!
//! The wire types come from `cerno-types`, the same crate the server serialises from, so the
//! client cannot drift from the service it talks to.

mod builder;
mod response;

pub use builder::{Levels, SystemOne};
pub use response::Answers;

pub use cerno_types::{
    Answer, Calibration, ErrorCode, ErrorResponse, ModelInfo, ModelsResponse, SystemOneRequest,
    SystemOneResponse,
};

use std::time::Duration;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not reach cerno: {0}")]
    Transport(#[from] reqwest::Error),

    /// The service answered with a structured failure. Branch on `code`, not on the message.
    #[error("cerno returned {status} ({code:?}): {}", .response.message)]
    Api {
        status: u16,
        code: ErrorCode,
        response: ErrorResponse,
    },

    /// A non-2xx response that was not shaped like an `ErrorResponse` — a proxy or a gateway
    /// between client and service, most likely.
    #[error("cerno returned {status}: {body}")]
    Unexpected { status: u16, body: String },

    #[error("no answer for question {0:?}")]
    MissingAnswer(String),

    #[error("question {id:?} answered with a {actual}, not a {expected}")]
    WrongType {
        id: String,
        expected: &'static str,
        actual: &'static str,
    },
}

#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base_url: String,
}

impl Client {
    pub fn new(base_url: impl Into<String>) -> Result<Self, Error> {
        Self::with_timeout(base_url, DEFAULT_TIMEOUT)
    }

    pub fn with_timeout(base_url: impl Into<String>, timeout: Duration) -> Result<Self, Error> {
        Ok(Self {
            http: reqwest::Client::builder().timeout(timeout).build()?,
            base_url: base_url.into().trim_end_matches('/').to_string(),
        })
    }

    /// Start a request. Questions are added to the returned builder.
    pub fn systemone(&self, state: impl Into<String>) -> SystemOne<'_> {
        SystemOne::new(self, state.into())
    }

    /// The models this service will answer for.
    pub async fn models(&self) -> Result<ModelsResponse, Error> {
        let response = self
            .http
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await?;
        decode(response).await
    }

    /// Whether the service is up. An unreachable or unresponsive service is simply not up, the
    /// same answer the Python and TypeScript clients give.
    pub async fn health(&self) -> bool {
        match self
            .http
            .get(format!("{}/health", self.base_url))
            .send()
            .await
        {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }

    pub(crate) async fn post_systemone(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse, Error> {
        let response = self
            .http
            .post(format!("{}/v1/systemone", self.base_url))
            .json(request)
            .send()
            .await?;
        decode(response).await
    }
}

/// Turn a response into a value, or into the most specific error we can justify.
async fn decode<T: serde::de::DeserializeOwned>(response: reqwest::Response) -> Result<T, Error> {
    let status = response.status();
    let body = response.text().await?;

    if status.is_success() {
        return serde_json::from_str(&body).map_err(|_| Error::Unexpected {
            status: status.as_u16(),
            body,
        });
    }

    match serde_json::from_str::<ErrorResponse>(&body) {
        Ok(response) => Err(Error::Api {
            status: status.as_u16(),
            code: response.code,
            response,
        }),
        // Not our error shape, so do not pretend to know what went wrong.
        Err(_) => Err(Error::Unexpected {
            status: status.as_u16(),
            body,
        }),
    }
}
