//! The seam between cerno and whatever runs the model.
//!
//! A host knows nothing about Noul, Choice or Score. It answers exactly one question: given a
//! prompt, what is the probability distribution over the *first* token the model would generate?
//! All primitive logic lives in `cerno-core` and therefore holds for every host equally.
//!
//! Two adapters cover the runtimes: [`OllamaHost`] for Ollama's native API, and
//! [`OpenAiCompatHost`] for everything speaking `/v1/chat/completions`. [`connect`] picks one
//! from a [`HostKind`].

mod ollama;
mod openai;

pub use ollama::OllamaHost;
pub use openai::{Flavour, OpenAiCompatHost};

use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

/// Which runtime a server talks to. One per process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostKind {
    Ollama,
    /// OpenAI itself, or any server that speaks its API and nothing more.
    OpenAi,
    Vllm,
    LlamaCpp,
    LmStudio,
}

impl HostKind {
    /// Where the runtime listens out of the box.
    pub fn default_url(self) -> &'static str {
        match self {
            HostKind::Ollama => "http://localhost:11434",
            HostKind::OpenAi => "https://api.openai.com/v1",
            HostKind::Vllm => "http://localhost:8000/v1",
            HostKind::LlamaCpp => "http://localhost:8080/v1",
            HostKind::LmStudio => "http://localhost:1234/v1",
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown host {0:?}; expected one of ollama, openai, vllm, llamacpp, lmstudio")]
pub struct UnknownHostKind(String);

impl std::str::FromStr for HostKind {
    type Err = UnknownHostKind;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ollama" => Ok(HostKind::Ollama),
            "openai" => Ok(HostKind::OpenAi),
            "vllm" => Ok(HostKind::Vllm),
            "llamacpp" | "llama.cpp" => Ok(HostKind::LlamaCpp),
            "lmstudio" => Ok(HostKind::LmStudio),
            _ => Err(UnknownHostKind(s.to_string())),
        }
    }
}

impl std::fmt::Display for HostKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            HostKind::Ollama => "ollama",
            HostKind::OpenAi => "openai",
            HostKind::Vllm => "vllm",
            HostKind::LlamaCpp => "llamacpp",
            HostKind::LmStudio => "lmstudio",
        })
    }
}

/// Build the adapter for `kind`. Ollama takes no API key; the others send it as a bearer token.
pub fn connect(
    kind: HostKind,
    url: &str,
    api_key: Option<String>,
    timeout: Duration,
) -> Result<Arc<dyn ModelHost>, HostError> {
    let flavour = match kind {
        HostKind::Ollama => return Ok(Arc::new(OllamaHost::new(url, timeout)?)),
        HostKind::OpenAi => Flavour::Generic,
        HostKind::Vllm => Flavour::Vllm,
        HostKind::LlamaCpp => Flavour::LlamaCpp,
        HostKind::LmStudio => Flavour::LmStudio,
    };
    Ok(Arc::new(OpenAiCompatHost::new(
        url, api_key, flavour, timeout,
    )?))
}

/// What a host can and cannot do. `cerno-core` reads this to size its label alphabet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCapabilities {
    /// How many ranked tokens the host will report for one position.
    ///
    /// Ollama and OpenAI cap this at 20. A label beyond that rank is invisible, which is exactly
    /// the bound that limits a Choice to 20 options.
    pub max_top_logprobs: usize,
}

/// One prompt, one token, full distribution.
#[derive(Debug, Clone)]
pub struct FirstTokenRequest {
    pub model: String,
    pub system: Option<String>,
    pub user: String,
    pub top_logprobs: usize,
    /// How long the host should keep the model resident after answering. Only Ollama has this;
    /// the OpenAI-compatible runtimes keep their model loaded for the life of the process.
    pub keep_alive: Option<String>,
}

/// The distribution over the first generated token.
#[derive(Debug, Clone)]
pub struct FirstTokenDistribution {
    /// Ranked tokens, highest logprob first.
    pub tokens: Vec<(String, f64)>,
    /// The lowest logprob the host reported.
    ///
    /// A label missing from `tokens` ranked below every entry, so this value is a strict upper
    /// bound on its logprob. `cerno-core` substitutes it and flags the answer as truncated.
    pub floor: f64,
    pub input_tokens: u32,
    pub latency: Duration,
}

impl FirstTokenDistribution {
    /// The logprob of `token`, or `None` when it fell outside the reported window.
    pub fn logprob(&self, token: &str) -> Option<f64> {
        self.tokens
            .iter()
            .find(|(t, _)| t == token)
            .map(|(_, lp)| *lp)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum HostError {
    #[error("host unreachable: {0}")]
    Unavailable(String),

    #[error("host timed out after {0:?}")]
    Timeout(Duration),

    #[error("host returned {status}: {body}")]
    Status { status: u16, body: String },

    /// The host answered, but not in the shape we need — most often because the model or the
    /// runtime does not report logprobs at all.
    #[error(
        "host response carried no logprobs (model {model:?}); \
             the runtime must support top_logprobs for cerno to work"
    )]
    NoLogprobs { model: String },

    #[error("could not parse host response: {0}")]
    Protocol(String),

    /// A base URL no request could be sent to. Caught when the host is built, because otherwise
    /// the server starts and every request fails with reqwest's "builder error".
    #[error("host URL {url:?} is not usable: {reason}")]
    InvalidUrl { url: String, reason: String },
}

/// The base URL with any trailing slash removed, or why it cannot be one.
fn base_url(raw: &str) -> Result<String, HostError> {
    let trimmed = raw.trim().trim_end_matches('/');
    let invalid = |reason: String| HostError::InvalidUrl {
        url: raw.to_string(),
        reason,
    };

    let parsed = reqwest::Url::parse(trimmed).map_err(|e| invalid(e.to_string()))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        // `localhost:11434` parses, with `localhost` as its scheme, so naming the scheme found
        // would only confuse.
        return Err(invalid("it must start with http:// or https://".into()));
    }
    if parsed.host_str().is_none() {
        return Err(invalid("it names no host".into()));
    }
    Ok(trimmed.to_string())
}

/// How much of a failed response's body travels on in the error. The whole body is logged; the
/// error reaches the caller of the service, and an upstream body can carry account or request
/// details that are the operator's business, not theirs.
const MAX_ERROR_BODY: usize = 300;

impl HostError {
    /// A non-success status from the host, with its body logged in full and kept short.
    ///
    /// An authentication failure keeps none of it. OpenAI answers a wrong key with "Incorrect
    /// API key provided: sk-…abcd", and part of the operator's key is not the caller's business
    /// however short the excerpt.
    fn status(status: u16, body: String) -> Self {
        tracing::warn!(status, body = %body, "host answered with an error");

        if matches!(status, 401 | 403) {
            return Self::Status {
                status,
                body: "the host refused cerno's credentials; the service log has its answer".into(),
            };
        }

        let body = match body.char_indices().nth(MAX_ERROR_BODY) {
            Some((cut, _)) => format!("{}…", &body[..cut]),
            None => body,
        };
        Self::Status { status, body }
    }

    /// Map a transport failure, keeping a timeout a timeout wherever in the exchange it struck —
    /// sending the request or reading the body.
    fn transport(error: &reqwest::Error, timeout: Duration) -> Self {
        if error.is_timeout() {
            Self::Timeout(timeout)
        } else {
            Self::Unavailable(error.to_string())
        }
    }
}

#[async_trait]
pub trait ModelHost: Send + Sync {
    fn capabilities(&self) -> HostCapabilities;

    /// A human-readable name for the host, used in logs and diagnostics.
    fn name(&self) -> &str;

    async fn first_token(
        &self,
        req: FirstTokenRequest,
    ) -> Result<FirstTokenDistribution, HostError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_host_kind_parses_from_its_own_name() {
        for kind in [
            HostKind::Ollama,
            HostKind::OpenAi,
            HostKind::Vllm,
            HostKind::LlamaCpp,
            HostKind::LmStudio,
        ] {
            assert_eq!(kind.to_string().parse::<HostKind>().unwrap(), kind);
        }
        assert_eq!(
            " LLAMA.CPP ".parse::<HostKind>().unwrap(),
            HostKind::LlamaCpp
        );
    }

    #[test]
    fn an_unknown_host_kind_names_the_choices() {
        let err = "tgi".parse::<HostKind>().unwrap_err().to_string();

        assert!(err.contains("\"tgi\""), "{err}");
        assert!(err.contains("lmstudio"), "{err}");
    }

    /// The OpenAI-compatible defaults carry the `/v1` prefix; Ollama's native API has none.
    #[test]
    fn default_urls_match_where_each_runtime_listens() {
        assert_eq!(HostKind::Ollama.default_url(), "http://localhost:11434");
        assert_eq!(HostKind::Vllm.default_url(), "http://localhost:8000/v1");
        assert_eq!(HostKind::LlamaCpp.default_url(), "http://localhost:8080/v1");
        assert_eq!(HostKind::LmStudio.default_url(), "http://localhost:1234/v1");
    }

    #[test]
    fn a_long_error_body_is_cut_short() {
        let HostError::Status { body, .. } = HostError::status(400, "x".repeat(5000)) else {
            panic!()
        };

        assert_eq!(
            body.chars().count(),
            MAX_ERROR_BODY + 1,
            "300 characters and an ellipsis"
        );
        assert!(body.ends_with('…'));
    }

    #[test]
    fn an_authentication_failure_passes_on_none_of_the_body() {
        for status in [401, 403] {
            let HostError::Status { body, .. } = HostError::status(
                status,
                r#"{"error":{"message":"Incorrect API key provided: sk-proj-********abcd."}}"#
                    .into(),
            ) else {
                panic!()
            };

            assert!(!body.contains("sk-"), "{status}: {body}");
            assert!(!body.contains("abcd"), "{status}: {body}");
        }
    }

    #[test]
    fn a_short_error_body_is_kept_whole() {
        let HostError::Status { body, .. } = HostError::status(404, "model not found".into())
        else {
            panic!()
        };

        assert_eq!(body, "model not found");
    }

    #[test]
    fn a_base_url_must_be_http_with_a_host() {
        assert_eq!(
            base_url(" http://localhost:11434/ ").unwrap(),
            "http://localhost:11434"
        );
        assert_eq!(
            base_url("https://api.openai.com/v1").unwrap(),
            "https://api.openai.com/v1"
        );

        for bad in ["localhost:11434", "ftp://x", "http://", "not a url", ""] {
            assert!(
                matches!(base_url(bad), Err(HostError::InvalidUrl { .. })),
                "{bad:?} was accepted"
            );
        }
    }

    #[test]
    fn connect_refuses_a_url_without_a_scheme() {
        let err = connect(
            HostKind::Ollama,
            "localhost:11434",
            None,
            Duration::from_secs(1),
        )
        .err()
        .expect("refused");

        assert!(err.to_string().contains("http://"), "{err}");
    }

    #[test]
    fn connect_names_the_host_it_built() {
        let timeout = Duration::from_secs(1);

        assert_eq!(
            connect(HostKind::Ollama, "http://x", None, timeout)
                .unwrap()
                .name(),
            "ollama"
        );
        assert_eq!(
            connect(HostKind::Vllm, "http://x/v1", None, timeout)
                .unwrap()
                .name(),
            "vllm"
        );
    }
}
