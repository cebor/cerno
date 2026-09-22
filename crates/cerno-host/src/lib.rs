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
