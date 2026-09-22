//! The seam between cerno and whatever runs the model.
//!
//! A host knows nothing about Noul, Choice or Score. It answers exactly one question: given a
//! prompt, what is the probability distribution over the *first* token the model would generate?
//! All primitive logic lives in `cerno-core` and therefore holds for every host equally.

mod ollama;

pub use ollama::OllamaHost;

use async_trait::async_trait;
use std::time::Duration;

/// What a host can and cannot do. `cerno-core` reads this to size its label alphabet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCapabilities {
    /// How many ranked tokens the host will report for one position.
    ///
    /// Ollama caps this at 20. A label beyond that rank is invisible, which is exactly the
    /// bound that limits a Choice to 20 options.
    pub max_top_logprobs: usize,
}

/// One prompt, one token, full distribution.
#[derive(Debug, Clone)]
pub struct FirstTokenRequest {
    pub model: String,
    pub system: Option<String>,
    pub user: String,
    pub top_logprobs: usize,
    /// How long the host should keep the model resident after answering.
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
