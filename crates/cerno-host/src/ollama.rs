//! Ollama adapter for [`ModelHost`].

use crate::{FirstTokenDistribution, FirstTokenRequest, HostCapabilities, HostError, ModelHost};
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Ollama reports at most this many ranked tokens per position; the server rejects more with
/// "top_logprobs must be between 0 and 20".
const OLLAMA_MAX_TOP_LOGPROBS: usize = 20;

/// Sampling settings that make the reported distribution usable.
///
/// These are correctness conditions, not tuning knobs, and are deliberately not configurable:
///
/// * `num_predict: 1` — we want one token, the answer label. Nothing is generated after it.
/// * `top_k: 0`, `top_p: 1`, `min_p: 0` — **the important one.** Ollama applies the sampling
///   transform *before* reporting logprobs, so the default `top_k: 40` truncates the
///   distribution and renormalises what survives. Measured against `gemma4:26b-a4b-it-q4_K_M`,
///   that collapsed a four-option question to `A=0.0` with every rival at `-17`, and the
///   remaining labels vanished from the list entirely. Disabling every filter restores the
///   model's actual distribution.
/// * `temperature: 1` — the identity for the same reason; calibration happens later, in
///   `cerno-core`, where it is explicit and reversible.
#[derive(Debug, Clone, Copy, Serialize)]
struct SamplingOptions {
    temperature: f64,
    top_k: u32,
    top_p: f64,
    min_p: f64,
    num_predict: u32,
}

impl SamplingOptions {
    const REQUIRED: Self = Self {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        num_predict: 1,
    };
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
    stream: bool,
    /// Suppresses the reasoning preamble. Without it the first generated token is a template
    /// control token such as `<|channel|>` rather than the answer label.
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    logprobs: bool,
    top_logprobs: usize,
    options: SamplingOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<&'a str>,
}

#[derive(Debug, Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

pub struct OllamaHost {
    client: reqwest::Client,
    base_url: String,
    timeout: Duration,
    /// Models that have refused `think` once. They are asked without it from then on, so the
    /// refusal costs one extra round trip per model rather than one per question.
    refused_thinking: Mutex<HashSet<String>>,
}

impl OllamaHost {
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> Result<Self, HostError> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| HostError::Unavailable(e.to_string()))?;
        Ok(Self {
            client,
            base_url: crate::base_url(&base_url.into())?,
            timeout,
            refused_thinking: Mutex::default(),
        })
    }

    fn body<'a>(&self, req: &'a FirstTokenRequest, think: Option<bool>) -> ChatRequest<'a> {
        let mut messages = Vec::with_capacity(2);
        if let Some(system) = req.system.as_deref() {
            messages.push(Message {
                role: "system",
                content: system,
            });
        }
        messages.push(Message {
            role: "user",
            content: &req.user,
        });

        ChatRequest {
            model: &req.model,
            messages,
            stream: false,
            think,
            logprobs: true,
            top_logprobs: req.top_logprobs.min(OLLAMA_MAX_TOP_LOGPROBS),
            options: SamplingOptions::REQUIRED,
            keep_alive: req.keep_alive.as_deref(),
        }
    }

    async fn post(&self, body: &ChatRequest<'_>) -> Result<(u16, String), HostError> {
        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(body)
            .send()
            .await
            .map_err(|e| HostError::transport(&e, self.timeout))?;

        let status = response.status().as_u16();
        let text = response
            .text()
            .await
            .map_err(|e| HostError::transport(&e, self.timeout))?;
        Ok((status, text))
    }
}

/// Whether a failed response is Ollama complaining that the model has no thinking mode.
///
/// Models without a reasoning preamble reject `think` outright, so the same request has to go
/// out again without the field. Only the message distinguishes this from a real failure, so it
/// has to say both that something is unsupported and that it is thinking: a 404 for a model
/// named `deepthinker` "does not exist", and a looser match would remember that model as unable
/// to think and stop sending `think: false` to it for good.
fn rejects_thinking(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("not support") && lower.contains("thinking")
}

#[async_trait]
impl ModelHost for OllamaHost {
    fn capabilities(&self) -> HostCapabilities {
        HostCapabilities {
            max_top_logprobs: OLLAMA_MAX_TOP_LOGPROBS,
        }
    }

    fn name(&self) -> &str {
        "ollama"
    }

    async fn first_token(
        &self,
        req: FirstTokenRequest,
    ) -> Result<FirstTokenDistribution, HostError> {
        let started = Instant::now();

        let known_to_reject = self
            .refused_thinking
            .lock()
            .expect("lock is never poisoned")
            .contains(&req.model);
        let think = if known_to_reject { None } else { Some(false) };

        let (mut status, mut text) = self.post(&self.body(&req, think)).await?;

        if think.is_some() && status >= 400 && rejects_thinking(&text) {
            tracing::debug!(
                model = %req.model,
                "model rejects the think flag; retrying without it, and from now on"
            );
            self.refused_thinking
                .lock()
                .expect("lock is never poisoned")
                .insert(req.model.clone());
            (status, text) = self.post(&self.body(&req, None)).await?;
        }

        if status >= 400 {
            return Err(HostError::status(status, text));
        }

        parse_distribution(&text, &req.model, started.elapsed())
    }
}

/// Pull the first position's ranked tokens out of an `/api/chat` response.
fn parse_distribution(
    text: &str,
    model: &str,
    latency: Duration,
) -> Result<FirstTokenDistribution, HostError> {
    let value: Value =
        serde_json::from_str(text).map_err(|e| HostError::Protocol(e.to_string()))?;

    let first = value
        .get("logprobs")
        .and_then(Value::as_array)
        .and_then(|positions| positions.first())
        .ok_or_else(|| HostError::NoLogprobs {
            model: model.to_string(),
        })?;

    // `top_logprobs` holds the ranked alternatives and always includes the sampled token itself.
    let ranked = first
        .get("top_logprobs")
        .and_then(Value::as_array)
        .ok_or_else(|| HostError::NoLogprobs {
            model: model.to_string(),
        })?;

    let mut tokens: Vec<(String, f64)> = ranked
        .iter()
        .filter_map(|entry| {
            let token = entry.get("token")?.as_str()?.to_string();
            let logprob = entry.get("logprob")?.as_f64()?;
            Some((token, logprob))
        })
        .collect();

    if tokens.is_empty() {
        return Err(HostError::NoLogprobs {
            model: model.to_string(),
        });
    }

    tokens.sort_by(|a, b| b.1.total_cmp(&a.1));

    let floor = tokens
        .last()
        .map(|(_, lp)| *lp)
        .expect("tokens is non-empty");

    let input_tokens = value
        .get("prompt_eval_count")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;

    Ok(FirstTokenDistribution {
        tokens,
        floor,
        input_tokens,
        latency,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real `/api/chat` response, trimmed. Captured from `gemma4:26b-a4b-it-q4_K_M`.
    const REAL_RESPONSE: &str = r#"{
        "model": "gemma4:26b-a4b-it-q4_K_M",
        "message": {"role": "assistant", "content": "D"},
        "done": true,
        "prompt_eval_count": 110,
        "logprobs": [{
            "token": "D",
            "logprob": -0.005,
            "top_logprobs": [
                {"token": "D", "logprob": -0.005},
                {"token": "A", "logprob": -5.246},
                {"token": "C", "logprob": -11.515},
                {"token": "B", "logprob": -13.662}
            ]
        }]
    }"#;

    #[test]
    fn parses_ranked_tokens_and_floor() {
        let dist = parse_distribution(REAL_RESPONSE, "m", Duration::from_millis(7)).unwrap();

        assert_eq!(dist.tokens.len(), 4);
        assert_eq!(dist.tokens[0].0, "D");
        assert_eq!(dist.input_tokens, 110);
        assert_eq!(dist.logprob("A"), Some(-5.246));
        assert_eq!(dist.logprob("Z"), None);
        // The floor is the weakest reported entry, the bound for anything not listed.
        assert_eq!(dist.floor, -13.662);
    }

    #[test]
    fn ranks_tokens_even_when_the_host_reports_them_unsorted() {
        let shuffled = r#"{"logprobs":[{"token":"B","top_logprobs":[
            {"token":"A","logprob":-5.0},{"token":"B","logprob":-0.5},{"token":"C","logprob":-9.0}]}]}"#;

        let dist = parse_distribution(shuffled, "m", Duration::ZERO).unwrap();

        assert_eq!(dist.tokens[0].0, "B");
        assert_eq!(dist.floor, -9.0);
    }

    /// A response with no logprobs is unusable, and saying so plainly beats a silent zero.
    #[test]
    fn missing_logprobs_is_an_error_naming_the_model() {
        let body = r#"{"model":"m","message":{"role":"assistant","content":"D"},"done":true}"#;

        let err = parse_distribution(body, "tiny", Duration::ZERO).unwrap_err();

        assert!(matches!(err, HostError::NoLogprobs { model } if model == "tiny"));
    }

    #[test]
    fn empty_top_logprobs_is_treated_as_missing() {
        let body = r#"{"logprobs":[{"token":"D","top_logprobs":[]}]}"#;

        assert!(matches!(
            parse_distribution(body, "m", Duration::ZERO),
            Err(HostError::NoLogprobs { .. })
        ));
    }

    #[test]
    fn recognises_the_thinking_rejection() {
        assert!(rejects_thinking(
            r#"{"error":"registry.ollama.ai/library/x does not support thinking"}"#
        ));
        assert!(!rejects_thinking(r#"{"error":"model not found"}"#));
        assert!(!rejects_thinking(
            r#"{"error":"model \"deepthinker\" does not exist"}"#
        ));
    }

    /// The sampling options are the whole reason the distribution is readable; pin them so a
    /// well-meaning edit cannot quietly reintroduce the default `top_k`.
    #[test]
    fn required_sampling_options_are_serialised_verbatim() {
        let json = serde_json::to_value(SamplingOptions::REQUIRED).unwrap();

        assert_eq!(json["top_k"], 0);
        assert_eq!(json["top_p"], 1.0);
        assert_eq!(json["min_p"], 0.0);
        assert_eq!(json["temperature"], 1.0);
        assert_eq!(json["num_predict"], 1);
    }

    #[test]
    fn request_body_disables_thinking_and_asks_for_logprobs() {
        let host = OllamaHost::new("http://localhost:11434", Duration::from_secs(5)).unwrap();
        let req = FirstTokenRequest {
            model: "m".into(),
            system: Some("sys".into()),
            user: "usr".into(),
            top_logprobs: 20,
            keep_alive: Some("5m".into()),
        };

        let json = serde_json::to_value(host.body(&req, Some(false))).unwrap();

        assert_eq!(json["think"], false);
        assert_eq!(json["logprobs"], true);
        assert_eq!(json["top_logprobs"], 20);
        assert_eq!(json["stream"], false);
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][1]["content"], "usr");
    }

    /// Asking for more than Ollama allows would be rejected outright, so clamp instead.
    #[test]
    fn top_logprobs_is_clamped_to_the_host_ceiling() {
        let host = OllamaHost::new("http://localhost:11434", Duration::from_secs(5)).unwrap();
        let req = FirstTokenRequest {
            model: "m".into(),
            system: None,
            user: "usr".into(),
            top_logprobs: 99,
            keep_alive: None,
        };

        let json = serde_json::to_value(host.body(&req, Some(false))).unwrap();

        assert_eq!(json["top_logprobs"], 20);
        assert!(json.get("keep_alive").is_none());
    }

    /// A model without a thinking mode refuses `think`. The first question pays for one retry;
    /// every later question to that model goes out without the field straight away.
    #[tokio::test]
    async fn a_refused_think_flag_is_remembered_per_model() {
        let mut server = mockito::Server::new_async().await;
        let refused = server
            .mock("POST", "/api/chat")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"think": false}),
            ))
            .with_status(400)
            .with_body(r#"{"error":"registry.ollama.ai/library/x does not support thinking"}"#)
            .expect(1)
            .create_async()
            .await;
        let answered = server
            .mock("POST", "/api/chat")
            // Only a body without the switch is answered. Without this guard mockito would hand a
            // repeated switch to this mock once the refusal had been used up, and the test
            // would pass whether or not the refusal was remembered.
            .match_request(|request| {
                !request
                    .utf8_lossy_body()
                    .is_ok_and(|body| body.contains(r#""think""#))
            })
            .with_body(REAL_RESPONSE)
            .expect(2)
            .create_async()
            .await;
        let host = OllamaHost::new(server.url(), Duration::from_secs(5)).unwrap();
        let req = FirstTokenRequest {
            model: "m".into(),
            system: None,
            user: "usr".into(),
            top_logprobs: 20,
            keep_alive: None,
        };

        host.first_token(req.clone()).await.unwrap();
        host.first_token(req).await.unwrap();

        refused.assert_async().await;
        answered.assert_async().await;
    }
}
