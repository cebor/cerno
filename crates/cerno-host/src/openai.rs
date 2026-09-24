//! Adapter for anything that speaks OpenAI's `/v1/chat/completions`: vLLM, llama.cpp's
//! `llama-server`, LM Studio, OpenAI itself.
//!
//! The response shape is the same everywhere, so there is one parser. What differs is which
//! sampling fields a runtime accepts, and that is all a [`Flavour`] carries.

use crate::{FirstTokenDistribution, FirstTokenRequest, HostCapabilities, HostError, ModelHost};
use async_trait::async_trait;
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// OpenAI rejects more than this, and it is vLLM's default `--max-logprobs`. Past 20 there are
/// no labels to observe anyway.
const OPENAI_MAX_TOP_LOGPROBS: usize = 20;

/// Which OpenAI-compatible runtime is on the other end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavour {
    /// Standard fields only. OpenAI answers 400 to a field it does not know, so nothing
    /// runtime-specific can be sent — `top_k` and `min_p` included.
    ///
    /// Measured behind Ollama's `/v1` on `gemma4:26b-a4b-it-q4_K_M`: this body returns the same
    /// distribution as the native path with its pinned options, to the third decimal.
    Generic,
    Vllm,
    LlamaCpp,
    LmStudio,
}

impl Flavour {
    fn name(self) -> &'static str {
        match self {
            Flavour::Generic => "openai",
            Flavour::Vllm => "vllm",
            Flavour::LlamaCpp => "llamacpp",
            Flavour::LmStudio => "lmstudio",
        }
    }

    /// Sampling settings on top of the standard ones, for the same reason as Ollama's
    /// `SamplingOptions::REQUIRED`: a runtime that applies its sampling transform before reporting
    /// logprobs hands back a truncated, renormalised distribution unless every filter is off.
    ///
    /// * vLLM reports raw logprobs by default, but pinning `top_k: -1` (its "disabled") costs
    ///   nothing and survives a server started with a different `--logprobs-mode`.
    /// * llama.cpp disables `top_k` with `0`, and `post_sampling_probs: false` asks for the
    ///   distribution before the sampler chain.
    fn sampling(self) -> Value {
        match self {
            Flavour::Generic => json!({}),
            Flavour::Vllm => json!({"top_k": -1, "min_p": 0.0}),
            Flavour::LlamaCpp => json!({"top_k": 0, "min_p": 0.0, "post_sampling_probs": false}),
            Flavour::LmStudio => json!({"top_k": 0, "min_p": 0.0}),
        }
    }

    /// Fields that switch the reasoning preamble off. Without them the first token is a template
    /// control token — measured on `gemma4:e2b-it-qat` behind Ollama's `/v1`: `<|channel>` at
    /// `-0.03`, the answer at `-3.5`. `reasoning_effort: "none"` is the standard switch and fixed
    /// it there; `chat_template_kwargs` is how vLLM and llama.cpp reach Qwen-style templates.
    fn no_thinking(self) -> Value {
        match self {
            Flavour::Vllm | Flavour::LlamaCpp => json!({
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": false},
            }),
            Flavour::Generic | Flavour::LmStudio => json!({"reasoning_effort": "none"}),
        }
    }
}

pub struct OpenAiCompatHost {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    flavour: Flavour,
    timeout: Duration,
    /// Models that have refused the thinking switch once. They are asked without it from then
    /// on, so the refusal costs one extra round trip per model rather than one per question.
    refused_thinking: Mutex<HashSet<String>>,
}

impl OpenAiCompatHost {
    /// `base_url` includes the version prefix, e.g. `http://localhost:8000/v1`.
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        flavour: Flavour,
        timeout: Duration,
    ) -> Result<Self, HostError> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| HostError::Unavailable(e.to_string()))?;
        Ok(Self {
            client,
            base_url: crate::base_url(&base_url.into())?,
            api_key,
            flavour,
            timeout,
            refused_thinking: Mutex::default(),
        })
    }

    fn body(&self, req: &FirstTokenRequest, think_off: bool) -> Value {
        let mut messages = Vec::with_capacity(2);
        if let Some(system) = req.system.as_deref() {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.push(json!({"role": "user", "content": req.user}));

        let mut body = json!({
            "model": req.model,
            "messages": messages,
            "stream": false,
            "logprobs": true,
            "top_logprobs": req.top_logprobs.min(OPENAI_MAX_TOP_LOGPROBS),
            "max_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
        });
        let object = body.as_object_mut().expect("body is an object");
        merge(object, self.flavour.sampling());
        if think_off {
            merge(object, self.flavour.no_thinking());
        }
        body
    }

    async fn post(&self, body: &Value) -> Result<(u16, String), HostError> {
        let mut request = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .json(body);
        if let Some(key) = self.api_key.as_deref() {
            request = request.bearer_auth(key);
        }

        let response = request
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

fn merge(into: &mut Map<String, Value>, extra: Value) {
    if let Value::Object(extra) = extra {
        into.extend(extra);
    }
}

/// Whether a failed response is the runtime refusing the thinking switch — OpenAI does for any
/// model without a reasoning mode. The request then goes out again without it.
fn rejects_thinking(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("reasoning_effort")
        || lower.contains("chat_template_kwargs")
        || lower.contains("enable_thinking")
}

#[async_trait]
impl ModelHost for OpenAiCompatHost {
    fn capabilities(&self) -> HostCapabilities {
        HostCapabilities {
            max_top_logprobs: OPENAI_MAX_TOP_LOGPROBS,
        }
    }

    fn name(&self) -> &str {
        self.flavour.name()
    }

    async fn first_token(
        &self,
        req: FirstTokenRequest,
    ) -> Result<FirstTokenDistribution, HostError> {
        let started = Instant::now();

        let think_off = !self
            .refused_thinking
            .lock()
            .expect("lock is never poisoned")
            .contains(&req.model);

        let (mut status, mut text) = self.post(&self.body(&req, think_off)).await?;

        if think_off && (400..500).contains(&status) && rejects_thinking(&text) {
            tracing::debug!(
                model = %req.model,
                host = self.flavour.name(),
                "host rejects the thinking switch; retrying without it, and from now on"
            );
            self.refused_thinking
                .lock()
                .expect("lock is never poisoned")
                .insert(req.model.clone());
            (status, text) = self.post(&self.body(&req, false)).await?;
        }

        if status >= 400 {
            return Err(HostError::status(status, text));
        }

        parse_distribution(&text, &req.model, started.elapsed())
    }
}

/// Pull the first position's ranked tokens out of a `/chat/completions` response.
fn parse_distribution(
    text: &str,
    model: &str,
    latency: Duration,
) -> Result<FirstTokenDistribution, HostError> {
    let value: Value =
        serde_json::from_str(text).map_err(|e| HostError::Protocol(e.to_string()))?;
    let no_logprobs = || HostError::NoLogprobs {
        model: model.to_string(),
    };

    let ranked = value
        .pointer("/choices/0/logprobs/content/0/top_logprobs")
        .and_then(Value::as_array)
        .ok_or_else(no_logprobs)?;

    let mut tokens: Vec<(String, f64)> = ranked
        .iter()
        .filter_map(|entry| {
            let token = entry.get("token")?.as_str()?.to_string();
            let logprob = entry.get("logprob")?.as_f64()?;
            Some((token, logprob))
        })
        .collect();

    if tokens.is_empty() {
        return Err(no_logprobs());
    }

    tokens.sort_by(|a, b| b.1.total_cmp(&a.1));

    let floor = tokens
        .last()
        .map(|(_, lp)| *lp)
        .expect("tokens is non-empty");

    let input_tokens = value
        .pointer("/usage/prompt_tokens")
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

    /// A real `/v1/chat/completions` response, trimmed of `bytes`. Captured from
    /// `gemma4:e2b-it-qat` behind Ollama's OpenAI endpoint, with `reasoning_effort: "none"`.
    const REAL_RESPONSE: &str = r#"{
        "id": "chatcmpl-147",
        "object": "chat.completion",
        "model": "gemma4:e2b-it-qat",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "B"},
            "finish_reason": "length",
            "logprobs": {"content": [{
                "token": "B",
                "logprob": -0.002,
                "top_logprobs": [
                    {"token": "B", "logprob": -0.002},
                    {"token": "b", "logprob": -7.721},
                    {"token": "**", "logprob": -7.946},
                    {"token": "", "logprob": -7.976},
                    {"token": "C", "logprob": -8.674}
                ]
            }]}
        }],
        "usage": {"prompt_tokens": 40, "completion_tokens": 1, "total_tokens": 41}
    }"#;

    fn request(top_logprobs: usize) -> FirstTokenRequest {
        FirstTokenRequest {
            model: "m".into(),
            system: Some("sys".into()),
            user: "usr".into(),
            top_logprobs,
            keep_alive: Some("5m".into()),
        }
    }

    fn body(flavour: Flavour, think_off: bool) -> Value {
        OpenAiCompatHost::new(
            "http://localhost:8000/v1",
            None,
            flavour,
            Duration::from_secs(5),
        )
        .unwrap()
        .body(&request(20), think_off)
    }

    #[test]
    fn parses_ranked_tokens_and_floor() {
        let dist = parse_distribution(REAL_RESPONSE, "m", Duration::from_millis(7)).unwrap();

        assert_eq!(dist.tokens.len(), 5);
        assert_eq!(dist.tokens[0].0, "B");
        assert_eq!(dist.input_tokens, 40);
        assert_eq!(dist.logprob("C"), Some(-8.674));
        assert_eq!(dist.logprob("A"), None);
        assert_eq!(dist.floor, -8.674);
    }

    #[test]
    fn ranks_tokens_even_when_the_host_reports_them_unsorted() {
        let shuffled = r#"{"choices":[{"logprobs":{"content":[{"token":"B","top_logprobs":[
            {"token":"A","logprob":-5.0},{"token":"B","logprob":-0.5},{"token":"C","logprob":-9.0}]}]}}]}"#;

        let dist = parse_distribution(shuffled, "m", Duration::ZERO).unwrap();

        assert_eq!(dist.tokens[0].0, "B");
        assert_eq!(dist.floor, -9.0);
    }

    /// A runtime that ignores `logprobs` answers normally with `"logprobs": null`.
    #[test]
    fn missing_logprobs_is_an_error_naming_the_model() {
        let body =
            r#"{"choices":[{"message":{"role":"assistant","content":"B"},"logprobs":null}]}"#;

        let err = parse_distribution(body, "tiny", Duration::ZERO).unwrap_err();

        assert!(matches!(err, HostError::NoLogprobs { model } if model == "tiny"));
    }

    #[test]
    fn empty_top_logprobs_is_treated_as_missing() {
        let body = r#"{"choices":[{"logprobs":{"content":[{"token":"B","top_logprobs":[]}]}}]}"#;

        assert!(matches!(
            parse_distribution(body, "m", Duration::ZERO),
            Err(HostError::NoLogprobs { .. })
        ));
    }

    #[test]
    fn the_standard_fields_are_the_same_for_every_flavour() {
        for flavour in [
            Flavour::Generic,
            Flavour::Vllm,
            Flavour::LlamaCpp,
            Flavour::LmStudio,
        ] {
            let json = body(flavour, true);

            assert_eq!(json["logprobs"], true, "{flavour:?}");
            assert_eq!(json["top_logprobs"], 20, "{flavour:?}");
            assert_eq!(json["max_tokens"], 1, "{flavour:?}");
            assert_eq!(json["temperature"], 1.0, "{flavour:?}");
            assert_eq!(json["top_p"], 1.0, "{flavour:?}");
            assert_eq!(json["stream"], false, "{flavour:?}");
            assert_eq!(json["reasoning_effort"], "none", "{flavour:?}");
            assert_eq!(json["messages"][0]["role"], "system");
            assert_eq!(json["messages"][1]["content"], "usr");
            // keep_alive is Ollama's; nobody else knows it.
            assert!(json.get("keep_alive").is_none(), "{flavour:?}");
        }
    }

    /// OpenAI answers 400 to a field it does not know, so the generic body carries none.
    #[test]
    fn the_generic_flavour_sends_nothing_nonstandard() {
        let json = body(Flavour::Generic, true);

        for field in [
            "top_k",
            "min_p",
            "post_sampling_probs",
            "chat_template_kwargs",
        ] {
            assert!(json.get(field).is_none(), "{field}");
        }
    }

    /// The sampling fields are the reason the distribution is readable; pin them verbatim.
    #[test]
    fn runtime_sampling_fields_are_serialised_verbatim() {
        let vllm = body(Flavour::Vllm, true);
        assert_eq!(vllm["top_k"], -1);
        assert_eq!(vllm["min_p"], 0.0);
        assert_eq!(vllm["chat_template_kwargs"]["enable_thinking"], false);

        let llamacpp = body(Flavour::LlamaCpp, true);
        assert_eq!(llamacpp["top_k"], 0);
        assert_eq!(llamacpp["min_p"], 0.0);
        assert_eq!(llamacpp["post_sampling_probs"], false);
        assert_eq!(llamacpp["chat_template_kwargs"]["enable_thinking"], false);

        let lmstudio = body(Flavour::LmStudio, true);
        assert_eq!(lmstudio["top_k"], 0);
        assert_eq!(lmstudio["min_p"], 0.0);
    }

    /// The retry drops the thinking switch and nothing else — least of all the sampling fields.
    #[test]
    fn the_retry_body_keeps_sampling_and_drops_only_the_thinking_switch() {
        let json = body(Flavour::Vllm, false);

        assert!(json.get("reasoning_effort").is_none());
        assert!(json.get("chat_template_kwargs").is_none());
        assert_eq!(json["top_k"], -1);
    }

    #[test]
    fn top_logprobs_is_clamped_to_the_host_ceiling() {
        let host = OpenAiCompatHost::new(
            "http://x/v1",
            None,
            Flavour::Generic,
            Duration::from_secs(5),
        )
        .unwrap();

        assert_eq!(host.body(&request(99), true)["top_logprobs"], 20);
    }

    #[test]
    fn recognises_the_thinking_rejection() {
        assert!(rejects_thinking(
            r#"{"error":{"message":"Unsupported parameter: 'reasoning_effort' is not supported with this model.","param":"reasoning_effort"}}"#
        ));
        assert!(!rejects_thinking(
            r#"{"error":{"message":"model not found"}}"#
        ));
    }

    #[tokio::test]
    async fn the_api_key_is_sent_as_a_bearer_token() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_header("authorization", "Bearer sk-test")
            .with_body(REAL_RESPONSE)
            .create_async()
            .await;
        let host = OpenAiCompatHost::new(
            format!("{}/v1/", server.url()),
            Some("sk-test".into()),
            Flavour::Generic,
            Duration::from_secs(5),
        )
        .unwrap();

        host.first_token(request(4)).await.unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn without_a_key_no_authorization_header_is_sent() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_header("authorization", mockito::Matcher::Missing)
            .with_body(REAL_RESPONSE)
            .create_async()
            .await;
        let host = OpenAiCompatHost::new(
            format!("{}/v1", server.url()),
            None,
            Flavour::LlamaCpp,
            Duration::from_secs(5),
        )
        .unwrap();

        host.first_token(request(4)).await.unwrap();

        mock.assert_async().await;
    }

    /// A model without a reasoning mode refuses the switch; the same question goes out again
    /// without it and is answered, and later questions to that model skip the refusal.
    #[tokio::test]
    async fn a_rejected_thinking_switch_is_retried_without_it_and_remembered() {
        let mut server = mockito::Server::new_async().await;
        let refused = server
            .mock("POST", "/v1/chat/completions")
            .match_body(mockito::Matcher::PartialJson(
                json!({"reasoning_effort": "none"}),
            ))
            .with_status(400)
            .with_body(r#"{"error":{"message":"Unsupported parameter: 'reasoning_effort'"}}"#)
            .expect(1)
            .create_async()
            .await;
        let answered = server
            .mock("POST", "/v1/chat/completions")
            // Only a body without the switch is answered. Without this guard mockito would hand a
            // repeated switch to this mock once the refusal had been used up, and the test
            // would pass whether or not the refusal was remembered.
            .match_request(|request| {
                !request
                    .utf8_lossy_body()
                    .is_ok_and(|body| body.contains(r#""reasoning_effort""#))
            })
            .with_body(REAL_RESPONSE)
            .expect(2)
            .create_async()
            .await;
        let host = OpenAiCompatHost::new(
            format!("{}/v1", server.url()),
            None,
            Flavour::Generic,
            Duration::from_secs(5),
        )
        .unwrap();

        let dist = host.first_token(request(4)).await.unwrap();
        assert_eq!(dist.tokens[0].0, "B");

        // The second question to the same model must not pay for the refusal again.
        host.first_token(request(4)).await.unwrap();

        refused.assert_async().await;
        answered.assert_async().await;
    }
}
