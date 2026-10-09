//! Ollama serves two endpoints cerno can ask, and [`OllamaRouter`] picks one per model.
//!
//! A model Ollama lists with the `decision` capability (`nimble`, `tev1`) is asked through
//! `/v1/systemone`, where Ollama writes the prompt and answers with a probability per offered
//! option. Every other model gets cerno's letter prompt over `/api/chat`. Detection is one
//! `/api/show` per model, cached for the life of the process.

use crate::{
    FirstTokenDistribution, FirstTokenRequest, HostCapabilities, HostError, ModelHost, OllamaHost,
    SystemOneHost,
};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

#[derive(Deserialize)]
struct Show {
    /// Absent on Ollama versions that predate the field; those have no decision models either.
    #[serde(default)]
    capabilities: Vec<String>,
}

pub struct OllamaRouter {
    client: reqwest::Client,
    base_url: String,
    timeout: Duration,
    chat: OllamaHost,
    systemone: SystemOneHost,
    /// Per model: whether Ollama lists it with the `decision` capability. Asked once per model.
    decision: Mutex<HashMap<String, bool>>,
}

impl OllamaRouter {
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> Result<Self, HostError> {
        let url = crate::base_url(&base_url.into())?;
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| HostError::Unavailable(e.to_string()))?;
        Ok(Self {
            client,
            chat: OllamaHost::new(url.clone(), timeout)?,
            systemone: SystemOneHost::new(url.clone(), timeout)?,
            base_url: url,
            timeout,
            decision: Mutex::default(),
        })
    }

    /// Whether `model` is a decision model.
    ///
    /// A refused `/api/show` is answered "no" and not remembered: `/api/chat` then fails the
    /// same way it always has, so an unknown model is still reported as one, and a model pulled
    /// later is recognised on its next question.
    async fn is_decision(&self, model: &str) -> Result<bool, HostError> {
        let known = self
            .decision
            .lock()
            .expect("lock is never poisoned")
            .get(model)
            .copied();
        if let Some(known) = known {
            return Ok(known);
        }

        let response = self
            .client
            .post(format!("{}/api/show", self.base_url))
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await
            .map_err(|e| HostError::transport(&e, self.timeout))?;

        let status = response.status().as_u16();
        if status >= 400 {
            tracing::debug!(
                model,
                status,
                "could not read the model's capabilities; asking /api/chat"
            );
            return Ok(false);
        }

        let text = response
            .text()
            .await
            .map_err(|e| HostError::transport(&e, self.timeout))?;
        let show: Show = serde_json::from_str(&text)
            .map_err(|e| HostError::Protocol(format!("could not read /api/show: {e}")))?;
        let decision = show.capabilities.iter().any(|c| c == "decision");

        // Concurrent first questions may each have asked; only the first to finish says so.
        let first = self
            .decision
            .lock()
            .expect("lock is never poisoned")
            .insert(model.to_string(), decision)
            .is_none();
        if decision && first {
            tracing::info!(model, "decision model; answering through /v1/systemone");
        }
        Ok(decision)
    }
}

#[async_trait]
impl ModelHost for OllamaRouter {
    /// The `/api/chat` ceiling. `/v1/systemone` takes more options, but `Engine` caps a choice at
    /// `MAX_OPTIONS` either way, so the smaller one costs nothing.
    fn capabilities(&self) -> HostCapabilities {
        self.chat.capabilities()
    }

    fn name(&self) -> &str {
        "ollama"
    }

    async fn first_token(
        &self,
        req: FirstTokenRequest,
    ) -> Result<FirstTokenDistribution, HostError> {
        if self.is_decision(&req.model).await? {
            self.systemone.first_token(req).await
        } else {
            self.chat.first_token(req).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Matcher;
    use serde_json::json;

    fn request(model: &str) -> FirstTokenRequest {
        FirstTokenRequest {
            model: model.into(),
            system: None,
            user: "Which?".into(),
            top_logprobs: 20,
            keep_alive: None,
            offer: crate::test_offer(),
        }
    }

    async fn show(
        server: &mut mockito::Server,
        model: &str,
        capabilities: &[&str],
    ) -> mockito::Mock {
        server
            .mock("POST", "/api/show")
            .match_body(Matcher::PartialJson(json!({ "model": model })))
            .with_body(json!({ "capabilities": capabilities }).to_string())
            .expect(1)
            .create_async()
            .await
    }

    #[tokio::test]
    async fn a_decision_model_is_answered_through_systemone_and_asked_once() {
        let mut server = mockito::Server::new_async().await;
        let show = show(&mut server, "nimble", &["decision"]).await;
        let systemone = server
            .mock("POST", "/v1/systemone")
            .with_body(
                json!({"answers": {"q": {"type": "choice", "probabilities": {"x": 0.8, "y": 0.2}}}})
                    .to_string(),
            )
            .expect(2)
            .create_async()
            .await;
        let chat = server
            .mock("POST", "/api/chat")
            .expect(0)
            .create_async()
            .await;
        let host = OllamaRouter::new(server.url(), Duration::from_secs(5)).unwrap();

        for _ in 0..2 {
            let dist = host.first_token(request("nimble")).await.unwrap();
            assert_eq!(dist.tokens[0].0, "A");
            assert!((dist.logprob("A").unwrap().exp() - 0.8).abs() < 1e-9);
        }

        show.assert_async().await;
        systemone.assert_async().await;
        chat.assert_async().await;
    }

    #[tokio::test]
    async fn any_other_model_gets_the_letter_prompt() {
        let mut server = mockito::Server::new_async().await;
        let show = show(&mut server, "gemma4", &["completion", "tools"]).await;
        let chat = server
            .mock("POST", "/api/chat")
            .with_body(
                json!({
                    "message": {"role": "assistant", "content": "B"},
                    "logprobs": [{"token": "B", "logprob": -0.1, "top_logprobs": [
                        {"token": "B", "logprob": -0.1},
                        {"token": "A", "logprob": -2.4}
                    ]}]
                })
                .to_string(),
            )
            .expect(1)
            .create_async()
            .await;
        let systemone = server
            .mock("POST", "/v1/systemone")
            .expect(0)
            .create_async()
            .await;
        let host = OllamaRouter::new(server.url(), Duration::from_secs(5)).unwrap();

        let dist = host.first_token(request("gemma4")).await.unwrap();

        assert_eq!(dist.tokens[0], ("B".to_string(), -0.1));
        show.assert_async().await;
        chat.assert_async().await;
        systemone.assert_async().await;
    }

    #[tokio::test]
    async fn an_unreadable_capability_list_falls_back_to_chat_and_is_asked_again() {
        let mut server = mockito::Server::new_async().await;
        let show = server
            .mock("POST", "/api/show")
            .with_status(404)
            .with_body(r#"{"error":"model 'x:latest' not found"}"#)
            .expect(2)
            .create_async()
            .await;
        let chat = server
            .mock("POST", "/api/chat")
            .with_status(404)
            .with_body(r#"{"error":"model \"x\" not found"}"#)
            .expect(2)
            .create_async()
            .await;
        let host = OllamaRouter::new(server.url(), Duration::from_secs(5)).unwrap();

        for _ in 0..2 {
            let err = host.first_token(request("x")).await.unwrap_err();
            assert!(
                matches!(err, HostError::Status { status: 404, ref body } if body.contains("not found")),
                "{err}"
            );
        }

        show.assert_async().await;
        chat.assert_async().await;
    }
}
