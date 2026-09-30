//! Adapter for Ollama's `POST /v1/systemone`, which its decision models (`nimble`, `tev1`) were
//! built for.
//!
//! Ollama writes the prompt itself and answers with a probability per offered option, so this
//! host reads the [`Offer`] and ignores the rendered prompt. It hands the answer back as a
//! distribution whose tokens are exactly the offered labels at `ln p`, which lets `cerno-core`
//! fold, calibrate and normalise it on the same path as every other host.

use crate::{
    FirstTokenDistribution, FirstTokenRequest, HostCapabilities, HostError, ModelHost, Offer, Shape,
};
use async_trait::async_trait;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// The endpoint takes 2–26 options per choice and returns a probability for every one, so no
/// option is ever outside a window. `Engine::max_options` still caps a choice at `MAX_OPTIONS`.
const SYSTEMONE_MAX_OPTIONS: usize = 26;

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    state: &'a str,
    /// Always exactly one entry, keyed by the question id: the engine asks one question per
    /// host call.
    questions: BTreeMap<&'a str, Ask<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<&'a str>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Ask<'a> {
    Noul {
        instructions: &'a str,
    },
    Choice {
        instructions: &'a str,
        criteria: NullCriteria<'a>,
    },
    Score {
        instructions: &'a str,
        /// Level descriptions, lowest first. The endpoint numbers them from 0.
        criteria: Vec<&'a str>,
    },
}

/// Options as a JSON object in the order given, each mapped to `null` so the name describes
/// itself. Hand-written because a map type would alphabetise the options and change what the
/// model sees, and serde_json's `preserve_order` would reorder JSON across the whole workspace.
struct NullCriteria<'a>(&'a [(String, String)]);

impl Serialize for NullCriteria<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (_, text) in self.0 {
            map.serialize_entry(text, &())?;
        }
        map.end()
    }
}

/// The request for one question. The endpoint refuses empty instructions, so a choice or score
/// asked without a question gets a neutral one.
fn body(req: &FirstTokenRequest) -> Request<'_> {
    let offer = &req.offer;
    let instructions = |default| offer.question.as_deref().unwrap_or(default);
    let ask = match offer.shape {
        Shape::YesNo => Ask::Noul {
            instructions: instructions("Is this true?"),
        },
        Shape::Pick => Ask::Choice {
            instructions: instructions("Which option applies?"),
            criteria: NullCriteria(&offer.options),
        },
        Shape::Scale => Ask::Score {
            instructions: instructions("Which level applies?"),
            criteria: offer.options.iter().map(|(_, t)| t.as_str()).collect(),
        },
    };
    Request {
        model: &req.model,
        state: &offer.state,
        questions: BTreeMap::from([(offer.question_id.as_str(), ask)]),
        keep_alive: req.keep_alive.as_deref(),
    }
}

#[derive(Deserialize)]
struct Response {
    answers: BTreeMap<String, Reply>,
    #[serde(default)]
    usage: Usage,
}

#[derive(Default, Deserialize)]
struct Usage {
    #[serde(default)]
    input_tokens: u32,
}

/// One answer. `choice`, `score`, `legend` and `confidence` are not read: cerno picks by argmax
/// and computes its own confidence, and a score's `score` is a probability-weighted level, not
/// the pick.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Reply {
    Noul {
        noul: f64,
    },
    Choice {
        probabilities: BTreeMap<String, f64>,
    },
    Score {
        probabilities: BTreeMap<String, f64>,
    },
}

pub struct SystemOneHost {
    client: reqwest::Client,
    base_url: String,
    timeout: Duration,
}

impl SystemOneHost {
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> Result<Self, HostError> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| HostError::Unavailable(e.to_string()))?;
        Ok(Self {
            client,
            base_url: crate::base_url(&base_url.into())?,
            timeout,
        })
    }
}

#[async_trait]
impl ModelHost for SystemOneHost {
    fn capabilities(&self) -> HostCapabilities {
        HostCapabilities {
            max_top_logprobs: SYSTEMONE_MAX_OPTIONS,
        }
    }

    fn name(&self) -> &str {
        "systemone"
    }

    async fn first_token(
        &self,
        req: FirstTokenRequest,
    ) -> Result<FirstTokenDistribution, HostError> {
        let started = Instant::now();

        let response = self
            .client
            .post(format!("{}/v1/systemone", self.base_url))
            .json(&body(&req))
            .send()
            .await
            .map_err(|e| HostError::transport(&e, self.timeout))?;

        let status = response.status().as_u16();
        let text = response
            .text()
            .await
            .map_err(|e| HostError::transport(&e, self.timeout))?;

        if status >= 400 {
            return Err(HostError::status(status, text));
        }

        parse_answer(&text, &req.offer, started.elapsed())
    }
}

/// `ln p`, clamped so a reported 0 stays finite: `raw_logprobs` is serialised to JSON, which has
/// no `-inf`. A softmax of these at T=1 gives back `p`.
fn ln(p: f64) -> f64 {
    p.max(1e-12).ln()
}

fn wire_type(shape: Shape) -> &'static str {
    match shape {
        Shape::YesNo => "noul",
        Shape::Pick => "choice",
        Shape::Scale => "score",
    }
}

/// Read the answer back into the offered labels, one token per label.
fn parse_answer(
    text: &str,
    offer: &Offer,
    latency: Duration,
) -> Result<FirstTokenDistribution, HostError> {
    let mut response: Response =
        serde_json::from_str(text).map_err(|e| HostError::Protocol(e.to_string()))?;
    let reply = response.answers.remove(&offer.question_id).ok_or_else(|| {
        HostError::Protocol(format!("no answer for question {:?}", offer.question_id))
    })?;

    let mut tokens: Vec<(String, f64)> = match (offer.shape, reply) {
        (Shape::YesNo, Reply::Noul { noul }) => vec![
            (offer.options[0].0.clone(), ln(noul)),
            (offer.options[1].0.clone(), ln(1.0 - noul)),
        ],
        (Shape::Pick, Reply::Choice { probabilities }) => offer
            .options
            .iter()
            .map(|(label, text)| {
                probabilities
                    .get(text)
                    .map(|&p| (label.clone(), ln(p)))
                    .ok_or_else(|| {
                        HostError::Protocol(format!("no probability for option {text:?}"))
                    })
            })
            .collect::<Result<_, _>>()?,
        (Shape::Scale, Reply::Score { probabilities }) => offer
            .options
            .iter()
            .enumerate()
            .map(|(k, (label, _))| {
                probabilities
                    .get(&k.to_string())
                    .map(|&p| (label.clone(), ln(p)))
                    .ok_or_else(|| HostError::Protocol(format!("no probability for level {k}")))
            })
            .collect::<Result<_, _>>()?,
        (shape, reply) => {
            let got = match reply {
                Reply::Noul { .. } => "noul",
                Reply::Choice { .. } => "choice",
                Reply::Score { .. } => "score",
            };
            return Err(HostError::Protocol(format!(
                "asked a {} question, got a {got} answer",
                wire_type(shape)
            )));
        }
    };

    tokens.sort_by(|a, b| b.1.total_cmp(&a.1));
    let floor = tokens.last().map(|(_, lp)| *lp).unwrap_or(ln(0.0));

    Ok(FirstTokenDistribution {
        tokens,
        floor,
        input_tokens: response.usage.input_tokens,
        latency,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn offer(shape: Shape, question: Option<&str>, texts: &[&str]) -> Offer {
        Offer {
            question_id: "q".into(),
            state: "s".into(),
            question: question.map(str::to_string),
            shape,
            options: texts
                .iter()
                .enumerate()
                .map(|(i, t)| (((b'A' + i as u8) as char).to_string(), t.to_string()))
                .collect(),
        }
    }

    fn request(offer: Offer, keep_alive: Option<&str>) -> FirstTokenRequest {
        FirstTokenRequest {
            model: "nimble".into(),
            system: Some("ignored".into()),
            user: "ignored".into(),
            top_logprobs: 20,
            keep_alive: keep_alive.map(str::to_string),
            offer,
        }
    }

    fn parse(
        offer: &Offer,
        response: serde_json::Value,
    ) -> Result<FirstTokenDistribution, HostError> {
        parse_answer(&response.to_string(), offer, Duration::ZERO)
    }

    #[test]
    fn a_choice_keeps_the_option_order() {
        let req = request(
            offer(
                Shape::Pick,
                Some("Wer?"),
                &["IT", "Personal", "Vertrieb", "Buchhaltung"],
            ),
            Some("5m"),
        );

        let body = serde_json::to_string(&body(&req)).unwrap();

        assert!(
            body.contains(
                r#""criteria":{"IT":null,"Personal":null,"Vertrieb":null,"Buchhaltung":null}"#
            ),
            "{body}"
        );
    }

    #[test]
    fn each_shape_is_sent_as_its_primitive() {
        let noul = request(
            offer(Shape::YesNo, Some("Urgent?"), &["Yes", "No"]),
            Some("5m"),
        );
        let json = serde_json::to_value(body(&noul)).unwrap();
        assert_eq!(
            json,
            json!({
                "model": "nimble",
                "state": "s",
                "questions": {"q": {"type": "noul", "instructions": "Urgent?"}},
                "keep_alive": "5m"
            })
        );

        let scale = request(
            offer(Shape::Scale, Some("How bad?"), &["1", "2", "3", "4", "5"]),
            None,
        );
        let json = serde_json::to_value(body(&scale)).unwrap();
        assert_eq!(
            json["questions"]["q"],
            json!({"type": "score", "instructions": "How bad?", "criteria": ["1", "2", "3", "4", "5"]})
        );
        assert!(json.get("keep_alive").is_none(), "{json}");

        let unasked = request(offer(Shape::Pick, None, &["x", "y"]), None);
        let json = serde_json::to_value(body(&unasked)).unwrap();
        assert_eq!(
            json["questions"]["q"]["instructions"],
            "Which option applies?"
        );
    }

    #[test]
    fn a_score_is_read_from_zero_based_levels() {
        let o = offer(Shape::Scale, None, &["1", "2", "3", "4", "5"]);
        let probabilities = [0.1, 0.2, 0.6, 0.05, 0.05];

        let dist = parse(
            &o,
            json!({
                "answers": {"q": {
                    "type": "score",
                    "score": 1.75,
                    "legend": {"0": "1", "1": "2", "2": "3", "3": "4", "4": "5"},
                    "probabilities": {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.05, "4": 0.05},
                    "confidence": 0.3
                }},
                "usage": {"input_tokens": 145, "output_tokens": 1}
            }),
        )
        .unwrap();

        assert_eq!(dist.tokens[0].0, "C");
        for (label, p) in ["A", "B", "C", "D", "E"].iter().zip(probabilities) {
            assert!(
                (dist.logprob(label).unwrap().exp() - p).abs() < 1e-9,
                "{label}"
            );
        }
        assert_eq!(dist.input_tokens, 145);
    }

    #[test]
    fn a_reported_zero_stays_finite() {
        let o = offer(Shape::YesNo, Some("?"), &["Yes", "No"]);

        let dist = parse(&o, json!({"answers": {"q": {"type": "noul", "noul": 1.0}}})).unwrap();

        assert_eq!(dist.tokens[0], ("A".to_string(), 0.0));
        assert_eq!(dist.logprob("B"), Some(1e-12_f64.ln()));
        assert!(dist.floor.is_finite());
    }

    #[test]
    fn an_answer_outside_the_offer_is_a_protocol_error() {
        let pick = offer(Shape::Pick, None, &["IT", "Personal", "Vertrieb"]);
        let scale = offer(Shape::Scale, None, &["low", "mid", "high"]);

        let cases = [
            (
                &pick,
                json!({"answers": {"q": {"type": "choice", "choice": "IT",
                    "probabilities": {"IT": 0.4, "Personal": 0.6}}}}),
                "Vertrieb",
            ),
            (
                &scale,
                json!({"answers": {"q": {"type": "score",
                    "probabilities": {"0": 0.4, "1": 0.6}}}}),
                "level 2",
            ),
            (
                &pick,
                json!({"answers": {"q": {"type": "noul", "noul": 0.9}}}),
                "asked a choice question, got a noul answer",
            ),
            (
                &pick,
                json!({"answers": {"other": {"type": "noul", "noul": 0.9}}}),
                "no answer for question \"q\"",
            ),
        ];

        for (offer, response, says) in cases {
            match parse(offer, response) {
                Err(HostError::Protocol(message)) => {
                    assert!(message.contains(says), "{message}")
                }
                other => panic!("{says}: {:?}", other.map(|d| d.tokens)),
            }
        }
    }

    #[tokio::test]
    async fn an_unknown_model_is_a_404() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/systemone")
            .with_status(404)
            .with_body(r#"{"error":"model \"does-not-exist\" not found, try pulling it first"}"#)
            .expect(1)
            .create_async()
            .await;
        let host = SystemOneHost::new(server.url(), Duration::from_secs(5)).unwrap();

        let err = host
            .first_token(request(
                offer(Shape::YesNo, Some("?"), &["Yes", "No"]),
                None,
            ))
            .await
            .unwrap_err();

        assert!(
            matches!(err, HostError::Status { status: 404, ref body } if body.contains("not found")),
            "{err}"
        );
        mock.assert_async().await;
    }
}
