//! Measures candidate models against a labelled dataset and writes the comparison table.
//!
//! Four things decide a model here, and the first one is a gate rather than a score:
//!
//! 1. **Label fidelity** — did the model answer with one of the letters it was offered? A model
//!    that writes prose instead is unusable for cerno at any accuracy.
//! 2. **Accuracy** — did it pick the right letter.
//! 3. **Latency** — the whole point of one forward pass.
//! 4. **Calibration** — how far its confidence has to be flattened to stop lying.
//!
//! Usage:
//!   cerno-bench --models a,b,c [--reference m] [--dataset p] [--out p] [--host url]

use cerno_core::{Engine, labels};
use cerno_host::{ModelHost, OllamaHost};
use cerno_types::{Answer, Calibration, ChoiceSpec, Question, QuestionKind, ScoreSpec};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Deserialize)]
struct Dataset {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    state: String,
    #[serde(default)]
    noul: Option<String>,
    #[serde(default)]
    expect_yes: Option<bool>,
    #[serde(default)]
    choice: Option<ChoiceSpec>,
    #[serde(default)]
    score: Option<ScoreSpec>,
    /// Correct option index for a choice, or correct 1-based level for a score.
    #[serde(default)]
    expect: Option<usize>,
    #[serde(default)]
    tolerance: Option<usize>,
}

impl Case {
    fn primitive(&self) -> &'static str {
        if self.noul.is_some() {
            "noul"
        } else if self.choice.is_some() {
            "choice"
        } else {
            "score"
        }
    }

    fn question(&self) -> Question {
        let kind = if let Some(q) = &self.noul {
            QuestionKind::Noul(q.clone())
        } else if let Some(c) = &self.choice {
            QuestionKind::Choice(c.clone())
        } else {
            QuestionKind::Score(self.score.clone().expect("case has no primitive"))
        };
        Question {
            id: self.id.clone(),
            kind,
        }
    }

    /// The label the model should have produced.
    fn correct_label(&self) -> String {
        let index = if self.noul.is_some() {
            // "Yes" is label A; see `cerno_core::engine::ballot_for`.
            if self.expect_yes.expect("noul case needs expect_yes") {
                0
            } else {
                1
            }
        } else if self.choice.is_some() {
            self.expect.expect("choice case needs expect")
        } else {
            self.expect.expect("score case needs expect") - 1
        };
        labels::label(index)
            .expect("expectation within the alphabet")
            .to_string()
    }

    /// How many levels a score answer may be off and still count as correct.
    fn tolerance(&self) -> usize {
        self.tolerance.unwrap_or(0)
    }
}

/// One model's result on one case.
struct Outcome {
    primitive: &'static str,
    /// `None` when the model produced no usable label at all.
    answer: Option<Answer>,
    correct: bool,
    /// Logprobs per label, for refitting the calibration temperature afterwards.
    logprobs: BTreeMap<String, f64>,
    correct_label: String,
    latency: Duration,
    truncated: bool,
    /// The label the model actually picked, for agreement against the reference.
    picked: Option<String>,
}

fn softmax_of(logprobs: &BTreeMap<String, f64>, temperature: f64) -> BTreeMap<String, f64> {
    let keys: Vec<&String> = logprobs.keys().collect();
    let values: Vec<f64> = keys.iter().map(|k| logprobs[*k]).collect();
    let probabilities = cerno_core::math::softmax(&values, temperature);
    keys.into_iter().cloned().zip(probabilities).collect()
}

async fn run_case(engine: &Engine, model: &str, case: &Case) -> Outcome {
    let question = case.question();
    let correct_label = case.correct_label();
    let started = Instant::now();

    let result = engine
        .answer(&case.state, &question, model, Calibration::default())
        .await;
    let latency = started.elapsed();

    match result {
        Ok((answer, _)) => {
            let logprobs = match &answer {
                Answer::Noul { raw_logprobs, .. }
                | Answer::Choice { raw_logprobs, .. }
                | Answer::Score { raw_logprobs, .. } => raw_logprobs.clone(),
            };

            let (picked_index, correct) = match &answer {
                Answer::Noul { noul, .. } => {
                    let yes = *noul >= 0.5;
                    (if yes { 0 } else { 1 }, yes == case.expect_yes.unwrap())
                }
                Answer::Choice { index, .. } => (*index, *index == case.expect.unwrap()),
                Answer::Score { score, .. } => {
                    let got = *score as usize;
                    let want = case.expect.unwrap();
                    (got - 1, got.abs_diff(want) <= case.tolerance())
                }
            };

            Outcome {
                primitive: case.primitive(),
                truncated: answer.truncated(),
                answer: Some(answer),
                correct,
                logprobs,
                correct_label,
                latency,
                picked: labels::label(picked_index).map(str::to_string),
            }
        }
        Err(err) => {
            eprintln!("  {} {}: {}", case.id, case.primitive(), err);
            Outcome {
                primitive: case.primitive(),
                answer: None,
                correct: false,
                logprobs: BTreeMap::new(),
                correct_label,
                latency,
                truncated: false,
                picked: None,
            }
        }
    }
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

/// Grid-search the temperature that best explains the labelled outcomes.
///
/// Minimising negative log-likelihood of the correct label is the standard temperature-scaling
/// fit. Cases where no label came back are skipped — they carry no evidence about calibration,
/// only about fidelity, which is counted separately.
fn fit_temperature(outcomes: &[Outcome]) -> (f64, f64) {
    let usable: Vec<&Outcome> = outcomes.iter().filter(|o| !o.logprobs.is_empty()).collect();
    if usable.is_empty() {
        return (1.0, f64::NAN);
    }

    let mut best = (1.0, f64::INFINITY);
    let mut t = 0.25;
    while t <= 8.0 {
        let nll: f64 = usable
            .iter()
            .map(|o| {
                let p = softmax_of(&o.logprobs, t)
                    .get(&o.correct_label)
                    .copied()
                    .unwrap_or(1e-12);
                -p.max(1e-12).ln()
            })
            .sum::<f64>()
            / usable.len() as f64;

        if nll < best.1 {
            best = (t, nll);
        }
        t += 0.05;
    }
    ((best.0 * 100.0).round() / 100.0, best.1)
}

/// Mean squared error between the predicted yes-probability and the truth. Noul cases only,
/// since Brier is defined on a binary outcome.
fn brier(outcomes: &[Outcome], cases: &[Case]) -> f64 {
    let mut total = 0.0;
    let mut n = 0;
    for (outcome, case) in outcomes.iter().zip(cases) {
        if let Some(Answer::Noul { noul, .. }) = &outcome.answer {
            let truth = if case.expect_yes.unwrap() { 1.0 } else { 0.0 };
            total += (noul - truth).powi(2);
            n += 1;
        }
    }
    if n == 0 { f64::NAN } else { total / n as f64 }
}

struct Report {
    model: String,
    fidelity: f64,
    accuracy: f64,
    per_primitive: BTreeMap<&'static str, f64>,
    p50: u128,
    p99: u128,
    truncation: f64,
    brier: f64,
    best_t: f64,
    agreement: Option<f64>,
    picked: Vec<Option<String>>,
}

fn summarise(model: &str, outcomes: &[Outcome], cases: &[Case]) -> Report {
    let n = outcomes.len() as f64;
    let answered: Vec<&Outcome> = outcomes.iter().filter(|o| o.answer.is_some()).collect();

    let mut latencies: Vec<u128> = outcomes.iter().map(|o| o.latency.as_millis()).collect();
    latencies.sort_unstable();

    let mut per_primitive = BTreeMap::new();
    for primitive in ["noul", "choice", "score"] {
        let subset: Vec<&Outcome> = outcomes
            .iter()
            .filter(|o| o.primitive == primitive)
            .collect();
        if !subset.is_empty() {
            let hits = subset.iter().filter(|o| o.correct).count() as f64;
            per_primitive.insert(primitive, hits / subset.len() as f64);
        }
    }

    let (best_t, _) = fit_temperature(outcomes);

    Report {
        model: model.to_string(),
        fidelity: answered.len() as f64 / n,
        accuracy: outcomes.iter().filter(|o| o.correct).count() as f64 / n,
        per_primitive,
        p50: percentile(&latencies, 0.50),
        p99: percentile(&latencies, 0.99),
        truncation: outcomes.iter().filter(|o| o.truncated).count() as f64 / n,
        brier: brier(outcomes, cases),
        best_t,
        agreement: None,
        picked: outcomes.iter().map(|o| o.picked.clone()).collect(),
    }
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let models: Vec<String> = arg(&args, "--models")
        .unwrap_or_else(|| "gemma4:e2b-it-qat,granite4:3b,phi4-mini:3.8b".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let reference = arg(&args, "--reference");
    let dataset_path =
        arg(&args, "--dataset").unwrap_or_else(|| "crates/cerno-bench/dataset.json".into());
    let out_path = arg(&args, "--out").unwrap_or_else(|| "docs/model-selection.md".into());
    let host_url = arg(&args, "--host").unwrap_or_else(|| "http://localhost:11434".into());

    let dataset: Dataset = serde_json::from_str(&std::fs::read_to_string(&dataset_path)?)?;
    println!("{} cases from {dataset_path}", dataset.cases.len());

    let host: Arc<dyn ModelHost> = Arc::new(OllamaHost::new(&host_url, Duration::from_secs(120))?);
    // Keep each model resident for the length of its run; unloading between cases would measure
    // model loading, not inference.
    let engine = Engine::new(host, Some("5m".to_string()));

    let mut reports = Vec::new();
    let mut reference_picks: Option<Vec<Option<String>>> = None;

    let all: Vec<String> = reference
        .iter()
        .cloned()
        .chain(models.iter().cloned())
        .collect();

    for model in &all {
        println!("\n{model}");

        // Warm-up: the first call pays for loading the model into VRAM and would otherwise
        // dominate the p99.
        let _ = engine
            .answer(
                "warm up",
                &Question {
                    id: "warmup".into(),
                    kind: QuestionKind::Noul("Is this a warm-up?".into()),
                },
                model,
                Calibration::default(),
            )
            .await;

        let mut outcomes = Vec::with_capacity(dataset.cases.len());
        for case in &dataset.cases {
            outcomes.push(run_case(&engine, model, case).await);
        }

        let mut report = summarise(model, &outcomes, &dataset.cases);

        if reference.as_deref() == Some(model.as_str()) {
            reference_picks = Some(report.picked.clone());
        } else if let Some(reference_picks) = &reference_picks {
            let agreed = report
                .picked
                .iter()
                .zip(reference_picks)
                .filter(|(a, b)| a.is_some() && a == b)
                .count();
            report.agreement = Some(agreed as f64 / dataset.cases.len() as f64);
        }

        println!(
            "  fidelity {:.0}%  accuracy {:.0}%  p50 {}ms  p99 {}ms  best T {:.2}",
            report.fidelity * 100.0,
            report.accuracy * 100.0,
            report.p50,
            report.p99,
            report.best_t
        );
        reports.push(report);
    }

    let markdown = render(&reports, &dataset, reference.as_deref());
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, &markdown)?;
    println!("\nwrote {out_path}");

    Ok(())
}

fn pct(v: f64) -> String {
    if v.is_nan() {
        "—".into()
    } else {
        format!("{:.0}%", v * 100.0)
    }
}

/// The best candidate: highest accuracy among models that answered every case with a usable
/// label, ties broken by median latency. Fidelity is a gate first — a model cerno cannot read is
/// not a candidate at any accuracy — and the reference is excluded, since it is the yardstick
/// rather than an option.
fn pick_winner<'a>(reports: &'a [Report], reference: Option<&str>) -> Option<&'a Report> {
    reports
        .iter()
        .filter(|r| Some(r.model.as_str()) != reference)
        .filter(|r| r.fidelity >= 1.0)
        .min_by(|a, b| {
            b.accuracy
                .total_cmp(&a.accuracy)
                .then_with(|| a.p50.cmp(&b.p50))
        })
}

fn render(reports: &[Report], dataset: &Dataset, reference: Option<&str>) -> String {
    let mut out = String::new();
    out.push_str("# Model selection\n\n");
    out.push_str(&format!(
        "Generated by `cargo run -p cerno-bench`. {} labelled cases: {} noul, {} choice, {} score.\n\n",
        dataset.cases.len(),
        dataset.cases.iter().filter(|c| c.noul.is_some()).count(),
        dataset.cases.iter().filter(|c| c.choice.is_some()).count(),
        dataset.cases.iter().filter(|c| c.score.is_some()).count(),
    ));

    out.push_str("| Model | Fidelity | Accuracy | noul | choice | score | p50 | p99 | Truncated | Brier | Best T | Agreement |\n");
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|\n");

    for r in reports {
        let marker = if Some(r.model.as_str()) == reference {
            " *(reference)*"
        } else {
            ""
        };
        out.push_str(&format!(
            "| `{}`{} | {} | {} | {} | {} | {} | {} ms | {} ms | {} | {:.3} | {:.2} | {} |\n",
            r.model,
            marker,
            pct(r.fidelity),
            pct(r.accuracy),
            pct(r.per_primitive.get("noul").copied().unwrap_or(f64::NAN)),
            pct(r.per_primitive.get("choice").copied().unwrap_or(f64::NAN)),
            pct(r.per_primitive.get("score").copied().unwrap_or(f64::NAN)),
            r.p50,
            r.p99,
            pct(r.truncation),
            r.brier,
            r.best_t,
            r.agreement.map(pct).unwrap_or_else(|| "—".into()),
        ));
    }

    if let Some(winner) = pick_winner(reports, reference) {
        out.push_str(&format!(
            "\n## Verdict\n\n\
             **`{}`** — the most accurate model that answered every case with a usable label, \
             ties broken by median latency.\n\n\
             Set it as the default with `CERNO_DEFAULT_MODEL={}`. Its best-fit calibration \
             temperature above is {:.2}; a value below 1 means the model is slightly *under*\
             confident and would need sharpening rather than flattening. cerno ships a default of \
             1.0 either way — {} labelled cases is thin evidence for baking in an adjustment, and \
             callers can always pass their own.\n",
            winner.model,
            winner.model,
            winner.best_t,
            dataset.cases.len(),
        ));
    }

    out.push_str(
        "\n## Reading the table\n\n\
         - **Fidelity** — share of cases answered with one of the offered letters. This is a gate,\n  \
           not a score: below 100% the model is ignoring the instruction on some inputs, and no\n  \
           amount of accuracy makes up for an answer cerno cannot read.\n\
         - **Accuracy** — share of cases where the picked label was the labelled one. Score cases\n  \
           allow the per-case tolerance in the dataset.\n\
         - **p50 / p99** — wall-clock per question, warm model, one question per request.\n\
         - **Truncated** — share of answers where some label fell outside the host's top-20 window,\n  \
           making its probability an upper bound rather than an observation.\n\
         - **Brier** — mean squared error of the yes-probability on noul cases; lower is better.\n\
         - **Best T** — the calibration temperature that best explains the labelled outcomes. Far\n  \
           above 1 means the model is overconfident and needs flattening.\n\
         - **Agreement** — share of cases where the model picked the same label as the reference.\n",
    );

    out
}
