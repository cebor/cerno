//! Measures candidate models against a labelled dataset and writes the comparison table.
//!
//! Four things decide a model here, and the first one is a gate rather than a score:
//!
//! 1. **Label fidelity** — was the model's most likely first token one of the letters it was
//!    offered? A model that writes prose instead is unusable for cerno at any accuracy, even
//!    when a letter turns up further down the ranking and an answer can still be read off it.
//! 2. **Accuracy** — did it pick the right letter.
//! 3. **Latency** — the whole point of one forward pass.
//! 4. **Calibration** — how far its confidence has to be flattened to stop lying.
//!
//! Usage:
//!   cerno-bench --models a,b,c [--reference m] [--dataset p] [--out p] [--host url]
//!               [--host-kind ollama|openai|vllm|llamacpp|lmstudio]
//!
//! An API key for the OpenAI-compatible hosts comes from `CERNO_HOST_API_KEY`.

use cerno_core::{Engine, EngineError, labels};
use cerno_host::HostKind;
use cerno_types::{Answer, Calibration, ChoiceSpec, Question, QuestionKind, ScoreSpec};
use serde::Deserialize;
use std::collections::BTreeMap;
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
    /// Why this case cannot be run, if it cannot. The accessors below assume a case passed, so a
    /// dataset is checked whole before the first model is loaded, rather than panicking partway
    /// through a run over whichever case happens to be malformed.
    fn problem(&self) -> Option<String> {
        let named = [
            self.noul.is_some(),
            self.choice.is_some(),
            self.score.is_some(),
        ];
        if named.iter().filter(|n| **n).count() != 1 {
            return Some("needs exactly one of noul, choice or score".into());
        }

        if self.noul.is_some() {
            return self
                .expect_yes
                .is_none()
                .then(|| "a noul case needs expect_yes".into());
        }

        let (offered, first) = match (&self.choice, &self.score) {
            (Some(choice), _) => (choice.options.len(), 0),
            (_, Some(score)) => (score.levels.count(), 1),
            _ => unreachable!("exactly one primitive, and it is not a noul"),
        };
        if offered == 0 {
            return Some(format!("this {} offers nothing to pick", self.primitive()));
        }
        let last = offered + first - 1;
        match self.expect {
            None => Some(format!("a {} case needs expect", self.primitive())),
            Some(want) if want < first || want > last => Some(format!(
                "expect is {want}, outside {first}..={last} for this {}",
                self.primitive()
            )),
            Some(_) => None,
        }
    }

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
    /// Whether the most likely first token was one of the offered labels. An answer can exist
    /// without this: the engine reads any label in the top 20 and renormalises, so a model that
    /// opens with `**` or `The` still gets an answer, and a confident-looking one.
    faithful: bool,
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
        .answer_with_distribution(&case.state, &question, model, Calibration::default())
        .await;
    let latency = started.elapsed();

    match result {
        Ok((answer, distribution)) => {
            let logprobs = match &answer {
                Answer::Noul { raw_logprobs, .. }
                | Answer::Choice { raw_logprobs, .. }
                | Answer::Score { raw_logprobs, .. } => raw_logprobs.clone(),
            };
            // `raw_logprobs` is keyed by exactly the labels offered.
            let faithful = top_token_is_a_label(&distribution.tokens, logprobs.keys());

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
                faithful,
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
                faithful: false,
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

/// Whether the highest-ranked token spells one of `offered`. `tokens` is ranked, highest first.
fn top_token_is_a_label<'a>(
    tokens: &[(String, f64)],
    mut offered: impl Iterator<Item = &'a String>,
) -> bool {
    tokens
        .first()
        .is_some_and(|(top, _)| offered.any(|label| labels::matches(top, label)))
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
        fidelity: outcomes.iter().filter(|o| o.faithful).count() as f64 / n,
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
    let host_kind: HostKind = arg(&args, "--host-kind")
        .unwrap_or_else(|| "ollama".into())
        .parse()?;
    let host_url = arg(&args, "--host").unwrap_or_else(|| host_kind.default_url().into());

    let dataset: Dataset = serde_json::from_str(&std::fs::read_to_string(&dataset_path)?)?;
    let problems: Vec<String> = dataset
        .cases
        .iter()
        .filter_map(|case| case.problem().map(|p| format!("  {}: {p}", case.id)))
        .collect();
    if !problems.is_empty() {
        eprintln!(
            "{dataset_path} has cases that cannot run:\n{}",
            problems.join("\n")
        );
        std::process::exit(1);
    }
    println!("{} cases from {dataset_path}", dataset.cases.len());

    let host = cerno_host::connect(
        host_kind,
        &host_url,
        std::env::var("CERNO_HOST_API_KEY").ok(),
        Duration::from_secs(120),
    )?;
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
        // dominate the p99. It is also where a misspelt model or an unreachable host shows:
        // measured anyway, every case would fail, the model would be reported at 0% fidelity for
        // a fault that is not its own, and the doc would be overwritten with that. A model that
        // answers without a letter is different — that is exactly what the run is here to count.
        let warm_up = engine
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
        if let Err(err @ (EngineError::Host(_) | EngineError::UnknownModel { .. })) = warm_up {
            eprintln!("{model}: the warm-up failed, so nothing was measured or written: {err}");
            std::process::exit(1);
        }

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

/// The best candidate: highest accuracy among models whose top token was an offered label in
/// every case, ties broken by median latency. Fidelity is a gate first — a model cerno cannot read is
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

/// Render a markdown table with padded cells, the first column left-aligned and the rest right.
///
/// Markdown does not need the padding, but a generated file is read as often in a terminal as in
/// a renderer, and an unpadded table with a column as wide and as variable as a model name is
/// hard to scan in either. Right-aligning the numbers lines up their digits.
fn markdown_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    // Column width in characters. Every cell here is ASCII apart from the dagger and the em
    // dash, neither of which is double-width, so counting chars is the right measure.
    let widths: Vec<usize> = headers
        .iter()
        .enumerate()
        .map(|(i, header)| {
            rows.iter()
                .filter_map(|row| row.get(i))
                .map(|cell| cell.chars().count())
                .chain(std::iter::once(header.chars().count()))
                .max()
                .unwrap_or(0)
        })
        .collect();

    let pad = |cell: &str, width: usize, right: bool| {
        let fill = " ".repeat(width.saturating_sub(cell.chars().count()));
        if right {
            format!("{fill}{cell}")
        } else {
            format!("{cell}{fill}")
        }
    };

    let mut out = String::new();

    let header: Vec<String> = headers
        .iter()
        .zip(&widths)
        .enumerate()
        .map(|(i, (h, w))| pad(h, *w, i > 0))
        .collect();
    out.push_str(&format!("| {} |\n", header.join(" | ")));

    // A cell occupies width + 2 columns, from the space either side of it. The rule has one
    // alignment colon and no spaces, so it needs width + 1 dashes to line up with the rest.
    let rule: Vec<String> = widths
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let dashes = "-".repeat(w + 1);
            if i == 0 {
                format!(":{dashes}")
            } else {
                format!("{dashes}:")
            }
        })
        .collect();
    out.push_str(&format!("|{}|\n", rule.join("|")));

    for row in rows {
        let cells: Vec<String> = row
            .iter()
            .zip(&widths)
            .enumerate()
            .map(|(i, (cell, w))| pad(cell, *w, i > 0))
            .collect();
        out.push_str(&format!("| {} |\n", cells.join(" | ")));
    }

    out
}

/// What a best-fit temperature says about the model, in words that stay true for any value.
///
/// Written out rather than fixed in the template: the verdict is regenerated with every run, and
/// a sentence that only fits one outcome turns false the first time another model wins.
fn calibration_reading(best_t: f64) -> &'static str {
    // The grid steps by 0.05, so anything this close to 1 is 1 within the fit's resolution.
    if (best_t - 1.0).abs() <= 0.1 {
        "that is close enough to 1 that the model's own probabilities can be taken as they are"
    } else if best_t < 1.0 {
        "a value below 1 means the model is *under*confident and would need sharpening rather \
         than flattening"
    } else {
        "a value above 1 means the model is *over*confident and its probabilities need \
         flattening before they mean anything"
    }
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

    let headers = [
        "Model",
        "Fidelity",
        "Accuracy",
        "noul",
        "choice",
        "score",
        "p50 (ms)",
        "p99 (ms)",
        "Truncated",
        "Brier",
        "Best T",
        "Agreement",
    ];

    let mut reference_seen = false;
    let rows: Vec<Vec<String>> = reports
        .iter()
        .map(|r| {
            // The reference is marked with a footnote numeral rather than an inline
            // "(reference)": the model column already holds the widest values in the table, and
            // a parenthetical there widened it by half again and left the column ragged. Not a
            // dagger: it reads as a cross to anyone who hasn't met it as a footnote mark.
            let is_reference = Some(r.model.as_str()) == reference;
            reference_seen |= is_reference;

            vec![
                format!("`{}`{}", r.model, if is_reference { " ¹" } else { "" }),
                pct(r.fidelity),
                pct(r.accuracy),
                pct(r.per_primitive.get("noul").copied().unwrap_or(f64::NAN)),
                pct(r.per_primitive.get("choice").copied().unwrap_or(f64::NAN)),
                pct(r.per_primitive.get("score").copied().unwrap_or(f64::NAN)),
                r.p50.to_string(),
                r.p99.to_string(),
                pct(r.truncation),
                format!("{:.3}", r.brier),
                format!("{:.2}", r.best_t),
                r.agreement.map(pct).unwrap_or_else(|| "—".into()),
            ]
        })
        .collect();

    out.push_str(&markdown_table(&headers, &rows));

    if reference_seen {
        out.push_str(
            "\n¹ Reference model — the yardstick the Agreement column is measured against, \
             not a candidate.\n",
        );
    }

    if let Some(winner) = pick_winner(reports, reference) {
        out.push_str(&format!(
            "\n## Verdict\n\n\
             **`{}`** — the most accurate model whose most likely token was an offered letter \
             in every case, ties broken by median latency.\n\n\
             Set it as the default with `CERNO_DEFAULT_MODEL={}`. Its best-fit calibration \
             temperature above is {:.2}; {}. cerno ships a default of 1.0 either way — {} \
             labelled cases is thin evidence for baking in an adjustment, and callers can always \
             pass their own.\n",
            winner.model,
            winner.model,
            winner.best_t,
            calibration_reading(winner.best_t),
            dataset.cases.len(),
        ));
    }

    out.push_str(
        "\n## Reading the table\n\n\
         - **Fidelity** — share of cases whose most likely first token was one of the offered\n  \
           letters. This is a gate, not a score: below 100% the model is ignoring the instruction\n  \
           on some inputs, and cerno then reads its answer off tokens it was not going to write.\n\
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The table is read in a terminal as often as in a renderer, so every row — the rule
    /// included — has to be the same width. Pinned character for character.
    fn case(json: serde_json::Value) -> Case {
        let mut base = serde_json::json!({"id": "c", "state": "s"});
        base.as_object_mut()
            .unwrap()
            .extend(json.as_object().unwrap().clone());
        serde_json::from_value(base).unwrap()
    }

    /// The checked-in dataset is what `cargo run -p cerno-bench` uses; it has to pass its own check.
    #[test]
    fn every_shipped_case_can_run() {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("dataset.json"),
        )
        .unwrap();
        let dataset: Dataset = serde_json::from_str(&text).unwrap();

        for case in &dataset.cases {
            assert_eq!(case.problem(), None, "{}", case.id);
        }
    }

    #[test]
    fn a_case_that_could_not_be_scored_is_found_before_running() {
        let bad = [
            serde_json::json!({}),
            serde_json::json!({"noul": "a?", "choice": {"options": ["x", "y"]}, "expect_yes": true}),
            serde_json::json!({"noul": "a?"}),
            serde_json::json!({"choice": {"options": ["x", "y"]}}),
            serde_json::json!({"choice": {"options": ["x", "y"]}, "expect": 2}),
            serde_json::json!({"choice": {"options": []}, "expect": 0}),
            // Score levels are 1-based; 0 would have underflowed computing the label.
            serde_json::json!({"score": {"levels": 5}, "expect": 0}),
            serde_json::json!({"score": {"levels": 5}, "expect": 6}),
        ];
        for json in bad {
            assert!(
                case(json.clone()).problem().is_some(),
                "{json} was accepted"
            );
        }

        for good in [
            serde_json::json!({"noul": "a?", "expect_yes": false}),
            serde_json::json!({"choice": {"options": ["x", "y"]}, "expect": 1}),
            serde_json::json!({"score": {"levels": 5}, "expect": 5}),
        ] {
            assert_eq!(case(good.clone()).problem(), None, "{good}");
        }
    }

    #[test]
    fn table_columns_line_up_exactly() {
        let table = markdown_table(
            &["Model", "p50 (ms)"],
            &[
                vec!["`gemma4:e2b-it-qat`".into(), "37".into()],
                vec!["`x`".into(), "1234".into()],
            ],
        );

        assert_eq!(
            table,
            "\
| Model               | p50 (ms) |
|:--------------------|---------:|
| `gemma4:e2b-it-qat` |       37 |
| `x`                 |     1234 |
"
        );
    }

    #[test]
    fn every_line_of_a_table_is_the_same_width() {
        let table = markdown_table(
            &["Model", "Brier", "Agreement"],
            &[
                vec![
                    "`gemma4:26b-a4b-it-q4_K_M` ¹".into(),
                    "0.029".into(),
                    "—".into(),
                ],
                vec!["`granite4:3b`".into(), "0.168".into(), "75%".into()],
            ],
        );

        let widths: Vec<usize> = table.lines().map(|l| l.chars().count()).collect();

        assert!(
            widths.windows(2).all(|w| w[0] == w[1]),
            "ragged table, line widths {widths:?}:\n{table}"
        );
    }

    /// A value wider than its header must widen the column, not overflow it.
    /// A letter further down the ranking is enough for an answer, not for fidelity.
    #[test]
    fn only_a_label_at_the_top_is_faithful() {
        let offered = ["A".to_string(), "B".to_string()];
        let ranked = |tokens: &[(&str, f64)]| -> Vec<(String, f64)> {
            tokens.iter().map(|(t, l)| (t.to_string(), *l)).collect()
        };

        assert!(top_token_is_a_label(
            &ranked(&[(" b", -0.1), ("A", -3.0)]),
            offered.iter()
        ));
        assert!(!top_token_is_a_label(
            &ranked(&[("**", -0.01), ("A", -8.0)]),
            offered.iter()
        ));
        assert!(!top_token_is_a_label(&[], offered.iter()));
    }

    /// The verdict is regenerated every run, so its sentence must fit whichever value comes out.
    #[test]
    fn the_calibration_reading_follows_the_temperature() {
        assert!(calibration_reading(0.8).contains("*under*confident"));
        assert!(calibration_reading(3.2).contains("*over*confident"));
        assert!(calibration_reading(1.05).contains("close enough to 1"));
    }

    #[test]
    fn a_long_cell_widens_its_column() {
        let table = markdown_table(&["T"], &[vec!["a-very-long-value".into()]]);

        assert!(table.starts_with("| T                 |\n"), "{table}");
    }
}
