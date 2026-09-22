# cerno

Typed decisions from a locally hosted model. Three primitives:

| | | |
|---|---|---|
| **noul** | how likely the answer is yes | `0.0 .. 1.0` |
| **choice** | which one of up to 20 options | the option, plus a probability for each |
| **score** | where on a rubric of 2–10 levels | the level, its legend, and a weighted mean |

Everything runs against Ollama on your own machine. Nothing leaves it.

*cernere*, Latin: to sift, to distinguish, to decide.

```bash
curl localhost:3000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Ticket: Serverraum-Klima ausgefallen, 31 Grad und steigend.",
  "questions": [
    {"id": "urgent", "noul": "Ist das dringend?"},
    {"id": "team", "choice": {"question": "Welches Team?", "options": ["IT", "Facility", "HR"]}},
    {"id": "sev",  "score":  {"question": "Wie schwer?", "levels": 5}}
  ]}'
```

```json
{"answers": {
   "urgent": {"type": "noul",   "noul": 0.9911, "confidence": 0.927, "truncated": false},
   "team":   {"type": "choice", "choice": "Facility", "index": 1, "confidence": 0.939},
   "sev":    {"type": "score",  "score": 5, "expected_score": 4.88, "legend": "5"}},
 "model": "gemma4:e2b-it-qat",
 "usage": {"input_tokens": 329, "questions": 3},
 "timing_ms": {"total": 144}}
```

Three questions, 144 ms, on a 4 GB model.

## How it works

A generative model answers by writing words, and then something has to parse the words back
into a decision. cerno never lets it get that far.

Each question becomes a short prompt whose options are lettered `A`, `B`, `C`. The model is
asked for exactly **one** token, and instead of taking the token cerno reads the *probability
distribution* over it. `P(A)`, `P(B)`, `P(C)` are all there, in one forward pass, whatever the
number of options.

```
question ──> options ──> labels A,B,C ──> prompt ──> model ──> distribution over one token
                                                                          │
                       typed answer <── calibrate <── fold variants, floor ┘
```

That is where the speed comes from — one token, not a sentence — and where the probabilities
come from, since they are the model's own, not a number it was asked to invent.

### What the measurements forced

The design is shaped by six things that turned out to be true of real models, each of which
would otherwise have produced quietly wrong answers:

| Measured | Consequence |
|---|---|
| The first generated token was `<\|channel\|>`, a chat-template control token | `think: false` on every request |
| The default `top_k: 40` truncates the distribution *before* logprobs are reported, and renormalises what survives | `top_k: 0, top_p: 1, min_p: 0` are pinned in `cerno-host` and not configurable |
| "Nein" is not one token — it arrives as `Ne` + `in` — and "Ja" competes with `JA`, ` Ja`, `ja` | Single-letter labels only; variants of one label are folded in probability space |
| When a model is certain, the losing label drops out of the top-20 window entirely | The weakest reported logprob becomes an upper bound, and the answer is flagged `truncated` |
| `gemma4:26b` answers clear cases at `P = 1.0000` | Temperature scaling, per request or per model |
| A 26B model answers in 87 ms; a 4 GB one in 36 ms | A small model is the default |

## Getting started

```bash
ollama pull gemma4:e2b-it-qat
cargo run -p cerno-server
```

Then `http://localhost:3000/docs` for Swagger UI, or one of the SDKs:

```rust
let answers = client.systemone(text)
    .noul("urgent", "Is this urgent?")
    .choice("team", "Which team?", ["IT", "Facility"])
    .send().await?;
```
```python
answers = (client.systemone(text)
    .noul("urgent", "Is this urgent?")
    .choice("team", "Which team?", ["IT", "Facility"])
    .send())
```
```ts
const answers = await client.systemone(text)
  .noul("urgent", "Is this urgent?")
  .choice("team", "Which team?", ["IT", "Facility"])
  .send();
```

All three are written by hand against `spec/openapi.json` and tested against the same
[conformance cases](spec/conformance/cases.json), so they cannot drift apart silently.

## The terminal front end

```bash
cargo run -p cerno-tui            # or --url http://host:3000, or CERNO_URL
```

```
┌ State (1) ───────────────────────┐┌ Answers ─────────────────────────────┐
│Ticket: Serverraum-Klima          ││sev      score → 5 "kritisch" conf 0.69│
│ausgefallen, 31 Grad und steigend.││          expected 4.80                │
└──────────────────────────────────┘│  1 unkritisch▎░░░░░░░░░░░░░░░░░  0.9% │
┌ Questions (2) ───────────────────┐│  4 hoch      ██░░░░░░░░░░░░░░░░  8.2% │
│  urgent    noul   Ist das dring… ││  5 kritisch  █████████████████░ 87.6% │
│  team      choice IT | Facility  ││                                       │
│▸ sev       score  unkritisch | … ││team     choice → Facility   conf 0.965│
│  + add question  (a)             ││  IT          ░░░░░░░░░░░░░░░░░░  0.2% │
│                                  ││  Facility    █████████████████▉ 99.4% │
└──────────────────────────────────┘└───────────────────────────────────────┘
 localhost:3000 ● · default model · ^S send · ? help
```

Type a state, add questions with `a`, send with `Ctrl+S`. `Tab` moves between panes, `t` and
`T` step the calibration temperature and `c` clears it, `m` cycles the models the service
offers, `?` lists the keys. The form is saved on exit and comes back on the next start.

The bars are the reason it exists. `curl` gives you the winner; the terminal shows you how close
the runner-up was, which is the difference between "Facility" and "Facility, but it was nearly
IT". A run at a higher temperature next to one at 1.0 shows the calibration working directly.

**The TUI validates nothing.** It builds the request through `cerno-sdk` and lets the service
decide; a refusal is shown with its error code, and the question the service blamed is marked in
the list. A fourth copy of the rules, after `cerno-core`, the OpenAPI document and the SDKs,
would be the copy that drifts.

## Choosing a model

`cargo run -p cerno-bench` scores candidates over a labelled dataset and writes
[docs/model-selection.md](docs/model-selection.md). The current result:

| Model                        | Fidelity | Accuracy | p50 (ms) | Brier | Best T |
|:-----------------------------|---------:|---------:|---------:|------:|-------:|
| `gemma4:26b-a4b-it-q4_K_M` † |     100% |      97% |      202 | 0.029 |   3.20 |
| `gemma4:e2b-it-qat`          |     100% |      97% |       40 | 0.006 |   0.80 |
| `granite4:3b`                |      97% |      78% |       23 | 0.167 |   2.20 |
| `phi4-mini:3.8b`             |     100% |      86% |       28 | 0.023 |   2.25 |

† Reference model — the yardstick, not a candidate. A snapshot of the generated document above;
latency depends on hardware and on what else is holding VRAM, so it moves between runs while
the accuracy and calibration columns stay put.

The 4 GB model matches the 26B reference's accuracy on a quarter of the footprint, several
times faster, and is *better* calibrated: the large model needs its logits flattened by 3.2
before its confidences mean anything, while the small one is very slightly under-confident.

**Fidelity is a gate, not a score.** A model that answers in prose instead of a letter is
unusable here at any accuracy, which is why the benchmark reports it first — and why
`granite4:3b`, which slipped to 97% on this run, is not a candidate regardless of its speed.

## Configuration

Deployment knobs are environment variables; the model table is an optional TOML file
(see [cerno.example.toml](cerno.example.toml)).

| Variable | Default | |
|---|---|---|
| `CERNO_BIND` | `0.0.0.0:3000` | |
| `CERNO_OLLAMA_URL` | `http://localhost:11434` | |
| `CERNO_DEFAULT_MODEL` | `gemma4:e2b-it-qat` | Alias or model name |
| `CERNO_CONFIG` | — | Path to the model table |
| `CERNO_STRICT_MODELS` | `false` | Only allow configured models |
| `CERNO_MAX_CONCURRENT_QUESTIONS` | `4` | In flight against the host at once |
| `CERNO_KEEP_ALIVE` | `5m` | Empty means: do not send it |
| `CERNO_HOST_TIMEOUT_SECS` | `30` | |
| `RUST_LOG` | `cerno_server=info,cerno_core=info` | |

## Layout

```
crates/
  cerno-types    wire types — serde always, utoipa behind a feature
  cerno-host     ModelHost trait + the Ollama adapter
  cerno-core     labels, prompt, logprob maths, engine
  cerno-server   axum + utoipa
  cerno-sdk      Rust client
  cerno-tui      terminal front end, on cerno-sdk
  cerno-bench    model benchmark
sdks/python      uv package `cerno`
sdks/typescript  npm package `@cerno/sdk`
spec/            openapi.json + conformance cases
```

`ModelHost` is the seam for a second runtime. It knows nothing about noul, choice or score — it
answers one question, "what is the distribution over the next token?" — so a llama.cpp adapter
would be a new file, not a new design.

## Limits

- **20 options.** Ollama reports at most 20 ranked tokens, so a 21st option could never be
  observed. Over that, the service answers 422 rather than degrading quietly. Splitting a large
  set across two questions works today; doing it automatically does not.
- **Position bias.** Options are always lettered in request order, and models have some
  preference for `A`. Shuffling and averaging would cost a second pass, so it is not done.
- **Logprobs wobble.** Identical requests can return logprobs differing in the third decimal —
  GPU reduction order, not calibration. It does not change answers; it does mean exact equality
  is the wrong assertion in a test against a live model.
- **The TUI has no mouse support and no history.** One form, one request, and the last one is
  remembered across restarts. Comparing two runs side by side means running them one after the
  other and reading the numbers.

## Development

```bash
cargo test --workspace          # includes the TUI, which needs no terminal to test
cd sdks/python && uv run pytest
cd sdks/typescript && npm test
```

`spec/openapi.json` is generated, and a test fails when it drifts from the code:

```bash
cargo run -p cerno-server --bin cerno-openapi > spec/openapi.json
```
