<p align="center">
  <img src="https://cebor.github.io/cerno/logo.svg" alt="cerno logo" width="112" height="112">
</p>

<h1 align="center">cerno</h1>

<p align="center">
  <strong>Typed decisions from a locally hosted model.</strong><br>
  <em>cernere</em>, Latin: to sift, to distinguish, to decide.
</p>

<p align="center">
  <a href="https://github.com/cebor/cerno/actions/workflows/ci.yml"><img src="https://github.com/cebor/cerno/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/cerno-server"><img src="https://img.shields.io/crates/v/cerno-server" alt="crates.io"></a>
  <a href="https://pypi.org/project/cerno-sdk/"><img src="https://img.shields.io/pypi/v/cerno-sdk" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/cerno-sdk"><img src="https://img.shields.io/npm/v/cerno-sdk" alt="npm"></a>
  <a href="https://cebor.github.io/cerno/api/"><img src="https://img.shields.io/badge/docs-api-1e1b4b" alt="API docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-f59e0b" alt="MIT licence"></a>
</p>

<p align="center">
  <a href="https://cebor.github.io/cerno/">Website</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#the-terminal-front-end">Terminal front end</a> ·
  <a href="#choosing-a-model">Choosing a model</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#hosts">Hosts</a> ·
  <a href="#limits">Limits</a>
</p>

Three primitives:

| Primitive | Asks | Returns |
|---|---|---|
| **noul** | how likely the answer is yes | `0.0 .. 1.0` |
| **choice** | which one of up to 20 options | the option, plus a probability for each |
| **score** | where on a rubric of 2–10 levels | the level, its legend, and a weighted mean |

Everything runs against a model on your own machine — Ollama by default, or vLLM, llama.cpp,
LM Studio or anything else that speaks OpenAI's API. Nothing leaves it unless you point it at
a remote endpoint.

```bash
curl localhost:3000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Ticket: Server room cooling failed, 31 degrees and rising.",
  "questions": [
    {"id": "urgent", "noul": "Is this urgent?"},
    {"id": "team", "choice": {"question": "Which team?", "options": ["IT", "Facility", "HR"]}},
    {"id": "sev",  "score":  {"question": "How severe?",
                           "levels": ["negligible", "minor", "moderate", "major", "critical"]}}
  ]}'
```

```json
{"answers": {
   "urgent": {"type": "noul",   "noul": 0.9946, "label_mass": 0.999, "truncated": false},
   "team":   {"type": "choice", "choice": "Facility", "index": 1,
              "confidence": 0.967, "label_mass": 0.999, "truncated": false},
   "sev":    {"type": "score",  "score": 5, "expected_score": 4.90, "legend": "critical",
              "confidence": 0.807, "label_mass": 0.999, "truncated": false}},
 "model": "gemma4:e2b-it-qat",
 "usage": {"input_tokens": 281, "questions": 3},
 "timing_ms": {"total": 85}}
```

Trimmed: every answer also carries `raw_logprobs` and `truncated_labels`, and a choice or score
its `probabilities`. `label_mass` is how much of the model's own probability fell on the offered
letters: near 1 here, because the model answered with a letter. The probabilities are normalised
over the letters alone, so a low `label_mass` is the only sign that it was about to write
something else and the answer was read off letters far down its ranking.
Three questions, 85 ms, on a 4 GB model.

## How it works

A generative model answers by writing words, and then something has to parse the words back
into a decision. cerno never lets it get that far.

Each question becomes a short prompt whose options are lettered `A`, `B`, `C`. The model is
asked for exactly **one** token, and instead of taking the token cerno reads the *probability
distribution* over it. `P(A)`, `P(B)`, `P(C)` are all there, in one forward pass, whatever the
number of options.

```
question ──> options ──> labels A,B,C ──> prompt ──> model
                                                       │
                                          distribution over one token
                                                       │
typed answer <── calibrate <── fold variants, floor <──┘
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

The service and the terminal front end are both on crates.io, so installing needs a Rust
toolchain and nothing else:

```bash
ollama pull gemma4:e2b-it-qat          # once, 4.3 GB
cargo install cerno-server cerno-tui
```

Two terminals: the service stays in the foreground, something else talks to it.

```bash
# terminal 1 — the service
cerno-server

# terminal 2 — the terminal front end
cerno-tui
```

The service listens on `127.0.0.1:3000` and the front end looks there by default, so there is
nothing to configure. `localhost:3000 ●` in its status bar means the two found each other.

**The first request takes 5–10 seconds** while Ollama loads the model into VRAM; after that it
is around 150 ms.

From a checkout, `cargo run --release -p cerno-server` and `cargo run --release -p cerno-tui`
do the same. `--release` matters most for the front end, where a debug build makes typing
noticeably sluggish.

### From code

With the service running, `http://localhost:3000/docs` is the Swagger UI, and `curl` or one of
the SDKs works just as well. All three are published as `cerno-sdk`:

```bash
cargo add cerno-sdk tokio --features tokio/macros,tokio/rt-multi-thread
uv add cerno-sdk                       # or: pip install cerno-sdk
npm install cerno-sdk
```

```rust
use cerno_sdk::Client;

let client = Client::new("http://localhost:3000")?;
let answers = client.systemone(text)
    .noul("urgent", "Is this urgent?")
    .choice("team", "Which team?", ["IT", "Facility"])
    .send().await?;
```
```python
from cerno import Client

client = Client("http://localhost:3000")
answers = (client.systemone(text)
    .noul("urgent", "Is this urgent?")
    .choice("team", "Which team?", ["IT", "Facility"])
    .send())
```
```ts
import { Client } from "cerno-sdk";

const client = new Client("http://localhost:3000");
const answers = await client.systemone(text)
  .noul("urgent", "Is this urgent?")
  .choice("team", "Which team?", ["IT", "Facility"])
  .send();
```

The Python package installs as `cerno-sdk` and imports as `cerno`.

All three are written by hand against `spec/openapi.json` and tested against the same
[conformance cases](spec/conformance/cases.json), so they cannot drift apart silently.

The website is [cebor.github.io/cerno](https://cebor.github.io/cerno/); the API documentation of
every crate, with links to the Python and TypeScript READMEs, is under
[/api](https://cebor.github.io/cerno/api/).

## The terminal front end

Started above. It points at `http://localhost:3000` unless `--url` or `CERNO_URL` says
otherwise.

<p align="center">
  <img src="https://cebor.github.io/cerno/tui.png" width="840" alt="cerno-tui with the server-room ticket as state, three questions, and the answers as bars: urgent 99.3% yes, team Facility at 99.5%, severity critical at 91.8%">
</p>

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
| `gemma4:26b-a4b-it-q4_K_M` ¹ |     100% |      97% |       64 | 0.029 |   3.20 |
| `gemma4:e2b-it-qat`          |     100% |      97% |       30 | 0.005 |   0.80 |
| `granite4:3b`                |     100% |      81% |       26 | 0.168 |   2.10 |
| `phi4-mini:3.8b`             |     100% |      86% |       22 | 0.023 |   2.25 |
| `nimble`                     |     100% |      97% |      124 | 0.014 |   0.65 |
| `tev1`                       |     100% |      94% |      122 | 0.030 |   0.25 |
| `tev1:0.8b`                  |     100% |      78% |       35 | 0.070 |   0.65 |
| `nimble` ²                   |        — |     100% |       92 | 0.008 |   0.70 |
| `tev1` ²                     |        — |     100% |       90 | 0.016 |   0.60 |
| `tev1:0.8b` ²                |        — |      86% |       39 | 0.075 |   0.45 |

¹ Reference model — the yardstick, not a candidate. A snapshot of the generated document above;
latency depends on hardware and on what else is holding VRAM, so it moves between runs while
the accuracy and calibration columns stay put.

² Answered by Ollama's own `/v1/systemone` endpoint rather than cerno's engine. Not a candidate
for the verdict; see below.

The 4 GB model matches the 26B reference's accuracy on a quarter of the footprint, several
times faster, and is *better* calibrated: the large model needs its logits flattened by 3.2
before its confidences mean anything, while the small one is very slightly under-confident.

**Fidelity is a gate, not a score.** It counts the cases where the model's most likely first
token was one of the offered letters. An answer can be read off a letter further down the
ranking even when the model was about to write prose, and it looks just as confident, so a
model below 100% is not a candidate regardless of its accuracy or speed.

**Ollama's decision models can be measured on their own path.** `nimble` and `tev1` ship with
Ollama's `POST /v1/systemone`, where Ollama writes the prompt and returns probabilities over the
offered answers. `--systemone nimble,tev1` runs the same dataset through that endpoint and adds
rows marked ²; `--systemone-host url` points it elsewhere (Ollama's default address otherwise,
whatever `--host` the engine rows use):

```bash
cargo run -p cerno-bench -- --reference gemma4:26b-a4b-it-q4_K_M \
  --models gemma4:e2b-it-qat,nimble,tev1 --systemone nimble,tev1
```

The service can answer through that endpoint too: `CERNO_HOST=systemone` sends every question to
Ollama's `/v1/systemone` instead of cerno's letter prompt (see [Hosts](#hosts)). With any other
host, a decision model set as `CERNO_DEFAULT_MODEL` is used like any other model, with letter
labels and first-token logprobs — the unmarked rows. Fidelity and Truncated have no meaning on the
² rows, which is why they are never picked as the verdict.

## Configuration

Deployment knobs are environment variables; the model table is an optional TOML file
(see [cerno.example.toml](cerno.example.toml)).

| Variable | Default | |
|---|---|---|
| `CERNO_BIND` | `127.0.0.1:3000` | Loopback only. cerno has no authentication; set `0.0.0.0:3000` only where the network is trusted |
| `CERNO_HOST` | `ollama` | `ollama`, `vllm`, `llamacpp`, `lmstudio`, `openai` or `systemone` |
| `CERNO_HOST_URL` | depends on `CERNO_HOST` | See [Hosts](#hosts) |
| `CERNO_HOST_API_KEY` | — | Bearer token for the OpenAI-compatible hosts |
| `CERNO_DEFAULT_MODEL` | `gemma4:e2b-it-qat` | Alias or model name |
| `CERNO_CONFIG` | — | Path to the model table |
| `CERNO_STRICT_MODELS` | `false` | Only allow configured models |
| `CERNO_MAX_CONCURRENT_QUESTIONS` | `4` | In flight against the host at once |
| `CERNO_KEEP_ALIVE` | `5m` | Empty means: do not send it |
| `CERNO_HOST_TIMEOUT_SECS` | `30` | Per question |
| `CERNO_REQUEST_TIMEOUT_SECS` | `50` | Per request, waiting for a free slot included; below the SDKs' 60 s so the caller hears it from cerno |
| `RUST_LOG` | `cerno_server=info,cerno_core=info` | |

## Hosts

| `CERNO_HOST` | Default URL | |
|---|---|---|
| `ollama` | `http://localhost:11434` | Native API; `CERNO_KEEP_ALIVE` applies |
| `vllm` | `http://localhost:8000/v1` | Also sends `top_k: -1`, `min_p: 0`, `chat_template_kwargs` |
| `llamacpp` | `http://localhost:8080/v1` | `llama-server`; also sends `top_k: 0`, `min_p: 0`, `post_sampling_probs: false` |
| `lmstudio` | `http://localhost:1234/v1` | Also sends `top_k: 0`, `min_p: 0` |
| `openai` | `https://api.openai.com/v1` | Standard fields only, for any other compatible server |
| `systemone` | `http://localhost:11434` | Ollama's `/v1/systemone`, which writes its own prompt; decision models (`nimble`, `tev1`) only |

Every host must report `top_logprobs`; one that does not answers with a 502 that says so. The
extra fields are there because a runtime that samples before it reports logprobs hands back a
truncated distribution otherwise. `openai` cannot send them, so it is only as good as the
server's own defaults.

`systemone` asks the question itself rather than a prompt and gets a probability for every
offered answer, so nothing is ever truncated. cerno's response, calibration and confidence stay
the same. Set `CERNO_DEFAULT_MODEL` to a decision model with it: any other model answers 502 with
Ollama's "not supported by System One".

```bash
CERNO_HOST=vllm cargo run --release -p cerno-server
CERNO_HOST=openai CERNO_HOST_URL=http://gpu-box:9000/v1 CERNO_HOST_API_KEY=… cargo run -p cerno-server
```

## Layout

```
crates/
  cerno-types    wire types — serde always, utoipa behind a feature
  cerno-host     ModelHost trait, the Ollama and OpenAI-compatible adapters
  cerno-core     labels, prompt, logprob maths, engine
  cerno-server   axum + utoipa
  cerno-sdk      Rust client
  cerno-tui      terminal front end, on cerno-sdk
  cerno-bench    model benchmark
sdks/python      uv package `cerno-sdk`, imported as `cerno`
sdks/typescript  npm package `cerno-sdk`
spec/            openapi.json + conformance cases
site/            the website, published with the API docs by docs.yml
```

`ModelHost` is the seam between cerno and a runtime. It knows nothing about noul, choice or
score — it answers one question, "what is the distribution over the next token?" — so every
host gets the same primitives for free.

## Limits

- **20 options.** Ollama and OpenAI report at most 20 ranked tokens, so a 21st option could never be
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
cargo doc --workspace --no-deps --open
```

`spec/openapi.json` is generated, and a test fails when it drifts from the code:

```bash
cargo run -p cerno-server --bin cerno-openapi > spec/openapi.json
```
