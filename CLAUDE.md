# Agent Instructions — cerno

A local reimplementation of JEV's system-one primitives (noul, choice, score) over Ollama
or any OpenAI-compatible runtime.
See [README.md](README.md) for the project overview and [docs/model-selection.md](docs/model-selection.md)
for the measured model comparison.

## Build & Test

```bash
cargo test --workspace
cd sdks/python && uv run pytest
cd sdks/typescript && npm test
```

Nothing in the test suites needs Ollama running; the hosts are mocked everywhere. The live
checks are `cargo run -p cerno-sdk --example smoke` and `cargo run -p cerno-bench`, both of
which do.

## The sampling options are correctness conditions, not tuning knobs

`SamplingOptions::REQUIRED` in `crates/cerno-host/src/ollama.rs` is pinned and deliberately
unreachable from outside the crate. **`top_k: 0, top_p: 1, min_p: 0` is the load-bearing part.**

Ollama applies the sampling transform *before* reporting logprobs. With the default `top_k: 40`
the distribution is truncated and what survives is renormalised, which measured against
`gemma4:26b-a4b-it-q4_K_M` collapsed a four-option question to `A = 0.0` with every rival at
`-17` — and B, C and D absent from the list entirely. The answer still looked plausible. Only
the probabilities were meaningless.

`num_predict: 1` and `temperature: 1` belong to the same set: one token, no sampling bias.
Calibration happens later, in `cerno-core`, where it is explicit and reversible.

`think: false` is on every request because without it the first generated token is a chat
template control token (`<|channel|>` on gemma4), not the answer label.
`rejects_thinking` retries without the field for models that have no thinking mode at all, and
the host remembers that per model, so the refusal costs one round trip per model rather than one
per question. The mock tests guard the answering mock so a repeated refusal cannot fall through
to it — without that guard they passed with the cache removed.

## The other hosts get the same treatment, per runtime

`crates/cerno-host/src/openai.rs` serves vLLM, llama.cpp, LM Studio and plain OpenAI through one
`/v1/chat/completions` adapter. The response shape is identical everywhere, so a `Flavour` only
decides which extra fields go out: `top_k`/`min_p` (with `-1` as vLLM's "off", `0` elsewhere),
`post_sampling_probs: false` for llama.cpp, and `chat_template_kwargs.enable_thinking: false` for
vLLM and llama.cpp. `Generic` sends none of these, because OpenAI answers 400 to a field it does
not know. `the_generic_flavour_sends_nothing_nonstandard` pins that.

`reasoning_effort: "none"` is the standard thinking switch and goes out on every flavour.
Without it, gemma4 behind Ollama's `/v1` put `<|channel>` first at `-0.03`, with the answer at
`-3.5`; `think` and `chat_template_kwargs` were ignored on that endpoint. The OpenAI adapter has
its own `rejects_thinking`, because a model without a reasoning mode refuses the field.

Measured against Ollama's `/v1` on `gemma4:26b-a4b-it-q4_K_M`: the `Generic` body returns the
exact same distribution as the native path with `REQUIRED` options, and `top_k: 40` sent there
changed nothing. vLLM, llama.cpp and LM Studio have not been run live yet; whether LM Studio
reports `top_logprobs` at all, and whether llama.cpp honours `post_sampling_probs` on the chat
endpoint, are the two open questions. A host without logprobs fails loudly as `NoLogprobs`.

One host per process, chosen by `CERNO_HOST` through `cerno_host::connect`. A second runtime
is a second instance, not a routing table.

## Labels are single capital letters, and that is not cosmetic

`crates/cerno-core/src/labels.rs`. Word labels do not survive tokenisation: measured on
`gemma4:26b`, a German yes/no produced `Ja`, `JA`, ` Ja` and `ja` as four separate entries each
holding part of the mass, while "Nein" was not a token at all — it arrived as `Ne` + `in` and so
could never be read from a single-token distribution.

Letters have neither problem. `labels::matches` still folds case and surrounding whitespace,
because some tokenisers hand back `" A"`, and every matching token is combined through
`math::logsumexp` rather than one being picked and the rest dropped.

The alphabet stops at 20 because that is Ollama's `top_logprobs` ceiling. A 21st label could
never be observed, so `MAX_OPTIONS` is that same 20 and a larger choice is a 422.

## A missing label is not a zero

When a model is confident, the losing label drops out of the top-20 window — measured at 39% of
cases for the 26B reference model. `read_labels` substitutes `distribution.floor`, the weakest
logprob the host reported, which is a strict upper bound on anything unreported, and sets
`truncated: true`. Every answer carries that flag and its `raw_logprobs`, so a caller acting on
a probability rather than a winner can tell an observation from a bound.

If *no* label matched, that is `NoLabelMatched` and a 502 — the request was fine, the model is
not following the instruction, and the error names the tokens that came back instead so an
operator can see what they are dealing with.

## Confidence is one definition for all three primitives

`1 - H(p)/log(n)`, in `math::confidence`. Normalising by `log(n)` puts a two-option noul and a
ten-level score on the same `0..=1` scale.

It is a scale, not an invariant: at a fixed top probability the value *rises* with the label
count (0.9 of ten scores 0.76, 0.9 of two scores 0.53). That is intended — narrowing ten
candidates to one is the stronger statement — but it means confidences are only directly
comparable between questions of the same shape. `confidence_is_not_invariant_to_label_count`
pins both numbers.

## Everything is one path

`Engine::answer` handles all three primitives through `ballot_for`, which is the only place they
differ: a noul is a two-option ballot with "Yes" first (so the noul is always option 0), a choice
is its options, a score is its legend. Everything after that — labelling, prompting, the single
host call, folding, flooring, calibrating, normalising — is shared. **Add a primitive by adding
a `ballot_for` arm and a response shape, not a second code path.**

## Validation happens before a model is loaded

`Engine::validate` runs over the whole request in the handler before any host call, so a bad
request costs nothing. `a_request_with_too_many_options_is_rejected_before_the_host_is_called`
pins this with a mock that expects zero calls.

## The prompt is in English, the content is not

`crates/cerno-core/src/prompt.rs`. The answer is a single letter, so the instruction language
never reaches the output, and small instruct-tuned models follow English formatting instructions
more reliably. State and question stay in whatever language the caller wrote them in — the
benchmark set is deliberately half German to keep that honest.

## The spec is generated, and drift is a test failure

`spec/openapi.json` comes from `cargo run -p cerno-server --bin cerno-openapi`.
`the_checked_in_spec_matches_the_code` fails when the checked-in copy is stale. Three
hand-written SDKs are built against that file, so silent drift would be a silent break in three
languages at once.

`spec/conformance/cases.json` is the shared truth all three SDKs test against: request cases pin
the JSON a builder must produce, response cases pin the values read back out, error cases pin the
failure mapping. The Rust suite additionally round-trips every case body through the server's own
types, so a case can never describe something the service would reject.

**When the wire format changes, update the conformance cases first**, then make all three SDKs
pass them again.

## Model choice is measured, not assumed

`cargo run -p cerno-bench` over `crates/cerno-bench/dataset.json`. Fidelity — did the model
answer with one of the offered letters — is reported first and treated as a gate, because a model
that writes prose is unusable here whatever its accuracy.

The winner is picked mechanically in `pick_winner` (highest accuracy among fully-faithful models,
ties by p50), so the verdict in the generated doc stays true when the benchmark is re-run.

## Two things that will bite in a live test

- **Logprobs are not bit-reproducible.** Identical requests return values differing in the third
  decimal, from GPU reduction order. Assert ranges against a live model, never equality. The
  mocked suites are exact because their inputs are fixed.
- **The first request loads the model.** Measured at 5–10 s cold against 36 ms warm. Benchmarks
  send a warm-up question first and discard it; anything else timing a live call should too.

## The TUI keeps its logic out of the terminal

`crates/cerno-tui` is split so that everything deciding *what* happens is testable without a
terminal: `draft.rs` parses the fields, `app.rs` holds the state, `keys.rs` turns key presses
into transitions, `session.rs` persists, and only `render.rs` draws. `tests/render.rs` covers
the drawing through `ratatui::backend::TestBackend`, including asserting that the refused
question is actually coloured red — a buffer carries styles, so that is checkable rather than a
matter of trust.

`tests/conformance.rs` drives the form from `spec/conformance/cases.json` and asserts the body
it produces is identical to the SDKs'. The cases already describe a question neutrally, which is
the same shape as a `QuestionDraft`, so the fourth client hangs on the same truth as the other
three for free.

**Dependencies are pinned by construction, not by care.** `ratatui-textarea` 0.9 and `ratatui`
0.30 agree because both sit on `ratatui-core ^0.1` and `ratatui-widgets ^0.3`. There is no direct
`crossterm` dependency at all — `ratatui::crossterm` is the re-export — so a second crossterm
version cannot appear. The older `tui-textarea` is stuck on ratatui 0.29 and is not an option.

## A default belongs beside a field, not inside it

Twice the editor pre-filled a field with its default and left the cursor in it, so typing joined
rather than replaced: a typed "urgent" became `urgentq1`, and a typed rubric became
`5unkritisch, gering, …`. Both shipped through a green test suite and were found by driving the
real binary in a pty.

The id and levels fields therefore start **empty**, their default is shown in the field's title
(`id (blank → q2)`), and `Editor::finish` substitutes it when the field is left blank. Every
field's cursor starts at the end of its contents so typing appends. If a third field ever gets a
default, it goes the same way.

## Two traps in the event loop

**Do not gate a `select!` branch on `JoinHandle::is_finished`.** The health probe did, and a
probe against a refusing port completes in under a millisecond — before the first `select!`. The
guard then disabled the branch and the result was never collected, so the status bar sat on "?"
instead of saying the service was unreachable. It now arrives over a channel, which holds the
message whether or not anyone was looking. `spawn_probe` says so at the call site.

**A cancelled request can still land.** `Esc` aborts the task, but a result already on the
channel arrives anyway. Without a guard, Esc appeared to work and then the screen changed by
itself a moment later. "Drop it unless we are sending" is not enough: after Esc and a second
send, the first result arrives *while* the second is in flight. Every send therefore gets a
generation from `App::begin_send`, the result travels with it, and `App::finish_send` accepts
only the current one.

## Testing a TUI end to end

`script` is not usable for it: `printf 'hello' | script -qec cat /dev/null` prints `hellohello`,
and that doubling looked exactly like an application bug for a while. Drive a pty directly
instead — `pty.openpty`, `TIOCSWINSZ` for a real size, write keys on a schedule, then replay the
ANSI stream into a grid to read the final screen. Without the window size the program draws
nothing at all and the capture is empty.

## Host ceilings

Every adapter reports `max_top_logprobs: 20` today. llama.cpp's own ceiling is higher, which is
why `Engine` takes the minimum of that and `MAX_OPTIONS` rather than hard-coding 20.
