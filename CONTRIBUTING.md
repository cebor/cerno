# Contributing to cerno

Thanks for looking. cerno is small, and most of what makes a change safe here is knowing a
handful of things that the code cannot tell you on its own. This page lists them, and points to
where each one is written up in full.

## Setting up

| | |
|---|---|
| Rust | stable, edition 2024 |
| Python | 3.10+, with [uv](https://docs.astral.sh/uv/) |
| Node | recent enough to run `.ts` files directly (22.18+ or 23.6+) — the tests have no build step |
| Ollama | only for live checks; nothing in the test suites needs it |

```bash
cargo test --workspace
cd sdks/python && uv run pytest
cd sdks/typescript && npm install && npm test
```

All three suites mock the host, so they run anywhere and their numbers are exact. The live
checks need Ollama and a pulled model:

```bash
ollama pull gemma4:e2b-it-qat
cargo run -p cerno-sdk --example smoke
cargo run -p cerno-bench
```

Against a live model, **assert ranges, never equality**: identical requests return logprobs
that differ in the third decimal. And discard the first request when timing anything — it
loads the model, 5–10 s cold against ~40 ms warm.

## Before you change something

[CLAUDE.md](CLAUDE.md) is the long version, written for humans and coding agents alike. The
short version:

- **The sampling options in `cerno-host` are correctness conditions.** `top_k: 0, top_p: 1,
  min_p: 0` keep Ollama from truncating the distribution before it reports it. Loosening them
  produces answers that still look plausible with probabilities that mean nothing. They are
  not configurable, on purpose.
- **Labels stay single capital letters.** Words split into several tokens and compete with
  their own spellings; letters do neither.
- **A missing label is a bound, not a zero.** It gets the weakest reported logprob and the
  answer is flagged `truncated`.
- **One code path for all primitives.** A new primitive is a `ballot_for` arm and a response
  shape in `cerno-core`, not a second path through the engine.
- **Validate before the host is called.** A bad request must not load a model; there is a test
  with a mock that expects zero calls.
- **The TUI validates nothing.** It builds requests through `cerno-sdk` and lets the service
  refuse them. Its logic lives outside `render.rs` so it can be tested without a terminal.

## Changing the wire format

The service has one contract and four clients built against it — Rust, Python, TypeScript and
the TUI — so the order matters:

1. **Update `spec/conformance/cases.json` first.** Request cases pin the JSON a builder must
   produce, response cases pin the values read back, error cases pin the failure mapping.
2. Change the types in `cerno-types` and the server.
3. Regenerate the spec:
   ```bash
   cargo run -p cerno-server --bin cerno-openapi > spec/openapi.json
   ```
   `the_checked_in_spec_matches_the_code` fails until you do.
4. Make all three SDK suites and the TUI's `tests/conformance.rs` pass again.

A change that passes in one language and not the others is not finished.

## Changing prompts, labels or the default model

Anything that could move what a model answers is measured, not argued:

```bash
cargo run -p cerno-bench
```

This scores every candidate over `crates/cerno-bench/dataset.json` and rewrites
[docs/model-selection.md](docs/model-selection.md). Fidelity — did the model answer with one of
the offered letters — is a gate before accuracy is looked at. Commit the regenerated document
with the change that caused it, and if the README's snapshot of the table no longer matches,
update it from the same run.

The prompt instructions stay in English; the state and the question stay in whatever language
the caller wrote. The dataset is half German to keep that honest — keep it that way when you add
cases.

## Testing the TUI for real

A green suite has let through bugs that only showed up in the running binary. For anything
touching input handling, drive the real binary in a pty (`pty.openpty`, with `TIOCSWINSZ` set,
or it draws nothing) and read the final screen. Do not use `script` for this: it doubles
stdin, which looks exactly like an application bug. Details in [CLAUDE.md](CLAUDE.md).

## Things to ask about first

Some things were left out deliberately rather than forgotten. Open an issue before building
any of these, so we can agree on the shape:

- a llama.cpp `ModelHost` (the seam exists; `max_top_logprobs` has to be reported honestly)
- choices beyond 20 options — today a 422, because Ollama cannot report a 21st label
- fitting calibration from labelled data
- shuffling options against position bias

The same goes for new dependencies. The TUI's dependency set in particular is pinned by
construction — there is no direct `crossterm` dependency, and that is intentional.

## Commit strategy

The history of `main` is linear and meant to be read. `git log` is where the reasoning behind
the code lives, so a commit is written for the person who runs `git blame` on it a year from
now.

### Branches and merging

- Work on a short-lived topic branch off `main`, one branch per change, and open a merge
  request from it.
- Keep it current by **rebasing** onto `main`, never by merging `main` into it.
- It lands **fast-forward only**. There are no merge commits on `main`, and no squash-on-merge
  either — the commits you push are the commits that land, so shape them before review
  finishes.
- Never force-push `main`. Force-pushing your own topic branch after a rebase is expected
  (`git push --force-with-lease`).

### What goes into one commit

- **One logical change per commit.** A feature, a fix, a refactor — not two of them, and not
  half of one. If the message needs "and also", it is two commits.
- **Every commit is green on its own.** All three suites pass at every commit, not just at the
  tip, so `git bisect` always lands somewhere meaningful.
- **Generated files travel with their cause.** A change to the wire types carries the
  regenerated `spec/openapi.json`; a change that moves benchmark results carries the
  regenerated `docs/model-selection.md`.
- **A wire format change is one commit across all clients.** Conformance cases, types, spec and
  all SDKs together — splitting them would leave commits where the clients disagree with the
  service, which is exactly what the conformance cases exist to rule out.
- **Docs change with the code they describe.** If a change makes a sentence in the README or
  CLAUDE.md untrue, fix the sentence in the same commit.
- Review fixes are folded into the commit they fix (`git commit --fixup`, then
  `git rebase -i --autosquash main`) rather than piled on top as "address review".

### Writing the message

```
Make `cargo run -p cerno-server` start the server

The package grew a second binary when the OpenAPI dump was added, and
cargo has refused to guess between them ever since:

    error: `cargo run` could not determine which binary to run

`default-run` settles it, and a sweep over `cargo metadata` confirms no
other package is ambiguous.

Changelog: fixed
```

- **Subject:** imperative, says what the change does, no trailing full stop, at most about 60
  characters. No `feat:`/`fix:` prefixes — the changelog comes from a trailer instead (below),
  so the subject can stay a sentence.
- **Body:** wrapped at 72 columns, and says **why**: what was wrong, what was measured or tried,
  and why this fix over the alternatives. A number you measured beats an adjective.
- A bug found along the way, or a trap the next person would fall into, belongs in the body —
  and, if it will outlive the commit, in CLAUDE.md too.
- Written in English, like the rest of the repository.

### The changelog trailer

The changelog is generated from a `Changelog:` trailer in the last paragraph of the message,
which is the trailer GitLab's changelog API reads by default. Add it when **someone using
cerno would notice the change** — through the HTTP API, an SDK, the TUI, configuration, or a
change in the answers a model gives. Leave it off for refactors, tests, CI and internal
documentation; those commits do not appear in the changelog at all.

| Value | For |
|---|---|
| `added` | a new capability: a primitive, an endpoint, an SDK method, a TUI key |
| `changed` | existing behaviour that works differently, including a new default model |
| `fixed` | a bug that users could hit |
| `performance` | the same answers, measurably faster or lighter — say by how much in the body |
| `deprecated` | still works, will be removed; the body names the replacement |
| `removed` | gone |
| `security` | a vulnerability fix |

One value per commit. If a commit seems to need two, it is usually two commits. The subject
line becomes the changelog entry, which is one more reason to write it as a sentence a user
can understand.

A breaking change to the wire format or an SDK's public API is `changed` or `removed`, and its
body says what callers have to do. `git commit --trailer "Changelog: fixed"` adds the line in
the right place.

The history before this convention has no trailers, so everything up to and including 0.1.0 is
summarised by hand.

## License

cerno is MIT-licensed. By contributing you agree that your contribution is released under the
same [license](LICENSE).
