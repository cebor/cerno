# Security policy

## Supported versions

Only the latest release gets fixes. cerno is maintained by one person, and there are no
backports to older versions.

## Reporting a vulnerability

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/cebor/cerno/security/advisories/new),
not as a public issue. Say what you did, what happened, and which version and host runtime
you ran it against.

Replies are best effort: expect an acknowledgement within a week. A confirmed issue is fixed in
a release whose changelog entry carries `Changelog: security`, and you are credited unless you
ask not to be.

## What counts

Things cerno itself gets wrong, for example:

- **Leaking `CERNO_HOST_API_KEY`**: in a log line, an error message or a response body.
- **Getting past validation**: a request that reaches the host even though `Engine::validate`
  should have rejected it, or one that crashes the server or holds resources without bound.
- **The SDKs and the TUI**: anything in the Python, TypeScript or Rust clients, or in the
  session file the TUI writes, that exposes data or runs something it should not.

## What does not

These are deliberate, and documented where they are decided:

- **No authentication.** The service has none of its own. It listens on `127.0.0.1:3000` by
  default and logs a warning when `CERNO_BIND` puts it anywhere else. If it has to be reachable
  from a network, put it behind something that authenticates.
- **Prompt injection through `state` or a question.** That text is what the model decides
  about, so it can steer the answer. An answer is a probability, not an authorisation. Do not
  use it as one.
- **Vulnerabilities in the host runtime.** Ollama, vLLM, llama.cpp and LM Studio each have
  their own maintainers, so report those upstream.
