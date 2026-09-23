# cerno — Python client

Three kinds of question about a piece of text, answered by a locally hosted model:

- **noul** — how likely is the answer yes
- **choice** — which one of up to 20 options
- **score** — where on a rubric of 2–10 levels

Each question is one forward pass, so answers come back in tens of milliseconds.

```bash
uv add cerno
```

```python
from cerno import Client

client = Client("http://localhost:3000")

answers = (
    client.systemone("Ticket: server room at 31C, rising, servers throttling.")
    .noul("urgent", "Is this urgent?")
    .choice("team", "Which team?", ["IT", "Facility", "HR"])
    .score("sev", "How severe?", ["harmless", "minor", "moderate", "high", "critical"])
    .send()
)

answers.noul("urgent")        # 0.991
answers.choice("team")        # "Facility"
answers.score("sev")          # 5
answers.legend("sev")         # "critical"
answers.confidence("team")    # 0.939
```

Questions in one call share their state, so the text is sent and prefilled once.

## Async

Same surface, `await` on `send()`:

```python
from cerno import AsyncClient

async with AsyncClient("http://localhost:3000") as client:
    answers = await client.systemone(text).noul("spam", "Is this spam?").send()
```

## Reading the answer honestly

Every answer carries the evidence it came from:

```python
answers.truncated("team")          # some label fell outside the host's reporting window
answers.truncated_labels("team")   # ("C",) — which ones; their raw_logprobs entry is a bound
answers["team"].raw_logprobs       # {"A": -4.54, "B": -0.02, ...}
answers["team"].probabilities      # per option, in request order
```

`truncated` is worth checking when you act on a probability rather than on the winner: it means
at least one option ranked below everything the host reported, so its probability is an upper
bound, not an observation.

To flatten an overconfident model, scale the logits before they are normalised:

```python
client.systemone(text).calibration(2.5).noul("urgent", "Is this urgent?").send()
```

Above 1 flattens, below 1 sharpens, and `raw_logprobs` is unaffected either way.

## Errors

```python
from cerno import ApiError

try:
    answers = client.systemone(text).choice("team", "Which?", options).send()
except ApiError as err:
    err.code         # "too_many_options" — branch on this, never on the message
    err.question_id  # "team"
```

`UnexpectedResponse` is raised instead when a body is not something cerno sends — a non-2xx
that is not a cerno error, or a 2xx that is not an answer — which usually means a proxy between
you and the service. `TransportError` means the service could not be reached or timed out; the
`httpx` exception is its `__cause__`. All three derive from `CernoError`.

The builder refuses two easy mistakes with a `TypeError` before anything is sent: a string where
the options belong (`.choice("team", "Which team?")` would otherwise ask about eleven single
letters), and a string or `bool` as a score rubric.
