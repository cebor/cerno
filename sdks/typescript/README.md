# @cerno/sdk

Three kinds of question about a piece of text, answered by a locally hosted model:

- **noul** — how likely is the answer yes
- **choice** — which one of up to 20 options
- **score** — where on a rubric of 2–10 levels

Each question is one forward pass, so answers come back in tens of milliseconds.

```bash
npm install @cerno/sdk
```

```ts
import { Client } from "@cerno/sdk";

const client = new Client("http://localhost:3000");

const answers = await client
  .systemone("Ticket: server room at 31C, rising, servers throttling.")
  .noul("urgent", "Is this urgent?")
  .choice("team", "Which team?", ["IT", "Facility", "HR"])
  .score("sev", "How severe?", ["harmless", "minor", "moderate", "high", "critical"])
  .send();

answers.noul("urgent");     // 0.991
answers.choice("team");     // "Facility"
answers.score("sev");       // 5
answers.legend("sev");      // "critical"
answers.confidence("team"); // 0.939
```

Questions in one call share their state, so the text is sent and prefilled once.

A choice whose options speak for themselves takes two arguments:

```ts
.choice("mood", ["positive", "neutral", "negative"])
```

## Reading the answer honestly

Every answer carries the evidence it came from:

```ts
answers.truncated("team");            // a label fell outside the host's reporting window
answers.get("team").raw_logprobs;     // { A: -4.54, B: -0.02, ... }
answers.get("team").probabilities;    // per option, in request order
```

`truncated` is worth checking when you act on a probability rather than on the winner: it means
at least one option ranked below everything the host reported, so its probability is an upper
bound, not an observation.

To flatten an overconfident model, scale the logits before they are normalised:

```ts
await client.systemone(text).calibration(2.5).noul("urgent", "Is this urgent?").send();
```

Above 1 flattens, below 1 sharpens, and `raw_logprobs` is unaffected either way.

`answers.get(id)` is a discriminated union, so narrowing on `type` gives you the full shape:

```ts
const answer = answers.get("team");
if (answer.type === "choice") {
  answer.probabilities[0].option; // typed
}
```

## Errors

```ts
import { ApiError } from "@cerno/sdk";

try {
  await client.systemone(text).choice("team", "Which?", options).send();
} catch (err) {
  if (err instanceof ApiError) {
    err.code;       // "too_many_options" — branch on this, never on the message
    err.questionId; // "team"
  }
}
```

`UnexpectedResponse` is thrown instead when a non-2xx body is not a cerno error at all, which
usually means a proxy between you and the service.

## Options

```ts
new Client({
  baseUrl: "http://localhost:3000",
  timeoutMs: 60_000,
  headers: { "x-request-id": id },
  fetch: customFetch, // injected for testing or custom transports
});
```
