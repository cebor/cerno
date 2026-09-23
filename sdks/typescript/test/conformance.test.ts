/**
 * Runs the shared conformance cases from spec/conformance/cases.json.
 *
 * Every cerno SDK runs these same cases, so if the three clients ever disagree about the wire
 * format, they disagree here first.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { ApiError, Client, MissingAnswer, UnexpectedResponse, WrongAnswerType } from "../src/index.ts";

const here = dirname(fileURLToPath(import.meta.url));
const cases = JSON.parse(
  readFileSync(join(here, "../../../spec/conformance/cases.json"), "utf8"),
);

/** A fetch that answers every request with one canned response. */
function stub(status: number, body: unknown, asText = false): typeof globalThis.fetch {
  return (async () =>
    new Response(asText ? (body as string) : JSON.stringify(body), {
      status,
      headers: { "content-type": asText ? "text/html" : "application/json" },
    })) as unknown as typeof globalThis.fetch;
}

function clientFor(status: number, body: unknown, asText = false): Client {
  return new Client({ baseUrl: "http://cerno.test", fetch: stub(status, body, asText) });
}

/** Replay one request case through the public builder. */
function build(client: Client, testCase: any) {
  let builder = client.systemone(testCase.state);

  if (testCase.model) builder = builder.model(testCase.model);
  if (testCase.calibration !== undefined) builder = builder.calibration(testCase.calibration);

  for (const question of testCase.questions) {
    const { id, kind, question: text } = question;

    if (kind === "noul") {
      builder = builder.noul(id, text);
    } else if (kind === "choice") {
      // A choice whose options speak for themselves is called without a question.
      builder =
        text === undefined
          ? builder.choice(id, question.options)
          : builder.choice(id, text, question.options);
    } else if (kind === "score") {
      builder = builder.score(id, text, question.levels);
    } else {
      throw new Error(`unknown kind ${kind}`);
    }
  }

  return builder;
}

test("request cases produce the expected body", () => {
  const client = new Client({ baseUrl: "http://unused.invalid" });

  for (const testCase of cases.requests) {
    assert.deepEqual(build(client, testCase).body(), testCase.expect_body, testCase.name);
  }
});

test("response cases read back the expected values", async () => {
  for (const testCase of cases.responses) {
    const answers = await clientFor(200, testCase.body)
      .systemone("state")
      .noul("ignored", "the stub answers regardless")
      .send();

    const expect = testCase.expect;
    if (expect.model) assert.equal(answers.model, expect.model, testCase.name);
    if (expect.usage) {
      assert.equal(answers.usage.input_tokens, expect.usage.input_tokens, testCase.name);
    }

    for (const [id, want] of Object.entries<any>(expect.answers)) {
      if (want.type === "noul" && want.noul !== undefined) {
        assert.equal(answers.noul(id), want.noul, `${testCase.name}/${id}`);
      } else if (want.type === "choice") {
        assert.equal(answers.choice(id), want.choice, `${testCase.name}/${id}`);
        assert.equal(answers.index(id), want.index, `${testCase.name}/${id}`);
      } else if (want.type === "score") {
        assert.equal(answers.score(id), want.score, `${testCase.name}/${id}`);
        if (want.legend) assert.equal(answers.legend(id), want.legend, `${testCase.name}/${id}`);
      }

      if (want.truncated !== undefined) {
        assert.equal(answers.truncated(id), want.truncated, `${testCase.name}/${id}`);
      }
      if (want.truncated_labels !== undefined) {
        assert.deepEqual(
          answers.truncatedLabels(id),
          want.truncated_labels,
          `${testCase.name}/${id}`,
        );
      }
    }
  }
});

test("error cases throw ApiError carrying the code", async () => {
  for (const testCase of cases.errors) {
    const client = clientFor(testCase.status, testCase.body);

    await assert.rejects(
      () => client.systemone("state").noul("q", "Urgent?").send(),
      (err: unknown) => {
        assert.ok(err instanceof ApiError, testCase.name);
        assert.equal(err.status, testCase.status, testCase.name);
        assert.equal(err.code, testCase.body.code, testCase.name);
        assert.equal(err.questionId, testCase.body.question_id, testCase.name);
        return true;
      },
    );
  }
});

test("a non-cerno error body is reported as unexpected", async () => {
  // A gateway in front of the service can return HTML. Calling that a cerno error code would be
  // a lie, so it surfaces as something distinctly different.
  const client = clientFor(503, "<html>service unavailable</html>", true);

  await assert.rejects(
    () => client.systemone("state").noul("q", "Urgent?").send(),
    (err: unknown) => err instanceof UnexpectedResponse && err.status === 503,
  );
});

test("reading an answer as the wrong type names both types", async () => {
  const answers = await clientFor(200, cases.responses[0].body)
    .systemone("state")
    .noul("urgent", "Urgent?")
    .send();

  assert.throws(
    () => answers.choice("urgent"),
    (err: unknown) =>
      err instanceof WrongAnswerType && err.expected === "choice" && err.actual === "noul",
  );

  // A noul carries no confidence, as in JEV, so asking for one is the same mistake.
  assert.throws(
    () => answers.confidence("urgent"),
    (err: unknown) => err instanceof WrongAnswerType && err.expected === "choice or score",
  );

  assert.throws(() => answers.noul("nope"), MissingAnswer);
});

test("answers expose ids, probabilities and the raw response", async () => {
  const answers = await clientFor(200, cases.responses[0].body)
    .systemone("state")
    .noul("urgent", "Urgent?")
    .send();

  assert.deepEqual(answers.ids().sort(), ["sev", "team", "urgent"]);
  assert.equal(answers.get("team").type, "choice");

  const team = answers.get("team");
  assert.ok(team.type === "choice");
  assert.equal(team.probabilities[1]?.option, "Facility");

  assert.equal(answers.raw().model, "gemma4:e2b-it-qat");
  assert.ok(answers.timingMs > 0);
});
