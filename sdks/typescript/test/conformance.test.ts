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
      if (want.label_mass !== undefined) {
        assert.equal(answers.labelMass(id), want.label_mass, `${testCase.name}/${id}`);
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

test("unknown code cases keep their code and message", async () => {
  // A newer service can send a code this client has never heard of. That is still the service
  // speaking, so it is an ApiError carrying the code as it came.
  for (const testCase of cases.unknown_codes) {
    const client = clientFor(testCase.status, testCase.body);

    await assert.rejects(
      () => client.systemone("state").noul("q", "Urgent?").send(),
      (err: unknown) => {
        assert.ok(err instanceof ApiError, testCase.name);
        assert.equal(err.status, testCase.status, testCase.name);
        assert.equal(err.code, testCase.body.code, testCase.name);
        assert.equal(err.questionId, testCase.body.question_id, testCase.name);
        assert.ok(err.message.includes(testCase.body.message), testCase.name);
        return true;
      },
    );
  }
});

test("unexpected cases are reported as unexpected", async () => {
  // A body that is not cerno's - a gateway's HTML page, some other JSON, a 2xx that is not an
  // answer - is reported as exactly that, never as a cerno error code it does not carry.
  for (const testCase of cases.unexpected) {
    const client = clientFor(testCase.status, testCase.body_text, true);

    await assert.rejects(
      () => client.systemone("state").noul("q", "Urgent?").send(),
      (err: unknown) => {
        assert.ok(err instanceof UnexpectedResponse, `${testCase.name}: ${String(err)}`);
        assert.equal(err.status, testCase.status, testCase.name);
        return true;
      },
    );
  }
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
  // Names every plain object inherits are not answers either.
  for (const inherited of ["toString", "constructor", "__proto__"]) {
    assert.throws(() => answers.get(inherited), MissingAnswer, inherited);
  }
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
