/**
 * Failure modes outside the shared conformance cases: an unreachable service and a hanging one.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { CernoError, Client, TransportError } from "../src/index.ts";

const refusing = (async () => {
  throw new TypeError("fetch failed");
}) as unknown as typeof globalThis.fetch;

/** A fetch that never answers, but gives up when its signal is aborted, as the real one does. */
const hanging = ((_url: string, init?: RequestInit) =>
  new Promise((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
  })) as unknown as typeof globalThis.fetch;

test("an unreachable service is a TransportError carrying the cause", async () => {
  const client = new Client({ baseUrl: "http://cerno.test", fetch: refusing });

  await assert.rejects(
    () => client.systemone("state").noul("q", "Urgent?").send(),
    (err: unknown) =>
      err instanceof TransportError &&
      err instanceof CernoError &&
      err.cause instanceof TypeError,
  );
});

test("a service that never answers times out as a TransportError", async () => {
  const client = new Client({ baseUrl: "http://cerno.test", fetch: hanging, timeoutMs: 20 });

  await assert.rejects(() => client.models(), TransportError);
});

test("health is false when the service cannot be reached", async () => {
  const client = new Client({ baseUrl: "http://cerno.test", fetch: refusing });

  assert.equal(await client.health(), false);
});
