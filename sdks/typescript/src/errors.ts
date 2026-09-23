import type { ErrorCode, ErrorResponse } from "./types.ts";

export class CernoError extends Error {}

/**
 * The service answered with a structured failure.
 *
 * Branch on `code`, never on `message` — the codes are the contract, the prose is not.
 */
export class ApiError extends CernoError {
  readonly status: number;
  readonly code: ErrorCode;
  readonly questionId?: string;

  constructor(status: number, body: ErrorResponse) {
    super(`cerno returned ${status} (${body.code}): ${body.message}`);
    this.name = "ApiError";
    this.status = status;
    this.code = body.code;
    this.questionId = body.question_id;
  }
}

/**
 * The service could not be reached, or did not answer within the timeout. The error `fetch`
 * threw is kept as `cause`.
 */
export class TransportError extends CernoError {
  constructor(message: string, cause: unknown) {
    super(message, { cause });
    this.name = "TransportError";
  }
}

/** A non-2xx response that was not shaped like a cerno error — a proxy, most likely. */
export class UnexpectedResponse extends CernoError {
  readonly status: number;
  readonly body: string;

  constructor(status: number, body: string) {
    super(`cerno returned ${status}: ${body.slice(0, 200)}`);
    this.name = "UnexpectedResponse";
    this.status = status;
    this.body = body;
  }
}

export class MissingAnswer extends CernoError {
  readonly questionId: string;

  constructor(questionId: string) {
    super(`no answer for question ${JSON.stringify(questionId)}`);
    this.name = "MissingAnswer";
    this.questionId = questionId;
  }
}

export class WrongAnswerType extends CernoError {
  readonly questionId: string;
  readonly expected: string;
  readonly actual: string;

  constructor(questionId: string, expected: string, actual: string) {
    super(`question ${JSON.stringify(questionId)} answered with a ${actual}, not a ${expected}`);
    this.name = "WrongAnswerType";
    this.questionId = questionId;
    this.expected = expected;
    this.actual = actual;
  }
}
