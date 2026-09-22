/**
 * TypeScript client for cerno.
 *
 * cerno answers three kinds of question about a piece of text — is it true (`noul`), which one
 * is it (`choice`), where on a scale does it sit (`score`) — using a locally hosted model. Each
 * question is one forward pass, so answers come back in tens of milliseconds.
 */

export { Answers } from "./answers.ts";
export { SystemOneBuilder } from "./builder.ts";
export { Client, type ClientOptions } from "./client.ts";
export {
  ApiError,
  CernoError,
  MissingAnswer,
  UnexpectedResponse,
  WrongAnswerType,
} from "./errors.ts";
export type * from "./types.ts";
