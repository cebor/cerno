import { MissingAnswer, WrongAnswerType } from "./errors.ts";
import type {
  Answer,
  ChoiceAnswer,
  NoulAnswer,
  ScoreAnswer,
  SystemOneResponse,
  Usage,
} from "./types.ts";

/**
 * The answers to one request, with accessors that fail loudly on the wrong id or type.
 *
 * Reaching for `noul("team")` when `team` was a choice is a programming mistake, not a runtime
 * condition, so it throws rather than handing back a default.
 */
export class Answers {
  readonly model: string;
  readonly usage: Usage;
  readonly timingMs: number;

  // Declared explicitly rather than as a constructor parameter property: Node's native
  // type-stripping does not support those, and the test suite runs the sources directly.
  private readonly response: SystemOneResponse;

  constructor(response: SystemOneResponse) {
    this.response = response;
    this.model = response.model;
    this.usage = response.usage;
    this.timingMs = response.timing_ms.total;
  }

  /** The raw answer for `id`. */
  get(id: string): Answer {
    const answer = this.response.answers[id];
    if (answer === undefined) throw new MissingAnswer(id);
    return answer;
  }

  ids(): string[] {
    return Object.keys(this.response.answers);
  }

  /** Probability that the answer to `id` is yes. */
  noul(id: string): number {
    return this.typed<NoulAnswer>(id, "noul").noul;
  }

  /** The winning option for `id`. */
  choice(id: string): string {
    return this.typed<ChoiceAnswer>(id, "choice").choice;
  }

  /** The winning option's position in the request. */
  index(id: string): number {
    return this.typed<ChoiceAnswer>(id, "choice").index;
  }

  /** The winning level for `id`, 1-based. */
  score(id: string): number {
    return this.typed<ScoreAnswer>(id, "score").score;
  }

  /** The probability-weighted mean level for `id`. */
  expectedScore(id: string): number {
    return this.typed<ScoreAnswer>(id, "score").expected_score;
  }

  legend(id: string): string {
    return this.typed<ScoreAnswer>(id, "score").legend;
  }

  /**
   * How peaked the distribution behind `id` was, in 0..=1. A noul has none, as in JEV: its
   * probability is already the whole answer.
   */
  confidence(id: string): number {
    const answer = this.get(id);
    if (answer.type === "noul") {
      throw new WrongAnswerType(id, "choice or score", answer.type);
    }
    return answer.confidence;
  }

  /**
   * Whether some label for `id` fell outside the host's reporting window. When true, that
   * label's probability is an upper bound rather than an observation.
   */
  truncated(id: string): boolean {
    return this.get(id).truncated;
  }

  /** The labels for `id` whose logprob is an upper bound rather than an observation. */
  truncatedLabels(id: string): string[] {
    return this.get(id).truncated_labels ?? [];
  }

  /** The response exactly as the service sent it. */
  raw(): SystemOneResponse {
    return this.response;
  }

  private typed<T extends Answer>(id: string, expected: T["type"]): T {
    const answer = this.get(id);
    if (answer.type !== expected) {
      throw new WrongAnswerType(id, expected, answer.type);
    }
    return answer as T;
  }
}
