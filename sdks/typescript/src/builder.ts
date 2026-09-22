import type { Answers } from "./answers.ts";
import type { LevelSpec, Question, SystemOneRequest } from "./types.ts";

/** A request under construction. Chain questions onto it, then `send()`. */
export class SystemOneBuilder {
  private readonly request: SystemOneRequest;
  private readonly dispatch: (body: SystemOneRequest) => Promise<Answers>;

  constructor(state: string, dispatch: (body: SystemOneRequest) => Promise<Answers>) {
    this.request = { state, questions: [] };
    this.dispatch = dispatch;
  }

  /** Name a model or a configured alias. The service's default applies otherwise. */
  model(model: string): this {
    this.request.model = model;
    return this;
  }

  /** Scale the label logits before normalising. Above 1 flattens, below 1 sharpens. */
  calibration(temperature: number): this {
    this.request.calibration = { temperature };
    return this;
  }

  /** How likely the answer to `question` is yes. */
  noul(id: string, question: string): this {
    return this.push({ id, noul: question });
  }

  /**
   * One of `options`.
   *
   * Called with two arguments the options come second and no question is sent, which is the
   * right shape when the options speak for themselves.
   */
  choice(id: string, question: string, options: string[]): this;
  choice(id: string, options: string[]): this;
  choice(id: string, second: string | string[], third?: string[]): this {
    const options = third ?? (second as string[]);
    const question = third === undefined ? undefined : (second as string);

    return this.push({
      id,
      choice: question === undefined ? { options } : { question, options },
    });
  }

  /** A position on a rubric: a number for generated levels, or the level texts. */
  score(id: string, question: string, levels: LevelSpec): this {
    return this.push({ id, score: { question, levels } });
  }

  /**
   * The request as it will be sent. Useful for logging, and for testing a chain without a
   * server.
   */
  body(): SystemOneRequest {
    return this.request;
  }

  send(): Promise<Answers> {
    return this.dispatch(this.request);
  }

  private push(question: Question): this {
    this.request.questions.push(question);
    return this;
  }
}
