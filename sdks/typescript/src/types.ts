/**
 * Wire types for the cerno API.
 *
 * These mirror `spec/openapi.json` exactly. A change here without a change there is a bug.
 */

/** Temperature scaling applied to label logits before the softmax. */
export interface Calibration {
  /** 1 leaves the model's own distribution alone; above 1 flattens, below 1 sharpens. */
  temperature: number;
}

export interface ChoiceSpec {
  question?: string;
  options: string[];
}

/** A rubric given either as a plain count or as the text of each level, lowest first. */
export type LevelSpec = number | string[];

export interface ScoreSpec {
  question?: string;
  levels: LevelSpec;
}

export type Question =
  | { id: string; noul: string }
  | { id: string; choice: ChoiceSpec }
  | { id: string; score: ScoreSpec };

export interface SystemOneRequest {
  state: string;
  model?: string;
  calibration?: Calibration;
  questions: Question[];
}

export interface OptionProbability {
  option: string;
  probability: number;
}

export interface LevelProbability {
  level: number;
  legend: string;
  probability: number;
}

interface AnswerBase {
  confidence: number;
  /**
   * The logprob fed into the softmax, per label. A floor substitution appears here like any
   * other value; `truncated` is what says one happened.
   */
  raw_logprobs: Record<string, number>;
  /**
   * Whether some label fell outside the host's reporting window, which makes its probability
   * an upper bound rather than an observation.
   */
  truncated: boolean;
}

export interface NoulAnswer extends AnswerBase {
  type: "noul";
  /** Probability that the answer is yes. */
  noul: number;
}

export interface ChoiceAnswer extends AnswerBase {
  type: "choice";
  choice: string;
  /** The winning option's position in the request. */
  index: number;
  probabilities: OptionProbability[];
}

export interface ScoreAnswer extends AnswerBase {
  type: "score";
  /** The most likely level, 1-based. */
  score: number;
  /** The probability-weighted mean level — often more useful than the argmax. */
  expected_score: number;
  legend: string;
  probabilities: LevelProbability[];
}

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

export interface Usage {
  input_tokens: number;
  questions: number;
}

export interface SystemOneResponse {
  answers: Record<string, Answer>;
  model: string;
  usage: Usage;
  timing_ms: { total: number };
}

export type ErrorCode =
  | "too_many_options"
  | "too_many_questions"
  | "invalid_request"
  | "too_few_options"
  | "invalid_levels"
  | "empty_state"
  | "empty_question"
  | "duplicate_question_id"
  | "no_questions"
  | "unknown_model"
  | "invalid_calibration"
  | "no_label_matched"
  | "host_unavailable"
  | "host_timeout"
  | "internal";

export interface ErrorResponse {
  code: ErrorCode;
  message: string;
  question_id?: string;
}

export interface ModelInfo {
  alias: string;
  model: string;
  calibration: Calibration;
}

export interface ModelsResponse {
  models: ModelInfo[];
  default: string;
}
